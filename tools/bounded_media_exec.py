"""跑一个媒体转码子进程:有超时、杀整组、stderr 有界、取消也回收。

## 为什么需要这个模块

Codex 在 PR #339 上报的 P1-3(feishu `_transcode_to_opus`)有三个独立缺口:

1. **无超时** —— ``await proc.communicate()`` 无限等。ffmpeg 卡在一个坏输入上
   (损坏容器、网络文件系统掉线、等 stdin)时,这条 ``await`` 永不返回。
2. **stderr 全缓存在内存** —— ``communicate()`` 把子进程 stderr 完整读进内存,
   没有上限。ffmpeg 在错误循环里能刷出几百 MB。
3. **``CancelledError`` 不进 ``except Exception``** —— 它继承自 ``BaseException``。
   会话被取消 / gateway 关停时,**子进程和临时文件都失去回收路径**。

⚠️ 前两条在 1C2G 无 swap 的板子上是**整机级**(同 2026-08-07 事故里
「无上限的并发 / 缓冲 / 读取」那一类)。而且这些调用点全都在 ``async def``
里 —— 卡住的不是一条消息,是**整个 gateway 事件循环**。

## 照抄的是谁(`copying-is-a-verifiable-claim` 三问,逐条作答)

⚠️ **⛔ 不是** ``whatsapp_cloud._convert_to_opus`` —— feishu 的 docstring 声称
照抄它,但它**自己带着同一个缺陷**,是兄弟调用点,不是正确先例。

真正做对的是 ``agent/secret_sources/command.py::_run_helper``(它的注释原话:
``start_new_session=True,  # so the hard timeout can kill the whole group``)。

| 先例的每个分支 | 本模块的对应分支 | 差异与理由 |
|---|---|---|
| ``start_new_session=True`` 建独立进程组 | ✅ 同 | — |
| ``communicate(timeout=…)`` 硬超时 | ✅ ``asyncio.wait_for(...)`` | 同步 API → 异步 API,语义相同 |
| 超时 → ``os.killpg(os.getpgid(pid), SIGKILL)`` | ✅ 逐字同 | ⭐ ``CommandContext`` / ``proc.kill()`` **只杀 leader**,孙子会孤儿化继续跑 |
| ``killpg`` 抛 ``ProcessLookupError``/``PermissionError``/``OSError`` → 退回 ``proc.kill()`` | ✅ 逐字同 | — |
| 杀完再 drain 一次(短超时)收尸 | ✅ 同 | 不 drain 会留 zombie |
| stderr **整个丢弃**(``_stderr_discarded``) | 🔴 **不同**:保留**前 N 字节** | 先例丢弃是因为 helper stderr 可能含密钥;我们需要 ffmpeg 的报错来给用户分类。⇒ 改成**有界保留**:边读边丢弃超出部分,管道始终被抽干(⛔ 不抽干会让子进程写满管道死锁) |
| —(同步代码没有取消这回事) | 🆕 ``CancelledError`` 分支 | asyncio 多出来的失败模式;先例里不存在,所以这一条**不是照抄,是新增**,单独说明 |

⭐ 三问都答完了。⛔ 没有「我参考了 X」这种含糊说法。
"""
from __future__ import annotations

import asyncio
import os
import signal as _signal
from typing import Optional, Sequence

#: 一次媒体转码的硬上限。
#:
#: ⛔ 不许拍脑袋、⛔ 不许照抄量纲不同的数字。推导:
#: 本仓**同为 async 子进程 + ffmpeg 相关**的既有上限是
#: ``gateway/platforms/qqbot/adapter.py`` 与
#: ``plugins/platforms/telegram/adapter.py`` 的 ``timeout=60``。
#: 语音条转码(``-b:a 32k`` 的 libopus,输入通常是几分钟以内的语音)在这个量级
#: 上有两个数量级的余量;真需要更久的是长视频转码,那类**不该**跑在事件循环上。
#: ⇒ 取 60s,与仓内同类一致。
DEFAULT_TRANSCODE_TIMEOUT = 60.0

#: stderr 最多保留多少字节交给调用方记日志。
#:
#: ⛔ 不许拍脑袋。推导:两个现有调用点都只把 stderr 截到
#: ``[:300]``(feishu)/ ``[:500]``(whatsapp)写进日志 —— 也就是说
#: **超过这个量的部分从来没有任何消费者**。取 8 KiB(≈ 两个数量级余量,
#: 足够容纳 ffmpeg 的多行报错),⛔ 而不是无上限地读进内存。
MAX_STDERR_BYTES = 8 * 1024

_DRAIN_CHUNK = 8192
#: 杀掉进程组之后,给内核收尸留的时间(照抄先例的 ``communicate(timeout=1.0)``)。
_REAP_TIMEOUT = 1.0


class TranscodeTimeout(Exception):
    """子进程超过硬上限,已连同整个进程组被杀掉。"""


async def _drain_capped(stream, cap: int) -> bytes:
    """把 stream 抽干,但**只保留前 ``cap`` 字节**。

    ⚠️ 必须一直抽到 EOF:只读 ``cap`` 字节就不读了,管道写满后子进程会**阻塞**,
    于是"有上限"反而制造了一个新的挂起。⇒ 继续读、把超出的部分丢掉。
    """
    if stream is None:
        return b""
    buf = bytearray()
    while True:
        chunk = await stream.read(_DRAIN_CHUNK)
        if not chunk:
            break
        if len(buf) < cap:
            buf.extend(chunk[: cap - len(buf)])
    return bytes(buf)


def _kill_process_group(proc) -> None:
    """杀掉**整个进程组** —— ⭐ ``proc.kill()`` 只杀 leader,孙子会孤儿化。

    逐字照抄 ``agent/secret_sources/command.py`` 的三段式:
    ``killpg(SIGKILL)`` → 三类异常退回 ``proc.kill()``。
    """
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)  # windows-footgun: ok
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


async def _reap(proc) -> None:
    """杀完收尸,⛔ 不留 zombie。失败不抛 —— 这一步是尽力而为。"""
    try:
        await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT)
    except (asyncio.TimeoutError, ProcessLookupError, OSError):
        pass


async def run_media_subprocess(
    argv: Sequence[str],
    *,
    timeout: float = DEFAULT_TRANSCODE_TIMEOUT,
    max_stderr_bytes: int = MAX_STDERR_BYTES,
) -> "tuple[Optional[int], bytes]":
    """跑一个转码子进程,返回 ``(returncode, 前 N 字节的 stderr)``。

    · 超时 ⇒ 杀**整组** + 收尸,抛 ``TranscodeTimeout``。
    · 被取消(``CancelledError``)⇒ 同样杀整组 + 收尸,**然后原样往上抛**
      (⛔ 不许吞掉取消)。这是同步先例里不存在的分支。
    · ⛔ 本函数不碰任何临时文件 —— 清理由调用方在 ``finally`` 里做,
      因为只有它知道产物路径。
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        # ⭐ 独立进程组,这样硬超时才能杀掉**整棵**子进程树。
        # ⚠️ Windows 上 start_new_session 无效;这些调用点是 POSIX 板端/服务端路径。
        start_new_session=True,
    )
    drain = asyncio.ensure_future(_drain_capped(proc.stderr, max_stderr_bytes))
    try:
        stderr, _ = await asyncio.wait_for(
            asyncio.gather(drain, proc.wait()), timeout=timeout
        )
    except asyncio.TimeoutError:
        _kill_process_group(proc)
        drain.cancel()
        await _reap(proc)
        raise TranscodeTimeout(f"exceeded {timeout:g}s") from None
    except asyncio.CancelledError:
        # 🔴 ``CancelledError`` 继承自 ``BaseException`` —— 调用点原先的
        # ``except Exception`` 根本接不到它,于是会话一被取消,子进程和临时文件
        # 就同时失去回收路径。这里必须收尸,然后**原样重抛**。
        _kill_process_group(proc)
        drain.cancel()
        await _reap(proc)
        raise
    return proc.returncode, stderr
