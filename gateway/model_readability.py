"""模型文件可读性契约 —— Hermes 侧（`SPEC-MODEL-FILE-READABILITY-CONTRACT.md`）。

## 这条契约要解决什么

族 A 的根因不是某一层写错了，而是：

> **「转运」责任在整条链上从没有人认领 —— 每层看自己都正常
> （PC 给了路径 ✅ / hermes 传了 ✅ / adapter 认出了 ✅），
> 缺的是层与层之间那一跳。**

Hermes 这一层的具体缺口：现在只把路径拼成一句 ``[file: <path>]`` 塞给模型，
**没有任何读取回执**。路径在模型的实际运行环境里不存在时，模型只会自己编或说
"读不到"，而链路上**没有任何一层报错**。

⇒ 唯一不变量（SPEC §1.1）：附件进入模型提示前必须满足闭集之一 ——
**内联可读** / **运行环境可读（`open` + `regular-file` + `read` 成功）** /
**明确失败**。⛔「路径长得像 cache」「后缀是 PDF」「字符串非空」都不在闭集内。

## 照抄的是谁（`copying-is-a-verifiable-claim` 三问）

先例是 LS `internal/channel/media/resolver.go:56-108` 的 `Resolve`。
⭐ 「照抄」是可验证的断言，所以把它的**每一个分支**列出来逐条对照：

| # | LS 分支 | 本模块 | 差异与理由 |
|---|---|---|---|
| 1 | 输入非空 → `empty_input` | ✅ 同 | — |
| 2 | `security.ValidatePath` → `invalid_path` | ✅ 同（拒 NUL / 非绝对 / `..`） | 语言差异，语义相同 |
| 3 | 在 **agent uploads 根**内 → `outside_agent_uploads` | 🔴 **不做** | LS 的 artifact 只来自 App uploads 根（封闭来源）；Hermes 的 `media_urls` 由二十多个 adapter 产出，落点不止 `_CACHE_DIRS`。照抄它会**误杀合法附件**（实测两条既有门变红）—— 而误杀比原缺陷严重得多。SPEC §1.1 的不变量只有「可读」，⛔ 没有「在某个根内」 |
| 4 | 根不存在 → `uploads_root_missing` | 🔴 **不做** | 同上：cache 根尚未创建时会把**所有**附件判成不可读 |
| 5 | `openNoFollow` → `open_failed` | ✅ `O_NOFOLLOW` | — |
| 6 | fd 真实路径未逃逸根 → `symlink_escape` | ✅ **fstat(fd) 与 stat(realpath) 的 dev/ino 双向比对**，再逐级上溯 | ⭐ 比路径字符串更硬（挡 TOCTOU）。⚠️ ⛔ **不能**照抄 LS 的 `fdpath.Of`：那依赖 `/dev/fd/N` 是符号链接，**macOS 上不是**，照抄会让判据恒假 |
| 7 | `IsRegular` → `not_regular` | ✅ 同 | — |
| 8 | `Size > 0` → `empty_file` | ✅ 同 | — |
| 9 | `Classify`（按内容分类） | 🔴 **不做** | Hermes 已有自己的 `media_types`，再分类一次会造出**第二个真相源**；且 SPEC §7.4 要求"任意文件类型不退化"，内容分类失败会把未知类型挡掉 |
| 10 | `KindHint=image` 必须真是 image | 🔴 **不做** | 同上，依赖 #9 |
| 11 | `Seek(0,0)` → `seek_failed` | ⚠️ 改为**实际 `read` 一个字节** | ⭐ **更强**：SPEC 要求的是 `read` 成功，`seek` 只证明可定位。挂载存在但 I/O 报错（远端 fs 掉线、稀疏文件洞）时 seek 会过、read 才会红 |
| 12 | `Size > Limits` → `too_large` | ✅ 同 | — |
| 13 | — | 🆕 **目标环境回执** | LS 只证明"设备文件系统可读"（SPEC §2.2 第 1 条明说它**不**证明 Docker/Modal 能读）。这一层是 SPEC 新增的 |

⭐ 第 6 条用 `st_dev/st_ino` 而不是路径字符串，是本模块**唯一比先例更严**的地方；
其余差异（#3/#9/#10/#11）都在上表给了理由，⛔ 没有"我参考了它"这种含糊说法。

## ⛔ 必须保持不变的行为（`fix-must-not-break-what-worked`）

1. **local 后端已能读的路径逐字不变** —— 它本来就是恒等，验证通过后返回同一字符串。
2. **Docker 已有的 cache mount 转换不变** —— 仍走 `to_agent_visible_cache_path`。
3. **任意文件类型不退化** —— ⛔ 不做内容分类，未知 MIME / 无扩展名照样通过。
4. **无附件会话不受影响** —— 本模块只在有 artifact 时被调用。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import stat
import threading
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────── 失败分类（SPEC §5.1 闭集） ───────────────────────────

#: ⛔ 闭集：每一类都必须能被用户区分并据此行动。
#  未知情况用 ``attachment_delivery_failed`` 兜底，⛔ 不许伪装成成功。
FAILURE_CODES: frozenset[str] = frozenset({
    "attachment_runtime_unavailable",   # 源存在，但当前执行环境未挂载/未同步/无回执
    "attachment_expired",               # 已被 TTL 清理 / 源路径不存在 / 摘要不符
    "attachment_permission_denied",     # 源或目标环境拒绝读取
    "attachment_download_failed",       # 渠道 / relay 下载、鉴权、网络失败
    "attachment_transfer_failed",       # 已下载但同步/挂载/摘要/目标打开失败
    "execution_environment_unsupported",# backend 不在闭集，或无 artifact 能力
    "attachment_delivery_failed",       # 兜底
})

#: 面向用户的安全提示。⭐ 判据是「不写它用户会不会做错事」——
#  会：他不知道附件没送到，会以为模型在偷懒。⛔ 但一个字都不许含路径 / 原始错误。
USER_MESSAGES: dict[str, str] = {
    "attachment_runtime_unavailable":
        "当前 Agent 的执行环境无法访问该文件。请重新发送文件，"
        "或切换到文件所在设备上的 Agent 后重试。",
    "attachment_expired":
        "该文件已不可用，请重新发送后再试。",
    "attachment_permission_denied":
        "当前 Agent 没有读取该文件的权限。请使用文件所属 Agent，"
        "或授予访问权限后重试。",
    "attachment_download_failed":
        "文件下载失败。请检查网络后重试；仍失败请重新发送文件。",
    "attachment_transfer_failed":
        "文件未能传入当前执行环境。请稍后重试；持续失败请重新发送文件。",
    "execution_environment_unsupported":
        "当前执行环境不支持处理该文件。请切换到支持文件处理的 Agent 后重试。",
    "attachment_delivery_failed":
        "文件未能处理完成。请稍后重试；持续失败请重新发送文件。",
}

#: 执行环境闭集（`tools/environments/__init__.py`）。
#  ⛔ 任何其它值一律 ``execution_environment_unsupported``，
#  ⛔ 不许按"本地路径大概可用"放行。
KNOWN_ENVIRONMENTS: frozenset[str] = frozenset({
    "local", "docker", "singularity", "modal",
    "daytona", "vercel_sandbox", "ssh",
})

#: 目标环境回执的证据等级。⭐ 分开是刻意的：把「真跑过 open+read」和
#  「只是按声明的 mount 换了个字符串」标成同一种，就是在假装有证据。
EVIDENCE_VERIFIED_READ = "verified_read"      # 在目标环境实际 open+read 成功
EVIDENCE_MOUNT_DECLARED = "mount_declared"    # 只有挂载声明，未在目标环境读过

#: 读取校验时实际读入的字节数上限（只为证明可读，⛔ 不是要把文件读进内存）。
_PROBE_BYTES = 4096


@dataclass(frozen=True)
class ArtifactReceipt:
    """一份 artifact 的交付回执。``model_path`` 只在验证成功后才有值。"""

    ok: bool
    source_path: str                     # ⛔ 仅日志/审计，绝不进用户文案或模型提示
    model_path: Optional[str] = None
    size: Optional[int] = None
    sha256: Optional[str] = None
    runtime_id: str = "local"
    evidence: Optional[str] = None
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None  # ⛔ 内部用：原始 error 摘要
    checks: tuple[str, ...] = field(default_factory=tuple)

    @property
    def user_message(self) -> Optional[str]:
        if self.ok:
            return None
        return USER_MESSAGES.get(
            self.failure_code or "", USER_MESSAGES["attachment_delivery_failed"])


def _fail(source: str, code: str, detail: str, checks, runtime_id: str) -> ArtifactReceipt:
    assert code in FAILURE_CODES, f"未登记的失败码:{code}"
    return ArtifactReceipt(
        ok=False, source_path=source, failure_code=code,
        failure_detail=detail, checks=tuple(checks), runtime_id=runtime_id,
    )


def _fd_matches_path(fd: int, source_path: str) -> bool:
    """打开的这个 fd，是不是 ``source_path`` **此刻**解析出来的那个文件。

    ⭐ 挡 TOCTOU：验证完再把路径换掉，``fstat(fd)`` 与 ``stat(realpath)`` 的
    ``(st_dev, st_ino)`` 就对不上了。LS 用 ``fdpath.Of`` 达到同一目的。

    ⚠️ ⛔ **不能**照抄 LS 的 ``fdpath.Of`` 实现：它依赖 ``/dev/fd/N`` 是符号
    链接，**macOS 上不是**，照抄会让判据恒假（实测 13 条门全红）。
    ⚠️ 也 ⛔ 不再检查「是否落在 cache 白名单目录内」—— 见
    ``verify_artifact_readable`` 里 #3/#4 处的说明：那条是从 LS 多抄来的，
    会误杀合法附件。
    """
    try:
        fd_st = os.fstat(fd)
    except OSError:
        return False
    try:
        real_st = os.stat(os.path.realpath(source_path))
    except OSError:
        return False
    return (fd_st.st_dev, fd_st.st_ino) == (real_st.st_dev, real_st.st_ino)


# 🔴 **整段探测必须离开 gateway 事件循环,并且有墙钟上限。**
# ``O_NONBLOCK`` 只对 FIFO/设备有效,**对普通文件不提供任何墙钟上限** ——
# 失联的 NFS / FUSE 挂载上,``lstat`` / ``open`` / ``fstat`` / ``os.read``
# 每一个都可能**无限挂起**。而 ``_build_media_placeholder()`` 是在事件循环上
# **同步**调用本函数的 ⇒ **一条附件就能堵住所有会话**,不只是当前这条消息。
# ⛔ 把 ``_build_media_placeholder`` 改异步会波及 gateway/run.py 的 7 个调用点,
#   作用域远大于缺陷 ⇒ 改在本文件内部,**公开签名一个字不变**。
#
# ⛔ 线程数必须有上限:否则每条卡死的附件都留一个线程,又是一处无界增长。
# 卡住的 worker 各自只占一个 fd + 4 KiB 缓冲,4 个足够并发探测且把
# 「卡死线程」封顶在 4。
_PROBE_MAX_WORKERS = 4
# 🔴 **准入必须有界,⛔ 不能靠线程池自己的无界工作队列排队。**
# 四个 worker 全被失联挂载卡死后,后续每个附件仍会被 ``submit`` 进
# ``ThreadPoolExecutor`` 的**无界队列**;超时分支既拒不掉、也移不走已排队的
# work item ⇒ 待处理 future / 参数 / 路径持续累积 ⇒ 约 2 GB 的共享设备内存
# 预算下最终 OOM。⇒ **提交前非阻塞拿票,拿不到就当场降级**。
# ⭐ 回收策略:票在 worker 的 ``finally`` 里归还 —— 挂载点恢复、syscall 返回的
#   那一刻票自动回来,**不需要重建线程池**(重建反而会把卡死线程变成无界增长)。
#   永久卡死的挂载没有「可回收」这回事,那时**持续快速降级就是正确行为**。
_probe_permits = threading.BoundedSemaphore(_PROBE_MAX_WORKERS)
# ⛔ 期限不许拍脑袋:健康的本地探测是**亚毫秒**级(几个 syscall + 4 KiB 读),
# 5s 已高出三个数量级;同时远低于任何用户可感知的发送预算 ⇒ 卡住时能立刻
# 按 ``attachment_transfer_failed`` 降级,而不是把整个网关拖住。
_PROBE_DEADLINE_S = 5.0
# 🔴 **整条消息共享一个探测预算。** 单个附件的 ``_PROBE_DEADLINE_S`` 挡不住
# 「一条消息挂 N 个异常附件」—— 那是 N 倍的累计停顿。⛔ 上限不许拍脑袋:
# 取单附件预算的 2 倍 —— 正常一条消息里健康附件是亚毫秒级,2 倍足以覆盖
# 「一两个坏附件 + 若干健康附件」,再多就该整条降级而不是让用户干等。
_MESSAGE_PROBE_BUDGET_S = _PROBE_DEADLINE_S * 2
_probe_pool_lock = threading.Lock()
_probe_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_probe_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _probe_pool
    with _probe_pool_lock:
        if _probe_pool is None:
            _probe_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=_PROBE_MAX_WORKERS,
                thread_name_prefix="hermes-artifact-probe",
            )
        return _probe_pool


def _probe_submit(source_path, max_bytes, runtime_id):
    """拿票 + 提交。拿不到票返回 ``None``(调用方当场降级)。"""
    if not _probe_permits.acquire(blocking=False):
        return None

    def _run():
        try:
            return _verify_artifact_readable_blocking(
                source_path, max_bytes=max_bytes, runtime_id=runtime_id)
        finally:
            _probe_permits.release()

    try:
        return _get_probe_pool().submit(_run)
    except Exception:
        _probe_permits.release()
        raise


def _probe_saturated(source_path, env):
    return _fail(str(source_path or ""), "attachment_transfer_failed",
                 "readability probe saturated (unresponsive mount?)", [], env)


def _probe_timed_out(source_path, env, budget):
    return _fail(str(source_path or ""), "attachment_transfer_failed",
                 f"readability probe exceeded {budget:.1f}s (unresponsive mount?)",
                 [], env)


def _probe_env(runtime_id):
    return (runtime_id or os.environ.get("TERMINAL_ENV", "local")
            or "local").strip().lower()


async def verify_artifact_readable_async(
    source_path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    runtime_id: Optional[str] = None,
    budget_s: Optional[float] = None,
) -> ArtifactReceipt:
    """⭐ **事件循环上的调用方必须用这个。**

    🔴 上一版只把**系统调用**丢进线程,却仍在事件循环上同步
    ``Future.result(timeout=…)`` —— **有上限也照样是停顿**。而
    ``_build_media_placeholder()`` 会**逐个附件**调用 ⇒ 一条含 N 个异常附件的
    消息让所有会话停顿约 N×deadline;四个 worker 都卡住后连**健康附件**也要
    先等满一个 deadline 才失败。
    ⇒ 这里改成 ``await``,并接受调用方传入的**整条消息共享**预算 ``budget_s``。
    """
    env = _probe_env(runtime_id)
    budget = _PROBE_DEADLINE_S if budget_s is None else budget_s
    if budget <= 0:
        return _probe_timed_out(source_path, env, 0.0)
    try:
        future = _probe_submit(source_path, max_bytes, runtime_id)
    except Exception as exc:
        return _fail(str(source_path or ""), "attachment_transfer_failed",
                     f"probe could not run: {exc!r}", [], env)
    if future is None:
        return _probe_saturated(source_path, env)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), timeout=budget)
    except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
        # ⛔ 不 cancel:worker 卡在 syscall 上,取消无效;票由它自己的 finally 归还。
        return _probe_timed_out(source_path, env, budget)


def verify_artifact_readable(
    source_path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    runtime_id: Optional[str] = None,
) -> ArtifactReceipt:
    """同步入口 —— ⛔ **不许从协程里调用**(有门钉住调用点全集)。

    留给 CLI / 后台线程等本来就不在事件循环上的调用方。
    """
    env = _probe_env(runtime_id)
    try:
        future = _probe_submit(source_path, max_bytes, runtime_id)
    except Exception as exc:
        return _fail(str(source_path or ""), "attachment_transfer_failed",
                     f"probe could not run: {exc!r}", [], env)
    if future is None:
        return _probe_saturated(source_path, env)
    try:
        return future.result(timeout=_PROBE_DEADLINE_S)
    except concurrent.futures.TimeoutError:
        return _probe_timed_out(source_path, env, _PROBE_DEADLINE_S)


def _verify_artifact_readable_blocking(
    source_path: str,
    *,
    max_bytes: int = 512 * 1024 * 1024,
    runtime_id: Optional[str] = None,
) -> ArtifactReceipt:
    """⚠️ 只在隔离 worker 里跑 —— ⛔ 不许从事件循环直接调用。"""
    checks: list[str] = []
    env = (runtime_id or os.environ.get("TERMINAL_ENV", "local") or "local").strip().lower()

    # ── #1 输入非空（LS: empty_input）
    if not source_path or not str(source_path).strip():
        return _fail("", "attachment_delivery_failed", "empty input", checks, env)
    source_path = str(source_path)

    # ── #2 路径合法（LS: invalid_path）
    if "\x00" in source_path or not os.path.isabs(source_path):
        return _fail(source_path, "attachment_delivery_failed",
                     "path must be absolute and NUL-free", checks, env)
    checks.append("path_shape")

    # ── 执行环境闭集（SPEC §3）。⛔ 未知值不许按"本地大概可用"放行。
    if env not in KNOWN_ENVIRONMENTS:
        return _fail(source_path, "execution_environment_unsupported",
                     f"unknown TERMINAL_ENV={env!r}", checks, env)
    checks.append("env_known")

    # ── #3/#4 🔴 **这里刻意不照抄 LS 的「必须在某个根内」**
    #
    # LS 的 ``outside_agent_uploads`` 成立，是因为它的 artifact 只可能来自
    # **App agent uploads 根**，那是个封闭来源。Hermes 不是：``media_urls``
    # 由二十多个 adapter 各自产出，落点不止 ``_CACHE_DIRS`` 那 8 个目录。
    #
    # ⚠️ 我上一版把它照抄成「必须在 Hermes cache 根内」，后果是：
    #   * cache 根尚未创建（首次启动 / HERMES_HOME 指向新位置）⇒ **所有**附件
    #     被判不可读；
    #   * artifact 落在 cache 之外的合法位置 ⇒ 同样被误杀。
    #   实测两条既有门因此变红（video / mixed-attachment）。
    #   ⭐ **误杀合法附件比原缺陷严重得多** —— 原缺陷只是「模型拿到读不到的
    #   路径」，误杀是「所有附件都不给模型」。
    #
    # ⭐ 回到 SPEC §1.1 的不变量本身：它要求的是
    #   「``open`` + ``regular-file`` + ``read`` 成功」，**⛔ 没有「在某个根内」**。
    #   ⇒ 准入条件就只有可读性；路径来源的信任边界由产出方负责
    #   （``media_urls`` 是我们自己写进去的，⛔ 不是用户可控输入）。
    #
    # 逃逸防护仍在，但只用 **fd 与路径的 dev/ino 一致性**（挡 TOCTOU）+
    # ``O_NOFOLLOW``（挡符号链接），⛔ 不再要求落在白名单目录里。

    # ── #4b 🔴 **先按 lstat 拒绝非普通文件,再 open。**
    #
    # 缺陷:``os.open(fifo, O_RDONLY)`` 在**没有 writer** 的 FIFO 上会**无限阻塞**,
    # 而普通文件检查(下面 #7 的 ``fstat`` + ``S_ISREG``)在 open **之后**。
    # 本函数由 ``_build_media_placeholder()`` 在 **gateway 事件循环上同步调用**
    # ⇒ **一条这样的附件就能永久卡住所有会话**。
    # ⭐ 「检查放在可能阻塞的动作之后」= 检查根本跑不到。
    #
    # ⛔ 不删下面的 ``fstat`` 检查:``lstat`` 是**路径**上的判断,存在 TOCTOU;
    # 两道一起才闭合(lstat 挡住阻塞、fstat+dev/ino 挡住换文件)。
    try:
        pre = os.lstat(source_path)
    except FileNotFoundError as exc:
        return _fail(source_path, "attachment_expired", repr(exc), checks, env)
    except OSError as exc:
        return _fail(source_path, "attachment_transfer_failed", repr(exc), checks, env)
    if not stat.S_ISREG(pre.st_mode):
        return _fail(source_path, "attachment_transfer_failed",
                     f"not a regular file (lstat mode={pre.st_mode:o})", checks, env)
    checks.append("lstat_regular")

    # ── #5 O_NOFOLLOW 打开（LS: open_failed）
    # ⭐ ``O_NONBLOCK`` 是第二道保险:即便在 lstat 与 open 之间路径被换成 FIFO,
    #   open 也会立刻返回而不是挂住。⛔ 它不替代上面的 lstat(有些类型只有
    #   lstat 分得出),两道并存。
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(source_path, flags)
    except FileNotFoundError as exc:
        return _fail(source_path, "attachment_expired", repr(exc), checks, env)
    except PermissionError as exc:
        return _fail(source_path, "attachment_permission_denied", repr(exc), checks, env)
    except OSError as exc:
        # ELOOP（是符号链接）也落这里 —— O_NOFOLLOW 的本意
        return _fail(source_path, "attachment_transfer_failed", repr(exc), checks, env)
    checks.append("open_nofollow")

    try:
        # ── #6 fd 指向的确实是这个路径（LS: symlink_escape 的 TOCTOU 那一半）
        if not _fd_matches_path(fd, source_path):
            return _fail(source_path, "attachment_permission_denied",
                         "artifact changed between resolve and open", checks, env)
        checks.append("fd_matches_path")

        # ── #7 常规文件（LS: not_regular）
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return _fail(source_path, "attachment_transfer_failed",
                         f"not a regular file (mode={st.st_mode:o})", checks, env)
        checks.append("regular_file")

        # ── #8 非空（LS: empty_file）
        if st.st_size <= 0:
            return _fail(source_path, "attachment_expired", "zero-length file", checks, env)
        checks.append("non_empty")

        # ── #12 大小上限（LS: too_large）
        if st.st_size > max_bytes:
            return _fail(source_path, "attachment_transfer_failed",
                         f"size {st.st_size} exceeds {max_bytes}", checks, env)
        checks.append("size_limit")

        # ── #11 实际读（LS 是 Seek；⭐ 这里更强：挂载在但 I/O 坏时只有 read 会红）
        #
        # 🔴 **有界探测，⛔ 不读到 EOF。**
        #
        # 上一版是 ``while True: os.read(...)``，把**整个文件**读完只为算一个
        # ``sha256``。三件事叠在一起就是设备级故障：
        #   ① 本函数的 7 个调用点**全部**在 ``async def`` 里
        #      （``_handle_message`` / ``_handle_active_session_busy_message`` /
        #      ``_run_agent_inner``）⇒ 它是**同步**的，读多久就把
        #      **整个 gateway 事件循环**堵多久，**所有其它会话**一起卡；
        #   ② 默认上限 512 MiB ⇒ 一条大附件能堵到分钟级；
        #   ③ 那个 ``sha256`` **没有任何生产消费者**（``gateway/`` 内
        #      ``receipt.sha256`` 引用数为 0）——整趟磁盘 I/O 买的是空气。
        # 在 1C2G 无 swap 的板子上这属于整机级（同 2026-08-07 事故里
        # 「无上限的并发 / 缓冲 / 读取」那一类）。
        #
        # ⭐ 回到不变量本身：SPEC §1.1 要的是「``open`` + regular-file +
        #   ``read`` 成功」，**⛔ 从来没有要求读到 EOF**。一次成功的 read
        #   就已经证明了「挂载在、且 I/O 没坏」——这正是本检查要答的问题。
        #
        # ⛔ 上限不许拍脑袋：直接用本文件既有的 ``_PROBE_BYTES``（4096，
        # 名字本身就是「探测」，且原本就是这里的分块大小）。⛔ 不新造一个常量、
        # ⛔ 不照抄别处的数字——量纲不同就不能抄。
        read_total = 0
        try:
            chunk = os.read(fd, _PROBE_BYTES)
            read_total = len(chunk)
        except OSError as exc:
            return _fail(source_path, "attachment_transfer_failed",
                         f"read failed after {read_total} bytes: {exc!r}", checks, env)
        if read_total <= 0:
            return _fail(source_path, "attachment_transfer_failed",
                         "file reported non-zero size but read returned nothing",
                         checks, env)
        checks.append("read_ok")
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    # ── #13 目标环境回执（LS 没有这一层）
    model_path, evidence, failure = _target_path_and_evidence(source_path, env)
    if failure is not None:
        return _fail(source_path, failure, f"no receipt for env={env}", checks, env)
    checks.append(f"target:{evidence}")

    # ``sha256=None``：我们**只读了头 4 KiB**，⛔ 不许把前缀摘要塞进一个叫
    # ``sha256`` 的字段冒充全文摘要 —— 那是制造假的确定性，比没有这个字段更坏
    # （下一个人会拿它去比对完整性，而它根本对不上）。该字段目前**无生产消费者**；
    # 真需要全文摘要时，另开一个显式的、⛔ 不在事件循环上跑的入口。
    return ArtifactReceipt(
        ok=True, source_path=source_path, model_path=model_path,
        size=st.st_size, sha256=None, runtime_id=env,
        evidence=evidence, checks=tuple(checks),
    )


def _target_path_and_evidence(
    source_path: str, env: str
) -> "tuple[Optional[str], Optional[str], Optional[str]]":
    """``(model_path, evidence, failure_code)``。

    ⚠️ 三档，⛔ 不许合并 —— 合并就是在假装有证据：

    * ``local``：gateway 与模型**同一个文件系统**，上面那次 ``open+read``
      **就是**目标环境的回执 ⇒ ``verified_read``。
    * ``docker``：cache 目录以只读 bind mount 进容器，路径由
      ``to_agent_visible_cache_path`` 转换。**没有在容器内 open 过** ⇒
      只能标 ``mount_declared``。
      ⛔ 不因为"没有真回执"就拒绝 —— SPEC §7.3 要求已有 Docker cache 挂载
      的成功路径**不得因新校验被拒**（那会弄坏原来对的东西）。
    * 其余（singularity / modal / daytona / vercel_sandbox / ssh）：
      SPEC §3 逐条确认过它们**没有**把 artifact 的目标路径回写给提示 sink
      ⇒ 🔴 **明确失败**，⛔ 不许把主机路径原样塞给模型（那就是族 A 本身）。
    """
    if env == "local":
        return source_path, EVIDENCE_VERIFIED_READ, None

    if env == "docker":
        from tools.credential_files import map_cache_path_to_container

        mapped = map_cache_path_to_container(source_path)
        if not mapped:
            # 在 cache 根里却映射不出容器路径 ⇒ mount 清单与 cache 闭集不一致
            return None, None, "attachment_runtime_unavailable"
        return mapped, EVIDENCE_MOUNT_DECLARED, None

    if env == "modal":
        # 🔴 **本模块是本批新增的,这条兜底把 Modal 的附件能力整个砍掉了。**
        # 按 backend 名字无条件拒 ⇒ 每个本地缓存附件在进提示前都被换成「不可用」,
        # 原本能在默认 ``/root/.hermes`` 布局下工作的 Modal 会话**升级后完全失去
        # 图片与文档处理能力**。⭐ 这是「修复弄坏原来对的东西」。
        #
        # 生产路径 ``tools/environments/modal.py`` 已通过 ``iter_cache_files()``
        # 把 cache 挂进容器、并用 ``FileSyncManager`` 持续同步 ⇒ **目标路径与传输
        # 机制都已存在**,这里只是没消费它们。
        # ⇒ 复用与 docker **完全相同**的 cache 映射(⛔ 不另造一套);证据等级如实
        # 标 ``mount_declared`` —— 我们**没有**在容器内 open 过它。
        try:
            from tools.credential_files import map_cache_path_to_container
        except Exception:  # noqa: BLE001 — 取不到映射器就退回原来的明确失败
            return None, None, "attachment_runtime_unavailable"
        mapped = map_cache_path_to_container(source_path)
        if not mapped:
            # 不在 cache 根里 ⇒ 容器里确实没有它,仍然明确失败。
            # ⛔ 不许退回原始主机路径(那正是族 A 的现场)。
            return None, None, "attachment_runtime_unavailable"
        return mapped, EVIDENCE_MOUNT_DECLARED, None

    return None, None, "attachment_runtime_unavailable"
