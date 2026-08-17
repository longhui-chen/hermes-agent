"""媒体转码子进程:超时杀整组、stderr 有界、取消也回收(Codex PR #339 P1-3)。

⭐ 判据一律是**行为**(真的 spawn 一个进程、真的数它的孙子还在不在、真的量
内存里留了多少字节),⛔ 不是「源码里有没有出现 start_new_session 这个词」——
那是词法判据,换个写法就绕过去了。
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import time

import pytest

from tools.bounded_media_exec import (
    DEFAULT_TRANSCODE_TIMEOUT,
    MAX_STDERR_BYTES,
    TranscodeTimeout,
    run_media_subprocess,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="进程组语义是 POSIX 的;这些调用点是板端/服务端路径"
)

PY = sys.executable


def _script(body: str) -> list:
    return [PY, "-c", body]


def _run(coro):
    """本仓 tests/tools 的既有形态:同步测试内部 asyncio.run(⛔ 不用 pytest-asyncio)。

    ⚠️ home-guard-tests.yml 刻意不装 pytest-asyncio,而 tests/tools/ 在它的文件
    清单里 —— 用 @pytest.mark.asyncio 会让那道门整体红在收集期。
    """
    return asyncio.run(coro)


class TestHardTimeoutKillsTheWholeGroup:
    def test_hanging_child_is_killed_and_raises(self):
        started = time.monotonic()
        with pytest.raises(TranscodeTimeout):
            _run(run_media_subprocess(_script("import time; time.sleep(30)"), timeout=1.0))
        elapsed = time.monotonic() - started
        assert elapsed < 10, f"超时没生效,等了 {elapsed:.1f}s"

    def test_grandchild_is_killed_too_not_just_the_leader(self):
        """🔴 ⭐ 超时存在 ≠ 防护存在:``proc.kill()`` **只杀 leader**。

        让子进程 fork 一个孙子并把孙子的 pid 写到文件,然后让两者都挂住。
        超时之后**孙子必须也死了** —— 这正是 2026-08-07 那次
        「20+ 层孙子进程一个没死」的判据。
        """
        import tempfile

        fd, pidfile = tempfile.mkstemp(prefix="grandchild-", suffix=".pid")
        os.close(fd)
        body = (
            "import os,sys,time\n"
            "pid=os.fork()\n"
            "if pid==0:\n"
            "    open(%r,'w').write(str(os.getpid()))\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "time.sleep(60)\n" % pidfile
        )
        try:
            with pytest.raises(TranscodeTimeout):
                _run(run_media_subprocess(_script(body), timeout=2.0))

            raw = open(pidfile).read().strip()
            assert raw, "夹具没生效:孙子没来得及写 pid ⇒ 这条断言是空的"
            grandchild = int(raw)

            # 给内核一点时间把 SIGKILL 送达
            for _ in range(50):
                try:
                    os.kill(grandchild, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(grandchild, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pytest.fail(
                    f"孙子进程 {grandchild} 在超时后仍然活着 —— "
                    f"只杀了 leader,整组没杀"
                )
        finally:
            try:
                os.unlink(pidfile)
            except OSError:
                pass


class TestStderrIsBounded:
    def test_huge_stderr_is_capped_in_memory(self):
        """子进程狂刷 stderr ⇒ 我们只留上限那么多,⛔ 不把它读进内存。"""
        body = (
            "import sys\n"
            "for _ in range(400):\n"
            "    sys.stderr.write('x'*8192)\n"
            "sys.stderr.flush()\n"
        )
        rc, stderr = _run(run_media_subprocess(_script(body), timeout=30.0))

        assert rc == 0, "子进程被上限逻辑弄挂了 —— 管道没抽干会死锁"
        assert len(stderr) == MAX_STDERR_BYTES, (
            f"stderr 留了 {len(stderr)} 字节,上限是 {MAX_STDERR_BYTES}"
        )
        # 阳性对照:它确实写了远超上限的量,⛔ 否则上面的断言恒真
        assert 400 * 8192 > MAX_STDERR_BYTES

    def test_small_stderr_is_preserved_verbatim(self):
        """🔴 必须保持不变:够短的 stderr 要逐字拿到(错误分类依赖它)。"""
        body = "import sys; sys.stderr.write('Invalid data found when processing input')"
        rc, stderr = _run(run_media_subprocess(_script(body), timeout=30.0))
        assert stderr.decode() == "Invalid data found when processing input"


class TestCancellationReclaims:
    def test_cancel_kills_the_child_and_reraises(self):
        """🔴 ``CancelledError`` 继承自 ``BaseException``,⛔ 不进 ``except Exception``。"""
        import tempfile

        fd, pidfile = tempfile.mkstemp(prefix="cancelled-", suffix=".pid")
        os.close(fd)
        body = (
            "import os,time\n"
            "open(%r,'w').write(str(os.getpid()))\n"
            "time.sleep(60)\n" % pidfile
        )

        async def _drive():
            task = asyncio.ensure_future(
                run_media_subprocess(_script(body), timeout=30.0)
            )
            for _ in range(100):          # 等它真的把 pid 写出来
                await asyncio.sleep(0.05)
                if open(pidfile).read().strip():
                    break
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return int(open(pidfile).read().strip())

        try:
            child = _run(_drive())
            for _ in range(50):
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                pytest.fail(f"取消之后子进程 {child} 还活着 —— 失去了回收路径")
        finally:
            try:
                os.unlink(pidfile)
            except OSError:
                pass


class TestPreservedBehaviour:
    """🔴 正常转码的行为必须逐字不变。"""

    def test_successful_run_returns_zero_and_its_stderr(self):
        body = "import sys; sys.stderr.write('ok'); sys.exit(0)"
        assert _run(run_media_subprocess(_script(body), timeout=30.0)) == (0, b"ok")

    def test_nonzero_exit_is_reported_not_raised(self):
        body = "import sys; sys.stderr.write('boom'); sys.exit(3)"
        rc, stderr = _run(run_media_subprocess(_script(body), timeout=30.0))
        assert rc == 3 and stderr == b"boom"

    def test_default_timeout_matches_the_repo_precedent(self):
        """上限⛔ 不许拍脑袋:与仓内同类 async ffmpeg 调用点(qqbot/telegram)一致。"""
        assert DEFAULT_TRANSCODE_TIMEOUT == 60.0
