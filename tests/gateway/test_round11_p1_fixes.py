"""第十一轮 4 条 P1 —— 全部驱动**生产入口**,⛔ 不测判据函数本身。

⭐ 每条都配了「删掉调用点」逆改:门若仍绿,说明它测的是 helper 而不是
生产控制流 —— 那种门是**出生即空转**。
"""

import asyncio
import os
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_probe_pool():
    """⚠️ ``_probe_permits`` / ``_probe_pool`` 是**模块级全局**:上一条用例里被卡住的
    worker 会攥着票不放,污染下一条。⭐ 这不是产品缺陷(卡死的探测本来就该让后续
    请求快速降级),而是**测试隔离**问题 ⇒ 每条用例换一套全新的票和池。"""
    import concurrent.futures
    import threading as _t

    import gateway.model_readability as mr

    old_permits, old_pool = mr._probe_permits, mr._probe_pool
    mr._probe_permits = _t.BoundedSemaphore(mr._PROBE_MAX_WORKERS)
    mr._probe_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=mr._PROBE_MAX_WORKERS, thread_name_prefix="test-probe")
    try:
        yield
    finally:
        mr._probe_pool.shutdown(wait=False)
        mr._probe_permits, mr._probe_pool = old_permits, old_pool


# ═════════════════ A · 每次上传都要硬期限 ═════════════════

class TestArtifactUploadHasAHardDeadline:
    """🔴 ``_ARTIFACT_RETRY_BUDGET_S`` 只限制**重试前的 sleep**,⛔ 不限制上传本身;
    而 Feishu 等适配器最终经**无超时**的 ``_run_blocking()`` 等 SDK。
    notifier **串行**处理订阅 ⇒ 一次卡死阻断该 profile **后续所有**任务完成通知。
    """

    @staticmethod
    def _mixin(monkeypatch, send_impl, deadline=0.15):
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin

        m = GatewayKanbanWatchersMixin.__new__(GatewayKanbanWatchersMixin)
        monkeypatch.setattr(
            GatewayKanbanWatchersMixin, "_ARTIFACT_UPLOAD_TIMEOUT_S", deadline,
            raising=False)
        calls = []

        async def _send(*, adapter, chat_id, metadata, path):
            calls.append(path)
            return await send_impl(len(calls))

        m._send_one_artifact = _send
        return m, calls

    def test_a_hung_upload_does_not_block_the_queue(self, monkeypatch):
        """✅ **应该改变**:上传永不返回 ⇒ 期限到了放弃,交付队列继续走。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin

        async def _never(_n):
            await asyncio.Event().wait()

        m, calls = self._mixin(monkeypatch, _never)

        async def _drive():
            t0 = time.monotonic()
            ok = await GatewayKanbanWatchersMixin._upload_artifact_with_retry(
                m, adapter=object(), chat_id="c", metadata={}, path="/x/a.png",
                budget=[8.0])
            return ok, time.monotonic() - t0

        ok, elapsed = asyncio.run(_drive())
        assert elapsed < 5.0, f"仍在无限等待({elapsed:.1f}s)⇒ 该 profile 后续通知全被堵住"
        assert ok is False, "卡死的上传不能报成功"
        assert len(calls) == 1, (
            f"超时后又重传了({len(calls)} 次)⇒ 上传非幂等,**用户会收到重复文件**"
        )

    def test_a_fast_success_is_unchanged(self, monkeypatch):
        """🔴 **必须保持不变**:正常上传一次成功,⛔ 不许被期限改写。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin

        async def _ok(_n):
            return SimpleNamespace(success=True, error=None, retryable=False)

        m, calls = self._mixin(monkeypatch, _ok)
        ok = asyncio.run(GatewayKanbanWatchersMixin._upload_artifact_with_retry(
            m, adapter=object(), chat_id="c", metadata={}, path="/x/a.png",
            budget=[8.0]))
        assert ok is True and len(calls) == 1

    def test_a_retryable_failure_still_retries(self, monkeypatch):
        """🔴 **必须保持不变**:瞬时故障仍然重试 —— 期限⛔不许把重试一起砍掉。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin

        async def _flaky(n):
            if n == 1:
                return SimpleNamespace(success=False, error="ConnectionResetError: x",
                                       retryable=True)
            return SimpleNamespace(success=True, error=None, retryable=False)

        m, calls = self._mixin(monkeypatch, _flaky)
        ok = asyncio.run(GatewayKanbanWatchersMixin._upload_artifact_with_retry(
            m, adapter=object(), chat_id="c", metadata={}, path="/x/a.png",
            budget=[8.0]))
        assert ok is True, "瞬时故障后的重试被砍掉了"
        assert len(calls) == 2, f"重试没发生(调用 {len(calls)} 次)"

    def test_the_deadline_is_far_above_the_retry_budget(self):
        """⛔ 期限不许拍脑袋:必须远大于**等待**预算,否则会掐断正常的大文件上传。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        assert G._ARTIFACT_UPLOAD_TIMEOUT_S > G._ARTIFACT_RETRY_BUDGET_S * 5


# ═════════════════ B · exc_info 绕过脱敏 ═════════════════

class TestArtifactLoggingDoesNotLeakCredentials:
    """🔴 ``safe_exc`` 的 docstring 自己写着「⛔ 挡不住 ``exc_info=True``」并把那一面
    标为**开集**。logging 在 ``exc_info=True`` 下会**重新格式化原始异常对象**
    ⇒ 完整签名 URL / userinfo / query token 落进 ``agent.log`` ——
    一次附件上传失败就把渠道凭据**持久化**了。
    """

    SECRETY = ("403, message='Forbidden', "
               "url='https://user:hunter2@cdn.example.com/f?token=TOPSECRET' "
               "/Users/x/.secret/token.json")

    def test_safe_traceback_strips_every_credential_shape(self):
        from gateway.platforms.base import safe_traceback

        try:
            raise ValueError(self.SECRETY)
        except ValueError as exc:
            out = safe_traceback(exc)
        for leak in ("TOPSECRET", "hunter2", "user:hunter2", "/Users/x/.secret"):
            assert leak not in out, f"仍然泄漏 {leak!r}:{out[:200]}"

    def test_safe_traceback_stays_diagnosable(self):
        """🔴 **必须保持不变**:脱敏 ≠ 丢掉可定位性(⛔ 别砍成一句「操作失败」)。"""
        from gateway.platforms.base import safe_traceback

        try:
            raise ValueError(self.SECRETY)
        except ValueError as exc:
            out = safe_traceback(exc)
        assert out.startswith("ValueError:"), "异常类型名丢了 ⇒ 分不清超时和未授权"
        assert "cdn.example.com" in out, "host 也被抹了 ⇒ 定位不到是哪个渠道"
        assert "test_round11_p1_fixes.py" in out, "调用栈没了 ⇒ 排查无从下手"

    def test_the_delivery_call_sites_no_longer_pass_exc_info(self):
        """⭐ 判据落在**生产调用点**:凭据真正流过的那三处。"""
        src = pathlib.Path("gateway/kanban_watchers.py").read_text()
        assert "exc_info=True" not in src, (
            "kanban 交付路径仍在用 exc_info=True ⇒ logging 会重新格式化原始异常,"
            "把签名 URL 写进 agent.log"
        )
        assert src.count("safe_traceback(") >= 3, "脱敏 traceback 没接到调用点上"

    def test_safe_exc_itself_is_untouched(self):
        """🔴 **必须保持不变**:⛔ 不许顺手改 ``safe_exc`` 的行为(它遍地在用)。"""
        from gateway.platforms.base import safe_exc

        try:
            raise ValueError(self.SECRETY)
        except ValueError as exc:
            one = safe_exc(exc)
        assert "\n" not in one, "safe_exc 必须仍是**一行**"
        assert one.startswith("ValueError:") and "TOPSECRET" not in one


# ═════════════════ C · 取消后的 MCP cleanup 有硬期限 ═════════════════

class TestCancelledMcpCleanupIsBounded:
    """🔴 上一版**无条件**吞掉每一次后续取消 ⇒ discovery cleanup 赖在 MCP loop 里,
    **profile reload/unload 再也收不回 transport 和子进程**。
    ⚠️ 它是**独立于**所有外层 deadline 的旁路 —— 外面加多少超时都罩不住。
    """

    def test_a_cleanup_that_never_finishes_stops_swallowing_cancels(self, monkeypatch):
        """✅ **应该改变**:窗口耗尽后取消真正传播出去。

        🔴 **这道门第一版出生即空转。** 它在 ``asyncio.run()`` **返回之后**才断言
        ``task.done()`` —— 而关闭事件循环会把 pending task 一律取消 ⇒ 那条断言
        **恒真**,把 ``if False:`` 的逆改也判成了绿。
        ⭐ 判据必须在**循环还活着的时候**取样。
        """
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "_MCP_CANCEL_REAP_SECONDS", 0.2)

        async def _drive():
            async def _never():
                await asyncio.Event().wait()

            task = asyncio.ensure_future(
                mcp_tool._await_cleanup_until_complete(_never()))
            await asyncio.sleep(0.02)
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            while not task.done() and loop.time() - t0 < 3.0:
                task.cancel()                     # 反复取消 —— 模拟外层不断催
                await asyncio.sleep(0.02)
            # ⭐ 在**循环仍然活着**时取样,⛔ 不许等 asyncio.run 收尾后再问
            sample = (task.done(), loop.time() - t0)
            if not task.done():
                task.cancel()
            # ⛔ 收尾也要有界:否则缺陷版本会让整个 pytest 挂死,
            #   红在「超时」而不是红在**断言**上。
            await asyncio.wait({task}, timeout=1.0)
            return sample

        done, elapsed = asyncio.run(_drive())
        assert done, (
            f"{elapsed:.1f}s 内反复取消都被吞掉 ⇒ discovery cleanup 赖在 MCP loop 里,"
            "profile reload/unload 永久收不回 transport 和子进程"
        )
        assert elapsed >= 0.1, "根本没等到窗口 ⇒ 期限没生效,barrier 语义被破坏"

    def test_a_cleanup_that_finishes_normally_returns_its_value(self):
        """🔴 **必须保持不变**:没人取消时,原样返回 worker 的结果。"""
        import tools.mcp_tool as mcp_tool

        async def _drive():
            async def _work():
                await asyncio.sleep(0)
                return "done"

            return await mcp_tool._await_cleanup_until_complete(_work())

        assert asyncio.run(_drive()) == "done"

    def test_cancels_inside_the_window_are_still_swallowed(self, monkeypatch):
        """🔴 **必须保持不变**:窗口内仍然吞取消 —— 这正是 barrier 的意义,
        ⛔ 不许提前放手让 transport 半死不活。"""
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(mcp_tool, "_MCP_CANCEL_REAP_SECONDS", 30.0)

        async def _drive():
            async def _slow():
                await asyncio.sleep(0.15)
                return "finished-anyway"

            task = asyncio.ensure_future(
                mcp_tool._await_cleanup_until_complete(_slow()))
            await asyncio.sleep(0.02)
            task.cancel()                          # 窗口内的取消必须被吞掉
            return await asyncio.gather(task, return_exceptions=True)

        [res] = asyncio.run(_drive())
        assert res == "finished-anyway", (
            f"窗口内的取消没被吞掉({res!r})⇒ transport 会被半路丢下"
        )


# ═════════════════ D · 可读性探测离开事件循环 ═════════════════

class TestReadabilityProbeCannotBlockTheEventLoop:
    """🔴 ``O_NONBLOCK`` 只对 FIFO/设备有效,**对普通文件不提供任何墙钟上限**。
    ``_build_media_placeholder()`` 在事件循环上**同步**调用它
    ⇒ 一条挂在失联 NFS/FUSE 上的附件能**堵住所有会话**。
    """

    def test_a_hanging_probe_degrades_instead_of_hanging(self, monkeypatch):
        """✅ **应该改变**:探测卡死 ⇒ 期限内按 ``attachment_transfer_failed`` 降级。"""
        import gateway.model_readability as mr

        monkeypatch.setattr(mr, "_PROBE_DEADLINE_S", 0.2)
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: time.sleep(2))
        t0 = time.monotonic()
        r = mr.verify_artifact_readable("/mnt/dead-nfs/a.png")
        elapsed = time.monotonic() - t0
        assert elapsed < 3.0, f"探测仍会把事件循环堵 {elapsed:.1f}s ⇒ 所有会话一起卡"
        assert r.ok is False and r.failure_code == "attachment_transfer_failed"
        assert r.model_path is None, "失败时 ⛔ 不许给出 model_path"

    def test_the_event_loop_keeps_ticking_while_the_probe_hangs(self, monkeypatch):
        """⭐ 直接钉「不堵事件循环」这个不变量本身。"""
        import gateway.model_readability as mr

        monkeypatch.setattr(mr, "_PROBE_DEADLINE_S", 0.4)
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: time.sleep(2))
        ticks = []

        async def _drive():
            async def _ticker():
                while True:
                    ticks.append(1)
                    await asyncio.sleep(0.02)

            t = asyncio.create_task(_ticker())
            await asyncio.sleep(0)
            await asyncio.to_thread(mr.verify_artifact_readable, "/mnt/dead/a.png")
            t.cancel()

        asyncio.run(_drive())
        assert len(ticks) > 3, f"探测期间事件循环只跑了 {len(ticks)} 次 ⇒ 仍在被堵"

    def test_a_healthy_file_still_gets_a_full_receipt(self, tmp_path, monkeypatch):
        """🔴 **必须保持不变**:正常文件的回执逐字不变(⛔ 不许被降级路径污染)。"""
        import gateway.model_readability as mr

        f = tmp_path / "ok.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        r = mr.verify_artifact_readable(str(f))
        assert r.ok is True, f"正常文件被判失败:{r}"
        assert r.model_path == str(f)
        assert r.size == f.stat().st_size
        assert r.sha256 is None, "⛔ 仍然不许塞前缀摘要冒充 sha256"
        assert "read_ok" in r.checks and "regular_file" in r.checks

    def test_existing_failure_codes_are_unchanged(self, tmp_path, monkeypatch):
        """🔴 **必须保持不变**:既有的每个失败码逐条不变。"""
        import gateway.model_readability as mr

        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert mr.verify_artifact_readable("").failure_code == "attachment_delivery_failed"
        assert mr.verify_artifact_readable(
            str(tmp_path / "nope.png")).failure_code == "attachment_expired"
        empty = tmp_path / "empty.png"; empty.write_bytes(b"")
        assert mr.verify_artifact_readable(str(empty)).failure_code == "attachment_expired"

    def test_the_worker_pool_is_bounded(self):
        """⛔ 线程不许无界增长:每条卡死的附件都留一个线程就是第二处无界。"""
        import gateway.model_readability as mr

        assert 0 < mr._PROBE_MAX_WORKERS <= 8
        pool = mr._get_probe_pool()
        assert pool is mr._get_probe_pool(), "每次新建线程池 = 无界增长"
        assert pool._max_workers == mr._PROBE_MAX_WORKERS


# ═══════════ E · 第十二轮:探测的【同步等待】与【无界排队】 ═══════════

class TestProbeWaitsAsynchronously:
    """🔴 上一版只把**系统调用**丢进线程,却仍在事件循环上同步
    ``Future.result(timeout=5)`` —— **有上限也照样是停顿**。
    ``_build_media_placeholder()`` 逐个附件调用 ⇒ N 个异常附件 = N×deadline。
    """

    def test_the_loop_keeps_running_while_the_probe_is_stuck(self, monkeypatch):
        """✅ **应该改变**:探测卡住时事件循环照跑,⛔ 不再整段停顿。"""
        import gateway.model_readability as mr

        monkeypatch.setattr(mr, "_PROBE_DEADLINE_S", 0.4)
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: time.sleep(2))
        ticks = []

        async def _drive():
            async def _ticker():
                while True:
                    ticks.append(1)
                    await asyncio.sleep(0.02)

            t = asyncio.create_task(_ticker())
            await asyncio.sleep(0)
            r = await mr.verify_artifact_readable_async("/mnt/dead/a.png")
            t.cancel()
            return r

        r = asyncio.run(_drive())
        assert r.ok is False and r.failure_code == "attachment_transfer_failed"
        assert len(ticks) > 5, (
            f"探测期间事件循环只跑了 {len(ticks)} 次 ⇒ 仍在同步等待,所有会话一起停"
        )

    def test_n_bad_attachments_do_not_multiply_the_stall(self, monkeypatch):
        """✅ **应该改变**:整条消息共享一个预算,⛔ 不是每个附件各一份。"""
        import gateway.model_readability as mr

        monkeypatch.setattr(mr, "_PROBE_DEADLINE_S", 0.3)
        monkeypatch.setattr(mr, "_MESSAGE_PROBE_BUDGET_S", 0.6)
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: time.sleep(2))

        async def _drive():
            deadline = time.monotonic() + mr._MESSAGE_PROBE_BUDGET_S
            t0 = time.monotonic()
            out = []
            for _ in range(6):                     # 6 个坏附件
                out.append(await mr.verify_artifact_readable_async(
                    "/mnt/dead/x.png",
                    budget_s=max(0.0, deadline - time.monotonic())))
            return out, time.monotonic() - t0

        out, elapsed = asyncio.run(_drive())
        assert all(r.ok is False for r in out)
        assert elapsed < 6 * 0.3, (
            f"6 个坏附件花了 {elapsed:.2f}s ⇒ 预算没共享,停顿仍随附件数线性增长"
        )

    def test_the_message_budget_is_a_multiple_of_the_per_file_one(self):
        """⛔ 上限不许拍脑袋:整条消息预算必须由单附件预算推导。"""
        import gateway.model_readability as mr

        assert mr._MESSAGE_PROBE_BUDGET_S == mr._PROBE_DEADLINE_S * 2

    def test_the_placeholder_builder_is_a_coroutine(self):
        """⭐ 判据落在**生产入口**:它必须是协程,否则调用方只能同步等。"""
        import inspect

        from gateway.run import _build_media_placeholder

        assert inspect.iscoroutinefunction(_build_media_placeholder)


class TestProbeAdmissionIsBounded:
    """🔴 四个 worker 全卡死后,后续附件仍会进 ``ThreadPoolExecutor`` 的**无界队列**
    ⇒ 待处理 future / 参数 / 路径持续累积 ⇒ 约 2 GB 设备内存预算下最终 OOM。
    """

    def test_a_saturated_pool_rejects_instead_of_queueing(self, monkeypatch):
        """✅ **应该改变**:饱和时**当场拒绝**,⛔ 不排队。"""
        import gateway.model_readability as mr

        monkeypatch.setattr(mr, "_PROBE_DEADLINE_S", 30.0)   # 期限不该是它退出的原因
        released = threading.Event()
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: released.wait(5))
        try:
            wedged = [mr._probe_submit(f"/mnt/dead/{i}", 1, None)
                      for i in range(mr._PROBE_MAX_WORKERS)]
            assert all(w is not None for w in wedged), "前 N 个应当都拿到票"
            t0 = time.monotonic()
            r = mr.verify_artifact_readable("/mnt/dead/extra.png")
            elapsed = time.monotonic() - t0
            assert elapsed < 1.0, f"饱和后仍等了 {elapsed:.1f}s ⇒ 排进了无界队列"
            assert r.ok is False and "saturated" in (r.failure_detail or ""), (
                f"饱和没有被当场拒绝:{r.failure_detail!r}"
            )
        finally:
            released.set()
            for w in wedged:
                if w is not None:
                    w.result(timeout=15)

    def test_permits_come_back_when_the_mount_recovers(self, monkeypatch):
        """⭐ **回收策略**:票在 worker 的 finally 里归还 ⇒ 挂载点一恢复就自愈,
        ⛔ 不需要重建线程池(重建会把卡死线程变成无界增长)。"""
        import gateway.model_readability as mr

        released = threading.Event()
        monkeypatch.setattr(mr, "_verify_artifact_readable_blocking",
                            lambda *a, **k: released.wait(5))
        wedged = [mr._probe_submit(f"/mnt/dead/{i}", 1, None)
                  for i in range(mr._PROBE_MAX_WORKERS)]
        assert mr._probe_submit("/x", 1, None) is None, "前置:此刻应当饱和"
        released.set()
        for w in wedged:
            w.result(timeout=15)
        again = mr._probe_submit("/y", 1, None)
        assert again is not None, "挂载点恢复后票没有归还 ⇒ 探测永久停摆"
        again.result(timeout=15)

    def test_a_healthy_probe_still_returns_a_receipt(self, tmp_path, monkeypatch):
        """🔴 **必须保持不变**:没饱和时正常文件的回执逐字不变。"""
        import gateway.model_readability as mr

        f = tmp_path / "ok.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"y" * 64)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        r = asyncio.run(mr.verify_artifact_readable_async(str(f)))
        assert r.ok is True and r.model_path == str(f) and r.sha256 is None
        assert "read_ok" in r.checks

    def test_the_sync_entrypoint_still_works_for_non_loop_callers(self, tmp_path, monkeypatch):
        """🔴 **必须保持不变**:CLI / 后台线程等本来就不在循环上的调用方不受影响。"""
        import gateway.model_readability as mr

        f = tmp_path / "ok2.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"z" * 64)
        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert mr.verify_artifact_readable(str(f)).ok is True


# ═══════════ F · 第十三轮:LSP owner 期限 · detached owner · multiplex discovery ═══════════

class TestLspOwnerHasItsOwnDeadline:
    """🔴 LSP 那一族的**第四个位置**。调用方 ``self._loop.run(..., timeout=1.0)``
    只取消 waiter,owner task 照跑;barrier 捕获取消后又 shield **同一个** task
    ⇒ ``_cleanup_tasks`` / ``_retiring_clients`` 永不结算 ⇒ 后续 unload/reload
    持续报 already-in-progress,子进程也回收不掉。
    """

    def test_a_wedged_shutdown_is_force_terminated(self):
        """✅ **应该改变**:owner 内期限耗尽 ⇒ **真的把进程杀掉**,⛔ 不是「不等了」。"""
        from agent.lsp.manager import LSPService

        killed = []

        class _Proc:
            returncode = None
            def terminate(self): killed.append("terminate")
            def kill(self): killed.append("kill")

        class _Client:
            def __init__(self): self._proc = _Proc(); self._stopping = True
            async def shutdown(self): await asyncio.Event().wait()

        c = _Client()

        async def _drive():
            with pytest.raises(asyncio.TimeoutError):
                await LSPService._shutdown_client_for_retry(c)

        import agent.lsp.manager as m
        old = m._CLIENT_SHUTDOWN_TIMEOUT_S
        m._CLIENT_SHUTDOWN_TIMEOUT_S = 0.1
        try:
            asyncio.run(_drive())
        finally:
            m._CLIENT_SHUTDOWN_TIMEOUT_S = old
        assert killed, "期限耗尽却没有终止进程 ⇒ 子进程留下来,回收不掉"
        assert c._stopping is False, "🔴 必须保留**可重试**状态,⛔ 不能把 client 判死"

    def test_a_fast_shutdown_is_unchanged(self):
        """🔴 **必须保持不变**:正常关闭路径逐字不变(不杀进程、不抛异常)。"""
        from agent.lsp.manager import LSPService

        killed = []

        class _Proc:
            returncode = 0
            def terminate(self): killed.append(1)
            def kill(self): killed.append(1)

        class _Client:
            def __init__(self): self._proc = _Proc(); self._stopping = True
            async def shutdown(self): return None

        asyncio.run(LSPService._shutdown_client_for_retry(_Client()))
        assert not killed, "正常路径不该动进程"

    def test_a_failing_shutdown_still_restores_the_process(self):
        """🔴 **必须保持不变**:非超时失败仍然恢复活着的 process handle。"""
        from agent.lsp.manager import LSPService

        class _Proc:
            returncode = None
            def terminate(self): raise AssertionError("非超时失败不该强杀")
            def kill(self): raise AssertionError("非超时失败不该强杀")

        class _Client:
            def __init__(self): self._proc = _Proc(); self._stopping = True
            async def shutdown(self): raise RuntimeError("boom")

        c = _Client()

        async def _drive():
            with pytest.raises(RuntimeError):
                await LSPService._shutdown_client_for_retry(c)

        asyncio.run(_drive())
        assert c._stopping is False, "失败后没恢复可重试状态"

    def test_the_owner_deadline_matches_the_in_file_precedent(self):
        """⛔ 上限不许拍脑袋:与本文件 ``LSPClient.stop`` 的 join 同量纲。"""
        import agent.lsp.manager as m

        assert m._CLIENT_SHUTDOWN_TIMEOUT_S == 2.0
        assert m._CLEANUP_BARRIER_TIMEOUT_S > m._CLIENT_SHUTDOWN_TIMEOUT_S, (
            "barrier 窗口必须**大于** owner 期限,否则会在 owner 正常收尾前就放手"
        )


class TestDetachedUploadOwnerIsTracked:
    """🔴 ``wait_for`` 超时**只取消协程**,停不掉适配器 ``_run_blocking()`` 已提交
    到 executor 的线程。⇒ 连续任务先占满 SDK worker,之后把上传持续堆进**无界队列**。
    """

    @staticmethod
    def _mixin(monkeypatch, send_impl, deadline=0.1):
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        m = G.__new__(G)
        monkeypatch.setattr(G, "_ARTIFACT_UPLOAD_TIMEOUT_S", deadline, raising=False)
        calls = []

        async def _send(*, adapter, chat_id, metadata, path):
            calls.append(path)
            return await send_impl(len(calls))

        m._send_one_artifact = _send
        return m, calls

    def test_a_timed_out_upload_records_its_owner(self, monkeypatch):
        """✅ **应该改变**:超时后 owner 被登记下来(⛔ 不是「不等了就忘了」)。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        async def _never(_n):
            await asyncio.Event().wait()

        m, _ = self._mixin(monkeypatch, _never)
        asyncio.run(G._upload_artifact_with_retry(
            m, adapter=object(), chat_id="c", metadata={}, path="/x/a.png",
            budget=[8.0]))
        assert m._detached_uploads(), "迟到上传的 owner 没被保存 ⇒ 无法拒绝同类新上传"

    def test_a_second_upload_of_the_same_file_is_refused(self, monkeypatch):
        """✅ **应该改变**:owner 未结束前拒绝同类新上传,⛔ 不排队。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        async def _never(_n):
            await asyncio.Event().wait()

        m, calls = self._mixin(monkeypatch, _never)
        ad = object()

        async def _drive():
            await G._upload_artifact_with_retry(
                m, adapter=ad, chat_id="c", metadata={}, path="/x/a.png", budget=[8.0])
            return await G._upload_artifact_with_retry(
                m, adapter=ad, chat_id="c", metadata={}, path="/x/a.png", budget=[8.0])

        second = asyncio.run(_drive())
        assert second is False
        assert len(calls) == 1, (
            f"同一附件被重复提交 {len(calls)} 次 ⇒ 队列堆积 + 用户可能收到两份"
        )

    def test_the_owner_table_is_an_instance_attribute(self):
        """🔴 **必须保持不变**:⛔ 不许挂到类上(会跨 runner 共享)。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        assert not hasattr(G, "_detached_upload_owners")
        a, b = G.__new__(G), G.__new__(G)
        a._detached_uploads()["k"] = object()
        assert b._detached_uploads() == {}, "owner 表被跨实例共享了"

    def test_a_normal_upload_is_unchanged(self, monkeypatch):
        """🔴 **必须保持不变**:成功路径不留任何 owner 账。"""
        from gateway.kanban_watchers import GatewayKanbanWatchersMixin as G

        async def _ok(_n):
            return SimpleNamespace(success=True, error=None, retryable=False)

        m, calls = self._mixin(monkeypatch, _ok)
        ok = asyncio.run(G._upload_artifact_with_retry(
            m, adapter=object(), chat_id="c", metadata={}, path="/x/a.png", budget=[8.0]))
        assert ok is True and len(calls) == 1
        assert m._detached_uploads() == {}, "成功路径不该留账"


class TestMultiplexDiscoveryCoversEveryProfile:
    """🔴 **半条链**:状态改成了 profile-scoped,启动侧还是单 profile
    ⇒ 除启动时那个 profile 外,其余 profile 的 MCP 工具**一个都不注册**。
    """

    def test_discovery_starts_once_per_served_profile(self, monkeypatch, tmp_path):
        import gateway.run as gr

        homes = {n: tmp_path / n for n in ("default", "b", "c")}
        for h in homes.values():
            h.mkdir()
        started = []
        monkeypatch.setattr("hermes_cli.mcp_startup.start_background_mcp_discovery",
                            lambda **kw: started.append(kw.get("thread_name")))
        monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve",
                            lambda multiplex: [(n, h) for n, h in homes.items()])
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir",
                            lambda n: homes.get(n, tmp_path / n))
        monkeypatch.setattr(gr, "_multiplex_active_profile_name", lambda: "default")
        monkeypatch.setattr(gr, "_profile_runtime_scope",
                            lambda home: __import__("contextlib").nullcontext())

        gr._spawn_mcp_discovery(logger=gr.logger, multiplex=True)
        assert len(started) == 3, f"只为 {len(started)} 个 profile 起了 discovery,应为 3"
        assert started[0].endswith("default"), "active profile 应当最先就绪(既有行为)"
        assert gr._mcp_discovery_homes and len(gr._mcp_discovery_homes) == 3, (
            "⭐ 谁跟踪:已启动的 profile 没有被登记"
        )

    def test_single_profile_mode_is_unchanged(self, monkeypatch):
        """🔴 **必须保持不变**:非 multiplex 仍然只起一次、且返回 None。"""
        import gateway.run as gr

        started = []
        monkeypatch.setattr("hermes_cli.mcp_startup.start_background_mcp_discovery",
                            lambda **kw: started.append(kw.get("thread_name")))
        assert gr._spawn_mcp_discovery(logger=gr.logger, multiplex=False) is None
        assert started == ["mcp-discovery"], f"单 profile 路径变了:{started}"
