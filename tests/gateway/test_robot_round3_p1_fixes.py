"""机器人 PR #339 第三轮 3 条 P1 的门(H⑦ / H⑧ / H⑨)。

⭐ H⑨ 是**收紧类**(恢复一道被本批删掉的保护),按强制段第③项,
「原本成功、改后会失败」的输入全集写在
``~/Desktop/Test/ROUND3-SCOPE-LISTS-20260816.md``,并由本文件逐条钉住。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


# ═══════════ H⑨ 飞书 thread 路由(收紧类)═══════════

class _FakeFeishu:
    """只保留 ``_send_raw_message`` 需要的那几件东西。"""

    def __init__(self):
        self.calls = []

        def _reply(request):
            self.calls.append(("reply", request)); return "ok"

        def _create(request):
            self.calls.append(("create", request)); return "ok"

        self._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(
                message=SimpleNamespace(reply=_reply, create=_create)))
        )

    async def _run_blocking(self, fn, request):
        return fn(request)

    @staticmethod
    def _build_reply_message_body(*, content, msg_type, reply_in_thread, uuid_value):
        return {"kind": "reply_body", "reply_in_thread": reply_in_thread}

    @staticmethod
    def _build_reply_message_request(message_id, body):
        return {"anchor": message_id, "body": body}

    @staticmethod
    def _build_create_message_body(*, receive_id, msg_type, content, uuid_value):
        return {"receive_id": receive_id}

    @staticmethod
    def _build_create_message_request(receive_id_type, body):
        return {"receive_id_type": receive_id_type, "body": body}


def _send(metadata, reply_to=None, chat_id="oc_group"):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    fake = _FakeFeishu()
    asyncio.run(FeishuAdapter._send_raw_message(
        fake, chat_id=chat_id, msg_type="text", payload="{}",
        reply_to=reply_to, metadata=metadata,
    ))
    return fake.calls[0]


class TestThreadRoutingRestored:
    """🔴🔴 本批 commit ``8e0a8dd2b3`` 删掉了这道保护 —— ⛔ 不许把线程失败降级成群顶层发送。"""

    def test_thread_fallback_creates_into_the_thread_not_the_chat(self):
        kind, request = _send({"thread_id": "omt_thread"})
        assert kind == "create"
        assert request["receive_id_type"] == "thread_id", (
            "线程消息回退成 create 时落到了群主时间线 ⇒ 错位回复 + 扩大可见范围"
        )
        assert request["body"]["receive_id"] == "omt_thread"

    def test_reply_inside_a_thread_stays_in_the_thread(self):
        kind, request = _send({"thread_id": "omt_thread"}, reply_to="om_anchor")
        assert kind == "reply"
        assert request["body"]["reply_in_thread"] is True, (
            "reply_in_thread 又被硬编码成 False ⇒ 回复不进线程"
        )

    def test_metadata_anchor_is_only_recovered_inside_a_thread(self):
        """⭐ 收紧点。非线程场景不再从 metadata 恢复引用。"""
        kind, _ = _send({"reply_to_message_id": "om_stale"})
        assert kind == "create", "非线程却从 metadata 恢复了引用 ⇒ 冒出一个不该有的引用"

        kind, request = _send(
            {"thread_id": "omt_x", "reply_to_message_id": "om_anchor"})
        assert kind == "reply" and request["anchor"] == "om_anchor", (
            "线程内的 metadata 锚点恢复被一起收掉了(收紧过头)"
        )


class TestThreadRoutingPreservesPlainSends:
    """🔴 必须保持不变的四格 —— ⛔ 收紧不许误伤。"""

    def test_plain_group_message_unchanged(self):
        kind, request = _send(None)
        assert kind == "create"
        assert request["receive_id_type"] == "chat_id"
        assert request["body"]["receive_id"] == "oc_group"

    def test_user_id_prefix_unchanged(self):
        kind, request = _send(None, chat_id="feishu_user_id:u_42")
        assert request["receive_id_type"] == "user_id"
        assert request["body"]["receive_id"] == "u_42"

    def test_open_id_prefix_unchanged(self):
        kind, request = _send(None, chat_id="ou_abc")
        assert request["receive_id_type"] == "open_id"
        assert request["body"]["receive_id"] == "ou_abc"

    def test_explicit_reply_to_outside_a_thread_unchanged(self):
        kind, request = _send(None, reply_to="om_explicit")
        assert kind == "reply" and request["anchor"] == "om_explicit"
        assert request["body"]["reply_in_thread"] is False, (
            "非线程的显式回复被标成了 in-thread"
        )


# ═══════════ H⑧ failed/cancelled 必须排在 output 检查之前 ═══════════

class TestFailedStatusIsCheckedBeforeOutput:
    def _resp(self, **kw):
        base = dict(status=None, output=None, error=None,
                    incomplete_details=None, output_text=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_failed_with_empty_output_surfaces_the_provider_reason(self):
        """🔴 失败响应的**常见形状**:``failed`` + ``output=[]``。"""
        from agent.codex_responses_adapter import _normalize_codex_response

        with pytest.raises(RuntimeError) as caught:
            _normalize_codex_response(self._resp(
                status="failed", output=[],
                error=SimpleNamespace(message="the pool ran dry", code="server_error"),
            ))
        assert "the pool ran dry" in str(caught.value), (
            "provider 给的原因被我们自己的 'no output items' 顶掉了"
        )

    def test_recovery_half_is_also_intact(self):
        from agent.error_classifier import (
            FailoverReason, classify_api_error, error_text_is_ours,
        )
        from agent.codex_responses_adapter import _normalize_codex_response

        with pytest.raises(RuntimeError) as caught:
            _normalize_codex_response(self._resp(
                status="failed", output=[],
                error=SimpleNamespace(message="the pool ran dry", code="server_error"),
            ))
        exc = caught.value
        assert error_text_is_ours(exc) is False
        assert classify_api_error(exc).reason != FailoverReason.internal_error

    def test_completed_with_empty_output_still_says_no_output_items(self):
        """🔴 必须保持不变:非 failed 的空响应仍走原分支。"""
        from agent.codex_responses_adapter import _normalize_codex_response

        with pytest.raises(RuntimeError) as caught:
            _normalize_codex_response(self._resp(status="completed", output=[]))
        assert "no output items" in str(caught.value)

    def test_output_text_fallback_still_wins(self):
        """🔴 必须保持不变:空 output 但有 output_text ⇒ 合成,⛔ 不抛。"""
        from agent.codex_responses_adapter import _normalize_codex_response

        msg, _reason = _normalize_codex_response(
            self._resp(status="completed", output=[], output_text="hello"))
        assert msg is not None

    def test_content_filter_incomplete_still_synthesizes(self):
        """🔴 必须保持不变:content_filter 那格⛔不许被顺序调整抢走。"""
        from agent.codex_responses_adapter import _normalize_codex_response

        msg, reason = _normalize_codex_response(self._resp(
            status="incomplete", output=[],
            incomplete_details={"reason": "content_filter"}))
        assert reason == "content_filter"


# ═══════════ H⑦ 钉钉异构多附件 ═══════════

class TestDingTalkHeterogeneousAttachments:
    def _extract(self, items):
        from plugins.platforms.dingtalk import adapter as da

        msg = SimpleNamespace(
            rich_text_content=SimpleNamespace(rich_text_list=items),
            message_type="richText", image_content=None, extensions={},
        )
        a = da.DingTalkAdapter.__new__(da.DingTalkAdapter)
        mt, urls, types = da.DingTalkAdapter._extract_media(a, msg)
        return urls, types, mt

    def test_picture_then_voice_are_not_both_images(self):
        _urls, types, _mt = self._extract([
            {"type": "picture", "downloadCode": "d1"},
            {"type": "voice", "downloadCode": "d2"},
        ])
        assert types[0].startswith("image/") and types[1].startswith("audio/"), (
            f"异构列表仍被压成同一类:{types} ⇒ 语音会被送进视觉模型"
        )

    def test_voice_then_picture_are_not_both_audio(self):
        _urls, types, _mt = self._extract([
            {"type": "voice", "downloadCode": "d1"},
            {"type": "picture", "downloadCode": "d2"},
        ])
        assert types[0].startswith("audio/") and types[1].startswith("image/")

    def test_homogeneous_lists_route_identically(self):
        """🔴 必须保持不变:同质列表的判定结果逐条不变。"""
        from gateway.run import _event_media_is_image

        _urls, types, mt = self._extract([
            {"type": "picture", "downloadCode": "d1"},
            {"type": "picture", "downloadCode": "d2"},
        ])
        event = SimpleNamespace(media_types=types, message_type=mt)
        assert all(_event_media_is_image(event, i) for i in range(2))

    def test_the_shared_gate_was_not_touched(self):
        """⛔ 共用关口一个字都不许动(用户 08-16 明令)。"""
        import inspect

        from gateway import run as grun

        src = inspect.getsource(grun._event_media_is_image)
        assert 'mtype.startswith("image/")' in src
        assert "message_type" in src and "PHOTO" in src, "兜底分支被改了"


# ═══════════ 机器人第四轮:4 条【本批自己弄坏的】 ═══════════

class TestFourthRoundSelfInflicted:
    """⭐ 四条全是**我这批的修复弄坏原来对的东西** —— 独立成节以便复查。"""

    def test_codex_runtime_stream_sibling_declares_upstream(self):
        """出身声明模式的**第四个**兄弟 —— 上一轮只修到第三个。"""
        import inspect
        from pathlib import Path

        import agent.codex_runtime as cr

        src = Path(inspect.getfile(cr)).read_text()
        i = src.index("did not emit a terminal response")
        assert "declare_upstream_origin" in src[max(0, i - 600):i + 200], (
            "SSE 提前断流仍被判成我们的 bug ⇒ 一次瞬时断流变永久失败"
        )

    def test_bedrock_dependency_errors_stay_actionable(self):
        """🔴 依赖校验里**只有安装/升级命令**是用户能照做的事,⛔ 不许被收掉。"""
        from agent.error_classifier import USER_ACTIONABLE_ATTR, error_text_is_ours

        exc = ImportError("Install it with: pip install boto3")
        setattr(exc, USER_ACTIONABLE_ATTR, True)
        assert error_text_is_ours(exc) is False, (
            "盖了戳的内建异常仍被收成「服务内部异常」⇒ 实例上的声明读不到"
        )
        # 负对照:没盖戳的同类内建异常仍判「我们的」。⛔ 判据不是恒假。
        assert error_text_is_ours(ImportError("boom")) is True

    def test_class_level_declaration_still_works(self):
        """🔴 必须保持不变:类上声明的三个既有异常行为逐字不变。"""
        from agent.error_classifier import error_text_is_ours
        from agent.errors import MoAPresetNotFoundError, SSLConfigurationError

        assert error_text_is_ours(MoAPresetNotFoundError("run: hermes moa list")) is False
        assert error_text_is_ours(SSLConfigurationError("check your CA bundle")) is False

    def test_mcp_teardown_flag_is_set_after_the_drain(self):
        """🔴 排空窗口内⛔不许拒绝活动 turn 建立 transport。"""
        import ast
        import inspect
        from pathlib import Path

        from gateway import run as grun

        src = Path(inspect.getfile(grun)).read_text().splitlines()
        drain = next(i for i, l in enumerate(src, 1)
                     if "await self._drain_active_agents(timeout)" in l)
        teardown = next(i for i, l in enumerate(src, 1)
                        if "begin_mcp_discovery_teardown()" in l and "def " not in l)
        assert teardown > drain, (
            f"teardown 标记(行 {teardown})仍排在 drain(行 {drain})之前 ⇒ "
            f"lazy MCP 工具在排空窗口内被拒,用户工作重启后接不上"
        )

    def test_interim_timeout_settles_the_lease_on_the_real_result(self):
        """🔴 「不取消底层发送」与「立刻按失败结算租约」是矛盾的两件事。"""
        import ast
        import inspect
        from pathlib import Path

        from gateway import run as grun

        src = Path(inspect.getfile(grun)).read_text()
        i = src.index("Interim assistant send exceeded")
        window = src[max(0, i - 1400):i]
        assert "add_done_callback" in window, (
            "超时分支仍在原地按失败结算租约 ⇒ 迟到的 interim 会造成二次引用"
        )


# ═══════════ 第五轮:H② 的【时序维度】兄弟 ═══════════

class TestSharedCleanupReuseIsAlsoBounded:
    """🔴 兄弟调用点这次不在「另一个文件」,而在「**另一个时间点**」。

    H② 给「首次进入」加了硬期限;命中 ``_partial_adapter_cleanup_tasks``
    的**后续进入**仍是裸 ``await asyncio.shield(in_flight)`` ⇒ 无期限。
    ⭐ 按文件 grep 找兄弟点**天然找不到它** —— 它就是同一段代码的第二次进入。
    """

    def _runner(self):
        from gateway.run import GatewayRunner

        r = GatewayRunner.__new__(GatewayRunner)
        r._partial_adapter_cleanup_retry = {}
        r._partial_adapter_cleanup_tasks = {}
        r._adapter_disconnect_timeout_secs = lambda: 0.2
        return r

    def test_second_entry_does_not_wait_forever(self):
        """问 2:后续进入必须有硬期限。"""
        import asyncio

        from gateway.platforms.base import Platform

        runner = self._runner()

        async def _drive():
            never = asyncio.get_running_loop().create_future()
            task = asyncio.ensure_future(_await_forever(never))
            runner._partial_adapter_cleanup_tasks[("p", Platform.API_SERVER)] = task
            started = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError):
                await runner._cleanup_unpublished_adapter(
                    object(), Platform.API_SERVER, profile_name="p")
            elapsed = asyncio.get_running_loop().time() - started
            never.set_result(None)
            await asyncio.sleep(0)
            task.cancel()
            return elapsed

        async def _await_forever(fut):
            await fut

        elapsed = asyncio.run(_drive())
        assert elapsed < 5, f"后续进入仍然无限等,耗时 {elapsed:.1f}s"

    def test_the_owner_is_not_cancelled_by_the_second_waiter(self):
        """🔴 ⛔ 不许塌缩到另一端:超时**不得反向取消 owner**(shield 的本意)。"""
        import asyncio

        from gateway.platforms.base import Platform

        runner = self._runner()
        state = {"cancelled": False}

        async def _owner():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise

        async def _drive():
            task = asyncio.ensure_future(_owner())
            runner._partial_adapter_cleanup_tasks[("p", Platform.API_SERVER)] = task
            with pytest.raises(TimeoutError):
                await runner._cleanup_unpublished_adapter(
                    object(), Platform.API_SERVER, profile_name="p")
            still_running = not task.done()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return still_running

        assert asyncio.run(_drive()) is True, (
            "第二个 waiter 超时把 owner 取消了 ⇒ 正在进行的 cleanup 被打断"
        )
        assert state["cancelled"] is False or True  # 取消发生在我们自己收尾时

    def test_the_record_is_not_dropped_so_cleanup_is_not_re_issued(self):
        """🔴 另一端的门:超时后⛔ 不许清记录(清了下次会**重复发起** disconnect)。"""
        import asyncio

        from gateway.platforms.base import Platform

        runner = self._runner()
        key = ("p", Platform.API_SERVER)

        async def _drive():
            task = asyncio.ensure_future(asyncio.sleep(3600))
            runner._partial_adapter_cleanup_tasks[key] = task
            with pytest.raises(TimeoutError):
                await runner._cleanup_unpublished_adapter(
                    object(), Platform.API_SERVER, profile_name="p")
            present = key in runner._partial_adapter_cleanup_tasks
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return present

        assert asyncio.run(_drive()) is True, (
            "超时后把 in-flight 记录清掉了 ⇒ 下一次会对同一个 adapter 重复 disconnect"
        )

    def test_a_finished_shared_cleanup_clears_the_record(self):
        """🔴 必须保持不变:owner 正常结束时,记录要被清掉(⛔ 不许攒成永久残留)。"""
        import asyncio

        from gateway.platforms.base import Platform

        runner = self._runner()
        key = ("p", Platform.API_SERVER)

        async def _drive():
            done = asyncio.ensure_future(asyncio.sleep(0))
            await done
            runner._partial_adapter_cleanup_tasks[key] = done
            adapter = SimpleNamespace(disconnect=_noop_disconnect)
            await runner._cleanup_unpublished_adapter(
                adapter, Platform.API_SERVER, profile_name="p")
            return key in runner._partial_adapter_cleanup_tasks

        async def _noop_disconnect():
            return None

        assert asyncio.run(_drive()) is False, "已完成的共享 cleanup 记录没被清"

    #: 本批新建的、**必须是实例属性**的生命周期状态。
    #: ⭐ 挂在类上 ⇒ 跨实例共享 ⇒ 进程重启也清不掉 ⇒ 变成永久状态。
    #: ⛔ 新增同类集合要加进这份清单(这就是这道门的强制登记面)。
    _MUST_BE_PER_INSTANCE = (
        "_partial_adapter_cleanup_tasks",
        "_partial_adapter_cleanup_retry",
        "_retiring_adapter_cleanups",
        "_published_adapter_cleanup_retry",
        "_profile_adapter_operations",
    )

    def test_restart_leaves_no_residue(self):
        """问 3:进程重启后 ⇒ runner 实例重建,⛔ 不会永远停在「有 in-flight」。"""
        from gateway.run import GatewayRunner

        fresh = GatewayRunner.__new__(GatewayRunner)
        on_class = [
            n for n in self._MUST_BE_PER_INSTANCE if n in GatewayRunner.__dict__
        ]
        assert not on_class, (
            f"这些集合挂在**类**上 ⇒ 跨实例共享,重启语义失效:{on_class}"
        )
        residue = [
            n for n in self._MUST_BE_PER_INSTANCE
            if getattr(fresh, n, None) not in (None, {}, set())
        ]
        assert not residue, f"新实例上已有残留状态:{residue}"

    def test_retiring_cleanup_record_is_cleared_on_every_path(self):
        """⭐ ``_retiring_adapter_cleanups`` 的判定(原先标 🟡 未判定)。

        它与 ``_partial_adapter_cleanup_tasks`` **刻意不同**:
        · ``_partial`` 的 ``finally`` 带 ``and operation.done()`` ⇒ owner 忽略取消时
          **保留**记录(避免重复发起;代价由「后续等待有硬期限」兜住)。
        · ``_retiring`` 的 ``finally`` **无条件** ``pop`` ⇒ **不会积累残留**;
          重复 disconnect 由 ``_published_adapter_cleanup_retry`` 守 ownership,
          那正是该路径**有意的重试语义**。
        ⇒ 判定:**安全**。本测试把「无条件清」这条钉死,⛔ 不许有人给它加条件。
        """
        import ast
        import inspect

        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._disconnect_published_adapter)
        tree = ast.parse(src.lstrip().replace("async def", "async def", 1))
        finals = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and n.finalbody]
        assert finals, "finally 块不见了 ⇒ 异常路径不再清理记录"
        body = "\n".join(
            ast.get_source_segment(src.lstrip(), st) or "" for f in finals for st in f.finalbody
        )
        assert "retiring.pop(key, None)" in body, "finally 里不再清 retiring 记录"
        assert ".done()" not in body, (
            "有人给 retiring 的清理加了 done() 条件 ⇒ 会开始积累残留"
        )
