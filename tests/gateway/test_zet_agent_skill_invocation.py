"""ZET fork: explicit skill invocation (metadata.skill_slug) + reload fix.

Covers two zet_agent changes:

1. ``_expand_inbound_skill_invocation`` — the App's skill quick-pick inserts
   a visible ``/<slug>`` token into the input text AND sends
   ``metadata.skill_slug`` with the message (the client drops the field when
   the user edits the token away). Only that explicit field triggers
   expansion; the message text is NEVER sniffed for slash commands. The hook
   loads the requested skill, strips the display token(s) from the task text
   and rebuilds the message with the canonical CLI-slash scaffolding.

2. ``_handle_skills_reload`` — ``gateway_runner._session_db`` is the
   AsyncSessionDB facade whose methods return coroutines; the handler must
   await ``clear_all_system_prompts()`` (un-awaited it was a silent no-op and
   the coroutine object 500'd the JSON response).
"""

import asyncio
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server import _strip_skill_display_token
from gateway.platforms.zet_agent import ZetAgentAdapter


def _make_adapter() -> ZetAgentAdapter:
    return ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))


def _expand(adapter, message, slug="deep-research", session_id=None):
    """Drive the async hook to completion (it offloads to a worker thread)."""
    return asyncio.run(
        adapter._expand_inbound_skill_invocation(message, slug, session_id=session_id)
    )


def _patch_skill_layer(monkeypatch, *, known=("deep-research",), load_raises=False, disabled=False):
    """Stub the skill_commands entry points the hook (and the canonical
    ``build_skill_invocation_message`` it delegates to) reach lazily."""
    import tools.skill_usage as skill_usage
    import tools.skills_tool as skills_tool

    from gateway.session_context import get_session_env

    calls = {}

    def fake_is_disabled(name, platform=None):
        calls["disabled_check"] = (name, platform)
        return disabled

    monkeypatch.setattr(skills_tool, "_is_skill_disabled", fake_is_disabled)

    def fake_scan():
        import os as _os

        calls["scan_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
        calls["scan_env_platform"] = _os.environ.get("HERMES_PLATFORM")
        return {
            f"/{name}": {"name": name, "skill_dir": f"/fake/skills/{name}"}
            for name in known
        }

    def fake_load(identifier, task_id=None):
        if load_raises:
            raise RuntimeError("boom")
        calls["load_identifier"] = identifier
        calls["load_task_id"] = task_id
        calls["load_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
        return ({"content": "SKILL BODY"}, Path("/fake/skills/deep-research"), "deep-research")

    def fake_build(loaded_skill, skill_dir, activation_note, user_instruction="", **kwargs):
        calls["user_instruction"] = user_instruction
        calls["activation_note"] = activation_note
        return f"<<EXPANDED:{loaded_skill['content']}|task={user_instruction}>>"

    monkeypatch.setattr(skill_commands, "scan_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "get_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "_load_skill_payload", fake_load)
    monkeypatch.setattr(skill_commands, "_build_skill_message", fake_build)
    monkeypatch.setattr(skill_usage, "bump_use", lambda name: None)
    return calls


def test_requested_skill_expands(monkeypatch):
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    out = _expand(adapter, "/deep-research 研究黄金为什么下跌")
    assert out.startswith("<<EXPANDED:SKILL BODY")
    assert calls["user_instruction"] == "研究黄金为什么下跌"
    assert calls["load_identifier"] == "/fake/skills/deep-research"


def test_display_token_stripped_wherever_it_sits(monkeypatch):
    # The quick-pick appends the visible token at the cursor, so it can sit
    # anywhere and repeat after re-selects; standalone occurrences are display
    # artifacts and must be stripped from the task text. Slash sequences glued
    # to other text (paths) are NOT the token and must survive.
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()

    _expand(adapter, "帮我研究黄金 /deep-research")
    assert calls["user_instruction"] == "帮我研究黄金"

    _expand(adapter, "/deep-research 黄金\n/deep-research")
    assert calls["user_instruction"] == "黄金"

    _expand(adapter, "看下 repo/deep-research 目录 /deep-research")
    assert calls["user_instruction"] == "看下 repo/deep-research 目录"


def test_trusted_task_strips_only_the_selected_skill_display_token():
    message = (
        "/video-edit-workflow-mini 请总结 [file: /data/input.mp4]\n"
        "保留 repo/video-edit-workflow-mini 路径"
    )

    assert _strip_skill_display_token(
        message,
        "video-edit-workflow-mini",
    ) == (
        "请总结 [file: /data/input.mp4]\n"
        "保留 repo/video-edit-workflow-mini 路径"
    )


def test_slug_without_token_in_text_still_expands(monkeypatch):
    # The explicit field is authoritative — the server does not require the
    # display token to be present in the text at all.
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    out = _expand(adapter, "研究黄金")
    assert out.startswith("<<EXPANDED:")
    assert calls["user_instruction"] == "研究黄金"


def test_unknown_slug_passes_through(monkeypatch):
    # App inventory drift (stale panel, uninstalled skill) must fail open.
    _patch_skill_layer(monkeypatch, known=("other-skill",))
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original


def test_empty_slug_and_multimodal_pass_through(monkeypatch):
    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    assert _expand(adapter, "你好，帮我查天气", slug="") == "你好，帮我查天气"
    multimodal = [{"type": "text", "text": "研究黄金"}]
    assert _expand(adapter, multimodal) is multimodal


def test_load_failure_falls_back_to_original(monkeypatch):
    _patch_skill_layer(monkeypatch, load_raises=True)
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original


def test_expansion_binds_zet_agent_platform_and_restores(monkeypatch):
    # The expansion runs BEFORE the session is bound, so the hook must bind
    # the platform contextvar itself: without it scan/load resolve platform
    # None and skills.platform_disabled.zet_agent is silently ignored (a
    # skill disabled only for zet_agent would still expand). The binding must
    # also be token-restored — it must not leak past the hook.
    from gateway.session_context import get_session_env

    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    before = get_session_env("HERMES_SESSION_PLATFORM")
    _expand(adapter, "/deep-research 研究黄金")
    assert calls["scan_platform"] == "zet_agent"
    assert calls["load_platform"] == "zet_agent"
    assert get_session_env("HERMES_SESSION_PLATFORM") == before


def test_expanded_payload_uses_canonical_memory_scaffolding(monkeypatch):
    # Memory compatibility: the expanded payload must round-trip through
    # extract_user_instruction_from_skill_message (what MemoryManager.
    # _strip_skill_scaffolding calls). A bespoke activation note fails the
    # canonical-prefix check and the FULL skill body would be fed to memory
    # providers as if the user typed it.
    import tools.skill_usage as skill_usage
    import tools.skills_tool as skills_tool

    def fake_scan():
        return {"/deep-research": {"name": "deep-research", "skill_dir": "/fake/skills/deep-research"}}

    def fake_load(identifier, task_id=None):
        return ({"content": "SKILL BODY"}, None, "deep-research")

    monkeypatch.setattr(skill_commands, "scan_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "get_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "_load_skill_payload", fake_load)
    monkeypatch.setattr(skill_usage, "bump_use", lambda name: None)
    monkeypatch.setattr(skills_tool, "_is_skill_disabled", lambda name, platform=None: False)

    adapter = _make_adapter()
    out = _expand(adapter, "/deep-research 研究黄金为什么下跌")
    assert out.startswith(skill_commands._SKILL_INVOCATION_PREFIX)
    assert (
        skill_commands.extract_user_instruction_from_skill_message(out)
        == "研究黄金为什么下跌"
    )
    # Bare invocation (only the display token, no task text) → no user
    # content worth remembering: extract must return None so memory callers
    # skip the turn entirely.
    bare = _expand(adapter, "/deep-research")
    assert bare.startswith(skill_commands._SKILL_INVOCATION_PREFIX)
    assert skill_commands.extract_user_instruction_from_skill_message(bare) is None


def test_session_id_forwarded_as_builder_task_id(monkeypatch):
    # ${HERMES_SESSION_ID} templates / session-scoped skill state must resolve
    # against the REAL chat session: the hook forwards session_id as the
    # canonical builder's task_id (CLI/gateway slash parity).
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    _expand(adapter, "/deep-research 研究黄金", session_id="sess-42")
    assert calls["load_task_id"] == "sess-42"
    # No session (defensive default) → builder gets None, not "".
    _expand(adapter, "/deep-research 研究黄金", session_id="")
    assert calls["load_task_id"] is None


def test_saturated_expansion_fails_open_to_passthrough(monkeypatch):
    # The expansion semaphore bounds how many scan/load jobs can occupy the
    # shared default executor. When saturated the hook must fail OPEN — the
    # message passes through untouched instead of queueing behind other
    # expansions and starving agent runs.
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    monkeypatch.setattr(ZetAgentAdapter, "_SKILL_INVOKE_ACQUIRE_TIMEOUT", 0.05)

    async def _run():
        adapter._skill_invoke_semaphore = asyncio.Semaphore(0)  # all slots busy
        return await adapter._expand_inbound_skill_invocation(
            "/deep-research 黄金", "deep-research"
        )

    original = "/deep-research 黄金"
    assert asyncio.run(_run()) == original
    assert "load_identifier" not in calls, "saturated path must not touch the skills layer"


def test_platform_disabled_skill_passes_through_even_with_foreign_env(monkeypatch):
    # The disabled gate must use the EXPLICIT platform argument: the
    # resolution chain reads the HERMES_PLATFORM process env before the
    # contextvar, so an externally provisioned value would shadow the
    # binding. With the skill disabled for zet_agent the message must pass
    # through as plain text — no skill payload injected.
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    calls = _patch_skill_layer(monkeypatch, disabled=True)
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original
    assert calls["disabled_check"] == ("deep-research", "zet_agent")
    assert "user_instruction" not in calls, "disabled skill must never be built"


def test_contextvar_binding_outranks_foreign_platform_env(monkeypatch):
    # ZET fork 语义：技能层平台解析 ContextVar 优先于进程 HERMES_PLATFORM。
    # 外部注入的 env 不再遮蔽展开窗口的绑定（zet_agent 允许、他平台禁用的
    # skill 不会被误判为未安装），且展开不改任何进程全局状态——共存平台
    # 的并发线程看到的 env 原封不动。
    import os

    from agent.skill_commands import _resolve_skill_commands_platform
    from gateway.session_context import (
        pop_session_platform,
        push_session_platform,
        reset_session_vars,
    )

    # 组合运行时其它用例可能在本上下文残留已绑定的 contextvar——先清到
    # _UNSET,让「绑定外回退 env」的断言确定性成立。
    reset_session_vars()
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    out = _expand(adapter, "/deep-research 研究黄金")
    assert out.startswith("<<EXPANDED:")
    assert calls["scan_env_platform"] == "telegram", "process env must never be mutated"

    # resolver 单元断言：绑定内 contextvar 赢，绑定外回退 env。
    tok = push_session_platform("zet_agent")
    try:
        assert _resolve_skill_commands_platform() == "zet_agent"
    finally:
        pop_session_platform(tok)
    assert _resolve_skill_commands_platform() == "telegram"
    assert os.environ.get("HERMES_PLATFORM") == "telegram"


def test_cancel_racing_semaphore_acquire_never_starts_worker(monkeypatch):
    # ``asyncio.wait_for(sema.acquire())`` can swallow an external cancel when
    # its inner acquire task completes in the same loop turn. A disconnect at
    # that boundary must not continue into the blocking skill worker.
    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()

    class YieldOnceSemaphore:
        async def acquire(self):
            await asyncio.sleep(0)
            return True

        def release(self):
            raise AssertionError("an unacquired permit must not be released")

    adapter._skill_invoke_semaphore = YieldOnceSemaphore()

    async def _run():
        loop = asyncio.get_running_loop()
        with monkeypatch.context() as loop_patch:
            loop_patch.setattr(
                loop,
                "run_in_executor",
                lambda executor, fn: pytest.fail(
                    "cancelled expansion must not start an executor worker"
                ),
            )
            task = asyncio.create_task(
                adapter._expand_inbound_skill_invocation("x", "deep-research")
            )
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(_run())


def test_queued_cancel_refunds_semaphore_permit(monkeypatch):
    # executor 满载时 worker 还在队列里就被取消：fn 永不执行、finally 永不
    # 触发——必须由等待方当场退款，否则 4 次断开就把许可漏光（review P1）。
    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()

    async def _run():
        loop = asyncio.get_running_loop()
        never = loop.create_future()  # 模拟排队中的 executor future
        executor_called = asyncio.Event()

        def queue_forever(executor, fn):
            executor_called.set()
            return never

        # Restore the loop method before ``asyncio.run`` starts its own
        # default-executor shutdown; otherwise that shutdown is also routed
        # to ``never`` and the test process hangs after all assertions pass.
        with monkeypatch.context() as loop_patch:
            loop_patch.setattr(loop, "run_in_executor", queue_forever)
            task = asyncio.create_task(
                adapter._expand_inbound_skill_invocation("x", "deep-research")
            )
            await asyncio.wait_for(executor_called.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            sema = adapter._skill_invoke_semaphore
            assert sema._value == adapter._SKILL_INVOKE_MAX_CONCURRENCY, (
                "queued-cancel must refund the permit immediately"
            )

    asyncio.run(_run())


def test_cancelled_caller_does_not_leak_semaphore_permit(monkeypatch):
    # 取消等待方（客户端断开）不会停掉 executor 里的 worker；许可必须绑定
    # worker 实际完成才归还，否则连发-断开循环可绕过并发上限堆满共享
    # executor（review P1）。
    import threading

    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    started = threading.Event()
    release_worker = threading.Event()

    def slow_blocking(msg, slug, session_id=None):
        started.set()
        release_worker.wait(5)
        return "done"

    monkeypatch.setattr(
        adapter, "_expand_inbound_skill_invocation_blocking", slow_blocking
    )

    async def _run():
        task = asyncio.create_task(
            adapter._expand_inbound_skill_invocation("x", "deep-research")
        )
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        sema = adapter._skill_invoke_semaphore
        cap = adapter._SKILL_INVOKE_MAX_CONCURRENCY
        assert sema._value == cap - 1, "permit must stay held while the worker runs"
        release_worker.set()
        for _ in range(200):
            if sema._value == cap:
                break
            await asyncio.sleep(0.01)
        assert sema._value == cap, "permit must return when the worker finishes"

    asyncio.run(_run())


def test_on_settled_fires_exactly_once_per_lifecycle_path(monkeypatch):
    # on_settled 是资源核算(profile 活跃计数)的释放信号:每条生命周期路径
    # 都必须恰好一次——正常完成/直通/饱和直通/排队取消退款,以及关键的
    # 「等待方被取消而 worker 仍在跑」(此时必须等 worker 真正结束才结算,
    # 否则 /v1/profile/unload 会在展开中拆掉 runtime,review P1)。
    import threading

    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()

    settled = []
    # 正常完成
    _ = asyncio.run(adapter._expand_inbound_skill_invocation(
        "/deep-research 黄金", "deep-research", on_settled=lambda: settled.append("ok")))
    assert settled == ["ok"]

    # 直通(空 slug)
    settled.clear()
    _ = asyncio.run(adapter._expand_inbound_skill_invocation(
        "你好", "", on_settled=lambda: settled.append("pass")))
    assert settled == ["pass"]

    # running-cancel:worker 在跑时取消等待方——结算必须延后到 worker 结束。
    settled.clear()
    started = threading.Event()
    release_worker = threading.Event()

    def slow_blocking(msg, slug, session_id=None):
        started.set()
        release_worker.wait(5)
        return "done"

    monkeypatch.setattr(
        adapter, "_expand_inbound_skill_invocation_blocking", slow_blocking
    )

    async def _run():
        task = asyncio.create_task(adapter._expand_inbound_skill_invocation(
            "x", "deep-research", on_settled=lambda: settled.append("late")))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert settled == [], "must NOT settle while the worker still runs"
        release_worker.set()
        for _ in range(200):
            if settled:
                break
            await asyncio.sleep(0.01)
        assert settled == ["late"], "must settle exactly once when the worker finishes"

    asyncio.run(_run())


def test_base_api_server_hook_is_noop():
    # HR4: plain api_server behavior must be unchanged — the base hook is
    # identity for every shape.
    base = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    for value in ("/deep-research x", "hello", ["parts"], None):
        assert asyncio.run(
            base._expand_inbound_skill_invocation(value, "deep-research")
        ) is value


@pytest.mark.asyncio
async def test_skills_reload_awaits_async_session_db_flow(monkeypatch):
    """Flow: /v1/skills/reload against an AsyncSessionDB-style runner.

    Regression for the un-awaited ``clear_all_system_prompts()``: with the
    async facade the old code put a coroutine into the JSON response (500)
    and never actually cleared the rows. The fixed handler must await and
    return the real row count.
    """
    import agent.prompt_builder as prompt_builder

    monkeypatch.setattr(
        prompt_builder, "clear_skills_system_prompt_cache", lambda **kw: None
    )
    monkeypatch.setattr(skill_commands, "scan_skill_commands", dict)

    async def _clear():
        return 7

    session_db = type(
        "AsyncFacade", (), {"clear_all_system_prompts": lambda self: _clear()}
    )()
    runner = type(
        "Runner",
        (),
        {"_session_db": session_db, "invalidate_all_cached_agents": lambda self: 3},
    )()

    adapter = _make_adapter()
    adapter.gateway_runner = runner
    app = web.Application()
    app.router.add_post("/v1/skills/reload", adapter._handle_skills_reload)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/skills/reload", headers={"Authorization": "Bearer test-key"}
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["db_rows_cleared"] == 7
    assert data["invalidated_sessions"] == 3
    assert data["cleared"] is True
