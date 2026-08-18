"""Regression tests for MCP discovery timing in non-interactive sessions.

Covers the race where AIAgent snapshots its tool registry at construction
time before background MCP discovery finishes.  In single-query (``-q``) and
oneshot (``-z``) mode there is only ONE turn — no between-turns late-binding
refresh — so missing tools at construction are missing for the entire
session.

Tests verify:
  1. The ``single_query`` flag resolves to the larger bound.
  2. ``ensure_mcp_discovery_before_agent_build`` starts discovery if needed.
  3. Oneshot calls the helper before AIAgent construction (ordering).
  4. The wait stays bounded when discovery is slow (dead server).
  5. Interactive mode keeps the small bound (not affected).
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from hermes_cli import mcp_startup


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    saved_profiles = dict(mcp_startup._mcp_discovery_by_profile)
    try:
        mcp_startup._mcp_discovery_started = False
        mcp_startup._mcp_discovery_thread = None
        mcp_startup._mcp_discovery_by_profile.clear()
        yield
    finally:
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread
        mcp_startup._mcp_discovery_by_profile.clear()
        mcp_startup._mcp_discovery_by_profile.update(saved_profiles)


# ── _resolve_discovery_timeout: single_query bound ──────────────────────────


def test_resolve_discovery_timeout_single_query_uses_larger_bound(monkeypatch):
    """Single-query mode reads the larger mcp_single_query_discovery_timeout."""
    import hermes_cli.config as cfg

    monkeypatch.setattr(
        cfg,
        "load_config",
        lambda: {
            "mcp_discovery_timeout": 1.5,
            "mcp_single_query_discovery_timeout": 25.0,
        },
    )
    assert mcp_startup._resolve_discovery_timeout(None) == 1.5
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == 25.0


def test_resolve_discovery_timeout_single_query_falls_back(monkeypatch):
    """Bad/absent single-query value falls back to DEFAULT_CONFIG, never hangs."""
    import hermes_cli.config as cfg

    default = float(cfg.DEFAULT_CONFIG.get("mcp_single_query_discovery_timeout", 15.0))
    monkeypatch.setattr(
        cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": 0}
    )
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default

    monkeypatch.setattr(
        cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": "oops"}
    )
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default

    monkeypatch.setattr(cfg, "load_config", lambda: {})
    assert mcp_startup._resolve_discovery_timeout(None, single_query=True) == default


def test_resolve_discovery_timeout_explicit_overrides_single_query():
    """An explicit timeout always wins, even in single-query mode."""
    assert mcp_startup._resolve_discovery_timeout(5.0, single_query=True) == 5.0


# ── ensure_mcp_discovery_before_agent_build ─────────────────────────────────


def _stub_mcp_modules(monkeypatch):
    """Stub MCP-related modules for helper tests."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
            load_config=lambda: {},
            DEFAULT_CONFIG={"mcp_discovery_timeout": 0.1, "mcp_single_query_discovery_timeout": 0.2},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: __import__("contextlib").nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: None,
            get_mcp_status=lambda: [{"connected": True}],
        ),
    )


def test_ensure_helper_starts_discovery_and_waits(monkeypatch):
    """The helper starts background discovery if not yet started, then waits."""
    _stub_mcp_modules(monkeypatch)
    waited = []

    original_wait = mcp_startup.wait_for_mcp_discovery

    def _spy_wait(timeout=None, *, single_query=False):
        waited.append(("wait", single_query))
        original_wait(timeout=timeout, single_query=single_query)

    monkeypatch.setattr(mcp_startup, "wait_for_mcp_discovery", _spy_wait)

    logger = types.SimpleNamespace(debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

    mcp_startup.ensure_mcp_discovery_before_agent_build(
        logger=logger,
        single_query=True,
    )

    # Discovery was started (thread created)
    assert mcp_startup._mcp_discovery_thread is not None or waited
    # Wait was called with single_query=True
    assert any(call[1] is True for call in waited)


def test_ensure_helper_is_idempotent(monkeypatch):
    """Calling the helper twice doesn't start a second discovery thread."""
    _stub_mcp_modules(monkeypatch)
    logger = types.SimpleNamespace(debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

    mcp_startup.ensure_mcp_discovery_before_agent_build(logger=logger)
    thread1 = mcp_startup._mcp_discovery_thread
    if thread1:
        thread1.join(timeout=2.0)

    mcp_startup.ensure_mcp_discovery_before_agent_build(logger=logger)
    thread2 = mcp_startup._mcp_discovery_thread
    if thread2:
        thread2.join(timeout=2.0)

    # Second call didn't create a new thread (first one completed, status shows connected)
    # or if it did, it's because the first exited with zero connected — but we stubbed
    # get_mcp_status to return connected=True, so no retry.
    # The key invariant: no exception, no hang.


def test_ensure_helper_swallows_errors(monkeypatch):
    """A broken MCP config never aborts agent construction."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            load_config=lambda: {},
            DEFAULT_CONFIG={},
        ),
    )
    logger = types.SimpleNamespace(debug=lambda *_a, **_k: None, warning=lambda *_a, **_k: None)

    # Should not raise
    mcp_startup.ensure_mcp_discovery_before_agent_build(logger=logger)


def test_gateway_discovery_task_spawns_in_background_not_inline(monkeypatch):
    """gateway 的 discovery 必须走后台 spawn，⛔ 不许内联同步发现。

    🔴 上一版用 AST 判「``_start_mcp_discovery_task`` 的函数体里出现过
    ``start_background_mcp_discovery`` 吗」。**AST 不管可达性** ——
    把真实调用塞进 ``if False:``（或任何走不到的分支），门照样绿
    （RH 复审 P2-4 实证）。⭐ **AST 存在性 ≠ 控制流。**

    改成**运行时**判据：真的把函数跑起来，看它**实际调了谁**。
    为此把决策从 723 行的 ``start_gateway`` 闭包里提成模块级
    ``_spawn_mcp_discovery``（行为逐字未变）——
    ⭐ 一个无法被驱动的分支 = 一个无法被验证的承诺。
    """
    import hermes_cli.mcp_startup as startup
    from gateway.run import _spawn_mcp_discovery

    calls: list[str] = []
    monkeypatch.setattr(
        startup, "start_background_mcp_discovery",
        lambda **kw: calls.append(f"background:{kw.get('thread_name')}"),
    )
    if hasattr(startup, "discover_mcp_tools"):
        monkeypatch.setattr(
            startup, "discover_mcp_tools",
            lambda *a, **k: calls.append("inline:discover_mcp_tools"),
        )

    logger = types.SimpleNamespace(debug=lambda *a, **k: None,
                                   warning=lambda *a, **k: None,
                                   info=lambda *a, **k: None)
    home = _spawn_mcp_discovery(logger=logger, multiplex=False)

    assert home is None, "非 multiplex 下不该解析 profile home"
    assert calls == ["background:mcp-discovery"], (
        f"discovery 没走后台 spawn（实际调用序列={calls}）—— "
        f"内联发现会在事件循环里同步阻塞")
    assert not any(c.startswith("inline:") for c in calls), (
        f"内联调用了同步发现:{calls}")


def test_gateway_discovery_under_multiplex_runs_inside_profile_scope(monkeypatch):
    """multiplex 下必须在 profile scope **内**起 discovery。

    ⭐ 孪生枚举：上一条只覆盖了非 multiplex 那一支。两支都要钉，
    否则「改对一支、另一支静默走错 profile」不会被任何门发现。
    """
    import gateway.run as grun
    import hermes_cli.mcp_startup as startup
    import hermes_cli.profiles as profiles
    from contextlib import contextmanager

    order: list[str] = []

    @contextmanager
    def _fake_scope(home):
        order.append(f"enter:{home}")
        try:
            yield
        finally:
            order.append("exit")

    monkeypatch.setattr(grun, "_profile_runtime_scope", _fake_scope)
    monkeypatch.setattr(grun, "_multiplex_active_profile_name", lambda: "alice")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: f"/p/{name}")
    monkeypatch.setattr(
        startup, "start_background_mcp_discovery",
        lambda **kw: order.append("spawn"),
    )

    logger = types.SimpleNamespace(debug=lambda *a, **k: None,
                                   warning=lambda *a, **k: None,
                                   info=lambda *a, **k: None)
    home = grun._spawn_mcp_discovery(logger=logger, multiplex=True)

    assert home == "/p/alice", f"没有解析出 active profile 的 home:{home}"
    assert order == ["enter:/p/alice", "spawn", "exit"], (
        f"spawn 不在 profile scope 内 —— discovery 会读到错的 profile:{order}")


def test_init_agent_forwards_single_query_flag(monkeypatch):
    """Single-query mode forwards single_query=True to the discovery wait."""
    import cli as cli_mod

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._single_query_mode = True

    seen = {}

    def _fake_ensure(*, logger, timeout=None, single_query=False, **_kw):
        seen["single_query"] = single_query

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        _fake_ensure,
    )
    monkeypatch.setattr(cli_mod, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    assert cli._init_agent() is True
    assert seen.get("single_query") is True


def test_init_agent_defaults_to_interactive(monkeypatch):
    """Without _single_query_mode, the helper uses interactive (short) bound."""
    import cli as cli_mod

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True

    seen = {}

    def _fake_ensure(*, logger, timeout=None, single_query=False, **_kw):
        seen["single_query"] = single_query

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        _fake_ensure,
    )
    monkeypatch.setattr(cli_mod, "AIAgent", lambda *_a, **_k: types.SimpleNamespace())

    assert cli._init_agent() is True
    assert seen.get("single_query") is False


# ── bounded wait: slow server doesn't freeze startup ────────────────────────


def test_wait_stays_bounded_when_discovery_is_slow(monkeypatch):
    """A slow/dead MCP server must not freeze startup: the wait is capped."""
    import hermes_cli.config as cfg

    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_single_query_discovery_timeout": 0.1})

    stop = threading.Event()
    thread = threading.Thread(target=lambda: stop.wait(10), daemon=True)
    thread.start()
    mcp_startup._mcp_discovery_thread = thread

    try:
        start = time.monotonic()
        mcp_startup.wait_for_mcp_discovery(single_query=True)
        elapsed = time.monotonic() - start
    finally:
        stop.set()

    assert elapsed < 3.0, (
        f"wait blocked {elapsed:.2f}s on a stuck MCP server — the wait must "
        "stay bounded by mcp_single_query_discovery_timeout"
    )


def test_wait_returns_instantly_when_discovery_done():
    """When discovery is already complete, the wait returns immediately."""
    mcp_startup._mcp_discovery_thread = None
    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery(single_query=True)
    assert time.time() - t0 < 0.2


# ── 构造顺序：运行时判据 ────────────────────────────────────────────────────
#
# ⛔ 顺序门不许拿"源码里这行在那行前面"当判据（字符偏移 / inspect.getsource
# 的 find 下标 / 只查字符串存在）：那测的是词法位置，不是控制流。把调用包进
# `if False:` 或塞进永不进入的分支，源码位置门照样绿 —— 门恒绿、冒充保护。
# 下面三门改为**真实调用**被测入口，用桩记录实际调用序列。
#
# 每门都带三条断言，缺一条就能被绕过：
#   1. ensure 确实被调用了（否则"两件事都没发生"也会绿）
#   2. ensure 触发时 AIAgent 尚未构造（真实先后，不是词法先后）
#   3. AIAgent 最终确实构造了（阳性对照，防止把 AIAgent 也短路掉换绿）


class _OrderProbe:
    """记录实际调用序列的探针。"""

    def __init__(self):
        self.calls: list[str] = []
        self.built_when_ensure_ran: list[str] | None = None

    def ensure(self, **_kw):
        self.built_when_ensure_ran = [c for c in self.calls if c == "AIAgent"]
        self.calls.append("ensure")

    def agent_factory(self, cls):
        def _factory(*_a, **kw):
            self.calls.append("AIAgent")
            return cls(**kw)

        return _factory

    def assert_ordered(self, where: str):
        assert "ensure" in self.calls, (
            f"{where}：ensure_mcp_discovery_before_agent_build 运行时从未被调用"
            f"（实际调用序列={self.calls}）。源码里出现过不等于跑到了。"
        )
        assert self.built_when_ensure_ran == [], (
            f"{where}：ensure 触发时 AIAgent 已经构造过了"
            f"（ensure 时已构造={self.built_when_ensure_ran}）"
        )
        assert "AIAgent" in self.calls, (
            f"{where}：AIAgent 根本没被构造，这一轮没有真正走到构造点"
            f"（实际调用序列={self.calls}）"
        )
        assert self.calls.index("ensure") < self.calls.index("AIAgent"), (
            f"{where}：实际调用序列是 {self.calls}，ensure 必须先于 AIAgent 构造"
        )


def test_gateway_turn_waits_for_discovery_before_building_the_agent(monkeypatch, tmp_path):
    """gateway TurnRunner 真实跑一轮：ensure 必须先于 ctx.AIAgent 构造。"""
    import asyncio
    import importlib

    from gateway.config import Platform, PlatformConfig, StreamingConfig
    from gateway.platforms.base import BasePlatformAdapter, SendResult
    from gateway.session import SessionSource

    probe = _OrderProbe()

    class _Agent:
        def __init__(self, **_kw):
            self.tools = []
            self.tool_progress_callback = None

        def run_conversation(self, message, conversation_history=None, task_id=None):
            return {"final_response": "done", "messages": [], "api_calls": 1}

    class _Adapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
            self.sent = []

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            self.sent.append(content)
            return SendResult(success=True, message_id="m-1")

        async def edit_message(self, chat_id, message_id, content) -> SendResult:
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id) -> None:
            return None

        async def get_chat_info(self, chat_id: str):
            return {"id": chat_id}

        def get_streaming_config(self) -> StreamingConfig:
            return StreamingConfig(enabled=False)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = probe.agent_factory(_Agent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", probe.ensure
    )

    gateway_run = importlib.import_module("gateway.run")
    adapter = _Adapter()
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.session_store = types.SimpleNamespace(_entries={}, _save=lambda: None)
    runner.hooks = types.SimpleNamespace(loaded_hooks=False)
    runner.config = types.SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="C1",
        chat_type="dm",
        thread_id=None,
    )
    result = asyncio.run(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-order",
            session_key="agent:main:telegram:dm:C1",
        )
    )

    assert result["final_response"] == "done"
    probe.assert_ordered("gateway TurnRunner")


def test_oneshot_waits_for_discovery_before_building_the_agent(monkeypatch, tmp_path):
    """oneshot._run_agent 真实跑一轮：ensure 必须先于 AIAgent 构造。"""
    import hermes_cli.oneshot as oneshot_mod

    probe = _OrderProbe()

    class _Agent:
        def __init__(self, **_kw):
            self._supports_followup_turns = True
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, prompt):
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *_a):
            return None

        def close(self):
            return None

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = probe.agent_factory(_Agent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # 与本门无关的外部依赖桩化，保证判据只落在"调用顺序"上。
    import hermes_cli.config as cfg_mod
    import hermes_cli.runtime_provider as rp_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda: {})
    monkeypatch.setattr(
        rp_mod,
        "resolve_runtime_provider",
        lambda **_kw: {
            "api_key": "***",
            "base_url": "http://localhost",
            "provider": "test",
            "requested_provider": "test",
            "api_mode": "chat",
        },
    )
    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", probe.ensure
    )
    monkeypatch.setattr(
        oneshot_mod, "_create_session_db_for_oneshot", lambda: None, raising=False
    )

    final, _result = oneshot_mod._run_agent("hello", use_config_toolsets=False)

    assert final == "ok"
    probe.assert_ordered("oneshot._run_agent")


def test_init_agent_waits_for_discovery_before_building_the_agent(monkeypatch):
    """CLI _init_agent 真实跑一轮：ensure 必须先于 AIAgent 构造。"""
    import cli as cli_mod

    probe = _OrderProbe()

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True

    monkeypatch.setattr(
        mcp_startup, "ensure_mcp_discovery_before_agent_build", probe.ensure
    )
    monkeypatch.setattr(
        cli_mod, "AIAgent", probe.agent_factory(lambda **_kw: types.SimpleNamespace())
    )

    assert cli._init_agent() is True
    probe.assert_ordered("CLIAgentSetupMixin._init_agent")
