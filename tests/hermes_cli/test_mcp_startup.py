"""Regression tests for bounded/lazy CLI MCP startup."""

from __future__ import annotations

from argparse import Namespace
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
import time
import types

import pytest

import cli as cli_mod
from hermes_cli import main as main_mod
from hermes_cli import mcp_startup


@pytest.fixture(autouse=True)
def _reset_mcp_startup_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    saved_profiles = dict(mcp_startup._mcp_discovery_by_profile)
    saved_teardown = mcp_startup._mcp_discovery_teardown_started
    try:
        mcp_startup._mcp_discovery_started = False
        mcp_startup._mcp_discovery_thread = None
        mcp_startup._mcp_discovery_by_profile.clear()
        mcp_startup._mcp_discovery_teardown_started = False
        yield
    finally:
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread
        mcp_startup._mcp_discovery_by_profile.clear()
        mcp_startup._mcp_discovery_by_profile.update(saved_profiles)
        mcp_startup._mcp_discovery_teardown_started = saved_teardown


def _agent_args(**overrides) -> Namespace:
    base = {
        "accept_hooks": False,
        "command": "chat",
        "cron_command": None,
        "gateway_command": None,
        "mcp_action": None,
        "tui": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat(monkeypatch):
    stop = threading.Event()
    calls = {"mcp": 0}

    def _blocking_discover():
        calls["mcp"] += 1
        stop.wait()

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
            load_config=lambda: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda *_a, **_k: None),
    )
    # Stub mcp_oauth so the background thread doesn't pay the real (cold,
    # ~0.75s) ``tools.mcp_oauth`` import before calling discovery. This test
    # asserts the *backgrounding contract* (main thread returns fast, discovery
    # runs off-thread), not OAuth suppression — the unrelated import latency
    # would otherwise blow the polling deadline on a loaded CI runner.
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_blocking_discover),
    )

    try:
        start = time.monotonic()
        main_mod._prepare_agent_startup(_agent_args())
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
        deadline = time.monotonic() + 3.0
        while calls["mcp"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls["mcp"] == 1
        assert mcp_startup._mcp_discovery_thread is not None
        assert mcp_startup._mcp_discovery_thread.is_alive()
    finally:
        stop.set()


def test_background_mcp_discovery_suppresses_interactive_oauth(monkeypatch):
    state = {"active": False, "during_discover": None}

    class SuppressInteractiveOAuth:
        def __enter__(self):
            state["active"] = True

        def __exit__(self, *_exc):
            state["active"] = False

    def _discover():
        state["during_discover"] = state["active"]

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"url": "https://mcp.example.test/mcp"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(
            suppress_interactive_oauth=lambda: SuppressInteractiveOAuth(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_discover),
    )

    mcp_startup.start_background_mcp_discovery(
        logger=types.SimpleNamespace(debug=lambda *_a, **_k: None),
        thread_name="test-mcp-discovery",
    )
    assert mcp_startup._mcp_discovery_thread is not None
    mcp_startup._mcp_discovery_thread.join(timeout=1.0)

    assert state["during_discover"] is True
    assert state["active"] is False


def test_background_mcp_discovery_is_single_flight_per_profile(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    release = threading.Event()
    started = []

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )

    def _discover():
        from hermes_constants import get_hermes_home

        started.append(str(get_hermes_home().resolve()))
        release.wait(timeout=2)

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_discover),
    )
    logger = _retry_logger()
    homes = [tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"]
    try:
        for home in homes:
            token = set_hermes_home_override(home)
            try:
                mcp_startup.start_background_mcp_discovery(
                    logger=logger,
                    thread_name=f"discover-{home.name}",
                )
                mcp_startup.start_background_mcp_discovery(
                    logger=logger,
                    thread_name=f"duplicate-{home.name}",
                )
            finally:
                reset_hermes_home_override(token)

        deadline = time.monotonic() + 2
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert set(started) == {str(home.resolve()) for home in homes}
        assert len(mcp_startup._mcp_discovery_by_profile) == 2
        assert mcp_startup.join_all_mcp_discovery(timeout=0) is False
    finally:
        release.set()
        for thread in list(mcp_startup._mcp_discovery_by_profile.values()):
            if thread is not None:
                thread.join(timeout=2)
    assert mcp_startup.join_all_mcp_discovery(timeout=0) is True


def test_background_discovery_copies_profile_secret_scope(monkeypatch, tmp_path):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    observed = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )

    def _discover():
        from agent.secret_scope import get_secret

        observed.append(get_secret("WECOM_CLI_CONFIG_DIR"))

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=_discover),
    )
    home = tmp_path / "profiles" / "secret-a"
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"WECOM_CLI_CONFIG_DIR": "/profile-a/creds"})
    try:
        mcp_startup.start_background_mcp_discovery(
            logger=_retry_logger(), thread_name="secret-scope-discovery"
        )
        thread = next(iter(mcp_startup._mcp_discovery_by_profile.values()))
        assert thread is not None
        thread.join(timeout=2)
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
    assert observed == ["/profile-a/creds"]


def _module_level_assigned_names(module) -> set[str]:
    """枚举 module-level 赋值名（⛔ 不按形状过滤）。

    ⚠️ 只遍历 `tree.body` 是不够的：`try` / `if` / `with` 块里的赋值同样是
    module-level 状态，却不在 `tree.body` 里。典型形态——

        try:
            import foo
            _foo_available = True
        except ImportError:
            _foo_available = False

    它是标量所以运行时容器判据（判据 ①）抓不到，又不在 `tree.body` 所以名字
    判据也抓不到 ⇒ **两道门都绿**，正是本门要消灭的那种免检。所以这里下降进
    控制流块，但 ⛔ 不进 FunctionDef / AsyncFunctionDef / ClassDef——那里面
    是局部变量，不是 module 状态。

    残留开集（判据无法覆盖，靠 review 兜）：`globals()["_x"] = ...` 这类
    动态赋值。
    """
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()

    def _walk(body) -> None:
        for node in body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        names.update(
                            el.id for el in target.elts if isinstance(el, ast.Name)
                        )
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(
                node, (ast.If, ast.Try, ast.For, ast.AsyncFor, ast.While,
                       ast.With, ast.AsyncWith)
            ):
                # 下降进控制流块；⛔ 不下降进函数 / 类体。
                for attr in ("body", "orelse", "finalbody"):
                    _walk(getattr(node, attr, []) or [])
                for handler in getattr(node, "handlers", []) or []:
                    _walk(handler.body)

    _walk(tree.body)
    return names


def test_mcp_startup_module_state_is_a_classified_closed_inventory():
    """闭集门：mcp_startup 的每一个 module-level 状态都必须被显式归类。

    ⛔ 旧判据按 `_mcp_discovery_` **变量名前缀**筛，等于给所有别的命名发免检：
    新增 `_profile_backoff = {}` 时它照样 GREEN（独立 reviewer 用
    `ordinary_owner_cache = {}` 实证过同一件事）。判据必须按"必须满足什么"，
    ⛔ 不按"长什么样"。

    形状照抄 tools/mcp_tool.py 的 `test_all_name_keyed_mcp_lifecycle_state_
    is_profile_scoped`（运行时枚举 module 里的可变容器，与生产侧分类清单精确
    相等）；另叠一层 AST 名字全集，因为本模块大半状态是 bool / Thread / Lock，
    纯容器判据抓不到 `_x_started = False` 这类新增。
    """
    from collections.abc import MutableMapping, MutableSet

    classified = (
        set(mcp_startup._MCP_STARTUP_PROCESS_GLOBAL_STATE)
        | set(mcp_startup._MCP_STARTUP_SINGLE_PROFILE_STATE)
        | set(mcp_startup._MCP_STARTUP_PROFILE_INDEX_STATE)
        | set(mcp_startup._MCP_STARTUP_IMMUTABLE_STATE)
        | set(mcp_startup._MCP_STARTUP_INVENTORY_STATE)
    )

    # ① 运行时可变容器（照抄 mcp_tool 的判据）
    module_mutables = {
        name
        for name, value in vars(mcp_startup).items()
        if not name.startswith("__")
        and isinstance(value, (MutableMapping, MutableSet, list))
    }
    assert module_mutables <= classified, (
        "mcp_startup 新增了未登记的 module-level 可变容器："
        f"{sorted(module_mutables - classified)}"
    )

    # ② AST 名字全集（补 ① 在本模块的盲区：标量 / Thread / Lock）
    actual = _module_level_assigned_names(mcp_startup)
    # 阳性对照：扫不到任何名字说明扫描本身坏了，⛔ 不许当"闭集成立"。
    assert actual, "校准失败：mcp_startup 一个 module-level 赋值都没扫到"
    assert actual == classified, (
        "mcp_startup 的 module-level 状态必须逐项显式归类："
        f"未归类={sorted(actual - classified)}，已消失={sorted(classified - actual)}"
    )

    # ③「往集合里塞东西」方向：把可变状态塞进"不可变常量"桶不能消音。
    mutable_in_immutable = sorted(
        name
        for name in mcp_startup._MCP_STARTUP_IMMUTABLE_STATE
        if isinstance(getattr(mcp_startup, name, None), (MutableMapping, MutableSet, list))
    )
    assert not mutable_in_immutable, (
        f"这些登记为不可变常量、运行时却是可变容器：{mutable_in_immutable}"
    )

    # ④ 用错桶消音：登记为 profile 索引的必须真的按 profile 分区。
    not_partitioned = sorted(
        name
        for name in mcp_startup._MCP_STARTUP_PROFILE_INDEX_STATE
        if not isinstance(getattr(mcp_startup, name, None), MutableMapping)
    )
    assert not not_partitioned, (
        f"这些登记为 profile 索引、运行时却不是按 profile 分区的映射：{not_partitioned}"
    )


def test_discovery_teardown_closes_admission_before_late_spawn(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=lambda: calls.append("spawn")),
    )

    mcp_startup.begin_mcp_discovery_teardown()
    mcp_startup.start_background_mcp_discovery(
        logger=_retry_logger(), thread_name="late-discovery"
    )

    assert calls == []
    assert mcp_startup._mcp_discovery_thread is None








def _retry_logger():
    return types.SimpleNamespace(
        debug=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
    )


def _install_retry_stubs(monkeypatch, *, connected: bool, calls: dict):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            read_raw_config=lambda: {"mcp_servers": {"demo": {"transport": "stdio"}}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.__setitem__("mcp", calls["mcp"] + 1),
            get_mcp_status=lambda: [{"connected": connected}],
        ),
    )




    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        types.SimpleNamespace(get_tool_definitions=lambda *_a, **_k: ["ok"]),
    )

    start = time.monotonic()
    result = cli_mod.get_tool_definitions(enabled_toolsets=["web"], quiet_mode=True)
    elapsed = time.monotonic() - start

    assert result == ["ok"]
    assert elapsed >= 0.04
    assert not thread.is_alive()


@pytest.mark.parametrize(
    ("single_query_mode", "supports_followup_turns"),
    [(False, True), (True, False)],
)
def test_init_agent_waits_for_mcp_discovery_before_agent_build(
    monkeypatch, single_query_mode, supports_followup_turns
):
    waited = {"done": False}

    cli = cli_mod.HermesCLI(compact=True)
    cli._session_db = object()
    cli._resumed = False
    cli.conversation_history = []
    cli._install_tool_callbacks = lambda: None
    cli._ensure_tirith_security = lambda: None
    cli._ensure_runtime_credentials = lambda: True
    cli._single_query_mode = single_query_mode

    monkeypatch.setattr(
        mcp_startup,
        "ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: waited.__setitem__("done", True),
    )

    def _fake_agent(*_a, **_k):
        assert waited["done"] is True
        return types.SimpleNamespace()

    monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)

    assert cli._init_agent() is True
    assert cli.agent._supports_followup_turns is supports_followup_turns
