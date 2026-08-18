"""Shared CLI/TUI-safe helpers for background MCP discovery."""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

_mcp_discovery_lock = threading.Lock()
# 无 profile pin 的单 profile 进程保留原状态槽，避免改变 CLI 启动语义。
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None
_mcp_discovery_by_profile: dict[str, Optional[threading.Thread]] = {}
_mcp_discovery_teardown_started = False

# ── module-level 状态的权威分类（闭集门读这里）──────────────────────────
#
# 形状与 tools/mcp_tool.py 的 `_MCP_*_MUTABLES` 一致：分类清单放生产侧，
# 让新增状态的人在同一个文件里就看到必须登记。
#
# ⛔ 判据不许按变量名前缀（或任何"长什么样"的形状）筛 —— 那样只罩住恰好长
# 成那样的状态，新增别的命名会静默免检，门恒绿、冒充保护。判据按"必须满足
# 什么"：本模块每一个 module-level 名字都必须出现在下面某一类里。
#
# ⚠️ 与 mcp_tool 的差异（有意为之，不是另发明一套）：mcp_tool 的 module 状态
# 几乎全是容器，纯运行时 MutableMapping/MutableSet/list 枚举就够；本模块大半
# 状态是 bool / Thread / Lock，纯容器判据会漏掉它们（新增 `_x_started = False`
# 扫不到）。所以在照抄那套运行时容器判据之外，额外用 AST 覆盖名字全集。
#
# 闭集边界：只覆盖本模块**仓内 Python 可见**的 module-level 状态。第三方 SDK
# 内部持有的进程级状态属于开集，由 MCP/LSP 的真实 child-env 集成测试兜底。
_MCP_STARTUP_PROCESS_GLOBAL_STATE = frozenset({
    "_mcp_discovery_lock",
    "_mcp_discovery_teardown_started",
})
# 无 profile pin 的单 profile 兼容槽。
_MCP_STARTUP_SINGLE_PROFILE_STATE = frozenset({
    "_mcp_discovery_started",
    "_mcp_discovery_thread",
})
# 按 profile 分区的索引；运行时必须是映射（闭集门会反证）。
_MCP_STARTUP_PROFILE_INDEX_STATE = frozenset({
    "_mcp_discovery_by_profile",
})
# 不可变常量 / 类型别名；⛔ 可变容器不许登记到这里（闭集门会反证）。
_MCP_STARTUP_IMMUTABLE_STATE: frozenset[str] = frozenset()
# 上面这些清单自身也是 module-level 赋值，AST 判据会扫到，所以要自登记。
# （mcp_tool 那套只枚举运行时可变容器，frozenset 不是 MutableSet，天然不用
# 自登记；本模块多了一层 AST 名字判据，就得把清单自己也算进去。）
_MCP_STARTUP_INVENTORY_STATE: frozenset[str] = frozenset({
    "_MCP_STARTUP_PROCESS_GLOBAL_STATE",
    "_MCP_STARTUP_SINGLE_PROFILE_STATE",
    "_MCP_STARTUP_PROFILE_INDEX_STATE",
    "_MCP_STARTUP_IMMUTABLE_STATE",
    "_MCP_STARTUP_INVENTORY_STATE",
})


def _discovery_profile_identity() -> str | None:
    """返回显式 profile 身份；单 profile 无 pin 时使用兼容状态槽。"""
    from hermes_constants import get_hermes_home_override

    override = get_hermes_home_override()
    return str(Path(override).expanduser().resolve()) if override else None


def _discovery_state(profile_identity: str | None) -> tuple[bool, Optional[threading.Thread]]:
    if profile_identity is None:
        return _mcp_discovery_started, _mcp_discovery_thread
    return (
        profile_identity in _mcp_discovery_by_profile,
        _mcp_discovery_by_profile.get(profile_identity),
    )


def _set_discovery_state(
    profile_identity: str | None,
    *,
    started: bool,
    thread: Optional[threading.Thread],
) -> None:
    global _mcp_discovery_started, _mcp_discovery_thread
    if profile_identity is None:
        _mcp_discovery_started = started
        _mcp_discovery_thread = thread
        return
    if not started:
        _mcp_discovery_by_profile.pop(profile_identity, None)
    else:
        _mcp_discovery_by_profile[profile_identity] = thread


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        mcp_servers = (read_raw_config() or {}).get("mcp_servers")
        return isinstance(mcp_servers, dict) and len(mcp_servers) > 0
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one shared background MCP discovery thread for this profile.

    If the first background discovery run exits without connecting any MCP
    server (for example after startup cancellation / OOM restart), later calls
    are allowed to retry instead of permanently pinning the process in a
    "discovery already started" state with zero MCP tools.
    """
    profile_identity = _discovery_profile_identity()

    with _mcp_discovery_lock:
        if _mcp_discovery_teardown_started:
            logger.debug("MCP discovery skipped: process teardown has started")
            return
        started, thread = _discovery_state(profile_identity)
        if started:
            if thread is not None and thread.is_alive():
                return
            try:
                from tools.mcp_tool import get_mcp_status

                status = get_mcp_status() or []
                if any(entry.get("connected") for entry in status):
                    return
            except Exception:
                return
            logger.warning(
                "Background MCP discovery previously exited with no connected "
                "servers; retrying discovery thread"
            )
            _set_discovery_state(profile_identity, started=False, thread=None)

        _set_discovery_state(profile_identity, started=True, thread=None)
        if not _has_configured_mcp_servers():
            return

        # Bare threads do not inherit ContextVars.  Copy the complete caller
        # context so profile home and secret scope travel together; copying
        # only HERMES_HOME lets a secondary profile discover with the wrong
        # credential namespace.
        discovery_context = contextvars.copy_context()

        def _discover() -> None:
            try:
                with _mcp_discovery_lock:
                    if _mcp_discovery_teardown_started:
                        return
                _discover_mcp_tools_without_interactive_oauth()
                try:
                    from tools.mcp_tool import get_mcp_status
                    status = get_mcp_status() or []
                    if not any(entry.get("connected") for entry in status):
                        logger.warning(
                            "Background MCP discovery completed with zero connected servers"
                        )
                except Exception:
                    logger.debug("Failed to inspect MCP status after background discovery", exc_info=True)
            except Exception:
                logger.debug("Background MCP tool discovery failed", exc_info=True)
            finally:
                with _mcp_discovery_lock:
                    _started, current = _discovery_state(profile_identity)
                    if current is threading.current_thread():
                        _set_discovery_state(
                            profile_identity, started=_started, thread=None
                        )

        thread = threading.Thread(
            target=discovery_context.run,
            args=(_discover,),
            name=thread_name,
            daemon=True,
        )
        _set_discovery_state(profile_identity, started=True, thread=thread)
        thread.start()


def _resolve_discovery_timeout(
    explicit: "float | None", *, single_query: bool = False
) -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.

    When ``single_query`` is True (``hermes -z "..."`` / ``-q``), the larger
    ``mcp_single_query_discovery_timeout`` bound is used instead. In single-query
    mode there is only ONE turn, so the between-turns late-binding refresh never
    runs — a server that misses the small interactive bound would be invisible to
    the LLM for the whole session. The wait still returns the instant discovery
    completes (see ``wait_for_mcp_discovery``), so fast servers pay ~0s; the
    larger bound only caps how long a genuinely slow cold-start may block.
    """
    if explicit is not None:
        return explicit
    key = (
        "mcp_single_query_discovery_timeout"
        if single_query
        else "mcp_discovery_timeout"
    )
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get(key, fallback))
        try:
            raw = (load_config() or {}).get(key, default)
            val = float(raw)
            return val if val > 0 else default
        except Exception:
            return default
    except Exception:
        return fallback


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def wait_for_mcp_discovery(
    timeout: "float | None" = None, *, single_query: bool = False
) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.

    When ``single_query`` is True, the bound comes from
    ``mcp_single_query_discovery_timeout`` instead (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.
    """
    _started, thread = _discovery_state(_discovery_profile_identity())
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def mcp_discovery_in_flight() -> bool:
    """返回当前 profile 的 discovery 线程是否仍在运行。

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight`` for the surfaces that
    start discovery through ``start_background_mcp_discovery`` here (the desktop
    app + dashboard WebSocket sidecar via ``tui_gateway/ws.py``, and
    ``hermes dashboard``).  Those processes populate THIS module's
    late-refresh scheduler 仍会检查 ``tui_gateway.entry`` 的兼容 owner；
    在本共享 owner 内，由调用方的 profile 身份选择线程。
    """
    _started, thread = _discovery_state(_discovery_profile_identity())
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block until THIS module's background discovery finishes, up to ``timeout``.

    Returns True if discovery has completed (thread absent or no longer alive),
    False if it is still running after the timeout.  Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.
    """
    _started, thread = _discovery_state(_discovery_profile_identity())
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def join_all_mcp_discovery(timeout: "float | None" = None) -> bool:
    """在进程 shutdown 前等待所有 profile 的 discovery，使用一个总预算。"""
    with _mcp_discovery_lock:
        threads = [
            thread
            for thread in [_mcp_discovery_thread, *_mcp_discovery_by_profile.values()]
            if thread is not None and thread.is_alive()
        ]
    deadline = None if timeout is None else time.monotonic() + timeout
    for thread in threads:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
    return not any(thread.is_alive() for thread in threads)


def begin_mcp_discovery_teardown() -> None:
    """关闭本进程 discovery admission；shutdown 期间不可重新打开。"""
    global _mcp_discovery_teardown_started
    with _mcp_discovery_lock:
        _mcp_discovery_teardown_started = True


def mcp_discovery_admission_open() -> bool:
    with _mcp_discovery_lock:
        return not _mcp_discovery_teardown_started


def clear_mcp_discovery_profile(profile_home: str | Path) -> None:
    """清除已经完整卸载的 profile discovery owner 状态。"""
    profile_identity = str(Path(profile_home).expanduser().resolve())
    with _mcp_discovery_lock:
        thread = _mcp_discovery_by_profile.get(profile_identity)
        if thread is not None and thread.is_alive():
            raise RuntimeError(
                f"cannot clear MCP discovery while it is running for {profile_identity}"
            )
        _set_discovery_state(profile_identity, started=False, thread=None)


def ensure_mcp_discovery_before_agent_build(
    *,
    logger,
    timeout: "float | None" = None,
    single_query: bool = False,
    thread_name: str = "cli-mcp-discovery",
) -> None:
    """Give configured MCP tools a bounded chance to register before AIAgent.

    Non-interactive first turns (``chat -q``, ``hermes -z``) can construct
    ``AIAgent`` before the normal banner or tool-list paths touch
    ``get_tool_definitions()``.  Because the agent snapshots its tool
    registry at construction time, the first and only model turn can miss
    native ``mcp__...`` tools even when the MCP server is healthy.

    ``wait_for_mcp_discovery()`` only joins an already-created discovery
    thread, so it no-ops if a direct/single-query path reaches agent
    construction before MCP startup created that thread.  This helper makes
    the construction site self-sufficient: start discovery if needed, then
    wait up to the configured bound.

    When ``single_query`` is True, the larger
    ``mcp_single_query_discovery_timeout`` bound is used (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.

    Failures are swallowed so a broken MCP config never aborts agent
    construction — the agent runs without MCP tools, same as before.
    """
    try:
        start_background_mcp_discovery(
            logger=logger,
            thread_name=thread_name,
        )
        wait_for_mcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug(
            "MCP discovery readiness check failed before agent build",
            exc_info=True,
        )
