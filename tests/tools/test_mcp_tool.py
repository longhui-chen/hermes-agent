"""Tests for the MCP (Model Context Protocol) client support.

All tests use mocks -- no real MCP servers or subprocesses are started.
"""

import asyncio
import ast
import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mcp_tool(name="read_file", description="Read a file", input_schema=None):
    """Create a fake MCP Tool object matching the SDK interface."""
    tool = SimpleNamespace()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema or {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    }
    return tool


def _make_call_result(text="file contents here", is_error=False):
    """Create a fake MCP CallToolResult."""
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block], isError=is_error)


def _make_mock_server(name, session=None, tools=None):
    """Create an MCPServerTask with mock attributes for testing."""
    from tools.mcp_tool import MCPServerTask
    server = MCPServerTask(name)
    server.session = session
    server._tools = tools or []
    return server


class TestFilterMCPChildren:
    def test_filters_gateway_children_by_argv_marker(self, monkeypatch):
        """Non-MCP children start with an interpreter/binary, not the marker."""
        import sys

        import tools.mcp_tool as mcp_tool

        cmdlines = {
            101: [
                "/usr/bin/python3",
                "-m",
                "tui_gateway.slash_worker",
                "--session-key",
                "abc",
            ],
            102: [
                "/usr/bin/java",
                "-jar",
                "/opt/jdtls/plugins/org.eclipse.equinox.launcher_1.7.0.jar",
            ],
            103: [
                "/usr/bin/python3",
                "/repo/tools/mcp_stdio_watchdog.py",
                "--",
                "/usr/bin/node",
                "server.js",
            ],
            104: ["/usr/local/bin/pyright-langserver", "--stdio"],
            105: ["/usr/local/bin/gopls"],
        }

        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid

            def cmdline(self):
                return cmdlines[self.pid]

        fake_psutil = SimpleNamespace(
            Process=FakeProcess,
            NoSuchProcess=ProcessLookupError,
            AccessDenied=PermissionError,
        )
        monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
        monkeypatch.setattr(mcp_tool.os, "name", "posix")

        assert mcp_tool._filter_mcp_children({101, 102, 103, 104, 105}) == {103}

    def test_windows_filter_accepts_only_expected_spawn_identity(
        self, monkeypatch
    ):
        """Windows 必须按本次 command/args 正向认领，不能靠 LSP 黑名单。"""
        import sys

        import tools.mcp_tool as mcp_tool

        cmdlines = {
            201: ["C:\\node.exe", "pyright-langserver.js", "--stdio"],
            202: ["C:\\node.exe", "mcp-server.js", "--stdio"],
            203: [
                "C:\\node.exe",
                "npx-cli.js",
                "-y",
                "@modelcontextprotocol/server-filesystem",
            ],
        }

        class _Process:
            def __init__(self, pid):
                self.pid = pid

            def cmdline(self):
                return cmdlines[self.pid]

        monkeypatch.setitem(
            sys.modules,
            "psutil",
            SimpleNamespace(
                Process=_Process,
                NoSuchProcess=ProcessLookupError,
                AccessDenied=PermissionError,
            ),
        )
        monkeypatch.setattr(mcp_tool.os, "name", "nt")

        assert mcp_tool._filter_mcp_children(
            {201, 202},
            expected_command="node.exe",
            expected_args=["mcp-server.js", "--stdio"],
        ) == {202}
        assert mcp_tool._filter_mcp_children(
            {201, 203},
            expected_command="npx.cmd",
            expected_args=["-y", "@modelcontextprotocol/server-filesystem"],
        ) == {203}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadMCPConfig:

    def test_valid_config_parsed(self):
        """Valid mcp_servers config is returned as-is."""
        servers = {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {},
            }
        }
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": servers}):
            from tools.mcp_tool import _load_mcp_config
            result = _load_mcp_config()
            assert "filesystem" in result
            assert result["filesystem"]["command"] == "npx"

    def test_mcp_servers_not_dict_returns_empty(self):
        """mcp_servers set to non-dict value -> empty dict."""
        with patch("hermes_cli.config.load_config", return_value={"mcp_servers": "invalid"}):
            from tools.mcp_tool import _load_mcp_config
            result = _load_mcp_config()
            assert result == {}


class TestMCPParallelSafetyProvenance:
    def test_parallel_safe_servers_keep_exact_raw_names(self, monkeypatch):
        import tools.mcp_tool as mcp_tool

        first = SimpleNamespace(session=object(), _registered_tool_names=[])
        second = SimpleNamespace(session=object(), _registered_tool_names=[])

        with mcp_tool._lock:
            saved_servers = dict(mcp_tool._servers)
            saved_parallel = set(mcp_tool._parallel_safe_servers)
            mcp_tool._servers.clear()
            mcp_tool._servers.update({"foo-bar": first, "foo_bar": second})
            mcp_tool._parallel_safe_servers.clear()

        try:
            monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
            monkeypatch.setattr(
                mcp_tool, "_filter_suspicious_mcp_servers", lambda servers: servers
            )
            mcp_tool.register_mcp_servers(
                {
                    "foo-bar": {"supports_parallel_tool_calls": True},
                    "foo_bar": {"supports_parallel_tool_calls": False},
                }
            )
            with mcp_tool._lock:
                assert "foo-bar" in mcp_tool._parallel_safe_servers
                assert "foo_bar" not in mcp_tool._parallel_safe_servers
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved_servers)
                mcp_tool._parallel_safe_servers.clear()
                mcp_tool._parallel_safe_servers.update(saved_parallel)

    def test_tool_provenance_keeps_exact_raw_server_names(self):
        import tools.mcp_tool as mcp_tool

        first_tool = "mcp__foo_bar__first"
        second_tool = "mcp__foo_bar__second"
        with mcp_tool._lock:
            saved_map = dict(mcp_tool._mcp_tool_server_names)
            saved_parallel = set(mcp_tool._parallel_safe_servers)
            mcp_tool._mcp_tool_server_names.clear()
            mcp_tool._parallel_safe_servers.clear()
            mcp_tool._parallel_safe_servers.add("foo-bar")

        try:
            mcp_tool._track_mcp_tool_server(first_tool, "foo-bar")
            mcp_tool._track_mcp_tool_server(second_tool, "foo_bar")

            assert mcp_tool.is_mcp_tool_parallel_safe(first_tool) is True
            assert mcp_tool.is_mcp_tool_parallel_safe(second_tool) is False
            assert mcp_tool.get_registered_mcp_server_names() == {
                "foo-bar",
                "foo_bar",
            }
        finally:
            with mcp_tool._lock:
                mcp_tool._mcp_tool_server_names.clear()
                mcp_tool._mcp_tool_server_names.update(saved_map)
                mcp_tool._parallel_safe_servers.clear()
                mcp_tool._parallel_safe_servers.update(saved_parallel)

class TestMCPStatus:
    def test_status_distinguishes_configured_connecting_failed_and_disabled(
        self, monkeypatch
    ):
        import tools.mcp_tool as mcp_tool

        monkeypatch.setattr(
            mcp_tool,
            "_load_mcp_config",
            lambda: {
                "configured": {"command": "docker", "args": ["mcp", "gateway", "run"]},
                "connecting": {"command": "slow-mcp"},
                "failed": {"command": "bad-mcp"},
                "disabled": {"command": "off-mcp", "enabled": False},
            },
        )
        with mcp_tool._lock:
            saved_servers = dict(mcp_tool._servers)
            saved_connecting = set(mcp_tool._server_connecting)
            saved_errors = dict(mcp_tool._server_connect_errors)
            mcp_tool._servers.clear()
            mcp_tool._server_connecting.clear()
            mcp_tool._server_connect_errors.clear()
            mcp_tool._server_connecting.add("connecting")
            mcp_tool._server_connect_errors["failed"] = "Connection closed"

        try:
            statuses = {
                entry["name"]: entry
                for entry in mcp_tool.get_mcp_status()
            }
        finally:
            with mcp_tool._lock:
                mcp_tool._servers.clear()
                mcp_tool._servers.update(saved_servers)
                mcp_tool._server_connecting.clear()
                mcp_tool._server_connecting.update(saved_connecting)
                mcp_tool._server_connect_errors.clear()
                mcp_tool._server_connect_errors.update(saved_errors)

        assert statuses["configured"]["status"] == "configured"
        assert statuses["configured"]["connected"] is False
        assert statuses["configured"]["disabled"] is False
        assert statuses["connecting"]["status"] == "connecting"
        assert statuses["failed"]["status"] == "failed"
        assert statuses["failed"]["error"] == "Connection closed"
        assert statuses["disabled"]["status"] == "disabled"
        assert statuses["disabled"]["disabled"] is True


class TestLifecycleConfig:
    def test_get_lifecycle_seconds_accepts_top_level_and_nested_values(self):
        from tools.mcp_tool import _get_lifecycle_seconds

        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": "3.5"},
                "idle_timeout_seconds",
            )
            == 3.5
        )
        assert _get_lifecycle_seconds(
            {"lifecycle": {"max_lifetime_seconds": 42}},
            "max_lifetime_seconds",
        ) == 42.0

    def test_get_lifecycle_seconds_ignores_invalid_values(self, caplog):
        from tools.mcp_tool import _get_lifecycle_seconds

        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": "soon"},
                "idle_timeout_seconds",
            )
            is None
        )
        assert (
            _get_lifecycle_seconds(
                {"idle_timeout_seconds": -1},
                "idle_timeout_seconds",
            )
            is None
        )

        messages = [record.getMessage() for record in caplog.records]
        assert any("must be a number of seconds" in msg for msg in messages)
        assert any("must be positive" in msg for msg in messages)


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------

class TestSchemaConversion:
    def test_converts_mcp_tool_to_hermes_schema(self):
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="read_file", description="Read a file")
        schema = _convert_mcp_schema("filesystem", mcp_tool)

        assert schema["name"] == "mcp__filesystem__read_file"
        assert schema["description"] == "Read a file"
        assert "properties" in schema["parameters"]

    def test_definitions_as_property_name_is_preserved(self):
        """A tool parameter literally named ``definitions`` must not be renamed.

        Regression: the rewrite that promotes the legacy ``definitions``
        meta-keyword to ``$defs`` used to fire for *any* key named
        ``definitions`` anywhere in the tree, including inside ``properties``
        dicts. That turned user-facing parameter names into ``$defs``, which
        Anthropic and OpenAI both reject because ``$`` is not in the
        ``^[a-zA-Z0-9_.-]{1,64}$`` property-name pattern. Real-world repro: a
        CI/pipelines MCP tool whose ``definitions`` parameter is an array of
        pipeline-definition IDs.
        """
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(
            name="pipelines_build",
            description="List pipeline builds",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "definitions": {
                        "description": "Array of build definition IDs to filter builds.",
                    },
                    "top": {"type": "integer"},
                },
            },
        )

        schema = _convert_mcp_schema("pipelines", mcp_tool)

        props = schema["parameters"]["properties"]
        assert "definitions" in props, "user-facing property name was renamed away"
        assert "$defs" not in props, "user-facing property name was rewritten to $defs"
        # And the meta-keyword promotion didn't happen at the root either,
        # because there was no `definitions` meta-keyword to promote.
        assert "$defs" not in schema["parameters"]
        assert "definitions" not in schema["parameters"]


    def test_optional_nullable_field_is_collapsed_to_non_null_schema(self):
        """Anthropic rejects MCP/Pydantic anyOf-null optional parameter schemas."""
        from tools.mcp_tool import _normalize_mcp_input_schema

        schema = _normalize_mcp_input_schema({
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "workdir": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Optional working directory",
                },
            },
            "required": ["command"],
        })

        assert schema["properties"]["workdir"] == {
            "type": "string",
            "nullable": True,
            "default": None,
            "description": "Optional working directory",
        }
        assert schema["required"] == ["command"]

    def test_hyphens_sanitized_to_underscores(self):
        """Hyphens in tool/server names are replaced with underscores for LLM compat."""
        from tools.mcp_tool import _convert_mcp_schema

        mcp_tool = _make_mcp_tool(name="get-sum")
        schema = _convert_mcp_schema("my-server", mcp_tool)

        assert schema["name"] == "mcp__my_server__get_sum"
        assert "-" not in schema["name"]


# ---------------------------------------------------------------------------
# Check function
# ---------------------------------------------------------------------------

class TestCheckFunction:
    def test_disconnected_returns_false(self):
        from tools.mcp_tool import _make_check_fn, _servers

        _servers.pop("test_server", None)
        check = _make_check_fn("test_server")
        assert check() is False


    def test_recycled_stdio_server_remains_available_for_lazy_reconnect(self):
        from tools.mcp_tool import _make_check_fn, _servers

        server = _make_mock_server("test_server", session=None)
        server._config = {"command": "npx"}
        server._recycled_reason = "idle_timeout_seconds"
        _servers["test_server"] = server
        try:
            check = _make_check_fn("test_server")
            assert check() is True
        finally:
            _servers.pop("test_server", None)


# ---------------------------------------------------------------------------
# MCP loop runner
# ---------------------------------------------------------------------------

class TestRunOnMcpLoop:
    def test_scheduler_failure_closes_factory_coroutine(self):
        """If run_coroutine_threadsafe raises, the factory's coroutine is closed."""
        import gc
        import warnings
        import tools.mcp_tool as mcp

        created = {"coro": None}

        async def _sample():
            return "ok"

        def factory():
            created["coro"] = _sample()
            return created["coro"]

        fake_loop = MagicMock()
        fake_loop.is_running.return_value = True

        with patch.object(mcp, "_mcp_loop", fake_loop):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "agent.async_utils.asyncio.run_coroutine_threadsafe",
                    side_effect=RuntimeError("scheduler down"),
                ):
                    with pytest.raises(RuntimeError):
                        mcp._run_on_mcp_loop(factory)
                gc.collect()

        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_sample" in str(w.message)
        ]
        assert runtime_warnings == []

    def test_dead_loop_closes_passed_coroutine(self):
        """If loop is None, a passed coroutine (not factory) is closed."""
        import gc
        import warnings
        import tools.mcp_tool as mcp

        async def _sample():
            return "ok"

        coro = _sample()
        with patch.object(mcp, "_mcp_loop", None):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError, match="not running"):
                    mcp._run_on_mcp_loop(coro)
                gc.collect()

        assert coro.cr_frame is None
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_sample" in str(w.message)
        ]
        assert runtime_warnings == []


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

class TestToolHandler:
    """Tool handlers are sync functions that schedule work on the MCP loop."""

    def _patch_mcp_loop(self, coro_side_effect=None):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)
        if coro_side_effect:
            return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=coro_side_effect)
        return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run)

    def test_successful_call(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("hello world", is_error=False)
        )
        server = _make_mock_server("test_srv", session=mock_session)
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "greet", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"name": "world"}))
            assert result["result"] == "hello world"
            mock_session.call_tool.assert_called_once_with("greet", arguments={"name": "world"})
        finally:
            _servers.pop("test_srv", None)

    def test_forward_context_meta_uses_trusted_session_context(self):
        from gateway.session_context import (
            clear_session_vars,
            clear_turn_vars,
            set_session_vars,
            set_turn_vars,
        )
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("stored", is_error=False)
        )
        server = _make_mock_server("memo_srv", session=mock_session)
        server._config = {"forward_context_meta": True}
        _servers["memo_srv"] = server
        session_tokens = set_session_vars(
            user_id="account-1",
            session_id="session-1",
            profile="main",
        )
        turn_tokens = set_turn_vars(turn_id="turn-1")

        try:
            handler = _make_tool_handler("memo_srv", "memo_write", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"statement": "用户喜欢咖啡"}))
            assert result["result"] == "stored"
            mock_session.call_tool.assert_called_once_with(
                "memo_write",
                arguments={"statement": "用户喜欢咖啡"},
                meta={
                    "zettlab/profile_id": "main",
                    "zettlab/session_id": "session-1",
                    "zettlab/turn_id": "turn-1",
                    "zettlab/account_id": "account-1",
                },
            )
        finally:
            clear_turn_vars(turn_tokens)
            clear_session_vars(session_tokens)
            _servers.pop("memo_srv", None)

    def test_managed_memo_forwards_context_without_migrated_config_flag(self):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("stored", is_error=False)
        )
        server = _make_mock_server("zettlab_memo", session=mock_session)
        server._config = {}
        _servers["zettlab_memo"] = server
        session_tokens = set_session_vars(
            user_id="account-ota", session_id="session-ota", profile="main"
        )

        try:
            handler = _make_tool_handler("zettlab_memo", "memo_write", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"statement": "remember this"}))
            assert result["result"] == "stored"
            mock_session.call_tool.assert_called_once_with(
                "memo_write",
                arguments={"statement": "remember this"},
                meta={
                    "zettlab/profile_id": "main",
                    "zettlab/session_id": "session-ota",
                    "zettlab/account_id": "account-ota",
                },
            )
        finally:
            clear_session_vars(session_tokens)
            _servers.pop("zettlab_memo", None)

    def test_managed_memo_metadata_uses_zet_account_not_session_owner_principal(self, monkeypatch):
        from gateway.config import PlatformConfig
        from gateway.platforms.zet_agent import ZetAgentAdapter, _zettlab_request_account_id
        from gateway.session_context import clear_session_vars, pop_zettlab_auth_principal, push_zettlab_auth_principal
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("stored", is_error=False)
        )
        server = _make_mock_server("zettlab_memo", session=mock_session)
        server._config = {}
        _servers["zettlab_memo"] = server
        adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
        account_token = _zettlab_request_account_id.set("account-1")
        principal_token = push_zettlab_auth_principal("iam:alice")
        session_tokens = adapter._bind_api_server_session(
            chat_id="session-1",
            session_key="zettlab:account-1:main:session-1",
            session_id="session-1",
            session_user_id="account-1",
        )

        try:
            handler = _make_tool_handler("zettlab_memo", "memo_write", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"statement": "remember this"}))
            assert result["result"] == "stored"
            assert mock_session.call_tool.call_args.kwargs["meta"]["zettlab/account_id"] == "account-1"
        finally:
            clear_session_vars(session_tokens)
            pop_zettlab_auth_principal(principal_token)
            _zettlab_request_account_id.reset(account_token)
            _servers.pop("zettlab_memo", None)


    def test_recycled_stdio_server_reconnects_lazily_on_tool_call(self):
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("reconnected", is_error=False)
        )
        server = _make_mock_server("test_srv", session=None)
        server._config = {"command": "npx"}
        server._recycled_reason = "idle_timeout_seconds"
        _servers["test_srv"] = server

        def fake_lazy_reconnect(server_name, srv):
            assert server_name == "test_srv"
            assert srv is server
            srv.session = mock_session
            srv._recycled_reason = None
            return True

        try:
            handler = _make_tool_handler("test_srv", "greet", 120)
            with patch("tools.mcp_tool._request_lazy_reconnect", side_effect=fake_lazy_reconnect) as reconnect, \
                 self._patch_mcp_loop():
                result = json.loads(handler({"name": "world"}))
            assert result["result"] == "reconnected"
            reconnect.assert_called_once()
            mock_session.call_tool.assert_called_once_with("greet", arguments={"name": "world"})
        finally:
            _servers.pop("test_srv", None)


class TestRunOnMCPLoopInterrupts:
    @staticmethod
    def _run_with_future(mcp_mod, future):
        loop = MagicMock()
        loop.is_running.return_value = True

        async def _unused_call():
            return "unused"

        def _schedule(coro, scheduled_loop, **_kwargs):
            assert scheduled_loop is loop
            coro.close()
            return future

        with patch.object(mcp_mod, "_mcp_loop", loop):
            with patch("agent.async_utils.safe_schedule_threadsafe", side_effect=_schedule):
                return mcp_mod._run_on_mcp_loop(_unused_call(), timeout=1)

    def test_interrupt_cancels_waiting_mcp_call(self):
        import tools.mcp_tool as mcp_mod
        from tools.interrupt import set_interrupt

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        cancelled = threading.Event()

        async def _slow_call():
            try:
                await asyncio.sleep(5)
                return "done"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        old_loop = mcp_mod._mcp_loop
        old_thread = mcp_mod._mcp_thread
        mcp_mod._mcp_loop = loop
        mcp_mod._mcp_thread = thread

        waiter_tid = threading.current_thread().ident

        def _interrupt_soon():
            time.sleep(0.02)
            set_interrupt(True, waiter_tid)

        interrupter = threading.Thread(target=_interrupt_soon, daemon=True)
        interrupter.start()

        try:
            with pytest.raises(InterruptedError, match="User sent a new message"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=10)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.01)
            assert cancelled.is_set()
        finally:
            set_interrupt(False, waiter_tid)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
            loop.close()
            mcp_mod._mcp_loop = old_loop
            mcp_mod._mcp_thread = old_thread

    def test_timeout_reports_elapsed_and_configured_timeout(self):
        import tools.mcp_tool as mcp_mod

        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()

        cancelled = threading.Event()

        async def _slow_call():
            try:
                await asyncio.sleep(5)
                return "done"
            except asyncio.CancelledError:
                cancelled.set()
                raise

        old_loop = mcp_mod._mcp_loop
        old_thread = mcp_mod._mcp_thread
        mcp_mod._mcp_loop = loop
        mcp_mod._mcp_thread = thread

        try:
            # 0.1s is the floor the MCP loop clamps short timeouts to.
            with pytest.raises(TimeoutError, match=r"MCP call timed out after .*configured timeout: 0.1s"):
                mcp_mod._run_on_mcp_loop(_slow_call(), timeout=0.1)

            deadline = time.time() + 2
            while time.time() < deadline and not cancelled.is_set():
                time.sleep(0.01)
            assert cancelled.is_set()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=10)
            loop.close()
            mcp_mod._mcp_loop = old_loop
            mcp_mod._mcp_thread = old_thread

    def test_strict_interrupt_waits_for_real_lifecycle_completion(self):
        """生产入口收到 interrupt 后必须进入 completion wait，不能提前返回。"""
        import tools.mcp_tool as mcp_mod
        real_event = threading.Event
        lifecycle_entered = real_event()
        interrupt_requested = real_event()
        observed = queue.Queue()
        outcomes = queue.Queue()
        release_holder = {}
        event_count = 0
        event_count_lock = threading.Lock()

        class _ObservedLifecycleEvent:
            def __init__(self):
                nonlocal event_count
                self._event = real_event()
                self._reported_wait = False
                with event_count_lock:
                    event_count += 1
                    self._index = event_count

            def set(self):
                return self._event.set()

            def is_set(self):
                return self._event.is_set()

            def wait(self, timeout=None):
                if self._index == 2 and not self._reported_wait:
                    self._reported_wait = True
                    observed.put("completion-wait")
                return self._event.wait(timeout)

        async def blocked_lifecycle():
            release = asyncio.Event()
            release_holder["event"] = release
            lifecycle_entered.set()
            await release.wait()

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        old_loop = mcp_mod._mcp_loop
        old_thread = mcp_mod._mcp_thread
        mcp_mod._mcp_loop = loop
        mcp_mod._mcp_thread = loop_thread

        def run_strict_call():
            try:
                mcp_mod._run_on_mcp_loop(
                    blocked_lifecycle,
                    timeout=10,
                    wait_for_completion_on_cancel=True,
                )
            except BaseException as exc:
                outcomes.put(exc)
            finally:
                observed.put("returned")

        caller = threading.Thread(target=run_strict_call, daemon=True)
        try:
            with patch.object(
                mcp_mod,
                "threading",
                SimpleNamespace(Event=_ObservedLifecycleEvent),
            ), patch(
                "tools.interrupt.is_interrupted",
                new=lambda: interrupt_requested.is_set(),
            ):
                caller.start()
                assert lifecycle_entered.wait(timeout=2)
                interrupt_requested.set()
                # 新实现必先进入 lifecycle_completed.wait；旧 wiring 会先返回。
                assert observed.get(timeout=2) == "completion-wait"
                assert caller.is_alive()
                loop.call_soon_threadsafe(release_holder["event"].set)
                caller.join(timeout=2)
                assert not caller.is_alive()
                assert observed.get(timeout=2) == "returned"
                assert isinstance(outcomes.get(timeout=2), InterruptedError)
        finally:
            release = release_holder.get("event")
            if release is not None:
                loop.call_soon_threadsafe(release.set)
            caller.join(timeout=2)
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=2)
            loop.close()
            mcp_mod._mcp_loop = old_loop
            mcp_mod._mcp_thread = old_thread

# ---------------------------------------------------------------------------
# Tool registration (discovery + register)
# ---------------------------------------------------------------------------

class TestDiscoverAndRegister:
    def test_tools_registered_in_registry(self):
        """_discover_and_register_server registers tools with correct names."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [
            _make_mcp_tool("read_file", "Read a file"),
            _make_mcp_tool("write_file", "Write a file"),
        ]
        mock_session = MagicMock()

        async def fake_connect(name, config, **_kwargs):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("fs", {"command": "npx", "args": []})
            )

        assert "mcp__fs__read_file" in registered
        assert "mcp__fs__write_file" in registered
        assert "mcp__fs__read_file" in mock_registry.get_all_tool_names()
        assert "mcp__fs__write_file" in mock_registry.get_all_tool_names()

        _servers.pop("fs", None)


    def test_same_server_normalization_collision_skips_all_ambiguous_tools(self, caplog):
        from tools.mcp_tool import _register_server_tools
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        server = _make_mock_server(
            "srv",
            session=MagicMock(),
            tools=[
                _make_mcp_tool("read-file"),
                _make_mcp_tool("read_file"),
                _make_mcp_tool("safe_tool"),
            ],
        )
        config = {"tools": {"resources": False, "prompts": False}}

        with patch("tools.registry.registry", registry), \
             patch("tools.mcp_tool._track_mcp_tool_server"), \
             caplog.at_level(logging.ERROR, logger="tools.mcp_tool"):
            registered = _register_server_tools("srv", server, config)

        assert registered == ["mcp__srv__safe_tool"]
        assert registry.get_entry("mcp__srv__read_file") is None
        assert registry.get_entry("mcp__srv__safe_tool") is not None
        assert any(
            "name normalization collision" in record.message
            and "tool 'read-file'" in record.message
            and "tool 'read_file'" in record.message
            for record in caplog.records
        )

# ---------------------------------------------------------------------------
# MCPServerTask (run / start / shutdown)
# ---------------------------------------------------------------------------

class TestMCPServerTask:
    """Test the MCPServerTask lifecycle with mocked MCP SDK."""

    def _mock_stdio_and_session(self, session):
        """Return patches for stdio_client and ClientSession as async CMs."""
        mock_read, mock_write = MagicMock(), MagicMock()

        mock_stdio_cm = MagicMock()
        mock_stdio_cm.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

        mock_cs_cm = MagicMock()
        mock_cs_cm.__aenter__ = AsyncMock(return_value=session)
        mock_cs_cm.__aexit__ = AsyncMock(return_value=False)

        return (
            patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm),
            patch("tools.mcp_tool.ClientSession", return_value=mock_cs_cm),
            mock_read, mock_write,
        )

    def test_start_connects_and_discovers_tools(self):
        """start() creates a Task that connects, discovers tools, and waits."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("echo")]
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        p_stdio, p_cs, _, _ = self._mock_stdio_and_session(mock_session)

        async def _test():
            with patch("tools.mcp_tool.StdioServerParameters"), p_stdio, p_cs, \
                 patch("tools.mcp_tool._observe_child_pids",
                       side_effect=[(True, set()), (True, {42424242})]), \
                 patch("tools.mcp_tool._filter_mcp_children",
                       return_value={42424242}):
                server = MCPServerTask("test_srv")
                await server.start({"command": "npx", "args": ["-y", "test"]})

                assert server.session is mock_session
                assert len(server._tools) == 1
                assert server._tools[0].name == "echo"
                mock_session.initialize.assert_called_once()

                await server.shutdown()
                assert server.session is None

        asyncio.run(_test())


    def test_stdio_recycle_deadline_pauses_while_rpc_active(self):
        from tools.mcp_tool import MCPServerTask

        async def _test():
            server = MCPServerTask("srv")
            server._config = {"command": "npx"}
            server._idle_timeout_seconds = 0.01
            server._last_tool_call_at = time.monotonic() - 1.0

            async with server._rpc_lock:
                assert server._stdio_recycle_reason() is None
                assert server._next_stdio_recycle_deadline() is None

        asyncio.run(_test())


# ---------------------------------------------------------------------------
# discover_mcp_tools toolset injection
# ---------------------------------------------------------------------------

class TestToolsetInjection:
    def test_mcp_tools_resolve_through_server_aliases(self):
        """Discovered MCP tools resolve through raw server-name aliases."""
        from tools.mcp_tool import MCPServerTask
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        mock_tools = [_make_mcp_tool("list_files", "List files")]
        mock_session = MagicMock()
        mock_registry = ToolRegistry()

        fresh_servers = {}

        async def fake_connect(name, config, **_kwargs):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_config = {"fs": {"command": "npx", "args": []}}

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()

            assert "mcp__fs__list_files" in result
            assert validate_toolset("fs") is True
            assert validate_toolset("mcp-fs") is True
            assert "mcp__fs__list_files" in resolve_toolset("fs")
            assert "mcp__fs__list_files" in resolve_toolset("mcp-fs")

    def test_partial_failure_retry_on_second_call(self):
        """Failed servers are retried on subsequent discover_mcp_tools() calls."""
        from tools.mcp_tool import MCPServerTask

        mock_tools = [_make_mcp_tool("ping", "Ping")]
        mock_session = MagicMock()

        # Use a real dict so idempotency logic works correctly
        fresh_servers = {}
        call_count = 0
        broken_fixed = False

        async def flaky_connect(name, config, **_kwargs):
            nonlocal call_count
            call_count += 1
            if name == "broken" and not broken_fixed:
                raise ConnectionError("cannot reach server")
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        fake_config = {
            "broken": {"command": "bad"},
            "good": {"command": "npx", "args": []},
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", fresh_servers), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=flaky_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            from tools.mcp_tool import discover_mcp_tools

            # First call: good connects, broken fails
            result1 = discover_mcp_tools()
            assert "mcp__good__ping" in result1
            assert "mcp__broken__ping" not in result1
            first_attempts = call_count

            # "Fix" the broken server
            broken_fixed = True
            call_count = 0

            # The failed server is now serving a post-failure backoff
            # (#50394: prevents a tight re-spawn storm across the frequent
            # per-worker-session discovery passes). Expire that cooldown to
            # simulate the retry window having elapsed.
            import tools.mcp_tool as _mcp_mod
            _mcp_mod._server_connect_retry_after.pop("broken", None)

            # Next call after the cooldown: should retry broken, skip good
            result2 = discover_mcp_tools()
            assert "mcp__good__ping" in result2
            assert "mcp__broken__ping" in result2
            assert call_count == 1  # Only broken retried


# ---------------------------------------------------------------------------
# Graceful fallback
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_mcp_unavailable_returns_empty(self):
        """When _MCP_AVAILABLE is False, discover_mcp_tools is a no-op."""
        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            from tools.mcp_tool import discover_mcp_tools
            result = discover_mcp_tools()
            assert result == []

# ---------------------------------------------------------------------------
# Shutdown (public API)
# ---------------------------------------------------------------------------

class TestShutdown:

    def test_shutdown_timeout_before_cleanup_keeps_server_owner_for_retry(self):
        """cleanup 未完成时的显式超时必须保留全局 exact owner。"""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import MCPServerTask, shutdown_mcp_servers

        server = MCPServerTask("parked")
        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
            mcp_mod._servers[server.name] = server

        try:
            with patch.object(
                mcp_mod, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
            ), patch.object(
                mcp_mod,
                "_run_on_mcp_loop",
                side_effect=TimeoutError("simulated bounded MCP shutdown timeout"),
            ):
                with pytest.raises(RuntimeError, match="timed out"):
                    shutdown_mcp_servers()

            assert mcp_mod._servers[server.name] is server
        finally:
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
                mcp_mod._retiring_mcp_profiles.clear()

    def test_shutdown_deregisters_registered_tools(self):
        """shutdown_mcp_servers removes MCP tools and their raw alias."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import MCPServerTask, shutdown_mcp_servers, _servers
        from tools.registry import registry
        from toolsets import resolve_toolset, validate_toolset

        _servers.clear()
        registry.register(
            name="mcp__test__ping",
            toolset="mcp-test",
            schema={
                "name": "mcp__test__ping",
                "description": "Ping",
                "parameters": {"type": "object", "properties": {}},
            },
            handler=lambda *_args, **_kwargs: "{}",
        )
        registry.register_toolset_alias("test", "mcp-test")

        server = MCPServerTask("test")
        server._registered_tool_names = ["mcp__test__ping"]
        _servers["test"] = server

        mcp_mod._ensure_mcp_loop()
        try:
            assert validate_toolset("test") is True
            assert "mcp__test__ping" in resolve_toolset("test")
            shutdown_mcp_servers()
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert "mcp__test__ping" not in registry.get_all_tool_names()
        assert validate_toolset("test") is False

    def test_shutdown_is_parallel(self):
        """Multiple servers are shut down in parallel via asyncio.gather."""
        import tools.mcp_tool as mcp_mod
        from tools.mcp_tool import shutdown_mcp_servers, _servers
        import time

        _servers.clear()

        # 4 servers each taking 50ms to shut down
        delay = 0.05
        for i in range(4):
            mock_server = MagicMock()
            mock_server.name = f"srv_{i}"
            async def slow_shutdown():
                await asyncio.sleep(delay)
            mock_server.shutdown = slow_shutdown
            _servers[f"srv_{i}"] = mock_server

        mcp_mod._ensure_mcp_loop()
        try:
            start = time.monotonic()
            shutdown_mcp_servers()
            elapsed = time.monotonic() - start
        finally:
            mcp_mod._mcp_loop = None
            mcp_mod._mcp_thread = None

        assert len(_servers) == 0
        # Parallel: ~1 delay, not 4. Margin covers scheduling jitter but stays
        # well under the serial total.
        assert elapsed < delay * 3, (
            f"Shutdown took {elapsed:.3f}s, expected ~{delay}s (parallel)"
        )


# ---------------------------------------------------------------------------
# _build_safe_env
# ---------------------------------------------------------------------------

class TestBuildSafeEnv:
    """Tests for _build_safe_env() environment filtering."""

    def test_only_safe_vars_passed(self):
        """Only safe baseline vars and XDG_* from os.environ are included."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "USER": "test",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C",
            "TERM": "xterm",
            "SHELL": "/bin/bash",
            "TMPDIR": "/tmp",
            "XDG_DATA_HOME": "/home/test/.local/share",
            "SECRET_KEY": "should_not_appear",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        # Safe vars present
        assert result["PATH"] == "/usr/bin"
        assert result["HOME"] == "/home/test"
        assert result["USER"] == "test"
        assert result["LANG"] == "en_US.UTF-8"
        assert result["XDG_DATA_HOME"] == "/home/test/.local/share"
        # Unsafe vars excluded
        assert "SECRET_KEY" not in result
        assert "AWS_ACCESS_KEY_ID" not in result

    def test_secret_vars_excluded(self):
        """Sensitive env vars from os.environ are NOT passed through."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": "/usr/bin",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "OPENAI_API_KEY": "sk-proj-abc123",
            "DATABASE_URL": "postgres://user:pass@localhost/db",
            "API_SECRET": "supersecret",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert "PATH" in result
        assert "AWS_SECRET_ACCESS_KEY" not in result
        assert "GITHUB_TOKEN" not in result
        assert "OPENAI_API_KEY" not in result
        assert "DATABASE_URL" not in result
        assert "API_SECRET" not in result

    def test_secret_source_injected_vars_are_passed(self, monkeypatch):
        """Vars tagged by an external secret source (Bitwarden/1Password) are
        deliberately allowed for MCP stdio servers."""
        from hermes_cli import env_loader
        from tools.mcp_tool import _build_safe_env

        monkeypatch.setitem(env_loader._SECRET_SOURCES, "ALPACA_API_KEY", "bitwarden")
        monkeypatch.setitem(env_loader._SECRET_SOURCES, "NOTION_TOKEN", "onepassword")
        fake_env = {
            "PATH": "/usr/bin",
            "ALPACA_API_KEY": "from-bws-key",
            "NOTION_TOKEN": "from-op",
            "UNTRACKED_SECRET_KEY": "still-filtered",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert result["PATH"] == "/usr/bin"
        assert result["ALPACA_API_KEY"] == "from-bws-key"
        assert result["NOTION_TOKEN"] == "from-op"
        assert "UNTRACKED_SECRET_KEY" not in result

    def test_windows_location_vars_passed_without_secrets(self):
        """Windows launcher tools need location vars, but secrets stay filtered."""
        from tools.mcp_tool import _build_safe_env

        fake_env = {
            "PATH": r"C:\Windows\System32",
            "ProgramFiles": r"C:\Program Files",
            "ProgramData": r"C:\ProgramData",
            "ProgramW6432": r"C:\Program Files",
            "LOCALAPPDATA": r"C:\Users\alice\AppData\Local",
            "APPDATA": r"C:\Users\alice\AppData\Roaming",
            "USERPROFILE": r"C:\Users\alice",
            "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "OPENAI_API_KEY": "sk-proj-abc123",
        }
        with patch.dict("os.environ", fake_env, clear=True):
            result = _build_safe_env(None)

        assert result["ProgramFiles"] == r"C:\Program Files"
        assert result["ProgramData"] == r"C:\ProgramData"
        assert result["ProgramW6432"] == r"C:\Program Files"
        assert result["LOCALAPPDATA"].endswith("Local")
        assert result["APPDATA"].endswith("Roaming")
        assert result["USERPROFILE"] == r"C:\Users\alice"
        assert "GITHUB_TOKEN" not in result
        assert "OPENAI_API_KEY" not in result


# ---------------------------------------------------------------------------
# _sanitize_error
# ---------------------------------------------------------------------------

class TestSanitizeError:
    """Tests for _sanitize_error() credential stripping."""

    def test_strips_credentials(self):
        from tools.mcp_tool import _sanitize_error

        for text, expected in (
            ("Error with ghp_abc123def456", "Error with [REDACTED]"),
            ("key sk-projABC123xyz", "key [REDACTED]"),
            ("Authorization: Bearer eyJabc123def", "Authorization: [REDACTED]"),
            ("url?token=secret123", "url?[REDACTED]"),
        ):
            assert _sanitize_error(text) == expected, text

        # Several credentials in one message are all masked.
        multi = _sanitize_error("ghp_abc123 and sk-projXyz789 and token=foo")
        assert "ghp_" not in multi and "sk-" not in multi and "token=" not in multi
        assert multi.count("[REDACTED]") == 3

    def test_no_credentials_unchanged(self):
        from tools.mcp_tool import _sanitize_error
        result = _sanitize_error("normal error message")
        assert result == "normal error message"

# ---------------------------------------------------------------------------
# HTTP config
# ---------------------------------------------------------------------------

class TestHTTPConfig:
    """Tests for HTTP transport detection and handling."""

    def test_is_http_with_url(self):
        from tools.mcp_tool import MCPServerTask
        server = MCPServerTask("remote")
        server._config = {"url": "https://example.com/mcp"}
        assert server._is_http() is True

    def test_http_unavailable_raises(self):
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("remote")
        config = {"url": "https://example.com/mcp"}

        async def _test():
            with patch("tools.mcp_tool._MCP_HTTP_AVAILABLE", False):
                with pytest.raises(ImportError, match="HTTP transport"):
                    await server._run_http(config)

        asyncio.run(_test())

    def test_stdio_unavailable_raises_importerror_not_nameerror(self):
        """Regression test for #30904.

        When the mcp SDK isn't installed, ``_run_stdio`` previously leaked a
        bare ``NameError: name 'StdioServerParameters' is not defined``. The
        gate now raises a clear ``ImportError`` with install instructions,
        mirroring ``_run_http``'s behaviour when the HTTP transport is
        unavailable.
        """
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask("local")
        config = {"command": "python3", "args": ["/tmp/echo.py"]}

        async def _test():
            with patch("tools.mcp_tool._MCP_AVAILABLE", False):
                with pytest.raises(ImportError, match=r"mcp.*SDK"):
                    await server._run_stdio(config)

        asyncio.run(_test())

# ---------------------------------------------------------------------------
# Reconnection logic
# ---------------------------------------------------------------------------

class TestReconnection:
    """Tests for automatic reconnection behavior in MCPServerTask.run()."""

    def test_reconnect_on_disconnect(self):
        """After initial success, a connection drop triggers reconnection."""
        from tools.mcp_tool import MCPServerTask

        run_count = 0
        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            nonlocal run_count, target_server
            run_count += 1
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            if run_count == 1:
                # First connection succeeds, then simulate disconnect
                self_srv.session = MagicMock()
                self_srv._tools = []
                self_srv._ready.set()
                raise ConnectionError("connection dropped")
            else:
                # Reconnection succeeds; signal shutdown so run() exits
                self_srv.session = MagicMock()
                self_srv._shutdown_event.set()
                await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"command": "test"})

            assert run_count >= 2  # At least one reconnection attempt

        asyncio.run(_test())


    def test_preflight_probe_runs_on_initial_http_connect(self):
        """The content-type preflight probe fires on the first HTTP connect."""
        from tools.mcp_tool import MCPServerTask

        target_server = None
        probe = AsyncMock()

        original_run_http = MCPServerTask._run_http

        async def patched_run_http(self_srv, config):
            if target_server is not self_srv:
                return await original_run_http(self_srv, config)
            # First connect succeeds; signal shutdown so run() exits cleanly.
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            self_srv._shutdown_event.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("http_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_http", patched_run_http), \
                 patch.object(MCPServerTask, "_preflight_content_type", probe), \
                 patch("asyncio.sleep", new_callable=AsyncMock):
                await server.run({"url": "https://example.com/mcp"})

            # Probe ran exactly once on the initial (pre-_ready) connect.
            assert probe.await_count == 1

        asyncio.run(_test())

# ---------------------------------------------------------------------------
# Configurable timeouts
# ---------------------------------------------------------------------------

class TestConfigurableTimeouts:
    """Tests for configurable per-server timeouts."""

    def test_custom_timeout(self):
        """Server with timeout=180 in config gets 180."""
        from tools.mcp_tool import MCPServerTask

        target_server = None

        original_run_stdio = MCPServerTask._run_stdio

        async def patched_run_stdio(self_srv, config):
            if target_server is not self_srv:
                return await original_run_stdio(self_srv, config)
            self_srv.session = MagicMock()
            self_srv._tools = []
            self_srv._ready.set()
            await self_srv._shutdown_event.wait()

        async def _test():
            nonlocal target_server
            server = MCPServerTask("test_srv")
            target_server = server

            with patch.object(MCPServerTask, "_run_stdio", patched_run_stdio):
                task = asyncio.ensure_future(
                    server.run({"command": "test", "timeout": 180})
                )
                await server._ready.wait()
                assert server.tool_timeout == 180
                server._shutdown_event.set()
                await task

        asyncio.run(_test())

    def test_timeout_passed_to_handler(self):
        """The tool handler uses the server's configured timeout."""
        from tools.mcp_tool import _make_tool_handler, _servers

        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(
            return_value=_make_call_result("ok", is_error=False)
        )
        server = _make_mock_server("test_srv", session=mock_session)
        server.tool_timeout = 180
        _servers["test_srv"] = server

        try:
            handler = _make_tool_handler("test_srv", "my_tool", 180)
            with patch("tools.mcp_tool._run_on_mcp_loop") as mock_run:
                def fake_run(coro, timeout=30):
                    coro.close()
                    return json.dumps({"result": "ok"})

                mock_run.side_effect = fake_run
                handler({})
                # Verify timeout=180 was passed
                call_kwargs = mock_run.call_args
                assert call_kwargs.kwargs.get("timeout") == 180 or \
                       (len(call_kwargs.args) > 1 and call_kwargs.args[1] == 180) or \
                       call_kwargs[1].get("timeout") == 180
        finally:
            _servers.pop("test_srv", None)


# ---------------------------------------------------------------------------
# Utility tool schemas (Resources & Prompts)
# ---------------------------------------------------------------------------

class TestUtilitySchemas:
    """Tests for _build_utility_schemas() and the schema format of utility tools."""

    def test_builds_four_utility_schemas(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("myserver")
        assert len(schemas) == 4
        names = [s["schema"]["name"] for s in schemas]
        assert "mcp__myserver__list_resources" in names
        assert "mcp__myserver__read_resource" in names
        assert "mcp__myserver__list_prompts" in names
        assert "mcp__myserver__get_prompt" in names

    def test_read_resource_schema_requires_uri(self):
        from tools.mcp_tool import _build_utility_schemas

        schemas = _build_utility_schemas("srv")
        rr = next(s for s in schemas if s["handler_key"] == "read_resource")
        params = rr["schema"]["parameters"]
        assert "uri" in params["properties"]
        assert params["properties"]["uri"]["type"] == "string"
        assert params["required"] == ["uri"]

# ---------------------------------------------------------------------------
# Utility tool handlers (Resources & Prompts)
# ---------------------------------------------------------------------------

class TestUtilityHandlers:
    """Tests for the MCP Resources & Prompts handler functions."""

    def _patch_mcp_loop(self):
        """Return a patch for _run_on_mcp_loop that runs the coroutine directly."""
        def fake_run(coro_or_factory, timeout=30):
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)
        return patch("tools.mcp_tool._run_on_mcp_loop", side_effect=fake_run)

    # -- list_resources --

    def test_list_resources_success(self):
        from tools.mcp_tool import _make_list_resources_handler, _servers

        mock_resource = SimpleNamespace(
            uri="file:///tmp/test.txt", name="test.txt",
            description="A test file", mimeType="text/plain",
        )
        mock_session = MagicMock()
        mock_session.list_resources = AsyncMock(
            return_value=SimpleNamespace(resources=[mock_resource])
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_list_resources_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({}))
            assert "resources" in result
            assert len(result["resources"]) == 1
            assert result["resources"][0]["uri"] == "file:///tmp/test.txt"
            assert result["resources"][0]["name"] == "test.txt"
        finally:
            _servers.pop("srv", None)


    # -- read_resource --


    # -- list_prompts --

    # -- get_prompt --

    def test_get_prompt_success(self):
        from tools.mcp_tool import _make_get_prompt_handler, _servers

        mock_msg = SimpleNamespace(
            role="assistant",
            content=SimpleNamespace(text="Here is a summary of your text."),
        )
        mock_session = MagicMock()
        mock_session.get_prompt = AsyncMock(
            return_value=SimpleNamespace(messages=[mock_msg], description=None)
        )
        server = _make_mock_server("srv", session=mock_session)
        _servers["srv"] = server

        try:
            handler = _make_get_prompt_handler("srv", 120)
            with self._patch_mcp_loop():
                result = json.loads(handler({"name": "summarize", "arguments": {"text": "hello"}}))
            assert "messages" in result
            assert len(result["messages"]) == 1
            assert result["messages"][0]["role"] == "assistant"
            assert "summary" in result["messages"][0]["content"].lower()
            mock_session.get_prompt.assert_called_once_with(
                "summarize", arguments={"text": "hello"}
            )
        finally:
            _servers.pop("srv", None)

# ---------------------------------------------------------------------------
# Utility tools registration in _discover_and_register_server
# ---------------------------------------------------------------------------

class TestUtilityToolRegistration:
    """Verify utility tools are registered alongside regular MCP tools."""

    def test_utility_tools_registered(self):
        """_discover_and_register_server registers all 4 utility tools."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()
        mock_tools = [_make_mcp_tool("read_file", "Read a file")]
        mock_session = MagicMock()

        async def fake_connect(name, config, **_kwargs):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("fs", {"command": "npx", "args": []})
            )

        # Regular tool + 4 utility tools
        assert "mcp__fs__read_file" in registered
        assert "mcp__fs__list_resources" in registered
        assert "mcp__fs__read_resource" in registered
        assert "mcp__fs__list_prompts" in registered
        assert "mcp__fs__get_prompt" in registered
        assert len(registered) == 5

        # All in the registry
        all_names = mock_registry.get_all_tool_names()
        for name in registered:
            assert name in all_names

        _servers.pop("fs", None)

# ===========================================================================
# SamplingHandler tests
# ===========================================================================


class _CompatType:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


try:
    from mcp.types import (
        CreateMessageResult,
        ErrorData,
        SamplingCapability,
        TextContent,
    )
except ImportError:
    CreateMessageResult = _CompatType
    ErrorData = _CompatType
    SamplingCapability = _CompatType
    TextContent = _CompatType

try:
    from mcp.types import CreateMessageResultWithTools
except ImportError:
    CreateMessageResultWithTools = _CompatType

try:
    from mcp.types import SamplingToolsCapability
except ImportError:
    SamplingToolsCapability = _CompatType

try:
    from mcp.types import ToolUseContent
except ImportError:
    ToolUseContent = _CompatType

from tools.mcp_tool import (
    CreateMessageResultWithTools,
    SamplingHandler,
    SamplingToolsCapability,
    ToolUseContent,
    _safe_numeric,
)


# ---------------------------------------------------------------------------
# Helpers for sampling tests
# ---------------------------------------------------------------------------

def _make_sampling_params(
    messages=None,
    max_tokens=100,
    system_prompt=None,
    model_preferences=None,
    temperature=None,
    stop_sequences=None,
    tools=None,
    tool_choice=None,
):
    """Create a fake CreateMessageRequestParams using SimpleNamespace.

    Each message must have a ``content_as_list`` attribute that mirrors
    the SDK helper so that ``_convert_messages`` works correctly.
    """
    if messages is None:
        content = SimpleNamespace(text="Hello")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        messages = [msg]

    params = SimpleNamespace(
        messages=messages,
        maxTokens=max_tokens,
        modelPreferences=model_preferences,
        temperature=temperature,
        stopSequences=stop_sequences,
        tools=tools,
        toolChoice=tool_choice,
    )
    if system_prompt is not None:
        params.systemPrompt = system_prompt
    return params


def _make_llm_response(
    content="LLM response",
    model="test-model",
    finish_reason="stop",
    tool_calls=None,
):
    """Create a fake OpenAI chat completion response (text)."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        finish_reason=finish_reason,
        message=message,
    )
    usage = SimpleNamespace(total_tokens=42)
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _make_llm_tool_response(tool_calls_data=None, model="test-model"):
    """Create a fake response with tool_calls.

    ``tool_calls_data``: list of (id, name, arguments_json) tuples.
    """
    if tool_calls_data is None:
        tool_calls_data = [("call_1", "get_weather", '{"city": "London"}')]

    tc_list = [
        SimpleNamespace(
            id=tc_id,
            function=SimpleNamespace(name=name, arguments=args),
        )
        for tc_id, name, args in tool_calls_data
    ]
    return _make_llm_response(
        content=None,
        model=model,
        finish_reason="tool_calls",
        tool_calls=tc_list,
    )


# ---------------------------------------------------------------------------
# 1. _safe_numeric helper
# ---------------------------------------------------------------------------

class TestSafeNumeric:
    def test_coercion_clamping_and_fallbacks(self):
        # (value, default, caster, kwargs, expected)
        cases = [
            (10, 5, int, {}, 10),
            ("20", 5, int, {}, 20),
            ("3.5", 1.0, float, {}, 3.5),
            (None, 7, int, {}, 7),
            ("abc", 42, int, {}, 42),
            (float("inf"), 3.0, float, {}, 3.0),
            (float("nan"), 4.0, float, {}, 4.0),
            (-5, 10, int, {"minimum": 1}, 1),
            (0, 10, int, {"minimum": 0}, 0),
        ]
        for value, default, caster, kwargs, expected in cases:
            assert _safe_numeric(value, default, caster, **kwargs) == expected, value

# ---------------------------------------------------------------------------
# 2. SamplingHandler initialization and config parsing
# ---------------------------------------------------------------------------

class TestSamplingHandlerInit:
    def test_defaults(self):
        h = SamplingHandler("srv", {})
        assert h.server_name == "srv"
        assert h.max_rpm == 10
        assert h.timeout == 30
        assert h.max_tokens_cap == 4096
        assert h.max_tool_rounds == 5
        assert h.model_override is None
        assert h.allowed_models == []
        assert h.metrics == {"requests": 0, "errors": 0, "tokens_used": 0, "tool_use_count": 0}

    def test_custom_config(self):
        cfg = {
            "max_rpm": 20,
            "timeout": 60,
            "max_tokens_cap": 2048,
            "max_tool_rounds": 3,
            "model": "gpt-4o",
            "allowed_models": ["gpt-4o", "gpt-3.5-turbo"],
            "log_level": "debug",
        }
        h = SamplingHandler("custom", cfg)
        assert h.max_rpm == 20
        assert h.timeout == 60.0
        assert h.max_tokens_cap == 2048
        assert h.max_tool_rounds == 3
        assert h.model_override == "gpt-4o"
        assert h.allowed_models == ["gpt-4o", "gpt-3.5-turbo"]

# ---------------------------------------------------------------------------
# 3. Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    def setup_method(self):
        self.handler = SamplingHandler("rl", {"max_rpm": 3})

    def test_rejects_over_limit(self):
        for _ in range(3):
            self.handler._check_rate_limit()
        assert self.handler._check_rate_limit() is False

    def test_window_expiry(self):
        """Old timestamps should be purged from the sliding window."""
        for _ in range(3):
            self.handler._check_rate_limit()
        # Simulate timestamps from 61 seconds ago
        self.handler._rate_timestamps[:] = [time.time() - 61] * 3
        assert self.handler._check_rate_limit() is True


# ---------------------------------------------------------------------------
# 4. Model resolution
# ---------------------------------------------------------------------------

class TestResolveModel:
    def setup_method(self):
        self.handler = SamplingHandler("mr", {})

    def test_config_override_wins(self):
        self.handler.model_override = "override-model"
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "override-model"

    def test_hint_used_when_no_override(self):
        prefs = SimpleNamespace(hints=[SimpleNamespace(name="hint-model")])
        assert self.handler._resolve_model(prefs) == "hint-model"

# ---------------------------------------------------------------------------
# 5. Message conversion
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def setup_method(self):
        self.handler = SamplingHandler("mc", {})

    def test_single_text_message(self):
        content = SimpleNamespace(text="Hello world")
        msg = SimpleNamespace(role="user", content=content, content_as_list=[content])
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello world"}


    def test_tool_use_message(self):
        tu_block = SimpleNamespace(
            id="call_2", name="get_weather", input={"city": "London"}
        )
        msg = SimpleNamespace(
            role="assistant",
            content=[tu_block],
            content_as_list=[tu_block],
        )
        params = _make_sampling_params(messages=[msg])
        result = self.handler._convert_messages(params)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {"city": "London"}

# ---------------------------------------------------------------------------
# 6. Text-only sampling callback (full flow)
# ---------------------------------------------------------------------------

class TestSamplingCallbackText:
    def setup_method(self):
        self.handler = SamplingHandler("txt", {})

    def test_text_response(self):
        """Full flow: text response returns CreateMessageResult."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response(
            content="Hello from LLM"
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResult)
        assert isinstance(result.content, TextContent)
        assert result.content.text == "Hello from LLM"
        assert result.model == "test-model"
        assert result.role == "assistant"
        assert result.stopReason == "endTurn"

    def test_server_tools_with_object_schema_are_normalized(self):
        """Server-provided tools should gain empty properties for object schemas."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()
        server_tool = SimpleNamespace(
            name="ask",
            description="Ask Crawl4AI",
            inputSchema={"type": "object"},
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ) as mock_call:
            params = _make_sampling_params(tools=[server_tool])
            asyncio.run(self.handler(None, params))

        tools = mock_call.call_args.kwargs["tools"]
        assert tools == [{
            "type": "function",
            "function": {
                "name": "ask",
                "description": "Ask Crawl4AI",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

# ---------------------------------------------------------------------------
# 7. Tool use sampling callback
# ---------------------------------------------------------------------------

class TestSamplingCallbackToolUse:
    def setup_method(self):
        self.handler = SamplingHandler("tu", {})

    def test_tool_use_response(self):
        """LLM tool_calls response returns CreateMessageResultWithTools."""
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            result = asyncio.run(self.handler(None, params))

        assert isinstance(result, CreateMessageResultWithTools)
        assert result.stopReason == "toolUse"
        assert result.model == "test-model"
        assert len(result.content) == 1
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.name == "get_weather"
        assert tc.id == "call_1"
        assert tc.input == {"city": "London"}

# ---------------------------------------------------------------------------
# 8. Tool loop governance
# ---------------------------------------------------------------------------

class TestToolLoopGovernance:
    def test_max_tool_rounds_enforcement(self):
        """After max_tool_rounds consecutive tool responses, an error is returned."""
        handler = SamplingHandler("tl", {"max_tool_rounds": 2})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            params = _make_sampling_params()
            # Round 1, 2: allowed
            r1 = asyncio.run(handler(None, params))
            assert isinstance(r1, CreateMessageResultWithTools)
            r2 = asyncio.run(handler(None, params))
            assert isinstance(r2, CreateMessageResultWithTools)
            # Round 3: exceeds limit
            r3 = asyncio.run(handler(None, params))
            assert isinstance(r3, ErrorData)
            assert "Tool loop limit exceeded" in r3.message

    def test_max_tool_rounds_zero_disables(self):
        """max_tool_rounds=0 means tool loops are disabled entirely."""
        handler = SamplingHandler("tl3", {"max_tool_rounds": 0})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "Tool loops disabled" in result.message


# ---------------------------------------------------------------------------
# 9. Error paths: rate limit, timeout, no provider
# ---------------------------------------------------------------------------

class TestSamplingErrors:
    def test_rate_limit_error(self):
        handler = SamplingHandler("rle", {"max_rpm": 1})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            # First call succeeds
            r1 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r1, CreateMessageResult)
            # Second call is rate limited
            r2 = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(r2, ErrorData)
            assert "rate limit" in r2.message.lower()
            assert handler.metrics["errors"] == 1

    def test_timeout_error(self):
        # Config values clamp to a 1s floor (_safe_numeric minimum), so set the
        # attribute directly to exercise the timeout branch without a 1s wait.
        handler = SamplingHandler("to", {})
        handler.timeout = 0.05

        def slow_call(**kwargs):
            import threading
            evt = threading.Event()
            # Outlives the 0.05s handler timeout, but short enough that the
            # abandoned worker thread doesn't stall loop shutdown.
            evt.wait(0.15)
            return _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            side_effect=slow_call,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "timed out" in result.message.lower()
            assert handler.metrics["errors"] == 1

    def test_empty_choices_returns_error(self):
        """LLM returning choices=[] is handled gracefully, not IndexError."""
        handler = SamplingHandler("ec", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            model="test-model",
            usage=SimpleNamespace(total_tokens=0),
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, ErrorData)
        assert "empty response" in result.message.lower()
        assert handler.metrics["errors"] == 1

# ---------------------------------------------------------------------------
# 10. Model whitelist
# ---------------------------------------------------------------------------

class TestModelWhitelist:
    def test_allowed_model_passes(self):
        handler = SamplingHandler("wl", {"allowed_models": ["gpt-4o", "test-model"]})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, CreateMessageResult)

    def test_disallowed_model_rejected(self):
        handler = SamplingHandler("wl2", {"allowed_models": ["gpt-4o"], "model": "test-model"})
        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))
            assert isinstance(result, ErrorData)
            assert "not allowed" in result.message
            assert handler.metrics["errors"] == 1

# ---------------------------------------------------------------------------
# 11. Malformed tool_call arguments
# ---------------------------------------------------------------------------

class TestMalformedToolCallArgs:
    def test_invalid_json_wrapped_as_raw(self):
        """Malformed JSON arguments get wrapped in {"_raw": ...}."""
        handler = SamplingHandler("mf", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_tool_response(
            tool_calls_data=[("call_x", "some_tool", "not valid json {{{")]
        )

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            result = asyncio.run(handler(None, _make_sampling_params()))

        assert isinstance(result, CreateMessageResultWithTools)
        tc = result.content[0]
        assert isinstance(tc, ToolUseContent)
        assert tc.input == {"_raw": "not valid json {{{"}

# ---------------------------------------------------------------------------
# 12. Metrics tracking
# ---------------------------------------------------------------------------

class TestMetricsTracking:
    def test_request_and_token_metrics(self):
        handler = SamplingHandler("met", {})
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _make_llm_response()

        with patch(
            "agent.auxiliary_client.call_llm",
            return_value=fake_client.chat.completions.create.return_value,
        ):
            asyncio.run(handler(None, _make_sampling_params()))

        assert handler.metrics["requests"] == 1
        assert handler.metrics["tokens_used"] == 42
        assert handler.metrics["errors"] == 0

# ---------------------------------------------------------------------------
# 13. session_kwargs()
# ---------------------------------------------------------------------------

class TestSessionKwargs:
    def test_returns_correct_keys(self):
        handler = SamplingHandler("sk", {})
        kwargs = handler.session_kwargs()
        assert "sampling_callback" in kwargs
        assert "sampling_capabilities" in kwargs
        assert kwargs["sampling_callback"] is handler

# ---------------------------------------------------------------------------
# 14. MCPServerTask integration
# ---------------------------------------------------------------------------

class TestMCPServerTaskSamplingIntegration:
    def test_sampling_handler_created_when_enabled(self):
        """MCPServerTask.run() creates a SamplingHandler when sampling is enabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test")
        config = {
            "command": "fake",
            "sampling": {"enabled": True, "max_rpm": 5},
        }
        # We only need to test the setup logic, not the actual connection.
        # Calling run() would attempt a real connection, so we test the
        # sampling setup portion directly.
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is not None
        assert isinstance(server._sampling, SamplingHandler)
        assert server._sampling.server_name == "int_test"
        assert server._sampling.max_rpm == 5

    def test_sampling_handler_none_when_disabled(self):
        """MCPServerTask._sampling is None when sampling is disabled."""
        from tools.mcp_tool import MCPServerTask, _MCP_SAMPLING_TYPES

        server = MCPServerTask("int_test2")
        config = {
            "command": "fake",
            "sampling": {"enabled": False},
        }
        server._config = config
        sampling_config = config.get("sampling", {})
        if sampling_config.get("enabled", True) and _MCP_SAMPLING_TYPES:
            server._sampling = SamplingHandler(server.name, sampling_config)
        else:
            server._sampling = None

        assert server._sampling is None

# ---------------------------------------------------------------------------
# Discovery failed_count tracking
# ---------------------------------------------------------------------------

class TestDiscoveryFailedCount:
    """Verify discover_mcp_tools() correctly tracks failed server connections."""

    def test_failed_server_increments_failed_count(self):
        """When _discover_and_register_server raises, failed_count increments."""
        from tools.mcp_tool import discover_mcp_tools, _servers, _ensure_mcp_loop

        fake_config = {
            "good_server": {"command": "npx", "args": ["good"]},
            "bad_server": {"command": "npx", "args": ["bad"]},
        }

        async def fake_register(name, cfg, **_kwargs):
            if name == "bad_server":
                raise ConnectionError("Connection refused")
            # Simulate successful registration
            from tools.mcp_tool import MCPServerTask
            server = MCPServerTask(name)
            server.session = MagicMock()
            server._tools = [_make_mcp_tool("tool_a")]
            _servers[name] = server
            return [f"mcp__{name}__tool_a"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__good_server__tool_a"]):
            _ensure_mcp_loop()

            # Capture the logger to verify failed_count in summary
            with patch("tools.mcp_tool.logger") as mock_logger:
                discover_mcp_tools()

                # Find the summary info call
                info_calls = [
                    str(call)
                    for call in mock_logger.info.call_args_list
                    if "failed" in str(call).lower() or "MCP:" in str(call)
                ]
                # The summary should mention the failure
                assert any("1 failed" in str(c) for c in info_calls), (
                    f"Summary should report 1 failed server, got: {info_calls}"
                )

        _servers.pop("good_server", None)
        _servers.pop("bad_server", None)

    def test_ok_servers_excludes_failures(self):
        """ok_servers count correctly excludes failed servers."""
        from tools.mcp_tool import discover_mcp_tools, _servers, _ensure_mcp_loop

        fake_config = {
            "ok1": {"command": "npx", "args": ["ok1"]},
            "ok2": {"command": "npx", "args": ["ok2"]},
            "fail1": {"command": "npx", "args": ["fail"]},
        }

        async def selective_register(name, cfg, **_kwargs):
            if name == "fail1":
                raise ConnectionError("Refused")
            from tools.mcp_tool import MCPServerTask
            server = MCPServerTask(name)
            server.session = MagicMock()
            server._tools = [_make_mcp_tool("t")]
            _servers[name] = server
            return [f"mcp__{name}__t"]

        with patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=selective_register), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__ok1__t", "mcp__ok2__t"]):
            _ensure_mcp_loop()

            with patch("tools.mcp_tool.logger") as mock_logger:
                discover_mcp_tools()

                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                # Should say "2 server(s)" not "3 server(s)"
                assert any("2 server" in str(c) for c in info_calls), (
                    f"Summary should report 2 ok servers, got: {info_calls}"
                )
                assert any("1 failed" in str(c) for c in info_calls), (
                    f"Summary should report 1 failed, got: {info_calls}"
                )

        _servers.pop("ok1", None)
        _servers.pop("ok2", None)
        _servers.pop("fail1", None)


class TestMCPSelectiveToolLoading:
    """Tests for per-server MCP filtering and utility tool policies."""

    def _make_server(self, name, tool_names, session=None):
        server = _make_mock_server(
            name,
            session=session or SimpleNamespace(),
            tools=[_make_mcp_tool(n, n) for n in tool_names],
        )
        return server

    def _run_discover(self, name, tool_names, config, session=None):
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers

        mock_registry = ToolRegistry()
        server = self._make_server(name, tool_names, session=session)

        async def fake_connect(_name, _config, **_kwargs):
            return server

        async def run():
            with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
                 patch("tools.registry.registry", mock_registry), \
                 patch("toolsets.create_custom_toolset"):
                return await _discover_and_register_server(name, config)

        try:
            registered = asyncio.run(run())
        finally:
            _servers.pop(name, None)
        return registered, mock_registry

    def test_include_takes_precedence_over_exclude(self):
        config = {
            "url": "https://mcp.example.com",
            "tools": {
                "include": ["create_service"],
                "exclude": ["create_service", "delete_service"],
            },
        }
        registered, _ = self._run_discover(
            "ink",
            ["create_service", "delete_service", "list_services"],
            config,
            session=SimpleNamespace(),
        )
        assert registered == ["mcp__ink__create_service"]


    def test_enabled_false_skips_connection_attempt(self):
        from tools.mcp_tool import discover_mcp_tools

        connect_called = []

        async def fake_connect(name, config, **_kwargs):
            connect_called.append(name)
            return self._make_server(name, ["create_service"])

        fake_config = {
            "ink": {
                "url": "https://mcp.example.com",
                "enabled": False,
            }
        }
        fake_toolsets = {
            "hermes-cli": {"tools": [], "description": "CLI", "includes": []},
        }

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._servers", {}), \
             patch("tools.mcp_tool._load_mcp_config", return_value=fake_config), \
             patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("toolsets.TOOLSETS", fake_toolsets):
            result = discover_mcp_tools()

        assert connect_called == []
        assert result == []


def test_probe_cleanup_failure_is_explicit_and_keeps_live_owner(monkeypatch):
    import tools.mcp_tool as mcp_tool

    created = []

    class _CleanupFailingServer(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            created.append(self)

        async def start(self, _config):
            self._tools = [_make_mcp_tool("tool")]

        async def shutdown(self):
            raise RuntimeError("child still alive")

    def run_inline(coro_or_factory, timeout=30, **_kwargs):
        del timeout
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp_tool,
        "_load_mcp_config",
        lambda: {"probe": {"command": "probe"}},
    )
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool, "MCPServerTask", _CleanupFailingServer)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)

    try:
        with pytest.raises(RuntimeError, match="probe cleanup failed"):
            mcp_tool.probe_mcp_server_tools()
        assert len(created) == 1
        with mcp_tool._lock:
            assert created[0] in mcp_tool._live_mcp_servers
    finally:
        with mcp_tool._lock:
            for server in created:
                mcp_tool._live_mcp_servers.discard(server)


def test_probe_carries_generation_captured_before_config_load(monkeypatch):
    """config load 推进 generation 后，probe 的旧代次不得启动 transport。"""
    import tools.mcp_tool as mcp_tool

    created = []

    class _Server(mcp_tool.MCPServerTask):
        def __init__(self, name):
            super().__init__(name)
            self.start_called = False
            created.append(self)

        async def start(self, _config):
            self.start_called = True

    profile_identity = mcp_tool._current_mcp_profile_identity()
    missing = object()
    with mcp_tool._lock:
        original_generation = mcp_tool._mcp_profile_generations.get(
            profile_identity, missing
        )

    def load_and_retire():
        with mcp_tool._lock:
            mcp_tool._mcp_profile_generations[profile_identity] = (
                mcp_tool._mcp_profile_generations.get(profile_identity, 0) + 1
            )
        return {"probe": {"command": "probe"}}

    def run_inline(coro_or_factory, timeout=30, **_kwargs):
        del timeout
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_load_mcp_config", load_and_retire)
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
    monkeypatch.setattr(mcp_tool, "_stop_mcp_loop_if_idle", lambda: True)
    monkeypatch.setattr(mcp_tool, "MCPServerTask", _Server)

    try:
        assert mcp_tool.probe_mcp_server_tools() == {}
        assert len(created) == 1
        assert created[0].start_called is False
        with mcp_tool._lock:
            assert created[0] not in mcp_tool._live_mcp_servers
    finally:
        with mcp_tool._lock:
            if original_generation is missing:
                mcp_tool._mcp_profile_generations.pop(profile_identity, None)
            else:
                mcp_tool._mcp_profile_generations[profile_identity] = (
                    original_generation
                )


# ---------------------------------------------------------------------------
# Tool name collision protection
# ---------------------------------------------------------------------------

class TestRegistryCollisionWarning:
    """registry.register() warns when a tool name is overwritten by a different toolset."""

    def test_overwrite_different_toolset_logs_warning(self, caplog):
        """Overwriting a tool from a different toolset is REJECTED with an error."""
        from tools.registry import ToolRegistry
        import logging

        reg = ToolRegistry()
        schema = {"name": "my_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}
        handler = lambda args, **kw: "{}"

        reg.register(name="my_tool", toolset="builtin", schema=schema, handler=handler)

        with caplog.at_level(logging.ERROR, logger="tools.registry"):
            reg.register(name="my_tool", toolset="mcp-ext", schema=schema, handler=handler)

        assert any("rejected" in r.message.lower() for r in caplog.records)
        assert any("builtin" in r.message and "mcp-ext" in r.message for r in caplog.records)
        # The original tool should still be from 'builtin', not overwritten
        assert reg.get_toolset_for_tool("my_tool") == "builtin"

class TestMCPBuiltinCollisionGuard:
    """MCP tools that collide with built-in tool names are skipped."""

    def test_mcp_tool_skipped_when_builtin_exists(self):
        """An MCP tool whose prefixed name collides with a built-in is skipped."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()

        # Pre-register a "built-in" tool with the name that the MCP tool would produce.
        # Server "abc", tool "search" → mcp_abc_search
        builtin_schema = {
            "name": "mcp__abc__search",
            "description": "A hypothetical built-in",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp__abc__search", toolset="web",
            schema=builtin_schema, handler=lambda a, **k: "{}",
        )

        mock_tools = [_make_mcp_tool("search", "Search the web")]
        mock_session = MagicMock()

        async def fake_connect(name, config, **_kwargs):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("abc", {"command": "test", "args": []})
            )

        # The MCP tool should have been skipped — built-in preserved.
        assert "mcp__abc__search" not in registered
        assert mock_registry.get_toolset_for_tool("mcp__abc__search") == "web"

        _servers.pop("abc", None)

    def test_mcp_tool_rejected_when_collision_is_another_mcp(self):
        """Cross-server MCP collisions preserve the existing owner."""
        from tools.registry import ToolRegistry
        from tools.mcp_tool import _discover_and_register_server, _servers, MCPServerTask

        mock_registry = ToolRegistry()

        # Pre-register an MCP tool from a different server.
        mcp_schema = {
            "name": "mcp__srv__do_thing",
            "description": "From another MCP server",
            "parameters": {"type": "object", "properties": {}},
        }
        mock_registry.register(
            name="mcp__srv__do_thing", toolset="mcp-old",
            schema=mcp_schema, handler=lambda a, **k: "{}",
        )

        mock_tools = [_make_mcp_tool("do_thing", "Do a thing")]
        mock_session = MagicMock()

        async def fake_connect(name, config, **_kwargs):
            server = MCPServerTask(name)
            server.session = mock_session
            server._tools = mock_tools
            return server

        with patch("tools.mcp_tool._connect_server", side_effect=fake_connect), \
             patch("tools.registry.registry", mock_registry):
            registered = asyncio.run(
                _discover_and_register_server("srv", {"command": "test", "args": []})
            )

        # Cross-server MCP collisions fail closed: the existing owner stays active.
        assert "mcp__srv__do_thing" not in registered
        entry = mock_registry.get_entry("mcp__srv__do_thing")
        assert entry is not None
        assert entry.toolset == "mcp-old"
        assert entry.schema["description"] == "From another MCP server"
        assert mock_registry.get_toolset_for_tool("mcp__srv__do_thing") == "mcp-old"

        _servers.pop("srv", None)


# ---------------------------------------------------------------------------
# sanitize_mcp_name_component
# ---------------------------------------------------------------------------


class TestSanitizeMcpNameComponent:
    """Verify sanitize_mcp_name_component handles all edge cases."""

    def test_hyphens_replaced(self):
        from tools.mcp_tool import sanitize_mcp_name_component
        assert sanitize_mcp_name_component("my-server") == "my_server"


    def test_slash_in_server_alias_resolution(self):
        """Server names with slashes resolve through their live MCP alias."""
        from tools.registry import ToolRegistry
        from toolsets import resolve_toolset, validate_toolset

        reg = ToolRegistry()
        reg.register(
            name="mcp__ai_exa_exa__search",
            toolset="mcp-ai.exa/exa",
            schema={"name": "mcp__ai_exa_exa__search", "description": "Search", "parameters": {"type": "object", "properties": {}}},
            handler=lambda *_args, **_kwargs: "{}",
        )
        reg.register_toolset_alias("ai.exa/exa", "mcp-ai.exa/exa")

        with patch("tools.registry.registry", reg):
            assert validate_toolset("ai.exa/exa") is True
            assert "mcp__ai_exa_exa__search" in resolve_toolset("ai.exa/exa")


# ---------------------------------------------------------------------------
# register_mcp_servers public API
# ---------------------------------------------------------------------------


class TestRegisterMcpServers:
    """Verify the new register_mcp_servers() public API."""

    def test_mcp_not_available_returns_empty(self):
        from tools.mcp_tool import register_mcp_servers

        with patch("tools.mcp_tool._MCP_AVAILABLE", False):
            result = register_mcp_servers({"srv": {"command": "test"}})
        assert result == []


    def test_connects_new_servers(self):
        from tools.mcp_tool import register_mcp_servers, _servers, _ensure_mcp_loop

        fake_config = {"my_server": {"command": "npx", "args": ["test"]}}

        async def fake_register(name, cfg, **_kwargs):
            server = _make_mock_server(name)
            server._registered_tool_names = ["mcp__my_server__tool1"]
            _servers[name] = server
            return ["mcp__my_server__tool1"]

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
             patch("tools.mcp_tool._existing_tool_names", return_value=["mcp__my_server__tool1"]):
            _ensure_mcp_loop()
            result = register_mcp_servers(fake_config)

        assert "mcp__my_server__tool1" in result
        _servers.pop("my_server", None)

    def test_same_named_server_is_reused_only_within_one_profile(
        self, tmp_path, monkeypatch
    ):
        """A/B profile 的同名 MCP 必须各自启动，不能复用对方的凭据进程。"""
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        server_name = "profile_isolation_probe"
        config = {server_name: {"command": "profile-mcp"}}
        spawned = []

        async def fake_register(name, _config, **_kwargs):
            env = mcp_tool._build_safe_env({})
            server = SimpleNamespace(
                name=name,
                session=object(),
                _registered_tool_names=[f"mcp__{name}__probe"],
            )
            spawned.append((mcp_tool._current_mcp_profile_identity(), env, server))
            mcp_tool._servers[name] = server
            return list(server._registered_tool_names)

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(
            mcp_tool, "_filter_suspicious_mcp_servers", lambda servers: servers
        )
        monkeypatch.setattr(mcp_tool, "_ensure_mcp_loop", lambda: None)
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        monkeypatch.setattr(
            mcp_tool, "_discover_and_register_server", fake_register
        )

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        try:
            token = set_hermes_home_override(profile_a)
            try:
                mcp_tool.register_mcp_servers(config)
                server_a = mcp_tool._servers[server_name]
                # 同一 profile 的重复装配仍应复用，不增加子进程。
                mcp_tool.register_mcp_servers(config)
            finally:
                reset_hermes_home_override(token)

            token = set_hermes_home_override(profile_b)
            try:
                mcp_tool.register_mcp_servers(config)
                server_b = mcp_tool._servers[server_name]
            finally:
                reset_hermes_home_override(token)

            assert len(spawned) == 2
            assert server_a is not server_b
            assert spawned[0][1]["WECOM_CLI_CONFIG_DIR"] == str(
                profile_a / "wecom-cli-config"
            )
            assert spawned[1][1]["WECOM_CLI_CONFIG_DIR"] == str(
                profile_b / "wecom-cli-config"
            )
        finally:
            for profile in (profile_a, profile_b):
                token = set_hermes_home_override(profile)
                try:
                    mcp_tool._servers.pop(server_name, None)
                    mcp_tool._server_connecting.discard(server_name)
                    mcp_tool._server_connect_errors.pop(server_name, None)
                    mcp_tool._parallel_safe_servers.discard(server_name)
                finally:
                    reset_hermes_home_override(token)

    def test_all_name_keyed_mcp_lifecycle_state_is_profile_scoped(
        self, tmp_path
    ):
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        mapping_names = tuple(mcp_tool._PROFILE_SCOPED_MCP_MAPPINGS)
        set_names = tuple(mcp_tool._PROFILE_SCOPED_MCP_SETS)
        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        probe_key = "profile-state-probe"

        token = set_hermes_home_override(profile_a)
        try:
            for name in mapping_names:
                getattr(mcp_tool, name)[probe_key] = object()
            for name in set_names:
                getattr(mcp_tool, name).add(probe_key)
        finally:
            reset_hermes_home_override(token)

        try:
            token = set_hermes_home_override(profile_b)
            try:
                for name in mapping_names:
                    assert probe_key not in getattr(mcp_tool, name), name
                for name in set_names:
                    assert probe_key not in getattr(mcp_tool, name), name
            finally:
                reset_hermes_home_override(token)
        finally:
            token = set_hermes_home_override(profile_a)
            try:
                for name in mapping_names:
                    getattr(mcp_tool, name).pop(probe_key, None)
                for name in set_names:
                    getattr(mcp_tool, name).discard(probe_key)
            finally:
                reset_hermes_home_override(token)

        tree = ast.parse(Path(mcp_tool.__file__).read_text(encoding="utf-8"))

        def module_control_flow(statements):
            """下降模块控制流块，但不把函数/类内部赋值算作全局状态。"""
            for node in statements:
                yield node
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                child_blocks = []
                for field in ("body", "orelse", "finalbody"):
                    block = getattr(node, field, None)
                    if isinstance(block, list):
                        child_blocks.append(block)
                for handler in getattr(node, "handlers", []):
                    child_blocks.append(handler.body)
                for case in getattr(node, "cases", []):
                    child_blocks.append(case.body)
                for block in child_blocks:
                    yield from module_control_flow(block)

        constructed = {
            node.target.id
            for node in module_control_flow(tree.body)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in {"_ProfileScopedMapping", "_ProfileScopedSet"}
        }
        assert constructed == set(mapping_names) | set(set_names)

        from collections.abc import MutableMapping, MutableSet

        module_mutables = {
            name
            for name, value in vars(mcp_tool).items()
            if not name.startswith("__")
            and isinstance(value, (MutableMapping, MutableSet, list))
        }

        classified = (
            set(mapping_names)
            | set(set_names)
            | set(mcp_tool._MCP_PROFILE_CUSTOM_MUTABLES)
            | set(mcp_tool._MCP_PROCESS_GLOBAL_MUTABLES)
            | set(mcp_tool._MCP_LIFECYCLE_INDEX_MUTABLES)
        )
        assert module_mutables == classified

    def test_profile_shutdown_keeps_sibling_profile_running(
        self, tmp_path, monkeypatch
    ):
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        server_a = SimpleNamespace(name="shared", shutdown=AsyncMock())
        server_b = SimpleNamespace(name="shared", shutdown=AsyncMock())

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        fake_loop = SimpleNamespace(is_running=lambda: True)
        monkeypatch.setattr(mcp_tool, "_mcp_loop", fake_loop)
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)

        token = set_hermes_home_override(profile_a)
        try:
            mcp_tool._servers["shared"] = server_a
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(profile_b)
        try:
            mcp_tool._servers["shared"] = server_b
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(profile_a)
        try:
            mcp_tool.shutdown_mcp_profile()
            assert "shared" not in mcp_tool._servers
        finally:
            reset_hermes_home_override(token)

        token = set_hermes_home_override(profile_b)
        try:
            assert mcp_tool._servers["shared"] is server_b
            mcp_tool._servers.pop("shared", None)
        finally:
            reset_hermes_home_override(token)

        server_a.shutdown.assert_awaited_once()
        server_b.shutdown.assert_not_awaited()

    def test_profile_shutdown_failure_keeps_state_for_retry(
        self, tmp_path, monkeypatch
    ):
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile = tmp_path / "profiles" / "a"
        server = SimpleNamespace(
            name="shared",
            shutdown=AsyncMock(side_effect=RuntimeError("close failed")),
        )

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(
            mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        token = set_hermes_home_override(profile)
        try:
            mcp_tool._servers["shared"] = server
            with pytest.raises(RuntimeError, match="failed to close 1 MCP server"):
                mcp_tool.shutdown_mcp_profile()
            assert mcp_tool._servers["shared"] is server
        finally:
            mcp_tool._servers.pop("shared", None)
            reset_hermes_home_override(token)

    def test_profile_shutdown_partial_failure_commits_success_before_retry(
        self, tmp_path, monkeypatch
    ):
        """profile 批量关闭部分失败时，只重试仍失败的 exact owner。"""
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile = tmp_path / "profiles" / "a"
        good = SimpleNamespace(name="good", shutdown=AsyncMock())
        flaky = SimpleNamespace(
            name="flaky",
            shutdown=AsyncMock(side_effect=[RuntimeError("still alive"), None]),
        )

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(
            mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", lambda **_kwargs: True)
        token = set_hermes_home_override(profile)
        try:
            mcp_tool._servers["good"] = good
            mcp_tool._servers["flaky"] = flaky
            with pytest.raises(RuntimeError, match="failed to close 1 MCP server"):
                mcp_tool.shutdown_mcp_profile()
            assert "good" not in mcp_tool._servers
            assert mcp_tool._servers["flaky"] is flaky

            mcp_tool.shutdown_mcp_profile()
            assert good.shutdown.await_count == 1
            assert flaky.shutdown.await_count == 2
            assert "flaky" not in mcp_tool._servers
        finally:
            mcp_tool._servers.pop("good", None)
            mcp_tool._servers.pop("flaky", None)
            reset_hermes_home_override(token)

    def test_global_shutdown_partial_failure_commits_success_before_retry(
        self, monkeypatch
    ):
        """全局批量关闭也只重试失败 owner，不能双清已成功 server。"""
        import tools.mcp_tool as mcp_tool

        good = SimpleNamespace(
            name="global-good",
            profile_identity=mcp_tool._current_mcp_profile_identity(),
            shutdown=AsyncMock(),
        )
        flaky = SimpleNamespace(
            name="global-flaky",
            profile_identity=mcp_tool._current_mcp_profile_identity(),
            shutdown=AsyncMock(side_effect=[RuntimeError("still alive"), None]),
        )

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(
            mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", lambda **_kwargs: True)
        mcp_tool._servers[good.name] = good
        mcp_tool._servers[flaky.name] = flaky
        try:
            with pytest.raises(RuntimeError, match="failed to close 1 MCP server"):
                mcp_tool.shutdown_mcp_servers()
            assert good.name not in mcp_tool._servers
            assert mcp_tool._servers[flaky.name] is flaky

            mcp_tool.shutdown_mcp_servers()
            assert good.shutdown.await_count == 1
            assert flaky.shutdown.await_count == 2
        finally:
            mcp_tool._servers.pop(good.name, None)
            mcp_tool._servers.pop(flaky.name, None)

    def test_mcp_stderr_log_is_profile_scoped(self, tmp_path):
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        paths = []
        for profile in (tmp_path / "profiles" / "a", tmp_path / "profiles" / "b"):
            token = set_hermes_home_override(profile)
            try:
                paths.append(Path(mcp_tool._get_mcp_stderr_log().name))
            finally:
                reset_hermes_home_override(token)

        assert paths == [
            tmp_path / "profiles" / "a" / "logs" / "mcp-stderr.log",
            tmp_path / "profiles" / "b" / "logs" / "mcp-stderr.log",
        ]
        mcp_tool._close_mcp_stderr_logs()

    def test_mcp_availability_check_is_never_cached_across_profile_switches(self):
        import tools.mcp_tool as mcp_tool

        check = mcp_tool._make_check_fn("shared", "mcp__shared__probe")
        assert check._session_scope_sensitive is True

    @pytest.mark.parametrize("registration_path", ["live", "lazy_cache"])
    def test_normalized_tool_name_routes_to_each_profiles_exact_handler(
        self, tmp_path, monkeypatch, registration_path
    ):
        """全局 Registry 的同名入口必须按 profile 路由到不同 handler。"""
        import tools.mcp_tool as mcp_tool
        import tools.registry as registry_module
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        created_handlers = []

        def make_handler(server_name, raw_tool_name, _timeout):
            marker = object()

            def handler(_args, **_kwargs):
                return json.dumps(
                    {"server": server_name, "raw_tool": raw_tool_name, "marker": id(marker)}
                )

            created_handlers.append(handler)
            return handler

        monkeypatch.setattr(registry_module, "registry", registry)
        monkeypatch.setattr(mcp_tool, "_make_tool_handler", make_handler)

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        expected = {}
        try:
            for profile, raw_name in (
                (profile_a, "foo-bar"),
                (profile_b, "foo_bar"),
            ):
                token = set_hermes_home_override(profile)
                try:
                    if registration_path == "live":
                        server = SimpleNamespace(
                            _tools=[_make_mcp_tool(raw_name)],
                            session=SimpleNamespace(),
                            tool_timeout=30,
                            initialize_result=None,
                        )
                        registered = mcp_tool._register_server_tools(
                            "shared", server, {}
                        )
                    else:
                        registered = mcp_tool._register_from_cache_sync(
                            "shared",
                            {},
                            {
                                "tools": [
                                    {
                                        "name": raw_name,
                                        "description": "probe",
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {},
                                        },
                                    }
                                ],
                                "utility_tools": [],
                            },
                        )
                    assert registered == ["mcp__shared__foo_bar"]
                    entry = registry.get_entry("mcp__shared__foo_bar")
                    expected[profile] = json.loads(entry.handler({}))
                finally:
                    reset_hermes_home_override(token)

            assert len(created_handlers) == 2
            assert created_handlers[0] is not created_handlers[1]
            assert expected[profile_a]["raw_tool"] == "foo-bar"
            assert expected[profile_b]["raw_tool"] == "foo_bar"

            for profile, raw_name in (
                (profile_a, "foo-bar"),
                (profile_b, "foo_bar"),
            ):
                token = set_hermes_home_override(profile)
                try:
                    routed = json.loads(
                        registry.get_entry("mcp__shared__foo_bar").handler({})
                    )
                    assert routed["raw_tool"] == raw_name
                    assert routed["marker"] == expected[profile]["marker"]
                finally:
                    reset_hermes_home_override(token)

            token = set_hermes_home_override(profile_a)
            try:
                mcp_tool._clear_current_mcp_profile_state(
                    mcp_tool._current_mcp_profile_identity()
                )
            finally:
                reset_hermes_home_override(token)

            token = set_hermes_home_override(profile_b)
            try:
                entry = registry.get_entry("mcp__shared__foo_bar")
                assert entry is not None
                routed = json.loads(entry.handler({}))
                assert routed["raw_tool"] == "foo_bar"
                mcp_tool._clear_current_mcp_profile_state(
                    mcp_tool._current_mcp_profile_identity()
                )
                assert registry.get_entry("mcp__shared__foo_bar") is None
            finally:
                reset_hermes_home_override(token)
        finally:
            for profile in (profile_a, profile_b):
                token = set_hermes_home_override(profile)
                try:
                    mcp_tool._clear_current_mcp_profile_state(
                        mcp_tool._current_mcp_profile_identity()
                    )
                finally:
                    reset_hermes_home_override(token)

    @pytest.mark.parametrize("registration_path", ["live", "lazy_cache"])
    def test_profile_handler_refresh_has_no_half_published_window(
        self, tmp_path, monkeypatch, registration_path
    ):
        """Registry entry 发布后，调用方不能看到旧 handler 或空 owner。"""
        import tools.mcp_tool as mcp_tool
        import tools.registry as registry_module
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        profile = tmp_path / "profiles" / "a"
        published = threading.Event()
        release_register = threading.Event()
        dispatch_done = threading.Event()
        schema_done = threading.Event()
        dispatch_waiting = threading.Event()
        schema_waiting = threading.Event()
        pause_register = threading.Event()
        result = {}

        class _ObservedRLock:
            def __init__(self):
                self._lock = threading.RLock()

            def __enter__(self):
                waiting = {
                    "mcp-dispatch": dispatch_waiting,
                    "mcp-schema": schema_waiting,
                }.get(threading.current_thread().name)
                if waiting is not None:
                    waiting.set()
                self._lock.acquire()
                return self

            def __exit__(self, *_exc):
                self._lock.release()

        monkeypatch.setattr(mcp_tool, "_lock", _ObservedRLock())

        def make_handler(_server_name, raw_tool_name, _timeout):
            return lambda _args, **_kwargs: json.dumps({"raw": raw_tool_name})

        def register_tool(raw_name):
            if registration_path == "live":
                server = SimpleNamespace(
                    _tools=[
                        _make_mcp_tool(
                            raw_name,
                            input_schema={
                                "type": "object",
                                "properties": {
                                    raw_name: {"type": "string"}
                                },
                            },
                        )
                    ],
                    session=SimpleNamespace(),
                    tool_timeout=30,
                    initialize_result=None,
                    _is_recycled_stdio=lambda: False,
                )
                mcp_tool._servers["shared"] = server
                return mcp_tool._register_server_tools("shared", server, {})
            return mcp_tool._register_from_cache_sync(
                "shared",
                {},
                {
                    "tools": [
                        {
                            "name": raw_name,
                            "description": "probe",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    raw_name: {"type": "string"}
                                },
                            },
                        }
                    ],
                    "utility_tools": [],
                },
            )

        original_register = registry.register

        def paused_register(*args, **kwargs):
            original_register(*args, **kwargs)
            if pause_register.is_set():
                published.set()
                assert release_register.wait(timeout=2)

        monkeypatch.setattr(registry_module, "registry", registry)
        monkeypatch.setattr(mcp_tool, "_make_tool_handler", make_handler)
        monkeypatch.setattr(registry, "register", paused_register)

        def refresh():
            token = set_hermes_home_override(profile)
            try:
                result["registered"] = register_tool("foo_bar")
            finally:
                reset_hermes_home_override(token)

        def dispatch():
            token = set_hermes_home_override(profile)
            try:
                entry = registry.get_entry("mcp__shared__foo_bar")
                result["dispatch"] = json.loads(entry.handler({}))["raw"]
            finally:
                reset_hermes_home_override(token)
                dispatch_done.set()

        def read_schema():
            token = set_hermes_home_override(profile)
            try:
                definitions = registry.get_definitions(
                    {"mcp__shared__foo_bar"}
                )
                result["schema_properties"] = set(
                    definitions[0]["function"]["parameters"]["properties"]
                )
            finally:
                reset_hermes_home_override(token)
                schema_done.set()

        token = set_hermes_home_override(profile)
        try:
            assert register_tool("foo-bar") == ["mcp__shared__foo_bar"]
        finally:
            reset_hermes_home_override(token)

        pause_register.set()
        refresh_thread = threading.Thread(target=refresh, name="mcp-refresh")
        dispatch_thread = threading.Thread(target=dispatch, name="mcp-dispatch")
        schema_thread = threading.Thread(target=read_schema, name="mcp-schema")
        try:
            refresh_thread.start()
            assert published.wait(timeout=2)
            dispatch_thread.start()
            schema_thread.start()
            assert dispatch_waiting.wait(timeout=2)
            assert schema_waiting.wait(timeout=2)
            assert not dispatch_done.is_set()
            assert not schema_done.is_set()
            release_register.set()
            refresh_thread.join(timeout=2)
            dispatch_thread.join(timeout=2)
            schema_thread.join(timeout=2)
            assert not refresh_thread.is_alive()
            assert not dispatch_thread.is_alive()
            assert not schema_thread.is_alive()
            assert result["registered"] == ["mcp__shared__foo_bar"]
            assert result["dispatch"] == "foo_bar"
            assert result["schema_properties"] == {"foo_bar"}
        finally:
            release_register.set()
            refresh_thread.join(timeout=2)
            dispatch_thread.join(timeout=2)
            schema_thread.join(timeout=2)
            token = set_hermes_home_override(profile)
            try:
                mcp_tool._clear_current_mcp_profile_state(
                    mcp_tool._current_mcp_profile_identity()
                )
            finally:
                reset_hermes_home_override(token)

    def test_retiring_and_generation_fence_reject_late_server_publish(
        self, tmp_path, monkeypatch
    ):
        """retiring 中及 clear 后，旧 generation server 都不得复活 owner。"""
        import tools.mcp_tool as mcp_tool
        import tools.registry as registry_module
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        monkeypatch.setattr(registry_module, "registry", registry)
        profile = tmp_path / "profiles" / "a"
        token = set_hermes_home_override(profile)
        try:
            old_server = mcp_tool.MCPServerTask("shared")
            old_server.session = SimpleNamespace()
            old_server._tools = [_make_mcp_tool("probe")]
            old_server.initialize_result = None

            with mcp_tool._lock:
                mcp_tool._retiring_mcp_profiles.add(
                    old_server.profile_identity
                )
            with pytest.raises(RuntimeError, match="retired profile generation"):
                mcp_tool._register_server_tools("shared", old_server, {})
            assert registry.get_entry("mcp__shared__probe") is None

            with mcp_tool._lock:
                mcp_tool._retiring_mcp_profiles.discard(
                    old_server.profile_identity
                )
            mcp_tool._clear_current_mcp_profile_state(
                old_server.profile_identity
            )
            with pytest.raises(RuntimeError, match="retired profile generation"):
                mcp_tool._register_server_tools("shared", old_server, {})
            assert registry.get_entry("mcp__shared__probe") is None

            new_server = mcp_tool.MCPServerTask("shared")
            new_server.session = SimpleNamespace()
            new_server._tools = [_make_mcp_tool("probe")]
            new_server.initialize_result = None
            assert mcp_tool._register_server_tools(
                "shared", new_server, {}
            ) == ["mcp__shared__probe"]
        finally:
            mcp_tool._clear_current_mcp_profile_state(
                mcp_tool._current_mcp_profile_identity()
            )
            reset_hermes_home_override(token)

    def test_inflight_lazy_cache_registration_cannot_adopt_new_generation(
        self, tmp_path, monkeypatch
    ):
        """旧 discovery 卡在 cache lookup 时，clear 后不得按新代次晚发布。"""
        import tools.mcp_schema_cache as cache_module
        import tools.mcp_tool as mcp_tool
        import tools.registry as registry_module
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.registry import ToolRegistry

        registry = ToolRegistry()
        monkeypatch.setattr(registry_module, "registry", registry)
        monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
        monkeypatch.setattr(cache_module, "config_fingerprint", lambda _cfg: "fp")
        entered = threading.Event()
        release = threading.Event()
        result = {}
        profile = tmp_path / "profiles" / "a"

        def get_cached_entry(_name, _fingerprint):
            entered.set()
            assert release.wait(timeout=2)
            return {
                "tools": [{
                    "name": "probe",
                    "description": "probe",
                    "inputSchema": {"type": "object", "properties": {}},
                }],
                "utility_tools": [],
            }

        monkeypatch.setattr(cache_module, "get_cached_entry", get_cached_entry)

        def register_old_generation():
            token = set_hermes_home_override(profile)
            try:
                result["value"] = mcp_tool.register_mcp_servers({
                    "shared": {"lazy": True, "command": "old-creds"}
                })
            except BaseException as exc:
                result["error"] = exc
            finally:
                reset_hermes_home_override(token)

        thread = threading.Thread(target=register_old_generation)
        thread.start()
        try:
            assert entered.wait(timeout=2)
            token = set_hermes_home_override(profile)
            try:
                profile_identity = mcp_tool._current_mcp_profile_identity()
                with mcp_tool._lock:
                    before = mcp_tool._mcp_profile_generations[profile_identity]
                mcp_tool._clear_current_mcp_profile_state(profile_identity)
                with mcp_tool._lock:
                    assert mcp_tool._mcp_profile_generations[profile_identity] == before + 1
            finally:
                reset_hermes_home_override(token)
            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert isinstance(result.get("error"), RuntimeError)
            assert "retired profile" in str(result["error"])
            assert "value" not in result
            assert registry.get_entry("mcp__shared__probe") is None
            token = set_hermes_home_override(profile)
            try:
                assert "mcp__shared__probe" not in mcp_tool._mcp_tool_handlers
                assert "shared" not in mcp_tool._lazy_server_configs
            finally:
                reset_hermes_home_override(token)
        finally:
            release.set()
            thread.join(timeout=2)
            token = set_hermes_home_override(profile)
            try:
                mcp_tool._clear_current_mcp_profile_state(
                    mcp_tool._current_mcp_profile_identity()
                )
            finally:
                reset_hermes_home_override(token)

    @pytest.mark.parametrize("cleanup_fails", [False, True])
    @pytest.mark.parametrize("repeat_cancel", [False, True], ids=["once", "twice"])
    def test_connect_cancellation_keeps_exact_live_owner_until_cleanup(
        self, monkeypatch, cleanup_fails, repeat_cancel
    ):
        """cancel 必须 await 严格 cleanup；失败时 live owner 留给重试。"""
        import tools.mcp_tool as mcp_tool

        started = None
        run_unwound = None
        shutdown_started = None
        release_shutdown = None
        created = []
        original_shutdown = mcp_tool.MCPServerTask.shutdown

        async def blocked_run(server, _config):
            created.append(server)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                run_unwound.set()

        async def observed_shutdown(server):
            assert run_unwound.is_set()
            shutdown_started.set()
            await release_shutdown.wait()
            if cleanup_fails:
                raise RuntimeError("cleanup failed")
            await original_shutdown(server)

        monkeypatch.setattr(mcp_tool.MCPServerTask, "run", blocked_run)
        monkeypatch.setattr(
            mcp_tool.MCPServerTask, "shutdown", observed_shutdown
        )

        async def exercise():
            nonlocal started, run_unwound, shutdown_started, release_shutdown
            started = asyncio.Event()
            run_unwound = asyncio.Event()
            shutdown_started = asyncio.Event()
            release_shutdown = asyncio.Event()
            task = asyncio.create_task(
                mcp_tool._connect_server("cancelled", {"command": "probe"})
            )
            await started.wait()
            task.cancel()
            await shutdown_started.wait()
            try:
                if repeat_cancel:
                    task.cancel()
                    cancellation_delivered = asyncio.Event()
                    asyncio.get_running_loop().call_soon(
                        cancellation_delivered.set
                    )
                    await cancellation_delivered.wait()
                    assert not task.done()
            finally:
                release_shutdown.set()
            if cleanup_fails:
                with pytest.raises(RuntimeError, match="cancellation cleanup failed"):
                    await task
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task

        try:
            asyncio.run(exercise())
            assert len(created) == 1
            with mcp_tool._lock:
                if cleanup_fails:
                    assert created[0] in mcp_tool._live_mcp_servers
                else:
                    assert created[0] not in mcp_tool._live_mcp_servers
            if cleanup_fails:
                assert mcp_tool._stop_mcp_loop(only_if_idle=True) is False
        finally:
            with mcp_tool._lock:
                for server in created:
                    mcp_tool._live_mcp_servers.discard(server)

    def test_standalone_start_and_cleanup_failure_is_explicit_and_keeps_owner(
        self, monkeypatch, caplog
    ):
        """无 discovery owner 时，二次 cleanup 失败也必须显性且可重试。"""
        import tools.mcp_tool as mcp_tool

        created = []

        class _StartAndCleanupFailingServer(mcp_tool.MCPServerTask):
            def __init__(self, name):
                super().__init__(name)
                created.append(self)

            async def start(self, _config):
                raise ConnectionError("backend start failed")

            async def shutdown(self):
                raise RuntimeError("child still alive")

        monkeypatch.setattr(
            mcp_tool, "MCPServerTask", _StartAndCleanupFailingServer
        )
        try:
            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError, match="start cleanup failed") as exc:
                    asyncio.run(
                        mcp_tool._connect_server(
                            "standalone", {"command": "probe"}
                        )
                    )
            assert isinstance(exc.value.__cause__, RuntimeError)
            assert str(exc.value.__cause__) == "child still alive"
            assert len(created) == 1
            with mcp_tool._lock:
                assert created[0] in mcp_tool._live_mcp_servers
            assert any(
                record.exc_info
                and isinstance(record.exc_info[1], ConnectionError)
                and str(record.exc_info[1]) == "backend start failed"
                for record in caplog.records
            )
        finally:
            with mcp_tool._lock:
                for server in created:
                    mcp_tool._live_mcp_servers.discard(server)

    def test_registration_and_shutdown_failure_keeps_owner_for_profile_retry(
        self, monkeypatch
    ):
        """注册失败后的 cleanup 也失败时，_servers owner 不得先删。"""
        import tools.mcp_tool as mcp_tool

        class _CleanupFailingServer(mcp_tool.MCPServerTask):
            fail_shutdown = True
            shutdown_calls = 0

            async def shutdown(self):
                self.shutdown_calls += 1
                if self.fail_shutdown:
                    raise RuntimeError("child still alive")
                await super().shutdown()

        server = _CleanupFailingServer("shared")
        server.session = object()
        monkeypatch.setattr(
            mcp_tool,
            "_connect_server",
            AsyncMock(return_value=server),
        )
        monkeypatch.setattr(
            mcp_tool,
            "_register_server_tools",
            MagicMock(side_effect=RuntimeError("registration failed")),
        )

        with pytest.raises(RuntimeError, match="child still alive"):
            asyncio.run(mcp_tool._discover_and_register_server("shared", {}))
        assert mcp_tool._servers["shared"] is server

        server.fail_shutdown = False

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(
            mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        monkeypatch.setattr(mcp_tool, "_stop_mcp_loop", lambda **_kwargs: True)
        mcp_tool.shutdown_mcp_profile()
        assert "shared" not in mcp_tool._servers
        assert server.shutdown_calls == 2

    def test_profile_shutdown_waits_for_unpublished_live_owner_single_flight(
        self, tmp_path, monkeypatch
    ):
        """未发布 connect 必须被 unload 等待，第二个 unload 不能拆 retiring 门。"""
        import tools.mcp_tool as mcp_tool
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        cleanup_entered = threading.Event()
        release_cleanup = threading.Event()
        finished = threading.Event()
        errors = []

        class _LiveServer:
            name = "shared"
            profile_identity = ""
            shutdown_calls = 0

            async def shutdown(self):
                self.shutdown_calls += 1
                cleanup_entered.set()
                assert await asyncio.to_thread(release_cleanup.wait, 2)
                with mcp_tool._lock:
                    mcp_tool._live_mcp_servers.discard(self)

        live = _LiveServer()
        token = set_hermes_home_override(profile_a)
        try:
            live.profile_identity = mcp_tool._current_mcp_profile_identity()
        finally:
            reset_hermes_home_override(token)

        def run_inline(coro_or_factory, timeout=30, **_kwargs):
            del timeout
            coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
            return asyncio.run(coro)

        monkeypatch.setattr(
            mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
        )
        monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
        token = set_hermes_home_override(profile_b)
        try:
            mcp_tool._servers["sibling"] = SimpleNamespace(
                name="sibling", session=object()
            )
        finally:
            reset_hermes_home_override(token)
        with mcp_tool._lock:
            mcp_tool._live_mcp_servers.add(live)

        def unload_a():
            token = set_hermes_home_override(profile_a)
            try:
                mcp_tool.shutdown_mcp_profile()
            except BaseException as exc:
                errors.append(exc)
            finally:
                reset_hermes_home_override(token)
                finished.set()

        thread = threading.Thread(target=unload_a)
        thread.start()
        try:
            assert cleanup_entered.wait(timeout=2)
            assert not finished.is_set()
            token = set_hermes_home_override(profile_a)
            try:
                with pytest.raises(RuntimeError, match="already in progress"):
                    mcp_tool.shutdown_mcp_profile()
                with pytest.raises(RuntimeError, match="already in progress"):
                    mcp_tool.shutdown_mcp_servers()
                with mcp_tool._lock:
                    assert live.profile_identity in mcp_tool._retiring_mcp_profiles
                assert live.shutdown_calls == 1
            finally:
                reset_hermes_home_override(token)
            release_cleanup.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert errors == []
            token = set_hermes_home_override(profile_b)
            try:
                assert "sibling" in mcp_tool._servers
            finally:
                reset_hermes_home_override(token)
        finally:
            release_cleanup.set()
            thread.join(timeout=2)
            with mcp_tool._lock:
                mcp_tool._live_mcp_servers.discard(live)
            token = set_hermes_home_override(profile_b)
            try:
                mcp_tool._servers.pop("sibling", None)
            finally:
                reset_hermes_home_override(token)

    def test_skips_servers_already_connecting(self):
        """Servers in _server_connecting must not be spawned again (#58862)."""
        from tools.mcp_tool import (
            register_mcp_servers, _servers, _server_connecting, _ensure_mcp_loop,
        )

        fake_config = {"my_srv": {"command": "npx", "args": ["test"]}}

        # Simulate a prior call that started connecting but hasn't finished
        _server_connecting.add("my_srv")
        connect_calls = []

        async def fake_register(name, cfg, **_kwargs):
            connect_calls.append(name)
            server = _make_mock_server(name)
            server._registered_tool_names = [f"mcp_{name}_tool"]
            _servers[name] = server
            return [f"mcp_{name}_tool"]

        try:
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._discover_and_register_server", side_effect=fake_register), \
                 patch("tools.mcp_tool._existing_tool_names", return_value=[]), \
                 patch("tools.mcp_tool._connect_cooldown_active", return_value=False):
                _ensure_mcp_loop()
                result = register_mcp_servers(fake_config)

            # Should NOT have attempted to connect my_srv again
            assert connect_calls == [], (
                f"Server already in _server_connecting should be skipped, "
                f"but connect was called for: {connect_calls}"
            )
            assert result == []
        finally:
            _server_connecting.discard("my_srv")
            _servers.pop("my_srv", None)

    def test_clears_stale_connecting_on_timeout(self):
        """Stale entries in _server_connecting are cleaned up after timeout (#58862)."""
        from tools.mcp_tool import (
            register_mcp_servers, _servers, _server_connecting,
            _server_connect_errors, _ensure_mcp_loop,
        )

        fake_config = {
            "srv_a": {"command": "npx", "args": ["a"]},
            "srv_b": {"command": "npx", "args": ["b"]},
        }

        # Simulate that srv_a is already connecting from another call
        _server_connecting.add("srv_a")

        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._run_on_mcp_loop", side_effect=TimeoutError("timed out")), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]), \
             patch("tools.mcp_tool._connect_cooldown_active", return_value=False):
            _ensure_mcp_loop()

            with pytest.raises(TimeoutError):
                register_mcp_servers(fake_config)

        # After timeout, srv_b (which was in new_servers and added to _server_connecting)
        # should have been cleaned up from _server_connecting.
        # srv_a should remain since it was added externally and not part of new_servers.
        assert "srv_b" not in _server_connecting, (
            "Stale server added during this call should have been removed from "
            "_server_connecting after timeout"
        )
        # Cleanup
        _server_connecting.discard("srv_a")
        _servers.pop("srv_a", None)
        _servers.pop("srv_b", None)

# ---------------------------------------------------------------------------
# Tests for parallel tool call support (port from openai/codex#17667)
# ---------------------------------------------------------------------------

class TestMcpParallelToolCalls:
    """Tests for the supports_parallel_tool_calls config option."""

    def test_is_mcp_tool_parallel_safe_with_flag(self):
        """MCP tool from a parallel-safe server returns True."""
        from tools.mcp_tool import (
            is_mcp_tool_parallel_safe, _mcp_tool_server_names,
            _parallel_safe_servers, _lock,
        )
        with _lock:
            _parallel_safe_servers.add("docs")
            _mcp_tool_server_names["mcp__docs__search"] = "docs"
            _mcp_tool_server_names["mcp__docs__read_file"] = "docs"
            _mcp_tool_server_names["mcp__github__list_repos"] = "github"
        try:
            assert is_mcp_tool_parallel_safe("mcp__docs__search") is True
            assert is_mcp_tool_parallel_safe("mcp__docs__read_file") is True
            # Different server should be False
            assert is_mcp_tool_parallel_safe("mcp__github__list_repos") is False
        finally:
            with _lock:
                _parallel_safe_servers.discard("docs")
                _mcp_tool_server_names.pop("mcp__docs__search", None)
                _mcp_tool_server_names.pop("mcp__docs__read_file", None)
                _mcp_tool_server_names.pop("mcp__github__list_repos", None)


    def test_register_mcp_servers_tracks_parallel_flag(self):
        """register_mcp_servers populates _parallel_safe_servers from config."""
        from tools.mcp_tool import (
            register_mcp_servers, _parallel_safe_servers, _lock,
            sanitize_mcp_name_component,
        )
        fake_config = {
            "parallel_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": True,
            },
            "serial_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": False,
            },
            "default_srv": {
                "command": "echo",
                # no supports_parallel_tool_calls key
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(fake_config)

        with _lock:
            assert sanitize_mcp_name_component("parallel_srv") in _parallel_safe_servers
            assert sanitize_mcp_name_component("serial_srv") not in _parallel_safe_servers
            assert sanitize_mcp_name_component("default_srv") not in _parallel_safe_servers
            # Cleanup
            _parallel_safe_servers.discard(sanitize_mcp_name_component("parallel_srv"))

    def test_register_mcp_servers_removes_parallel_flag_on_toggle(self):
        """Toggling supports_parallel_tool_calls to false removes server from the set."""
        from tools.mcp_tool import (
            register_mcp_servers, _parallel_safe_servers, _lock,
            sanitize_mcp_name_component,
        )
        config_on = {
            "toggle_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": True,
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(config_on)
        with _lock:
            assert sanitize_mcp_name_component("toggle_srv") in _parallel_safe_servers

        config_off = {
            "toggle_srv": {
                "command": "echo",
                "supports_parallel_tool_calls": False,
            },
        }
        with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._ensure_mcp_loop"), \
             patch("tools.mcp_tool._run_on_mcp_loop"), \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            register_mcp_servers(config_off)
        with _lock:
            assert sanitize_mcp_name_component("toggle_srv") not in _parallel_safe_servers

# ---------------------------------------------------------------------------
# Cross-process MCP discovery lock (issue #62771)
# ---------------------------------------------------------------------------


class TestMCPDiscoveryCrossProcessLock:
    """Tests for the cross-process MCP discovery guard in discover_mcp_tools()."""

    @staticmethod
    def _lock_exclusive(fh):
        """Lock a file handle exclusively, cross-platform.

        Mirrors production _try_acquire_mcp_discovery_lock: fcntl on POSIX,
        portalocker on Windows (portalocker only ships on win32 installs).
        """
        if sys.platform == "win32":
            import portalocker

            self._lock_exclusive(fh)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @pytest.fixture(autouse=True)
    def _fast_retries(self):
        """Override retry constants so tests are fast."""
        import tools.mcp_tool as mcp_tool
        orig_max = mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES
        orig_delay = mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S
        mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES = 3
        mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S = 0.01
        yield
        mcp_tool._MCP_DISCOVERY_LOCK_MAX_RETRIES = orig_max
        mcp_tool._MCP_DISCOVERY_LOCK_RETRY_DELAY_S = orig_delay

    def test_lock_acquired_path(self, tmp_path):
        """Lock acquired -> discovery runs normally, lock released at end."""
        from tools.mcp_tool import (
            _LockCookie,
            discover_mcp_tools,
        )

        lock_file = tmp_path / ".mcp-discovery.lock"
        fh = open(lock_file, "w", encoding="utf-8")
        cookie = _LockCookie(fh)

        def mock_acquire(profile_identity=None):
            # 锁按 profile 派生后,调用点会把已捕获的 identity 传下来。
            return cookie

        mock_config = {"test_srv": {"command": "echo", "enabled": True}}
        with patch.object(cookie, "release", wraps=cookie.release) as release_spy:
            with patch("tools.mcp_tool._try_acquire_mcp_discovery_lock", mock_acquire), \
                 patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._load_mcp_config", return_value=mock_config), \
                 patch("tools.mcp_tool.register_mcp_servers", return_value=["mcp__test_srv__ping"]) as reg_spy:
                result = discover_mcp_tools()
            assert result == ["mcp__test_srv__ping"]
            release_spy.assert_called_once()

    def test_lock_held_retries_exhausted_fallback(self):
        """All retry attempts see lock held -> runs discovery unguarded."""
        import tools.mcp_tool as mcp_tool
        from tools.mcp_tool import (
            _LOCK_UNAVAILABLE,
            discover_mcp_tools,
            _MCP_DISCOVERY_LOCK_MAX_RETRIES,
        )

        mock_config = {"test_srv": {"command": "echo", "enabled": True}}
        # Every attempt returns None (lock held)
        with patch("tools.mcp_tool._try_acquire_mcp_discovery_lock", return_value=None), \
             patch("tools.mcp_tool._MCP_AVAILABLE", True), \
             patch("tools.mcp_tool._load_mcp_config", return_value=mock_config), \
             patch("tools.mcp_tool.register_mcp_servers") as reg_spy, \
             patch("tools.mcp_tool._existing_tool_names", return_value=[]):
            result = discover_mcp_tools()
        # Must still run local discovery
        reg_spy.assert_called_once_with(
            mock_config,
            profile_identity=mcp_tool._current_mcp_profile_identity(),
            profile_generation=0,
        )

    def test_posix_flock_acquire_and_release(self):
        """_acquire_lock_on_fh uses fcntl.flock on POSIX."""
        import sys
        import tempfile
        from unittest.mock import MagicMock

        mock_fcntl = MagicMock()
        mock_fcntl.LOCK_EX = 2
        mock_fcntl.LOCK_NB = 4

        with tempfile.NamedTemporaryFile(prefix="mcp-lock-", suffix=".tmp", delete=False) as tf:
            lock_path = tf.name

        try:
            fh = open(lock_path, "w", encoding="utf-8")
            with patch.dict("sys.modules", {"fcntl": mock_fcntl}), \
                 patch("tools.mcp_tool.os.name", "posix"):
                from tools.mcp_tool import _acquire_lock_on_fh
                result = _acquire_lock_on_fh(fh)
            assert result is True
            mock_fcntl.flock.assert_called_once_with(
                fh.fileno(), mock_fcntl.LOCK_EX | mock_fcntl.LOCK_NB
            )
            fh.close()
        finally:
            try:
                os.unlink(lock_path)
            except Exception:
                pass


class TestBuildSafeEnvHomeContract:
    """``_build_safe_env`` must route the stdio-subprocess env through the
    shared subprocess HOME contract (hermes_constants.apply_subprocess_home_env).

    ``_build_safe_env`` is a *whitelist* filter, so two things have to hold for
    the contract to work through it:
      * ``HERMES_HOME`` must survive the filter (the contract needs it to locate
        ``{HERMES_HOME}/home`` and the child should carry it onward);
      * ``HERMES_HOME_FALLBACK`` (the nested-chain anti-flip marker) must pass
        through so a grandchild spawned by an MCP server inside a nested hermes
        chain keeps its fallback-injected profile HOME instead of "repairing" it
        back to a pwd-guessed real home (mirrors execute_code's allowlist).
    """

    def _host_mode(self, monkeypatch):
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)

    def test_missing_home_falls_back_to_profile_home(self, tmp_path, monkeypatch):
        """systemd/cron host: no HOME anywhere → inject ``{HERMES_HOME}/home``
        so an MCP stdio server's ``~``-addressed credentials resolve."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = _build_safe_env(None)
        assert env.get("HOME") == str(profile_home)

    def test_real_home_is_preserved(self, tmp_path, monkeypatch):
        """Host with a real HOME → auto mode keeps it untouched."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = _build_safe_env(None)
        assert env.get("HOME") == str(real_home)

    def test_hermes_home_is_whitelisted(self, tmp_path, monkeypatch):
        """HERMES_HOME must survive the whitelist filter so the contract can
        resolve the profile home and the child carries it forward."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(tmp_path / "real-home"))
        (tmp_path / "real-home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = _build_safe_env(None)
        assert env.get("HERMES_HOME") == str(hermes_home)

    def test_fallback_marker_passes_through_whitelist(self, tmp_path, monkeypatch):
        """A HERMES_HOME_FALLBACK marker riding in os.environ (nested hermes
        chain) must not be filtered out — dropping it would let the child's
        apply_subprocess_home_env() re-hijack a fallback-injected HOME."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        # Two profiles: parent injected A/home and marked it; this hop is B.
        base = tmp_path / ".hermes" / "profiles"
        a = base / "alpha"
        b = base / "beta"
        (a / "home").mkdir(parents=True)
        (b / "home").mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HOME", str(a / "home"))
        monkeypatch.setenv("HERMES_HOME", str(b))
        monkeypatch.setenv("HERMES_HOME_FALLBACK", str(a / "home"))

        env = _build_safe_env(None)
        # The marker survived the whitelist (present in the built env)...
        assert "HERMES_HOME_FALLBACK" in env
        # ...and the contract re-pointed HOME at B's own profile home, updating
        # the marker to match (cross-profile leak closed).
        assert env.get("HOME") == str(b / "home")
        assert env.get("HERMES_HOME_FALLBACK") == str(b / "home")

    def test_terminal_home_mode_passes_through_whitelist(self, tmp_path, monkeypatch):
        """TERMINAL_HOME_MODE (the profile/real/auto mode string) must survive
        the whitelist. If it is dropped, a nested hermes launched by the MCP
        server re-enters ``auto`` and repairs HOME back to the real dir on the
        second hop, breaking profile isolation the parent explicitly pinned."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")

        env = _build_safe_env(None)
        # The mode string is transmitted so the second hop honors it too...
        assert env.get("TERMINAL_HOME_MODE") == "profile"
        # ...and profile mode pinned HOME at the profile home this hop.
        assert env.get("HOME") == str(profile_home)

    def test_real_home_var_not_leaked_to_child(self, tmp_path, monkeypatch):
        """apply_subprocess_home_env unconditionally writes HERMES_REAL_HOME
        (the OS-account home, e.g. /Users/alice) into the env. That username-
        bearing path must NOT reach a third-party MCP server: the whitelist's
        whole purpose is to keep host env off untrusted subprocesses, and
        HERMES_REAL_HOME is not needed by the child (HOME is already resolved).
        """
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = _build_safe_env(None)
        assert "HERMES_REAL_HOME" not in env

    def test_real_home_pop_does_not_break_missing_home_fallback(self, tmp_path, monkeypatch):
        """Popping HERMES_REAL_HOME must not disturb the HOME fallback: HOME is
        already set by apply, REAL_HOME is only a repair-branch reference the
        MCP child never uses."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        env = _build_safe_env(None)
        assert env.get("HOME") == str(profile_home)
        assert "HERMES_REAL_HOME" not in env
        # Fallback marker still recorded so nested hops don't re-hijack.
        assert env.get("HERMES_HOME_FALLBACK") == str(profile_home)

    def test_same_profile_nested_marker_passes_through(self, tmp_path, monkeypatch):
        """same-profile nested MCP chain: a marker that already equals this
        hop's HOME/profile home must survive _build_safe_env so a grandchild
        MCP server does not lose it and re-break ZET-1938 on the second hop."""
        from tools.mcp_tool import _build_safe_env

        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        # Parent already injected the profile home and marked it (same profile).
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_HOME_FALLBACK", str(profile_home))

        env = _build_safe_env(None)
        # same-profile keep: HOME and the marker both stay at the profile home.
        assert env.get("HOME") == str(profile_home)
        assert env.get("HERMES_HOME_FALLBACK") == str(profile_home)
        assert "HERMES_REAL_HOME" not in env

    def test_contextvar_override_bridges_hermes_home(self, tmp_path, monkeypatch):
        """A per-request set_hermes_home_override(A) while the process env still
        carries HERMES_HOME=B must reach the child consistently: the contract
        resolves HOME from override A, so the child's HERMES_HOME must be A too,
        not the stale process-global B (else a nested hermes-as-MCP flips it)."""
        from tools.mcp_tool import _build_safe_env
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        self._host_mode(monkeypatch)
        a = tmp_path / "profileA" / ".hermes"
        b = tmp_path / "profileB" / ".hermes"
        (a / "home").mkdir(parents=True)
        (b / "home").mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(b))  # stale process-global

        token = set_hermes_home_override(str(a))
        try:
            env = _build_safe_env(None)
        finally:
            reset_hermes_home_override(token)

        assert env.get("HERMES_HOME") == str(a)
        assert env.get("HOME") == str(a / "home")


def test_single_server_reload_keeps_owner_when_loop_is_unavailable(monkeypatch):
    import tools.mcp_tool as mcp_tool

    original = SimpleNamespace(shutdown=AsyncMock())
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_servers", {"demo": original})
    monkeypatch.setattr(mcp_tool, "_mcp_loop", None)
    monkeypatch.setattr(
        mcp_tool,
        "discover_mcp_tools",
        lambda: pytest.fail("reload must not discover after failed shutdown"),
    )

    with pytest.raises(RuntimeError, match="loop unavailable"):
        mcp_tool.reload_single_mcp_server("demo")

    assert mcp_tool._servers["demo"] is original


def test_single_server_reload_blocks_profile_shutdown_until_cleanup_finishes(
    monkeypatch
):
    """单 server cleanup in-flight 时，profile shutdown 必须显式拒绝双清。"""
    import tools.mcp_tool as mcp_tool

    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    reload_finished = threading.Event()
    errors = []

    class _Server:
        name = "demo"

        def __init__(self):
            self.shutdown_calls = 0

        async def shutdown(self):
            self.shutdown_calls += 1
            cleanup_entered.set()
            assert await asyncio.to_thread(release_cleanup.wait, 2)

    server = _Server()

    def run_inline(coro_or_factory, timeout=30, **_kwargs):
        del timeout
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro)

    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(
        mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
    )
    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", run_inline)
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", lambda **_kwargs: [])
    mcp_tool._servers[server.name] = server

    def reload_server():
        try:
            mcp_tool.reload_single_mcp_server(server.name)
        except BaseException as exc:
            errors.append(exc)
        finally:
            reload_finished.set()

    thread = threading.Thread(target=reload_server, daemon=True)
    thread.start()
    try:
        assert cleanup_entered.wait(timeout=2)
        with pytest.raises(RuntimeError, match="operation already in progress"):
            mcp_tool.shutdown_mcp_profile()
        assert not reload_finished.is_set()

        release_cleanup.set()
        assert reload_finished.wait(timeout=2)
        assert errors == []
        assert server.shutdown_calls == 1
        assert server.name not in mcp_tool._servers
    finally:
        release_cleanup.set()
        thread.join(timeout=2)
        mcp_tool._servers.pop(server.name, None)
        with mcp_tool._lock:
            mcp_tool._retiring_mcp_server_operations.clear()
            mcp_tool._retiring_mcp_profiles.clear()


def test_single_server_reload_uses_identity_cas_before_removal(monkeypatch):
    import tools.mcp_tool as mcp_tool

    original = SimpleNamespace(shutdown=AsyncMock())
    replacement = SimpleNamespace(shutdown=AsyncMock())
    servers = {"demo": original}
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_servers", servers)
    monkeypatch.setattr(
        mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
    )

    def replace_during_shutdown(_factory, timeout, **_kwargs):
        assert timeout == 20
        servers["demo"] = replacement

    monkeypatch.setattr(mcp_tool, "_run_on_mcp_loop", replace_during_shutdown)
    monkeypatch.setattr(
        mcp_tool,
        "discover_mcp_tools",
        lambda **_kwargs: pytest.fail("reload must not discover after losing ownership"),
    )

    with pytest.raises(RuntimeError, match="lost ownership"):
        mcp_tool.reload_single_mcp_server("demo")

    assert servers["demo"] is replacement


def test_single_server_reload_keeps_owner_when_shutdown_fails(monkeypatch):
    import tools.mcp_tool as mcp_tool

    original = SimpleNamespace(shutdown=AsyncMock())
    servers = {"demo": original}
    monkeypatch.setattr(mcp_tool, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_servers", servers)
    monkeypatch.setattr(
        mcp_tool, "_mcp_loop", SimpleNamespace(is_running=lambda: True)
    )
    monkeypatch.setattr(
        mcp_tool,
        "_run_on_mcp_loop",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("stuck")),
    )
    monkeypatch.setattr(
        mcp_tool,
        "discover_mcp_tools",
        lambda **_kwargs: pytest.fail("reload must not discover after failed shutdown"),
    )

    with pytest.raises(TimeoutError, match="stuck"):
        mcp_tool.reload_single_mcp_server("demo")

    assert servers["demo"] is original


# 下面六条走本文件既有的 `async def _test()` + `asyncio.run(_test())` 形态
# （同文件已有 29 处先例），而不是 @pytest.mark.asyncio：
# home-guard-tests.yml 这道 CI 门刻意不装 pytest-asyncio（它的注释里写明
# 「the rest of that dir uses pytest-asyncio … extra deps」，为此只挑了 LSP
# 目录里唯一那个同步文件），而 tests/tools/test_mcp_tool.py 正在它的文件清单里。
# 用 pytest.mark.asyncio 会让这道门整体红在 "async def functions are not
# natively supported"，而不是红在被测契约上。
def test_server_shutdown_concurrent_callers_share_one_success():
    """两个并发 cleanup 必须共享同一轮成功，底层资源只能提交一次。"""
    import tools.mcp_tool as mcp_tool

    async def _test():
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()

        class _Server(mcp_tool.MCPServerTask):
            def __init__(self):
                super().__init__("demo")
                self.cleanup_calls = 0

            async def _shutdown_owned(self):
                self.cleanup_calls += 1
                cleanup_entered.set()
                await release_cleanup.wait()

        server = _Server()
        first = asyncio.create_task(server.shutdown())
        await cleanup_entered.wait()
        second = asyncio.create_task(server.shutdown())
        await asyncio.sleep(0)
        assert not second.done()

        release_cleanup.set()
        await asyncio.gather(first, second)
        assert server.cleanup_calls == 1

    asyncio.run(_test())


def test_server_shutdown_failure_does_not_latch_success():
    """第一轮失败不能置成功 latch，下一轮必须真正重试。"""
    import tools.mcp_tool as mcp_tool

    async def _test():
        class _Server(mcp_tool.MCPServerTask):
            def __init__(self):
                super().__init__("demo")
                self.cleanup_calls = 0

            async def _shutdown_owned(self):
                self.cleanup_calls += 1
                if self.cleanup_calls == 1:
                    raise RuntimeError("still alive")

        server = _Server()
        with pytest.raises(RuntimeError, match="still alive"):
            await server.shutdown()
        await server.shutdown()
        assert server.cleanup_calls == 2

    asyncio.run(_test())


def test_server_shutdown_keeps_registry_when_child_survives(monkeypatch):
    import tools.mcp_tool as mcp_tool

    server = mcp_tool.MCPServerTask("demo")
    deregister = MagicMock()
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", deregister)

    def keep_child(_include_active, server_name, owner_identity):
        assert server_name == "demo"
        assert owner_identity == (server.profile_identity, "demo", server)
        return [4242]

    monkeypatch.setattr(
        mcp_tool,
        "_kill_orphaned_mcp_children",
        keep_child,
    )

    async def _test():
        with pytest.raises(RuntimeError, match="still owns child processes"):
            await server.shutdown()

    asyncio.run(_test())

    deregister.assert_not_called()


def test_server_task_cannot_start_transport_after_process_teardown(monkeypatch):
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    server = mcp_tool.MCPServerTask("demo")
    run_stdio = AsyncMock(return_value="shutdown")
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", run_stdio)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_teardown_started", True)

    async def _test():
        await server.run({"command": "demo"})

    asyncio.run(_test())

    run_stdio.assert_not_awaited()
    assert server._shutdown_event.is_set()
    assert server._ready.is_set()


def test_preflight_does_not_construct_http_client_after_shutdown(monkeypatch):
    import httpx
    import tools.mcp_tool as mcp_tool

    server = mcp_tool.MCPServerTask("demo")
    server._shutdown_event.set()
    construct = MagicMock(side_effect=AssertionError("client constructed"))
    monkeypatch.setattr(httpx, "AsyncClient", construct)

    async def _test():
        await server._preflight_content_type("https://example.invalid/mcp")

    asyncio.run(_test())

    construct.assert_not_called()


def test_streamable_http_does_not_construct_client_after_shutdown(monkeypatch):
    import httpx
    import tools.mcp_tool as mcp_tool

    server = mcp_tool.MCPServerTask("demo")
    server._shutdown_event.set()
    construct = MagicMock(side_effect=AssertionError("client constructed"))
    monkeypatch.setattr(httpx, "AsyncClient", construct)
    monkeypatch.setattr(mcp_tool, "_MCP_HTTP_AVAILABLE", True)
    monkeypatch.setattr(mcp_tool, "_MCP_NEW_HTTP", True)

    async def _test():
        return await server._run_http({"url": "https://example.invalid/mcp"})

    result = asyncio.run(_test())

    assert result == "shutdown"
    construct.assert_not_called()
