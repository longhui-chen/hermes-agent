"""Tests for MCP stability fixes — event loop handler, PID tracking, shutdown robustness."""

import asyncio
import os
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1: MCP event loop exception handler
# ---------------------------------------------------------------------------

class TestMCPLoopExceptionHandler:
    """_mcp_loop_exception_handler suppresses benign 'Event loop is closed'."""

    def test_suppresses_event_loop_closed(self):
        from tools.mcp_tool import _mcp_loop_exception_handler
        loop = MagicMock()
        context = {"exception": RuntimeError("Event loop is closed")}
        # Should NOT call default handler
        _mcp_loop_exception_handler(loop, context)
        loop.default_exception_handler.assert_not_called()


    def test_probe_cleanup_does_not_stop_loop_with_registered_servers(self):
        """Probe cleanup must not kill the shared loop used by live MCP tools."""
        import tools.mcp_tool as mcp_mod

        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        try:
            mcp_mod._ensure_mcp_loop()
            with mcp_mod._lock:
                loop = mcp_mod._mcp_loop
                mcp_mod._servers["live"] = MagicMock(session=object())

            assert mcp_mod._stop_mcp_loop_if_idle() is False

            with mcp_mod._lock:
                assert mcp_mod._mcp_loop is loop
                assert mcp_mod._mcp_thread is not None
            assert loop is not None
            assert loop.is_running()
        finally:
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
            mcp_mod._stop_mcp_loop()


# ---------------------------------------------------------------------------
# Fix 2: stdio PID tracking
# ---------------------------------------------------------------------------

class TestStdioPidTracking:
    """_snapshot_child_pids and _stdio_pids track subprocess PIDs."""

    def test_snapshot_returns_set(self):
        from tools.mcp_tool import _snapshot_child_pids
        result = _snapshot_child_pids()
        assert isinstance(result, set)
        # All elements should be ints
        for pid in result:
            assert isinstance(pid, int)


    def test_kill_orphaned_handles_dead_pids(self):
        """_kill_orphaned_mcp_children gracefully handles already-dead PIDs."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pids,
            _lock,
        )

        # Use a PID that definitely doesn't exist
        fake_pid = 999999999
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"
            _orphan_stdio_pid_start_times[fake_pid] = 100

        # Should not raise (ProcessLookupError is caught)
        with patch("tools.mcp_tool.time.sleep"), patch(
            "tools.mcp_tool.os.kill", side_effect=ProcessLookupError
        ):
            _kill_orphaned_mcp_children()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_kill_orphaned_keeps_legacy_owner_when_generation_unknown(
        self, monkeypatch
    ):
        """legacy owner 缺 start-time 时必须保留且不得向未知 PID 发信号。"""
        import tools.mcp_tool as mcp_tool
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pids,
            _lock,
        )

        fake_pid = 424242
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = "orphan"

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        # 进程仍存活但缺少已记录的 start-time，不能判断 PID 是否已复用。
        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("tools.mcp_tool.time.sleep") as mock_sleep:
            assert _kill_orphaned_mcp_children() == [fake_pid]
        mock_kill.assert_not_called()
        mock_sleep.assert_not_called()

        with _lock:
            assert fake_pid in _orphan_stdio_pids
            assert not any(
                key[1] == fake_pid for key in mcp_tool._retiring_stdio_owners
            )

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("tools.mcp_tool.os.kill"), \
             patch("tools.mcp_tool.time.sleep"):
            assert _kill_orphaned_mcp_children() == [fake_pid]
        with _lock:
            assert fake_pid in _orphan_stdio_pids
            _orphan_stdio_pids.discard(fake_pid)
            _orphan_stdio_pid_servers.pop(fake_pid, None)


    def test_run_stdio_reaps_orphans_before_spawn(self):
        """_run_stdio kills orphaned PIDs from prior failed attempts (#57355)."""
        import tools.mcp_tool as mcp_tool
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pids,
            _stdio_pids,
            _stdio_pgids,
            _lock,
            MCPServerTask,
        )
        from unittest.mock import patch, MagicMock, AsyncMock

        fake_pid = 999999997

        server = MCPServerTask.__new__(MCPServerTask)
        server.name = "test-zombie-reap"
        server.profile_identity = mcp_tool._current_mcp_profile_identity()
        server._ready = MagicMock()
        server._shutdown_event = MagicMock()
        server._shutdown_event.is_set.return_value = True
        server._reconnect_event = MagicMock()
        server._sampling = None
        server._elicitation = None
        server._registered_tool_names = []
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_servers[fake_pid] = (
                server.profile_identity,
                server.name,
                server,
            )
            _orphan_stdio_pid_start_times[fake_pid] = 100

        config = {"command": "echo", "args": ["hello"]}

        import asyncio

        async def _run():
            # _run_stdio should reap orphans before it gets to the
            # stdio_client spawn.  Patch the OSV check (local import)
            # and stdio_client so no real subprocess is spawned.
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._build_safe_env", return_value={}), \
                 patch("tools.mcp_tool._resolve_stdio_command",
                       return_value=("echo", {})), \
                 patch("tools.mcp_tool._write_stderr_log_header"), \
                 patch("tools.mcp_tool._get_mcp_stderr_log",
                       return_value=None), \
                 patch("tools.mcp_tool.check_package_for_malware",
                       return_value=None, create=True), \
                 patch("tools.osv_check.check_package_for_malware",
                       return_value=None):
                # Patch stdio_client to raise so the test exits quickly
                cm = MagicMock()
                cm.__aenter__ = AsyncMock(side_effect=RuntimeError("test"))
                cm.__aexit__ = AsyncMock(return_value=False)
                with patch("tools.mcp_tool.stdio_client", return_value=cm), \
                     patch("tools.mcp_tool.time.sleep"):
                    try:
                        await server._run_stdio(config)
                    except Exception:
                        pass

        asyncio.run(_run())

        # The orphan must have been reaped before the spawn attempt.
        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    def test_run_stdio_does_not_spawn_while_exact_orphan_lingers(self):
        """同一 owner 的旧进程杀不掉时，本轮重连必须显性失败。"""
        import tools.mcp_tool as mcp_tool
        from tools.mcp_tool import MCPServerTask

        server = MCPServerTask.__new__(MCPServerTask)
        server.name = "test-lingering-owner"
        server.profile_identity = mcp_tool._current_mcp_profile_identity()
        server._sampling = None
        server._elicitation = None

        config = {"command": "echo", "args": ["hello"]}

        async def _run():
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._build_safe_env", return_value={}), \
                 patch("tools.mcp_tool._resolve_stdio_command",
                       return_value=("echo", {})), \
                 patch("tools.mcp_tool.check_package_for_malware",
                       return_value=None, create=True), \
                 patch("tools.osv_check.check_package_for_malware",
                       return_value=None), \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children",
                       return_value=[4242]) as reap, \
                 patch("tools.mcp_tool.stdio_client") as stdio:
                with pytest.raises(RuntimeError, match="still owns child processes"):
                    await server._run_stdio(config)
                stdio.assert_not_called()
                reap.assert_called_once_with(
                    False,
                    server.name,
                    (server.profile_identity, server.name, server),
                )

        asyncio.run(_run())

    def test_posix_run_stdio_fails_when_spawn_identity_is_unverified(self):
        """POSIX 也必须 fail-closed，不能建立无 PID owner 的 session。"""
        import tools.mcp_tool as mcp_tool

        async def _run():
            server = mcp_tool.MCPServerTask("unverified")
            transport = MagicMock()
            transport.__aenter__ = AsyncMock(return_value=(object(), object()))
            transport.__aexit__ = AsyncMock(return_value=False)
            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._build_safe_env", return_value={}), \
                 patch("tools.mcp_tool._resolve_stdio_command",
                       return_value=("echo", {})), \
                 patch("tools.osv_check.check_package_for_malware",
                       return_value=None), \
                 patch("tools.mcp_tool._wrap_command_with_watchdog",
                       side_effect=lambda command, args: (command, args)), \
                 patch("tools.mcp_tool._write_stderr_log_header"), \
                 patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None), \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children",
                       return_value=[]), \
                 patch("tools.mcp_tool._observe_child_pids",
                       side_effect=[(True, set()), (True, {4242})]), \
                 patch("tools.mcp_tool._filter_mcp_children", return_value=set()), \
                 patch("tools.mcp_tool.stdio_client", return_value=transport), \
                 patch("tools.mcp_tool.ClientSession") as session, \
                 patch.object(mcp_tool.os, "name", "posix"):
                with pytest.raises(
                    RuntimeError, match="ownership could not be verified"
                ):
                    await server._run_stdio({"command": "echo"})
            transport.__aexit__.assert_awaited_once()
            session.assert_not_called()

        asyncio.run(_run())

    def test_concurrent_run_stdio_serializes_spawn_owner_capture(self, monkeypatch):
        """并发 profile 只能在各自 spawn 窗口登记各自 PID owner。"""
        import tools.mcp_tool as mcp_tool

        live_pids = set()
        next_pid = iter((51001, 51002))
        spawn_state = {"active": 0, "max_active": 0}

        class _ObservedLock:
            def __init__(self):
                self._lock = asyncio.Lock()
                self.second_waiting = asyncio.Event()

            async def __aenter__(self):
                if self._lock.locked():
                    self.second_waiting.set()
                await self._lock.acquire()

            async def __aexit__(self, *_exc):
                self._lock.release()

        observed_lock = _ObservedLock()
        first_spawn_entered = asyncio.Event()
        release_first_spawn = asyncio.Event()

        class _Transport:
            async def __aenter__(self):
                self.pid = next(next_pid)
                spawn_state["active"] += 1
                spawn_state["max_active"] = max(
                    spawn_state["max_active"], spawn_state["active"]
                )
                if self.pid == 51001:
                    first_spawn_entered.set()
                    await release_first_spawn.wait()
                live_pids.add(self.pid)
                spawn_state["active"] -= 1
                return object(), object()

            async def __aexit__(self, *_exc):
                live_pids.discard(self.pid)

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def initialize(self):
                return SimpleNamespace()

        async def _run():
            release = asyncio.Event()
            servers = [
                mcp_tool.MCPServerTask("shared"),
                mcp_tool.MCPServerTask("shared"),
            ]
            servers[0].profile_identity = "/profiles/a"
            servers[1].profile_identity = "/profiles/b"

            async def wait_for_release():
                await release.wait()
                return "shutdown"

            async def discover_tools(_server):
                return None

            async def wait_for_lifecycle(_server):
                return await wait_for_release()

            with patch("tools.mcp_tool._MCP_AVAILABLE", True), \
                 patch("tools.mcp_tool._build_safe_env", return_value={}), \
                 patch("tools.mcp_tool._resolve_stdio_command",
                       return_value=("echo", {})), \
                 patch("tools.mcp_tool.check_package_for_malware",
                       return_value=None, create=True), \
                 patch("tools.osv_check.check_package_for_malware",
                       return_value=None), \
                 patch("tools.mcp_tool._wrap_command_with_watchdog",
                       side_effect=lambda command, args: (command, args)), \
                 patch("tools.mcp_tool._write_stderr_log_header"), \
                 patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None), \
                 patch("tools.mcp_tool._kill_orphaned_mcp_children",
                       return_value=[]), \
                 patch("tools.mcp_tool._get_stdio_spawn_lock",
                       return_value=observed_lock), \
                 patch("tools.mcp_tool._observe_child_pids",
                       side_effect=lambda: (True, set(live_pids))), \
                 patch("tools.mcp_tool._filter_mcp_children",
                       side_effect=lambda pids, **_kwargs: pids), \
                 patch("tools.mcp_tool.stdio_client",
                       side_effect=lambda *_args, **_kwargs: _Transport()), \
                 patch("tools.mcp_tool.ClientSession",
                       side_effect=lambda *_args, **_kwargs: _Session()), \
                 patch("tools.mcp_tool.os.getpgid", side_effect=lambda pid: pid), \
                 patch("gateway.status.get_process_start_time",
                       side_effect=lambda pid: pid * 10), \
                 patch("gateway.status._pid_exists", return_value=False), \
                 patch.object(mcp_tool.MCPServerTask, "_discover_tools",
                              discover_tools), \
                 patch.object(mcp_tool.MCPServerTask,
                              "_wait_for_lifecycle_event", wait_for_lifecycle):
                tasks = [asyncio.create_task(
                    servers[0]._run_stdio({"command": "echo"})
                )]
                await asyncio.wait_for(first_spawn_entered.wait(), timeout=1)
                tasks.append(asyncio.create_task(
                    servers[1]._run_stdio({"command": "echo"})
                ))
                await asyncio.wait_for(observed_lock.second_waiting.wait(), timeout=1)
                assert spawn_state == {"active": 1, "max_active": 1}
                release_first_spawn.set()
                await asyncio.gather(*(server._ready.wait() for server in servers))
                with mcp_tool._lock:
                    assert set(mcp_tool._stdio_pids) == {51001, 51002}
                    assert {
                        pid: mcp_tool._stdio_pids[pid]
                        for pid in (51001, 51002)
                    } == {
                        51001: ("/profiles/a", "shared", servers[0]),
                        51002: ("/profiles/b", "shared", servers[1]),
                    }
                    assert {
                        pid: mcp_tool._stdio_pid_start_times[pid]
                        for pid in (51001, 51002)
                    } == {
                        51001: 510010,
                        51002: 510020,
                    }
                assert spawn_state["max_active"] == 1
                release.set()
                assert await asyncio.gather(*tasks) == ["shutdown", "shutdown"]

        try:
            asyncio.run(_run())
        finally:
            with mcp_tool._lock:
                for pid in (51001, 51002):
                    mcp_tool._stdio_pids.pop(pid, None)
                    mcp_tool._stdio_pid_start_times.pop(pid, None)
                    mcp_tool._stdio_pgids.pop(pid, None)
                    mcp_tool._orphan_stdio_pids.discard(pid)
                    mcp_tool._orphan_stdio_pid_servers.pop(pid, None)
                    mcp_tool._orphan_stdio_pid_start_times.pop(pid, None)
                    mcp_tool._retiring_stdio_owners.clear()

    def test_kill_orphaned_can_filter_by_server_name(self):
        """Reconnect cleanup reaps only the orphan owned by that MCP server."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pids,
            _retiring_stdio_owners,
            _lock,
        )

        target_pid = 454545
        other_pid = 464646
        with _lock:
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _retiring_stdio_owners.clear()
            _orphan_stdio_pids.update({target_pid, other_pid})
            _orphan_stdio_pid_servers[target_pid] = "feishu"
            _orphan_stdio_pid_servers[other_pid] = "mimir"
            _orphan_stdio_pid_start_times[target_pid] = 100
            _orphan_stdio_pid_start_times[other_pid] = 200

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=False), \
             patch("tools.mcp_tool.time.sleep") as mock_sleep:
            _kill_orphaned_mcp_children(server_name="feishu")

        mock_kill.assert_not_called()
        mock_sleep.assert_not_called()
        with _lock:
            assert target_pid not in _orphan_stdio_pids
            assert target_pid not in _orphan_stdio_pid_servers
            assert other_pid in _orphan_stdio_pids
            assert _orphan_stdio_pid_servers[other_pid] == "mimir"

    def test_exact_reaper_reports_same_owner_cleanup_in_flight(self, monkeypatch):
        """并发严格回收必须返回 pending，不能把 in-flight owner 当成功。"""
        import tools.mcp_tool as mcp_tool

        fake_pid = 474747
        owner = ("/profiles/a", "shared", object())
        sleeping = threading.Event()
        release = threading.Event()
        alive = {"value": True}
        first_result = {}

        with mcp_tool._lock:
            mcp_tool._orphan_stdio_pids.add(fake_pid)
            mcp_tool._orphan_stdio_pid_servers[fake_pid] = owner
            mcp_tool._orphan_stdio_pid_start_times[fake_pid] = 100

        def fake_sleep(_seconds):
            sleeping.set()
            assert release.wait(timeout=2)
            alive["value"] = False

        monkeypatch.setattr(mcp_tool.time, "sleep", fake_sleep)
        monkeypatch.setattr(
            "gateway.status._pid_exists", lambda _pid: alive["value"]
        )
        monkeypatch.setattr(
            "gateway.status.get_process_start_time", lambda _pid: 100
        )
        monkeypatch.setattr(mcp_tool.os, "kill", lambda _pid, _sig: None)

        def first_reap():
            first_result["value"] = mcp_tool._kill_orphaned_mcp_children(
                False, "shared", owner
            )

        thread = threading.Thread(target=first_reap)
        try:
            thread.start()
            assert sleeping.wait(timeout=2)
            assert mcp_tool._kill_orphaned_mcp_children(
                False, "shared", owner
            ) == [fake_pid]
            release.set()
            thread.join(timeout=2)
            assert not thread.is_alive()
            assert first_result["value"] == []
        finally:
            release.set()
            thread.join(timeout=2)
            with mcp_tool._lock:
                mcp_tool._orphan_stdio_pids.discard(fake_pid)
                mcp_tool._orphan_stdio_pid_servers.pop(fake_pid, None)
                mcp_tool._orphan_stdio_pid_start_times.pop(fake_pid, None)
                mcp_tool._retiring_stdio_owners.clear()

    def test_reaper_does_not_signal_reused_pid(self, monkeypatch):
        """PID start time 已变化时，旧 owner 必须按已退出处理。"""
        import tools.mcp_tool as mcp_tool

        fake_pid = 484848
        owner = ("/profiles/a", "shared", object())
        with mcp_tool._lock:
            mcp_tool._orphan_stdio_pids.add(fake_pid)
            mcp_tool._orphan_stdio_pid_servers[fake_pid] = owner
            mcp_tool._orphan_stdio_pid_start_times[fake_pid] = 100

        kill = MagicMock()
        monkeypatch.setattr(mcp_tool.os, "kill", kill)
        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
        monkeypatch.setattr(
            "gateway.status.get_process_start_time", lambda _pid: 200
        )
        monkeypatch.setattr(mcp_tool.time, "sleep", lambda _seconds: None)

        assert mcp_tool._kill_orphaned_mcp_children(
            False, "shared", owner
        ) == []
        kill.assert_not_called()
        with mcp_tool._lock:
            assert fake_pid not in mcp_tool._orphan_stdio_pids
            assert fake_pid not in mcp_tool._orphan_stdio_pid_servers
            assert fake_pid not in mcp_tool._orphan_stdio_pid_start_times

    @pytest.mark.parametrize(
        ("expected_start", "live_start"),
        [(None, 200), (100, None)],
    )
    def test_reaper_keeps_owner_when_pid_generation_is_unknown(
        self, monkeypatch, expected_start, live_start
    ):
        """generation 任一侧不可读时不得发信号或删除 owner。"""
        import tools.mcp_tool as mcp_tool

        fake_pid = 494949
        owner = ("/profiles/a", "shared", object())
        with mcp_tool._lock:
            mcp_tool._orphan_stdio_pids.add(fake_pid)
            mcp_tool._orphan_stdio_pid_servers[fake_pid] = owner
            mcp_tool._orphan_stdio_pid_start_times[fake_pid] = expected_start

        kill = MagicMock()
        monkeypatch.setattr(mcp_tool.os, "kill", kill)
        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
        monkeypatch.setattr(
            "gateway.status.get_process_start_time", lambda _pid: live_start
        )
        monkeypatch.setattr(mcp_tool.time, "sleep", lambda _seconds: None)

        assert mcp_tool._kill_orphaned_mcp_children(
            False, "shared", owner
        ) == [fake_pid]
        kill.assert_not_called()
        with mcp_tool._lock:
            assert mcp_tool._orphan_stdio_pid_servers[fake_pid] == owner
            assert (
                mcp_tool._orphan_stdio_pid_start_times[fake_pid]
                == expected_start
            )
            assert not mcp_tool._retiring_stdio_owners
            mcp_tool._orphan_stdio_pids.discard(fake_pid)
            mcp_tool._orphan_stdio_pid_servers.pop(fake_pid, None)
            mcp_tool._orphan_stdio_pid_start_times.pop(fake_pid, None)

    def test_reaper_keeps_unknown_generation_when_direct_pid_exited_but_group_lives(
        self, monkeypatch
    ):
        """direct PID 退出后仍可能有同组后代；generation 未知时不得发组信号。"""
        import tools.mcp_tool as mcp_tool

        fake_pid = 499949
        owner = ("/profiles/a", "shared", object())
        with mcp_tool._lock:
            mcp_tool._orphan_stdio_pids.add(fake_pid)
            mcp_tool._orphan_stdio_pid_servers[fake_pid] = owner
            mcp_tool._orphan_stdio_pid_start_times[fake_pid] = None
            mcp_tool._stdio_pgids[fake_pid] = fake_pid

        def signal_group(_pgid, sig):
            return None

        killpg = MagicMock(side_effect=signal_group)
        monkeypatch.setattr(mcp_tool.os, "killpg", killpg)
        monkeypatch.setattr(mcp_tool.os, "getpgrp", lambda: fake_pid + 1)
        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: False)
        monkeypatch.setattr(mcp_tool.time, "sleep", lambda _seconds: None)

        try:
            result = mcp_tool._kill_orphaned_mcp_children(
                False, "shared", owner
            )
            assert call(fake_pid, signal.SIGTERM) not in killpg.call_args_list
            assert result == [fake_pid]
            with mcp_tool._lock:
                assert mcp_tool._orphan_stdio_pid_servers[fake_pid] == owner
                assert mcp_tool._stdio_pgids[fake_pid] == fake_pid
        finally:
            with mcp_tool._lock:
                mcp_tool._orphan_stdio_pids.discard(fake_pid)
                mcp_tool._orphan_stdio_pid_servers.pop(fake_pid, None)
                mcp_tool._orphan_stdio_pid_start_times.pop(fake_pid, None)
                mcp_tool._stdio_pgids.pop(fake_pid, None)
                mcp_tool._retiring_stdio_owners.clear()

    def test_reused_pid_new_owner_is_not_hidden_by_old_retiring_owner(
        self, monkeypatch
    ):
        """同 PID 的旧 orphan in-flight 不能让新 active strict cleanup 假成功。"""
        import tools.mcp_tool as mcp_tool

        fake_pid = 505050
        old_owner = ("/profiles/a", "shared", object())
        new_owner = ("/profiles/b", "shared", object())
        with mcp_tool._lock:
            mcp_tool._orphan_stdio_pids.add(fake_pid)
            mcp_tool._orphan_stdio_pid_servers[fake_pid] = old_owner
            mcp_tool._orphan_stdio_pid_start_times[fake_pid] = 100
            mcp_tool._stdio_pids[fake_pid] = new_owner
            mcp_tool._stdio_pid_start_times[fake_pid] = 200
            mcp_tool._retiring_stdio_owners.add(
                ("orphan", fake_pid, old_owner, 100)
            )

        monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
        monkeypatch.setattr(
            "gateway.status.get_process_start_time", lambda _pid: 200
        )
        monkeypatch.setattr(mcp_tool.os, "kill", lambda _pid, _sig: None)
        monkeypatch.setattr(mcp_tool.time, "sleep", lambda _seconds: None)

        assert mcp_tool._kill_orphaned_mcp_children(
            True, "shared", new_owner
        ) == [fake_pid]
        with mcp_tool._lock:
            assert mcp_tool._stdio_pids[fake_pid] == new_owner
            assert mcp_tool._stdio_pid_start_times[fake_pid] == 200
            mcp_tool._retiring_stdio_owners.clear()
            mcp_tool._stdio_pids.pop(fake_pid, None)
            mcp_tool._stdio_pid_start_times.pop(fake_pid, None)
            mcp_tool._orphan_stdio_pids.discard(fake_pid)
            mcp_tool._orphan_stdio_pid_servers.pop(fake_pid, None)
            mcp_tool._orphan_stdio_pid_start_times.pop(fake_pid, None)

    def test_same_named_real_pids_are_reaped_by_exact_profile_task(
        self, tmp_path
    ):
        """卸载 A profile 只能终止 A 的同名真实子进程。"""
        import tools.mcp_tool as mcp_tool
        from gateway.status import get_process_start_time
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        profile_a = tmp_path / "profiles" / "a"
        profile_b = tmp_path / "profiles" / "b"
        processes = []
        servers = []
        for profile in (profile_a, profile_b, profile_a):
            token = set_hermes_home_override(profile)
            try:
                servers.append(mcp_tool.MCPServerTask("shared"))
            finally:
                reset_hermes_home_override(token)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    start_new_session=True,
                )
            )

        try:
            with mcp_tool._lock:
                for process, server in zip(processes, servers):
                    mcp_tool._stdio_pids[process.pid] = (
                        server.profile_identity,
                        server.name,
                        server,
                    )
                    start_time = get_process_start_time(process.pid)
                    assert start_time is not None
                    mcp_tool._stdio_pid_start_times[process.pid] = start_time

            asyncio.run(servers[0].shutdown())

            processes[0].wait(timeout=5)
            assert processes[1].poll() is None
            assert processes[2].poll() is None
            with mcp_tool._lock:
                assert processes[0].pid not in mcp_tool._stdio_pids
                assert mcp_tool._stdio_pids[processes[1].pid][2] is servers[1]
                assert mcp_tool._stdio_pids[processes[2].pid][2] is servers[2]
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            with mcp_tool._lock:
                for process in processes:
                    mcp_tool._stdio_pids.pop(process.pid, None)
                    mcp_tool._orphan_stdio_pids.discard(process.pid)
                    mcp_tool._orphan_stdio_pid_servers.pop(process.pid, None)
                    mcp_tool._orphan_stdio_pid_start_times.pop(process.pid, None)
                    mcp_tool._stdio_pgids.pop(process.pid, None)
                    mcp_tool._stdio_pid_start_times.pop(process.pid, None)
                    mcp_tool._retiring_stdio_owners.clear()


# ---------------------------------------------------------------------------
# Fix 2b: stdio descendant reaping via process group (issue #23799)
# ---------------------------------------------------------------------------
#
# When a stdio MCP wrapper (e.g. ``openclaw mcp serve``) itself spawns a
# helper subprocess (``claude mcp serve``) and then exits, the helper
# reparents to systemd-user and is invisible to the per-pid orphan reaper.
# The fix captures the wrapper's pgid at spawn time and reaps via killpg,
# which reaches same-group descendants whether or not the direct pid is alive.

class TestStdioPgroupReaping:
    """_kill_orphaned_mcp_children reaps via killpg when a pgid is tracked."""

    def _reset_state(self):
        from tools.mcp_tool import (
            _orphan_stdio_pid_servers,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pids,
            _retiring_stdio_owners,
            _stdio_pgids,
            _stdio_pid_start_times,
            _stdio_pids,
            _lock,
        )
        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pid_start_times.clear()
            _retiring_stdio_owners.clear()
            _stdio_pgids.clear()
            _stdio_pid_start_times.clear()

    def test_killpg_used_when_pgid_tracked(self, monkeypatch):
        """SIGTERM and SIGKILL route through killpg when pgid is known."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pid_start_times,
            _stdio_pgids,
            _lock,
        )

        self._reset_state()
        fake_pid = 525252
        fake_pgid = 525252  # session leader: pgid == pid
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_start_times[fake_pid] = 100
            _stdio_pgids[fake_pid] = fake_pgid

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        # Ensure os.killpg exists on this platform for the test to make sense;
        # the production fallback path is covered by the per-pid tests above.
        if not hasattr(os, "killpg"):
            pytest.skip("os.killpg not available on this platform")

        with patch("tools.mcp_tool.os.killpg") as mock_killpg, \
             patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch("gateway.status.get_process_start_time", return_value=100), \
             patch("time.sleep"):
            assert _kill_orphaned_mcp_children() == [fake_pid]

        # Both phases should have used killpg (pgroup reach), not per-pid kill.
        mock_killpg.assert_any_call(fake_pgid, signal.SIGTERM)
        mock_killpg.assert_any_call(fake_pgid, fake_sigkill)
        assert mock_killpg.call_count == 2
        mock_kill.assert_not_called()

        with _lock:
            assert fake_pid in _orphan_stdio_pids
            assert _stdio_pgids[fake_pid] == fake_pgid

        with patch("gateway.status._pid_exists", return_value=False), \
             patch("tools.mcp_tool.os.killpg", side_effect=ProcessLookupError), \
             patch("tools.mcp_tool.os.kill"), \
             patch("time.sleep"):
            assert _kill_orphaned_mcp_children() == []
        with _lock:
            assert fake_pid not in _orphan_stdio_pids
            assert fake_pid not in _stdio_pgids

    def test_killpg_skipped_when_pgid_matches_gateway_own_pgroup(self, monkeypatch):
        """#47134: when a tracked MCP child shares the gateway's OWN process
        group, killpg(pgid) would signal the gateway itself and crash it.
        The guard must skip killpg for that pgid and fall through to per-pid
        os.kill instead."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _orphan_stdio_pid_start_times,
            _stdio_pgids,
            _lock,
        )

        if not hasattr(os, "killpg") or not hasattr(os, "getpgrp"):
            pytest.skip("os.killpg/os.getpgrp not available on this platform")

        self._reset_state()
        gateway_pgid = 424242
        fake_pid = 717171  # a child pid that resolves to the gateway's pgid
        other_pid = 818181  # a normal child in its OWN (non-gateway) group
        other_pgid = 818181
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_start_times[fake_pid] = 100
            _stdio_pgids[fake_pid] = gateway_pgid  # == gateway's own pgid
            _orphan_stdio_pids.add(other_pid)
            _orphan_stdio_pid_start_times[other_pid] = 200
            _stdio_pgids[other_pid] = other_pgid  # distinct group → killpg OK

        fake_sigkill = 9
        monkeypatch.setattr(signal, "SIGKILL", fake_sigkill, raising=False)

        with patch("tools.mcp_tool.os.getpgrp", return_value=gateway_pgid), \
             patch("tools.mcp_tool.os.killpg") as mock_killpg, \
             patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=True), \
             patch(
                 "gateway.status.get_process_start_time",
                 side_effect=lambda pid: 100 if pid == fake_pid else 200,
             ), \
             patch("time.sleep"):
            _kill_orphaned_mcp_children()

        # killpg must NEVER be called for the gateway's own pgid (would self-kill).
        killpg_pgids = [call.args[0] for call in mock_killpg.call_args_list]
        assert gateway_pgid not in killpg_pgids, (
            "killpg was called with the gateway's own pgid — self-kill (#47134)"
        )
        # The shared-pgid child must be reaped via per-pid kill instead.
        mock_kill.assert_any_call(fake_pid, signal.SIGTERM)
        mock_kill.assert_any_call(fake_pid, fake_sigkill)
        # NEGATIVE CONTROL: a child in a DISTINCT group must STILL use killpg —
        # the guard must skip only the gateway's own group, not all pgids.
        assert other_pgid in killpg_pgids, (
            "killpg must still be used for a non-gateway pgid (guard too broad)"
        )


    def test_no_pgid_uses_per_pid_kill(self, monkeypatch):
        """When no pgid is recorded (e.g. Windows), fall back to os.kill."""
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pids,
            _orphan_stdio_pid_start_times,
            _stdio_pgids,
            _lock,
        )

        self._reset_state()
        fake_pid = 747474
        with _lock:
            _orphan_stdio_pids.add(fake_pid)
            _orphan_stdio_pid_start_times[fake_pid] = 100
            # No entry in _stdio_pgids.

        with patch("tools.mcp_tool.os.kill") as mock_kill, \
             patch("gateway.status._pid_exists", return_value=False), \
             patch("time.sleep"):
            # killpg may or may not exist; either way the no-pgid path skips it.
            _kill_orphaned_mcp_children()

        mock_kill.assert_not_called()

        with _lock:
            assert fake_pid not in _orphan_stdio_pids

    @pytest.mark.live_system_guard_bypass
    @pytest.mark.skipif(
        not hasattr(os, "killpg") or not hasattr(os, "setsid"),
        reason="POSIX-only: requires os.killpg and os.setsid",
    )
    def test_grandchild_reaped_via_pgroup(self, tmp_path):
        """End-to-end: parent spawns grandchild, parent exits, killpg reaps grandchild.

        Mirrors issue #23799: a stdio MCP wrapper (parent) launches a long-lived
        helper subprocess (grandchild) in the same process group, then the
        wrapper exits while the grandchild keeps running.  killpg on the pgid
        captured at spawn time must still deliver the signal to the grandchild.

        Marked ``live_system_guard_bypass`` because this test genuinely needs
        real signal delivery to its own subprocess tree (the conftest guard
        only knows the test's *initial* children; the spawned tree here is
        outside that allowlist).
        """
        import subprocess
        import sys
        import time as _time

        psutil = pytest.importorskip("psutil")

        # Grandchild: sleep forever, write its pid then wait.  The pid file
        # is written to a temp path and os.replace()d into place so the
        # polling reader below can never observe a created-but-empty file
        # (CI flake: int('') ValueError when the reader won the race between
        # open('w') creating the file and write() filling it).
        grandchild_pid_file = tmp_path / "grandchild.pid"
        grandchild_script = tmp_path / "grandchild.py"
        grandchild_script.write_text(
            "import os, signal, sys, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"tmp = {str(grandchild_pid_file)!r} + '.tmp'\n"
            "with open(tmp, 'w') as f:\n"
            "    f.write(str(os.getpid()))\n"
            f"os.replace(tmp, {str(grandchild_pid_file)!r})\n"
            "while True:\n"
            "    time.sleep(0.5)\n"
        )

        # Parent: spawn grandchild, exit immediately (without killing it).
        parent_script = tmp_path / "parent.py"
        parent_script.write_text(
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(grandchild_script)!r}])\n"
            "time.sleep(0.2)\n"
            # Parent exits — grandchild reparents to init.
        )

        # Spawn parent in its own session (mirrors stdio_client behaviour).
        parent = subprocess.Popen(
            [sys.executable, str(parent_script)],
            start_new_session=True,
        )
        from gateway.status import get_process_start_time

        parent_pgid = os.getpgid(parent.pid)
        parent_start_time = get_process_start_time(parent.pid)
        assert parent_start_time is not None
        # Wait for parent to exit and grandchild to spin up.
        parent.wait(timeout=15)
        deadline = _time.time() + 15  # fresh CPython spinup dilates under CI load
        while _time.time() < deadline and not grandchild_pid_file.exists():
            _time.sleep(0.05)
        assert grandchild_pid_file.exists(), "grandchild did not start"
        grandchild_pid = int(grandchild_pid_file.read_text().strip())

        # Sanity: grandchild is alive and shares the parent's pgid.
        assert psutil.pid_exists(grandchild_pid)
        assert os.getpgid(grandchild_pid) == parent_pgid

        # Drive the reaper: register the parent pid + pgid as an orphan.
        from tools.mcp_tool import (
            _kill_orphaned_mcp_children,
            _orphan_stdio_pid_servers,
            _orphan_stdio_pid_start_times,
            _orphan_stdio_pids,
            _stdio_pgids,
            _stdio_pids,
            _lock,
        )
        with _lock:
            _stdio_pids.clear()
            _orphan_stdio_pids.clear()
            _orphan_stdio_pid_servers.clear()
            _orphan_stdio_pid_start_times.clear()
            _stdio_pgids.clear()
            _orphan_stdio_pids.add(parent.pid)
            _orphan_stdio_pid_servers[parent.pid] = "orphan"
            _orphan_stdio_pid_start_times[parent.pid] = parent_start_time
            _stdio_pgids[parent.pid] = parent_pgid
        def _grandchild_is_running() -> bool:
            try:
                return psutil.Process(grandchild_pid).status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False

        try:
            _kill_orphaned_mcp_children()

            # 孙进程忽略 SIGTERM；即使直接父 PID 已退出，第三阶段也必须发现
            # 仍存活的进程组并发送 SIGKILL。孤儿退出后短暂成为 zombie 也算已退出。
            deadline = _time.time() + 10
            while _time.time() < deadline and _grandchild_is_running():
                _time.sleep(0.05)
            assert not _grandchild_is_running(), (
                "grandchild survived killpg-based reaping (issue #23799 regression)"
            )
        finally:
            # 断言失败后才兜底清理，不能让测试自己制造成功条件。
            if _grandchild_is_running():
                try:
                    os.kill(grandchild_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


# ---------------------------------------------------------------------------
# Fix 3: MCP reload timeout (cli.py)
# ---------------------------------------------------------------------------

class TestMCPReloadTimeout:
    """_check_config_mcp_changes uses a timeout on _reload_mcp."""

    def test_reload_timeout_does_not_block_forever(self, tmp_path, monkeypatch):
        """If _reload_mcp hangs, the config watcher times out and returns."""
        import time

        # Create a mock HermesCLI-like object with the needed attributes
        class FakeCLI:
            _config_mtime = 0.0
            _config_mcp_servers = {}
            _last_config_check = 0.0
            _command_running = False
            config = {}
            agent = None

            def _reload_mcp(self):
                # Simulate a hang — sleep longer than the timeout
                time.sleep(60)

            def _slow_command_status(self, cmd):
                return cmd

        # This test verifies the timeout mechanism exists in the code
        # by checking that _check_config_mcp_changes doesn't call
        # _reload_mcp directly (it uses a thread now)
        import inspect
        from cli import HermesCLI
        source = inspect.getsource(HermesCLI._check_config_mcp_changes)
        # The fix adds threading.Thread for _reload_mcp
        assert "Thread" in source or "thread" in source.lower(), \
            "_check_config_mcp_changes should use a thread for _reload_mcp"


# ---------------------------------------------------------------------------
# Fix 4: MCP initial connection retry with backoff
# (Ported from Kilo Code's MCP resilience fix)
# ---------------------------------------------------------------------------

class TestMCPInitialConnectionRetry:
    """MCPServerTask.run() retries initial connection failures instead of giving up."""

    def test_initial_connect_retries_constant_exists(self):
        """_MAX_INITIAL_CONNECT_RETRIES should be defined."""
        from tools.mcp_tool import _MAX_INITIAL_CONNECT_RETRIES
        assert _MAX_INITIAL_CONNECT_RETRIES >= 1


    def test_initial_connect_retry_respects_shutdown(self):
        """Shutdown during initial retry backoff aborts cleanly."""
        from tools.mcp_tool import MCPServerTask

        async def _run():
            server = MCPServerTask("test-shutdown")
            attempt = 0

            async def fake_run_stdio(self_inner, config):
                nonlocal attempt
                attempt += 1
                if attempt == 1:
                    raise ConnectionError("transient failure")
                # Should not reach here because shutdown fires during sleep
                raise AssertionError("Should not attempt after shutdown")

            with patch.object(MCPServerTask, '_run_stdio', fake_run_stdio), \
                 patch('tools.mcp_tool._jittered', lambda s: 0.01):
                task = asyncio.ensure_future(server.run({"command": "fake"}))

                # Give the first attempt time to fail, then set shutdown
                # during the backoff sleep
                await asyncio.sleep(0.1)
                server._shutdown_event.set()
                await server._ready.wait()

                # Should have the error set and be done
                assert server._error is not None
                await task

        asyncio.get_event_loop().run_until_complete(_run())


# ---------------------------------------------------------------------------
# Fix: drain pending tasks before closing the MCP loop
# ---------------------------------------------------------------------------

class TestMCPLoopDrainOnStop:
    """_stop_mcp_loop reaps pending tasks while the loop is still open."""

    def test_pending_task_cleanup_runs_before_close(self):
        """A task left on the loop must finish its cleanup before close.

        Regression for #60197: a task still suspended when the loop closes is
        resumed later by the GC, and its ``finally`` then drives ``cancel()``
        -> ``call_soon()`` against the closed loop, surfacing as an ignored
        ``RuntimeError: Event loop is closed``. Servers absent from
        ``_servers`` (e.g. parked after an initial-connect failure) never get
        ``shutdown()``, so this drain is their only reaper.
        """
        import time
        import tools.mcp_tool as mcp_mod

        state = {
            "started": False,
            "cleanup_ran": False,
            "cleanup_error": None,
            "task": None,
        }

        async def _parked():
            state["task"] = asyncio.current_task()
            state["started"] = True
            try:
                await asyncio.sleep(3600)
            finally:
                # Mirrors _wait_for_reconnect_or_shutdown's finally: cancelling
                # a helper task needs call_soon(), which a closed loop rejects.
                try:
                    helper = asyncio.ensure_future(asyncio.sleep(0))
                    helper.cancel()
                    state["cleanup_ran"] = True
                except BaseException as exc:
                    state["cleanup_error"] = exc

        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        try:
            mcp_mod._ensure_mcp_loop()
            with mcp_mod._lock:
                loop = mcp_mod._mcp_loop
            assert loop is not None
            asyncio.run_coroutine_threadsafe(_parked(), loop)

            deadline = time.monotonic() + 5
            while not state["started"] and time.monotonic() < deadline:
                time.sleep(0.01)
            assert state["started"], "task never started on the MCP loop"

            stop_saw_cleanup = []
            call_soon = loop.call_soon
            call_soon_threadsafe = loop.call_soon_threadsafe

            def record_stop_order(schedule, callback, *args, **kwargs):
                if (
                    getattr(callback, "__self__", None) is loop
                    and getattr(callback, "__name__", None) == "stop"
                ):
                    stop_saw_cleanup.append(state["cleanup_ran"])
                return schedule(callback, *args, **kwargs)

            with (
                patch.object(
                    loop,
                    "call_soon",
                    side_effect=lambda callback, *args, **kwargs: record_stop_order(
                        call_soon, callback, *args, **kwargs
                    ),
                ),
                patch.object(
                    loop,
                    "call_soon_threadsafe",
                    side_effect=lambda callback, *args, **kwargs: record_stop_order(
                        call_soon_threadsafe, callback, *args, **kwargs
                    ),
                ),
            ):
                mcp_mod._stop_mcp_loop()

            assert state["task"] is not None
            assert state["task"].done(), "task left pending when the loop closed"
            assert state["cleanup_error"] is None, (
                f"cleanup ran against a closed loop: {state['cleanup_error']!r}"
            )
            assert state["cleanup_ran"], "task cleanup never ran"
            assert stop_saw_cleanup == [True], "loop.stop ran before task cleanup"
        finally:
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
            mcp_mod._stop_mcp_loop()

    def test_drain_is_bounded_when_task_ignores_cancellation(self, caplog):
        """A cancellation-resistant task must not hang final MCP shutdown."""
        import tools.mcp_tool as mcp_mod

        async def _run():
            release = asyncio.Event()

            async def cancellation_resistant():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()

            task = asyncio.create_task(cancellation_resistant())
            await asyncio.sleep(0)
            try:
                with caplog.at_level("WARNING", logger=mcp_mod.logger.name):
                    async with asyncio.timeout(0.5):
                        await mcp_mod._drain_mcp_loop_tasks(timeout=0.01)
                assert not task.done(), "drain waited indefinitely for resistant task"
            finally:
                release.set()
                if not task.done() and task.cancelling() == 0:
                    task.cancel()
                await task

        asyncio.run(_run())

        assert any(
            "still pending after" in record.getMessage()
            for record in caplog.records
        )

    def test_outer_timeout_still_allows_loop_owned_drain_before_stop(
        self, caplog, monkeypatch
    ):
        """A blocked loop must drain after it resumes, not stop ahead of the drain."""
        import threading
        import tools.mcp_tool as mcp_mod

        parked_started = threading.Event()
        cleanup_ran = threading.Event()
        blocker_started = threading.Event()
        release_blocker = threading.Event()

        async def parked_task():
            parked_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_ran.set()

        def block_loop():
            blocker_started.set()
            release_blocker.wait(timeout=5)

        with mcp_mod._lock:
            mcp_mod._servers.clear()
            mcp_mod._server_connecting.clear()
        mcp_mod._ensure_mcp_loop()
        with mcp_mod._lock:
            loop = mcp_mod._mcp_loop
        assert loop is not None

        future = asyncio.run_coroutine_threadsafe(parked_task(), loop)
        assert parked_started.wait(timeout=2)
        loop.call_soon_threadsafe(block_loop)
        assert blocker_started.wait(timeout=2)

        monkeypatch.setattr(mcp_mod, "_MCP_LOOP_DRAIN_TIMEOUT", 0.01)
        release_timer = threading.Timer(1.2, release_blocker.set)
        release_timer.start()
        try:
            with caplog.at_level("WARNING", logger=mcp_mod.logger.name):
                mcp_mod._stop_mcp_loop()

            assert cleanup_ran.is_set(), "drain was overtaken by loop.stop"
            assert future.done(), "parked task remained pending after loop resumed"
            assert loop.is_closed()
        finally:
            release_timer.cancel()
            release_blocker.set()
            future.cancel()
            with mcp_mod._lock:
                mcp_mod._servers.clear()
                mcp_mod._server_connecting.clear()
            mcp_mod._stop_mcp_loop()

        assert any(
            "Timed out waiting for MCP loop drain" in record.getMessage()
            for record in caplog.records
        )
