import asyncio
import sys
import threading
import types

import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner
from gateway.status import read_runtime_status


class _RetryableFailureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_connect_error",
            "Telegram startup failed: temporary DNS resolution failure.",
            retryable=True,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _DisabledAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=False, token="***"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        raise AssertionError("connect should not be called for disabled platforms")

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SuccessfulAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_stays_alive_for_retryable_startup_errors(monkeypatch, tmp_path):
    """Retryable startup errors should leave the gateway running in
    degraded mode so the reconnect watcher can recover the platform when
    the underlying problem clears.  Previously this returned False from
    ``start()`` and exited the process, which converted a single broken
    platform (e.g. unpaired WhatsApp, DNS blip on Telegram) into a
    systemd restart loop and killed cron jobs in the meantime.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _RetryableFailureAdapter())

    ok = await runner.start()

    # Gateway stays alive in degraded mode; reconnect watcher takes over.
    assert ok is True
    assert runner.should_exit_cleanly is False
    state = read_runtime_status()
    assert state["gateway_state"] in {"degraded", "running"}
    # Telegram was queued for retry, not given up on.
    assert Platform.TELEGRAM in runner._failed_platforms
    assert state["platforms"]["telegram"]["state"] == "retrying"
    assert state["platforms"]["telegram"]["error_code"] == "telegram_connect_error"


@pytest.mark.asyncio
async def test_runner_allows_cron_only_mode_when_no_platforms_are_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=False, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.adapters == {}
    state = read_runtime_status()
    assert state["gateway_state"] == "running"


@pytest.mark.asyncio
async def test_runner_records_connected_platform_state_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _SuccessfulAdapter())
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    ok = await runner.start()

    assert ok is True
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    assert state["platforms"]["discord"]["state"] == "connected"
    assert state["platforms"]["discord"]["error_code"] is None
    assert state["platforms"]["discord"]["error_message"] is None


@pytest.mark.asyncio
async def test_start_gateway_verbosity_imports_redacting_formatter(monkeypatch, tmp_path):
    """Verbosity != None must not crash with NameError on RedactingFormatter (#8044)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    # verbosity=1 triggers the code path that uses RedactingFormatter.
    # Before the fix this raised NameError.
    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=1)

    assert ok is True


@pytest.mark.asyncio
async def test_start_gateway_prepares_trusted_worker_before_memory_monitor_flow(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    events: list[str] = []

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            events.append("runner.start")
            return True

        async def stop(self):
            return None

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name", "noop")

        def start(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr(
        "hermes_logging.setup_logging",
        lambda hermes_home, mode: events.append("setup-logging") or tmp_path,
    )
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.threading.Thread", _NoopThread)
    monkeypatch.setattr(
        "gateway.run._prepare_trusted_video_edit_runtime_before_gateway_threads",
        lambda: events.append("trusted-worker") or True,
    )
    monkeypatch.setattr(
        "gateway.memory_monitor.start_memory_monitoring",
        lambda **kwargs: events.append("memory-monitor") or True,
    )
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(
        config=GatewayConfig(),
        replace=False,
        verbosity=None,
    )

    assert ok is True
    assert events[:4] == [
        "trusted-worker",
        "setup-logging",
        "memory-monitor",
        "runner.start",
    ]


def test_trusted_worker_startup_failure_is_non_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path))
    from gateway import run as gateway_run
    import tools

    terminal_tool = types.SimpleNamespace(
        _late_prepare_video_edit_worker_before_terminal=lambda: (
            _ for _ in ()
        ).throw(PermissionError("late fork"))
    )
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", terminal_tool)
    monkeypatch.setattr(tools, "terminal_tool", terminal_tool, raising=False)

    assert (
        gateway_run._prepare_trusted_video_edit_runtime_before_gateway_threads()
        is False
    )


@pytest.mark.asyncio
async def test_start_gateway_schedules_mcp_discovery_after_runner_start(monkeypatch, tmp_path):
    """A slow/broken MCP server must not block gateway readiness.

    local-server waits on /health from the platform adapter; MCP discovery is
    optional tool setup and should run only after the gateway has started.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    events: list[str] = []
    release = asyncio.Event()
    discovered = asyncio.Event()
    loop = asyncio.get_running_loop()

    class _RunningRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}
            self._restart_requested = False
            self._restart_via_service = False

        async def start(self):
            events.append("runner.start")
            return True

        async def wait_for_shutdown(self):
            events.append("runner.wait")
            await release.wait()

        async def stop(self):
            return None

    def _discover_mcp_tools():
        events.append("mcp.discovery")
        loop.call_soon_threadsafe(discovered.set)

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda _cfg, accept_hooks=False: None,
    )
    monkeypatch.setattr("gateway.run.GatewayRunner", _RunningRunner)
    monkeypatch.setattr("gateway.run._start_cron_ticker", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=_discover_mcp_tools,
            shutdown_mcp_servers=lambda: None,
        ),
    )

    from gateway.run import start_gateway

    task = asyncio.create_task(
        start_gateway(config=GatewayConfig(), replace=False, verbosity=None)
    )
    await asyncio.wait_for(discovered.wait(), timeout=1)
    assert "runner.wait" in events

    release.set()
    assert await task is True
    assert events[0] == "runner.start"
    assert events.index("mcp.discovery") > events.index("runner.start")


@pytest.mark.asyncio
async def test_start_gateway_does_not_wait_for_slow_mcp_discovery(monkeypatch, tmp_path):
    """Gateway readiness must not wait for a slow MCP discovery thread."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    events: list[str] = []
    release = asyncio.Event()
    start_finished = asyncio.Event()
    release_mcp = threading.Event()
    mcp_started = threading.Event()

    class _RunningRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = False
            self.should_exit_with_failure = False
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}
            self._restart_requested = False
            self._restart_via_service = False
            self._mcp_discovery_task = None

        async def start(self):
            events.append("runner.start")
            return True

        async def wait_for_shutdown(self):
            events.append("runner.wait")
            start_finished.set()
            await release.wait()

        async def stop(self):
            return None

    def _discover_mcp_tools():
        mcp_started.set()
        release_mcp.wait(timeout=2)
        events.append("mcp.discovery.done")

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda _cfg, accept_hooks=False: None,
    )
    monkeypatch.setattr("gateway.run.GatewayRunner", _RunningRunner)
    monkeypatch.setattr("gateway.run._start_cron_ticker", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=_discover_mcp_tools,
            shutdown_mcp_servers=lambda: None,
        ),
    )

    from gateway.run import start_gateway

    task = asyncio.create_task(
        start_gateway(config=GatewayConfig(), replace=False, verbosity=None)
    )
    await asyncio.wait_for(start_finished.wait(), timeout=1)

    assert events == ["runner.start", "runner.wait"]
    assert mcp_started.wait(timeout=1)

    release_mcp.set()
    release.set()
    assert await asyncio.wait_for(task, timeout=1) is True
    assert "mcp.discovery.done" in events


@pytest.mark.asyncio
async def test_start_gateway_shutdown_cleanup_runs_on_failure_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls: list[str] = []

    class _FailureRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = False
            self.should_exit_with_failure = True
            self.exit_reason = "adapter failed"
            self.exit_code = None
            self.adapters = {}
            self._restart_requested = False
            self._restart_via_service = False
            self._mcp_discovery_task = None

        async def start(self):
            return True

        async def wait_for_shutdown(self):
            calls.append("runner.wait")

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "agent.shell_hooks.register_from_config",
        lambda _cfg, accept_hooks=False: None,
    )
    monkeypatch.setattr("gateway.run.GatewayRunner", _FailureRunner)
    monkeypatch.setattr("gateway.run._start_cron_ticker", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(
            discover_mcp_tools=lambda: calls.append("mcp.discovery"),
            shutdown_mcp_servers=lambda: calls.append("mcp.shutdown"),
        ),
    )

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=False, verbosity=None)

    assert ok is False
    assert "runner.wait" in calls
    assert "mcp.shutdown" in calls


@pytest.mark.asyncio
async def test_reload_mcp_returns_busy_during_startup_discovery():
    runner = GatewayRunner.__new__(GatewayRunner)

    async def _never_finishes():
        await asyncio.Event().wait()

    task = asyncio.create_task(_never_finishes())
    runner._mcp_discovery_task = task
    try:
        result = await GatewayRunner._execute_mcp_reload(runner, types.SimpleNamespace())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert "still initializing" in result


@pytest.mark.asyncio
async def test_start_gateway_replace_force_uses_terminate_pid(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls = []

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    # get_running_pid returns 42 before we kill the old gateway, then None
    # after remove_pid_file() clears the record (reflects real behavior).
    _pid_state = {"alive": True}
    def _mock_get_running_pid():
        return 42 if _pid_state["alive"] else None
    def _mock_remove_pid_file():
        _pid_state["alive"] = False
    monkeypatch.setattr("gateway.status.get_running_pid", _mock_get_running_pid)
    monkeypatch.setattr("gateway.status.remove_pid_file", _mock_remove_pid_file)
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks",
        lambda **kwargs: 0,
    )
    # force-kill reaps the process: terminate_pid(force=True) flips it dead,
    # and the post-kill re-poll via _pid_exists then sees it gone so the
    # replacement proceeds.
    def _mock_terminate_pid(pid, force=False):
        calls.append((pid, force))
        if force:
            _pid_state["alive"] = False
    monkeypatch.setattr("gateway.status.terminate_pid", _mock_terminate_pid)
    monkeypatch.setattr(
        "gateway.status._pid_exists", lambda pid: _pid_state["alive"]
    )
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("gateway.run.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    assert calls == [(42, False), (42, True)]


@pytest.mark.asyncio
async def test_start_gateway_replace_aborts_when_force_killed_pid_still_alive(
    monkeypatch, tmp_path
):
    """Regression for #19471 (duplicate-gateway half).

    If SIGKILL fails to reap the old gateway, --replace must NOT clear the PID
    file / scoped locks and start a fresh instance — that leaves two live
    gateways fighting over the same token. It should abort instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    calls = []
    removed_pid = False
    released_locks = False

    class _RunnerShouldNotStart:
        def __init__(self, config):
            raise AssertionError("replacement must not start while old PID is alive")

    def _mock_remove_pid_file():
        nonlocal removed_pid
        removed_pid = True

    def _mock_release_all_scoped_locks(**kwargs):
        nonlocal released_locks
        released_locks = True
        return 0

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.remove_pid_file", _mock_remove_pid_file)
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks",
        _mock_release_all_scoped_locks,
    )
    monkeypatch.setattr(
        "gateway.status.terminate_pid",
        lambda pid, force=False: calls.append((pid, force)),
    )
    # _pid_exists never goes False — the force-kill did not take.
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: True)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("gateway.run.os.kill", lambda pid, sig: None)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _RunnerShouldNotStart)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is False
    assert calls == [(42, False), (42, True)]
    assert removed_pid is False
    assert released_locks is False


@pytest.mark.asyncio
async def test_start_gateway_replace_writes_takeover_marker_before_sigterm(
    monkeypatch, tmp_path
):
    """--replace must write a takeover marker BEFORE sending SIGTERM.

    The marker lets the target's shutdown handler identify the signal as a
    planned takeover (→ exit 0) rather than an unexpected kill (→ exit 1).
    Without the marker, PR #5646's signal-recovery path would revive the
    target via systemd Restart=on-failure, starting a flap loop.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Record the ORDER of marker-write + terminate_pid calls
    events: list[str] = []
    marker_paths_seen: list = []

    def record_write_marker(target_pid: int) -> bool:
        events.append(f"write_marker(target_pid={target_pid})")
        # Also check that the marker file actually exists after this call
        marker_paths_seen.append(
            (tmp_path / ".gateway-takeover.json").exists() is False  # not yet
        )
        # Actually write the marker so we can verify cleanup later
        from gateway.status import _get_takeover_marker_path, _write_json_file
        _write_json_file(_get_takeover_marker_path(), {
            "target_pid": target_pid,
            "target_start_time": 0,
            "replacer_pid": 100,
            "written_at": "2026-04-17T00:00:00+00:00",
        })
        return True

    def record_terminate(pid, force=False):
        events.append(f"terminate_pid(pid={pid}, force={force})")

    class _CleanExitRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = None
            self.exit_code = None
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    _pid_state = {"alive": True}
    def _mock_get_running_pid():
        return 42 if _pid_state["alive"] else None
    def _mock_remove_pid_file():
        _pid_state["alive"] = False
    monkeypatch.setattr("gateway.status.get_running_pid", _mock_get_running_pid)
    monkeypatch.setattr("gateway.status.remove_pid_file", _mock_remove_pid_file)
    monkeypatch.setattr(
        "gateway.status.release_all_scoped_locks",
        lambda **kwargs: 0,
    )
    monkeypatch.setattr("gateway.status.write_takeover_marker", record_write_marker)
    monkeypatch.setattr("gateway.status.terminate_pid", record_terminate)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    # Simulate old process exiting on first check so we don't loop into force-kill
    monkeypatch.setattr(
        "gateway.run.os.kill",
        lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _CleanExitRunner)

    from gateway.run import start_gateway

    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is True
    # Ordering: marker written BEFORE SIGTERM
    assert events[0] == "write_marker(target_pid=42)"
    assert any(e.startswith("terminate_pid(pid=42") for e in events[1:])
    # Marker file cleanup: replacer cleans it after loop completes
    assert not (tmp_path / ".gateway-takeover.json").exists()


@pytest.mark.asyncio
async def test_start_gateway_replace_clears_marker_on_permission_denied(
    monkeypatch, tmp_path
):
    """If we fail to kill the existing PID (permission denied), clean up the
    marker so it doesn't grief an unrelated future shutdown."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write_marker(target_pid: int) -> bool:
        from gateway.status import _get_takeover_marker_path, _write_json_file
        _write_json_file(_get_takeover_marker_path(), {
            "target_pid": target_pid,
            "target_start_time": 0,
            "replacer_pid": 100,
            "written_at": "2026-04-17T00:00:00+00:00",
        })
        return True

    def raise_permission(pid, force=False):
        raise PermissionError("simulated EPERM")

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 42)
    monkeypatch.setattr("gateway.status.write_takeover_marker", write_marker)
    monkeypatch.setattr("gateway.status.terminate_pid", raise_permission)
    monkeypatch.setattr("gateway.run.os.getpid", lambda: 100)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)

    from gateway.run import start_gateway

    # Should return False due to permission error
    ok = await start_gateway(config=GatewayConfig(), replace=True, verbosity=None)

    assert ok is False
    # Marker must NOT be left behind
    assert not (tmp_path / ".gateway-takeover.json").exists()


@pytest.mark.asyncio
async def test_runner_degrades_gracefully_when_all_adapters_missing(monkeypatch, tmp_path, caplog):
    """When all enabled platforms have no adapter (missing library or credentials),
    the gateway should NOT return failure — it should warn and continue running for
    cron job execution, matching the behaviour of 'no platforms enabled' (#5196).

    In fleet deployments the same config.yaml is shared across nodes that may only
    have credentials for a subset of platforms.  Requiring perfect credentials on
    every node makes fleet operation impossible."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    # Simulate _create_adapter returning None for ALL platforms (missing library /
    # missing credentials — no connection attempt ever made).
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)

    import logging
    with caplog.at_level(logging.WARNING):
        ok = await runner.start()

    # Must NOT return False — gateway should keep running for cron.
    assert ok is True
    assert runner.should_exit_cleanly is False
    assert runner.adapters == {}
    # Runtime state must remain "running", not "startup_failed".
    state = read_runtime_status()
    assert state["gateway_state"] == "running"
    # A warning must be emitted explaining why no platforms connected.
    assert any(
        "No adapter could be created" in record.message
        for record in caplog.records
    ), "Expected degraded-mode warning when all adapters are missing"


class _NonRetryableFailureAdapter(BasePlatformAdapter):
    """Simulates a fatal config error like token collision."""
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "discord-bot-token_lock",
            "Discord bot token already in use (PID 999). Stop the other gateway first.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_exits_with_ex_config_on_nonretryable_startup_error(monkeypatch, tmp_path):
    """Non-retryable startup errors (token collision, no platforms) must
    set exit_code to 78 (EX_CONFIG) so the s6 finish script can translate
    it to exit 125 (permanent failure).  See #51228."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(runner, "_create_adapter", lambda platform, platform_config: _NonRetryableFailureAdapter())

    ok = await runner.start()

    assert ok is True  # start() returns True (clean exit requested)
    assert runner.should_exit_cleanly is True
    assert runner.exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    state = read_runtime_status()
    assert state["gateway_state"] == "startup_failed"


@pytest.mark.asyncio
async def test_start_gateway_propagates_fatal_config_exit_code(monkeypatch, tmp_path):
    """A clean exit carrying GATEWAY_FATAL_CONFIG_EXIT_CODE must surface as a
    process-level SystemExit(78) — NOT a truthy return — so main() exits 78
    and the s6 finish script can translate it to 125 (no restart).

    This guards the propagation gap: runner.start() stamps exit_code=78 and
    requests a clean exit, but start_gateway()'s clean-exit branch used to
    `return True` before the SystemExit(exit_code) site, so main() exited 0
    and s6 crash-looped anyway (#51228)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    class _FatalConfigRunner:
        def __init__(self, config):
            self.config = config
            self.should_exit_cleanly = True
            self.exit_reason = "discord: Discord bot token already in use"
            self.exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
            self.adapters = {}

        async def start(self):
            return True

        async def stop(self):
            return None

    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: tmp_path)
    monkeypatch.setattr("hermes_logging._add_rotating_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", _FatalConfigRunner)

    from gateway.run import start_gateway

    with pytest.raises(SystemExit) as exc_info:
        await start_gateway(config=GatewayConfig(), replace=False, verbosity=0)

    assert exc_info.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE


def test_runner_warns_when_docker_gateway_lacks_explicit_output_mount(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", '["/etc/localtime:/etc/localtime:ro"]')
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")
        },
        sessions_dir=tmp_path / "sessions",
    )

    with caplog.at_level("WARNING"):
        GatewayRunner(config)

    assert any(
        "host-visible output mount" in record.message
        for record in caplog.records
    )
