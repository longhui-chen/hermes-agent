import importlib
import json
import subprocess
import sys
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner


class _SuccessfulDiscordAdapter(BasePlatformAdapter):
    def __init__(self, events):
        super().__init__(
            PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD,
        )
        self._events = events

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._events.append("connect:discord")
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _SuccessfulZetAdapter(_SuccessfulDiscordAdapter):
    def __init__(self, events):
        BasePlatformAdapter.__init__(
            self,
            PlatformConfig(enabled=True, extra={"key": "test-key"}),
            Platform.ZET_AGENT,
        )
        self._events = events

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._events.append("connect:zet_agent")
        return True


def test_importing_generic_gateway_does_not_harden_process_flow():
    code = """
import json
import tools.process_security as process_security

calls = []
process_security.harden_sensitive_process = (
    lambda **kwargs: calls.append(kwargs) or True
)
import gateway.run
print(json.dumps(calls, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(completed.stdout) == []


def test_sensitive_boundary_is_lazy_and_does_not_request_no_new_privs(
    monkeypatch,
):
    from tools import process_security

    original_boundary = sys.modules.get("gateway.sensitive_process_boundary")
    calls = []
    monkeypatch.setattr(
        process_security,
        "harden_sensitive_process",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    try:
        sys.modules.pop("gateway.sensitive_process_boundary", None)
        boundary = importlib.import_module("gateway.sensitive_process_boundary")

        assert calls == []
        assert boundary.initialize_gateway_sensitive_process_boundary() is True
        assert calls == [{"no_new_privs": False, "drop_ptrace": True}]
        assert boundary.initialize_gateway_sensitive_process_boundary() is True
        assert calls == [{"no_new_privs": False, "drop_ptrace": True}]
        assert boundary.gateway_sensitive_process_boundary_ready() is True
    finally:
        if original_boundary is None:
            sys.modules.pop("gateway.sensitive_process_boundary", None)
        else:
            sys.modules["gateway.sensitive_process_boundary"] = original_boundary


def test_sensitive_boundary_failure_is_not_retried_late(monkeypatch):
    from tools import process_security

    original_boundary = sys.modules.get("gateway.sensitive_process_boundary")
    calls = []
    monkeypatch.setattr(
        process_security,
        "harden_sensitive_process",
        lambda **kwargs: calls.append(kwargs) or False,
    )
    try:
        sys.modules.pop("gateway.sensitive_process_boundary", None)
        boundary = importlib.import_module("gateway.sensitive_process_boundary")

        assert boundary.initialize_gateway_sensitive_process_boundary() is False
        assert boundary.initialize_gateway_sensitive_process_boundary() is False
        assert calls == [{"no_new_privs": False, "drop_ptrace": True}]
        assert boundary.gateway_sensitive_process_boundary_ready() is False
    finally:
        if original_boundary is None:
            sys.modules.pop("gateway.sensitive_process_boundary", None)
        else:
            sys.modules["gateway.sensitive_process_boundary"] = original_boundary


@pytest.mark.asyncio
async def test_generic_gateway_start_does_not_initialize_boundary_flow(
    monkeypatch,
    tmp_path,
):
    events = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    monkeypatch.setattr(
        "gateway.run.initialize_gateway_sensitive_process_boundary",
        lambda: events.append("boundary") or False,
    )
    monkeypatch.setattr(
        runner,
        "_create_adapter",
        lambda platform, _config: (
            events.append(f"create:{platform.value}")
            or _SuccessfulDiscordAdapter(events)
        ),
    )
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    assert await runner.start() is True
    assert events == ["create:discord", "connect:discord"]


@pytest.mark.asyncio
async def test_boundary_failure_keeps_zet_listener_and_discord_flow(
    monkeypatch,
    tmp_path,
):
    events = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = GatewayConfig(
        platforms={
            Platform.ZET_AGENT: PlatformConfig(
                enabled=True,
                extra={"key": "test-key"},
            ),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    monkeypatch.setattr(
        "gateway.run.initialize_gateway_sensitive_process_boundary",
        lambda: events.append("boundary") or False,
        raising=False,
    )

    def create_adapter(platform, _platform_config):
        events.append(f"create:{platform.value}")
        if platform == Platform.ZET_AGENT:
            return _SuccessfulZetAdapter(events)
        return _SuccessfulDiscordAdapter(events)

    monkeypatch.setattr(runner, "_create_adapter", create_adapter)
    monkeypatch.setattr(runner.hooks, "discover_and_load", lambda: None)
    monkeypatch.setattr(runner.hooks, "emit", AsyncMock())

    ok = await runner.start()

    assert ok is True
    assert runner.should_exit_cleanly is False
    assert list(runner.adapters) == [Platform.ZET_AGENT, Platform.DISCORD]
    assert events == [
        "boundary",
        "create:zet_agent",
        "connect:zet_agent",
        "create:discord",
        "connect:discord",
    ]


@pytest.mark.asyncio
async def test_zet_boundary_failure_is_not_a_fatal_adapter_error(monkeypatch):
    import gateway.platforms.zet_agent as zet_agent

    monkeypatch.setattr(
        zet_agent,
        "initialize_gateway_sensitive_process_boundary",
        lambda: False,
    )
    monkeypatch.setattr(zet_agent, "AIOHTTP_AVAILABLE", False)
    adapter = zet_agent.ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    assert await adapter.connect() is False
    assert adapter.has_fatal_error is False


@pytest.mark.asyncio
async def test_boundary_failure_does_not_read_or_parse_token_flow(monkeypatch):
    import gateway.platforms.zet_agent as zet_agent

    monkeypatch.setattr(
        zet_agent,
        "initialize_gateway_sensitive_process_boundary",
        lambda: True,
    )
    adapter = zet_agent.ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = type(
        "Request",
        (),
        {
            "headers": {
                "X-Zettlab-Business-Execution-Action": "a" * 64,
                "X-Zettlab-Business-Execution-Action-Version": "1",
            },
            "read": AsyncMock(
                side_effect=AssertionError("request must not be read")
            ),
        },
    )()
    monkeypatch.setattr(
        zet_agent,
        "gateway_sensitive_process_boundary_ready",
        lambda: False,
    )

    response = await adapter._diagnostic_chat_completions(request)

    assert response.status == 503
    request.read.assert_not_awaited()
