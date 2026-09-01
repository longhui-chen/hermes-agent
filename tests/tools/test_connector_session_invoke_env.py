"""Focused Connector Session runner environment boundary tests."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

from tools.environments.local import (
    build_connector_runtime_env,
    hermes_subprocess_env,
)


def _load_session_context(monkeypatch):
    gateway = types.ModuleType("gateway")
    gateway.__path__ = []
    monkeypatch.setitem(sys.modules, "gateway", gateway)
    path = Path(__file__).resolve().parents[2] / "gateway" / "session_context.py"
    spec = importlib.util.spec_from_file_location("gateway.session_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "gateway.session_context", module)
    spec.loader.exec_module(module)
    return module


def test_connector_session_invoke_flag_only_reaches_dedicated_runner(monkeypatch):
    """Tokenless mode and turn capability stay inside the trusted runner."""
    session_context = _load_session_context(monkeypatch)
    session_context._session_context_engaged = True
    session_context.set_zettlab_connector_route_capability("C" * 43)
    monkeypatch.setenv("ZETTLAB_CONNECTOR_SESSION_INVOKE_V1", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "FOREIGN")
    monkeypatch.delenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", raising=False)

    connector_env = build_connector_runtime_env()
    assert connector_env["ZETTLAB_CONNECTOR_SESSION_INVOKE_V1"] == "1"
    assert connector_env["HERMES_SESSION_KEY"] == "C" * 43
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in connector_env

    generic_env = hermes_subprocess_env()
    assert "ZETTLAB_CONNECTOR_SESSION_INVOKE_V1" not in generic_env
    assert "HERMES_SESSION_KEY" not in generic_env


def test_connector_route_capability_is_task_local(monkeypatch):
    session_context = _load_session_context(monkeypatch)

    async def worker(capability):
        session_context.set_zettlab_connector_route_capability(capability)
        await asyncio.sleep(0)
        assert session_context.zettlab_connector_route_capability() == capability
        session_context.set_zettlab_connector_route_capability("")

    async def run_workers():
        await asyncio.gather(worker("A" * 43), worker("B" * 43))

    asyncio.run(run_workers())
    assert session_context.zettlab_connector_route_capability() == ""
