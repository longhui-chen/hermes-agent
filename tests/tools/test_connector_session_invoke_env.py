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

    forged_base = build_connector_runtime_env(
        {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "legacy-bearer"}
    )
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in forged_base

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


def test_stable_session_and_profile_identity_flow_without_route_capability(monkeypatch, tmp_path):
    import json
    from agent import secret_scope
    from tools.trusted_direct_runner import run_trusted_python_script

    session_context = _load_session_context(monkeypatch)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "foreign-profile")
    monkeypatch.setenv("HERMES_SESSION_KEY", "foreign-session")
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({"ZET_AGENT_ID": "profile-a", "ZETTLAB_AGENT_ACTION_TOKEN": "profile-a-token"})
    script = tmp_path / "probe.py"
    script.write_text("import os,json\nprint(json.dumps({key:os.environ.get(key) for key in ['ZETTLAB_CONNECTOR_SESSION_ID','ZETTLAB_AGENT_ACTION_TOKEN','HERMES_SESSION_KEY']}))\n")
    try:
        for session in ("zettlab:owner:profile-a:chat-work", "zettlab:owner:profile-a:chat-personal"):
            tokens = session_context.set_session_vars(session_key=session, session_id="api-compacted-lineage")
            try:
                env = build_connector_runtime_env({"ZETTLAB_CONNECTOR_SESSION_ID":"stale", "HERMES_SESSION_KEY":"stale"})
                assert env["ZETTLAB_CONNECTOR_SESSION_ID"] == session
                assert env["ZETTLAB_AGENT_ACTION_TOKEN"] == "profile-a-token"
                assert "HERMES_SESSION_KEY" not in env
                result = run_trusted_python_script(script=script, argv=[], cwd=tmp_path, base_env={}, injected_env=env, timeout=10, secret_values=[], stdlib_only=True)
                received = json.loads(result.output.strip())
                assert received == {"ZETTLAB_CONNECTOR_SESSION_ID":session,"ZETTLAB_AGENT_ACTION_TOKEN":"profile-a-token","HERMES_SESSION_KEY":None}
            finally:
                session_context.clear_session_vars(tokens)
        assert "ZETTLAB_AGENT_ACTION_TOKEN" not in hermes_subprocess_env()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)


def test_cron_keeps_direct_route_instead_of_chat_session(monkeypatch):
    session_context = _load_session_context(monkeypatch)
    tokens = session_context.set_session_vars(session_key="cron-session")
    session_context._CRON_SESSION.set("1")
    session_context.set_zettlab_connector_route_capability("C" * 43)
    try:
        env = build_connector_runtime_env()
        assert "ZETTLAB_CONNECTOR_SESSION_ID" not in env
        assert env["HERMES_SESSION_KEY"] == "C" * 43
    finally:
        session_context.clear_session_vars(tokens)
