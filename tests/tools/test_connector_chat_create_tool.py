"""The Agent connector-create path uses a direct scoped RPC, never shell argv."""

import json
import urllib.request

import pytest

from gateway.platforms import zet_agent_connector_chat_tool as connector_tool


def _env() -> dict[str, str]:
    return {
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=agent-1",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-action-token",
        "ZETTLAB_CONNECTOR_SESSION_ID": "chat-1",
        "ZETTLAB_TURN_ID": "turn-1",
    }


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        return self._body[:n] if n > 0 else self._body


def _fake_transport(monkeypatch, body: bytes):
    holder = {"timeout": None}

    class _FakeOpener:
        def open(self, request, timeout):
            holder["timeout"] = timeout
            return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: _FakeOpener())
    return holder


def test_connector_rpc_parses_tools_list_within_read_budget(monkeypatch):
    holder = _fake_transport(monkeypatch, json.dumps({
        "result": {"tools": [{"name": "connector.create_readonly_template"}]},
    }).encode())
    monkeypatch.setattr(connector_tool, "_runtime_env", _env)

    response = connector_tool._call(_env(), "tools/list", {}, 20.0)
    assert response["result"]["tools"][0]["name"] == "connector.create_readonly_template"
    assert holder["timeout"] == 20.0


def test_connector_rpc_fails_closed_when_response_exceeds_read_budget(monkeypatch):
    # AC publishes 338 tools (~383KB); the historic 64KB read silently
    # truncated the JSON mid-string. An over-budget body must surface as an
    # error, never as a swallowed parse failure on truncated bytes.
    _fake_transport(monkeypatch, b"x" * (connector_tool._MAX_RPC_RESPONSE_BYTES + 10))
    monkeypatch.setattr(connector_tool, "_runtime_env", _env)

    with pytest.raises(ValueError):
        connector_tool._call(_env(), "tools/list", {}, 20.0)


def test_connector_chat_create_full_capability_flow_keeps_secret_out_of_result(monkeypatch):
    monkeypatch.setattr(connector_tool, "_runtime_env", _env)
    seen = []

    def fake_rpc(request, timeout):
        body = json.loads(request.data)
        seen.append((request, body, timeout))
        if body["method"] == "tools/list":
            return {"result": {"tools": [{"name": "connector.create_readonly_template"}]}}
        return {"result": {
            "connection_id": "conn-1", "connection_status": "active",
            "template_id": "dify-workflow-api", "session_ready": True,
            "verified": False,
            "secret": "synthetic-pat-not-real",
        }}

    monkeypatch.setattr(connector_tool, "_rpc", fake_rpc)
    probe = json.loads(connector_tool.connector_chat_create_tool({"action": "probe"}))
    assert probe == {"available": True, "ok": True}

    created = json.loads(connector_tool.connector_chat_create_tool({
        "action": "create", "template_id": "dify-workflow-api",
        "template_version": 1, "variables": {},
        "secret": "synthetic-pat-not-real",
        "idempotency_key": "chat-create-dify-0123456789",
    }))
    assert created == {
        "ok": True, "connection_id": "conn-1", "connection_status": "active",
        "template_id": "dify-workflow-api", "session_ready": True,
        "verified": False,
    }
    assert "synthetic-pat-not-real" not in json.dumps(created)
    assert seen[1][1]["params"]["arguments"]["secret"] == "synthetic-pat-not-real"
    assert seen[1][1]["params"]["arguments"]["template_id"] == "dify-workflow-api"
    for request, _, _ in seen:
        assert request.get_header("X-zettlab-agent-action-token") == "profile-action-token"
        assert request.get_header("X-zettlab-connector-session-id") == "chat-1"
        assert request.get_header("X-zettlab-turn-id") == "turn-1"
        assert request.full_url.startswith("http://127.0.0.1:9090/")


def test_connector_chat_create_rejects_external_url_and_missing_turn(monkeypatch):
    monkeypatch.setattr(connector_tool, "_runtime_env", lambda: {
        **_env(), "ZETTLAB_CONNECTORS_URL": "https://api.example.com/rpc",
    })
    assert json.loads(connector_tool.connector_chat_create_tool({"action": "probe"}))["available"] is False


def test_connector_chat_create_rejects_invalid_template_identifier(monkeypatch):
    monkeypatch.setattr(connector_tool, "_runtime_env", _env)
    result = json.loads(connector_tool.connector_chat_create_tool({
        "action": "create", "template_id": "../untrusted-template",
        "template_version": 1, "variables": {}, "secret": "synthetic",
        "idempotency_key": "chat-create-invalid-0123456789",
    }))
    assert result == {"code": "connector_chat_create_invalid_request", "ok": False}
    monkeypatch.setattr(connector_tool, "_runtime_env", lambda: {
        **_env(), "ZETTLAB_TURN_ID": "",
    })
    assert json.loads(connector_tool.connector_chat_create_tool({"action": "probe"}))["available"] is False


def test_connector_chat_create_tool_reachable_in_zet_agent_catalog(monkeypatch):
    import toolsets

    assert "connector_chat_create" in toolsets._HERMES_CORE_TOOLS
    assert "connector_chat_create" in toolsets.TOOLSETS["zettlab_connectors"]["tools"]
    assert "connector_chat_create" in toolsets.resolve_toolset("hermes-zet-agent")
    monkeypatch.setattr(connector_tool, "_runtime_env", _env)
    definition = connector_tool.registry.get_definitions({"connector_chat_create"}, quiet=True)[0]
    description = definition["function"]["description"]
    assert "next turn" in description
    assert "published Connector template" in description
    assert "read-only Jira/GitLab" not in description
    assert "no automatic tool authorization" not in description.lower()


def test_connector_chat_create_uses_current_profile_session_and_turn_not_process_env(monkeypatch):
    from agent import secret_scope
    from gateway import session_context
    from tools.registry import registry

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "foreign-profile-token")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "https://foreign.example/rpc")
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({
        "ZET_AGENT_ID": "profile-a",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-a-token",
        "ZETTLAB_CONNECTORS_URL": _env()["ZETTLAB_CONNECTORS_URL"],
    })
    tokens = session_context.set_session_vars(session_key="chat-current")
    session_context.set_zettlab_turn_id("turn-current")
    try:
        env = connector_tool._validated_env()
        assert env is not None
        assert env["ZETTLAB_AGENT_ACTION_TOKEN"] == "profile-a-token"
        assert env["ZETTLAB_CONNECTOR_SESSION_ID"] == "chat-current"
        assert env["ZETTLAB_TURN_ID"] == "turn-current"
        assert registry.get_definitions({"connector_chat_create"}, quiet=True)
        session_context.set_zettlab_turn_id("")
        assert connector_tool._validated_env() is None
        assert not registry.get_definitions({"connector_chat_create"}, quiet=True)
    finally:
        session_context.set_zettlab_turn_id("")
        session_context.clear_session_vars(tokens)
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)
