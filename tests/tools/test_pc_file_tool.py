import json

import tools.pc_file_tool as module


class _Response:
    status_code = 200
    content = b'{"success":true,"result":{"entries":[]}}'

    def json(self):
        return {"success": True, "result": {"entries": []}}


class _Client:
    trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:8080/api/v1/internal/pc/action"
        assert self.trust_env is False
        assert kwargs["json"] == {
            "session_id": "zettlab:alice:agent-a:chat-1",
            "action": "file.list",
            "params": {"path": ".", "limit": 100},
        }
        assert kwargs["headers"]["X-Zettlab-Agent-Action-Token"] == "action-token"
        assert kwargs["headers"]["X-Zettlab-Browser-Session-Token"] == "session-token"
        return _Response()


def test_pc_file_defaults_generic_list_to_authorized_root(monkeypatch):
    values = {
        "ZETTLAB_BROWSER_ACTION_URL": "http://127.0.0.1:8080/api/v1/internal/browser/action",
        "ZETTLAB_AGENT_ACTION_TOKEN": "action-token",
    }
    monkeypatch.setattr(
        module, "get_secret", lambda name, default="": values.get(name, default)
    )
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": (
            "zettlab:alice:agent-a:chat-1"
            if name == "HERMES_SESSION_KEY"
            else default
        ),
    )
    monkeypatch.setattr(module, "zettlab_browser_session_token", lambda: "session-token")
    monkeypatch.setattr(module.requests, "Session", _Client)

    result = json.loads(module.pc_file_tool({"action": "list", "limit": 100}))
    assert result == {"success": True, "result": {"entries": []}}


def test_pc_file_rejects_unknown_action_without_network(monkeypatch):
    called = False

    def session():
        nonlocal called
        called = True
        return _Client()

    monkeypatch.setattr(module.requests, "Session", session)
    result = json.loads(module.pc_file_tool({"action": "delete", "path": "."}))
    assert result["code"] == "invalid_action"
    assert called is False


def test_pc_file_is_reachable_only_from_the_zet_agent_composite():
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset

    assert "pc_file" in resolve_toolset("hermes-zet-agent")
    assert "pc_file" not in _HERMES_CORE_TOOLS
    assert "pc_file" not in resolve_toolset("hermes-cli")


def test_pc_file_schema_directs_generic_requests_to_authorized_root():
    description = module.PC_FILE_SCHEMA["description"]
    path_description = module.PC_FILE_SCHEMA["parameters"]["properties"]["path"][
        "description"
    ]

    assert "immediately list path '.'" in description
    assert "do not ask them for a directory path" in description
    assert "Use '.' for the authorized root" in path_description
