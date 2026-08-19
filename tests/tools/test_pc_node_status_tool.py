import json

from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tools import pc_node_status_tool as module
from tools.registry import invalidate_check_fn_cache
from toolsets import _HERMES_CORE_TOOLS, resolve_toolset


class _Response:
    content = b'{"success":true,"result":{"connected":true,"status":"online"}}'

    def json(self):
        return {
            "success": True,
            "result": {"connected": True, "status": "online"},
        }


class _Client:
    trust_env = True

    def __init__(self, captured):
        self.captured = captured

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, **kwargs):
        self.captured.update(url=url, trust_env=self.trust_env, **kwargs)
        return _Response()


def _trusted_context(monkeypatch):
    values = {
        "ZETTLAB_BROWSER_ACTION_URL": (
            "http://127.0.0.1:8080/api/v1/internal/browser/action"
        ),
        "ZETTLAB_AGENT_ACTION_TOKEN": "action-token",
    }
    monkeypatch.setattr(
        module,
        "get_secret",
        lambda name, default="": values.get(name, default),
    )
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": (
            "zettlab:alice:agent-a:chat-1" if name == "HERMES_SESSION_KEY" else default
        ),
    )
    monkeypatch.setattr(
        module,
        "zettlab_browser_session_token",
        lambda: "session-token",
    )


def test_pc_node_status_reads_only_broker_status(monkeypatch):
    captured = {}
    _trusted_context(monkeypatch)
    monkeypatch.setattr(module.requests, "Session", lambda: _Client(captured))

    result = json.loads(module.pc_node_status_tool({}))

    assert result["result"] == {"connected": True, "status": "online"}
    assert captured["url"] == ("http://127.0.0.1:8080/api/v1/internal/pc/action")
    assert captured["trust_env"] is False
    assert captured["json"] == {
        "session_id": "zettlab:alice:agent-a:chat-1",
        "action": "status",
    }
    assert captured["headers"]["X-Zettlab-Agent-Action-Token"] == ("action-token")
    assert captured["headers"]["X-Zettlab-Browser-Session-Token"] == ("session-token")


def test_pc_node_status_is_hidden_without_a_bound_chat_turn(monkeypatch):
    _trusted_context(monkeypatch)
    monkeypatch.setattr(module, "get_session_env", lambda *_: "")

    assert module._check_pc_node_status() is False


def test_pc_node_status_flow_is_reachable_only_on_zet_agent(monkeypatch):
    _trusted_context(monkeypatch)
    invalidate_check_fn_cache()

    assert "pc_node_status" in resolve_toolset(
        "hermes-zet-agent",
        include_registry=False,
    )
    assert "pc_node_status" not in _HERMES_CORE_TOOLS
    assert "pc_node_status" not in resolve_toolset(
        "hermes-cli",
        include_registry=False,
    )

    _clear_tool_defs_cache()
    definitions = get_tool_definitions(
        enabled_toolsets=["hermes-zet-agent"],
        quiet_mode=True,
    )
    assert "pc_node_status" in {item["function"]["name"] for item in definitions}
