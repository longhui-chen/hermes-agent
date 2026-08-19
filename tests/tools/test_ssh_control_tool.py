import json

from toolsets import resolve_toolset
from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tools import ssh_control_tool as module
from tools.registry import registry
from tools.tool_search import classify_tools


class _Response:
    status_code = 200
    content = b'{"data":{"action":"shell.exec","output":"ok"}}'

    def json(self):
        return {"data": {"action": "shell.exec", "output": "ok"}}


class _Session:
    def __init__(self, captured):
        self.captured = captured
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def post(self, url, **kwargs):
        self.captured.update(url=url, trust_env=self.trust_env, **kwargs)
        return _Response()


def _trusted_context(monkeypatch):
    monkeypatch.setattr(module, "get_secret", lambda name, default="": {
        "ZETTLAB_LOCAL_SERVER_URL": "http://127.0.0.1:19090",
        "ZETTLAB_AGENT_ACTION_TOKEN": "action-token",
    }.get(name, default))
    monkeypatch.setattr(module, "hardware_execution_token", lambda: "execution-token")
    monkeypatch.setattr(module, "get_session_env", lambda name, default="": {
        "HERMES_SESSION_ID": "zettlab:owner:agent:chat",
        "HERMES_TURN_ID": "turn-1",
    }.get(name, default))


def test_ssh_control_forwards_unrestricted_command_through_trusted_loopback(monkeypatch):
    _trusted_context(monkeypatch)
    captured = {}
    monkeypatch.setattr(module.requests, "Session", lambda: _Session(captured))

    result = json.loads(module.ssh_control_tool({
        "action": "shell_exec",
        "connection_id": "protocol-1",
        "command": "cd / && find . -type f | sort > /tmp/all-files",
        "timeout_seconds": 120,
    }))

    assert result == {"success": True, "result": {"action": "shell.exec", "output": "ok"}}
    assert captured["url"] == "http://127.0.0.1:19090/api/v1/agent/hardware/protocol/actions"
    assert captured["trust_env"] is False
    assert captured["json"]["command"] == "cd / && find . -type f | sort > /tmp/all-files"
    assert captured["headers"]["X-Zettlab-Agent-Action-Token"] == "action-token"
    assert captured["headers"]["X-Zettlab-Hardware-Execution-Token"] == "execution-token"
    assert captured["timeout"] == 125


def test_ssh_control_encodes_file_write_and_never_accepts_remote_credentials(monkeypatch):
    _trusted_context(monkeypatch)
    captured = {}
    monkeypatch.setattr(module.requests, "Session", lambda: _Session(captured))

    module.ssh_control_tool({
        "action": "file_write",
        "connection_id": "protocol-1",
        "path": "/etc/example.conf",
        "content": "enabled=true\n",
    })

    assert captured["json"] == {
        "action": "file.write",
        "connection_id": "protocol-1",
        "path": "/etc/example.conf",
        "content_base64": "ZW5hYmxlZD10cnVlCg==",
    }
    serialized = json.dumps(captured, default=str)
    assert "password" not in serialized
    assert "private_key" not in serialized


def test_ssh_control_stays_discoverable_but_dispatch_denies_without_execution_token(monkeypatch):
    _trusted_context(monkeypatch)
    monkeypatch.setattr(module, "hardware_execution_token", lambda: "")

    assert module._check_ssh_control() is True
    assert json.loads(module.ssh_control_tool({"action": "list_connections"}))["code"] == "ssh_authorization_unavailable"


def test_ssh_control_is_hidden_without_a_bound_chat_turn(monkeypatch):
    _trusted_context(monkeypatch)
    monkeypatch.setattr(module, "get_session_env", lambda *_: "")

    assert module._check_ssh_control() is False


def test_ssh_control_is_available_only_to_the_trusted_zettlab_agent_surface():
    assert "ssh_control" in resolve_toolset("hermes-zet-agent", include_registry=False)
    assert "ssh_control" not in resolve_toolset("hermes-cli", include_registry=False)


def test_ssh_control_stays_directly_visible_when_the_turn_is_authorized(monkeypatch):
    _trusted_context(monkeypatch)

    definitions = registry.get_definitions({"ssh_control"}, quiet=True)
    visible, deferred = classify_tools(definitions)

    assert [item["function"]["name"] for item in visible] == ["ssh_control"]
    assert deferred == []

    _clear_tool_defs_cache()
    model_definitions = get_tool_definitions(
        enabled_toolsets=["hermes-zet-agent"],
        quiet_mode=True,
    )
    assert "ssh_control" in {
        item["function"]["name"] for item in model_definitions
    }


def test_ssh_control_survives_the_real_zettlab_platform_mapping(monkeypatch):
    from hermes_cli.tools_config import _get_platform_tools

    _trusted_context(monkeypatch)
    enabled = sorted(_get_platform_tools(
        {"platform_toolsets": {"zet_agent": ["hermes-zet-agent", "cronjob"]}},
        "zet_agent",
        include_default_mcp_servers=False,
    ))

    assert "zettlab_ssh" in enabled
    _clear_tool_defs_cache()
    definitions = get_tool_definitions(
        enabled_toolsets=enabled,
        quiet_mode=True,
    )
    assert "ssh_control" in {
        item["function"]["name"] for item in definitions
    }
