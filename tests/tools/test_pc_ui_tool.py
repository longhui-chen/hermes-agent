import json

import tools.pc_ui_tool as module
from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tools.registry import invalidate_check_fn_cache


class _Response:
    content = b'{"success":true,"result":{"pid":42,"window_id":7}}'

    def json(self):
        return {"success": True, "result": {"pid": 42, "window_id": 7}}


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
            "action": "ui.snapshot",
            "params": {"app": "Finder"},
        }
        assert kwargs["headers"]["X-Zettlab-Agent-Action-Token"] == "action-token"
        assert kwargs["headers"]["X-Zettlab-Browser-Session-Token"] == "session-token"
        return _Response()


def _configure(monkeypatch):
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
            "zettlab:alice:agent-a:chat-1" if name == "HERMES_SESSION_KEY" else default
        ),
    )
    monkeypatch.setattr(
        module, "zettlab_browser_session_token", lambda: "session-token"
    )


def test_pc_ui_snapshot_uses_session_scoped_pc_action(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(module.requests, "Session", _Client)

    result = json.loads(module.pc_ui_tool({"action": "snapshot", "app": "Finder"}))

    assert result == {"success": True, "result": {"pid": 42, "window_id": 7}}


def test_pc_ui_rejects_unknown_or_incomplete_action_without_network(monkeypatch):
    called = False

    def session():
        nonlocal called
        called = True
        return _Client()

    monkeypatch.setattr(module.requests, "Session", session)
    assert (
        json.loads(module.pc_ui_tool({"action": "shell", "command": "id"}))["code"]
        == "invalid_action"
    )
    assert (
        json.loads(module.pc_ui_tool({"action": "invoke", "pid": 42}))["code"]
        == "invalid_parameters"
    )
    assert called is False


def test_pc_ui_is_reachable_only_from_the_zet_agent_composite(monkeypatch):
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset

    _configure(monkeypatch)
    invalidate_check_fn_cache()
    assert "pc_ui" in resolve_toolset("hermes-zet-agent")
    assert "pc_ui" not in _HERMES_CORE_TOOLS
    assert "pc_ui" not in resolve_toolset("hermes-cli")

    _clear_tool_defs_cache()
    definitions = get_tool_definitions(
        enabled_toolsets=["hermes-zet-agent"],
        quiet_mode=True,
    )
    assert "pc_ui" in {item["function"]["name"] for item in definitions}


def test_pc_ui_schema_forbids_shell_paths_and_arbitrary_properties():
    properties = module.PC_UI_SCHEMA["parameters"]["properties"]
    assert module.PC_UI_SCHEMA["parameters"]["additionalProperties"] is False
    assert "command" not in properties
    assert "path" not in properties
    assert "cdp" not in properties


def test_zet_agent_prompt_requires_snapshot_first_for_local_apps():
    from gateway.platforms.zet_agent import _ZET_WORKDIR_SECTION

    assert "使用 `pc_ui`" in _ZET_WORKDIR_SECTION
    assert "必须先 `snapshot`" in _ZET_WORKDIR_SECTION
    assert "调用 `pc_node_status`" in _ZET_WORKDIR_SECTION
    assert "`connected=true` 且 `computer_use=false`" in _ZET_WORKDIR_SECTION
    assert "而不是笼统声称当前会话没有工具" in _ZET_WORKDIR_SECTION
    assert "不要改用 terminal" in _ZET_WORKDIR_SECTION
