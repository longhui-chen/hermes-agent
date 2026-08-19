import json
import base64

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


class _LaunchClient(_Client):
    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:8080/api/v1/internal/pc/action"
        assert kwargs["json"] == {
            "session_id": "zettlab:alice:agent-a:chat-1",
            "action": "ui.launch",
            "params": {"app": "Google Chrome"},
        }
        return _Response()


class _ListAppsClient(_Client):
    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:8080/api/v1/internal/pc/action"
        assert kwargs["json"] == {
            "session_id": "zettlab:alice:agent-a:chat-1",
            "action": "ui.list-apps",
            "params": {},
        }
        return _Response()


class _ListWindowsClient(_Client):
    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:8080/api/v1/internal/pc/action"
        assert kwargs["json"] == {
            "session_id": "zettlab:alice:agent-a:chat-1",
            "action": "ui.list-windows",
            "params": {"app": "Feishu"},
        }
        return _Response()


class _ExactSnapshotClient(_Client):
    def post(self, url, **kwargs):
        assert url == "http://127.0.0.1:8080/api/v1/internal/pc/action"
        assert kwargs["json"] == {
            "session_id": "zettlab:alice:agent-a:chat-1",
            "action": "ui.snapshot",
            "params": {"app": "Feishu", "pid": 42, "window_id": 7},
        }
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


def test_pc_ui_launches_an_exact_app_without_accepting_urls(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(module.requests, "Session", _LaunchClient)

    result = json.loads(
        module.pc_ui_tool({"action": "launch", "app": "Google Chrome"})
    )

    assert result["success"] is True
    assert module._params(
        {"app": "Google Chrome", "value": "https://example.com"}, "launch"
    ) is None


def test_pc_ui_lists_apps_without_accepting_parameters(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(module.requests, "Session", _ListAppsClient)

    result = json.loads(module.pc_ui_tool({"action": "list_apps"}))

    assert result["success"] is True
    assert module._params({"path": "/Applications"}, "list_apps") is None


def test_pc_ui_lists_windows_and_snapshots_one_exact_window(monkeypatch):
    _configure(monkeypatch)
    monkeypatch.setattr(module.requests, "Session", _ListWindowsClient)

    result = json.loads(module.pc_ui_tool({"action": "list_windows", "app": "Feishu"}))

    assert result["success"] is True
    monkeypatch.setattr(module.requests, "Session", _ExactSnapshotClient)
    result = json.loads(
        module.pc_ui_tool(
            {
                "action": "snapshot",
                "app": "Feishu",
                "pid": 42,
                "window_id": 7,
            }
        )
    )
    assert result["success"] is True
    assert module._params({"action": "snapshot", "pid": 42}, "snapshot") is None
    assert module._params({"app": "Feishu", "include_hidden": True}, "snapshot") is None


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


def test_pc_ui_complete_bounded_action_surface_and_target_rules():
    valid = {
        "desktop_snapshot": {},
        "click": {"pid": 42, "window_id": 7, "x": 10, "y": 20},
        "move_cursor": {"scope": "desktop", "x": 10, "y": 20},
        "drag": {"pid": 42, "window_id": 7, "from_x": 1, "from_y": 2, "to_x": 30, "to_y": 40},
        "type_text": {"pid": 42, "window_id": 7, "element": 3, "text": "hello"},
        "invoke_menu": {"pid": 42, "window_id": 7, "path": ["Window", "Zoom"]},
        "verify": {"pid": 42, "window_id": 7, "expect": [{"window": {"exists": True}}]},
        "zoom": {"pid": 42, "window_id": 7, "x1": 1, "y1": 2, "x2": 30, "y2": 40},
        "set_window_frame": {"pid": 42, "window_id": 7, "x": 0, "y": 0, "width": 800, "height": 600},
        "clipboard_read": {"include_text": True},
        "clipboard_write": {"text": "hello"},
        "kill_app": {"pid": 42},
    }
    for action, params in valid.items():
        assert module._params(params, action) == params
    assert module._params(
        {"pid": 42, "window_id": 7, "element": 3, "x": 1, "y": 2}, "click"
    ) is None
    assert module._params(
        {"pid": 42, "window_id": 7, "element": 3, "count": 2, "button": "right"}, "click"
    ) is None
    assert module._params(
        {"pid": 42, "window_id": 7, "element": 3, "x": 1, "y": 2, "text": "hello"}, "type_text"
    ) is None
    assert module._params(
        {"scope": "desktop", "pid": 42, "x": 1, "y": 2}, "click"
    ) is None
    assert module._params(
        {"scope": "desktop", "x": 1, "y": 2, "modifiers": ["cmd"]}, "click"
    ) is None
    assert module._params({"x": 1, "y": 2}, "move_cursor") is None
    assert module._params({"file_path": "/tmp/secret"}, "clipboard_write") is None


def test_pc_ui_returns_screenshot_as_bounded_multimodal_content(monkeypatch):
    _configure(monkeypatch)
    image = base64.b64encode(b"window-image").decode("ascii")

    class _ScreenshotResponse:
        content = json.dumps({
            "success": True,
            "result": {
                "pid": 42,
                "window_id": 7,
                "screenshot_b64": image,
                "screenshot_mime_type": "image/png",
            },
        }).encode()

        def json(self):
            return json.loads(self.content)

    class _ScreenshotClient(_Client):
        def post(self, url, **kwargs):
            assert kwargs["json"]["action"] == "ui.snapshot"
            assert kwargs["json"]["params"]["include_screenshot"] is True
            return _ScreenshotResponse()

    monkeypatch.setattr(module.requests, "Session", _ScreenshotClient)
    result = module.pc_ui_tool({"action": "snapshot", "app": "Feishu", "include_screenshot": True})

    assert result["_multimodal"] is True
    assert result["content"][1]["image_url"]["url"] == f"data:image/png;base64,{image}"
    assert "screenshot_b64" not in result["content"][0]["text"]


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
    assert "file_path" not in properties
    assert "cdp" not in properties
    assert "launch" in module.PC_UI_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "list_apps" in module.PC_UI_SCHEMA["parameters"]["properties"]["action"]["enum"]
    assert "list_windows" in module.PC_UI_SCHEMA["parameters"]["properties"]["action"]["enum"]
    for action in module._ACTIONS:
        assert action in module.PC_UI_SCHEMA["parameters"]["properties"]["action"]["enum"]


def test_zet_agent_prompt_requires_snapshot_first_for_local_apps():
    from gateway.platforms.zet_agent import _ZET_WORKDIR_SECTION

    assert "使用 `pc_ui`" in _ZET_WORKDIR_SECTION
    assert "先调用 `list_apps`" in _ZET_WORKDIR_SECTION
    assert "调用 `list_windows`" in _ZET_WORKDIR_SECTION
    assert "最多再检查 3 个" in _ZET_WORKDIR_SECTION
    assert "不要立刻要求用户手工切窗" in _ZET_WORKDIR_SECTION
    assert "不得自动关闭登录、授权、权限、未保存内容" in _ZET_WORKDIR_SECTION
    assert "调用 `pc_node_status`" in _ZET_WORKDIR_SECTION
    assert "`connected=true` 且 `computer_use=false`" in _ZET_WORKDIR_SECTION
    assert "而不是笼统声称当前会话没有工具" in _ZET_WORKDIR_SECTION
    assert "不要改用 terminal" in _ZET_WORKDIR_SECTION
    assert "`snapshot(include_screenshot=true)`" in _ZET_WORKDIR_SECTION
    assert "先 `desktop_snapshot`" in _ZET_WORKDIR_SECTION
    assert "动作后用 `verify`" in _ZET_WORKDIR_SECTION
