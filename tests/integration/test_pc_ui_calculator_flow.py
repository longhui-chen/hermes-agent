import json

import tools.pc_ui_tool as module


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.content = json.dumps(payload).encode()

    def json(self):
        return self._payload


def test_schema_visible_context_survives_snapshot_to_calculator_mutation(monkeypatch):
    calls = []

    class _Client:
        trust_env = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def post(self, _url, **kwargs):
            calls.append(kwargs["json"])
            if kwargs["json"]["action"] == "ui.snapshot":
                return _Response({
                    "success": True,
                    "result": {
                        "pid": 42,
                        "window_id": 7,
                        "snapshot_revision": 4,
                        "user_input_epoch": 0,
                    },
                })
            return _Response({"success": True, "result": {"verified": True}})

    secrets = {
        "ZETTLAB_BROWSER_ACTION_URL": "http://127.0.0.1:8080/api/v1/internal/browser/action",
        "ZETTLAB_AGENT_ACTION_TOKEN": "action-token",
    }
    monkeypatch.setattr(module, "get_secret", lambda name, default="": secrets.get(name, default))
    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": (
            "zettlab:alice:agent-a:calculator" if name == "HERMES_SESSION_KEY" else default
        ),
    )
    monkeypatch.setattr(module, "zettlab_browser_session_token", lambda: "session-token")
    monkeypatch.setattr(module.requests, "Session", _Client)

    snapshot = json.loads(module.pc_ui_tool(
        {
            "action": "snapshot",
            "app": "Calculator",
            "pid": 42,
            "window_id": 7,
            "include_text": True,
        },
        task_id="turn-calculator",
        tool_call_id="call-snapshot",
    ))
    assert snapshot["success"] is True

    mutation = json.loads(module.pc_ui_tool(
        {
            "action": "invoke",
            "app": "Calculator",
            "pid": 42,
            "window_id": 7,
            "element": 23,
            "snapshot_revision": snapshot["result"]["snapshot_revision"],
            "user_input_epoch": snapshot["result"]["user_input_epoch"],
            "postcondition": [{"element": {"value_equals": "17"}}],
        },
        task_id="turn-calculator",
        tool_call_id="call-invoke-17",
    ))
    assert mutation["success"] is True

    assert calls[0]["params"] == {
        "app": "Calculator",
        "pid": 42,
        "window_id": 7,
    }
    assert calls[1]["params"] == {"pid": 42, "window_id": 7, "element": 23}
    assert calls[1]["task_control"]["snapshot_revision"] == 4
    assert calls[1]["task_control"]["user_input_epoch"] == 0
    assert calls[1]["task_control"]["postcondition"] == [
        {"element": {"value_equals": "17"}}
    ]
