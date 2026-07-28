"""Flow coverage for trusted turn context reaching the native install tool."""

import json

import model_tools
from tools import skillhub_install_tool


def test_runtime_forwards_user_turn_context_to_skillhub_install(monkeypatch):
    observed = {}

    def capture(identifier, **kwargs):
        observed["identifier"] = identifier
        observed.update(kwargs)
        return json.dumps({"status": "captured"})

    monkeypatch.setattr(skillhub_install_tool, "skillhub_install", capture)

    result = model_tools.handle_function_call(
        "skillhub_install",
        {"identifier": "owner/repo/example"},
        task_id="task-1",
        session_id="session-1",
        turn_id="turn-2",
        tool_call_id="call-1",
        user_task="确认安装",
        previous_assistant_message="候选是 owner/repo/example，是否安装？",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )

    assert json.loads(result) == {"status": "captured"}
    assert observed == {
        "identifier": "owner/repo/example",
        "session_id": "session-1",
        "turn_id": "turn-2",
        "user_message": "确认安装",
        "previous_assistant_message": "候选是 owner/repo/example，是否安装？",
    }
