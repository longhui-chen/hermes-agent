"""Camera pre-dispatch failures must not masquerade as device grant failures."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import zet_agent_response_mode as response_mode
from agent import tool_executor


@pytest.fixture(autouse=True)
def registered_camera_command(monkeypatch):
    # Installed signed helper discovery is an external packaging boundary.
    # Keep the actual scope guard and execution middleware unmocked.
    monkeypatch.setattr(
        response_mode, "_camera_runtime_argv",
        lambda args: ["python3", "camera_connector.py", "list"]
        if args.get("command") == "python3 camera_connector.py list" else None,
    )


def test_camera_scope_failure_has_stable_reason_without_claiming_permission():
    blocked = response_mode.trusted_skill_operation_block_message(
        SimpleNamespace(), function_name="terminal",
        function_args={"command": "python3 camera_connector.py list"},
    )
    assert isinstance(blocked, str)
    assert blocked.code == "camera_task_scope_missing"
    assert blocked.authorization_status == "not_checked"
    assert "Do not claim" in blocked


def test_camera_scope_error_survives_real_execution_middleware(monkeypatch):
    post = Mock()
    execute = Mock(side_effect=AssertionError("must not dispatch camera"))
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", post)
    set_halt = Mock()
    agent = SimpleNamespace(
        platform="zet_agent", session_id="camera-session",
        _set_tool_guardrail_halt=set_halt,
    )
    outcome = tool_executor._run_agent_tool_execution_middleware(
        agent, function_name="terminal",
        function_args={"command": "python3 camera_connector.py list"},
        effective_task_id="camera-task", tool_call_id="camera-call",
        execute=execute,
    )
    result = json.loads(outcome.result)
    assert outcome.blocked
    assert result["code"] == "camera_task_scope_missing"
    assert result["authorization_status"] == "not_checked"
    assert isinstance(result["error"], str)
    execute.assert_not_called()
    set_halt.assert_called_once()
    assert set_halt.call_args.args[0].code == "camera_task_scope_missing"
    # Preserve the existing event contract while making the tool payload precise.
    assert post.call_args.kwargs["error_type"] == "zet_agent_plan_mode_block"


def test_other_policy_blocks_do_not_claim_camera_scope_failure(monkeypatch):
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", Mock())
    execute = Mock(side_effect=AssertionError("must not dispatch"))
    outcome = tool_executor._run_agent_tool_execution_middleware(
        SimpleNamespace(platform="zet_agent"), function_name="terminal",
        function_args={"command": "python3 camera_connector.py list"},
        effective_task_id="camera-task", tool_call_id="camera-call",
        scope_block="Tool is not available in this scope", execute=execute,
    )
    assert json.loads(outcome.result) == {
        "error": "Tool is not available in this scope",
    }
    execute.assert_not_called()
