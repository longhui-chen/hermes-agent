import json

from agent.tool_executor import _tool_error_log_preview


def test_generic_tool_error_keeps_existing_head_only_behavior():
    result = "head-" + ("x" * 220) + "\nRuntimeError: generic tail"

    preview = _tool_error_log_preview("terminal", result)

    assert preview == result[:200]
    assert "generic tail" not in preview
