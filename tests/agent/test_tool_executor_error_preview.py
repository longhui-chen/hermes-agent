import json

from agent.tool_executor import _tool_error_log_preview


def test_generic_tool_error_keeps_existing_head_only_behavior():
    result = "head-" + ("x" * 220) + "\nRuntimeError: generic tail"

    preview = _tool_error_log_preview("terminal", result)

    assert preview == result[:200]
    assert "generic tail" not in preview


def test_trusted_video_worker_error_includes_bounded_traceback_tail():
    result = json.dumps(
        {
            "output": (
                "Traceback (most recent call last):\n"
                + ('  File "/trusted/worker.py", line 218, in _execute\n' * 8)
                +
                "PermissionError: operation denied"
            ),
            "exit_code": 1,
            "error": None,
            "video_edit_runtime_direct": True,
        }
    )

    preview = _tool_error_log_preview("terminal", result)

    assert preview.startswith(result[:200])
    assert preview.endswith("tail: PermissionError: operation denied")


def test_trusted_video_worker_tail_is_bounded():
    tail = "RuntimeError: " + ("y" * 300)
    result = json.dumps(
        {
            "output": f"Traceback\n{tail}",
            "exit_code": 1,
            "video_edit_runtime_direct": True,
        }
    )

    preview = _tool_error_log_preview("terminal", result, max_chars=40)

    assert preview == f"{result[:40]} | tail: {tail[:40]}"
