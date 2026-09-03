"""JSON tool results are judged by their own verdict fields, never by a
substring scan: a healthy status payload carrying "failed": false must not be
counted as a failure (09-03: create_pipeline tripped the same-tool loop guard
on every turn), and an explicit ok:false must be."""

import json

from agent.display import _detect_tool_failure
from agent.tool_guardrails import classify_tool_failure


STATUS_OK = json.dumps({"ok": True, "run_id": "pipe-app-1", "next": "…",
                        "status": {"run_id": "pipe-app-1", "step": "guide", "done": False, "failed": False,
                                   "last_error": "应用名要小写字母开头"}})
REJECTED = json.dumps({"ok": False, "success": False, "code": "spec_rejected_final",
                       "problems": [{"code": "req_shape", "msg": "x" * 600}]})


def test_ok_true_with_failed_false_inside_is_not_a_failure():
    for fn in (_detect_tool_failure, classify_tool_failure):
        failed, _ = fn("create_pipeline", STATUS_OK)
        assert failed is False


def test_ok_false_is_a_failure_even_when_failed_word_is_out_of_window():
    for fn in (_detect_tool_failure, classify_tool_failure):
        failed, _ = fn("create_pipeline", REJECTED)
        assert failed is True


def test_plain_json_without_verdict_or_error_is_success():
    for fn in (_detect_tool_failure, classify_tool_failure):
        assert fn("read_file", json.dumps({"content": "1|package main", "total_lines": 1})) == (False, "")


def test_error_field_still_counts():
    for fn in (_detect_tool_failure, classify_tool_failure):
        failed, _ = fn("read_file", json.dumps({"content": "", "error": "Binary file - cannot display as text"}))
        assert failed is True


def test_text_results_keep_heuristic():
    for fn in (_detect_tool_failure, classify_tool_failure):
        assert fn("web_search", "Error: timeout")[0] is True
        assert fn("web_search", "found 3 results")[0] is False
