from __future__ import annotations

import random

from gateway.platforms.tool_display import ARGS_MAX_BYTES, SUMMARY_MAX_BYTES, build_tool_result_display, build_tool_start_display, result_display


def test_random_redaction_and_utf8_bounds():
    secrets = ["token=abc123", "Cookie: sid=secret", "/Users/alice/private.txt", "普通话🙂"]
    for seed in range(40):
        rng = random.Random(seed)
        raw = " ".join(rng.choice(secrets) for _ in range(500))
        start = build_tool_start_display("mcp.search", {"query": raw}, {"kind": "mcp", "server": "search", "label": "Search"})
        result = build_tool_result_display(raw)
        assert len(start["display"]["args_summary"].encode()) <= ARGS_MAX_BYTES
        assert len(result["display"]["summary"].encode()) <= SUMMARY_MAX_BYTES
        assert "token=abc123" not in start["display"]["args_summary"]
        assert "/Users/alice" not in result["display"]["summary"]


def test_error_always_uses_error_content_type_and_exact_bytes():
    display = build_tool_result_display({"ignored": True}, error="token=bad")['display']
    assert display["content_type"] == "error"
    assert display["bytes"] == len(display["summary"].encode())
    assert "token=bad" not in display["summary"]


def test_truncation_sets_flag_and_keeps_character_boundary():
    display = build_tool_result_display("🙂" * (SUMMARY_MAX_BYTES + 100))["display"]
    assert display["truncated"] is True
    assert display["bytes"] <= SUMMARY_MAX_BYTES
    display["summary"].encode("utf-8")


def test_structured_secret_keys_are_redacted_before_serialization():
    display = build_tool_start_display("x", {"api_key": "secret", "nested": {"password": "pw"}})["display"]
    assert "secret" not in display["args_summary"]
    assert "pw" not in display["args_summary"]


def test_zero_false_and_invalid_unicode_are_safe_display_values():
    assert result_display(0)["summary"] == "0"
    assert result_display(False)["summary"] == "False"
    assert "INVALID_TEXT" in result_display("bad\ud800")["summary"]


def test_string_json_sensitive_keys_are_redacted():
    display = build_tool_start_display("x", '{"api_key":"secret","ok":true}')['display']
    assert "secret" not in display["args_summary"]
