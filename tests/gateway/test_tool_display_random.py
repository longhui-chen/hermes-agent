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


def test_oversized_atoms_are_rejected_before_redaction_or_encoding(monkeypatch):
    from gateway.platforms import tool_display

    class Unencodable(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("oversized input must not be encoded")

    original = tool_display.redact_sensitive_text
    inspected = []

    def checked(text, **kwargs):
        assert len(text) <= SUMMARY_MAX_BYTES
        assert kwargs == {"force": True}
        inspected.append(text)
        return original(text, **kwargs)

    monkeypatch.setattr(tool_display, "redact_sensitive_text", checked)
    oversized = Unencodable("secret" * (SUMMARY_MAX_BYTES + 1))
    for value in (oversized, {"nested": oversized}, {oversized: "value"}):
        display = result_display(value)
        assert display["truncated"]
        assert display["bytes"] == len(display["summary"].encode())
        assert "secret" not in display["summary"]
    assert all(len(text) <= SUMMARY_MAX_BYTES for text in inspected)


def test_shared_container_limit_bounds_total_iteration():
    visited = 0

    class CountingList(list):
        def __iter__(self):
            nonlocal visited
            for item in super().__iter__():
                visited += 1
                assert visited <= 272
                yield item

    tree = CountingList(["x"] * 256)
    for _ in range(8):
        tree = CountingList([tree] * 256)
    display = result_display(tree)
    assert display["truncated"]
    assert visited <= 272


def test_cycles_depth_and_unsupported_values_fail_closed():
    cycle = []
    cycle.append(cycle)
    deep = "text"
    for _ in range(30):
        deep = [deep]

    class Dangerous:
        def __str__(self):
            raise AssertionError("must not stringify unknown objects")

    for value in (cycle, deep, Dangerous(), 1 << 100000):
        display = result_display(value)
        assert display["truncated"]
        assert display["bytes"] <= SUMMARY_MAX_BYTES


def test_redactor_failure_does_not_disclose_input(monkeypatch):
    from gateway.platforms import tool_display

    def broken(*args, **kwargs):
        raise RuntimeError("redactor failed")

    monkeypatch.setattr(tool_display, "redact_sensitive_text", broken)
    assert result_display("secret material")["summary"] == "[INVALID_TEXT]"


def test_random_short_inputs_use_shared_redactor_and_private_path_policy():
    for seed in range(50):
        rng = random.Random(seed)
        secret = "".join(rng.choice("abcdef0123456789") for _ in range(16))
        for raw in (f'{{"cookie":"{secret}"}}', f'{{"access_token":"{secret}"}}',
                    f'Cookie: sid={secret}', f'/Users/{secret}/document',
                    f'ghp_{secret}', f'token={secret}'):
            display = result_display(raw)
            assert secret not in display["summary"]
            assert display["bytes"] == len(display["summary"].encode())


def test_random_utf8_character_boundaries_and_preprocessing_truncation():
    for seed in range(100):
        rng = random.Random(seed)
        text = "".join(rng.choice("a中🙂") for _ in range(rng.randrange(128, 400)))
        display = build_tool_start_display("x", text)["display"]
        summary = display["args_summary"]
        assert len(summary.encode()) <= ARGS_MAX_BYTES
        assert display["truncated"] == (len(text.encode()) > ARGS_MAX_BYTES)
    # Redaction/omission can shrink the output below the cap; lost input still
    # requires truncated=true rather than guessing from the final byte length.
    display = result_display({"safe": "x" * (SUMMARY_MAX_BYTES + 1)})
    assert display["truncated"] and display["bytes"] < SUMMARY_MAX_BYTES


def test_mcp_uses_server_identity_and_server_display_name():
    from gateway.platforms.tool_display import source_from_registration

    source = source_from_registration("mcp_search_query", {
        "kind": "mcp", "id": "query", "server": "search_server",
        "server_label": "Search Server", "label": "Query tool",
    })
    assert source == {"kind": "mcp", "id": "search_server", "label": "Search Server"}
