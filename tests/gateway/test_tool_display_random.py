from __future__ import annotations

import random

import pytest

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


def test_fallback_args_summary_is_never_structured_and_may_be_empty():
    """D1 兜底：其它工具取第一个非空字符串参数值，没有就空 —— 永不是 JSON。

    AC-1379：兜底原来直接把整个 arguments 丢下去，skill 行于是显示
    `{"file_path":"","name":"command-execution"}`。「做了什么」要么是人读短语、
    要么空着（客户端据此不显示 chip），没有第三种。
    """
    cases = [
        # skill：file_path 为空 ⇒ 落到 name
        ({"file_path": "", "name": "command-execution"}, "command-execution"),
        # 路径型字段与 read_file 同口径取 basename
        ({"file_path": "/skills/a/Skill.md", "name": "x"}, "Skill.md"),
        # 一个字符串参数都没有 ⇒ 空串
        ({"n": 1, "ok": False, "items": [1, 2]}, ""),
        ({}, ""),
        # 只有空白字符串也算没有
        ({"a": "   ", "b": ""}, ""),
        # 敏感键跳过，不因为「第一个字符串」就把 token 摆上屏
        ({"api_key": "sk-live-xxx", "name": "deploy"}, "deploy"),
        ({"password": "pw"}, ""),
    ]
    # 非字典 arguments 同理：字符串原样，其余空着
    assert build_tool_start_display("t", "跑一下构建", None)["display"]["args_summary"] == "跑一下构建"
    for weird in ([1, 2, 3], 42, None, True):
        assert build_tool_start_display("t", weird, None)["display"]["args_summary"] == "", weird
    for arguments, expected in cases:
        summary = build_tool_start_display("some.unknown.tool", arguments, None)["display"]["args_summary"]
        assert summary == expected, (arguments, summary)
        assert not summary.startswith("{"), summary
        assert not summary.startswith("["), summary

    # 随机结构也不许渲染成 JSON
    rng = random.Random(1379)
    for _ in range(50):
        arguments = {
            "flag": rng.choice([True, False]),
            "count": rng.randint(0, 10),
            "nested": {"deep": rng.randint(0, 5)},
            "items": [rng.randint(0, 3) for _ in range(3)],
        }
        summary = build_tool_start_display("another.unknown.tool", arguments, None)["display"]["args_summary"]
        assert summary == "", summary


def test_error_always_uses_error_content_type_and_exact_bytes():
    display = build_tool_result_display({"ignored": True}, error="token=bad")['display']
    assert display["content_type"] == "error"
    assert display["bytes"] == len("token=bad".encode())
    assert "token=bad" not in display["summary"]


def test_truncation_sets_flag_and_keeps_character_boundary():
    display = build_tool_result_display("🙂" * (SUMMARY_MAX_BYTES + 100))["display"]
    assert display["truncated"] is True
    assert display["bytes"] == len(("🙂" * (SUMMARY_MAX_BYTES + 100)).encode())
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
    oversized = Unencodable("token=" + "secret" * (SUMMARY_MAX_BYTES + 1))
    for value in (oversized, {"nested": oversized}, {oversized: "value"}):
        display = result_display(value)
        assert display["truncated"]
        if isinstance(value, str):
            assert display["bytes"] == len(value)
        else:
            assert "bytes" not in display
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
        assert "bytes" not in display


def test_redactor_failure_does_not_disclose_input(monkeypatch):
    from gateway.platforms import tool_display

    def broken(*args, **kwargs):
        raise RuntimeError("redactor failed")

    monkeypatch.setattr(tool_display, "redact_sensitive_text", broken)
    assert result_display("secret material")["summary"] == "Result: [INVALID_TEXT]"


def test_random_short_inputs_use_shared_redactor_and_private_path_policy():
    for seed in range(50):
        rng = random.Random(seed)
        secret = "".join(rng.choice("abcdef0123456789") for _ in range(16))
        for raw in (f'{{"cookie":"{secret}"}}', f'{{"access_token":"{secret}"}}',
                    f'Cookie: sid={secret}', f'/Users/{secret}/document',
                    f'ghp_{secret}', f'token={secret}'):
            display = result_display(raw)
            assert secret not in display["summary"]
            assert display["bytes"] == len(raw.encode())


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
    assert display["truncated"] and len(display["summary"].encode()) <= SUMMARY_MAX_BYTES
    assert "bytes" not in display


def test_mcp_uses_server_identity_and_server_display_name():
    from gateway.platforms.tool_display import source_from_registration

    source = source_from_registration("mcp_search_query", {
        "kind": "mcp", "id": "query", "server": "search_server",
        "server_label": "Search Server", "label": "Query tool",
    })
    assert source == {"kind": "mcp", "id": "search_server", "label": "Search Server"}

@pytest.mark.parametrize(
    ("tool_id", "arguments", "expected"),
    [
        ("read_file", {"path": "/Users/alice/notes.txt"}, "notes.txt"),
        ("terminal", {"command": "git status\nnpm test\necho done"}, "git status + 2"),
        ("browser_navigate", {"url": "https://example.com/a/path?x=1"}, "example.com"),
        ("search_files", {"pattern": "landing page"}, "landing page"),
        ("nas_search", {"query": "海边照片"}, "海边照片"),
    ],
)
def test_args_summary_is_derived_by_tool_type(tool_id, arguments, expected):
    display = build_tool_start_display(tool_id, arguments)["display"]
    assert display == {"args_summary": expected, "truncated": False}


def test_terminal_args_summary_scans_only_display_budget():
    class NoSplitLines(str):
        def splitlines(self, *args, **kwargs):
            raise AssertionError("terminal display must not materialize every line")

    command = NoSplitLines("first\n" + ("short\n" * 100) + "x" * 512)
    display = build_tool_start_display("terminal", {"command": command})["display"]
    assert display == {"args_summary": "[TRUNCATED]", "truncated": True}


@pytest.mark.parametrize(
    ("tool_id", "arguments"),
    [
        ("read_file", {"path": "/Users/alice/token=secret.txt"}),
        ("terminal", {"command": "curl token=secret\ntrue"}),
        ("browser_navigate", {"url": "https://token:secret@example.com/path"}),
        ("search_files", {"pattern": "token=secret"}),
        ("nas_search", {"query": "/Users/alice/document"}),
    ],
)
def test_derived_args_summary_is_redacted(tool_id, arguments):
    display = build_tool_start_display(tool_id, arguments)["display"]
    assert "secret" not in display["args_summary"]
    assert "alice" not in display["args_summary"]
    assert len(display["args_summary"].encode()) <= ARGS_MAX_BYTES
