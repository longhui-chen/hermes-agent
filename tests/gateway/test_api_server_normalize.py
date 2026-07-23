"""Tests for _normalize_chat_content in the API server adapter."""

from gateway.platforms import api_server
from gateway.platforms.api_server import (
    _extract_plan_ack,
    _extract_plan_auto_execute,
    _extract_turn_id,
    _normalize_chat_content,
    _resolve_plan_auto_execute,
)


class TestExtractPlanAck:
    def test_snake_case_cancelled_ack(self):
        assert _extract_plan_ack({
            "metadata": {
                "plan_ack": {
                    "status": "cancelled",
                    "revision_requested": False,
                },
            },
        }) == {"status": "cancelled", "revision_requested": False}

    def test_camel_case_revision_ack(self):
        assert _extract_plan_ack({
            "metadata": {
                "planAck": {
                    "status": "cancelled",
                    "revisionRequested": True,
                },
            },
        }) == {"status": "cancelled", "revision_requested": True}

    def test_unknown_or_malformed_ack_is_ignored(self):
        assert _extract_plan_ack({"metadata": {"plan_ack": "cancelled"}}) == {}
        assert _extract_plan_ack({"metadata": {"plan_ack": {"status": "other"}}}) == {}


class TestExtractPlanAutoExecute:
    def test_absent_returns_none(self):
        assert _extract_plan_auto_execute({}) is None
        assert _extract_plan_auto_execute({"metadata": {}}) is None

    def test_explicit_bool(self):
        assert _extract_plan_auto_execute({"metadata": {"plan_auto_execute": True}}) is True
        assert _extract_plan_auto_execute({"metadata": {"plan_auto_execute": False}}) is False

    def test_camel_case_and_string(self):
        assert _extract_plan_auto_execute({"metadata": {"planAutoExecute": "false"}}) is False
        assert _extract_plan_auto_execute({"metadata": {"planAutoExecute": "true"}}) is True


class TestResolvePlanAutoExecute:
    def test_default_is_manual(self, monkeypatch):
        # 默认 manual（capability negotiation）：未 opt-in（meta None）+ 无 env → 不 auto。
        monkeypatch.delenv("HERMES_ZET_AGENT_PLAN_AUTO_EXECUTE", raising=False)
        assert _resolve_plan_auto_execute(None) is False

    def test_meta_override_beats_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_ZET_AGENT_PLAN_AUTO_EXECUTE", raising=False)
        assert _resolve_plan_auto_execute(False) is False
        assert _resolve_plan_auto_execute(True) is True

    def test_env_opt_in_and_kill_switch(self, monkeypatch):
        # env 可全局 opt-in auto（"1"）或强制 manual（"0"）；per-turn meta 仍优先。
        monkeypatch.setenv("HERMES_ZET_AGENT_PLAN_AUTO_EXECUTE", "1")
        assert _resolve_plan_auto_execute(None) is True
        monkeypatch.setenv("HERMES_ZET_AGENT_PLAN_AUTO_EXECUTE", "0")
        assert _resolve_plan_auto_execute(None) is False
        assert _resolve_plan_auto_execute(True) is True


class TestExtractTurnId:
    """metadata.turn_id is plumbed to the NAS agent-search fallback header."""

    def test_snake_case_turn_id(self):
        assert _extract_turn_id({"metadata": {"turn_id": "t_abc-123"}}) == "t_abc-123"

    def test_camel_case_turn_id(self):
        assert _extract_turn_id({"metadata": {"turnId": "t_xyz"}}) == "t_xyz"

    def test_missing_metadata_returns_empty(self):
        assert _extract_turn_id({}) == ""

    def test_metadata_not_dict_returns_empty(self):
        assert _extract_turn_id({"metadata": "nope"}) == ""

    def test_missing_turn_id_returns_empty(self):
        assert _extract_turn_id({"metadata": {"response_mode": "plan"}}) == ""

    def test_surrounding_whitespace_stripped(self):
        assert _extract_turn_id({"metadata": {"turn_id": "  t_abc  "}}) == "t_abc"

    def test_crlf_injection_dropped(self):
        # The value lands in an HTTP header; CR/LF (and any header-unsafe byte)
        # must be rejected outright rather than forwarded.
        assert _extract_turn_id({"metadata": {"turn_id": "t_a\r\nX-Evil: 1"}}) == ""

    def test_internal_whitespace_dropped(self):
        assert _extract_turn_id({"metadata": {"turn_id": "t_a b"}}) == ""

    def test_uuid_form_preserved(self):
        tid = "t_550e8400-e29b-41d4-a716-446655440000"
        assert _extract_turn_id({"metadata": {"turn_id": tid}}) == tid


class TestNormalizeChatContent:
    """Content normalization converts array-based content parts to plain text."""

    def test_none_returns_empty_string(self):
        assert _normalize_chat_content(None) == ""

    def test_plain_string_returned_as_is(self):
        assert _normalize_chat_content("hello world") == "hello world"

    def test_empty_string_returned_as_is(self):
        assert _normalize_chat_content("") == ""

    def test_text_content_part(self):
        content = [{"type": "text", "text": "hello"}]
        assert _normalize_chat_content(content) == "hello"

    def test_input_text_content_part(self):
        content = [{"type": "input_text", "text": "user input"}]
        assert _normalize_chat_content(content) == "user input"

    def test_output_text_content_part(self):
        content = [{"type": "output_text", "text": "assistant output"}]
        assert _normalize_chat_content(content) == "assistant output"

    def test_multiple_text_parts_joined_with_newline(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        assert _normalize_chat_content(content) == "first\nsecond"

    def test_mixed_string_and_dict_parts(self):
        content = ["plain string", {"type": "text", "text": "dict part"}]
        assert _normalize_chat_content(content) == "plain string\ndict part"

    def test_image_url_parts_silently_skipped(self):
        content = [
            {"type": "text", "text": "check this:"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        assert _normalize_chat_content(content) == "check this:"

    def test_integer_content_converted(self):
        assert _normalize_chat_content(42) == "42"

    def test_boolean_content_converted(self):
        assert _normalize_chat_content(True) == "True"

    def test_deeply_nested_list_respects_depth_limit(self):
        """Nesting beyond max_depth returns empty string."""
        content = [[[[[[[[[[[["deep"]]]]]]]]]]]]
        result = _normalize_chat_content(content)
        # The deep nesting should be truncated, not crash
        assert isinstance(result, str)

    def test_large_list_capped(self):
        """Lists beyond MAX_CONTENT_LIST_SIZE are truncated."""
        content = [{"type": "text", "text": f"item{i}"} for i in range(2000)]
        result = _normalize_chat_content(content)
        # Should not contain all 2000 items
        assert result.count("item") <= 1000

    def test_oversized_string_truncated(self):
        """Strings beyond 64KB are truncated."""
        huge = "x" * 100_000
        result = _normalize_chat_content(huge)
        assert len(result) == 65_536

    def test_empty_text_parts_filtered(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "actual"},
            {"type": "text", "text": ""},
        ]
        assert _normalize_chat_content(content) == "actual"

    def test_dict_without_type_skipped(self):
        content = [{"foo": "bar"}, {"type": "text", "text": "real"}]
        assert _normalize_chat_content(content) == "real"

    def test_empty_list_returns_empty(self):
        assert _normalize_chat_content([]) == ""

    def test_many_small_parts_normalize_without_quadratic_rescan(self, monkeypatch):
        """Large content arrays should normalize in linear time."""
        content = [{"type": "text", "text": "x"} for _ in range(1000)]
        sum_calls = 0

        def counting_sum(values):
            nonlocal sum_calls
            sum_calls += 1
            return sum(values)

        monkeypatch.setattr(api_server, "sum", counting_sum, raising=False)
        result = _normalize_chat_content(content)

        assert result.count("x") == 1000
        assert sum_calls == 0
