"""Tests for _normalize_chat_content in the API server adapter."""

from gateway.platforms import api_server
from gateway.platforms.api_server import (
    _extract_business_execution_action,
    _extract_plan_ack,
    _extract_plan_auto_execute,
    _extract_response_mode,
    _extract_turn_id,
    _normalize_chat_content,
    _resolve_plan_auto_execute,
)


class TestExtractBusinessExecutionAction:
    def test_valid_opaque_token_is_preserved(self):
        request = type("Request", (), {"headers": {
            "X-Zettlab-Business-Execution-Action": "a" * 64,
            "X-Zettlab-Business-Execution-Action-Version": "1",
        }})()
        assert _extract_business_execution_action(request) == {
            "action": "a" * 64,
            "action_version": "1",
        }

    def test_malformed_or_header_injected_token_is_dropped(self):
        for action, version in (
            ("short", "1"),
            ("a" * 64 + "\r\nX-Evil: 1", "1"),
            ("A" * 64, "1"),
            ("a" * 64, "2"),
        ):
            request = type("Request", (), {"headers": {
                "X-Zettlab-Business-Execution-Action": action,
                "X-Zettlab-Business-Execution-Action-Version": version,
            }})()
            assert _extract_business_execution_action(request) == {}


class TestExtractResponseMode:
    def test_plan_mode_is_preserved(self):
        assert _extract_response_mode(
            {"metadata": {"response_mode": "plan"}}
        ) == "plan"

    def test_missing_direct_or_unknown_mode_preserves_default(self):
        assert _extract_response_mode({}) == ""
        assert _extract_response_mode(
            {"metadata": {"responseMode": " DIRECT "}}
        ) == ""
        assert _extract_response_mode({"metadata": {"response_mode": "auto"}}) == ""


class TestExtractPlanAck:
    def test_snake_case_cancelled_ack(self):
        assert _extract_plan_ack({
            "metadata": {
                "plan_ack": {
                    "turn_id": "turn-plan-1",
                    "status": "cancelled",
                    "revision_requested": False,
                },
            },
        }) == {
            "turn_id": "turn-plan-1",
            "status": "cancelled",
            "revision_requested": False,
        }

    def test_camel_case_revision_ack(self):
        assert _extract_plan_ack({
            "metadata": {
                "planAck": {
                    "turnId": "turn-plan-2",
                    "status": "cancelled",
                    "revisionRequested": True,
                },
            },
        }) == {
            "turn_id": "turn-plan-2",
            "status": "cancelled",
            "revision_requested": True,
        }

    def test_legacy_ack_uses_metadata_turn_id(self):
        assert _extract_plan_ack({
            "metadata": {
                "turn_id": "legacy-plan-turn",
                "plan_ack": {
                    "status": "confirmed",
                    "revision_requested": False,
                },
            },
        }) == {
            "turn_id": "legacy-plan-turn",
            "status": "confirmed",
            "revision_requested": False,
        }

    def test_released_ack_without_turn_id_preserves_receipt_without_binding(self):
        ack = _extract_plan_ack({
            "metadata": {
                "plan_ack": {
                    "status": "confirmed",
                    "revision_requested": False,
                },
            },
        })

        assert ack == {
            "status": "confirmed",
            "revision_requested": False,
        }
        assert "turn_id" not in ack

    def test_unknown_or_malformed_ack_is_ignored(self):
        assert _extract_plan_ack({"metadata": {"plan_ack": "cancelled"}}) == {}
        assert _extract_plan_ack({"metadata": {"plan_ack": {"status": "other"}}}) == {}
        assert _extract_plan_ack({
            "metadata": {"plan_ack": {"status": "confirmed", "turn_id": ""}},
        }) == {}


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

    def test_unparseable_value_returns_none_not_auto(self):
        # 字段存在但值无法解析（null / 空串 / 未知字符串）→ None（回落到
        # env/default manual），绝不能被提成 auto，否则绕过 capability opt-in。
        for bad in (None, "", "  ", "maybe", "yesnt", "1.5x", [], {}, object()):
            assert (
                _extract_plan_auto_execute({"metadata": {"plan_auto_execute": bad}})
                is None
            ), bad

    def test_numeric_optin_still_parses(self):
        # 明确的数值型 bool 仍算显式 opt-in（1 → True，0 → False）。
        assert _extract_plan_auto_execute({"metadata": {"plan_auto_execute": 1}}) is True
        assert _extract_plan_auto_execute({"metadata": {"plan_auto_execute": 0}}) is False

    def test_unparseable_value_flows_to_manual_default(self, monkeypatch):
        # 端到端：坏值 → extract None → resolve 默认 manual（无 env）。
        monkeypatch.delenv("HERMES_ZET_AGENT_PLAN_AUTO_EXECUTE", raising=False)
        extracted = _extract_plan_auto_execute({"metadata": {"plan_auto_execute": None}})
        assert _resolve_plan_auto_execute(extracted) is False


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


    def test_text_content_part(self):
        content = [{"type": "text", "text": "hello"}]
        assert _normalize_chat_content(content) == "hello"

    def test_input_text_content_part(self):
        content = [{"type": "input_text", "text": "user input"}]
        assert _normalize_chat_content(content) == "user input"

    def test_output_text_content_part(self):
        content = [{"type": "output_text", "text": "assistant output"}]
        assert _normalize_chat_content(content) == "assistant output"


    def test_empty_text_parts_filtered(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "actual"},
            {"type": "text", "text": ""},
        ]
        assert _normalize_chat_content(content) == "actual"

