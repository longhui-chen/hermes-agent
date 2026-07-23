from __future__ import annotations

import base64
import json
import re

from tests.plugins.test_creation_governor_plugin import _load_plugin


def _candidate() -> dict[str, object]:
    return {
        "decision": "agent",
        "suggested_name": "广告分析助手",
        "reason": "后续判断需要保留投放背景",
        "evidence_turn_ids": ["turn-1"],
        "confidence": 0.9,
        "dedup_key": "ads-analyst",
        "proposal_text": "要创建广告分析助手吗？",
    }


def _decode_envelope(text: str) -> dict[str, object]:
    match = re.search(r"<!--creation-recommendation:start ([A-Za-z0-9_-]+)-->", text)
    assert match is not None
    encoded = match.group(1)
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))


def _action(payload: dict[str, object], *, proposal_id: str | None = None) -> str:
    response = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": "create",
        "proposal_id": proposal_id
        if proposal_id is not None
        else payload["proposal_id"],
        "creation_type": payload["creation_type"],
        "title": payload["title"],
        "dedup_key": payload["dedup_key"],
        "evidence_turn_ids": payload["evidence_turn_ids"],
    }
    return (
        "[creation_recommendation_response]\n"
        f"{json.dumps(response, ensure_ascii=False)}\n"
        "[/creation_recommendation_response]"
    )


def _show_card(plugin, session_id: str) -> dict[str, object]:
    plugin._on_pre_llm_call(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="turn-1",
        user_message="分析近期广告效果",
        conversation_history=[],
    )
    result = plugin._detect_creation_opportunity(
        _candidate(),
        session_id=session_id,
        sender_id="owner-a",
    )
    assert json.loads(result)["status"] == "proposal_ready"
    transformed = plugin._transform_llm_output(
        session_id=session_id,
        sender_id="owner-a",
        response_text="分析完成。",
        completed=True,
        failed=False,
    )
    assert transformed is not None
    return _decode_envelope(transformed)


def test_recommendation_envelope_has_a_proposal_id_bound_to_the_current_card():
    plugin = _load_plugin()

    payload = _show_card(plugin, "bound-card")

    assert payload["proposal_id"]
    assert payload["source_turn_id"] == "turn-1"


def test_structured_create_requires_the_current_proposal_id_and_owner():
    plugin = _load_plugin()
    payload = _show_card(plugin, "bound-action")

    stale = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )
    assert stale is None

    wrong_owner = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-b",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert wrong_owner is None

    accepted = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert accepted is not None
    assert "native creation flow" in accepted["context"]

    replay = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert replay is None


def test_recommendations_fail_closed_for_unsupported_or_failed_turns():
    plugin = _load_plugin()

    assert (
        plugin._on_pre_llm_call(
            session_id="one-shot",
            platform="cli",
            supports_followup_turns=False,
            user_message="分析广告效果",
            conversation_history=[],
        )
        is None
    )
    blocked = json.loads(
        plugin._detect_creation_opportunity(
            _candidate(),
            session_id="one-shot",
            platform="cli",
            supports_followup_turns=False,
        )
    )
    assert blocked == {"status": "not_proposed", "reason": "unsupported_runtime"}

    _show_card(plugin, "failed-turn")
    assert (
        plugin._transform_llm_output(
            session_id="failed-turn",
            sender_id="owner-a",
            response_text="部分结果",
            completed=False,
            failed=True,
        )
        is None
    )
    state_key = plugin._session_key(
        {"session_id": "failed-turn", "sender_id": "owner-a"}
    )
    assert plugin._session_states[state_key]["last_proposal"] is None


def test_explicit_creation_and_expired_cards_cannot_enter_recommendation_flow():
    plugin = _load_plugin()

    assert (
        plugin._on_pre_llm_call(
            session_id="native-create",
            sender_id="owner-a",
            user_message="请创建一个广告分析 Agent",
            conversation_history=[],
        )
        is None
    )

    payload = _show_card(plugin, "expired-card")
    state_key = plugin._session_key(
        {"session_id": "expired-card", "sender_id": "owner-a"}
    )
    plugin._session_states[state_key]["last_proposal"]["expires_at"] = 0

    assert (
        plugin._on_pre_llm_call(
            session_id="expired-card",
            sender_id="owner-a",
            user_message=_action(payload),
            conversation_history=[],
        )
        is None
    )
