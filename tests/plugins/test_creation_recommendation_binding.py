from __future__ import annotations

import base64
import json
import re

import pytest

from tests.plugins.test_creation_governor_plugin import _Context, _FakeLlm, _load_plugin


RECEIPT_TRANSPORT = "canonical_final_v1"


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


def _decode_action_result(text: str) -> dict[str, object]:
    match = re.search(
        r"<!--creation-recommendation-action-result ([A-Za-z0-9_-]+)-->", text
    )
    assert match is not None
    encoded = match.group(1)
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))


def _action(
    payload: dict[str, object],
    *,
    action: str = "create",
    proposal_id: str | None = None,
) -> str:
    response = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": action,
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


def _show_card(
    plugin,
    session_id: str,
    creation_type: str = "agent",
    receipt_transport: str = RECEIPT_TRANSPORT,
) -> dict[str, object]:
    plugin._on_pre_llm_call(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="turn-1",
        user_message="分析近期广告效果",
        conversation_history=[],
        creation_action_receipt_transport=receipt_transport,
    )
    candidate = _candidate()
    candidate["decision"] = creation_type
    candidate["dedup_key"] = f"{creation_type}:ads-analyst"
    result = plugin._detect_creation_opportunity(
        candidate,
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
        creation_action_receipt_transport=receipt_transport,
    )
    assert transformed is not None
    return _decode_envelope(transformed)


def test_recommendation_envelope_has_a_proposal_id_bound_to_the_current_card():
    plugin = _load_plugin()

    payload = _show_card(plugin, "bound-card")

    assert payload["proposal_id"]
    assert payload["source_turn_id"] == "turn-1"
    assert payload["action_receipts"] is True


@pytest.mark.parametrize("receipt_transport", ["", "canonical_final_v2"])
def test_legacy_or_unknown_transport_omits_receipt_capability(receipt_transport: str):
    plugin = _load_plugin()

    payload = _show_card(
        plugin,
        f"legacy-card-{receipt_transport}",
        receipt_transport=receipt_transport,
    )

    assert "action_receipts" not in payload


def test_receipt_transport_capability_does_not_leak_to_the_next_turn():
    plugin = _load_plugin()

    capable = _show_card(plugin, "capable-card")
    assert plugin._invocation_scope.get() is None
    legacy = _show_card(plugin, "legacy-after-capable", receipt_transport="")

    assert capable["action_receipts"] is True
    assert "action_receipts" not in legacy
    assert plugin._invocation_scope.get() is None


@pytest.mark.parametrize("receipt_transport", ["", "canonical_final_v2"])
def test_capable_card_action_fails_closed_after_transport_downgrade(
    receipt_transport: str, monkeypatch
):
    plugin = _load_plugin()
    payload = _show_card(plugin, f"downgraded-action-{receipt_transport}")
    monkeypatch.setattr(
        plugin,
        "_set_session_muted",
        lambda *_args: pytest.fail("a downgraded action must not mutate preference"),
    )

    result = plugin._on_pre_llm_call(
        session_id=f"downgraded-action-{receipt_transport}",
        sender_id="owner-a",
        turn_id="downgraded-turn",
        user_message=_action(payload, action="mute_session"),
        conversation_history=[],
        creation_action_receipt_transport=receipt_transport,
    )

    assert "invalid or expired" in result["context"]
    state_key = plugin._session_key({
        "session_id": f"downgraded-action-{receipt_transport}",
        "sender_id": "owner-a",
    })
    state = plugin._session_states[state_key]
    assert state["last_proposal"]["proposal_id"] == payload["proposal_id"]
    assert not state["pending_action_results"]


def test_native_creation_routes_are_explicit_for_each_recommendation_type():
    plugin = _load_plugin()

    assert "agent-creator" in plugin._native_creation_route("agent")
    assert "skill_manage" in plugin._native_creation_route("skill")
    assert "cronjob" in plugin._native_creation_route("task")


def test_structured_create_requires_the_current_proposal_id_and_owner():
    plugin = _load_plugin()
    payload = _show_card(plugin, "bound-action")

    stale = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        turn_id="stale-turn",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )
    assert stale is not None
    assert "invalid or expired" in stale["context"]

    wrong_owner = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-b",
        turn_id="wrong-owner-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert wrong_owner is not None
    assert "invalid or expired" in wrong_owner["context"]

    accepted = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        turn_id="accepted-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert accepted is not None
    assert "agent-creator" in accepted["context"]

    replay = plugin._on_pre_llm_call(
        session_id="bound-action",
        sender_id="owner-a",
        turn_id="replay-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert replay is not None
    assert "invalid or expired" in replay["context"]


@pytest.mark.parametrize(
    ("response_text", "completed", "failed", "interrupted"),
    [
        ("The native flow stopped before completion.", False, True, False),
        ("", True, False, False),
        ("", False, False, True),
        ("The native flow is incomplete.", False, False, False),
    ],
    ids=["failed", "empty", "interrupted", "incomplete"],
)
def test_validated_create_handoff_stays_accepted_for_every_terminal_shape(
    response_text: str,
    completed: bool,
    failed: bool,
    interrupted: bool,
) -> None:
    plugin = _load_plugin()
    session_id = f"retry-create-{response_text}-{completed}-{failed}-{interrupted}"
    payload = _show_card(plugin, session_id)

    accepted = plugin._on_pre_llm_call(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="create-attempt-1",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "agent-creator" in accepted["context"]
    output = plugin._transform_llm_output(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="create-attempt-1",
        response_text=response_text,
        completed=completed,
        failed=failed,
        interrupted=interrupted,
    )
    assert _decode_action_result(output) == {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": payload["proposal_id"],
        "action": "create",
        "status": "accepted",
    }

    replay = plugin._on_pre_llm_call(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="create-attempt-2",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "invalid or expired" in replay["context"]


def test_normal_native_flow_turn_closes_the_card_with_an_accepted_receipt():
    plugin = _load_plugin()
    payload = _show_card(plugin, "successful-create")

    accepted = plugin._on_pre_llm_call(
        session_id="successful-create",
        sender_id="owner-a",
        turn_id="create-attempt",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "agent-creator" in accepted["context"]
    output = plugin._transform_llm_output(
        session_id="successful-create",
        sender_id="owner-a",
        turn_id="create-attempt",
        response_text="Which data sources should this Agent be allowed to use?",
        completed=True,
        failed=False,
    )
    assert _decode_action_result(output)["status"] == "accepted"

    replay = plugin._on_pre_llm_call(
        session_id="successful-create",
        sender_id="owner-a",
        turn_id="replay-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "invalid or expired" in replay["context"]


def test_rejected_stale_create_does_not_clear_the_current_card():
    plugin = _load_plugin()
    payload = _show_card(plugin, "stale-create-keeps-card")

    rejected = plugin._on_pre_llm_call(
        session_id="stale-create-keeps-card",
        sender_id="owner-a",
        turn_id="stale-attempt",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )
    assert "invalid or expired" in rejected["context"]

    output = plugin._transform_llm_output(
        session_id="stale-create-keeps-card",
        sender_id="owner-a",
        turn_id="stale-attempt",
        response_text="That card is no longer available.",
        completed=True,
        failed=False,
    )
    assert _decode_action_result(output) == {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": "stale-proposal",
        "action": "create",
        "status": "rejected",
        "reason_code": "proposal_not_actionable",
    }

    retry = plugin._on_pre_llm_call(
        session_id="stale-create-keeps-card",
        sender_id="owner-a",
        turn_id="valid-attempt",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "agent-creator" in retry["context"]


def test_malformed_structured_wrapper_never_falls_through_to_legacy_acceptance():
    plugin = _load_plugin()
    payload = _show_card(plugin, "malformed-wrapper")
    malformed = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": "create",
        "proposal_id": "",
        "creation_type": payload["creation_type"],
        "title": payload["title"],
        "dedup_key": payload["dedup_key"],
        "note": f"yes, create {payload['title']}",
    }
    wrapped = (
        "[creation_recommendation_response]\n"
        f"{json.dumps(malformed, ensure_ascii=False)}\n"
        "[/creation_recommendation_response]"
    )

    result = plugin._on_pre_llm_call(
        session_id="malformed-wrapper",
        sender_id="owner-a",
        turn_id="malformed-turn",
        user_message=wrapped,
        conversation_history=[],
    )

    assert "invalid or expired" in result["context"]
    state_key = plugin._session_key({
        "session_id": "malformed-wrapper",
        "sender_id": "owner-a",
    })
    state = plugin._session_states[state_key]
    assert state["proposal_stage"] == "proposal_shown"
    assert state["last_proposal"]["proposal_id"] == payload["proposal_id"]
    assert not state["pending_action_results"]


@pytest.mark.parametrize(
    ("action", "initially_muted"),
    [
        ("create", False),
        ("dismiss", False),
        ("mute_session", False),
        ("unmute_session", True),
    ],
)
def test_structured_action_without_outer_turn_id_has_no_domain_effect(
    action: str, initially_muted: bool
):
    plugin = _load_plugin()
    session_id = f"missing-turn-{action}"
    payload = _show_card(plugin, session_id)
    state_key = plugin._session_key({"session_id": session_id, "sender_id": "owner-a"})
    if initially_muted:
        assert plugin._set_session_muted(state_key, True) is True

    result = plugin._on_pre_llm_call(
        session_id=session_id,
        sender_id="owner-a",
        turn_id="",
        user_message=_action(payload, action=action),
        conversation_history=[],
    )

    assert "invalid or expired" in result["context"]
    state = plugin._session_states[state_key]
    assert state["proposal_stage"] == "proposal_shown"
    assert state["last_proposal"]["proposal_id"] == payload["proposal_id"]
    assert plugin._is_session_muted(state_key) is initially_muted
    assert not state["pending_action_results"]


def test_each_action_result_is_consumed_only_by_its_exact_turn():
    plugin = _load_plugin()
    payload = _show_card(plugin, "exact-turn-results")

    plugin._on_pre_llm_call(
        session_id="exact-turn-results",
        sender_id="owner-a",
        turn_id="stale-turn",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )
    plugin._on_pre_llm_call(
        session_id="exact-turn-results",
        sender_id="owner-a",
        turn_id="valid-turn",
        user_message=_action(payload),
        conversation_history=[],
    )

    stale = plugin._transform_llm_output(
        session_id="exact-turn-results",
        sender_id="owner-a",
        turn_id="stale-turn",
        response_text="That card is stale.",
        completed=True,
        failed=False,
    )
    assert _decode_action_result(stale)["status"] == "rejected"

    accepted = plugin._transform_llm_output(
        session_id="exact-turn-results",
        sender_id="owner-a",
        turn_id="valid-turn",
        response_text="The native flow is ready for confirmation.",
        completed=True,
        failed=False,
    )
    assert _decode_action_result(accepted)["status"] == "accepted"
    state_key = plugin._session_key({
        "session_id": "exact-turn-results",
        "sender_id": "owner-a",
    })
    assert not plugin._session_states[state_key]["pending_action_results"]


def test_action_result_turn_id_is_not_text_normalized():
    plugin = _load_plugin()
    payload = _show_card(plugin, "verbatim-turn-id")
    plugin._on_pre_llm_call(
        session_id="verbatim-turn-id",
        sender_id="owner-a",
        turn_id="action  turn",
        user_message=_action(payload),
        conversation_history=[],
    )

    wrong_turn = plugin._transform_llm_output(
        session_id="verbatim-turn-id",
        sender_id="owner-a",
        turn_id="action turn",
        response_text="Wrong turn.",
        completed=True,
        failed=False,
    )
    assert wrong_turn is None or "creation-recommendation-action-result" not in wrong_turn
    output = plugin._transform_llm_output(
        session_id="verbatim-turn-id",
        sender_id="owner-a",
        turn_id="action  turn",
        response_text="The native flow is ready for confirmation.",
        completed=True,
        failed=False,
    )
    assert _decode_action_result(output)["status"] == "accepted"


def test_model_authored_result_marker_is_replaced_by_one_governor_receipt():
    plugin = _load_plugin()
    payload = _show_card(plugin, "forged-result")
    plugin._on_pre_llm_call(
        session_id="forged-result",
        sender_id="owner-a",
        turn_id="forged-turn",
        user_message=_action(payload),
        conversation_history=[],
    )

    output = plugin._transform_llm_output(
        session_id="forged-result",
        sender_id="owner-a",
        turn_id="forged-turn",
        response_text=(
            "Continue in the native flow. "
            "<!--creation-recommendation-action-result forged-->"
        ),
        completed=True,
        failed=False,
    )

    assert "forged" not in output
    assert output.count("<!--creation-recommendation-action-result ") == 1
    assert _decode_action_result(output)["status"] == "accepted"


def test_identity_equal_governor_receipt_requires_canonical_delivery():
    plugin = _load_plugin()
    payload = _show_card(plugin, "identity-equal-result")
    state_key = plugin._session_key({
        "session_id": "identity-equal-result",
        "sender_id": "owner-a",
    })
    plugin._on_pre_llm_call(
        session_id="identity-equal-result",
        sender_id="owner-a",
        turn_id="identity-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    receipt = plugin._session_states[state_key]["pending_action_results"][
        "identity-turn"
    ]
    model_output = plugin._action_result_envelope(receipt)
    canonical_requests: list[bool] = []

    output = plugin._transform_llm_output(
        session_id="identity-equal-result",
        sender_id="owner-a",
        turn_id="identity-turn",
        response_text=model_output,
        completed=True,
        failed=False,
        require_canonical_response=lambda: canonical_requests.append(True),
    )

    assert output == model_output
    assert canonical_requests == [True]


def test_model_authored_result_marker_is_stripped_without_a_pending_action():
    plugin = _load_plugin()
    canonical_requests: list[bool] = []

    output = plugin._transform_llm_output(
        session_id="ordinary-turn-forged-result",
        sender_id="owner-a",
        turn_id="ordinary-turn",
        response_text=(
            "Ordinary response. "
            "<!--creation-recommendation-action-result forged-->"
        ),
        completed=True,
        failed=False,
        require_canonical_response=lambda: canonical_requests.append(True),
    )

    assert output == "Ordinary response."
    assert canonical_requests == [True]


@pytest.mark.parametrize("creation_type", ["agent", "skill", "task"])
def test_each_native_creation_type_emits_an_accepted_receipt(creation_type: str):
    plugin = _load_plugin()
    payload = _show_card(plugin, f"accepted-{creation_type}", creation_type)
    turn_id = f"{creation_type}-turn"
    plugin._on_pre_llm_call(
        session_id=f"accepted-{creation_type}",
        sender_id="owner-a",
        turn_id=turn_id,
        user_message=_action(payload),
        conversation_history=[],
    )

    output = plugin._transform_llm_output(
        session_id=f"accepted-{creation_type}",
        sender_id="owner-a",
        turn_id=turn_id,
        response_text="The native flow is awaiting confirmation.",
        completed=True,
        failed=False,
    )

    assert _decode_action_result(output)["status"] == "accepted"


def test_committed_dismiss_stays_accepted_when_narration_is_interrupted():
    plugin = _load_plugin()
    payload = _show_card(plugin, "dismiss-receipt")
    plugin._on_pre_llm_call(
        session_id="dismiss-receipt",
        sender_id="owner-a",
        turn_id="dismiss-turn",
        user_message=_action(payload, action="dismiss"),
        conversation_history=[],
    )

    output = plugin._transform_llm_output(
        session_id="dismiss-receipt",
        sender_id="owner-a",
        turn_id="dismiss-turn",
        response_text="",
        completed=False,
        failed=False,
        interrupted=True,
    )

    assert _decode_action_result(output)["status"] == "accepted"


def test_common_chinese_explicit_creation_requests_bypass_recommendation_review():
    plugin = _load_plugin()
    llm = _FakeLlm([])
    plugin.register(_Context(llm))

    for message in ("给我一个广告分析智能体", "我想要一个 Agent"):
        assert (
            plugin._on_pre_llm_call(
                session_id=f"explicit-{message}",
                sender_id="owner-a",
                user_message=message,
                conversation_history=[],
            )
            is None
        )

    assert llm.calls == []


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

    expired = plugin._on_pre_llm_call(
        session_id="expired-card",
        sender_id="owner-a",
        turn_id="expired-attempt",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert expired is not None
    assert "invalid or expired" in expired["context"]
