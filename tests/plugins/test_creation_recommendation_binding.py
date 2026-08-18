from __future__ import annotations

import base64
import json
import re

import pytest

from tests.plugins.test_creation_governor_plugin import _Context, _FakeLlm
from tests.plugins.test_creation_governor_plugin import _load_plugin as _load_plugin_module


def _load_plugin():
    """本文件钉的是 agent 原生创建流程的绑定语义，因此显式打开 agent 品类。

    该品类在本部署默认关闭（AGENT_RECOMMENDATION_ENABLED=False，落地工具缺失），
    但这些用例要验的是「接受推荐后如何绑定到原生创建流程」，与开关无关。开关在
    _normalize_candidate / _detector_instructions 里都是调用时读取，加载后改写即可；
    schema enum 虽在 exec 时定死，但这些用例走 _FakeLlm 灌固定候选，不过 schema。
    """
    module = _load_plugin_module()
    module.AGENT_RECOMMENDATION_ENABLED = True
    return module


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


def test_preference_read_error_rejects_retry_without_write_or_proposal_consumption(
    monkeypatch,
):
    plugin = _load_plugin()
    payload = _show_card(plugin, "preference-read-error")
    state_key = plugin._session_key(
        {"session_id": "preference-read-error", "sender_id": "owner-a"}
    )
    monkeypatch.setattr(plugin.Path, "exists", lambda _path: True)
    monkeypatch.setattr(
        plugin.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            plugin.sqlite3.OperationalError("database is locked")
        ),
    )
    monkeypatch.setattr(
        plugin,
        "_set_session_muted",
        lambda *_args: pytest.fail("read_error must not write preference state"),
    )

    plugin._on_pre_llm_call(
        session_id="preference-read-error",
        sender_id="owner-a",
        turn_id="mute-attempt",
        user_message=_action(payload, action="mute_session"),
        conversation_history=[],
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )
    output = plugin._transform_llm_output(
        session_id="preference-read-error",
        sender_id="owner-a",
        turn_id="mute-attempt",
        response_text="请稍后重试。",
        completed=True,
        failed=False,
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )

    assert _decode_action_result(output) == {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": payload["proposal_id"],
        "action": "mute_session",
        "status": "rejected",
        "reason_code": "preference_not_persisted",
    }
    state = plugin._session_states[state_key]
    assert state["proposal_stage"] == "proposal_shown"
    assert state["last_proposal"]["proposal_id"] == payload["proposal_id"]


def test_preference_read_error_suppresses_new_recommendations(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "_read_persisted_session_preference",
        lambda _session_id: "read_error",
    )
    llm = _FakeLlm([])
    setattr(plugin, "_plugin_llm", llm)

    result = plugin._on_pre_llm_call(
        session_id="preference-read-error-display",
        sender_id="owner-a",
        turn_id="display-turn",
        user_message="分析近期广告效果",
        conversation_history=[],
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )

    assert result is None
    assert llm.calls == []


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


def test_rejected_action_blocks_creation_tools_for_that_turn():
    """被拒的动作正文照样进模型，闸门必须落在工具派发上。

    往上下文里塞一句「别创建」是劝阻不是约束：模型完全可以照着那段动作正文
    去调 skill_manage(create) / cronjob(create)，于是一个已过期/已重放的推荐
    仍然能把资源建出来。
    """
    plugin = _load_plugin()
    payload = _show_card(plugin, "deny-gate")

    rejected = plugin._on_pre_llm_call(
        session_id="deny-gate",
        sender_id="owner-a",
        turn_id="rejected-turn",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )
    assert "invalid or expired" in rejected["context"]

    for tool_name in ("skill_manage", "cronjob"):
        blocked = plugin._on_pre_tool_call(
            tool_name=tool_name,
            args={"action": "create", "name": "whatever"},
            turn_id="rejected-turn",
        )
        assert blocked is not None
        assert blocked["action"] == "block"

    # 闸门只挡创建。同一轮里读取/修改类动作照常放行——被拒的是「建东西」，
    # 不是整个会话。
    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args={"action": "patch", "name": "whatever"},
            turn_id="rejected-turn",
        )
        is None
    )
    assert (
        plugin._on_pre_tool_call(
            tool_name="cronjob", args={"action": "list"}, turn_id="rejected-turn"
        )
        is None
    )
    # 作用域是单个 turn：别的轮次不受牵连。
    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args={"action": "create"},
            turn_id="some-other-turn",
        )
        is None
    )
    # 与创建无关的工具永远不过这道门。
    assert (
        plugin._on_pre_tool_call(
            tool_name="web_search", args={"query": "x"}, turn_id="rejected-turn"
        )
        is None
    )

    # 本轮收尾后闸门失效，同一个 turn_id 复用不会被莫名挡住。
    plugin._transform_llm_output(
        session_id="deny-gate",
        sender_id="owner-a",
        turn_id="rejected-turn",
        response_text="这条推荐已经不能用了。",
        completed=True,
    )
    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args={"action": "create"},
            turn_id="rejected-turn",
        )
        is None
    )


def test_accepted_action_leaves_creation_tools_open():
    """正向对照：动作被接管的那一轮，创建工具必须照常可用。"""
    plugin = _load_plugin()
    payload = _show_card(plugin, "deny-gate-control")

    accepted = plugin._on_pre_llm_call(
        session_id="deny-gate-control",
        sender_id="owner-a",
        turn_id="accepted-turn",
        user_message=_action(payload),
        conversation_history=[],
    )
    assert "invalid or expired" not in accepted["context"]
    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args={"action": "create"},
            turn_id="accepted-turn",
        )
        is None
    )


def test_rejected_action_blocks_creation_when_tool_args_are_unparseable():
    """参数读不出来时按拒绝处理：这一轮本来就不该有任何创建落地。"""
    plugin = _load_plugin()
    payload = _show_card(plugin, "deny-gate-opaque")
    plugin._on_pre_llm_call(
        session_id="deny-gate-opaque",
        sender_id="owner-a",
        turn_id="opaque-turn",
        user_message=_action(payload, proposal_id="stale-proposal"),
        conversation_history=[],
    )

    for args in ("{not json", None, 42):
        blocked = plugin._on_pre_tool_call(
            tool_name="skill_manage", args=args, turn_id="opaque-turn"
        )
        assert blocked is not None and blocked["action"] == "block"

    # JSON 字符串形式的参数要能被解析出 action，不能一律拦。
    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args=json.dumps({"action": "patch"}),
            turn_id="opaque-turn",
        )
        is None
    )


def test_pending_receipt_key_is_bounded_for_oversized_turn_ids():
    """turn_id 原样当 pending 键时，条数上限拦不住单条键的体积。"""
    plugin = _load_plugin()
    oversized = "t" * (plugin.MAX_RAW_PENDING_TURN_KEY_LEN + 1)

    key = plugin._pending_turn_key(oversized)
    assert key.startswith("sha256:")
    assert len(key) < plugin.MAX_RAW_PENDING_TURN_KEY_LEN
    # 存和取走同一个归一化，查找语义不变。
    assert plugin._pending_turn_key(oversized) == key
    assert plugin._pending_turn_key(oversized + "x") != key
    # 上限之内的 turn_id 原样保留。
    at_limit = "t" * plugin.MAX_RAW_PENDING_TURN_KEY_LEN
    assert plugin._pending_turn_key(at_limit) == at_limit


# 卡片的 title / reason 里合法地含一个 `}`（"JSON {schema}" 这种）时，按花括号
# 定界的非贪婪正则会在字符串内部就收尾，解出来的是残片。
#
# 直接测正则而不是走整条 _on_pre_llm_call：结构化解析失败时会退到「卡片名字
# 出现在消息里就算数」的 legacy 分支，端到端断言会被那条兜底掩盖成绿的。
def test_recommendation_response_envelope_tolerates_a_right_brace_in_a_string():
    plugin = _load_plugin()
    payload = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": "create",
        "proposal_id": "proposal-brace",
        "creation_type": "agent",
        "title": "解析 JSON {schema} 的助手",
    }
    text = (
        "[creation_recommendation_response]"
        + json.dumps(payload, ensure_ascii=False)
        + "[/creation_recommendation_response]"
    )

    match = plugin._RECOMMENDATION_RESPONSE_RE.search(text)
    assert match is not None
    decoded = json.loads(match.group(1))
    assert decoded["title"] == "解析 JSON {schema} 的助手"
    assert decoded["proposal_id"] == "proposal-brace"


# 动作信封挂在消息末尾，Web 会在它前面放一段给模型看的动作说明文案。文案一长
# 就能把信封挤出 governor 的字符预算——那样 governor 完全看不到这次动作，既不
# 接管也不生成回执，请求却以普通模型结果收尾，Web 把这次创建永久停在
# 「不确定且不能重试」。
def test_action_envelope_survives_a_long_leading_instruction():
    plugin = _load_plugin()
    payload = _show_card(plugin, "long-prefix")
    long_prefix = "请注意：" + "这是一段很长的动作说明文案。" * 400
    assert len(long_prefix) > plugin.USER_MESSAGE_LIMIT

    result = plugin._on_pre_llm_call(
        session_id="long-prefix",
        sender_id="owner-a",
        turn_id="long-prefix-turn",
        user_message=long_prefix + "\n\n" + _action(payload),
        conversation_history=[],
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )

    # 同上：断言精确到接管本身。信封被截掉时这一轮退化成普通聊天，而 carry
    # context 里照样有路由名，宽松断言会假绿。
    assert result is not None
    assert "The user accepted the previous recommendation" in result["context"]


def test_bounded_user_message_keeps_the_trailing_envelope():
    plugin = _load_plugin()
    envelope = (
        "[creation_recommendation_response]"
        '{"version":1,"type":"creation_recommendation_response"}'
        "[/creation_recommendation_response]"
    )
    bounded = plugin._bounded_user_message("x" * 5000 + envelope)
    assert bounded.endswith(envelope)
    assert len(bounded) <= plugin.USER_MESSAGE_LIMIT
    # 对照：没有信封时就是普通截断。
    plain = plugin._bounded_user_message("x" * 5000)
    assert len(plain) == plugin.USER_MESSAGE_LIMIT
    # 对照：预算之内原样返回。
    assert plugin._bounded_user_message("短消息") == "短消息"


# 双击 / 传输重发会让两个并发请求复用同一个 turn_id：先到的原子消费掉 proposal
# 拿到 accepted，后到的因为 proposal 已被消费而落拒绝闸门。闸门只按 turn_id 记
# 的话会把先到那个请求真实的创建也挡掉——客户端收到 accepted，资源却没建出来。
def test_accepted_action_survives_a_concurrent_replay_deny():
    plugin = _load_plugin()
    payload = _show_card(plugin, "concurrent-replay")

    accepted = plugin._on_pre_llm_call(
        session_id="concurrent-replay",
        sender_id="owner-a",
        turn_id="shared-turn",
        user_message=_action(payload),
        conversation_history=[],
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )
    assert "invalid or expired" not in accepted["context"]

    # 同一个 turn_id 的第二份请求：proposal 已被消费，判为无效重放。
    replay = plugin._on_pre_llm_call(
        session_id="concurrent-replay",
        sender_id="owner-a",
        turn_id="shared-turn",
        user_message=_action(payload),
        conversation_history=[],
        creation_action_receipt_transport=RECEIPT_TRANSPORT,
    )
    assert "invalid or expired" in replay["context"]

    assert (
        plugin._on_pre_tool_call(
            tool_name="skill_manage",
            args={"action": "create"},
            turn_id="shared-turn",
        )
        is None
    ), "重放请求落下的闸门挡掉了先到那个请求真实的创建"


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
    canonical_requests: list[str | None] = []

    output = plugin._transform_llm_output(
        session_id="identity-equal-result",
        sender_id="owner-a",
        turn_id="identity-turn",
        response_text=model_output,
        completed=True,
        failed=False,
        require_canonical_response=lambda receipt=None: canonical_requests.append(receipt),
    )

    assert output == model_output
    # 有真回执的轮次：把回执原值交给 finalizer，它据此重建链末文本，
    # 而不是在 hook 链结果里猜哪个 marker 是权威的。
    assert canonical_requests == [model_output]


def test_model_authored_result_marker_is_stripped_without_a_pending_action():
    plugin = _load_plugin()
    canonical_requests: list[str | None] = []

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
        require_canonical_response=lambda receipt=None: canonical_requests.append(receipt),
    )

    assert output == "Ordinary response."
    # 只做了伪造清洗、本轮没有回执：必须交 None。传别的值会让 finalizer 把
    # 后置 hook 塞进来的 marker 当成权威回执发出去。
    assert canonical_requests == [None]


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


class _ReentrantSecondClick:
    """在「校验刚结束」的那次 _state_lock 释放点上同步插入第二次点击。

    这个释放点就是双击 / 重发 / 并发流真正撞上的位置：校验与消费分处两个
    临界区时它落在两者之间，插进来的第二次点击会同样看到 proposal_shown
    并同样成交，于是原生创建流程被拉起两次。
    """

    def __init__(self, plugin, second_click):
        self._real = plugin._state_lock
        self._second_click = second_click
        self.armed = False
        self._fired = False

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc_info):
        released = self._real.__exit__(*exc_info)
        if self.armed and not self._fired:
            self._fired = True
            self._second_click()
        return released


def test_a_second_click_landing_mid_validation_cannot_also_be_accepted(monkeypatch):
    plugin = _load_plugin()
    payload = _show_card(plugin, "double-click-create")
    message = _action(payload)

    def _click(turn_id: str) -> None:
        plugin._on_pre_llm_call(
            session_id="double-click-create",
            sender_id="owner-a",
            turn_id=turn_id,
            user_message=message,
            conversation_history=[],
        )

    seam = _ReentrantSecondClick(plugin, lambda: _click("click-2"))
    monkeypatch.setattr(plugin, "_state_lock", seam)
    original_parse = plugin._parse_recommendation_response

    def _parse_and_arm(user_message: str):
        parsed = original_parse(user_message)
        if parsed is not None:
            seam.armed = True
        return parsed

    monkeypatch.setattr(plugin, "_parse_recommendation_response", _parse_and_arm)

    _click("click-1")

    state_key = plugin._session_key(
        {"session_id": "double-click-create", "sender_id": "owner-a"}
    )
    receipts = plugin._session_states[state_key]["pending_action_results"]
    assert sorted(receipts) == ["click-1", "click-2"]
    statuses = [receipt.status for receipt in receipts.values()]
    assert statuses.count("accepted") == 1
    assert receipts["click-2"].reason_code == "proposal_not_actionable"


def test_tool_discovered_card_survives_a_transcript_scoped_tool_dispatch():
    """工具链只递 transcript 级 session_id，卡片仍要落在钩子的稳定作用域里。

    ``model_tools`` 的 registry.dispatch 不转发 conversation_session_id，
    候选一旦写进 transcript 作用域，transform 钩子就再也读不到它，卡片静默消失。
    """
    plugin = _load_plugin()

    plugin._on_pre_llm_call(
        session_id="transcript-1",
        conversation_session_id="stable-conversation",
        sender_id="owner-a",
        turn_id="turn-7",
        user_message="分析近期广告效果",
        conversation_history=[],
    )
    discovered = plugin._detect_creation_opportunity(
        _candidate(),
        session_id="transcript-1",
        turn_id="turn-7",
    )
    assert json.loads(discovered)["status"] == "proposal_ready"

    transformed = plugin._transform_llm_output(
        session_id="transcript-1",
        conversation_session_id="stable-conversation",
        sender_id="owner-a",
        turn_id="turn-7",
        response_text="分析完成。",
        completed=True,
        failed=False,
    )
    assert transformed is not None
    assert _decode_envelope(transformed)["source_turn_id"] == "turn-7"


def test_card_is_only_delivered_by_the_turn_that_produced_it():
    plugin = _load_plugin()

    plugin._on_pre_llm_call(
        session_id="turn-bound-card",
        sender_id="owner-a",
        turn_id="producing-turn",
        user_message="分析近期广告效果",
        conversation_history=[],
    )
    assert (
        json.loads(
            plugin._detect_creation_opportunity(
                _candidate(),
                session_id="turn-bound-card",
                sender_id="owner-a",
                turn_id="producing-turn",
            )
        )["status"]
        == "proposal_ready"
    )

    assert (
        plugin._transform_llm_output(
            session_id="turn-bound-card",
            sender_id="owner-a",
            turn_id="another-concurrent-turn",
            response_text="另一轮的回复。",
            completed=True,
            failed=False,
        )
        is None
    )


def test_two_long_session_keys_do_not_share_one_governor_scope():
    """会话作用域不能被截断/空白折叠塌在一起。

    APIServerAdapter 允许最长 256 字符且保留内部空白的 session key。若 scope 用
    展示用的 _text()（折叠空白 + 截断 160）算，两个合法的不同会话会共享
    last_proposal、mute 偏好和 pending receipt——一个会话里的动作会作用到另一个。
    """
    plugin = _load_plugin()

    shared_prefix = "zettlab:user-with-a-very-long-identity:" + "x" * 150
    first = plugin._raw_session_key({"conversation_session_id": shared_prefix + "-alpha"})
    second = plugin._raw_session_key({"conversation_session_id": shared_prefix + "-beta"})
    assert first != second, "两个仅后缀不同的超长 key 塌成了同一个 scope"

    spaced = plugin._raw_session_key({"conversation_session_id": "conv  a"})
    collapsed = plugin._raw_session_key({"conversation_session_id": "conv a"})
    assert spaced != collapsed, "内部空白被折叠后两个不同 key 撞在了一起"


def test_short_plain_session_key_keeps_its_original_scope():
    """短且无需归一的 key 原样保留——升级不该把用户既有的会话偏好重置掉。"""
    plugin = _load_plugin()
    assert plugin._raw_session_key({"conversation_session_id": "conv-1"}) == "conv-1"
    assert plugin._raw_session_key({"session_id": "sess-2"}) == "sess-2"
    assert plugin._raw_session_key({}) == ""
