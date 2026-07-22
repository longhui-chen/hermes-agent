import importlib.util
import json
from pathlib import Path

import pytest


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location("creation_governor_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._reset_state_for_tests()
    return module


def _proposal(
    *,
    creation_type="agent",
    suggested_name="Meta Ads Analyst",
    reason="Retained context will improve future analysis.",
    evidence="The user asked for a substantive domain analysis.",
    confidence=0.78,
    dedup_key="meta-ads-analysis",
    proposal_text="This could be reusable as a Meta Ads Analyst Agent—would you like me to prepare the creation plan?",
):
    return {
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
    }


def test_unicode_dedup_keys_are_nonempty_and_do_not_collapse():
    plugin = _load_plugin()
    news = plugin._semantic_dedup_key("每日新闻", "scheduled_task", "每日新闻简报")
    meeting = plugin._semantic_dedup_key("会议纪要", "skill", "会议纪要流程")

    assert news.startswith("scheduled_task:")
    assert meeting.startswith("skill:")
    assert news != meeting
    assert len(news) > len("scheduled_task:")


def test_first_turn_receives_zero_shot_review_without_keyword_classification():
    plugin = _load_plugin()
    inputs = (
        "帮我看看今天有什么 AI 新闻",
        "Take a look at our customer pipeline.",
        "Analiza el rendimiento reciente de la campaña.",
    )

    contexts = []
    for index, message in enumerate(inputs):
        result = plugin._on_pre_llm_call(
            session_id=f"language-{index}",
            user_message=message,
            conversation_history=[],
        )
        assert result is not None
        contexts.append(result["context"])

    assert contexts[0] == contexts[1] == contexts[2]
    assert "zero-shot" in contexts[0]
    assert "not from keywords" in contexts[0]
    assert "One substantive request is enough" in contexts[0]
    assert "first tool call" in contexts[0]
    assert "future fresh information" in contexts[0]
    assert "must not mention, draft, or paraphrase" in contexts[0]
    assert not hasattr(plugin, "_TASK_HINT_RE")
    assert not hasattr(plugin, "_TOPIC_RULES")
    assert not hasattr(plugin, "_judge_creation_opportunity")


def test_third_turn_forces_a_silent_checkpoint_and_repeats_every_three_turns():
    plugin = _load_plugin()
    results = []
    for turn in range(1, 7):
        results.append(
            plugin._on_pre_llm_call(
                session_id="checkpoint-session",
                user_message=f"message {turn}",
                conversation_history=[],
            )["context"]
        )

    assert "Silently consider" in results[0]
    assert "scheduled checkpoint" not in results[1]
    assert "scheduled checkpoint" in results[2]
    assert "scheduled checkpoint" in results[5]
    assert plugin._session_states["checkpoint-session"]["last_evaluation_turn"] == 6


def test_tool_approved_english_proposal_is_delivered_exactly_once():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="english-session",
        user_message="Review my recent Meta ads performance.",
        conversation_history=[],
    )
    args = _proposal()
    result = json.loads(plugin._propose_creation(args, session_id="english-session"))

    assert result["status"] == "proposal_ready"
    assert result["delivery"] == "deferred_to_transform_hook"
    assert "user_prompt" not in result
    assert args["proposal_text"] not in result["next_step"]
    assert "Review my recent Meta ads performance" in result["next_step"]
    assert "is not a deliverable" in result["next_step"]

    transformed = plugin._transform_llm_output(
        session_id="english-session",
        response_text="Here is the performance analysis.",
    )
    assert transformed.endswith(args["proposal_text"])
    assert plugin._transform_llm_output(
        session_id="english-session",
        response_text=transformed,
    ) is None


def test_tool_preserves_chinese_and_mixed_language_proposal_text():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="mixed-session",
        user_message="帮我 review 一下最近的 Meta ads",
        conversation_history=[],
    )
    args = _proposal(
        suggested_name="Meta 广告分析师",
        reason="保留账户背景和判断口径，后续分析会更稳定。",
        evidence="用户希望分析近期 Meta ads 表现。",
        dedup_key="Meta 广告分析",
        proposal_text="这类分析以后可能还会用到，要不要为你准备一个「Meta 广告分析师」Agent 的创建方案？",
    )
    result = json.loads(plugin._propose_creation(args, session_id="mixed-session"))

    assert result["status"] == "proposal_ready"
    assert result["delivery"] == "deferred_to_transform_hook"
    stored = plugin._session_states["mixed-session"]["last_proposal"]
    assert stored["dedup_key"].startswith("agent:")
    assert stored["dedup_key"] != "agent:"


def test_response_that_already_names_the_proposal_is_not_duplicated():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="already-rendered",
        user_message="Please review this campaign.",
        conversation_history=[],
    )
    args = _proposal()
    plugin._propose_creation(args, session_id="already-rendered")

    response = f"Done. {args['proposal_text']}"
    assert plugin._transform_llm_output(
        session_id="already-rendered",
        response_text=response,
    ) is None


def test_prompt_cooldown_suppresses_the_next_ten_user_turns():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="cooldown-session",
        user_message="Analyze this account.",
        conversation_history=[],
    )
    assert json.loads(
        plugin._propose_creation(_proposal(), session_id="cooldown-session")
    )["status"] == "proposal_ready"

    for turn in range(1, 11):
        plugin._on_pre_llm_call(
            session_id="cooldown-session",
            user_message=f"follow up {turn}",
            conversation_history=[],
        )
        blocked = _proposal(
            suggested_name=f"Analyst {turn}",
            dedup_key=f"blocked-{turn}",
            proposal_text=f"Would you like an Analyst {turn} creation plan?",
        )
        assert json.loads(
            plugin._propose_creation(blocked, session_id="cooldown-session")
        ) == {"status": "not_proposed", "reason": "prompt_cooldown"}

    plugin._on_pre_llm_call(
        session_id="cooldown-session",
        user_message="one more follow up",
        conversation_history=[],
    )
    allowed = _proposal(
        suggested_name="New Analyst",
        dedup_key="after-ten",
        proposal_text="Would you like me to prepare a New Analyst creation plan?",
    )
    assert json.loads(
        plugin._propose_creation(allowed, session_id="cooldown-session")
    )["status"] == "proposal_ready"


def test_cooldown_removes_review_nudge_but_keeps_short_acceptance_context():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="carry-session",
        user_message="Analyze this account.",
        conversation_history=[],
    )
    plugin._propose_creation(_proposal(), session_id="carry-session")

    next_turn = plugin._on_pre_llm_call(
        session_id="carry-session",
        user_message="Yes, do that.",
        conversation_history=[],
    )
    assert "previous user-facing response" in next_turn["context"]
    assert "zero-shot review" not in next_turn["context"]

    for index in range(3):
        result = plugin._on_pre_llm_call(
            session_id="carry-session",
            user_message=f"continuation {index}",
            conversation_history=[],
        )
    assert result is None


def test_recent_duplicate_suppression_expires():
    plugin = _load_plugin()
    assert plugin._claim_proposal("session-a", "skill:meeting-notes", 10.0) is True
    assert plugin._claim_proposal("session-a", "skill:meeting-notes", 11.0) is False
    later = 10.0 + plugin.PROPOSAL_TTL_SECONDS + 1
    assert plugin._claim_proposal("session-a", "skill:meeting-notes", later) is True


def test_low_confidence_and_invalid_types_are_rejected():
    plugin = _load_plugin()
    low = _proposal(confidence=0.3)
    invalid = _proposal(creation_type="artifact")

    assert json.loads(plugin._propose_creation(low, session_id="low")) == {
        "status": "not_proposed",
        "reason": "confidence_below_threshold",
    }
    assert json.loads(plugin._propose_creation(invalid, session_id="invalid")) == {
        "status": "invalid",
        "error": "unsupported_creation_type",
    }


def test_missing_user_facing_proposal_text_is_rejected():
    plugin = _load_plugin()
    args = _proposal(proposal_text="")
    assert json.loads(plugin._propose_creation(args, session_id="missing")) == {
        "status": "invalid",
        "error": "missing_proposal_fields",
    }


def test_plugin_self_query_reports_zero_shot_status_without_reviewing():
    plugin = _load_plugin()
    context = plugin._on_pre_llm_call(
        session_id="self-query-session",
        user_message="Do you have creation governor?",
        conversation_history=[],
    )

    assert plugin.PLUGIN_VERSION in context["context"]
    assert "zero-shot semantic opportunity judgment" in context["context"]
    assert "internal zero-shot review" not in context["context"]


def test_tool_schema_uses_semantic_definitions_without_scenario_examples():
    plugin = _load_plugin()

    class Context:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

    context = Context()
    plugin.register(context)
    schema = context.tools[0]["schema"]
    description = schema["description"]

    assert "zero-shot semantic discovery" in description
    assert "An Agent is appropriate" in description
    assert "A Skill is appropriate" in description
    assert "A scheduled_task is appropriate" in description
    assert "phrased as being for today" in description
    assert "before a long tool chain" in description
    assert "explicitly asks to create" in description
    assert "Meta" not in description
    assert "AI news" not in description
    assert context.tools[0]["toolset"] == "creation_governor"
    assert schema["parameters"]["properties"]["creation_type"]["enum"] == [
        "agent",
        "skill",
        "scheduled_task",
    ]
    assert "proposal_text" in schema["parameters"]["required"]


@pytest.mark.parametrize(
    ("creation_type", "name", "proposal_text"),
    [
        ("agent", "Research Partner", "Would you like me to prepare a Research Partner Agent plan?"),
        ("skill", "摘要整理流程", "要不要为你准备一个「摘要整理流程」Skill 的创建方案？"),
        ("scheduled_task", "Informe semanal", "¿Quieres que prepare el plan de esta tarea semanal?"),
    ],
)
def test_all_creation_types_and_languages_share_the_same_governor(
    creation_type,
    name,
    proposal_text,
):
    plugin = _load_plugin()
    session_id = f"type-{creation_type}"
    plugin._on_pre_llm_call(
        session_id=session_id,
        user_message="A substantive request in the user's own language.",
        conversation_history=[],
    )
    args = _proposal(
        creation_type=creation_type,
        suggested_name=name,
        dedup_key=f"{creation_type}-semantic-purpose",
        proposal_text=proposal_text,
    )
    result = json.loads(plugin._propose_creation(args, session_id=session_id))

    assert result["status"] == "proposal_ready"
    transformed = plugin._transform_llm_output(
        session_id=session_id,
        response_text="The user's current task is complete.",
    )
    assert transformed.endswith(proposal_text)
