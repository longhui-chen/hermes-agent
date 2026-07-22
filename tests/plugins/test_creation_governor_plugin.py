import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "creation_governor_plugin", PLUGIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._reset_state_for_tests()
    return module


def _candidate(
    *,
    decision="agent",
    suggested_name="Google Ads Analyst",
    reason="Retained account context and judgment will improve future analysis.",
    confidence=0.82,
    dedup_key="google-ads-analyst",
    proposal_text="Would you like me to create this Google Ads Analyst Agent?",
):
    return {
        "decision": decision,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence_turn_ids": ["evidence-1"],
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
    }


class _FakeLlm:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.results.pop(0)
        return SimpleNamespace(parsed=parsed)


class _ResponseFormatFallbackLlm:
    def __init__(self, result):
        self.result = result
        self.structured_calls = []
        self.complete_calls = []

    def complete_structured(self, **kwargs):
        self.structured_calls.append(kwargs)
        raise RuntimeError("This response_format type is unavailable now")

    def complete(self, messages, **kwargs):
        self.complete_calls.append((messages, kwargs))
        return SimpleNamespace(text=f"```json\n{json.dumps(self.result)}\n```")


class _Context:
    def __init__(self, llm=None):
        self.llm = llm
        self.tools = []
        self.hooks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))


def _decode_envelope(text):
    prefix = "<!--creation-recommendation:start "
    encoded = text.split(prefix, 1)[1].split("-->", 1)[0].strip()
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))


def test_first_turn_and_every_third_turn_run_bounded_structured_checks():
    plugin = _load_plugin()
    none = _candidate(
        decision="none",
        suggested_name="",
        reason="",
        confidence=0,
        dedup_key="",
        proposal_text="",
    )
    llm = _FakeLlm([none, none, none])
    plugin.register(_Context(llm))

    for turn in range(1, 7):
        plugin._on_pre_llm_call(
            session_id="checkpoint-session",
            user_message=f"message {turn}",
            conversation_history=[],
        )

    assert len(llm.calls) == 3
    assert plugin._session_states["checkpoint-session"]["last_evaluation_turn"] == 6
    assert all(
        call["purpose"] == "creation_opportunity_checkpoint" for call in llm.calls
    )
    assert all(call["max_tokens"] == 500 for call in llm.calls)
    assert "high-recall zero-shot" in llm.calls[0]["instructions"]
    assert "ongoing external work domain" in llm.calls[0]["instructions"]
    assert "today" in llm.calls[0]["instructions"]
    assert "not by itself a future trigger" in llm.calls[0]["instructions"]


def test_positive_checkpoint_preserves_answer_and_appends_card_envelope_once():
    plugin = _load_plugin()
    llm = _FakeLlm([_candidate()])
    plugin.register(_Context(llm))

    context = plugin._on_pre_llm_call(
        session_id="english-session",
        turn_id="turn-1",
        user_message="Tell me which Google Ads campaign performed best today.",
        conversation_history=[],
    )

    assert "already completed" in context["context"]
    transformed = plugin._transform_llm_output(
        session_id="english-session",
        response_text="Campaign A had the strongest ROAS.",
    )
    assert transformed.startswith("Campaign A had the strongest ROAS.")
    assert "<!--creation-recommendation:start " in transformed
    assert "This could become a reusable Agent" in transformed
    payload = _decode_envelope(transformed)
    assert payload == {
        "version": 1,
        "type": "creation_recommendation",
        "creation_type": "agent",
        "title": "Google Ads Analyst",
        "reason": "Retained account context and judgment will improve future analysis.",
        "dedup_key": "agent:google-ads-analyst",
        "confidence": 0.82,
        "evidence_turn_ids": ["evidence-1"],
    }
    assert (
        plugin._transform_llm_output(
            session_id="english-session",
            response_text=transformed,
        )
        is None
    )


def test_checkpoint_falls_back_when_provider_rejects_response_format():
    plugin = _load_plugin()
    llm = _ResponseFormatFallbackLlm(_candidate())
    plugin.register(_Context(llm))

    plugin._on_pre_llm_call(
        session_id="fallback-session",
        user_message="Tell me which Google Ads campaign performed best today.",
        conversation_history=[],
    )
    transformed = plugin._transform_llm_output(
        session_id="fallback-session",
        response_text="Campaign A had the strongest ROAS.",
    )

    assert len(llm.structured_calls) == 1
    assert len(llm.complete_calls) == 1
    assert llm.complete_calls[0][1]["purpose"].endswith("json_fallback")
    assert "<!--creation-recommendation:start " in transformed
    assert _decode_envelope(transformed)["creation_type"] == "agent"


def test_none_checkpoint_is_completely_invisible():
    plugin = _load_plugin()
    none = _candidate(
        decision="none",
        suggested_name="",
        reason="",
        confidence=0,
        dedup_key="",
        proposal_text="",
    )
    plugin.register(_Context(_FakeLlm([none])))

    plugin._on_pre_llm_call(
        session_id="none-session",
        user_message="Hello",
        conversation_history=[],
    )
    assert (
        plugin._transform_llm_output(
            session_id="none-session",
            response_text="Hello!",
        )
        is None
    )


def test_display_cooldown_does_not_stop_background_checkpoints():
    plugin = _load_plugin()
    candidates = [
        _candidate(),
        _candidate(suggested_name="Second candidate", dedup_key="second"),
        _candidate(suggested_name="Third candidate", dedup_key="third"),
        _candidate(suggested_name="Fourth candidate", dedup_key="fourth"),
        _candidate(suggested_name="After cooldown", dedup_key="after-cooldown"),
    ]
    llm = _FakeLlm(candidates)
    plugin.register(_Context(llm))

    for turn in range(1, 13):
        plugin._on_pre_llm_call(
            session_id="cooldown-session",
            user_message=f"request {turn}",
            conversation_history=[],
        )
        transformed = plugin._transform_llm_output(
            session_id="cooldown-session",
            response_text=f"answer {turn}",
        )
        if turn == 1:
            assert transformed is not None
        elif turn < 12:
            assert transformed is None
        else:
            assert transformed is not None
            assert "After cooldown" in transformed

    assert len(llm.calls) == 5


def test_dismissal_latches_the_same_semantic_candidate():
    plugin = _load_plugin()
    first = _candidate(
        suggested_name="广告分析助手",
        reason="保留账户背景后，后续判断会更稳定。",
        proposal_text="要为你创建这个广告分析助手吗？",
    )
    plugin.register(_Context(_FakeLlm([first])))
    plugin._on_pre_llm_call(
        session_id="dismiss-session",
        user_message="看看广告效果",
        conversation_history=[],
    )
    shown = plugin._transform_llm_output(
        session_id="dismiss-session",
        response_text="分析完成。",
    )
    assert "可以沉淀为一个 Agent" in shown

    action = plugin._on_pre_llm_call(
        session_id="dismiss-session",
        user_message="暂时不要创建「广告分析助手」",
        conversation_history=[],
    )
    assert "dismissed" in action["context"]

    with plugin._state_lock:
        plugin._session_states["dismiss-session"]["last_prompt_turn"] = -10_000
    result = json.loads(
        plugin._detect_creation_opportunity(first, session_id="dismiss-session")
    )
    assert result == {"status": "candidate_recorded", "reason": "dismissed"}


def test_optional_tool_accepts_none_and_rejects_invalid_or_low_confidence():
    plugin = _load_plugin()
    assert json.loads(
        plugin._detect_creation_opportunity(
            _candidate(
                decision="none",
                suggested_name="",
                reason="",
                confidence=0,
                dedup_key="",
                proposal_text="",
            ),
            session_id="none-tool",
        )
    ) == {"status": "no_candidate", "reason": "none"}
    assert json.loads(
        plugin._detect_creation_opportunity(
            _candidate(decision="artifact"), session_id="invalid"
        )
    ) == {"status": "not_proposed", "reason": "unsupported_creation_type"}
    assert json.loads(
        plugin._detect_creation_opportunity(
            _candidate(confidence=0.2), session_id="low"
        )
    ) == {"status": "not_proposed", "reason": "confidence_below_threshold"}


def test_unicode_dedup_keys_are_stable_and_nonempty():
    plugin = _load_plugin()
    news = plugin._semantic_dedup_key("每日新闻", "task", "每日新闻简报")
    meeting = plugin._semantic_dedup_key("会议纪要", "skill", "会议纪要流程")
    assert news.startswith("task:")
    assert meeting.startswith("skill:")
    assert news != meeting


def test_self_query_reports_runtime_without_triggering_evaluation():
    plugin = _load_plugin()
    llm = _FakeLlm([])
    plugin.register(_Context(llm))
    context = plugin._on_pre_llm_call(
        session_id="self-query",
        user_message="Do you have creation governor?",
        conversation_history=[],
    )
    assert plugin.PLUGIN_VERSION in context["context"]
    assert "first turn and every third turn" in context["context"]
    assert llm.calls == []


def test_tool_schema_is_zero_shot_and_supports_all_outcomes():
    plugin = _load_plugin()
    context = _Context()
    plugin.register(context)
    schema = context.tools[0]["schema"]
    description = schema["description"]

    assert context.tools[0]["name"] == "detect_creation_opportunity"
    assert context.tools[0]["toolset"] == "creation_governor"
    assert "zero-shot" in description
    assert "today is not by itself a trigger" in description
    assert "Missing connectors" in description
    assert "Meta" not in description
    assert "AI news" not in description
    assert schema["parameters"]["properties"]["decision"]["enum"] == [
        "agent",
        "skill",
        "task",
        "none",
    ]


@pytest.mark.parametrize("decision", ["agent", "skill", "task"])
def test_all_creation_types_share_the_same_envelope(decision):
    plugin = _load_plugin()
    candidate = _candidate(decision=decision, dedup_key=f"{decision}-example")
    plugin._on_pre_llm_call(
        session_id=f"type-{decision}",
        user_message="A reusable request.",
        conversation_history=[],
    )
    result = json.loads(
        plugin._detect_creation_opportunity(candidate, session_id=f"type-{decision}")
    )
    assert result["status"] == "proposal_ready"
    transformed = plugin._transform_llm_output(
        session_id=f"type-{decision}",
        response_text="The current task is complete.",
    )
    assert _decode_envelope(transformed)["creation_type"] == decision
