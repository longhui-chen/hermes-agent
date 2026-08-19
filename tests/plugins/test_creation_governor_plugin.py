import base64
import importlib.util
import json
import sys
import time
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
    # Import the leaf response-filter module without executing gateway/__init__.py,
    # whose full runtime dependency graph is unrelated to this plugin unit test.
    if "gateway.response_filters" not in sys.modules:
        response_filters_path = PLUGIN_PATH.parents[2] / "gateway" / "response_filters.py"
        response_filters_spec = importlib.util.spec_from_file_location(
            "gateway.response_filters", response_filters_path
        )
        response_filters = importlib.util.module_from_spec(response_filters_spec)
        assert response_filters_spec.loader is not None
        response_filters_spec.loader.exec_module(response_filters)
        sys.modules["gateway.response_filters"] = response_filters
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
    # agent 品类在本部署禁用（AGENT_RECOMMENDATION_ENABLED=False，落地工具缺失），
    # 通用用例改用 skill 承载"任一创建品类"的语义，不影响被测行为。
    decision="skill",
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


def _recommendation_response(
    action, *, proposal_id="", title="Google Ads Analyst"
):
    payload = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": action,
        "proposal_id": proposal_id,
        "creation_type": "skill",
        "title": title,
        "dedup_key": "skill:google-ads-analyst",
        "evidence_turn_ids": ["evidence-1"],
    }
    return (
        "[creation_recommendation_response]\n"
        f"{json.dumps(payload)}\n"
        "[/creation_recommendation_response]"
    )


class _FakeLlm:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        parsed = self.results.pop(0)
        return SimpleNamespace(text=json.dumps(parsed))


class _FailingFastRouteLlm:
    def __init__(self, fallback_result):
        self.fallback_result = fallback_result
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs.get("auxiliary_task"):
            raise RuntimeError("404 route is not in public manifest")
        return SimpleNamespace(
            text=json.dumps(self.fallback_result),
            provider="custom",
            model="active-main-model",
        )


class _PlainFailureStructuredFallbackLlm:
    def __init__(self, result):
        self.result = result
        self.complete_calls = []
        self.structured_calls = []

    def complete(self, messages, **kwargs):
        self.complete_calls.append((messages, kwargs))
        raise RuntimeError("ordinary completion is temporarily unavailable")

    def complete_structured(self, **kwargs):
        self.structured_calls.append(kwargs)
        return SimpleNamespace(parsed=self.result)


class _Context:
    def __init__(self, llm=None):
        self.llm = llm
        self.tools = []
        self.hooks = []
        self.auxiliary_tasks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))

    def register_auxiliary_task(self, **kwargs):
        self.auxiliary_tasks.append(kwargs)


def _decode_envelope(text):
    prefix = "<!--creation-recommendation:start "
    encoded = text.split(prefix, 1)[1].split("-->", 1)[0].strip()
    encoded += "=" * (-len(encoded) % 4)
    return json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))


def _onboarding_welcome_marker(payload):
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"<!--zettlab-onboarding-welcome {encoded}-->"


def test_onboarding_profile_skips_governor_checkpoint_entirely():
    plugin = _load_plugin()
    llm = _FakeLlm([_candidate()])
    plugin.register(_Context(llm))

    result = plugin._on_pre_llm_call(
        profile_name="onboarding",
        session_id="onboarding-session",
        user_message="Frank",
        conversation_history=[],
    )

    assert result is None
    assert llm.calls == []
    assert plugin._session_states == {}


def test_final_onboarding_welcome_emits_existing_cards_without_auxiliary_model(monkeypatch):
    plugin = _load_plugin()
    llm = _FakeLlm([])
    context = _Context(llm)
    emitted = []
    context.emit_attachment = lambda attachment: emitted.append(attachment) or True
    plugin.register(context)
    monkeypatch.setattr(
        plugin,
        "_connection_inventory",
        lambda _session_id, _now: {
            "fetched": True,
            "channels_connected": [],
            "channels_available": ["feishu", "wecom"],
            "channels_recommendable": ["feishu", "wecom"],
            "connectors_connected": [],
            "connectors_recommendable": [],
        },
    )
    marker = _onboarding_welcome_marker(
        {
            "version": 1,
            "type": "zettlab_onboarding_welcome",
            "channel": {"requested": True},
            "task": {
                "title": "持续跟进产品进展",
                "reason": "让变化中的进展保持更新。",
                "proposalText": "要现在设置吗？",
            },
            "artifact": {
                "title": "产品工作台",
                "reason": "集中查看资料和进展。",
                "artifactType": "app",
            },
        }
    )

    hook_context = plugin._on_pre_llm_call(
        # Older mixed deployments may still route the final handoff through the
        # onboarding profile; the explicit marker is the only allowed exception.
        profile_name="onboarding",
        session_id="welcome-session",
        turn_id="turn-welcome",
        user_message=f"Please welcome the user.\n{marker}",
        conversation_history=[],
    )
    transformed = plugin._transform_llm_output(
        session_id="welcome-session",
        response_text="Frank，很高兴认识你。",
        completed=True,
    )

    assert llm.calls == []
    assert "final onboarding welcome" in hook_context["context"]
    assert [attachment["kind"] for attachment in emitted] == [
        "channel.connect",
        "artifact.recommendation",
    ]
    assert emitted[0]["payload"] == {"channel_kind": "feishu"}
    assert emitted[1]["payload"]["artifact_type"] == "app"
    task = _decode_envelope(transformed)
    assert task["creation_type"] == "task"
    assert task["title"] == "持续跟进产品进展"
    assert plugin._session_states[next(iter(plugin._session_states))]["last_proposal"]["proposal_id"] == task["proposal_id"]


def test_onboarding_welcome_channel_is_omitted_when_inventory_has_no_supported_target(monkeypatch):
    plugin = _load_plugin()
    context = _Context(_FakeLlm([]))
    emitted = []
    context.emit_attachment = lambda attachment: emitted.append(attachment) or True
    plugin.register(context)
    monkeypatch.setattr(
        plugin,
        "_connection_inventory",
        lambda _session_id, _now: {
            "fetched": True,
            "channels_connected": ["feishu"],
            "channels_available": [],
            "channels_recommendable": [],
            "connectors_connected": [],
            "connectors_recommendable": [],
        },
    )
    marker = _onboarding_welcome_marker(
        {
            "version": 1,
            "type": "zettlab_onboarding_welcome",
            "channel": {"requested": True},
            "task": {"title": "跟进进展", "reason": "持续更新。", "proposalText": "要设置吗？"},
            "artifact": {"title": "工作台", "reason": "集中查看。", "artifactType": "app"},
        }
    )

    pre = plugin._on_pre_llm_call(
        session_id="no-channel", user_message=marker, conversation_history=[]
    )
    transformed = plugin._transform_llm_output(
        session_id="no-channel", response_text="欢迎，随时可以开始。", completed=True
    )

    assert [attachment["kind"] for attachment in emitted] == ["artifact.recommendation"]
    assert _decode_envelope(transformed)["creation_type"] == "task"
    # 不发卡还不够：App 的 brief 是在拿到实时 inventory 之前写好的，已经命令模型
    # 「引导用户点击本消息末尾的 IM 连接卡」。必须在同一轮显式否决，否则新用户
    # 的第一条消息就指向一个永远不会出现的卡片。
    assert "NO IM connection card will be attached this turn" in pre["context"]
    assert "do not tell them to tap a connection card" in pre["context"]
    # 否决只针对「指向卡片」，不禁止解释 IM 的价值——那是 onboarding 需求本身，
    # 没有卡片时依然成立。
    assert "You may still briefly explain what connecting an IM channel would do" in pre["context"]


def test_onboarding_welcome_keeps_channel_promotion_when_target_exists(monkeypatch):
    plugin = _load_plugin()
    context = _Context(_FakeLlm([]))
    context.emit_attachment = lambda attachment: True
    plugin.register(context)
    monkeypatch.setattr(
        plugin,
        "_connection_inventory",
        lambda _session_id, _now: {
            "fetched": True,
            "channels_connected": [],
            "channels_available": ["feishu"],
            "channels_recommendable": ["feishu"],
            "connectors_connected": [],
            "connectors_recommendable": [],
        },
    )
    marker = _onboarding_welcome_marker(
        {
            "version": 1,
            "type": "zettlab_onboarding_welcome",
            "channel": {"requested": True},
            "task": {"title": "跟进进展", "reason": "持续更新。", "proposalText": "要设置吗？"},
        }
    )

    pre = plugin._on_pre_llm_call(
        session_id="has-channel", user_message=marker, conversation_history=[]
    )

    assert "NO IM connection card" not in pre["context"]


def test_onboarding_welcome_emits_real_agent_template_cards(monkeypatch):
    plugin = _load_plugin()
    context = _Context(_FakeLlm([]))
    emitted = []
    context.emit_attachment = lambda attachment: emitted.append(attachment) or True
    plugin.register(context)
    monkeypatch.setattr(
        plugin,
        "_connection_inventory",
        lambda _session_id, _now: {
            "fetched": True,
            "channels_connected": [],
            "channels_available": [],
            "channels_recommendable": [],
            "connectors_connected": [],
            "connectors_recommendable": [],
        },
    )
    marker = _onboarding_welcome_marker(
        {
            "version": 1,
            "type": "zettlab_onboarding_welcome",
            "channel": {"requested": True},
            "task": {"title": "跟进 SEO", "reason": "持续更新。", "proposalText": "要设置吗？"},
            "agentTemplates": [
                {
                    "templateId": "cn/seo-advisor",
                    "title": "SEO 顾问",
                    "reason": "持续完成技术 SEO 审计和内容优化。",
                },
                {
                    "templateId": "cn/competitor-analysis",
                    "title": "竞品分析",
                    "reason": "持续跟踪和比较竞品。",
                },
            ],
        }
    )

    plugin._on_pre_llm_call(
        session_id="agent-template-welcome",
        turn_id="turn-template",
        user_message=marker,
        conversation_history=[],
    )
    transformed = plugin._transform_llm_output(
        session_id="agent-template-welcome",
        response_text="欢迎回来。",
        completed=True,
    )

    assert [attachment["kind"] for attachment in emitted] == [
        "agent-template.recommendation",
        "agent-template.recommendation",
    ]
    assert emitted[0]["payload"] == {
        "template_id": "cn/seo-advisor",
        "title": "SEO 顾问",
        "reason": "持续完成技术 SEO 审计和内容优化。",
    }
    assert emitted[0]["actions"] == [
        {"id": "dismiss"},
        {"id": "open", "style": "primary"},
    ]
    assert _decode_envelope(transformed)["creation_type"] == "task"


def test_first_turn_and_every_third_turn_run_bounded_json_checks():
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
    state_key = plugin._session_key({"session_id": "checkpoint-session"})
    assert plugin._session_states[state_key]["last_evaluation_turn"] == 6
    assert all(
        call[1]["purpose"] == "creation_opportunity_checkpoint_json"
        for call in llm.calls
    )
    assert all(call[1]["max_tokens"] == 500 for call in llm.calls)
    assert all(
        call[1]["timeout"] == plugin.EVALUATION_TIMEOUT_SECONDS
        for call in llm.calls
    )
    assert all(
        call[1]["auxiliary_task"] == plugin.AUXILIARY_TASK_NAME
        for call in llm.calls
    )
    assert all(call[1]["fail_fast"] is True for call in llm.calls)
    instructions = llm.calls[0][0][0]["content"]
    assert "high-recall zero-shot" in instructions
    assert "Bounded one-shot veto" in instructions
    assert "today" in instructions
    assert "not by itself a future trigger" in instructions


def test_freshness_maintenance_prefers_task_without_topic_keywords():
    plugin = _load_plugin()
    context = _Context()
    plugin.register(context)

    instructions = plugin._DETECTOR_INSTRUCTIONS
    description = context.tools[0]["schema"]["description"]

    assert "Freshness-over-method rule" in instructions
    assert "An explicit cadence is not required to recommend task" in instructions
    assert "keeping one persistent result" in instructions
    assert "Never\ninvent a daily, weekly, or other schedule" in instructions
    assert "Existing-capability gate takes priority" in instructions
    assert "configuring, seeding, previewing, or using" in instructions
    assert "never write internal reasoning" in instructions
    assert "keeping a derived result current as its source changes" in description
    assert "An explicit cadence is not required" in description
    assert "must not be invented" in description
    assert "keep one persistent result fresh" in description
    assert "Flomo" not in instructions
    assert "user.md" not in instructions


def test_bounded_one_shot_policy_and_pending_upload_gate():
    plugin = _load_plugin()
    context = _Context()
    plugin.register(context)

    instructions = plugin._DETECTOR_INSTRUCTIONS
    description = context.tools[0]["schema"]["description"]
    assert "one finite file, table" in instructions
    assert "friction, not evidence for a durable Agent" in instructions
    assert "the analysis verb alone is not evidence for an Agent" in description
    assert "one finite file, table, questionnaire" in description

    response = (
        "目前最直接的替代办法是把 Google 表格下载成 CSV，然后把 CSV 文件上传给我。"
        "我拿到文件后就能直接完成问卷总结。"
    )
    assert plugin._response_delivery_block_reason(response) == "blocked_or_unexecuted"


def test_missing_fast_route_retries_once_on_active_main_model():
    plugin = _load_plugin()
    llm = _FailingFastRouteLlm(_candidate())
    plugin.register(_Context(llm))

    context = plugin._on_pre_llm_call(
        session_id="fast-route-fallback",
        user_message="Analyze which campaign is performing best.",
        conversation_history=[],
    )

    assert "already completed" in context["context"]
    assert len(llm.calls) == 2
    assert llm.calls[0][1]["auxiliary_task"] == plugin.AUXILIARY_TASK_NAME
    assert llm.calls[0][1]["timeout"] == plugin.EVALUATION_TIMEOUT_SECONDS
    assert "auxiliary_task" not in llm.calls[1][1]
    assert (
        llm.calls[1][1]["timeout"]
        == plugin.MAIN_MODEL_FALLBACK_TIMEOUT_SECONDS
    )
    assert (
        llm.calls[1][1]["purpose"]
        == "creation_opportunity_checkpoint_main_fallback"
    )


def test_api_server_never_evaluates_or_transforms_recommendations():
    plugin = _load_plugin()
    llm = _FakeLlm([_candidate()])
    plugin.register(_Context(llm))

    assert plugin._on_pre_llm_call(
        session_id="openai-client-session",
        platform="api_server",
        user_message="Analyze my Google Ads account.",
        conversation_history=[],
    ) is None
    assert llm.calls == []
    assert plugin._transform_llm_output(
        session_id="openai-client-session",
        platform="api_server",
        response_text="Here is the analysis.",
    ) is None


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
    assert "This could become a reusable Skill" in transformed
    payload = _decode_envelope(transformed)
    assert payload == {
        "version": 1,
        "type": "creation_recommendation",
        "proposal_id": payload["proposal_id"],
        "expires_at": payload["expires_at"],
        "creation_type": "skill",
        "title": "Google Ads Analyst",
        # 保留 main 新增的 proposal_text / action_label / action_consequence 三个
        # 字段，但措辞取 skill：agent 品类在本部署禁用（见 _candidate 注释），
        # 这条链路实际产出的是 Skill 文案。
        "reason": (
            "Retained account context and judgment will improve future analysis. "
            "Accepting opens the native Skill creation flow and asks you to "
            "confirm the configuration before creation."
        ),
        "proposal_text": (
            "Would you like me to create this Google Ads Analyst Agent?"
        ),
        "action_label": "Create Skill",
        "action_consequence": (
            "Accepting opens the native Skill creation flow and asks you to "
            "confirm the configuration before creation."
        ),
        "dedup_key": "skill:google-ads-analyst",
        "confidence": 0.82,
        "evidence_turn_ids": ["evidence-1"],
        "source_turn_id": "turn-1",
    }
    assert (
        plugin._transform_llm_output(
            session_id="english-session",
            response_text=transformed,
        )
        is None
    )


def test_unexecuted_connector_block_suppresses_card_without_starting_cooldown():
    plugin = _load_plugin()
    candidate = _candidate(
        suggested_name="谷歌广告效果分析",
        reason="持续保留账户背景可以改善后续判断。",
        proposal_text="要创建谷歌广告分析助手吗？",
    )
    plugin.register(_Context(_FakeLlm([candidate])))
    plugin._on_pre_llm_call(
        session_id="blocked-delivery",
        user_message="最近谷歌广告效果如何",
        conversation_history=[],
    )

    blocked_response = (
        "你是指你自己账户里的谷歌广告投放效果，还是想了解谷歌广告整体的行业趋势？\n\n"
        "如果你问的是自己的广告账户数据，我目前没有接入 Google Ads 连接器，"
        "无法直接读取你的投放报表。确认一下方向我好帮你。"
    )
    assert plugin._transform_llm_output(
        session_id="blocked-delivery",
        response_text=blocked_response,
        completed=True,
        failed=False,
    ) is None

    state_key = plugin._session_key({"session_id": "blocked-delivery"})
    state = plugin._session_states[state_key]
    assert state["last_proposal"] is None
    assert state["last_prompt_turn"] == -10_000

    retry = json.loads(
        plugin._detect_creation_opportunity(
            candidate,
            session_id="blocked-delivery",
        )
    )
    assert retry["status"] == "proposal_ready"
    delivered = plugin._transform_llm_output(
        session_id="blocked-delivery",
        response_text="广告系列 A 的 ROAS 最高，主要由品牌搜索贡献。",
    )
    assert delivered is not None
    assert "creation-recommendation:start" in delivered


def test_clarification_only_suppresses_card_but_optional_followup_does_not():
    plugin = _load_plugin()
    plugin.register(_Context(_FakeLlm([_candidate(), _candidate()])))

    plugin._on_pre_llm_call(
        session_id="clarification-only",
        user_message="分析一下效果",
        conversation_history=[],
    )
    assert plugin._transform_llm_output(
        session_id="clarification-only",
        response_text="你是指广告账户效果，还是网站自然流量效果？",
    ) is None

    plugin._on_pre_llm_call(
        session_id="completed-with-followup",
        user_message="分析广告效果",
        conversation_history=[],
    )
    delivered = plugin._transform_llm_output(
        session_id="completed-with-followup",
        response_text=(
            "Campaign A had the strongest ROAS at 4.2, led by branded search. "
            "Would you like a campaign-level breakdown?"
        ),
    )
    assert delivered is not None
    assert "creation-recommendation:start" in delivered


def test_same_turn_existing_capability_delivery_suppresses_parallel_recommendation():
    plugin = _load_plugin()
    plugin.register(_Context(_FakeLlm([_candidate()])))

    plugin._on_pre_llm_call(
        session_id="dashboard-already-delivered",
        user_message="你先摘要两条新闻",
        conversation_history=[],
    )
    response = (
        "已在存储新闻 Dashboard 预置两条新闻，并更新了定时归档任务。"
        "打开即可查看最新内容。"
    )
    assert plugin._transform_llm_output(
        session_id="dashboard-already-delivered",
        response_text=response,
    ) is None

    state_key = plugin._session_key({"session_id": "dashboard-already-delivered"})
    state = plugin._session_states[state_key]
    assert state["last_proposal"] is None
    assert state["last_prompt_turn"] == -10_000


def test_checkpoint_failure_degrades_without_a_second_blocking_model_call():
    plugin = _load_plugin()
    llm = _PlainFailureStructuredFallbackLlm(_candidate())
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

    assert len(llm.complete_calls) == 1
    assert len(llm.structured_calls) == 0
    assert transformed is None


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
    assert "可以沉淀为一个 Skill" in shown

    action = plugin._on_pre_llm_call(
        session_id="dismiss-session",
        user_message="暂时不要创建「广告分析助手」",
        conversation_history=[],
    )
    assert "dismissed" in action["context"]

    state_key = plugin._session_key({"session_id": "dismiss-session"})
    with plugin._state_lock:
        plugin._session_states[state_key]["last_prompt_turn"] = -10_000
    result = json.loads(
        plugin._detect_creation_opportunity(first, session_id="dismiss-session")
    )
    assert result == {"status": "candidate_recorded", "reason": "dismissed"}


def test_session_mute_persists_across_plugin_state_reset_and_can_be_undone(
    tmp_path, monkeypatch
):
    plugin = _load_plugin()
    preferences_db = tmp_path / "creation-governor-test.db"
    monkeypatch.setattr(plugin, "_preferences_db_path", lambda: preferences_db)
    llm = _FakeLlm([_candidate()])
    plugin.register(_Context(llm))

    plugin._on_pre_llm_call(
        session_id="muted-session",
        user_message="Analyze my Google Ads account.",
        conversation_history=[],
    )
    shown = plugin._transform_llm_output(
        session_id="muted-session",
        response_text="Here is the analysis.",
    )
    assert shown
    proposal_id = _decode_envelope(shown)["proposal_id"]

    mute_context = plugin._on_pre_llm_call(
        session_id="muted-session",
        user_message=_recommendation_response(
            "mute_session", proposal_id=proposal_id
        ),
        conversation_history=[],
    )
    assert "disabled proactive creation recommendations" in mute_context["context"]
    state_key = plugin._session_key({"session_id": "muted-session"})
    assert plugin._is_session_muted(state_key) is True
    assert preferences_db.exists()

    plugin._reset_state_for_tests()
    after_restart_llm = _FakeLlm([_candidate()])
    plugin.register(_Context(after_restart_llm))
    assert plugin._is_session_muted(state_key) is True
    assert (
        plugin._on_pre_llm_call(
            session_id="muted-session",
            user_message="Analyze a different campaign.",
            conversation_history=[],
        )
        is None
    )
    assert after_restart_llm.calls == []
    assert json.loads(
        plugin._detect_creation_opportunity(
            _candidate(), session_id="muted-session"
        )
    ) == {"status": "candidate_recorded", "reason": "session_muted"}
    assert (
        plugin._transform_llm_output(
            session_id="muted-session",
            response_text="No recommendation should be appended.",
        )
        is None
    )

    unmute_context = plugin._on_pre_llm_call(
        session_id="muted-session",
        user_message=_recommendation_response("unmute_session"),
        conversation_history=[],
    )
    assert "re-enabled proactive creation recommendations" in unmute_context["context"]
    assert plugin._is_session_muted(state_key) is False


def test_mute_is_rejected_when_its_preference_cannot_be_persisted(monkeypatch):
    plugin = _load_plugin()
    llm = _FakeLlm([_candidate()])
    plugin.register(_Context(llm))
    plugin._on_pre_llm_call(
        session_id="unpersisted-mute",
        user_message="Analyze my Google Ads account.",
        conversation_history=[],
    )
    shown = plugin._transform_llm_output(
        session_id="unpersisted-mute",
        response_text="Here is the analysis.",
    )
    proposal_id = _decode_envelope(shown)["proposal_id"]
    monkeypatch.setattr(plugin, "_set_session_muted", lambda *_args: False)

    rejected = plugin._on_pre_llm_call(
        session_id="unpersisted-mute",
        user_message=_recommendation_response("mute_session", proposal_id=proposal_id),
        conversation_history=[],
    )
    assert "invalid or expired" in rejected["context"]
    result = plugin._transform_llm_output(
        session_id="unpersisted-mute",
        response_text="I could not save that preference.",
    )
    assert result is None
    state_key = plugin._session_key({"session_id": "unpersisted-mute"})
    assert plugin._is_session_muted(state_key) is False


def test_known_unmuted_sessions_are_bounded_and_pruned_with_session_state(
    tmp_path, monkeypatch
):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_preferences_db_path", lambda: tmp_path / "missing.db")

    for index in range(plugin.MAX_SESSION_STATES + 1):
        assert plugin._is_session_muted(f"unmuted-{index}") is False

    assert len(plugin._known_unmuted_sessions) <= plugin.MAX_SESSION_STATES
    plugin._prune_session_states(time.monotonic() + plugin.SESSION_STATE_TTL_SECONDS + 1)
    assert not plugin._known_unmuted_sessions


def test_muted_sessions_are_bounded_and_pruned_with_session_state():
    plugin = _load_plugin()
    now = time.monotonic()

    for index in range(plugin.MAX_SESSION_STATES + 1):
        plugin._remember_muted_session(f"muted-{index}", now)

    assert len(plugin._muted_sessions) <= plugin.MAX_SESSION_STATES
    plugin._prune_session_states(now + plugin.SESSION_STATE_TTL_SECONDS + 1)
    assert not plugin._muted_sessions


def test_mute_transform_guard_wins_when_a_candidate_is_already_pending(
    tmp_path, monkeypatch
):
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin, "_preferences_db_path", lambda: tmp_path / "creation-governor-test.db"
    )
    plugin.register(_Context(_FakeLlm([_candidate()])))
    plugin._on_pre_llm_call(
        session_id="in-flight-session",
        user_message="Analyze my Google Ads account.",
        conversation_history=[],
    )

    state_key = plugin._session_key({"session_id": "in-flight-session"})
    assert plugin._set_session_muted(state_key, True) is True
    assert (
        plugin._transform_llm_output(
            session_id="in-flight-session",
            response_text="The analysis completed after the mute action.",
        )
        is None
    )


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
            _candidate(decision="workflow"), session_id="invalid"
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
    # agent 品类由 AGENT_RECOMMENDATION_ENABLED 开关控制：关闭时不出现在 enum 里，
    # 模型看不到这个选项（事后硬闸另有一道）。
    assert schema["parameters"]["properties"]["decision"]["enum"] == (
        (["agent"] if plugin.AGENT_RECOMMENDATION_ENABLED else []) + [
        "skill",
        "task",
        "channel",
        "connector",
        "artifact",
        "none",
    ])


def test_registers_region_safe_fast_auxiliary_model_alias():
    plugin = _load_plugin()
    context = _Context()
    plugin.register(context)

    assert context.auxiliary_tasks == [
        {
            "key": plugin.AUXILIARY_TASK_NAME,
            "display_name": "Creation opportunity checkpoint",
            "description": "Fast bounded Agent, Skill, Task, or none classification.",
            "defaults": {
                "model": "zettlab-creation-fast",
                "timeout": 25.0,
            },
        }
    ]


@pytest.mark.parametrize("decision", ["skill", "task"])
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
