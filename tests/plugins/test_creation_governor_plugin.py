from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


def _load_plugin():
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "creation-governor"
    namespace = types.ModuleType("hermes_plugins")
    namespace.__path__ = []
    sys.modules.setdefault("hermes_plugins", namespace)
    module_name = "hermes_plugins.creation_governor_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module._reset_state_for_tests()
    return module


def test_high_confidence_implicit_opportunity_returns_proposal():
    plugin = _load_plugin()
    result = json.loads(
        plugin._propose_creation(
            {
                "creation_type": "skill",
                "suggested_name": "会议纪要整理",
                "reason": "相同整理流程已经重复出现",
                "evidence": "用户希望以后都按同一结构处理会议转写",
                "confidence": 0.91,
                "dedup_key": "meeting-notes-workflow",
            },
            session_id="session-a",
        )
    )
    assert result["status"] == "proposal_ready"
    assert result["choices"] == ["生成方案", "暂不创建"]
    assert "本插件不执行创建" in result["next_step"]


def test_low_confidence_and_recent_duplicates_are_suppressed():
    plugin = _load_plugin()
    base = {
        "creation_type": "agent",
        "suggested_name": "客户跟进助手",
        "reason": "可能需要长期负责客户跟进",
        "evidence": "用户提到持续跟进客户",
        "confidence": 0.54,
        "dedup_key": "customer-follow-up",
    }
    low = json.loads(plugin._propose_creation(base, session_id="session-a"))
    assert low == {"status": "not_proposed", "reason": "confidence_below_threshold"}

    base["confidence"] = 0.55
    first = json.loads(plugin._propose_creation(base, session_id="session-a"))
    duplicate = json.loads(plugin._propose_creation(base, session_id="session-a"))
    other_session = json.loads(plugin._propose_creation(base, session_id="session-b"))
    assert first["status"] == "proposal_ready"
    assert duplicate == {"status": "not_proposed", "reason": "prompt_cooldown"}
    assert other_session["status"] == "proposal_ready"


def test_single_ordinary_task_can_clear_the_broad_discovery_threshold():
    plugin = _load_plugin()
    result = json.loads(
        plugin._propose_creation(
            {
                "creation_type": "agent",
                "suggested_name": "Meta 广告分析师",
                "reason": "保存投放背景和判断口径可提升后续广告诊断效率",
                "evidence": "用户请求检查近期 Meta 广告效果",
                "confidence": 0.6,
                "dedup_key": "meta-ads-analysis",
            },
            session_id="session-meta",
        )
    )
    assert result["status"] == "proposal_ready"
    assert result["creation_type"] == "agent"
    assert "这类任务可以沉淀成" in result["user_prompt"]


def test_prompt_cooldown_suppresses_the_next_ten_turns():
    plugin = _load_plugin()

    def propose(key: str):
        return json.loads(
            plugin._propose_creation(
                {
                    "creation_type": "agent",
                    "suggested_name": f"分析助手-{key}",
                    "reason": "任务可以复用",
                    "evidence": "用户提交了一项分析任务",
                    "confidence": 0.7,
                    "dedup_key": key,
                },
                session_id="cooldown-session",
            )
        )

    assert propose("first")["status"] == "proposal_ready"
    for turn in range(1, 11):
        plugin._on_pre_llm_call(
            session_id="cooldown-session",
            user_message="你好",
            conversation_history=[],
        )
        assert propose(f"blocked-{turn}") == {
            "status": "not_proposed",
            "reason": "prompt_cooldown",
        }

    plugin._on_pre_llm_call(
        session_id="cooldown-session",
        user_message="你好",
        conversation_history=[],
    )
    assert propose("after-ten")["status"] == "proposal_ready"


def test_cooldown_rejection_does_not_consume_the_dedup_key():
    plugin = _load_plugin()

    def propose(key: str):
        return json.loads(
            plugin._propose_creation(
                {
                    "creation_type": "agent",
                    "suggested_name": f"分析助手-{key}",
                    "reason": "任务可以复用",
                    "evidence": "用户提交了一项分析任务",
                    "confidence": 0.7,
                    "dedup_key": key,
                },
                session_id="atomic-session",
            )
        )

    assert propose("first")["status"] == "proposal_ready"
    assert propose("held-back") == {"status": "not_proposed", "reason": "prompt_cooldown"}
    for _ in range(11):
        plugin._on_pre_llm_call(
            session_id="atomic-session",
            user_message="你好",
            conversation_history=[],
        )
    assert propose("held-back")["status"] == "proposal_ready"


def test_first_task_is_judged_and_missing_prompt_is_appended(monkeypatch):
    plugin = _load_plugin()
    calls = []

    def judge(user_message, history):
        calls.append((user_message, history))
        return {
            "creation_type": "scheduled_task",
            "suggested_name": "AI 新闻简报",
            "reason": "新闻查询适合沉淀为持续能力",
            "evidence": user_message,
            "confidence": 0.72,
            "dedup_key": "ai-news-briefing",
        }

    monkeypatch.setattr(plugin, "_judge_creation_opportunity", judge)
    injected = plugin._on_pre_llm_call(
        session_id="news-session",
        user_message="帮我看看今天有什么 AI 新闻",
        conversation_history=[],
    )
    assert len(calls) == 1
    assert "AI 新闻简报" in injected["context"]

    transformed = plugin._transform_llm_output(
        session_id="news-session",
        response_text="今天值得关注的 AI 新闻有三条。",
    )
    assert transformed.startswith("今天值得关注的 AI 新闻有三条。")
    assert "AI 新闻简报" in transformed
    assert "要不要为你生成创建方案" in transformed
    assert plugin._transform_llm_output(
        session_id="news-session",
        response_text=transformed,
    ) is None


def test_local_judge_maps_single_tasks_without_an_auxiliary_model():
    plugin = _load_plugin()
    cases = (
        (
            "帮我看看今天有什么 AI 新闻",
            "scheduled_task",
            "AI 新闻简报",
        ),
        (
            "帮我看一下 Meta 广告最近的效果",
            "agent",
            "Meta 广告分析师",
        ),
        (
            "帮我整理这份会议转录",
            "skill",
            "会议纪要流程",
        ),
        (
            "帮我分析这个客户最近的情况",
            "agent",
            "客户跟进助手",
        ),
    )
    for message, expected_type, expected_name in cases:
        proposal = plugin._judge_creation_opportunity(message, [])
        assert proposal["creation_type"] == expected_type
        assert proposal["suggested_name"] == expected_name
        assert proposal["confidence"] >= plugin.MIN_CONFIDENCE


def test_third_turn_forces_a_hidden_judgment_without_forcing_a_prompt(monkeypatch):
    plugin = _load_plugin()
    calls = []

    def judge(user_message, history):
        calls.append(user_message)
        return None

    monkeypatch.setattr(plugin, "_judge_creation_opportunity", judge)
    for message in ("你好", "谢谢", "嗯"):
        result = plugin._on_pre_llm_call(
            session_id="three-turn-session",
            user_message=message,
            conversation_history=[],
        )
    assert calls == ["嗯"]
    assert result is None
    assert plugin._transform_llm_output(
        session_id="three-turn-session",
        response_text="好的。",
    ) is None


def test_direct_native_creation_and_schedule_requests_skip_plugin_judge(monkeypatch):
    plugin = _load_plugin()
    calls = []
    monkeypatch.setattr(
        plugin,
        "_judge_creation_opportunity",
        lambda *args: calls.append(args),
    )
    messages = (
        "创建一个 Meta 广告分析 Agent",
        "新增一个合同审查助手",
        "每天早上九点把 AI 新闻汇总给我",
        "帮我新建一个会议纪要 Skill",
    )
    for message in messages:
        plugin._on_pre_llm_call(
            session_id="native-session",
            user_message=message,
            conversation_history=[],
        )
    assert calls == []


def test_native_creation_bypass_is_enforced_in_the_tool_handler():
    plugin = _load_plugin()
    plugin._on_pre_llm_call(
        session_id="native-handler-session",
        user_message="新增一个合同审查助手",
        conversation_history=[],
    )
    result = json.loads(
        plugin._propose_creation(
            {
                "creation_type": "agent",
                "suggested_name": "合同审查助手",
                "reason": "可复用",
                "evidence": "用户明确要求新增助手",
                "confidence": 0.9,
                "dedup_key": "agent:合同审查助手",
            },
            session_id="native-handler-session",
        )
    )
    assert result == {"status": "not_proposed", "reason": "native_creation_path"}


def test_unicode_dedup_keys_preserve_semantic_identity():
    plugin = _load_plugin()
    assert plugin._dedup_key("skill:会议纪要流程") == "skill:会议纪要流程"
    assert plugin._dedup_key("agent:合同审查助手") == "agent:合同审查助手"
    assert plugin._dedup_key("skill:会议纪要流程") != plugin._dedup_key("skill:合同审查流程")


def test_noninteractive_execution_origins_never_evaluate_or_propose(monkeypatch):
    plugin = _load_plugin()
    calls = []
    monkeypatch.setattr(plugin, "_judge_creation_opportunity", lambda *args: calls.append(args))
    cases = (
        {"platform": "cron"},
        {"platform": "subagent"},
        {"platform": "api_server", "execution_origin": "background_review"},
        {"platform": "api_server", "is_kanban_worker": True},
    )
    for index, extra in enumerate(cases):
        session_id = f"background-{index}"
        result = plugin._on_pre_llm_call(
            session_id=session_id,
            user_message="帮我分析这份合同",
            conversation_history=[],
            **extra,
        )
        assert result is None
        denied = json.loads(
            plugin._propose_creation(
                {
                    "creation_type": "agent",
                    "suggested_name": "合同审查助手",
                    "reason": "可复用",
                    "evidence": "后台任务",
                    "confidence": 0.9,
                    "dedup_key": "agent:合同审查助手",
                },
                session_id=session_id,
            )
        )
        assert denied == {"status": "not_proposed", "reason": "noninteractive_session"}
    assert calls == []


def test_failed_or_partial_turn_never_appends_a_proposal(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "_judge_creation_opportunity",
        lambda message, history: {
            "creation_type": "agent",
            "suggested_name": "合同审查助手",
            "reason": "可复用",
            "evidence": message,
            "confidence": 0.9,
            "dedup_key": "agent:合同审查助手",
        },
    )
    plugin._on_pre_llm_call(
        session_id="failed-session",
        user_message="帮我分析这份合同",
        conversation_history=[],
    )
    assert plugin._transform_llm_output(
        session_id="failed-session",
        response_text="供应商调用失败。",
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="provider_error",
    ) is None


def test_acceptance_generates_a_draft_then_requires_explicit_creation_confirmation(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "_judge_creation_opportunity",
        lambda message, history: {
            "creation_type": "skill",
            "suggested_name": "会议纪要流程",
            "reason": "固化整理步骤",
            "evidence": message,
            "confidence": 0.9,
            "dedup_key": "skill:会议纪要流程",
        },
    )
    plugin._on_pre_llm_call(
        session_id="confirm-session",
        user_message="帮我整理这份会议转录",
        conversation_history=[],
    )
    proposal = plugin._transform_llm_output(
        session_id="confirm-session",
        response_text="会议纪要已整理。",
        completed=True,
        failed=False,
    )

    draft = plugin._on_pre_llm_call(
        session_id="confirm-session",
        user_message="生成方案",
        conversation_history=[{"role": "assistant", "content": proposal}],
    )
    assert "draft only" in draft["context"]
    assert "确认创建" in draft["context"]
    block = plugin._on_pre_tool_call(
        session_id="confirm-session",
        tool_name="skill_manage",
        args={},
    )
    assert block["action"] == "block"
    assert plugin._on_pre_tool_call(
        session_id="confirm-session",
        tool_name="write_file",
        args={},
    )["action"] == "block"

    draft_response = plugin._transform_llm_output(
        session_id="confirm-session",
        response_text="方案包含输入格式、整理步骤和输出模板。",
        completed=True,
        failed=False,
    )
    assert "确认创建" in draft_response

    confirmed = plugin._on_pre_llm_call(
        session_id="confirm-session",
        user_message="确认创建",
        conversation_history=[{"role": "assistant", "content": draft_response}],
    )
    assert "authorized native creation" in confirmed["context"]
    assert plugin._on_pre_tool_call(
        session_id="confirm-session",
        tool_name="skill_manage",
        args={},
    ) is None


def test_persisted_proposal_rehydrates_after_plugin_restart():
    plugin = _load_plugin()
    prompt = plugin._proposal_payload(
        "skill",
        "会议纪要流程",
        "固化整理步骤",
        "会议转录",
        0.9,
    )["user_prompt"]

    restarted = _load_plugin()
    draft = restarted._on_pre_llm_call(
        session_id="restarted-session",
        user_message="生成方案",
        conversation_history=[
            {"role": "assistant", "content": "已整理。\n\n" + prompt},
            {"role": "user", "content": "生成方案"},
        ],
    )
    assert "会议纪要流程" in draft["context"]
    assert "draft only" in draft["context"]


def test_persisted_draft_rehydrates_final_confirmation_after_restart():
    plugin = _load_plugin()
    prompt = plugin._proposal_payload(
        "skill",
        "会议纪要流程",
        "固化整理步骤",
        "会议转录",
        0.9,
    )["user_prompt"]
    draft_response = "方案包含输入、处理步骤和输出模板。\n\n" + plugin._DRAFT_CONFIRM_PROMPT

    restarted = _load_plugin()
    confirmed = restarted._on_pre_llm_call(
        session_id="restarted-confirm-session",
        user_message="确认创建",
        conversation_history=[
            {"role": "assistant", "content": "已整理。\n\n" + prompt},
            {"role": "user", "content": "生成方案"},
            {"role": "assistant", "content": draft_response},
            {"role": "user", "content": "确认创建"},
        ],
    )
    assert "会议纪要流程" in confirmed["context"]
    assert "authorized native creation" in confirmed["context"]


def test_old_proposal_is_not_rehydrated_past_a_newer_assistant_reply():
    plugin = _load_plugin()
    prompt = plugin._proposal_payload(
        "skill", "旧流程", "旧原因", "旧证据", 0.9
    )["user_prompt"]
    result = plugin._on_pre_llm_call(
        session_id="stale-proposal-session",
        user_message="生成方案",
        conversation_history=[
            {"role": "assistant", "content": prompt},
            {"role": "user", "content": "先聊别的"},
            {"role": "assistant", "content": "好的，我们先聊别的。"},
            {"role": "user", "content": "生成方案"},
        ],
    )
    assert result is None


def test_plugin_self_query_reports_real_status_and_never_proposes(monkeypatch):
    plugin = _load_plugin()
    calls = []
    monkeypatch.setattr(
        plugin,
        "_judge_creation_opportunity",
        lambda *args: calls.append(args),
    )
    context = plugin._on_pre_llm_call(
        session_id="self-query-session",
        user_message="你有 creation governor 吗？",
        conversation_history=[],
    )
    assert calls == []
    assert "installed, enabled, and running" in context["context"]
    assert plugin.PLUGIN_VERSION in context["context"]
    assert plugin._transform_llm_output(
        session_id="self-query-session",
        response_text="有，creation-governor 已安装并运行。",
    ) is None
    state = plugin._session_states["self-query-session"]
    assert state["last_evaluation_turn"] == 0
    assert state["last_prompt_turn"] == -10_000


def test_unmatched_task_uses_safe_generic_name():
    plugin = _load_plugin()
    proposal = plugin._judge_creation_opportunity(
        "帮我处理一下这件比较复杂但没有明显领域标签的事情",
        [],
    )
    assert proposal["suggested_name"] == "专属任务助手"
    assert "帮我处理一下" not in proposal["suggested_name"]


def test_duplicate_suppression_expires():
    plugin = _load_plugin()
    assert plugin._claim_proposal("session-a", "meeting-notes", 10.0) is True
    assert plugin._claim_proposal("session-a", "meeting-notes", 11.0) is False
    later = 10.0 + plugin.PROPOSAL_TTL_SECONDS + 1
    assert plugin._claim_proposal("session-a", "meeting-notes", later) is True
