"""Suggest creation opportunities without replacing native creators.

The plugin lets interactive chats surface an implicit creation opportunity.
It gates the suggested path as proposal -> draft -> explicit confirmation, then
hands confirmed work to Hermes' existing native creator.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any

from gateway.response_filters import is_intentional_silence_response
from hermes_constants import get_hermes_home


TOOL_NAME = "propose_creation"
PLUGIN_VERSION = "0.4.2"
MIN_CONFIDENCE = 0.55
PROPOSAL_TTL_SECONDS = 30 * 60
MAX_RECENT_PROPOSALS = 128
EVALUATION_INTERVAL_TURNS = 3
PROMPT_COOLDOWN_TURNS = 10
SESSION_STATE_TTL_SECONDS = 24 * 60 * 60
MAX_SESSION_STATES = 512
CREATION_TYPES = {"agent", "skill", "scheduled_task"}

_recent_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_session_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
_recent_lock = threading.Lock()
_invocation_scope: ContextVar[tuple[str, str, str | None] | None] = ContextVar(
    "creation_governor_invocation_scope",
    default=None,
)


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _dedup_key(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    normalized = re.sub(r"[^\w.:-]+", "-", value, flags=re.UNICODE)
    return normalized.strip("-")[:120]


def _claim_proposal(session_id: str, dedup_key: str, now: float) -> bool:
    """Return False when the same opportunity was proposed recently."""
    identity = (session_id, dedup_key)
    with _recent_lock:
        expired_before = now - PROPOSAL_TTL_SECONDS
        for key, created_at in tuple(_recent_proposals.items()):
            if created_at < expired_before:
                _recent_proposals.pop(key, None)
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return False
        _recent_proposals[identity] = now
        while len(_recent_proposals) > MAX_RECENT_PROPOSALS:
            _recent_proposals.popitem(last=False)
    return True


def _raw_session_key(kwargs: dict[str, Any]) -> str:
    return _text(kwargs.get("session_id") or kwargs.get("task_id"), 160)


def _scoped_session_key(raw_session_id: str, owner_id: str) -> str:
    profile = str(get_hermes_home().resolve())
    return f"{profile}|{_text(owner_id, 160)}|{raw_session_id}"


def _session_key(kwargs: dict[str, Any]) -> str:
    raw_session_id = _raw_session_key(kwargs)
    if not raw_session_id:
        return ""
    owner_fields = ("sender_id", "owner_id", "user_id")
    if any(field in kwargs for field in owner_fields):
        owner_id = next(
            (_text(kwargs.get(field), 160) for field in owner_fields if kwargs.get(field)),
            "",
        )
        return _scoped_session_key(raw_session_id, owner_id)
    invocation = _invocation_scope.get()
    if invocation is not None and invocation[0] == raw_session_id:
        return invocation[1]
    return _scoped_session_key(raw_session_id, "")


def _prune_session_states(now: float) -> None:
    expired_before = now - SESSION_STATE_TTL_SECONDS
    for key, state in tuple(_session_states.items()):
        if float(state.get("last_seen", 0)) < expired_before:
            _session_states.pop(key, None)
    while len(_session_states) > MAX_SESSION_STATES:
        _session_states.popitem(last=False)


def _state_locked(session_id: str, now: float) -> dict[str, Any]:
    _prune_session_states(now)
    state = _session_states.get(session_id)
    if state is None:
        state = {
            "turn": 0,
            "last_evaluation_turn": 0,
            "last_prompt_turn": -10_000,
            "pending_proposal": None,
            "last_proposal": None,
            "proposal_stage": None,
            "draft_only_turn": None,
            "draft_delivered_turn": None,
            "awaiting_proposal_id": None,
            "authorized_turn": None,
            "native_bypass_turn": None,
            "last_user_message": "",
            "last_seen": now,
        }
        _session_states[session_id] = state
    else:
        state["last_seen"] = now
        _session_states.move_to_end(session_id)
    return state


def _prompt_is_cooling_down(state: dict[str, Any]) -> bool:
    return int(state["turn"]) - int(state["last_prompt_turn"]) <= PROMPT_COOLDOWN_TURNS


def _proposal_identity(proposal: dict[str, Any] | None) -> str:
    if not isinstance(proposal, dict):
        return ""
    return _dedup_key(
        f"{proposal.get('creation_type')}:{proposal.get('suggested_name')}"
    )


def _clear_draft_state(state: dict[str, Any], *, stage: str | None) -> None:
    state["proposal_stage"] = stage
    state["draft_only_turn"] = None
    state["draft_delivered_turn"] = None
    state["awaiting_proposal_id"] = None
    state["authorized_turn"] = None


def _latest_assistant_text(history: list[dict[str, Any]]) -> str:
    for message in reversed(history):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _proposal_prompt_is_latest(
    proposal: dict[str, Any], history: list[dict[str, Any]]
) -> bool:
    latest = _latest_assistant_text(history)
    if not latest:
        return False
    payload = _proposal_payload(
        proposal["creation_type"],
        proposal["suggested_name"],
        str(proposal.get("reason") or ""),
        str(proposal.get("evidence") or ""),
        float(proposal.get("confidence") or MIN_CONFIDENCE),
    )
    return payload["user_prompt"] in latest


def _commit_proposal(
    session_id: str,
    proposal: dict[str, Any],
    dedup_key: str,
    now: float,
) -> str | None:
    """Atomically claim cooldown, dedup, and session state for one visible prompt."""
    identity = (session_id, dedup_key)
    with _recent_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return "prompt_cooldown"
        expired_before = now - PROPOSAL_TTL_SECONDS
        for key, created_at in tuple(_recent_proposals.items()):
            if created_at < expired_before:
                _recent_proposals.pop(key, None)
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return "recent_duplicate"
        _recent_proposals[identity] = now
        while len(_recent_proposals) > MAX_RECENT_PROPOSALS:
            _recent_proposals.popitem(last=False)
        state["last_prompt_turn"] = state["turn"]
        state["last_proposal"] = dict(proposal)
        _clear_draft_state(state, stage="proposal_shown")
        state["pending_proposal"] = None
    return None


_TASK_HINT_RE = re.compile(
    r"(?:帮我|请|看看|看一下|查一下|查询|搜索|找一下|分析|整理|总结|写|做|"
    r"检查|评估|研究|优化|对比|翻译|汇总|监控|跟踪|提醒|"
    r"help\s+me|check|find|search|analy[sz]e|review|summari[sz]e|write|research)",
    re.IGNORECASE,
)
_EXPLICIT_CREATION_RE = re.compile(
    r"(?:(?:创建|新建|新增|建立|建个|建一个|再来一个|安装|做成|保存成|生成|我想要|给我).{0,48}"
    r"(?:agent|智能体|助手|skill|技能|定时任务|scheduled\s*task|task))|"
    r"(?:(?:create|build|make|new|add|install|spin\s+up)\s+.{0,48}"
    r"(?:agent|assistant|skill|scheduled\s*task))",
    re.IGNORECASE,
)
_DIRECT_SCHEDULE_RE = re.compile(
    r"(?:每天|每日|每周|每月|每个工作日|定时|到点|早上\s*\d|上午\s*\d|"
    r"下午\s*\d|晚上\s*\d|\d{1,2}\s*点|提醒我|"
    r"every\s+(?:day|week|month)|daily|weekly|monthly|remind\s+me|at\s+\d{1,2})",
    re.IGNORECASE,
)
_SMALL_TALK_RE = re.compile(
    r"^(?:你好|您好|嗨|哈喽|谢谢|多谢|好的|好|行|可以|继续|嗯+|哦+|再见|"
    r"hi|hello|thanks|thank\s+you|ok|okay|continue)[!！,.，。\s]*$",
    re.IGNORECASE,
)
_SELF_QUERY_RE = re.compile(
    r"(?:creation[\s_-]*governor|propose_creation)",
    re.IGNORECASE,
)
_ACCEPT_DRAFT_RE = re.compile(
    r"^(?:生成方案|先生成方案|看看方案|可以，?生成方案|生成吧|那就生成|好|好的|可以|行)$"
)
_CONFIRM_CREATE_RE = re.compile(r"^(?:确认创建|按方案创建|就按这个方案创建|现在创建)$")
_REJECT_PROPOSAL_RE = re.compile(r"^(?:暂不创建|不创建|不用了|先不用|取消)$")
_PERSISTED_PROPOSAL_RE = re.compile(
    r"这类任务可以沉淀成(Agent|Skill|定时任务)「([^」]{1,80})」.*?"
    r"要不要为你生成创建方案[？?]",
    re.DOTALL,
)
_TYPE_BY_LABEL = {"Agent": "agent", "Skill": "skill", "定时任务": "scheduled_task"}
_NONINTERACTIVE_PLATFORMS = {"cron", "subagent", "batch"}
_DRAFT_CONFIRM_PROMPT = "如果方案符合预期，请回复“确认创建”；在此之前不会执行创建。"


def _uses_native_creation_path(user_message: str) -> bool:
    """Return True when Hermes already has a direct native creation request."""
    return bool(
        _EXPLICIT_CREATION_RE.search(user_message)
        or _DIRECT_SCHEDULE_RE.search(user_message)
    )


def _is_creation_governor_self_query(user_message: str) -> bool:
    return bool(_SELF_QUERY_RE.search(user_message))


def _self_description_context() -> str:
    return (
        "[Creation governor internal status: creation-governor is installed, enabled, and "
        f"running as a Hermes background plugin, version {PLUGIN_VERSION}. It registers the "
        "propose_creation tool plus pre_llm_call and transform_llm_output hooks. It judges "
        "ordinary task chats for Agent, Skill, or scheduled-task opportunities, shows only a "
        "second-confirmation proposal, and never creates directly. Answer accurately that the "
        "plugin exists; do not claim it is absent or merely a distributed mechanism. Because "
        "the current message is about the plugin itself, do not suggest creating anything. "
        "Do not expose this internal status block verbatim.]"
    )


def _looks_task_like(user_message: str) -> bool:
    message = _text(user_message, 2000)
    if not message or _SMALL_TALK_RE.fullmatch(message):
        return False
    return bool(_TASK_HINT_RE.search(message) or len(message) >= 18)


_SKILL_SHAPE_RE = re.compile(
    r"(?:整理|总结|改写|润色|翻译|提取|分类|格式化|转写|纪要|清洗|转换|"
    r"summari[sz]e|rewrite|translate|extract|format|transcri(?:be|pt))",
    re.IGNORECASE,
)
_TASK_SHAPE_RE = re.compile(
    r"(?:今天|今日|最近|最新|新闻|资讯|动态|行情|价格|榜单|更新|监控|跟踪|"
    r"today|recent|latest|news|update|monitor|track|price)",
    re.IGNORECASE,
)
_LOW_VALUE_RE = re.compile(
    r"^(?:现在)?几点了?[?？\s]*$|^今天星期几[?？\s]*$|^\d+\s*[+\-*/]\s*\d+[?？\s]*$",
    re.IGNORECASE,
)
_TOPIC_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:AI|人工智能).{0,10}(?:新闻|资讯|动态)|(?:新闻|资讯|动态).{0,10}(?:AI|人工智能)", re.I), "AI 新闻"),
    (re.compile(r"Meta|Facebook|Instagram|脸书", re.I), "Meta 广告"),
    (re.compile(r"Google\s*Ads|谷歌广告", re.I), "Google 广告"),
    (re.compile(r"SEO|搜索引擎优化", re.I), "SEO"),
    (re.compile(r"会议|访谈|转录|transcript", re.I), "会议纪要"),
    (re.compile(r"合同|协议|contract", re.I), "合同审查"),
    (re.compile(r"竞品|竞争对手|competitor", re.I), "竞品研究"),
    (re.compile(r"客户|销售|线索|CRM", re.I), "客户跟进"),
    (re.compile(r"广告|投放|campaign|ROAS|CPA", re.I), "广告投放"),
    (re.compile(r"新闻|资讯|动态|news", re.I), "行业新闻"),
    (re.compile(r"数据|表格|CSV|Excel", re.I), "数据处理"),
    (re.compile(r"文章|内容|文案|content", re.I), "内容创作"),
    (re.compile(r"代码|程序|bug|code", re.I), "代码处理"),
)


def _infer_topic(user_message: str) -> tuple[str, bool]:
    for pattern, topic in _TOPIC_RULES:
        if pattern.search(user_message):
            return topic, True
    cleaned = re.sub(
        r"^(?:请|麻烦|能不能|可以|帮我|帮忙)?\s*(?:看看|看一下|查一下|查询|"
        r"搜索|找一下|分析|整理|总结|写|做|检查|评估|研究|优化|对比|翻译|汇总)?\s*",
        "",
        user_message,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[，。！？,.!?\s]+", " ", cleaned).strip()
    if not cleaned:
        return "任务", False
    return _text(cleaned, 16), False


def _proposal_payload(
    creation_type: str,
    suggested_name: str,
    reason: str,
    evidence: str,
    confidence: float,
) -> dict[str, Any]:
    labels = {
        "agent": "Agent",
        "skill": "Skill",
        "scheduled_task": "定时任务",
    }
    return {
        "status": "proposal_ready",
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "user_prompt": (
            f"顺便问一下：这类任务可以沉淀成{labels[creation_type]}"
            f"「{suggested_name}」，以后直接复用。要不要为你生成创建方案？"
        ),
        "choices": ["生成方案", "暂不创建"],
        "next_step": (
            "先完成并回答用户当前交付的任务，再把 user_prompt 作为轻量建议展示。"
            "用户选择生成方案后，只输出草案并要求再次明确回复“确认创建”；"
            "确认后才转交当前环境已有的原生创建流程执行，本插件不执行创建。"
        ),
    }


def _judge_creation_opportunity(
    user_message: str,
    conversation_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Run a cheap local judgment; no auxiliary model or extra credentials required."""
    message = _text(user_message, 2000)
    if (
        not _looks_task_like(message)
        or _SMALL_TALK_RE.fullmatch(message)
        or _LOW_VALUE_RE.fullmatch(message)
        or _is_creation_governor_self_query(message)
        or _uses_native_creation_path(message)
    ):
        return None
    topic, matched_topic = _infer_topic(message)
    if not matched_topic:
        topic = "专属任务"
    agent_topics = {
        "Meta 广告",
        "Google 广告",
        "SEO",
        "合同审查",
        "竞品研究",
        "客户跟进",
        "广告投放",
    }
    if topic in {"AI 新闻", "行业新闻"} or (
        _TASK_SHAPE_RE.search(message) and topic not in agent_topics
    ):
        creation_type = "scheduled_task"
        suggested_name = f"{topic}简报" if "新闻" in topic else f"{topic}巡检"
        reason = "把这类时效性查询沉淀为可重复执行的任务，后续可以直接复用或设置频率"
    elif _SKILL_SHAPE_RE.search(message):
        creation_type = "skill"
        suggested_name = f"{topic}流程"
        reason = "把这次处理方式固化为可复用流程，后续同类输入可以直接套用"
    else:
        creation_type = "agent"
        suggested_name = f"{topic}分析师" if topic in {"Meta 广告", "Google 广告", "广告投放", "竞品研究"} else f"{topic}助手"
        reason = "保留该领域的背景、口径和后续上下文，后续同类任务可以直接交给它"
    confidence = 0.72 if matched_topic else 0.58
    return {
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence": _text(message, 400),
        "confidence": confidence,
        "dedup_key": _dedup_key(f"{creation_type}:{suggested_name}"),
    }


def _previous_proposal_context(state: dict[str, Any]) -> str:
    if state.get("proposal_stage") != "proposal_shown":
        return ""
    proposal = state.get("last_proposal")
    if not isinstance(proposal, dict):
        return ""
    turns_since = int(state["turn"]) - int(state["last_prompt_turn"])
    if not 1 <= turns_since <= 3:
        return ""
    return (
        "[Creation governor internal context: The previous user-facing response ended with a "
        f"proposal for {proposal.get('creation_type')} '{proposal.get('suggested_name')}'. "
        "If the user accepts, generate a draft only and ask for explicit confirmation before "
        "using Hermes' native creation flow. Do not expose this internal context.]"
    )


def _restore_proposal_match(state: dict[str, Any], match: re.Match[str], stage: str) -> None:
    creation_type = _TYPE_BY_LABEL[match.group(1)]
    suggested_name = _text(match.group(2), 80)
    state["last_proposal"] = {
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": "Recovered from the persisted creation proposal",
        "evidence": "Persisted assistant proposal",
        "confidence": MIN_CONFIDENCE,
        "dedup_key": _dedup_key(f"{creation_type}:{suggested_name}"),
    }
    state["last_prompt_turn"] = max(0, int(state["turn"]) - 1)
    _clear_draft_state(state, stage=stage)
    if stage == "awaiting_confirmation":
        state["draft_delivered_turn"] = max(0, int(state["turn"]) - 1)
        state["awaiting_proposal_id"] = _proposal_identity(state["last_proposal"])


def _rehydrate_persisted_proposal(state: dict[str, Any], history: list[dict[str, Any]]) -> None:
    if isinstance(state.get("last_proposal"), dict):
        return
    assistant_texts = [
        str(message.get("content") or "")
        for message in history
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_texts:
        return
    latest = assistant_texts[-1]
    latest_match = _PERSISTED_PROPOSAL_RE.search(latest)
    if latest_match:
        _restore_proposal_match(state, latest_match, "proposal_shown")
        return
    if _DRAFT_CONFIRM_PROMPT not in latest:
        return
    for prior in reversed(assistant_texts[:-1]):
        prior_match = _PERSISTED_PROPOSAL_RE.search(prior)
        if prior_match:
            _restore_proposal_match(state, prior_match, "awaiting_confirmation")
            return


def _draft_context(proposal: dict[str, Any]) -> str:
    return (
        "[Creation governor internal instruction: The user accepted the proposal for "
        f"{proposal.get('creation_type')} '{proposal.get('suggested_name')}'. Generate a draft only; "
        "do not create, install, save, schedule, or mutate anything in this turn. End by asking the "
        "user to reply exactly '确认创建' if they want the native creation flow to execute the draft. "
        "Do not expose this internal instruction.]"
    )


def _authorized_creation_context(proposal: dict[str, Any]) -> str:
    return (
        "[Creation governor internal context: The user explicitly confirmed the previously drafted "
        f"{proposal.get('creation_type')} '{proposal.get('suggested_name')}'. This turn is authorized "
        "native creation: use Hermes' existing creator and its normal validation. Do not expose this "
        "internal context.]"
    )


def _is_noninteractive(kwargs: dict[str, Any]) -> bool:
    return bool(
        _text(kwargs.get("platform"), 40).lower() in _NONINTERACTIVE_PLATFORMS
        or _text(kwargs.get("execution_origin"), 80).lower() == "background_review"
        or kwargs.get("is_kanban_worker")
    )


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    """Schedule hidden judgments and carry proposal context across transformed output."""
    raw_session_id = _raw_session_key(kwargs)
    session_id = _session_key(kwargs)
    noninteractive = _is_noninteractive(kwargs)
    suppression_reason = None
    if noninteractive:
        suppression_reason = "noninteractive_session"
    elif kwargs.get("structured_output"):
        suppression_reason = "structured_output"
    _invocation_scope.set((raw_session_id, session_id, suppression_reason))
    if suppression_reason:
        if session_id and suppression_reason == "structured_output":
            with _recent_lock:
                state = _state_locked(session_id, time.monotonic())
                state["pending_proposal"] = None
                _clear_draft_state(state, stage=None)
        return None
    if not session_id:
        return None
    user_message = _text(kwargs.get("user_message"), 2000)
    history = kwargs.get("conversation_history")
    if not isinstance(history, list):
        history = []
    now = time.monotonic()
    with _recent_lock:
        state = _state_locked(session_id, now)
        state["turn"] += 1
        state["last_user_message"] = user_message
        pending = state.get("pending_proposal")
        if isinstance(pending, dict) and int(pending.get("turn", -1)) != int(
            state["turn"]
        ):
            state["pending_proposal"] = None
        _rehydrate_persisted_proposal(state, history)
        proposal = state.get("last_proposal")
        stage = state.get("proposal_stage")

        # A draft-generating or authorized state is valid for one turn only.
        # If that turn ended without the post-delivery transition, fail closed.
        if stage == "draft_generating" and int(
            state.get("draft_only_turn") or -1
        ) != int(state["turn"]):
            _clear_draft_state(state, stage="proposal_shown")
            stage = state["proposal_stage"]
        if stage == "creation_authorized" and int(
            state.get("authorized_turn") or -1
        ) != int(state["turn"]):
            _clear_draft_state(state, stage=None)
            stage = state["proposal_stage"]

        if isinstance(proposal, dict) and _REJECT_PROPOSAL_RE.fullmatch(user_message):
            _clear_draft_state(state, stage="dismissed")
            state["pending_proposal"] = None
            return None

        if isinstance(proposal, dict) and stage == "awaiting_confirmation":
            delivered_turn = state.get("draft_delivered_turn")
            confirmation_is_current = bool(
                isinstance(delivered_turn, int)
                and delivered_turn == int(state["turn"]) - 1
                and state.get("awaiting_proposal_id")
                == _proposal_identity(proposal)
                and _DRAFT_CONFIRM_PROMPT in _latest_assistant_text(history)
            )
            if confirmation_is_current and _CONFIRM_CREATE_RE.fullmatch(user_message):
                _clear_draft_state(state, stage="creation_authorized")
                state["authorized_turn"] = state["turn"]
                state["pending_proposal"] = None
                return {"context": _authorized_creation_context(proposal)}
            # Any other turn, including a stale or mismatched confirmation,
            # invalidates the one-shot draft authorization.
            _clear_draft_state(state, stage=None)
            stage = state["proposal_stage"]

        # A confirmation phrase is meaningful only for the immediately preceding,
        # durably delivered draft. Never reinterpret a stale confirmation as a new
        # creation opportunity or expose context for an older proposal.
        if _CONFIRM_CREATE_RE.fullmatch(user_message):
            state["pending_proposal"] = None
            _clear_draft_state(state, stage=None)
            return None

        if (
            isinstance(proposal, dict)
            and stage == "proposal_shown"
            and _ACCEPT_DRAFT_RE.fullmatch(user_message)
        ):
            if not _proposal_prompt_is_latest(proposal, history):
                _clear_draft_state(state, stage=None)
                return None
            state["proposal_stage"] = "draft_generating"
            state["draft_only_turn"] = state["turn"]
            state["pending_proposal"] = None
            return {"context": _draft_context(proposal)}
        carry_context = _previous_proposal_context(state)
        if _is_creation_governor_self_query(user_message):
            _clear_draft_state(state, stage=None)
            state["pending_proposal"] = None
            return {"context": _self_description_context()}
        if _uses_native_creation_path(user_message):
            _clear_draft_state(state, stage=None)
            state["native_bypass_turn"] = state["turn"]
            state["pending_proposal"] = None
            return None
        if _prompt_is_cooling_down(state):
            return {"context": carry_context} if carry_context else None
        turn = int(state["turn"])
        last_evaluation_turn = int(state["last_evaluation_turn"])
        due = (
            (last_evaluation_turn == 0 and _looks_task_like(user_message))
            or (last_evaluation_turn == 0 and turn >= EVALUATION_INTERVAL_TURNS)
            or (
                last_evaluation_turn > 0
                and turn - last_evaluation_turn >= EVALUATION_INTERVAL_TURNS
            )
        )
        if not due:
            return {"context": carry_context} if carry_context else None
        # Mark before the out-of-band call so failures cannot cause repeated calls in one turn.
        state["last_evaluation_turn"] = turn

    proposal = _judge_creation_opportunity(user_message, history)
    if proposal is None:
        return {"context": carry_context} if carry_context else None

    with _recent_lock:
        state = _state_locked(session_id, time.monotonic())
        state["pending_proposal"] = dict(proposal, turn=state["turn"])
    payload = _proposal_payload(
        proposal["creation_type"],
        proposal["suggested_name"],
        proposal["reason"],
        proposal["evidence"],
        proposal["confidence"],
    )
    context = (
        "[Creation governor internal instruction: First complete the user's current task. Then "
        "end the answer with the following lightweight capability proposal exactly once: "
        f"{payload['user_prompt']} Do not say this was generated by a plugin or evaluator.]"
    )
    if carry_context:
        context = carry_context + "\n" + context
    return {"context": context}


def _transform_llm_output(**kwargs: Any) -> str | None:
    """Append a missed scheduled proposal and start the ten-turn prompt cooldown."""
    session_id = _session_key(kwargs)
    response_text = str(kwargs.get("response_text") or "")
    if (
        not session_id
        or not response_text
        or _is_noninteractive(kwargs)
        or kwargs.get("structured_output")
        or is_intentional_silence_response(response_text)
    ):
        return None
    if kwargs.get("failed") or kwargs.get("interrupted") or kwargs.get("completed") is False:
        with _recent_lock:
            state = _state_locked(session_id, time.monotonic())
            state["pending_proposal"] = None
            if state.get("proposal_stage") == "draft_generating":
                _clear_draft_state(state, stage="proposal_shown")
        return None
    now = time.monotonic()
    with _recent_lock:
        state = _state_locked(session_id, now)
        if (
            state.get("proposal_stage") == "draft_generating"
            and int(state.get("draft_only_turn") or -1) == int(state["turn"])
        ):
            if _DRAFT_CONFIRM_PROMPT in response_text:
                return None
            return response_text.rstrip() + "\n\n" + _DRAFT_CONFIRM_PROMPT
        pending = state.get("pending_proposal")
        if not isinstance(pending, dict) or int(pending.get("turn", -1)) != int(state["turn"]):
            return None
        # A proactive tool call in the same turn may already have claimed the slot.
        if _prompt_is_cooling_down(state):
            state["pending_proposal"] = None
            return None
        proposal = dict(pending)

    dedup_key = _dedup_key(proposal.get("dedup_key"))
    payload = _proposal_payload(
        proposal["creation_type"],
        proposal["suggested_name"],
        proposal["reason"],
        proposal["evidence"],
        proposal["confidence"],
    )
    rejection = _commit_proposal(session_id, proposal, dedup_key, now)
    if rejection:
        return None
    user_prompt = payload["user_prompt"]
    if user_prompt in response_text or (
        proposal["suggested_name"] in response_text and "创建方案" in response_text
    ):
        return None
    return response_text.rstrip() + "\n\n" + user_prompt


def _on_post_llm_call(**kwargs: Any) -> None:
    """Commit draft confirmation only after the final response is durable."""
    if _is_noninteractive(kwargs) or kwargs.get("structured_output"):
        return
    session_id = _session_key(kwargs)
    if not session_id:
        return
    assistant_response = str(kwargs.get("assistant_response") or "")
    with _recent_lock:
        state = _state_locked(session_id, time.monotonic())
        if not (
            state.get("proposal_stage") == "draft_generating"
            and int(state.get("draft_only_turn") or -1) == int(state["turn"])
        ):
            return
        proposal = state.get("last_proposal")
        delivered = bool(
            isinstance(proposal, dict)
            and kwargs.get("completed") is not False
            and not kwargs.get("failed")
            and not kwargs.get("interrupted")
            and kwargs.get("persistence_succeeded") is True
            and _DRAFT_CONFIRM_PROMPT in assistant_response
        )
        if not delivered:
            _clear_draft_state(state, stage="proposal_shown")
            return
        state["proposal_stage"] = "awaiting_confirmation"
        state["draft_delivered_turn"] = state["turn"]
        state["awaiting_proposal_id"] = _proposal_identity(proposal)
        state["draft_only_turn"] = None


def _propose_creation(args: dict[str, Any], **kwargs: Any) -> str:
    creation_type = _text(args.get("creation_type"), 40).lower()
    suggested_name = _text(args.get("suggested_name"), 80)
    reason = _text(args.get("reason"), 400)
    evidence = _text(args.get("evidence"), 400)
    dedup_key = _dedup_key(args.get("dedup_key"))
    try:
        confidence = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    if creation_type not in CREATION_TYPES:
        return json.dumps({"status": "invalid", "error": "unsupported_creation_type"})
    if not suggested_name or not reason or not evidence or not dedup_key:
        return json.dumps({"status": "invalid", "error": "missing_proposal_fields"})
    if not math.isfinite(confidence) or confidence < MIN_CONFIDENCE:
        return json.dumps(
            {"status": "not_proposed", "reason": "confidence_below_threshold"}
        )

    session_id = _session_key(kwargs)
    if not session_id:
        return json.dumps({"status": "not_proposed", "reason": "missing_session"})
    now = time.monotonic()
    proposal = {
        "creation_type": creation_type,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence": evidence,
        "confidence": confidence,
        "dedup_key": dedup_key,
    }
    with _recent_lock:
        state = _state_locked(session_id, now)
        invocation = _invocation_scope.get()
        if invocation is not None and invocation[0] == _raw_session_key(kwargs) and invocation[2]:
            return json.dumps({"status": "not_proposed", "reason": invocation[2]})
        if int(state.get("native_bypass_turn") or -1) == int(state["turn"]):
            return json.dumps({"status": "not_proposed", "reason": "native_creation_path"})
        if _prompt_is_cooling_down(state):
            return json.dumps({"status": "not_proposed", "reason": "prompt_cooldown"})
        pending = state.get("pending_proposal")
        if isinstance(pending, dict) and int(pending.get("turn", -1)) == int(
            state["turn"]
        ):
            return json.dumps({"status": "not_proposed", "reason": "prompt_cooldown"})
        expired_before = now - PROPOSAL_TTL_SECONDS
        for key, created_at in tuple(_recent_proposals.items()):
            if created_at < expired_before:
                _recent_proposals.pop(key, None)
        identity = (session_id, dedup_key)
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return json.dumps({"status": "not_proposed", "reason": "recent_duplicate"})
        state["pending_proposal"] = dict(proposal, turn=state["turn"])
    return json.dumps(
        _proposal_payload(creation_type, suggested_name, reason, evidence, confidence),
        ensure_ascii=False,
    )


def _on_pre_tool_call(**kwargs: Any) -> dict[str, str] | None:
    """Mechanically prevent creation side effects during the draft-only turn."""
    session_id = _session_key(kwargs)
    if not session_id:
        return None
    with _recent_lock:
        state = _state_locked(session_id, time.monotonic())
        if int(state.get("draft_only_turn") or -1) != int(state["turn"]):
            return None
    return {
        "action": "block",
        "message": "Creation is blocked while presenting the draft. Wait for explicit '确认创建'.",
    }


def _reset_state_for_tests() -> None:
    with _recent_lock:
        _recent_proposals.clear()
        _session_states.clear()
    _invocation_scope.set(None)


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_tool(
        name=TOOL_NAME,
        toolset="skills",
        schema={
            "name": TOOL_NAME,
            "description": (
                "After completing the user's current task, proactively suggest one persistent "
                "Agent, Skill, or scheduled task whenever the task itself could plausibly recur "
                "or benefit from saved context, a specialized workflow, or ongoing ownership. "
                "Use a broad discovery threshold: a single ordinary task is enough; the user "
                "does not need to mention repetition, persistence, automation, or future use. "
                "Apply this same permissive rule equally to all three creation types; do not require "
                "proof of ongoing ownership for Agent, repeated use for Skill, or an existing cadence "
                "for scheduled_task. For example, one request to check recent Meta ads can justify a "
                "Meta Ads analyst Agent; one transcript cleanup can justify a transcript Skill; and "
                "one AI-news lookup can justify proposing an AI-news briefing task. Choose the type "
                "that would make the clearest useful next capability, and propose only one at a time. "
                "Do not call when the user explicitly asks to create an Agent/Skill/task, or gives "
                "a direct scheduled instruction such as 'every day at 9'; those use Hermes' native "
                "creation behavior. Do not call for artifacts. This tool only proposes; it never "
                "creates, and the user must confirm before the native creation flow begins."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "creation_type": {
                        "type": "string",
                        "enum": ["agent", "skill", "scheduled_task"],
                        "description": (
                            "The single best next capability. Apply the same broad discovery "
                            "threshold to Agent, Skill, and scheduled_task."
                        ),
                    },
                    "suggested_name": {
                        "type": "string",
                        "description": "A concise user-facing name.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why persistence would create long-term value.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "A concise paraphrase of evidence already in context.",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": (
                            "Confidence that persistence could be useful. Use 0.55 or above; "
                            "false-positive suggestions are acceptable because creation requires confirmation."
                        ),
                    },
                    "dedup_key": {
                        "type": "string",
                        "description": "Stable semantic key for suppressing repeat suggestions.",
                    },
                },
                "required": [
                    "creation_type",
                    "suggested_name",
                    "reason",
                    "evidence",
                    "confidence",
                    "dedup_key",
                ],
            },
        },
        handler=_propose_creation,
        description="Suggest an implicit creation opportunity",
        emoji="💡",
    )
