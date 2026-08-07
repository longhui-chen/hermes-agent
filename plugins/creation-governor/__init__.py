"""Discover reusable Agent, Skill, and Task opportunities without interrupting work.

The plugin separates three product controls that should not be conflated:

* evaluation cadence: first turn, every third turn, plus optional main-model calls;
* candidate quality: a zero-shot semantic rubric with a structured ``none`` outcome;
* display cadence: cooldown, semantic deduplication, and dismissal latching.

The user's current task always remains the primary response.  A valid candidate is
rendered as a backwards-compatible recommendation envelope: new Zettlab clients
show a card, while older clients see the enclosed plain-text fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
import unicodedata
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from gateway.response_filters import is_intentional_silence_response
from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

TOOL_NAME = "detect_creation_opportunity"
PLUGIN_VERSION = "0.9.3"
MIN_CONFIDENCE = 0.55
AUXILIARY_TASK_NAME = "creation_governor_checkpoint"
AUXILIARY_MODEL_ALIAS = "zettlab-creation-fast"
EVALUATION_TIMEOUT_SECONDS = 25.0
MAIN_MODEL_FALLBACK_TIMEOUT_SECONDS = 15.0
PROPOSAL_TTL_SECONDS = 30 * 60
DISMISS_TTL_SECONDS = 30 * 24 * 60 * 60
MAX_RECENT_PROPOSALS = 128
MAX_DISMISSALS = 128
EVALUATION_INTERVAL_TURNS = 3
PROMPT_COOLDOWN_TURNS = 10
SESSION_STATE_TTL_SECONDS = 24 * 60 * 60
MAX_SESSION_STATES = 512
CREATION_TYPES = {"agent", "skill", "task"}
RECOMMENDATION_ACTIONS = {"create", "dismiss", "mute_session", "unmute_session"}
SESSION_PREFERENCES_DB = "creation_governor.db"
UNSUPPORTED_API_MODES = {"codex_app_server"}
UNSUPPORTED_PLATFORMS = {"acp", "api_server"}
_NONINTERACTIVE_PLATFORMS = {"cron", "subagent", "batch"}

_recent_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_dismissed_proposals: OrderedDict[tuple[str, str], float] = OrderedDict()
_session_states: OrderedDict[str, dict[str, Any]] = OrderedDict()
_muted_sessions: OrderedDict[str, float] = OrderedDict()
_known_unmuted_sessions: OrderedDict[str, float] = OrderedDict()
_state_lock = threading.Lock()
_plugin_llm: Any = None
_invocation_scope: ContextVar[tuple[str, str, str | None, str] | None] = ContextVar(
    "creation_governor_invocation_scope",
    default=None,
)

_SELF_QUERY_RE = re.compile(
    r"(?:creation[\s_-]*governor|detect_creation_opportunity|propose_creation)",
    re.IGNORECASE,
)
_FAST_ROUTE_UNAVAILABLE_RE = re.compile(
    r"(?:404|not found|not in public manifest|unknown (?:model|route)|"
    r"model .+ does not exist|invalid model)",
    re.IGNORECASE,
)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_DISMISS_RE = re.compile(
    r"(?:暂时不要|不用创建|不要创建|先不创建|dismiss|not now|no thanks|do not create)",
    re.IGNORECASE,
)
_ACCEPT_RE = re.compile(
    r"(?:创建|开始创建|就这个|create it|create this|yes[, ]+create)",
    re.IGNORECASE,
)
_EXPLICIT_CREATION_RE = re.compile(
    r"(?:(?:创建|新建|新增|建立|建个|建一个|再来一个|安装|做成|保存成|生成).{0,48}"
    r"(?:agent|智能体|助手|skill|技能|定时任务|scheduled\s*task|task))|"
    r"(?:(?:给我|我想要|我需要|帮我|来(?:一个|个)|要(?:一个|个)).{0,48}"
    r"(?:agent|智能体|助手|skill|技能|定时任务|scheduled\s*task|task))|"
    r"(?:(?:create|build|make|new|add|install|spin\s+up)\s+.{0,48}"
    r"(?:agent|assistant|skill|scheduled\s*task))",
    re.IGNORECASE,
)
_DIRECT_SCHEDULE_RE = re.compile(
    r"(?:每天|每日|每周|每月|每个工作日|定时|提醒我|"
    r"every\s+(?:day|week|month)|daily|weekly|monthly|remind\s+me)",
    re.IGNORECASE,
)
_UNFINISHED_TASK_RE = re.compile(
    r"(?:无法|不能|没法|尚不能|暂时不能).{0,64}"
    r"(?:读取|访问|获取|查询|分析|执行|完成|继续)|"
    r"(?:没有|尚未|还没|未能).{0,32}(?:接入|连接|授权|获得|拿到).{0,64}"
    r"(?:连接器|账号|账户|权限|数据|文件|报表|系统)|"
    r"(?:需要|请).{0,16}(?:先)?(?:连接|接入|授权|提供|上传).{0,80}"
    r"(?:才能|之后|以后|再)|"
    r"(?:cannot|can't|unable to|not able to).{0,64}"
    r"(?:access|read|retrieve|query|analy[sz]e|execute|complete|continue)|"
    r"(?:not connected|isn't connected|missing (?:access|authorization|permission|data))|"
    r"(?:please|need you to).{0,24}(?:connect|authorize|provide|upload).{0,80}"
    r"(?:before|then|so I can)|"
    r"(?:拿到|收到|获得).{0,48}(?:文件|数据|表格|问卷|CSV)?.{0,32}"
    r"(?:后|以后|之后).{0,24}(?:就能|才能|才可以|可以继续).{0,64}"
    r"(?:完成|继续|总结|分析|处理)|"
    r"(?:once|after).{0,80}(?:upload|send|provide|receive|have).{0,80}"
    r"(?:can|will be able to).{0,64}(?:complete|continue|analy[sz]e|summari[sz]e)",
    re.IGNORECASE,
)
_CLARIFICATION_REQUIRED_RE = re.compile(
    r"(?:你是指|你的意思是|请确认一下|需要你确认|还需要确认|我需要先确认).{0,180}[？?]|"
    r"(?:do you mean|which .{0,80} do you mean|please clarify|could you clarify).{0,180}[?]?",
    re.IGNORECASE,
)
_CAPABILITY_DELIVERED_RE = re.compile(
    r"(?:已|已经).{0,20}(?:创建|新建|设置|配置|更新|部署|预置|同步|写入).{0,80}"
    r"(?:agent|智能体|助手|skill|技能|定时任务|dashboard|看板|应用|app)|"
    r"(?:已|已经).{0,20}(?:agent|智能体|助手|skill|技能|定时任务|dashboard|看板|应用|app)"
    r".{0,80}(?:创建|新建|设置|配置|更新|部署|预置|同步|写入)|"
    r"(?:created|configured|updated|deployed|seeded|synchronized|synced).{0,80}"
    r"(?:agent|assistant|skill|scheduled task|dashboard|application|app)|"
    r"(?:agent|assistant|skill|scheduled task|dashboard|application|app).{0,80}"
    r"(?:was|has been|is now).{0,24}"
    r"(?:created|configured|updated|deployed|seeded|synchronized|synced)",
    re.IGNORECASE,
)
_RECOMMENDATION_RESPONSE_RE = re.compile(
    r"\[creation_recommendation_response\]\s*(\{.*?\})\s*"
    r"\[/creation_recommendation_response\]",
    re.DOTALL,
)
def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normalize_creation_type(value: Any) -> str:
    normalized = _text(value, 40).lower().replace("-", "_")
    if normalized == "scheduled_task":
        return "task"
    return normalized


def _semantic_dedup_key(value: Any, creation_type: str, suggested_name: str) -> str:
    raw = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    slug = re.sub(r"[^a-z0-9._:-]+", "-", raw).strip("-")
    prefix = creation_type if creation_type in CREATION_TYPES else "proposal"
    if slug:
        if not slug.startswith(f"{prefix}:"):
            slug = f"{prefix}:{slug}"
        return slug[:120]

    fallback = unicodedata.normalize("NFKC", suggested_name).strip().casefold()
    digest = hashlib.sha256(f"{prefix}:{fallback}".encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _prune_timed_map(
    values: OrderedDict[tuple[str, str], float],
    *,
    expired_before: float,
    max_items: int,
) -> None:
    for key, created_at in tuple(values.items()):
        if created_at < expired_before:
            values.pop(key, None)
    while len(values) > max_items:
        values.popitem(last=False)


def _claim_proposal(session_id: str, dedup_key: str, now: float) -> bool:
    identity = (session_id, dedup_key)
    with _state_lock:
        _prune_timed_map(
            _recent_proposals,
            expired_before=now - PROPOSAL_TTL_SECONDS,
            max_items=MAX_RECENT_PROPOSALS,
        )
        if identity in _recent_proposals:
            _recent_proposals.move_to_end(identity)
            return False
        _recent_proposals[identity] = now
        _prune_timed_map(
            _recent_proposals,
            expired_before=now - PROPOSAL_TTL_SECONDS,
            max_items=MAX_RECENT_PROPOSALS,
        )
    return True


def _is_dismissed(session_id: str, dedup_key: str, now: float) -> bool:
    identity = (session_id, dedup_key)
    with _state_lock:
        _prune_timed_map(
            _dismissed_proposals,
            expired_before=now - DISMISS_TTL_SECONDS,
            max_items=MAX_DISMISSALS,
        )
        return identity in _dismissed_proposals


def _latch_dismissal(session_id: str, dedup_key: str, now: float) -> None:
    with _state_lock:
        _dismissed_proposals[(session_id, dedup_key)] = now
        _prune_timed_map(
            _dismissed_proposals,
            expired_before=now - DISMISS_TTL_SECONDS,
            max_items=MAX_DISMISSALS,
        )


def _raw_session_key(kwargs: dict[str, Any]) -> str:
    return _text(kwargs.get("session_id") or kwargs.get("task_id"), 160)


def _scoped_session_key(raw_session_id: str, owner_id: str) -> str:
    profile = str(get_hermes_home().resolve())
    return f"{profile}|{_text(owner_id, 160)}|{raw_session_id}"


def _session_key(kwargs: dict[str, Any]) -> str:
    raw_session_id = _raw_session_key(kwargs)
    if not raw_session_id:
        return ""
    invocation = _invocation_scope.get()
    profile_prefix = f"{get_hermes_home().resolve()}|"
    owner_fields = ("sender_id", "owner_id", "user_id")
    explicit_owner = next(
        (_text(kwargs.get(field), 160) for field in owner_fields if kwargs.get(field)),
        "",
    )
    if (
        invocation is not None
        and invocation[0] == raw_session_id
        and invocation[1].startswith(profile_prefix)
        and (not explicit_owner or explicit_owner == invocation[3])
    ):
        return invocation[1]
    return _scoped_session_key(raw_session_id, explicit_owner)


def _preferences_db_path() -> Path:
    return get_hermes_home() / SESSION_PREFERENCES_DB


def _open_preferences_db() -> sqlite3.Connection:
    path = _preferences_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=2.0)
    connection.execute("PRAGMA busy_timeout = 2000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS creation_session_preferences (
            session_id TEXT PRIMARY KEY,
            recommendations_muted INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        )
        """
    )
    return connection


def _set_session_muted(session_id: str, muted: bool) -> bool:
    """Persist the user's explicit per-conversation recommendation preference.

    SQLite commits before memory changes so a client never receives a durable
    success result for a preference that would disappear after a restart.
    """
    try:
        with _open_preferences_db() as connection:
            connection.execute(
                """
                INSERT INTO creation_session_preferences (
                    session_id, recommendations_muted, updated_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    recommendations_muted = excluded.recommendations_muted,
                    updated_at = excluded.updated_at
                """,
                (session_id, int(muted), time.time()),
            )
    except (OSError, sqlite3.Error):
        logger.warning(
            "creation recommendation session preference persistence failed",
            exc_info=True,
        )
        return False
    with _state_lock:
        if muted:
            _remember_muted_session(session_id, time.monotonic())
            _known_unmuted_sessions.pop(session_id, None)
        else:
            _muted_sessions.pop(session_id, None)
            _remember_unmuted_session(session_id, time.monotonic())
    return True


def _is_session_muted(session_id: str) -> bool:
    now = time.monotonic()
    with _state_lock:
        _prune_known_unmuted_sessions(now)
        _prune_muted_sessions(now)
        if session_id in _muted_sessions:
            _remember_muted_session(session_id, now)
            return True
        if session_id in _known_unmuted_sessions:
            _remember_unmuted_session(session_id, now)
            return False
    path = _preferences_db_path()
    if not path.exists():
        with _state_lock:
            _remember_unmuted_session(session_id, now)
        return False
    try:
        with sqlite3.connect(path, timeout=2.0) as connection:
            row = connection.execute(
                """
                SELECT recommendations_muted
                FROM creation_session_preferences
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
    except (OSError, sqlite3.Error):
        logger.warning(
            "creation recommendation session preference read failed",
            exc_info=True,
        )
        return False
    muted = bool(row and row[0])
    with _state_lock:
        if muted:
            _remember_muted_session(session_id, now)
            _known_unmuted_sessions.pop(session_id, None)
        else:
            _remember_unmuted_session(session_id, now)
    return muted


def _remember_unmuted_session(session_id: str, now: float) -> None:
    _known_unmuted_sessions[session_id] = now
    _known_unmuted_sessions.move_to_end(session_id)
    _prune_known_unmuted_sessions(now)


def _remember_muted_session(session_id: str, now: float) -> None:
    _muted_sessions[session_id] = now
    _muted_sessions.move_to_end(session_id)
    _prune_muted_sessions(now)


def _prune_known_unmuted_sessions(now: float) -> None:
    expired_before = now - SESSION_STATE_TTL_SECONDS
    for session_id, last_seen in tuple(_known_unmuted_sessions.items()):
        if last_seen < expired_before:
            _known_unmuted_sessions.pop(session_id, None)
    while len(_known_unmuted_sessions) > MAX_SESSION_STATES:
        _known_unmuted_sessions.popitem(last=False)


def _prune_muted_sessions(now: float) -> None:
    expired_before = now - SESSION_STATE_TTL_SECONDS
    for session_id, last_seen in tuple(_muted_sessions.items()):
        if last_seen < expired_before:
            _muted_sessions.pop(session_id, None)
    while len(_muted_sessions) > MAX_SESSION_STATES:
        _muted_sessions.popitem(last=False)


def _prune_session_states(now: float) -> None:
    _prune_known_unmuted_sessions(now)
    _prune_muted_sessions(now)
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
            "last_delivery_turn": -10_000,
            "candidate_turn": -10_000,
            "last_candidate": None,
            "last_proposal": None,
            "proposal_stage": None,
            "draft_only_turn": None,
            "draft_delivered_turn": None,
            "awaiting_proposal_id": None,
            "authorized_turn": None,
            "native_bypass_turn": None,
            "last_user_message": "",
            "last_turn_id": "",
            "last_seen": now,
        }
        _session_states[session_id] = state
    else:
        state["last_seen"] = now
        _session_states.move_to_end(session_id)
    return state


def _prompt_is_cooling_down(state: dict[str, Any]) -> bool:
    return int(state["turn"]) - int(state["last_prompt_turn"]) <= PROMPT_COOLDOWN_TURNS


def _claim_prompt_slot(session_id: str, candidate: dict[str, Any], now: float) -> bool:
    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return False
        identity_source = (
            f"{session_id}|{candidate['dedup_key']}|{state['turn']}|"
            f"{candidate.get('source_turn_id') or ''}"
        )
        candidate["proposal_id"] = hashlib.sha256(
            identity_source.encode("utf-8")
        ).hexdigest()[:32]
        candidate["expires_at"] = time.time() + PROPOSAL_TTL_SECONDS
        state["candidate_turn"] = state["turn"]
        state["last_candidate"] = dict(candidate)
        state["last_proposal"] = dict(candidate)
        state["proposal_stage"] = "proposal_shown"
        return True


def _discard_staged_proposal_locked(
    session_id: str, state: dict[str, Any], *, release_claim: bool
) -> None:
    proposal = state.get("last_proposal")
    if release_claim and isinstance(proposal, dict):
        dedup_key = _text(proposal.get("dedup_key"), 120)
        if dedup_key:
            _recent_proposals.pop((session_id, dedup_key), None)
    state["last_candidate"] = None
    state["last_proposal"] = None
    state["proposal_stage"] = None
    state["candidate_turn"] = -10_000


def _response_delivery_block_reason(response_text: str) -> str:
    """Explain why a staged card must stay hidden for an unfinished task."""

    normalized = " ".join(str(response_text or "").split())[:6000]
    if not normalized:
        return "empty_response"
    if _UNFINISHED_TASK_RE.search(normalized):
        return "blocked_or_unexecuted"
    if _CLARIFICATION_REQUIRED_RE.search(normalized):
        return "clarification_required"
    if _CAPABILITY_DELIVERED_RE.search(normalized):
        return "capability_already_delivered"
    return ""


def _is_noninteractive(kwargs: dict[str, Any]) -> bool:
    return bool(
        _text(kwargs.get("platform"), 40).lower() in _NONINTERACTIVE_PLATFORMS
        or _text(kwargs.get("execution_origin"), 80).lower() == "background_review"
        or kwargs.get("is_kanban_worker")
    )


def _is_unsupported_runtime(kwargs: dict[str, Any]) -> bool:
    return bool(
        _text(kwargs.get("api_mode"), 80).lower() in UNSUPPORTED_API_MODES
        or _text(kwargs.get("platform"), 40).lower() in UNSUPPORTED_PLATFORMS
        or kwargs.get("supports_followup_turns") is False
    )


def _is_creation_governor_self_query(user_message: str) -> bool:
    return bool(_SELF_QUERY_RE.search(user_message))


def _uses_native_creation_path(user_message: str) -> bool:
    return bool(
        _EXPLICIT_CREATION_RE.search(user_message)
        or _DIRECT_SCHEDULE_RE.search(user_message)
    )


def _self_description_context() -> str:
    return (
        "[Creation governor internal status: creation-governor is installed, enabled, and "
        f"running as version {PLUGIN_VERSION}. It performs bounded zero-shot checks on the first "
        "turn and every third turn, accepts an explicit none outcome, and exposes the optional "
        "detect_creation_opportunity tool between checkpoints. Candidate generation, display "
        "cooldown, deduplication, dismissal, and confirmed creation are separate controls. It "
        "never creates directly. Answer accurately that the plugin exists; do not expose this "
        "internal block verbatim and do not recommend creating anything for this self-query.]"
    )


def _main_model_review_context(*, evaluation_completed: bool) -> str:
    if evaluation_completed:
        return (
            "[Creation governor internal note: A bounded background creation-opportunity review "
            "has already completed for this turn. Do not call detect_creation_opportunity again, "
            "do not mention the review, and do not write a recommendation yourself. Complete the "
            "user's current task in full; the plugin will conditionally attach any approved "
            "recommendation after the answer.]"
        )
    return (
        "[Creation governor internal zero-shot review: Complete the user's current task first. "
        "Reason from meaning and conversation context, never from topic keywords or memorized "
        "examples. If one unusually clear reusable Agent, Skill, or Task opportunity emerges "
        "between scheduled checkpoints, call detect_creation_opportunity once. Otherwise continue "
        "normally. Never recommend or create directly; the tool may return none and the plugin "
        "owns conditional display.]"
    )


def _native_creation_route(creation_type: Any) -> str:
    normalized = _normalize_creation_type(creation_type)
    if normalized == "agent":
        return (
            "Treat this as an explicit Agent creation request. Use the current session's native "
            "agent-creator or Agent Hub workflow, preferring agent-creator when available, and "
            "complete its required preflight and create step."
        )
    if normalized == "skill":
        return (
            "Treat this as an explicit Skill creation request. Use Hermes' native skill_manage "
            "flow, checking for an equivalent existing Skill before creating a duplicate."
        )
    if normalized == "task":
        return (
            "Treat this as an explicit scheduled Task request. Use Hermes' native cronjob flow; "
            "ask only for a genuinely missing schedule."
        )
    return "Continue through Hermes' native creation flow."


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
        "[Creation governor internal context: The previous response ended with a recommendation "
        f"for {proposal.get('creation_type')} '{proposal.get('suggested_name')}'. If the user "
        f"accepts, {_native_creation_route(proposal.get('creation_type'))} Preserve the native "
        "confirmation boundaries. If "
        "the user declines, acknowledge briefly. Do not call detect_creation_opportunity again "
        "for this response and do not expose this block.]"
    )


def _join_context(*parts: str) -> dict[str, str] | None:
    content = "\n".join(part for part in parts if part)
    return {"context": content} if content else None


def _conversation_evidence(history: Any, user_message: str) -> str:
    rows: list[tuple[str, str]] = []
    if isinstance(history, list):
        for message in history:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            rows.append((role, _text(content, 900)))
    if not rows or rows[-1][0] != "user" or rows[-1][1] != _text(user_message, 900):
        rows.append(("user", _text(user_message, 900)))
    rows = rows[-8:]
    rendered = []
    for index, (role, content) in enumerate(rows, start=1):
        rendered.append(f"[evidence-{index}] {role.upper()}: {content}")
    return "\n".join(rendered)[:6000]


_DETECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["agent", "skill", "task", "none"]},
        "suggested_name": {"type": "string"},
        "reason": {"type": "string"},
        "evidence_turn_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "dedup_key": {"type": "string"},
        "proposal_text": {"type": "string"},
    },
    "required": [
        "decision",
        "suggested_name",
        "reason",
        "evidence_turn_ids",
        "confidence",
        "dedup_key",
        "proposal_text",
    ],
    "additionalProperties": False,
}


_DETECTOR_INSTRUCTIONS = """Perform one high-recall zero-shot product judgment.

Return exactly one of agent, skill, task, or none. Do not classify by topic words and do not use
memorized examples. A single substantive request can be enough only when the conversation itself
supports durable future value; the mere possibility that a capability could be reused is not enough.
Do not require magic words such as repetition, saving, or creation, but require affirmative semantic
evidence that the account, project, source, responsibility, or class of future inputs continues beyond
this bounded request.

Definitions and conflict order:
1. task: the desired future value depends on a recurring time trigger, event trigger, background
   monitoring, repeated refresh of new information, or keeping a derived result current as its
   source changes. A word such as 'today' merely scopes the current data; it is not by itself a future trigger.
2. agent: future work needs a long-lived responsible role, retained domain context, judgment,
   autonomous choice among tools, decisions about the next step, or repeated interpretation of a
   changing real-world business domain, account, operation, project, or body of evidence.
3. skill: future inputs vary but a stable input-to-output method can be reused without an
   independent identity or durable state. Do not choose skill when the primary future value is
   keeping one persistent result, profile, summary, index, report, or state up to date.
4. none: small talk, a trivial transformation, a low-value closed-world fact lookup, an explicit
   request to create/configure/schedule something through Hermes' native flow, or no reasonable
   reuse value.

Bounded one-shot veto: return none when the user only wants a result from one finite file, table,
questionnaire, document, import, dataset, or other bounded item and the conversation does not support
future recurrence, ongoing ownership, background refresh, or retained-context judgment. Needing an
upload, authorization, connector, or other setup step to finish the current request is execution
friction, not evidence for a durable Agent. If the same method is expected across future inputs,
skill may qualify; if freshness or a future trigger is the value, task may qualify; if continuing
responsibility and autonomous judgment are both present, agent may qualify.

High-recall boundary: a substantive request to inspect, compare, diagnose, research, optimize, or
make a judgment may qualify on the first request when it concerns an ongoing external account,
project, operation, or responsibility whose future state and decisions remain after this turn. The
verb alone never makes it an Agent. Choose none when durable reuse value is absent from the meaning
of the conversation, including bounded one-shot work.

Existing-capability gate takes priority over high recall. If the conversation shows that an
existing Agent, Skill, scheduled Task, Dashboard, or application already performs the same future
job, return none unless the new object would add a materially different responsibility that the
existing capability cannot provide. Do not recommend a parallel Agent or Skill merely because the
user is configuring, seeding, previewing, or using an object that was just created.

Freshness-over-method rule: prefer task over skill when at least two of these semantic properties
are clearly supported by the conversation: (a) the source, account, file, feed, or evidence changes
over time; (b) the generated result becomes stale when the source changes; (c) automatic refresh
would remove repeated manual work. An explicit cadence is not required to recommend task. Never
invent a daily, weekly, or other schedule in the recommendation. After the user confirms creation,
Hermes' native task/cronjob flow must ask for any missing schedule or event trigger. Stable refresh
steps do not make the opportunity a skill when freshness is the core value.

Apply this semantic gate before returning a positive decision. Ask, in order: (a) is this merely one
bounded item whose requested result ends the work; (b) will the underlying information, account,
project, or operating environment continue after this turn; (c) would a responsible role with
retained context make a future judgment better; (d) is there evidence that a stable method will be
used on materially different future inputs? If (a) is yes and (b)-(d) are no, return none. Otherwise,
when no existing capability already covers the need, choose task for a future trigger,
background refresh, or freshness maintenance; otherwise agent for continuing ownership/judgment,
otherwise skill for the reusable method. Do not infer recurrence merely because a method is
theoretically reusable. Do not reduce an ongoing analytical responsibility to a fact lookup merely
because the current data or connector is unavailable.

Judge reuse value separately from current execution availability. Missing authorization,
connectors, data, or tools may still reveal a long-term need, but the plugin separately suppresses
display unless the current response actually delivers the user's task. Recommend only the
first-layer object the user most needs, never multiple objects. Match the user's language. For a
positive decision, write concise user-facing card copy: name the object clearly; explain what it
will do for the user and why it helps; never write internal reasoning such as "the user..." or
“用户……”. Make proposal_text a direct call to action that names the creation action and says that
Hermes will enter its native creation/configuration flow. Provide confidence, a stable semantic
dedup_key, and evidence_turn_ids chosen only from the supplied labels. For none, use empty strings,
an empty evidence list, and confidence 0. Never claim anything was created.

For a task recommendation, name the ongoing outcome that should stay current instead of naming a
generic method. Explain what changing source would make the current result stale, but do not claim
or imply a cadence the user did not provide.

中文请求必须按同一套语义规则判断，不依赖“重复”“以后”“保存”或“创建”等触发词，但必须从语义
上找到任务在本轮之后仍会继续的证据。只处理一份确定的文件、表格、问卷、文档、导入数据或其他
有限对象，并且完成本次结果后工作即结束时，优先返回 none；不能因为还需要用户上传文件、授权或
连接 Connector 才能完成本轮任务，就推断需要一个长期 Agent。先判断需求所涉及的账户、项目、
业务环境或信息是否会继续变化；如果会变化且后续判断需要
保留背景、综合数据或自主选择工具，选择 agent。如果价值来自未来的时间、事件、后台监控、提醒，
或让一个随来源变化而过期的画像、摘要、索引、报告或状态持续保持最新，选择 task。只要“来源会
变化”“结果会过期”“自动刷新能减少反复手工操作”中至少两项在语义上成立，就可以优先 task，
不要求用户先说每天、每周或具体频率；推荐时不得虚构周期，用户确认后再由 Hermes 原生 cronjob
流程补问缺失的时间或事件条件。如果对话显示已有 Agent、Skill、定时任务、Dashboard 或应用已经
覆盖同一长期需求，优先返回 none；不能因为用户正在配置、预置、预览或使用刚创建的对象，就再推荐
一个平行的 Agent 或 Skill。如果有证据表明未来还会处理不同输入，且价值只是重复使用一套稳定
方法、又不存在保持结果新鲜的需求，才选择 skill。只有寒暄、低价值封闭事实、微小的一次性转换、
有限对象的一次性处理、用户已经明确要求创建，或 Agent/Skill/Task 三种长期价值都确实不存在时，
才选择 none。
“今天”“最近”“当前”只是本次数据范围，不等于没有长期价值。缺少授权、连接器或数据只影响本次
执行，但插件会在当前任务没有实际交付时阻止卡片展示。名称、原因和 proposal_text 必须使用面向
用户的语言，不能写“用户已……”这类内部判定；proposal_text 要明确说明将进入哪种原生创建流程。"""


def _run_forced_evaluation(
    *,
    user_message: str,
    conversation_history: Any,
) -> dict[str, Any] | None:
    llm = _plugin_llm
    if llm is None:
        return None
    evidence = _conversation_evidence(conversation_history, user_message)

    # Prefer an ordinary bounded JSON completion.  Some OpenAI-compatible
    # gateways accept ``response_format`` but collapse optional semantic
    # judgments to the schema's empty ``none`` shape.  The same model produces
    # materially better zero-shot classifications when asked for JSON in the
    # prompt, and the candidate still passes strict local normalization before
    # it can be displayed.
    messages = [
        {
            "role": "system",
            "content": (
                _DETECTOR_INSTRUCTIONS
                + "\n\nReturn only one compact JSON object with exactly these keys: "
                "decision, suggested_name, reason, evidence_turn_ids, confidence, "
                "dedup_key, proposal_text. Do not use Markdown fences."
            ),
        },
        {"role": "user", "content": evidence},
    ]
    try:
        result = llm.complete(
            messages,
            temperature=0.0,
            max_tokens=500,
            timeout=EVALUATION_TIMEOUT_SECONDS,
            fail_fast=True,
            purpose="creation_opportunity_checkpoint_json",
            auxiliary_task=AUXILIARY_TASK_NAME,
        )
    except Exception as fast_error:
        if not _FAST_ROUTE_UNAVAILABLE_RE.search(str(fast_error)):
            logger.warning(
                "creation opportunity fast-model checkpoint failed",
                exc_info=True,
            )
            return None
        logger.warning(
            "creation opportunity fast-model route unavailable; retrying once "
            "on the active main model: %s",
            fast_error,
        )
        try:
            result = llm.complete(
                messages,
                temperature=0.0,
                max_tokens=500,
                timeout=MAIN_MODEL_FALLBACK_TIMEOUT_SECONDS,
                fail_fast=True,
                purpose="creation_opportunity_checkpoint_main_fallback",
            )
        except Exception:
            logger.warning(
                "creation opportunity main-model fallback failed",
                exc_info=True,
            )
            return None

    try:
        parsed = _parse_detector_json(result.text)
        logger.info(
            "creation opportunity JSON decision=%s confidence=%s title=%s "
            "provider=%s model=%s",
            parsed.get("decision") if parsed else None,
            parsed.get("confidence") if parsed else None,
            _text(parsed.get("suggested_name"), 80) if parsed else "",
            getattr(result, "provider", ""),
            getattr(result, "model", ""),
        )
        if parsed is not None:
            return parsed
    except Exception:
        logger.warning("creation opportunity JSON checkpoint parse failed", exc_info=True)
        return None


def _parse_detector_json(value: Any) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_candidate(
    args: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    decision = _normalize_creation_type(
        args.get("decision") or args.get("creation_type")
    )
    if decision == "none":
        return None, "none"
    if decision not in CREATION_TYPES:
        return None, "unsupported_creation_type"

    suggested_name = _text(args.get("suggested_name"), 80)
    reason = _text(args.get("reason"), 400)
    proposal_text = _text(args.get("proposal_text"), 500)
    try:
        confidence = float(args.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < MIN_CONFIDENCE:
        return None, "confidence_below_threshold"
    if not suggested_name or not reason or not proposal_text:
        return None, "missing_candidate_fields"

    evidence_turn_ids = args.get("evidence_turn_ids")
    if not isinstance(evidence_turn_ids, list):
        evidence_turn_ids = []
    evidence_turn_ids = [
        _text(value, 80) for value in evidence_turn_ids[:8] if _text(value, 80)
    ]
    dedup_key = _semantic_dedup_key(args.get("dedup_key"), decision, suggested_name)
    return {
        "creation_type": decision,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence_turn_ids": evidence_turn_ids,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
        "current_request": _text(state.get("last_user_message"), 1000),
        "source_turn_id": _text(state.get("last_turn_id"), 160),
    }, "candidate"


def _proposal_payload(candidate: dict[str, Any], *, status: str) -> dict[str, Any]:
    current_request = (
        candidate.get("current_request") or "the current request already in context"
    )
    return {
        "status": status,
        "decision": candidate["creation_type"],
        "suggested_name": candidate["suggested_name"],
        "delivery": "deferred_to_transform_hook"
        if status == "proposal_ready"
        else "not_displayed",
        "next_step": (
            "Now complete the user's current task in full. A placeholder such as 'done' or "
            "'ready' is not a deliverable. Do not mention, quote, or paraphrase the creation "
            f"candidate. Current request: {current_request}"
        ),
    }


def _consider_candidate(
    session_id: str, args: dict[str, Any], now: float
) -> dict[str, Any]:
    if _is_session_muted(session_id):
        return {"status": "candidate_recorded", "reason": "session_muted"}
    with _state_lock:
        state = _state_locked(session_id, now)
        candidate, reason = _normalize_candidate(args, state)
        state["last_candidate"] = dict(candidate) if candidate else None
    if candidate is None:
        return {
            "status": "no_candidate" if reason == "none" else "not_proposed",
            "reason": reason,
        }

    if _is_dismissed(session_id, candidate["dedup_key"], now):
        return {"status": "candidate_recorded", "reason": "dismissed"}
    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return _proposal_payload(candidate, status="candidate_recorded") | {
                "reason": "prompt_cooldown"
            }
    if not _claim_proposal(session_id, candidate["dedup_key"], now):
        return {"status": "candidate_recorded", "reason": "recent_duplicate"}
    if not _claim_prompt_slot(session_id, candidate, now):
        return _proposal_payload(candidate, status="candidate_recorded") | {
            "reason": "prompt_cooldown"
        }
    return _proposal_payload(candidate, status="proposal_ready")


def _parse_recommendation_response(user_message: str) -> dict[str, Any] | None:
    match = _RECOMMENDATION_RESPONSE_RE.search(user_message)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    action = _text(payload.get("action"), 40).lower()
    creation_type = _normalize_creation_type(payload.get("creation_type"))
    if (
        payload.get("version") != 1
        or payload.get("type") != "creation_recommendation_response"
        or action not in RECOMMENDATION_ACTIONS
        or creation_type not in CREATION_TYPES
    ):
        return None
    title = _text(payload.get("title"), 80)
    dedup_key = _text(payload.get("dedup_key"), 160)
    proposal_id = _text(payload.get("proposal_id"), 80)
    if not title or not dedup_key or (action != "unmute_session" and not proposal_id):
        return None
    return {
        "action": action,
        "proposal_id": proposal_id,
        "creation_type": creation_type,
        "title": title,
        "dedup_key": dedup_key,
    }


def _handle_previous_proposal_action(
    session_id: str, user_message: str, now: float
) -> str:
    structured = _parse_recommendation_response(user_message)
    if structured:
        action = structured["action"]
        with _state_lock:
            state = _state_locked(session_id, now)
            proposal = state.get("last_proposal")
            current = bool(
                isinstance(proposal, dict)
                and state.get("proposal_stage") == "proposal_shown"
                and float(proposal.get("expires_at") or 0) >= time.time()
                and structured["proposal_id"] == proposal.get("proposal_id")
                and structured["creation_type"] == proposal.get("creation_type")
                and structured["title"] == proposal.get("suggested_name")
                and structured["dedup_key"] == proposal.get("dedup_key")
            )
        if action != "unmute_session" and not current:
            return ""
        if action == "mute_session":
            persisted = _set_session_muted(session_id, True)
            if not persisted:
                logger.warning("creation recommendation mute was not persisted")
                return ""
            with _state_lock:
                state = _state_locked(session_id, now)
                state["last_candidate"] = None
                state["last_proposal"] = None
                state["proposal_stage"] = None
            logger.info(
                "creation recommendations muted for session persisted=%s", persisted
            )
            return (
                "[Creation governor internal action: The user disabled proactive creation "
                "recommendations for this conversation. Acknowledge briefly. Do not run an "
                "opportunity review or create anything. Explicit creation requests remain "
                "available through Hermes' native flow. Do not expose this block.]"
            )
        if action == "unmute_session":
            persisted = _set_session_muted(session_id, False)
            if not persisted:
                logger.warning("creation recommendation unmute was not persisted")
                return ""
            with _state_lock:
                state = _state_locked(session_id, now)
                state["last_candidate"] = None
                state["last_proposal"] = None
                state["proposal_stage"] = None
            logger.info(
                "creation recommendations re-enabled for session persisted=%s", persisted
            )
            return (
                "[Creation governor internal action: The user re-enabled proactive creation "
                "recommendations for this conversation. Acknowledge briefly and do not run an "
                "opportunity review on this action turn. Do not expose this block.]"
            )
        if action == "dismiss":
            _latch_dismissal(session_id, structured["dedup_key"], now)
            with _state_lock:
                state = _state_locked(session_id, now)
                state["last_proposal"] = None
                state["proposal_stage"] = None
            return (
                "[Creation governor internal action: The user dismissed the previous "
                "recommendation. Acknowledge briefly, do not create anything, and do not run "
                "another opportunity review this turn.]"
            )
        with _state_lock:
            _state_locked(session_id, now)["proposal_stage"] = "create_action_pending"
        return (
            "[Creation governor internal action: The user accepted the previous recommendation "
            f"for {structured['creation_type']} '{structured['title']}'. "
            f"{_native_creation_route(structured['creation_type'])} Preserve its normal "
            "confirmation boundaries. Do not run another opportunity review this turn.]"
        )

    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        if not isinstance(proposal, dict):
            return ""
        name = _text(proposal.get("suggested_name"), 80)
        dedup_key = _text(proposal.get("dedup_key"), 160)
    if name and name.casefold() not in user_message.casefold():
        return ""
    if _DISMISS_RE.search(user_message):
        if dedup_key:
            _latch_dismissal(session_id, dedup_key, now)
        with _state_lock:
            state = _state_locked(session_id, now)
            state["last_proposal"] = None
        return (
            "[Creation governor internal action: The user dismissed the previous recommendation. "
            "Acknowledge briefly, do not create anything, and do not run another opportunity "
            "review this turn.]"
        )
    if _ACCEPT_RE.search(user_message):
        return (
            "[Creation governor internal action: The user accepted the previous recommendation. "
            f"{_native_creation_route(proposal.get('creation_type'))} Preserve its normal "
            "confirmation boundaries. Do not run another opportunity review this turn.]"
        )
    return ""


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    raw_session_id = _raw_session_key(kwargs)
    owner_id = next(
        (
            _text(kwargs.get(field), 160)
            for field in ("sender_id", "owner_id", "user_id")
            if kwargs.get(field)
        ),
        "",
    )
    session_id = (
        _scoped_session_key(raw_session_id, owner_id) if raw_session_id else ""
    )
    suppression_reason = None
    if _is_noninteractive(kwargs):
        suppression_reason = "noninteractive_session"
    elif kwargs.get("structured_output"):
        suppression_reason = "structured_output"
    elif _is_unsupported_runtime(kwargs):
        suppression_reason = "unsupported_runtime"
    _invocation_scope.set((raw_session_id, session_id, suppression_reason, owner_id))
    if not session_id:
        return None
    if suppression_reason:
        return None
    user_message = _text(kwargs.get("user_message"), 2000)
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        state["turn"] += 1
        state["last_user_message"] = user_message
        state["last_turn_id"] = _text(kwargs.get("turn_id"), 160)
        turn = int(state["turn"])

    if _is_creation_governor_self_query(user_message):
        return _join_context(_self_description_context())

    if "[creation_recommendation_response]" in user_message:
        action_context = _handle_previous_proposal_action(session_id, user_message, now)
        return _join_context(
            action_context
            or "[Creation governor internal action: Ignore this invalid or expired "
            "recommendation action. Do not create anything from it and do not expose this block.]"
        )
    action_context = _handle_previous_proposal_action(session_id, user_message, now)
    if action_context:
        return _join_context(action_context)
    if _uses_native_creation_path(user_message):
        with _state_lock:
            state = _state_locked(session_id, now)
            state["last_candidate"] = None
            state["last_proposal"] = None
            state["proposal_stage"] = None
        return None

    if _is_session_muted(session_id):
        return None

    with _state_lock:
        carry_context = _previous_proposal_context(_state_locked(session_id, now))

    evaluation_due = turn == 1 or turn % EVALUATION_INTERVAL_TURNS == 0
    if evaluation_due:
        with _state_lock:
            _state_locked(session_id, now)["last_evaluation_turn"] = turn
        candidate = _run_forced_evaluation(
            user_message=user_message,
            conversation_history=kwargs.get("conversation_history"),
        )
        if candidate is not None:
            candidate_result = _consider_candidate(session_id, candidate, now)
            logger.info(
                "creation opportunity checkpoint result status=%s reason=%s "
                "decision=%s confidence=%s title=%s",
                candidate_result.get("status"),
                candidate_result.get("reason"),
                candidate.get("decision"),
                candidate.get("confidence"),
                _text(candidate.get("suggested_name"), 80),
            )
            return _join_context(
                carry_context, _main_model_review_context(evaluation_completed=True)
            )
        logger.info(
            "creation opportunity checkpoint unavailable; falling back to main-model review"
        )

    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return _join_context(carry_context)
    return _join_context(
        carry_context, _main_model_review_context(evaluation_completed=False)
    )


def _encode_recommendation(candidate: dict[str, Any]) -> str:
    action_label, action_consequence = _recommendation_action_copy(candidate)
    payload = {
        "version": 1,
        "type": "creation_recommendation",
        "proposal_id": candidate["proposal_id"],
        "expires_at": candidate["expires_at"],
        "creation_type": candidate["creation_type"],
        "title": candidate["suggested_name"],
        "reason": f"{candidate['reason']} {action_consequence}",
        "proposal_text": candidate["proposal_text"],
        "action_label": action_label,
        "action_consequence": action_consequence,
        "dedup_key": candidate["dedup_key"],
        "confidence": candidate["confidence"],
        "evidence_turn_ids": candidate.get("evidence_turn_ids") or [],
        "source_turn_id": candidate.get("source_turn_id") or "",
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _recommendation_action_copy(candidate: dict[str, Any]) -> tuple[str, str]:
    creation_type = candidate["creation_type"]
    sample = (
        str(candidate.get("suggested_name") or "")
        + str(candidate.get("reason") or "")
        + str(candidate.get("proposal_text") or "")
    )
    if _CJK_RE.search(sample):
        labels = {
            "agent": "创建助手",
            "skill": "创建 Skill",
            "task": "设置定时任务",
        }
        consequences = {
            "agent": "接受后会进入“创建助手”流程，并在真正创建前让你确认配置。",
            "skill": "接受后会进入“创建 Skill”流程，并在真正创建前让你确认配置。",
            "task": "接受后会进入“设置定时任务”流程，并在真正创建前确认执行时间。",
        }
    else:
        labels = {
            "agent": "Create assistant",
            "skill": "Create Skill",
            "task": "Set up scheduled task",
        }
        consequences = {
            "agent": (
                "Accepting opens the native assistant creation flow and asks you to "
                "confirm the configuration before creation."
            ),
            "skill": (
                "Accepting opens the native Skill creation flow and asks you to "
                "confirm the configuration before creation."
            ),
            "task": (
                "Accepting opens the native scheduled-task flow and asks you to "
                "confirm the execution time before creation."
            ),
        }
    return labels[creation_type], consequences[creation_type]


def _fallback_text(candidate: dict[str, Any]) -> str:
    creation_type = candidate["creation_type"]
    name = candidate["suggested_name"]
    reason = candidate["reason"]
    proposal_text = candidate["proposal_text"]
    action_label, action_consequence = _recommendation_action_copy(candidate)
    chinese = bool(_CJK_RE.search(name + reason + proposal_text))
    if chinese:
        label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
        return (
            f"💡 可以沉淀为一个 {label}\n\n**「{name}」**\n\n{reason}\n\n"
            f"**{action_label}：** {action_consequence}\n\n{proposal_text}"
        )
    label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
    return (
        f"💡 This could become a reusable {label}\n\n**{name}**\n\n{reason}\n\n"
        f"**{action_label}:** {action_consequence}\n\n{proposal_text}"
    )


def _recommendation_envelope(candidate: dict[str, Any]) -> str:
    encoded = _encode_recommendation(candidate)
    return (
        f"<!--creation-recommendation:start {encoded}-->\n\n"
        f"{_fallback_text(candidate)}\n\n"
        "<!--creation-recommendation:end-->"
    )


def _transform_llm_output(**kwargs: Any) -> str | None:
    session_id = _session_key(kwargs)
    response_text = str(kwargs.get("response_text") or "")
    if not session_id:
        return None
    if _is_noninteractive(kwargs) or _is_unsupported_runtime(kwargs) or kwargs.get(
        "structured_output"
    ):
        return None
    if not response_text or is_intentional_silence_response(response_text):
        with _state_lock:
            state = _state_locked(session_id, time.monotonic())
            if state.get("proposal_stage") == "create_action_pending":
                state["proposal_stage"] = "proposal_shown"
        return None
    if kwargs.get("failed") or kwargs.get("interrupted") or kwargs.get("completed") is False:
        with _state_lock:
            state = _state_locked(session_id, time.monotonic())
            if state.get("proposal_stage") == "create_action_pending":
                state["proposal_stage"] = "proposal_shown"
            elif state.get("proposal_stage") == "proposal_shown":
                _discard_staged_proposal_locked(
                    session_id, state, release_claim=True
                )
        return None
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        if state.get("proposal_stage") == "create_action_pending":
            _discard_staged_proposal_locked(
                session_id, state, release_claim=False
            )
    if _is_session_muted(session_id):
        return None
    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        current_turn = int(state["turn"])
        if (
            not isinstance(proposal, dict)
            or int(state["candidate_turn"]) != current_turn
            or int(state["last_delivery_turn"]) == current_turn
        ):
            return None

    if "<!--creation-recommendation:start " in response_text:
        return None
    if _is_session_muted(session_id):
        return None
    delivery_block_reason = _response_delivery_block_reason(response_text)
    if delivery_block_reason:
        with _state_lock:
            state = _state_locked(session_id, now)
            current = state.get("last_proposal")
            if (
                isinstance(current, dict)
                and current.get("proposal_id") == proposal.get("proposal_id")
            ):
                _discard_staged_proposal_locked(
                    session_id, state, release_claim=True
                )
        logger.info(
            "creation recommendation suppressed reason=%s title=%s turn=%s",
            delivery_block_reason,
            _text(proposal.get("suggested_name"), 80),
            current_turn,
        )
        return None
    with _state_lock:
        state = _state_locked(session_id, now)
        current = state.get("last_proposal")
        if (
            not isinstance(current, dict)
            or current.get("proposal_id") != proposal.get("proposal_id")
            or int(state["last_delivery_turn"]) == current_turn
        ):
            return None
        state["last_prompt_turn"] = current_turn
        state["last_delivery_turn"] = current_turn
        state["candidate_turn"] = -10_000
    logger.info(
        "creation recommendation attached type=%s confidence=%s title=%s turn=%s",
        proposal.get("creation_type"),
        proposal.get("confidence"),
        _text(proposal.get("suggested_name"), 80),
        current_turn,
    )
    return response_text + "\n\n" + _recommendation_envelope(proposal)


def _detect_creation_opportunity(args: dict[str, Any], **kwargs: Any) -> str:
    invocation = _invocation_scope.get()
    if invocation is not None and invocation[2]:
        return json.dumps({"status": "not_proposed", "reason": invocation[2]})
    if _is_noninteractive(kwargs) or _is_unsupported_runtime(kwargs) or kwargs.get(
        "structured_output"
    ):
        return json.dumps({"status": "not_proposed", "reason": "unsupported_runtime"})
    session_id = _session_key(kwargs)
    if not session_id:
        return json.dumps({"status": "invalid", "error": "missing_session_id"})
    result = _consider_candidate(session_id, args, time.monotonic())
    return json.dumps(result, ensure_ascii=False)


def _reset_state_for_tests() -> None:
    global _plugin_llm
    with _state_lock:
        _recent_proposals.clear()
        _dismissed_proposals.clear()
        _session_states.clear()
        _muted_sessions.clear()
        _known_unmuted_sessions.clear()
    _plugin_llm = None
    _invocation_scope.set(None)


def register(ctx: Any) -> None:
    global _plugin_llm
    try:
        _plugin_llm = ctx.llm
    except Exception:
        _plugin_llm = None

    ctx.register_auxiliary_task(
        key=AUXILIARY_TASK_NAME,
        display_name="Creation opportunity checkpoint",
        description="Fast bounded Agent, Skill, Task, or none classification.",
        defaults={
            "model": AUXILIARY_MODEL_ALIAS,
            "timeout": EVALUATION_TIMEOUT_SECONDS,
        },
    )
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    ctx.register_tool(
        name=TOOL_NAME,
        toolset="creation_governor",
        schema={
            "name": TOOL_NAME,
            "description": (
                "Perform one high-recall zero-shot semantic choice among Agent, Skill, Task, and "
                "none. Do not require magic words such as repetition, saving, or creation, but "
                "require semantic evidence that durable future value continues beyond the current "
                "bounded request. Use "
                "meaning and conversation context, never topic keyword matching or memorized "
                "examples. Task means desired future time/event/background execution or keeping "
                "a derived result current as its source changes; a current data range such as "
                "today is not by itself a trigger. Prefer Task over Skill when at least two are "
                "true: the source changes over time, the result becomes stale, and automatic "
                "refresh removes repeated manual work. An explicit cadence is not required for "
                "the recommendation, must not be invented, and is collected by Hermes' native "
                "task flow after confirmation. Agent means a long-lived "
                "responsible role with retained context, judgment, autonomous tool choice, or "
                "interpretation of a changing real-world work domain. A first request may qualify "
                "only when it concerns an ongoing account, project, operation, or responsibility; "
                "the analysis verb alone is not evidence for an Agent. A request that only processes "
                "one finite file, table, questionnaire, document, import, or dataset returns none "
                "unless future recurrence, freshness, or ongoing ownership is supported. Requiring "
                "an upload, authorization, or connector to finish the current request is execution "
                "friction, not reuse evidence. "
                "If an existing Agent, Skill, scheduled Task, Dashboard, or application already "
                "does the same future job, return none unless the new object adds a materially "
                "different responsibility. Configuring, seeding, previewing, or using a newly "
                "created object is not a reason to recommend a parallel Agent or Skill. "
                "Skill means a stable reusable input-to-output method without an independent "
                "identity or a need to keep one persistent result fresh. Missing connectors or "
                "authorization affect current execution, not "
                "reuse value. Explicit creation requests use Hermes' native flow and return none. "
                "Write user-facing card copy that states what will be created, why it helps, and "
                "that acceptance enters Hermes' native creation/configuration flow; never expose "
                "internal reasoning such as 'the user...'. "
                "Call only between scheduled checkpoints when one unusually clear opportunity "
                "emerges. The tool never creates and may return none. The plugin owns cooldown, "
                "deduplication, dismissal, and conditional card/text delivery only after the "
                "current task has actually delivered a result."
            ),
            "parameters": _DETECTOR_SCHEMA,
        },
        handler=_detect_creation_opportunity,
        description="Detect one reusable creation opportunity or none",
        emoji="💡",
    )
