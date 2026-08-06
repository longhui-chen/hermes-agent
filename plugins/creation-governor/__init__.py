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
PLUGIN_VERSION = "0.9.0"
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
# Connection recommendations (Zettlab 需求 2/5)：channel/connector 走结构化
# attachment 通道（ctx.emit_attachment → channel.connect / connector.connect 卡），
# 与 agent/skill/task 的文本信封通道并行；共用同一套评估节奏 / 冷却 / 去重 /
# 拒绝闩锁（展示节奏三控不变）。
CONNECTION_TYPES = {"channel", "connector"}
# 可推荐的 IM 渠道 kind 白名单：App/Web 绑定向导都支持的交集。连接态与区域可连
# 范围来自 local-server 真实清单（list_my_channels 的 installed_channels +
# available_kinds，后者已按设备区域过滤，CN 设备不含 telegram/discord/slack）；
# 这里只约束"平台支持范围"，老版本 local-server 无 available_kinds 时作全集兜底。
RECOMMENDABLE_CHANNEL_KINDS = {"feishu", "wecom", "wechat", "telegram", "discord", "slack"}
# connector 连接态里视为"未连接、可推荐"的状态值（projection UnifiedAuthState 的窄投影）。
CONNECTOR_RECOMMENDABLE_STATES = {"not_connected", "expired", "revoked", "disconnected"}
ARTIFACT_TYPE = "artifact"
# attachment 通道交付的全部品类（连接推荐 + artifact 推荐）；agent/skill/task
# 保持文本信封通道不变。
ATTACHMENT_DELIVERED_TYPES = CONNECTION_TYPES | {ARTIFACT_TYPE}
CONNECTION_INVENTORY_TTL_SECONDS = 600.0
MAX_EMITTED_CONNECTION_PROPOSALS = 256
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
# attachment_id → (scoped_session_key, dedup_key)：attachment_action hook 的
# dismiss 回执靠它落 30 天闩锁（有界，最老先逐出）。
_emitted_connection_proposals: OrderedDict[str, tuple[str, str]] = OrderedDict()
_state_lock = threading.Lock()
_plugin_llm: Any = None
_plugin_ctx: Any = None
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
_RECOMMENDATION_RESPONSE_RE = re.compile(
    r"\[creation_recommendation_response\]\s*(\{.*?\})\s*"
    r"\[/creation_recommendation_response\]",
    re.DOTALL,
)
_ACTION_RESULT_ENVELOPE_RE = re.compile(
    r"<!--creation-recommendation-action-result\s+[A-Za-z0-9_-]+\s*-->"
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
    prefix = (
        creation_type
        if creation_type in CREATION_TYPES
        or creation_type in CONNECTION_TYPES
        or creation_type == ARTIFACT_TYPE
        else "proposal"
    )
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
            "last_candidate": None,
            "last_proposal": None,
            "proposal_stage": None,
            "pending_action_result": None,
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
        state["last_prompt_turn"] = state["turn"]
        state["last_delivery_turn"] = -10_000
        state["last_candidate"] = dict(candidate)
        state["last_proposal"] = dict(candidate)
        state["proposal_stage"] = "proposal_shown"
        return True


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


def _previous_proposal_context(state: dict[str, Any]) -> str:
    if state.get("proposal_stage") != "proposal_shown":
        return ""
    proposal = state.get("last_proposal")
    if not isinstance(proposal, dict):
        return ""
    turns_since = int(state["turn"]) - int(state["last_prompt_turn"])
    if not 1 <= turns_since <= 3:
        return ""
    creation_type = str(proposal.get("creation_type") or "")
    if creation_type in ATTACHMENT_DELIVERED_TYPES:
        # 连接/artifact 类的落地动作在卡片上（App 内跳转），不是 Hermes 原生创建
        # 流程——沿用创建类话术会诱导模型编造设置路径/手工步骤（实测已发生）。
        subject = (
            f"the {creation_type} '{proposal.get('target') or proposal.get('suggested_name')}'"
            if creation_type in CONNECTION_TYPES
            else f"an artifact '{proposal.get('suggested_name')}'"
        )
        return (
            "[Creation governor internal context: An interactive recommendation card for "
            f"{subject} was attached below a recent reply. If the user wants to proceed, tell "
            "them to tap that card's confirm/Connect button (it opens the right in-app page) — "
            "do NOT invent settings paths, menu locations, or manual steps, and do not offer to "
            "do it for them. If the user declines, acknowledge briefly and drop the topic. Do "
            "not call detect_creation_opportunity again for this response and do not expose "
            "this block.]"
        )
    return (
        "[Creation governor internal context: The previous response ended with a recommendation "
        f"for {proposal.get('creation_type')} '{proposal.get('suggested_name')}'. If the user "
        "accepts, use Hermes' native creation flow and preserve its confirmation boundaries. If "
        "the user declines, acknowledge briefly. Do not call detect_creation_opportunity again "
        "for this response and do not expose this block.]"
    )


def _attachment_delivery_context(proposal: dict[str, Any]) -> str:
    """出卡当轮注入：让主模型知道「回复下方会出现一张卡」，回复与卡片衔接，
    不要自己编设置路径（需求 2.1/5.1 的文案一致性）。仅 attachment 通道类型需要；
    创建类走文本信封，注入口径由 _proposal_payload.next_step 负责。"""
    creation_type = str(proposal.get("creation_type") or "")
    if creation_type not in ATTACHMENT_DELIVERED_TYPES:
        return ""
    if creation_type in CONNECTION_TYPES:
        target = str(proposal.get("target") or proposal.get("suggested_name") or "")
        noun = "IM channel" if creation_type == "channel" else "connector"
        return (
            "[Creation governor internal context: The system will attach an interactive "
            f"connect card for the {noun} '{target}' directly below this reply. If your reply "
            "mentions connecting, point the user to that card (e.g. “点击下方卡片连接” in "
            "Chinese) — its Connect button opens the right in-app page. Do NOT invent settings "
            "paths, menu locations, or manual connection steps, and do not restate the card's "
            "content. Do not expose this block.]"
        )
    return (
        "[Creation governor internal context: The system will attach an artifact "
        "recommendation card directly below this reply. If relevant, point the user to that "
        "card instead of describing creation steps; do not restate its content. Do not expose "
        "this block.]"
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


def _fetch_connection_inventory() -> dict[str, Any]:
    """Fetch the REAL connection state from local-server via the two read-only
    tools (需求 2.2：可推荐范围来自实际清单，不由模型猜测)。

    Any failure degrades to ``fetched=False`` — connection recommendations are
    then disabled for the round instead of blocking or guessing.  Runs inside
    the evaluation checkpoint only (first turn + every third turn), and the
    result is cached per session for CONNECTION_INVENTORY_TTL_SECONDS.
    """
    inventory: dict[str, Any] = {
        "fetched": False,
        "channels_connected": [],
        "channels_recommendable": [],
        "connectors_connected": [],
        "connectors_recommendable": [],
    }
    try:
        from tools.list_my_channels_tool import (
            _check_list_my_channels,
            list_my_channels_tool,
        )

        if _check_list_my_channels():
            parsed = json.loads(list_my_channels_tool({}))
            channels = parsed.get("installed_channels")
            if isinstance(channels, list):
                connected = set()
                for item in channels:
                    if not isinstance(item, dict):
                        continue
                    kind = _text(
                        item.get("kind") or item.get("channel_kind") or item.get("platform"),
                        40,
                    ).lower()
                    if kind:
                        connected.add(kind)
                inventory["channels_connected"] = sorted(connected)
                available = parsed.get("available_kinds")
                if isinstance(available, list):
                    # 新版 local-server 返回区域感知的可连清单（Supported 且未连，
                    # 例如 CN 设备不含 telegram/discord/slack）；推荐范围 = 可连 ∩
                    # 平台白名单。老版本无此字段时降级回「白名单 − 已连」旧公式。
                    kinds = {
                        _text(value, 40).lower()
                        for value in available
                        if _text(value, 40)
                    }
                    inventory["channels_recommendable"] = sorted(
                        (kinds & RECOMMENDABLE_CHANNEL_KINDS) - connected
                    )
                else:
                    inventory["channels_recommendable"] = sorted(
                        RECOMMENDABLE_CHANNEL_KINDS - connected
                    )
                inventory["fetched"] = True
    except Exception:
        logger.debug("connection inventory: channel fetch failed", exc_info=True)
    try:
        from tools.list_my_connectors_tool import (
            _check_list_my_connectors,
            list_my_connectors_tool,
        )

        if _check_list_my_connectors():
            parsed = json.loads(list_my_connectors_tool({}))
            connectors = parsed.get("connectors")
            if isinstance(connectors, list):
                connected: list[str] = []
                recommendable: list[str] = []
                for item in connectors:
                    if not isinstance(item, dict):
                        continue
                    provider = _text(item.get("provider"), 80).lower()
                    state = _text(item.get("state"), 40).lower()
                    if not provider:
                        continue
                    if state in CONNECTOR_RECOMMENDABLE_STATES:
                        recommendable.append(provider)
                    else:
                        connected.append(provider)
                inventory["connectors_connected"] = sorted(set(connected))
                inventory["connectors_recommendable"] = sorted(set(recommendable))
                inventory["fetched"] = True
    except Exception:
        logger.debug("connection inventory: connector fetch failed", exc_info=True)
    return inventory


def _connection_inventory(session_id: str, now: float) -> dict[str, Any]:
    with _state_lock:
        state = _state_locked(session_id, now)
        cached = state.get("connection_inventory")
        if (
            isinstance(cached, dict)
            and now - float(cached.get("_at") or float("-inf")) < CONNECTION_INVENTORY_TTL_SECONDS
        ):
            return cached
    inventory = _fetch_connection_inventory()
    inventory["_at"] = now
    with _state_lock:
        _state_locked(session_id, now)["connection_inventory"] = inventory
    return inventory


def _connection_inventory_context(inventory: dict[str, Any]) -> str:
    if not inventory.get("fetched"):
        return (
            "[connection-inventory] unavailable — channel and connector "
            "decisions are forbidden this round."
        )
    return (
        "[connection-inventory] "
        f"channels connected: {', '.join(inventory['channels_connected']) or '(none)'}; "
        f"channels recommendable: {', '.join(inventory['channels_recommendable']) or '(none)'}; "
        f"connectors connected: {', '.join(inventory['connectors_connected']) or '(none)'}; "
        f"connectors recommendable: {', '.join(inventory['connectors_recommendable']) or '(none)'}"
    )


_DETECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["agent", "skill", "task", "channel", "connector", "artifact", "none"],
        },
        "suggested_name": {"type": "string"},
        "reason": {"type": "string"},
        "target": {"type": "string"},
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

Return exactly one of agent, skill, task, channel, connector, artifact, or none. Do not classify by topic words and do not use
memorized examples. A single substantive request is enough when a reasonable user would benefit
from reusing the capability. Do not require the user to mention repetition, frequency, saving, or
creation. Ask whether a durable capability would materially reduce friction or improve judgment
the next time a related need appears.

Definitions and conflict order:
1. task: the desired future value depends on a recurring time trigger, event trigger, background
   monitoring, or repeated refresh of new information. A word such as 'today' that merely scopes
   the current data is not by itself a future trigger.
2. agent: future work needs a long-lived responsible role, retained domain context, judgment,
   autonomous choice among tools, decisions about the next step, or repeated interpretation of a
   changing real-world business domain, account, operation, project, or body of evidence.
3. skill: future inputs vary but a stable input-to-output method can be reused without an
   independent identity or durable state.
4. none: small talk, a trivial transformation, a low-value closed-world fact lookup, an explicit
   request to create/configure/schedule something through Hermes' native flow, or no reasonable
   reuse value.
5. channel: the durable value of this need depends on reminders, results, or notifications
   reaching the user inside an IM app, and a `[connection-inventory]` line in the evidence lists
   that channel kind under "channels recommendable". Set target to that exact channel kind.
6. connector: completing this class of request materially needs the user's own external data
   (mail, notes, code, calendar, ...) and the inventory lists that provider under "connectors
   recommendable". Set target to that exact provider id.

7. artifact: the most valuable durable outcome of this conversation is an openable product —
   a page, mini-app, dashboard, or report the user would revisit or share — rather than a
   capability. Prefer skill when the value is a reusable method; prefer artifact when the value
   is the produced thing itself. suggested_name is the artifact title in the user's language.

Grounding rule for channel/connector: these two decisions are FORBIDDEN unless the evidence
contains a `[connection-inventory]` line that explicitly lists the target as recommendable.
Never invent, guess, or generalize a channel kind or provider that is not in the inventory;
already-connected entries must never be recommended again. When both a creation decision
(agent/skill/task) and a connection decision seem plausible, prefer the one the user most
needs next; never return multiple objects. channel/connector recommendations are delivered
as a card by the client — proposal_text should be one sentence asking whether to connect.

High-recall boundary: a substantive request to inspect, compare, diagnose, research, optimize, or
make a judgment about an ongoing external work domain should normally be agent rather than none,
even on the first request and even when the requested snapshot is scoped to today/current/latest.
Choose none only when reuse value is genuinely absent, not merely unstated.

Apply this semantic gate before returning none. Ask, in order: (a) will the underlying information,
account, project, or operating environment change after this turn; (b) would a responsible role with
retained context make a future judgment better; (c) would a stable method save meaningful effort on
a different future input? If any answer is yes, none is forbidden: choose task for a future trigger,
otherwise agent for continuing ownership/judgment, otherwise skill for the reusable method. Ambiguity
about whether the user will repeat the request is not evidence for none. Do not reduce an analytical
request to a fact lookup merely because the current data or connector is unavailable.

Judge reuse value separately from current execution availability. Missing authorization,
connectors, data, or tools may block today's execution but is not a reason to ignore a clear
long-term need. Recommend only the first-layer object the user most needs, never multiple objects.
Match the user's language. For a positive decision, provide a concise name, concrete reason,
one-sentence optional proposal_text asking whether to create it, confidence, a stable semantic
dedup_key, and evidence_turn_ids chosen only from the supplied labels. For none, use empty strings,
an empty evidence list, and confidence 0. Never claim anything was created.

中文请求必须按同一套语义规则判断，不要因为用户没有说“重复”“以后”“保存”或“创建”就返回
none。先判断需求所涉及的账户、项目、业务环境或信息是否会继续变化；如果会变化且后续判断需要
保留背景、综合数据或自主选择工具，选择 agent。如果价值来自未来的时间、事件、后台监控或提醒，
选择 task。如果输入会变化但处理方法相对稳定，选择 skill。只有寒暄、低价值封闭事实、微小的一次性
转换、用户已经明确要求创建，或 Agent/Skill/Task 三种长期价值都确实不存在时，才选择 none。
“今天”“最近”“当前”只是本次数据范围，不等于没有长期价值。缺少授权、连接器或数据只影响本次
执行，不能作为返回 none 的理由。名称、原因和询问是否创建的 proposal_text 使用用户的语言。

channel 与 connector 的中文规则相同：只有当证据里存在 [connection-inventory] 行、且目标
明确出现在 recommendable 列表中时才允许返回这两类；已连接的渠道或数据源绝不重复推荐；
target 必须逐字取自清单，禁止猜测或泛化。channel 用于"提醒/结果需要直达用户的 IM"，
connector 用于"这类任务实质上需要用户自己的外部数据"。artifact 用于"这段对话最有价值的
沉淀是一件可打开的作品（页面/小应用/报告）而非一种能力"——方法可复用选 skill，产物本身
有长期价值选 artifact。"""


def _run_forced_evaluation(
    *,
    user_message: str,
    conversation_history: Any,
    connection_context: str = "",
) -> dict[str, Any] | None:
    llm = _plugin_llm
    if llm is None:
        return None
    evidence = _conversation_evidence(conversation_history, user_message)
    if connection_context:
        evidence = f"{evidence}\n{connection_context}"

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
    if (
        decision not in CREATION_TYPES
        and decision not in CONNECTION_TYPES
        and decision != ARTIFACT_TYPE
    ):
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

    target = ""
    if decision in CONNECTION_TYPES:
        # 事后过滤是硬闸（prompt 只是引导）：target 必须逐字命中真实库存的
        # recommendable 集合；库存缺失/为空 → 该轮禁止连接类推荐。
        inventory = state.get("connection_inventory") or {}
        target = _text(args.get("target"), 80).lower()
        pool_key = (
            "channels_recommendable" if decision == "channel" else "connectors_recommendable"
        )
        pool = inventory.get(pool_key) if inventory.get("fetched") else None
        if not target or not isinstance(pool, list) or target not in pool:
            return None, "connection_target_unavailable"

    evidence_turn_ids = args.get("evidence_turn_ids")
    if not isinstance(evidence_turn_ids, list):
        evidence_turn_ids = []
    evidence_turn_ids = [
        _text(value, 80) for value in evidence_turn_ids[:8] if _text(value, 80)
    ]
    if decision in CONNECTION_TYPES:
        # 连接类账本键与模型给的 dedup_key 解耦：同一渠道/数据源 30 天拒绝
        # 闩锁必须稳定命中，不能被下次评估换个说法绕开。
        dedup_key = _semantic_dedup_key(target, decision, suggested_name)
    else:
        dedup_key = _semantic_dedup_key(args.get("dedup_key"), decision, suggested_name)
    return {
        "creation_type": decision,
        "suggested_name": suggested_name,
        "reason": reason,
        "evidence_turn_ids": evidence_turn_ids,
        "confidence": confidence,
        "dedup_key": dedup_key,
        "proposal_text": proposal_text,
        **({"target": target} if target else {}),
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
            f"for {structured['creation_type']} '{structured['title']}'. Continue through "
            "Hermes' native creation flow, preserving its normal clarification and confirmation "
            "boundaries. Do not run another opportunity review this turn.]"
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
            "Continue through Hermes' native creation flow, preserving its normal clarification "
            "and confirmation boundaries. Do not run another opportunity review this turn.]"
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
        structured = _parse_recommendation_response(user_message)
        action_context = _handle_previous_proposal_action(session_id, user_message, now)
        if structured:
            with _state_lock:
                state = _state_locked(session_id, now)
                state["pending_action_result"] = {
                    "proposal_id": structured["proposal_id"],
                    "action": structured["action"],
                    "status": "accepted" if action_context else "rejected",
                }
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
        inventory = _connection_inventory(session_id, now)
        candidate = _run_forced_evaluation(
            user_message=user_message,
            conversation_history=kwargs.get("conversation_history"),
            connection_context=_connection_inventory_context(inventory),
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
            delivery_context = ""
            if candidate_result.get("status") == "proposal_ready":
                with _state_lock:
                    proposal = _state_locked(session_id, now).get("last_proposal")
                if isinstance(proposal, dict):
                    delivery_context = _attachment_delivery_context(proposal)
            return _join_context(
                carry_context,
                _main_model_review_context(evaluation_completed=True),
                delivery_context,
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
    payload = {
        "version": 1,
        "type": "creation_recommendation",
        "proposal_id": candidate["proposal_id"],
        "expires_at": candidate["expires_at"],
        "creation_type": candidate["creation_type"],
        "title": candidate["suggested_name"],
        "reason": candidate["reason"],
        "dedup_key": candidate["dedup_key"],
        "confidence": candidate["confidence"],
        "evidence_turn_ids": candidate.get("evidence_turn_ids") or [],
        "source_turn_id": candidate.get("source_turn_id") or "",
        "action_receipts": True,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fallback_text(candidate: dict[str, Any]) -> str:
    creation_type = candidate["creation_type"]
    name = candidate["suggested_name"]
    reason = candidate["reason"]
    proposal_text = candidate["proposal_text"]
    chinese = bool(_CJK_RE.search(name + reason + proposal_text))
    if chinese:
        label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
        return f"💡 可以沉淀为一个 {label}\n\n**「{name}」**\n\n{reason}\n\n{proposal_text}"
    label = {"agent": "Agent", "skill": "Skill", "task": "Task"}[creation_type]
    return f"💡 This could become a reusable {label}\n\n**{name}**\n\n{reason}\n\n{proposal_text}"


def _recommendation_envelope(candidate: dict[str, Any]) -> str:
    encoded = _encode_recommendation(candidate)
    return (
        f"<!--creation-recommendation:start {encoded}-->\n\n"
        f"{_fallback_text(candidate)}\n\n"
        "<!--creation-recommendation:end-->"
    )


def _action_result_envelope(result: dict[str, Any]) -> str:
    payload = {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": result["proposal_id"],
        "action": result["action"],
        "status": result["status"],
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"<!--creation-recommendation-action-result {encoded}-->"


def _emit_recommendation_attachment(session_key: str, proposal: dict[str, Any]) -> bool:
    """Deliver a channel/connector/artifact proposal as a structured attachment.

    需求 6.2：wire 只带语义（kind/payload/action id），推荐卡文案由客户端
    i18n 决定（artifact 的 title/reason 是模型按用户语言产出的内容字段）。
    发射失败（无活跃流 / 老客户端链路）静默降级——推荐是锦上添花，绝不
    进入正文文本通道。
    """
    ctx = _plugin_ctx
    if ctx is None or not hasattr(ctx, "emit_attachment"):
        return False
    creation_type = proposal.get("creation_type")
    target = _text(proposal.get("target"), 80).lower()
    proposal_id = _text(proposal.get("proposal_id"), 80)
    if creation_type not in ATTACHMENT_DELIVERED_TYPES or not proposal_id:
        return False
    if creation_type in CONNECTION_TYPES and not target:
        return False
    attachment_id = f"cg-{proposal_id}"
    if creation_type == "channel":
        kind = "channel.connect"
        payload: dict[str, Any] = {"channel_kind": target}
        actions = [{"id": "dismiss"}, {"id": "connect", "style": "primary"}]
    elif creation_type == "connector":
        kind = "connector.connect"
        # 推荐永远是非阻塞的（需求 5.1）；强依赖场景的 blocking 卡由执行路径
        # 自己发，不走推荐通道。
        payload = {"provider": target, "blocking": False}
        actions = [{"id": "dismiss"}, {"id": "connect", "style": "primary"}]
    else:
        kind = "artifact.recommendation"
        payload = {
            "title": _text(proposal.get("suggested_name"), 80),
            "reason": _text(proposal.get("reason"), 400),
            "confidence": proposal.get("confidence"),
        }
        actions = [{"id": "dismiss"}, {"id": "accept", "style": "primary"}]
    expires_at = proposal.get("expires_at")
    attachment = {
        "id": attachment_id,
        "kind": kind,
        "v": 1,
        "state": "active",
        "payload": payload,
        "actions": actions,
        "dedup_key": proposal.get("dedup_key") or "",
        **(
            {"expires_at": int(float(expires_at) * 1000)}
            if isinstance(expires_at, (int, float)) and expires_at > 0
            else {}
        ),
    }
    try:
        emitted = bool(ctx.emit_attachment(attachment))
    except Exception:
        logger.warning("connection recommendation emit failed", exc_info=True)
        return False
    if emitted:
        with _state_lock:
            _emitted_connection_proposals[attachment_id] = (
                session_key,
                str(proposal.get("dedup_key") or ""),
            )
            while len(_emitted_connection_proposals) > MAX_EMITTED_CONNECTION_PROPOSALS:
                _emitted_connection_proposals.popitem(last=False)
    return emitted


def _on_attachment_action(**kwargs: Any) -> None:
    """attachment_action hook：连接推荐卡的回执入账本。

    dismiss → 30 天拒绝闩锁（同 dedup_key 不再推荐）；connect → 不闩锁——
    授权完成后库存自然把该目标移出 recommendable，未完成则冷却窗口后允许
    再推。回执与发射同进程（per-agent gateway），映射表按 attachment_id 定位。
    """
    attachment_id = _text(kwargs.get("attachment_id"), 160)
    action_id = _text(kwargs.get("action_id"), 40).lower()
    if not attachment_id or action_id != "dismiss":
        return None
    with _state_lock:
        entry = _emitted_connection_proposals.get(attachment_id)
    if not entry:
        return None
    session_key, dedup_key = entry
    if dedup_key:
        _latch_dismissal(session_key, dedup_key, time.monotonic())
        logger.info(
            "connection recommendation dismissed; latched dedup_key=%s", dedup_key
        )
    return None


def _transform_llm_output(**kwargs: Any) -> str | None:
    session_id = _session_key(kwargs)
    response_text = str(kwargs.get("response_text") or "")
    if not session_id:
        return None
    if _is_noninteractive(kwargs) or _is_unsupported_runtime(kwargs) or kwargs.get(
        "structured_output"
    ):
        with _state_lock:
            _state_locked(session_id, time.monotonic())["pending_action_result"] = None
        return None
    if not response_text or is_intentional_silence_response(response_text):
        with _state_lock:
            state = _state_locked(session_id, time.monotonic())
            action_result = state.get("pending_action_result")
            state["pending_action_result"] = None
            if (
                isinstance(action_result, dict)
                and action_result.get("action") == "create"
                and state.get("proposal_stage") == "create_action_pending"
            ):
                state["proposal_stage"] = "proposal_shown"
                action_result = action_result | {"status": "rejected"}
        if (
            isinstance(action_result, dict)
            and (
                action_result.get("status") == "rejected"
                or action_result.get("action") in {"dismiss", "mute_session", "unmute_session"}
            )
        ):
            return _action_result_envelope(action_result)
        return None
    if kwargs.get("failed") or kwargs.get("interrupted") or kwargs.get("completed") is False:
        with _state_lock:
            state = _state_locked(session_id, time.monotonic())
            action_result = state.get("pending_action_result")
            state["pending_action_result"] = None
            retryable_create = isinstance(action_result, dict) and action_result.get("action") == "create"
            if retryable_create and state.get("proposal_stage") == "create_action_pending":
                state["proposal_stage"] = "proposal_shown"
                action_result = action_result | {"status": "rejected"}
            if state.get("proposal_stage") == "proposal_shown" and not retryable_create:
                state["last_candidate"] = None
                state["last_proposal"] = None
                state["proposal_stage"] = None
        if (
            isinstance(action_result, dict)
            and (
                action_result.get("status") == "rejected"
                or action_result.get("action") in {"dismiss", "mute_session", "unmute_session"}
            )
        ):
            return _action_result_envelope(action_result)
        return None
    response_without_action_results = _ACTION_RESULT_ENVELOPE_RE.sub("", response_text)
    stripped_forged_action_result = response_without_action_results != response_text
    response_text = response_without_action_results
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        action_result = state.get("pending_action_result")
        state["pending_action_result"] = None
        if (
            isinstance(action_result, dict)
            and action_result.get("action") == "create"
            and action_result.get("status") == "accepted"
        ):
            state["last_candidate"] = None
            state["last_proposal"] = None
            state["proposal_stage"] = None
    result_suffix = (
        "\n\n" + _action_result_envelope(action_result)
        if isinstance(action_result, dict)
        else ""
    )
    response_with_result = response_text + result_suffix
    # Transform hooks use ``None``/empty to mean "leave the original response
    # unchanged". Return whitespace when a forged marker was the entire
    # response, so the finalizer can still replace (and therefore remove) it.
    sanitized_response = response_with_result or ("\n" if stripped_forged_action_result else None)
    if _is_session_muted(session_id):
        return sanitized_response if (result_suffix or stripped_forged_action_result) else None
    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        current_turn = int(state["turn"])
        if (
            not isinstance(proposal, dict)
            or int(state["last_prompt_turn"]) != current_turn
            or int(state["last_delivery_turn"]) == current_turn
        ):
            return sanitized_response if (result_suffix or stripped_forged_action_result) else None
        state["last_delivery_turn"] = current_turn

    if "<!--creation-recommendation:start " in response_text:
        return sanitized_response if (result_suffix or stripped_forged_action_result) else None
    if _is_session_muted(session_id):
        return sanitized_response if (result_suffix or stripped_forged_action_result) else None
    if proposal.get("creation_type") in ATTACHMENT_DELIVERED_TYPES:
        # 连接/artifact 推荐走结构化 attachment 通道（channel.connect /
        # connector.connect / artifact.recommendation 卡），不追加文本信封；
        # 发射失败（无活跃流）静默降级，正文原样返回。
        emitted = _emit_recommendation_attachment(session_id, proposal)
        logger.info(
            "attachment recommendation %s type=%s target=%s turn=%s",
            "emitted" if emitted else "skipped (no active stream)",
            proposal.get("creation_type"),
            _text(proposal.get("target"), 80),
            current_turn,
        )
        return sanitized_response if (result_suffix or stripped_forged_action_result) else None
    logger.info(
        "creation recommendation attached type=%s confidence=%s title=%s turn=%s",
        proposal.get("creation_type"),
        proposal.get("confidence"),
        _text(proposal.get("suggested_name"), 80),
        current_turn,
    )
    return response_with_result + "\n\n" + _recommendation_envelope(proposal)


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
    global _plugin_llm, _plugin_ctx
    with _state_lock:
        _recent_proposals.clear()
        _dismissed_proposals.clear()
        _session_states.clear()
        _muted_sessions.clear()
        _known_unmuted_sessions.clear()
        _emitted_connection_proposals.clear()
    _plugin_llm = None
    _plugin_ctx = None
    _invocation_scope.set(None)


def register(ctx: Any) -> None:
    global _plugin_llm, _plugin_ctx
    _plugin_ctx = ctx
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
    # 连接推荐卡（channel.connect / connector.connect）的按钮回执：dismiss
    # 落 30 天拒绝闩锁。hook 由 zet_agent 的 attachment/action 入站派发。
    ctx.register_hook("attachment_action", _on_attachment_action)
    ctx.register_tool(
        name=TOOL_NAME,
        toolset="creation_governor",
        schema={
            "name": TOOL_NAME,
            "description": (
                "Perform one high-recall zero-shot semantic choice among Agent, Skill, Task, and "
                "none. A single substantive request is enough; never require the user to mention "
                "repetition, saving, or creation. Use "
                "meaning and conversation context, never topic keyword matching or memorized "
                "examples. Task means desired future time/event/background execution; a current "
                "data range such as today is not by itself a trigger. Agent means a long-lived "
                "responsible role with retained context, judgment, autonomous tool choice, or "
                "interpretation of a changing real-world work domain. A substantive first request "
                "to inspect, compare, diagnose, research, optimize, or make a judgment about an "
                "ongoing external work domain should normally be Agent rather than none. "
                "Skill means a stable reusable input-to-output method without an independent "
                "identity. Missing connectors or authorization affect current execution, not "
                "reuse value. Explicit creation requests use Hermes' native flow and return none. "
                "Call only between scheduled checkpoints when one unusually clear opportunity "
                "emerges. The tool never creates and may return none. The plugin owns cooldown, "
                "deduplication, dismissal, and conditional card/text delivery after the current "
                "task is complete."
            ),
            "parameters": _DETECTOR_SCHEMA,
        },
        handler=_detect_creation_opportunity,
        description="Detect one reusable creation opportunity or none",
        emoji="💡",
    )
