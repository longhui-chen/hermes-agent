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
from typing import Any, Literal, NamedTuple

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
MAX_PENDING_ACTION_RESULTS = 16
CREATION_TYPES = {"agent", "skill", "task"}
# agent 品类落地缺口（2026-08-07 真机实测）：hermes 侧没有 create_agent 工具
# （task 走原生定时、skill 有 skill_manager_tool，唯独 agent 档案的增删归
# zls 管理 API / App 界面，从未暴露给会话），用户点「确认」必然收到
# "Agent 创建服务当前不可用"。推荐一个建不成的东西比不推更伤，先关掉。
# **恢复方式：补齐 create_agent 工具后把这里改回 True，无需改动其他代码。**
AGENT_RECOMMENDATION_ENABLED = False
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
AGENT_TEMPLATE_TYPE = "agent_template"
# attachment 通道交付的全部品类。agent_template 指向云端真实模板详情/克隆
# 流程，不走当前缺失的 Hermes create_agent 工具；普通 agent/skill/task 仍走
# 文本信封通道。
ATTACHMENT_DELIVERED_TYPES = CONNECTION_TYPES | {ARTIFACT_TYPE, AGENT_TEMPLATE_TYPE}
CONNECTION_INVENTORY_TTL_SECONDS = 600.0
MAX_EMITTED_CONNECTION_PROPOSALS = 256
RECOMMENDATION_ACTIONS = {"create", "dismiss", "mute_session", "unmute_session"}
SESSION_PREFERENCES_DB = "creation_governor.db"
UNSUPPORTED_API_MODES = {"codex_app_server"}
UNSUPPORTED_PLATFORMS = {"acp"}
_NONINTERACTIVE_PLATFORMS = {"cron", "subagent", "batch"}
_ACTION_RECEIPT_TRANSPORT = "canonical_final_v1"

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
# (raw_session_id, scoped_session_id, suppression_reason, owner_id,
#  receipt_transport, turn_id)
_invocation_scope: ContextVar[
    tuple[str, str, str | None, str, str, str] | None
] = ContextVar(
    "creation_governor_invocation_scope",
    default=None,
)

_SELF_QUERY_RE = re.compile(
    r"(?:creation[\s_-]*governor|detect_creation_opportunity|propose_creation)",
    re.IGNORECASE,
)
_FAST_ROUTE_UNAVAILABLE_RE = re.compile(
    r"(?:404|not found|not in public manifest|unknown (?:model|route)|"
    r"model .+ does not exist|invalid model|model_not_found|"
    r"no available channel for model)",
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
# 花括号定界是刻意的，别改成 `(.*?)`。结束标签在后面锚着，非贪婪匹配会一路
# 回溯扩展到配对的那个 `}`——title 里含 `}`（"JSON {schema}"）或 payload 有嵌套
# 对象都解得对。反过来 `(.*?)` 在 JSON 字符串里恰好出现结束标签时会提前收尾，
# 比现在更脆弱。（2026-08-19 有一轮 review 按「非贪婪会在第一个 `}` 收尾」报过
# 这里，实测不成立。）
_RECOMMENDATION_RESPONSE_RE = re.compile(
    r"\[creation_recommendation_response\]\s*(\{.*?\})\s*"
    r"\[/creation_recommendation_response\]",
    re.DOTALL,
)
_ACTION_RESULT_ENVELOPE_RE = re.compile(
    r"<!--creation-recommendation-action-result(?:\s+[^>]*)?-->"
)


class _ActionReceipt(NamedTuple):
    proposal_id: str
    action: str
    status: Literal["accepted", "rejected"]
    reason_code: Literal[
        "proposal_not_actionable",
        "preference_not_persisted",
    ] | None = None


class _ActionHandlingOutcome(NamedTuple):
    context: str
    receipt: _ActionReceipt | None
    # 这次动作针对的创建品类。只有 create 被接管时才有意义——创建配额按品类
    # 发放，接受一张 Skill 卡换来的票不该放行一次 cronjob(create)。
    creation_type: str = ""

_ONBOARDING_WELCOME_RE = re.compile(
    r"<!--zettlab-onboarding-welcome\s+([A-Za-z0-9_-]+)-->",
)


def _decode_onboarding_welcome_spec(user_message: str) -> dict[str, Any] | None:
    """Decode the App-owned final-onboarding-turn marker.

    This is intentionally strict and bounded.  It never runs a model and never
    trusts the marker to name a channel; the channel target is resolved later
    from local-server's region-aware live inventory.
    """
    match = _ONBOARDING_WELCOME_RE.search(str(user_message or ""))
    if not match or len(match.group(1)) > 12_000:
        return None
    encoded = match.group(1)
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if len(raw) > 8_000:
            return None
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("type") != "zettlab_onboarding_welcome"
    ):
        return None
    channel = payload.get("channel")
    task = payload.get("task")
    artifact = payload.get("artifact")
    agent_templates = payload.get("agentTemplates", [])
    if (
        not isinstance(channel, dict)
        or not isinstance(task, dict)
        or (artifact is not None and not isinstance(artifact, dict))
        or not isinstance(agent_templates, list)
        or len(agent_templates) > 2
    ):
        return None
    task_title = _text(task.get("title"), 80)
    task_reason = _text(task.get("reason"), 400)
    task_proposal = _text(task.get("proposalText"), 500)
    artifact_title = _text(artifact.get("title"), 80) if artifact else ""
    artifact_reason = _text(artifact.get("reason"), 400) if artifact else ""
    parsed_agent_templates = []
    seen_template_ids = set()
    for item in agent_templates:
        if not isinstance(item, dict):
            return None
        template_id = _text(item.get("templateId"), 160)
        title = _text(item.get("title"), 80)
        reason = _text(item.get("reason"), 400)
        if not template_id or not title or not reason:
            return None
        if template_id in seen_template_ids:
            continue
        seen_template_ids.add(template_id)
        parsed_agent_templates.append(
            {"template_id": template_id, "title": title, "reason": reason}
        )
    if (
        not task_title
        or not task_reason
        or not task_proposal
        or (
            artifact is not None
            and (
                not artifact_title
                or not artifact_reason
                or artifact.get("artifactType") != "app"
            )
        )
    ):
        return None
    return {
        "channel_requested": channel.get("requested") is True,
        "task_title": task_title,
        "task_reason": task_reason,
        "task_proposal": task_proposal,
        "artifact_title": artifact_title,
        "artifact_reason": artifact_reason,
        "agent_templates": parsed_agent_templates,
    }


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


# 用户消息进 governor 前的字符预算。动作信封挂在消息末尾，而 Web 会在它前面
# 放一段给模型看的动作说明文案——文案一长就能把信封挤出这个窗口。
USER_MESSAGE_LIMIT = 2000


def _bounded_user_message(raw: Any) -> str:
    """把用户消息压到预算内，但**不能把结尾的动作信封切掉**。

    信封被切掉的后果不是「少看见一段文字」：governor 完全看不到这次动作，
    既不接管也不生成回执，而 HTTP 请求照常以普通模型结果收尾——版本化端点
    那边已经按「这是一次动作」放行了，Web 于是把这次创建永久停在「不确定
    且不能重试」。准入和这里必须看到同一个信封。
    """
    normalized = " ".join(str(raw or "").split())
    if len(normalized) <= USER_MESSAGE_LIMIT:
        return normalized
    match = _RECOMMENDATION_RESPONSE_RE.search(normalized)
    if match is None:
        return normalized[:USER_MESSAGE_LIMIT]
    envelope = normalized[match.start():match.end()]
    if len(envelope) >= USER_MESSAGE_LIMIT:
        # 信封本身就超预算。截断它只会让解析失败，原样交出去反而是更诚实的
        # 输入——payload 大小另有上限把关。
        return envelope
    head = normalized[: USER_MESSAGE_LIMIT - len(envelope) - 1].rstrip()
    return f"{head} {envelope}" if head else envelope


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
        or creation_type in {ARTIFACT_TYPE, AGENT_TEMPLATE_TYPE}
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
    """会话作用域的内部标识。

    不能用 `_text()`：它是给展示文本用的，会折叠内部空白并截断到 160 字符，
    而 `APIServerAdapter._parse_session_key_header()` 允许最长 256 且保留内部
    空白。两个合法的不同会话因此可能塌成同一个 scope，`last_proposal`、mute
    偏好和 pending receipt 会串到别人的会话上。

    短且无需归一的键原样保留（与既有 scope 兼容，不会因升级重置用户偏好）；
    其余用完整值的摘要，碰撞由 sha256 保证而不是由截断决定。
    """
    raw = str(
        kwargs.get("conversation_session_id")
        or kwargs.get("session_id")
        or kwargs.get("task_id")
        or ""
    )
    if not raw:
        return ""
    if len(raw) <= 160 and raw == " ".join(raw.split()):
        return raw
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _scoped_session_key(raw_session_id: str, owner_id: str) -> str:
    profile = str(get_hermes_home().resolve())
    return f"{profile}|{_text(owner_id, 160)}|{raw_session_id}"


def _is_current_invocation(kwargs: dict[str, Any]) -> bool:
    """Whether this hook/tool call belongs to the turn that set the scope.

    钩子（pre_llm_call / transform_llm_output）拿得到稳定的
    conversation_session_id，直接按它认；工具链拿不到——
    ``model_tools`` 只把 transcript 级 ``session_id`` 递进
    ``registry.dispatch``，稳定 scope 传不进来。turn_id 两侧都在且比 session
    更细，用它作为同一次调用的凭据，让工具写候选与钩子读候选落在同一个
    conversation scope。
    """
    invocation = _invocation_scope.get()
    if invocation is None:
        return False
    raw_session_id = _raw_session_key(kwargs)
    if raw_session_id and invocation[0] == raw_session_id:
        return True
    turn_id = _text(kwargs.get("turn_id"), 160)
    return bool(turn_id and invocation[5] == turn_id)


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
        and _is_current_invocation(kwargs)
        and invocation[1].startswith(profile_prefix)
        and (not explicit_owner or explicit_owner == invocation[3])
    ):
        return invocation[1]
    return _scoped_session_key(raw_session_id, explicit_owner)


def _receipt_transport(kwargs: dict[str, Any]) -> str:
    raw = kwargs.get("creation_action_receipt_transport")
    return _ACTION_RECEIPT_TRANSPORT if raw == _ACTION_RECEIPT_TRANSPORT else ""


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


def _read_persisted_session_preference(
    session_id: str,
) -> Literal["muted", "unmuted", "absent", "read_error"]:
    path = _preferences_db_path()
    if not path.exists():
        return "absent"
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
        return "read_error"
    if row is None:
        return "absent"
    return "muted" if bool(row[0]) else "unmuted"


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
    preference = _read_persisted_session_preference(session_id)
    if preference == "read_error":
        return True
    muted = preference == "muted"
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
            "pending_action_results": OrderedDict(),
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


# pending_action_results 有条目数上限，但没有单条键长上限：调用方给的 turn_id
# 原样当键，MAX_PENDING_ACTION_RESULTS 条超长键就能在端侧吃掉数百 MB。超过这个
# 长度的 turn_id 改用定长摘要——存和取走同一个归一化，查找语义不变。
MAX_RAW_PENDING_TURN_KEY_LEN = 256


def _pending_turn_key(turn_id: str) -> str:
    if len(turn_id) <= MAX_RAW_PENDING_TURN_KEY_LEN:
        return turn_id
    return "sha256:" + hashlib.sha256(turn_id.encode("utf-8")).hexdigest()


# 被判为无效/过期的 recommendation action，本轮不允许落地任何创建。
# 只靠 pre_llm_call 往上下文里塞一句「别创建」是劝阻不是约束：那条被拒的动作
# 正文照样进模型，模型完全可以照着它去调 skill_manage(create) / cronjob(create)。
# 闸门必须落在执行点，也就是 pre_tool_call。
#
# 作用域是单个 turn_id：拿不到 turn_id 时无法界定范围，此时保持放行——宁可漏挡
# 一次，也不能因为一个无 id 的请求把全设备的创建工具锁死。
#
# 只列了 skill 和 task 两种品类的落地工具，因为 agent 品类的推荐当前是关的
# （`AGENT_RECOMMENDATION_ENABLED = False`，硬闸在 `_detect_creation_opportunity`
# 里事后过滤），生产上不存在 creation_type=agent 的 proposal。
# **打开那个开关时必须回来补这里**：agent 的落地路径是 agent-creator 经 terminal
# 跑 create_agent.py，不是一个能按工具名挡住的独立工具，需要单独设计执行点。
DENIED_CREATION_TOOL_ACTIONS = {
    "skill_manage": {"create"},
    "cronjob": {"create"},
}
# 哪个工具落地哪个品类。票是按品类发的：接受一张 Skill 卡不该顺带放行一次
# cronjob(create)。
CREATION_TOOL_TYPES = {
    "skill_manage": "skill",
    "cronjob": "task",
}
MAX_DENIED_CREATION_TURNS = 256

# 闸门是**配额**，不是布尔开关：这个 turn 上允许落地的创建次数，等于被真正
# 接管的动作次数。
#
# 为什么不能用布尔。双击或传输重发会让两个并发请求复用同一个 turn_id：先到的
# 原子消费掉 proposal 拿到 accepted，后到的因为 proposal 已被消费而判无效。
#   - 只按 turn_id 记「拒绝」→ 后到那个的拒绝会把先到那个真实的创建也挡掉，
#     客户端收到 accepted 而资源没建出来；
#   - 反过来让「接管」全局压过「拒绝」→ 后到那个重放请求的创建也被放行，
#     重复创建又回来了。
# 两者都是拿一个 turn 级的开关去表达一个请求级的事实。配额能同时挡住两边：
# 一次 accepted 只买一张票，谁先用掉都行，但总共只有一张。
#
# 键存在 = 这个 turn 上出现过推荐动作、进入配额管控；键不存在 = 普通轮次，
# 用户直接说「帮我建个 skill」不受影响。
_creation_turn_quota: "OrderedDict[str, dict[str, int]]" = OrderedDict()
# 同一个 turn_id 上还有几个在途请求。见 _enter_creation_quota_for_turn 的说明。
_creation_turn_refs: "OrderedDict[str, int]" = OrderedDict()


def _trim_creation_turn_quota_locked() -> None:
    while len(_creation_turn_quota) > MAX_DENIED_CREATION_TURNS:
        evicted, _ = _creation_turn_quota.popitem(last=False)
        _creation_turn_refs.pop(evicted, None)


def _enter_creation_quota_for_turn(turn_id: str) -> None:
    """这一轮出现了推荐动作：从此刻起创建工具受配额管控。

    同一个 turn_id 可能有多个在途请求（双击 / 传输重发），所以记引用计数——
    先完成的那个请求不能把还在等模型返回工具调用的那个的闸门一起撤掉，
    否则后者会因为「键不存在」被当成不受管控的普通轮次，创建照样放行。
    """
    key = _pending_turn_key(turn_id)
    if not key:
        return
    with _state_lock:
        _creation_turn_quota.setdefault(key, {})
        _creation_turn_quota.move_to_end(key)
        _creation_turn_refs[key] = _creation_turn_refs.get(key, 0) + 1
        _trim_creation_turn_quota_locked()


def _grant_creation_for_turn(turn_id: str, creation_type: str) -> None:
    """一次 create 动作被真正接管，为**它那个品类**发一张票。

    只给 create 发票：dismiss / mute_session / unmute_session 也会拿到 accepted
    回执，但用户表达的恰恰是「别建」或「只改偏好」——给它们发票等于模型无视
    内部提示去调 create 时闸门主动让路。

    票绑定品类：接受一张 Skill 卡换来的票不该放行一次 cronjob(create)。
    """
    key = _pending_turn_key(turn_id)
    normalized = _normalize_creation_type(creation_type)
    if not key or not normalized:
        return
    with _state_lock:
        quota = _creation_turn_quota.setdefault(key, {})
        quota[normalized] = quota.get(normalized, 0) + 1
        _creation_turn_quota.move_to_end(key)
        _trim_creation_turn_quota_locked()


def _release_creation_deny(turn_id: str) -> None:
    """一个请求收尾。同 turn 还有在途请求时不撤闸门，等最后一个再撤。"""
    key = _pending_turn_key(turn_id)
    if not key:
        return
    with _state_lock:
        remaining = _creation_turn_refs.get(key)
        if remaining is None:
            # 从没进过配额管控的普通轮次，或者已经被容量淘汰了。
            _creation_turn_quota.pop(key, None)
            return
        if remaining > 1:
            _creation_turn_refs[key] = remaining - 1
            return
        _creation_turn_refs.pop(key, None)
        _creation_turn_quota.pop(key, None)


def _consume_creation_quota(turn_id: str, creation_type: str) -> bool:
    """这次创建能不能放行。该品类有票就消耗一张放行，没票就挡。

    不受管控的普通轮次（键不存在）永远放行——这道闸门只针对推荐动作那条路径。
    """
    key = _pending_turn_key(turn_id)
    if not key:
        return True
    with _state_lock:
        quota = _creation_turn_quota.get(key)
        if quota is None:
            return True
        remaining = quota.get(creation_type, 0)
        if remaining <= 0:
            return False
        quota[creation_type] = remaining - 1
        return True


def _on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    turn_id: str = "",
    **_: Any,
) -> dict[str, str] | None:
    denied_actions = DENIED_CREATION_TOOL_ACTIONS.get(tool_name)
    if not denied_actions:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            args = None
    # args 解析不出来时无法确认这次是不是 create，按拒绝处理：这一轮本来就
    # 不该有任何创建落地。
    action = ""
    if isinstance(args, dict):
        action = str(args.get("action") or "").strip().lower()
        if action and action not in denied_actions:
            return None
    # 配额在这里消耗：判定「是不是一次创建」之后、放行之前。放在更早会让
    # list / patch 这类调用白白吃掉一张票。
    if _consume_creation_quota(turn_id, CREATION_TOOL_TYPES.get(tool_name, "")):
        return None
    return {
        "action": "block",
        "message": (
            "creation-governor refused this creation: it originates from a "
            "creation recommendation action that was already rejected as "
            "invalid or expired. Tell the user the recommendation is no longer "
            "actionable instead of creating anything."
        ),
    }


def _store_pending_action_result_locked(
    state: dict[str, Any], turn_id: str, receipt: _ActionReceipt
) -> None:
    if not turn_id:
        return
    turn_id = _pending_turn_key(turn_id)
    pending = state["pending_action_results"]
    existing = pending.get(turn_id)
    # 同一个 turn_id 上的结果是**单调**的：接管过就接管过了，后到的重放请求
    # 拿到的 rejected 不能把它盖掉。双击 / 传输重发会让两个请求复用同一个
    # turn_id，先到的消费掉 proposal 拿到 accepted、后到的必然判无效——覆盖
    # 之后先结束的那个请求会取走 rejected，客户端把一次已经接管的创建显示成
    # 失败，用户重来一次就是重复创建。
    if (
        isinstance(existing, _ActionReceipt)
        and existing.status == "accepted"
        and receipt.status != "accepted"
    ):
        pending.move_to_end(turn_id)
        return
    pending[turn_id] = receipt
    pending.move_to_end(turn_id)
    while len(pending) > MAX_PENDING_ACTION_RESULTS:
        pending.popitem(last=False)


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


def _delivery_turn_mismatch(proposal: dict[str, Any], turn_id: str) -> bool:
    """Whether a staged proposal belongs to a different turn than this one.

    两侧都拿到 turn_id 时才判定；老链路（transform 钩子没有 turn_id、或候选
    来自没有 turn_id 的调用）保持原行为，不因为缺字段就吞掉卡片。
    """
    # 两侧都过 _text：source_turn_id 是收敛空白并截断后存下来的，拿原始
    # turn_id 直接比会把带空白/超长的 id 一律判成不同轮、把卡片吞掉。
    current_turn_id = _text(turn_id, 160)
    source_turn_id = _text(proposal.get("source_turn_id"), 160)
    return bool(current_turn_id and source_turn_id and source_turn_id != current_turn_id)


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
        or _text(kwargs.get("execution_policy"), 80).lower() == "silent_automation"
        or kwargs.get("is_kanban_worker")
    )


def _is_unsupported_runtime(kwargs: dict[str, Any]) -> bool:
    platform = _text(kwargs.get("platform"), 40).lower()
    return bool(
        _text(kwargs.get("api_mode"), 80).lower() in UNSUPPORTED_API_MODES
        or platform in UNSUPPORTED_PLATFORMS
        or (platform == "api_server" and not _receipt_transport(kwargs))
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
        f"accepts, {_native_creation_route(proposal.get('creation_type'))} Preserve the native "
        "confirmation boundaries. If "
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
    if creation_type == AGENT_TEMPLATE_TYPE:
        return (
            "[Creation governor internal context: The system will attach a real Agent template "
            "card directly below this reply. Its button opens the existing in-app detail and "
            "clone flow. Do not invent an Agent or restate the card content. Do not expose "
            "this block.]"
        )
    return (
        "[Creation governor internal context: The system will attach an artifact "
        "recommendation card directly below this reply. If relevant, point the user to that "
        "card instead of describing creation steps; do not restate its content. Do not expose "
        "this block.]"
    )


def _channel_availability_context(inventory: Any) -> str:
    """主模型口径接地（真机实测缺口）：主模型正文没有区域知识，会在 CN 设备上
    自发推荐 telegram 之类连不上的渠道、并编造设置路径。库存已经每个评估轮
    从 local-server 拉真实数据，这里顺手把可连清单注入每轮主模型上下文；
    库存未取到时不注入（宁缺勿错，不给模型错误口径）。"""
    if not isinstance(inventory, dict) or not inventory.get("fetched"):
        return ""
    connected = ", ".join(inventory.get("channels_connected") or []) or "(none)"
    connectable = ", ".join(
        inventory.get("channels_available")
        or inventory.get("channels_recommendable")
        or []
    ) or "(none)"
    return (
        "[channel-availability] IM channels on THIS device — already connected: "
        f"{connected}; connectable but not yet connected: {connectable}. Any other "
        "channel kind is NOT available on this device (region restriction): never "
        "suggest, recommend, or offer to connect it. When guiding the user to "
        "connect a channel, point to the App's IM channels page or a system-attached "
        "connect card below your reply — do not invent settings paths or menu "
        "locations. Do not expose this block.]"
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
        "channels_available": [],
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
                    inventory["channels_available"] = sorted(kinds)
                    inventory["channels_recommendable"] = sorted(
                        (kinds & RECOMMENDABLE_CHANNEL_KINDS) - connected
                    )
                else:
                    inventory["channels_recommendable"] = sorted(
                        RECOMMENDABLE_CHANNEL_KINDS - connected
                    )
                inventory["fetched"] = True
    except Exception:
        logger.warning("connection inventory: channel fetch failed", exc_info=True)
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
                # 从未连接过的 provider 不会出现在 connectors 里（那份只列已建立
                # 的连接），但它们恰恰是最该被推荐去连的。缺了这一路，用户一个
                # 连接器都没连时可推荐池恒为空，connector 推荐被硬闸拒死。
                available = parsed.get("available_providers")
                if isinstance(available, list):
                    for item in available:
                        provider = _text(item, 80).lower()
                        if provider and provider not in connected:
                            recommendable.append(provider)
                inventory["connectors_connected"] = sorted(set(connected))
                inventory["connectors_recommendable"] = sorted(set(recommendable))
                inventory["fetched"] = True
    except Exception:
        logger.warning("connection inventory: connector fetch failed", exc_info=True)
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
    # 库存是 channel/connector 推荐的硬闸输入：池为空即禁止推荐。此前它不落日志，
    # 出不出卡只能靠猜——池空到底是"云端真没有"还是"链路把数据丢了"分不清
    # （2026-08-10 排查教训）。
    logger.info(
        "connection inventory: fetched=%s channels_recommendable=%d "
        "connectors_connected=%d connectors_recommendable=%d",
        inventory.get("fetched"),
        len(inventory.get("channels_recommendable") or []),
        len(inventory.get("connectors_connected") or []),
        len(inventory.get("connectors_recommendable") or []),
    )
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
            "enum": (
                ["agent"] if AGENT_RECOMMENDATION_ENABLED else []
            ) + ["skill", "task", "channel", "connector", "artifact", "none"],
        },
        "suggested_name": {"type": "string"},
        "reason": {"type": "string"},
        "target": {
            "type": "string",
            "description": (
                "REQUIRED when decision is channel/connector: the exact kind/"
                "provider id copied verbatim from the recommendable list in "
                "[connection-inventory] (e.g. 'wechat', 'gmail'). Never invent "
                "values; leave empty only for non-connection decisions."
            ),
        },
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


_AGENT_RULE_ENABLED = """2. agent: future work needs a long-lived responsible role, retained domain context, judgment,
   autonomous choice among tools, decisions about the next step, or repeated interpretation of a
   changing real-world business domain, account, operation, project, or body of evidence."""

_AGENT_RULE_DISABLED = """2. agent: DISABLED on this deployment — never return "agent". If a case looks like a long-lived
   responsible role, evaluate whether a skill or task covers it; otherwise return none."""

_DETECTOR_INSTRUCTIONS = """Perform one high-recall zero-shot product judgment.

Return exactly one of agent, skill, task, channel, connector, artifact, or none. Do not classify by
topic words and do not use memorized examples. A single substantive request can be enough only when
the conversation itself supports durable future value; the mere possibility that a capability could be
reused is not enough. Do not require magic words such as repetition, saving, or creation, but require
affirmative semantic evidence that the account, project, source, responsibility, or class of future
inputs continues beyond this bounded request.

Definitions and conflict order:
1. task: the desired future value depends on a recurring time trigger, event trigger, background
   monitoring, repeated refresh of new information, or keeping a derived result current as its
   source changes. A word such as 'today' merely scopes the current data; it is not by itself a future trigger.
__AGENT_RULE__
3. skill: future inputs vary but a stable input-to-output method can be reused without an
   independent identity or durable state. Do not choose skill when the primary future value is
   keeping one persistent result, profile, summary, index, report, or state up to date.
4. none: small talk, a trivial transformation, a low-value closed-world fact lookup, an explicit
   request to create/configure/schedule something through Hermes' native flow, or no reasonable
   reuse value.
5. channel: the durable value of this need depends on reminders, results, or notifications
   reaching the user inside an IM app, and a `[connection-inventory]` line in the evidence lists
   that channel kind under "channels recommendable". Set target to that exact channel kind.
   ALSO decide channel (high confidence) when the user EXPLICITLY asks how to connect, use,
   or message through a specific IM channel that the inventory lists as recommendable — an
   explicit ask is the strongest possible signal; the card gives them a one-tap path.
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
用户的语言，不能写“用户已……”这类内部判定；proposal_text 要明确说明将进入哪种原生创建流程。

channel 与 connector 的中文规则相同：只有当证据里存在 [connection-inventory] 行、且目标
明确出现在 recommendable 列表中时才允许返回这两类；已连接的渠道或数据源绝不重复推荐；
target 必须逐字取自清单，禁止猜测或泛化。channel 用于"提醒/结果需要直达用户的 IM"，
connector 用于"这类任务实质上需要用户自己的外部数据"。artifact 用于"这段对话最有价值的
沉淀是一件可打开的作品（页面/小应用/报告）而非一种能力"——方法可复用选 skill，产物本身
有长期价值选 artifact。"""


def _detector_instructions() -> str:
    """按开关渲染检测器指令：agent 品类关闭时给出明确禁令而不是判定规则。"""
    rule = _AGENT_RULE_ENABLED if AGENT_RECOMMENDATION_ENABLED else _AGENT_RULE_DISABLED
    return _DETECTOR_INSTRUCTIONS.replace("__AGENT_RULE__", rule)


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
                _detector_instructions()
                + "\n\nReturn only one compact JSON object with exactly these keys: "
                "decision, suggested_name, reason, target, evidence_turn_ids, "
                "confidence, dedup_key, proposal_text. suggested_name, reason and "
                "proposal_text are ALWAYS required and must be non-empty. For channel/connector "
                "decisions target is MANDATORY: copy the exact kind/provider id "
                "verbatim from the recommendable list in [connection-inventory]; "
                "use an empty string for other decisions. Do not use Markdown fences."
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
            "creation opportunity JSON decision=%s target=%s confidence=%s title=%s "
            "provider=%s model=%s",
            parsed.get("decision") if parsed else None,
            _text(parsed.get("target"), 80) if parsed else "",
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
    args: dict[str, Any], state: dict[str, Any], turn_id: str = ""
) -> tuple[dict[str, Any] | None, str]:
    decision = _normalize_creation_type(
        args.get("decision") or args.get("creation_type")
    )
    if decision == "none":
        return None, "none"
    if decision == "agent" and not AGENT_RECOMMENDATION_ENABLED:
        # 事后过滤是硬闸：prompt 只是引导，模型仍可能选 agent。
        return None, "agent_recommendation_disabled"
    if (
        decision not in CREATION_TYPES
        and decision not in CONNECTION_TYPES
        and decision != ARTIFACT_TYPE
    ):
        return None, "unsupported_creation_type"

    suggested_name = _text(args.get("suggested_name"), 80)
    if decision in CONNECTION_TYPES and not suggested_name:
        # flash 档检测器在 target MANDATORY 强调后偶发漏填 suggested_name（真机
        # 实测）。连接卡标题由客户端 i18n 渲染、语义 dedup 键也走 target——这里
        # 用 target 兜底，不因展示面冗余字段拒掉合法推荐。
        suggested_name = _text(args.get("target"), 80).lower()
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
        if not target and isinstance(pool, list):
            # 容错（真机实测）：flash 档检测器常把渠道 kind 填进 suggested_name
            # 而漏掉 target。仅当 suggested_name 逐字命中库存池时回退采用——
            # 仍然在"目标必须命中真实可连清单"的硬闸之内，不放宽任何约束。
            fallback = _text(args.get("suggested_name"), 80).lower()
            if fallback in pool:
                target = fallback
        if not target or not isinstance(pool, list) or target not in pool:
            # 拒绝原因必须可诊断：真机排障时需要区分「检测器没给 target」「库存
            # 未取到」「target 不在可推荐集合」三种完全不同的故障面。
            logger.info(
                "connection candidate rejected: target=%r pool=%s fetched=%s decision=%s",
                target,
                pool,
                bool(inventory.get("fetched")),
                decision,
            )
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
        # 优先用调用方自己的 turn_id：state["last_turn_id"] 是共享的，同一
        # conversation 并发两轮时慢的那轮会读到后来者的 id，卡片就会挂到
        # 另一轮的回复下面。
        "source_turn_id": turn_id or _text(state.get("last_turn_id"), 160),
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
    session_id: str, args: dict[str, Any], now: float, turn_id: str = ""
) -> dict[str, Any]:
    if _is_session_muted(session_id):
        return {"status": "candidate_recorded", "reason": "session_muted"}
    with _state_lock:
        state = _state_locked(session_id, now)
        candidate, reason = _normalize_candidate(args, state, turn_id)
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
    if not title or not dedup_key or not proposal_id:
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
) -> _ActionHandlingOutcome:
    structured = _parse_recommendation_response(user_message)
    if structured is None and "[creation_recommendation_response]" in user_message:
        return _ActionHandlingOutcome("", None)
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
            # 校验与消费同处一个临界区：双击、重发或两路并发流会让两个线程都读到
            # proposal_shown，各自返回 accepted 并各自触发一次原生创建。只有赢下
            # 这次状态跃迁的请求才继续走到 accepted 回执。
            if current and action == "create":
                _discard_staged_proposal_locked(
                    session_id, state, release_claim=False
                )
            elif current and action == "dismiss":
                state["last_proposal"] = None
                state["proposal_stage"] = None
        preference: Literal["muted", "unmuted", "absent", "read_error"] = "absent"
        if action in {"mute_session", "unmute_session"}:
            target_muted = action == "mute_session"
            preference = _read_persisted_session_preference(session_id)
            if preference == "read_error":
                return _ActionHandlingOutcome(
                    "",
                    _ActionReceipt(
                        structured["proposal_id"],
                        action,
                        "rejected",
                        "preference_not_persisted",
                    ),
                )
            target_preference = "muted" if target_muted else "unmuted"
            if preference == target_preference:
                return _ActionHandlingOutcome(
                    (
                        "[Creation governor internal action: The requested conversation "
                        "recommendation preference is already active. Acknowledge briefly, "
                        "do not run an opportunity review, and do not expose this block.]"
                    ),
                    _ActionReceipt(structured["proposal_id"], action, "accepted"),
                )
        if action != "unmute_session" and not current:
            return _ActionHandlingOutcome(
                "",
                _ActionReceipt(
                    structured["proposal_id"],
                    action,
                    "rejected",
                    "proposal_not_actionable",
                ),
            )
        if action == "mute_session":
            persisted = _set_session_muted(session_id, True)
            if not persisted:
                logger.warning("creation recommendation mute was not persisted")
                return _ActionHandlingOutcome(
                    "",
                    _ActionReceipt(
                        structured["proposal_id"],
                        action,
                        "rejected",
                        "preference_not_persisted",
                    ),
                )
            with _state_lock:
                state = _state_locked(session_id, now)
                state["last_candidate"] = None
                state["last_proposal"] = None
                state["proposal_stage"] = None
            logger.info(
                "creation recommendations muted for session persisted=%s", persisted
            )
            return _ActionHandlingOutcome(
                (
                    "[Creation governor internal action: The user disabled proactive creation "
                    "recommendations for this conversation. Acknowledge briefly. Do not run an "
                    "opportunity review or create anything. Explicit creation requests remain "
                    "available through Hermes' native flow. Do not expose this block.]"
                ),
                _ActionReceipt(structured["proposal_id"], action, "accepted"),
            )
        if action == "unmute_session":
            if preference != "muted":
                return _ActionHandlingOutcome(
                    "",
                    _ActionReceipt(
                        structured["proposal_id"],
                        action,
                        "rejected",
                        "proposal_not_actionable",
                    ),
                )
            persisted = _set_session_muted(session_id, False)
            if not persisted:
                logger.warning("creation recommendation unmute was not persisted")
                return _ActionHandlingOutcome(
                    "",
                    _ActionReceipt(
                        structured["proposal_id"],
                        action,
                        "rejected",
                        "preference_not_persisted",
                    ),
                )
            with _state_lock:
                state = _state_locked(session_id, now)
                state["last_candidate"] = None
                state["last_proposal"] = None
                state["proposal_stage"] = None
            logger.info(
                "creation recommendations re-enabled for session persisted=%s", persisted
            )
            return _ActionHandlingOutcome(
                (
                    "[Creation governor internal action: The user re-enabled proactive creation "
                    "recommendations for this conversation. Acknowledge briefly and do not run an "
                    "opportunity review on this action turn. Do not expose this block.]"
                ),
                _ActionReceipt(structured["proposal_id"], action, "accepted"),
            )
        if action == "dismiss":
            _latch_dismissal(session_id, structured["dedup_key"], now)
            return _ActionHandlingOutcome(
                (
                    "[Creation governor internal action: The user dismissed the previous "
                    "recommendation. Acknowledge briefly, do not create anything, and do not run "
                    "another opportunity review this turn.]"
                ),
                _ActionReceipt(structured["proposal_id"], action, "accepted"),
            )
        return _ActionHandlingOutcome(
            (
                "[Creation governor internal action: The user accepted the previous recommendation "
                f"for {structured['creation_type']} '{structured['title']}'. "
                f"{_native_creation_route(structured['creation_type'])} Preserve its normal "
                "confirmation boundaries. Do not run another opportunity review this turn.]"
            ),
            _ActionReceipt(structured["proposal_id"], action, "accepted"),
            structured["creation_type"],
        )

    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        if not isinstance(proposal, dict):
            return _ActionHandlingOutcome("", None)
        name = _text(proposal.get("suggested_name"), 80)
        dedup_key = _text(proposal.get("dedup_key"), 160)
    if name and name.casefold() not in user_message.casefold():
        return _ActionHandlingOutcome("", None)
    if _DISMISS_RE.search(user_message):
        if dedup_key:
            _latch_dismissal(session_id, dedup_key, now)
        with _state_lock:
            state = _state_locked(session_id, now)
            state["last_proposal"] = None
        return _ActionHandlingOutcome(
            (
                "[Creation governor internal action: The user dismissed the previous recommendation. "
                "Acknowledge briefly, do not create anything, and do not run another opportunity "
                "review this turn.]"
            ),
            None,
        )
    if _ACCEPT_RE.search(user_message):
        return _ActionHandlingOutcome(
            (
                "[Creation governor internal action: The user accepted the previous recommendation. "
                f"{_native_creation_route(proposal.get('creation_type'))} Preserve its normal "
                "confirmation boundaries. Do not run another opportunity review this turn.]"
            ),
            None,
        )
    return _ActionHandlingOutcome("", None)


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    raw_user_message = str(kwargs.get("user_message") or "")
    # Marker is appended after the visible welcome instruction. Inspect a
    # bounded tail before the onboarding-profile fast bypass: current App sends
    # the final welcome from main, but older mixed deployments may still route
    # that one turn through onboarding.
    onboarding_welcome = _decode_onboarding_welcome_spec(raw_user_message[-16_000:])
    # Onboarding is a fixed, latency-sensitive state machine and never offers
    # reusable-object recommendations.  Skip the checkpoint before it mutates
    # governor state or invokes its auxiliary model.  The marked final welcome
    # is the sole deterministic exception and never invokes the auxiliary model.
    if (
        _text(kwargs.get("profile_name"), 80).strip().lower() == "onboarding"
        and onboarding_welcome is None
    ):
        return None

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
    receipt_transport = _receipt_transport(kwargs)
    _invocation_scope.set(
        (
            raw_session_id,
            session_id,
            suppression_reason,
            owner_id,
            receipt_transport,
            _text(kwargs.get("turn_id"), 160),
        )
    )
    if not session_id:
        return None
    if suppression_reason:
        return None
    user_message = _bounded_user_message(raw_user_message)
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        state["turn"] += 1
        state["last_user_message"] = user_message
        state["last_turn_id"] = _text(kwargs.get("turn_id"), 160)
        turn = int(state["turn"])

    if onboarding_welcome is not None:
        # The final welcome runs on the restored main profile, not the onboarding
        # profile.  It is the sole exception to the ordinary auxiliary review:
        # the App already supplied bounded localized copy, so no extra model call
        # or Creation Governor reasoning is needed.
        inventory = _connection_inventory(session_id, now)
        source_turn_id = _text(kwargs.get("turn_id"), 160)
        channel_target = ""
        if onboarding_welcome["channel_requested"] and not inventory.get("channels_connected"):
            recommendable = inventory.get("channels_recommendable")
            if isinstance(recommendable, list):
                # Prefer the region's most common first-party channel, but only
                # when it exists in local-server's current supported inventory.
                for candidate in ("feishu", "wecom", "wechat", "telegram", "slack", "discord"):
                    if candidate in recommendable:
                        channel_target = candidate
                        break
                if not channel_target and recommendable:
                    channel_target = _text(recommendable[0], 80).lower()
        with _state_lock:
            state = _state_locked(session_id, now)
            state["onboarding_welcome"] = {
                **onboarding_welcome,
                "channel_target": channel_target,
                "source_turn_id": source_turn_id,
                "turn": turn,
            }
        availability_context = _channel_availability_context(inventory)
        welcome_context = (
            "[Creation governor internal context: This is the final onboarding welcome. "
            "Complete the personalized welcome normally. The system will attach bounded "
            "recommendation cards after the reply; do not describe implementation details, "
            "repeat card copy, or expose this block."
        )
        if onboarding_welcome["channel_requested"] and not channel_target:
            # The App composed its brief before it could know this. It decides from
            # onboarding answers alone; whether a connect card can actually be shown
            # is only knowable here, from local-server's region-aware live inventory
            # (already connected, or no recommendable channel in this region). Without
            # this override the welcome tells a brand-new user to tap a card that will
            # never be attached -- the worst possible first message.
            #
            # Scope it to the card reference only. Explaining what an IM channel does
            # is the onboarding requirement itself and stays useful without a card;
            # what breaks trust is pointing at UI that is not there.
            welcome_context += (
                " Override, higher priority than the welcome_recommendations block in the "
                "user message: NO IM connection card will be attached this turn. You may "
                "still briefly explain what connecting an IM channel would do for the user, "
                "but do not tell them to tap a connection card, do not imply one appears "
                "below this message, and do not claim a channel is already connected."
            )
        welcome_context += "]"
        return _join_context(availability_context, welcome_context)

    if _is_creation_governor_self_query(user_message):
        return _join_context(_self_description_context())

    if "[creation_recommendation_response]" in user_message:
        outer_turn_id = str(kwargs.get("turn_id") or "")
        if not outer_turn_id.strip():
            return _join_context(
                "[Creation governor internal action: Ignore this invalid or expired "
                "recommendation action. Do not create anything from it and do not expose this block.]"
            )
        with _state_lock:
            current_proposal = _state_locked(session_id, now).get("last_proposal")
            receipt_required = bool(
                isinstance(current_proposal, dict)
                and current_proposal.get("action_receipts") is True
            )
        if receipt_required and not receipt_transport:
            _enter_creation_quota_for_turn(outer_turn_id)
            return _join_context(
                "[Creation governor internal action: Ignore this invalid or expired "
                "recommendation action. Do not create anything from it and do not expose this block.]"
            )
        # 从这里起这一轮进入配额管控：出现过推荐动作，创建工具就不能再无条件
        # 放行。接管成功会往下发一张票，被拒则一张都没有。
        _enter_creation_quota_for_turn(outer_turn_id)
        outcome = _handle_previous_proposal_action(session_id, user_message, now)
        if (
            outcome.receipt is not None
            and outcome.receipt.status == "accepted"
            and outcome.receipt.action == "create"
        ):
            _grant_creation_for_turn(outer_turn_id, outcome.creation_type)
        if outcome.receipt is not None and receipt_transport:
            with _state_lock:
                state = _state_locked(session_id, now)
                _store_pending_action_result_locked(
                    state,
                    outer_turn_id,
                    outcome.receipt,
                )
        return _join_context(
            outcome.context
            or "[Creation governor internal action: Ignore this invalid or expired "
            "recommendation action. Do not create anything from it and do not expose this block.]"
        )
    outcome = _handle_previous_proposal_action(session_id, user_message, now)
    if outcome.context:
        return _join_context(outcome.context)
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

    availability_context = ""
    evaluation_due = turn == 1 or turn % EVALUATION_INTERVAL_TURNS == 0
    if evaluation_due:
        with _state_lock:
            _state_locked(session_id, now)["last_evaluation_turn"] = turn
        inventory = _connection_inventory(session_id, now)
        availability_context = _channel_availability_context(inventory)
        candidate = _run_forced_evaluation(
            user_message=user_message,
            conversation_history=kwargs.get("conversation_history"),
            connection_context=_connection_inventory_context(inventory),
        )
        if candidate is not None:
            candidate_result = _consider_candidate(
                session_id, candidate, now, _text(kwargs.get("turn_id"), 160)
            )
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
                    if proposal.get("creation_type") in ATTACHMENT_DELIVERED_TYPES:
                        # 采集即发射（真机体验修复）：判定在 pre_llm 就完成了，
                        # 原先卡片却等到 turn 收尾的 transform 钩子才发射——用户
                        # 要多等整个回复生成（Pro 模型 20s+）才能看到卡。此处
                        # 判定通过立即发射，卡片先于正文出现；发射成功记
                        # last_delivery_turn 让 transform 钩子跳过重复发射，
                        # 失败（无活跃流等）则保持原状、由 transform 收尾兜底。
                        if _emit_recommendation_attachment(session_id, proposal):
                            logger.info(
                                "attachment recommendation emitted early type=%s target=%s turn=%s",
                                proposal.get("creation_type"),
                                _text(proposal.get("target"), 80),
                                turn,
                            )
                            with _state_lock:
                                # 两个字段都要推进：last_delivery_turn 让本轮的
                                # _transform_llm_output 认得「已投递」不再重复；
                                # last_prompt_turn 是后续轮话术（指向卡片那句）的
                                # 唯一依据——只设前者会让 transform 提前 return，
                                # 后续轮 turns_since 永远算不出来，话术整条丢失。
                                early_state = _state_locked(session_id, now)
                                early_state["last_delivery_turn"] = turn
                                early_state["last_prompt_turn"] = turn
            return _join_context(
                carry_context,
                availability_context,
                _main_model_review_context(evaluation_completed=True),
                delivery_context,
            )
        logger.info(
            "creation opportunity checkpoint unavailable; falling back to main-model review"
        )
    else:
        # 非评估轮不发起网络请求，只复用会话内缓存的库存（TTL 内），
        # 保证主模型每一轮都有区域口径而不增加时延。
        with _state_lock:
            availability_context = _channel_availability_context(
                _state_locked(session_id, now).get("connection_inventory")
            )

    with _state_lock:
        state = _state_locked(session_id, now)
        if _prompt_is_cooling_down(state):
            return _join_context(carry_context, availability_context)
    return _join_context(
        carry_context,
        availability_context,
        _main_model_review_context(evaluation_completed=False),
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
    if candidate.get("action_receipts") is True:
        payload["action_receipts"] = True
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


def _action_result_envelope(result: _ActionReceipt) -> str:
    payload = {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": result.proposal_id,
        "action": result.action,
        "status": result.status,
    }
    if result.reason_code is not None:
        payload["reason_code"] = result.reason_code
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"<!--creation-recommendation-action-result {encoded}-->"


def _build_recommendation_attachment(proposal: dict[str, Any]) -> dict[str, Any] | None:
    """Project one proposal onto the existing attachment wire contract."""
    creation_type = proposal.get("creation_type")
    target = _text(proposal.get("target"), 80).lower()
    proposal_id = _text(proposal.get("proposal_id"), 80)
    if creation_type not in ATTACHMENT_DELIVERED_TYPES or not proposal_id:
        return None
    if creation_type in CONNECTION_TYPES and not target:
        return None
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
    elif creation_type == AGENT_TEMPLATE_TYPE:
        template_id = _text(proposal.get("template_id"), 160)
        title = _text(proposal.get("suggested_name"), 80)
        reason = _text(proposal.get("reason"), 400)
        if not template_id or not title or not reason:
            return None
        kind = "agent-template.recommendation"
        payload = {"template_id": template_id, "title": title, "reason": reason}
        actions = [{"id": "dismiss"}, {"id": "open", "style": "primary"}]
    else:
        kind = "artifact.recommendation"
        payload = {
            "title": _text(proposal.get("suggested_name"), 80),
            "reason": _text(proposal.get("reason"), 400),
            "confidence": proposal.get("confidence"),
            **(
                {"artifact_type": _text(proposal.get("artifact_type"), 40)}
                if _text(proposal.get("artifact_type"), 40)
                else {}
            ),
        }
        actions = [{"id": "dismiss"}, {"id": "accept", "style": "primary"}]
    expires_at = proposal.get("expires_at")
    return {
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


def _emit_recommendation_attachment(session_key: str, proposal: dict[str, Any]) -> bool:
    """Deliver a connection/artifact/Agent-template proposal as an attachment.

    需求 6.2：wire 只带语义（kind/payload/action id），推荐卡文案由客户端
    i18n 决定（artifact 的 title/reason 是模型按用户语言产出的内容字段）。
    发射失败（无活跃流 / 老客户端链路）静默降级——推荐是锦上添花，绝不
    进入正文文本通道。
    """
    ctx = _plugin_ctx
    if ctx is None or not hasattr(ctx, "emit_attachment"):
        return False
    attachment = _build_recommendation_attachment(proposal)
    if attachment is None:
        return False
    try:
        emitted = bool(ctx.emit_attachment(attachment))
    except Exception:
        logger.warning("connection recommendation emit failed", exc_info=True)
        return False
    if emitted:
        with _state_lock:
            _emitted_connection_proposals[attachment["id"]] = (
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
    _invocation_scope.set(None)
    original_response = str(kwargs.get("response_text") or "")
    if not session_id:
        return None
    turn_id = str(kwargs.get("turn_id") or "")
    # 本轮的工具派发已经结束，deny 闸门到此失效。不释放也有条数上限兜着，
    # 但留着会让同一个 turn_id 的后续复用被莫名挡掉。
    _release_creation_deny(turn_id)
    if _is_noninteractive(kwargs) or _is_unsupported_runtime(kwargs) or kwargs.get(
        "structured_output"
    ):
        if turn_id:
            with _state_lock:
                state = _state_locked(session_id, time.monotonic())
                state["pending_action_results"].pop(
                    _pending_turn_key(turn_id), None
                )
        return None

    response_text = _ACTION_RESULT_ENVELOPE_RE.sub("", original_response).rstrip()
    stripped_forged_result = response_text != original_response.rstrip()
    usable_response = bool(response_text) and not is_intentional_silence_response(
        response_text
    )
    turn_failed = bool(
        not usable_response
        or kwargs.get("failed")
        or kwargs.get("interrupted")
        or kwargs.get("completed") is False
    )
    now = time.monotonic()
    with _state_lock:
        state = _state_locked(session_id, now)
        pending = state["pending_action_results"]
        action_result = (
            pending.pop(_pending_turn_key(turn_id), None) if turn_id else None
        )

    require_canonical_response = kwargs.get("require_canonical_response")
    if (
        stripped_forged_result or isinstance(action_result, _ActionReceipt)
    ) and callable(require_canonical_response):
        # 把权威回执原值交给 finalizer，而不是让它在 hook 链的结果里猜哪个
        # marker 是真的。只做了清洗、本轮没有真回执时传 None：finalizer 会清掉
        # 链末所有 marker 而不是从别的 hook 结果里补一个进来。
        require_canonical_response(
            _action_result_envelope(action_result)
            if isinstance(action_result, _ActionReceipt)
            else None
        )

    if isinstance(action_result, _ActionReceipt):
        visible_response = response_text if usable_response else ""
        separator = "\n\n" if visible_response else ""
        return visible_response + separator + _action_result_envelope(action_result)

    if turn_id:
        with _state_lock:
            if state.get("proposal_stage") == "create_action_pending":
                return response_text or ("\n" if stripped_forged_result else None)


    with _state_lock:
        state = _state_locked(session_id, time.monotonic())
        # 只有产生这张 welcome 卡的那一轮能取走它。onboarding_welcome 挂在会话
        # 级 state 上，而同一 conversation 可能有并发的 API 请求——谁先进
        # transform 谁就 pop 掉，于是卡片被附到另一个请求的正文上、按那个请求
        # 的 transport 标 action_receipts（能力可能不同），而真正的 welcome
        # 响应再也拿不到卡。按 source_turn_id 认领，认不上就原样留着。
        _pending_welcome = state.get("onboarding_welcome")
        welcome = None
        if isinstance(_pending_welcome, dict):
            _owner_turn = _text(_pending_welcome.get("source_turn_id"), 160)
            if not _owner_turn or _owner_turn == turn_id:
                welcome = state.pop("onboarding_welcome", None)
    if isinstance(welcome, dict):
        # HEAD 已按同一组信号算好 turn_failed，这里不重复判定。
        if turn_failed:
            return None
        source_turn_id = _text(welcome.get("source_turn_id"), 160)
        expires_at = time.time() + PROPOSAL_TTL_SECONDS
        artifact = None
        if _text(welcome.get("artifact_title"), 80):
            artifact = {
                "creation_type": ARTIFACT_TYPE,
                "artifact_type": "app",
                "suggested_name": _text(welcome.get("artifact_title"), 80),
                "reason": _text(welcome.get("artifact_reason"), 400),
                "confidence": 1.0,
                "dedup_key": _semantic_dedup_key(
                    "onboarding-app", ARTIFACT_TYPE, _text(welcome.get("artifact_title"), 80)
                ),
                "proposal_id": hashlib.sha256(
                    f"{session_id}|onboarding-app|{source_turn_id}".encode("utf-8")
                ).hexdigest()[:32],
                "expires_at": expires_at,
                "source_turn_id": source_turn_id,
            }
        task = {
            "creation_type": "task",
            "suggested_name": _text(welcome.get("task_title"), 80),
            "reason": _text(welcome.get("task_reason"), 400),
            "proposal_text": _text(welcome.get("task_proposal"), 500),
            "confidence": 1.0,
            "dedup_key": _semantic_dedup_key(
                "onboarding-task", "task", _text(welcome.get("task_title"), 80)
            ),
            "proposal_id": hashlib.sha256(
                f"{session_id}|onboarding-task|{source_turn_id}".encode("utf-8")
            ).hexdigest()[:32],
            "expires_at": expires_at,
            "evidence_turn_ids": [source_turn_id] if source_turn_id else [],
            "source_turn_id": source_turn_id,
            # 引导页最后一屏的 Task 卡跟常规推荐走同一个文本信封，却是从这里
            # 直接返回的，绕过了下面那次统一的 action_receipts 标注。漏标的后果
            # 不是「少个字段」：Web 会把它当老卡按猜测结算，而下一轮用户真点
            # 「创建」时 receipt_required 读到 False、不落 pending receipt，
            # local-server 那边照样要收据，于是这张卡必然 fail-closed。
            "action_receipts": bool(_receipt_transport(kwargs)),
        }
        channel_target = _text(welcome.get("channel_target"), 80).lower()
        channel_emitted = False
        if channel_target:
            channel = {
                "creation_type": "channel",
                "target": channel_target,
                "suggested_name": channel_target,
                "reason": "",
                "confidence": 1.0,
                "dedup_key": _semantic_dedup_key(channel_target, "channel", channel_target),
                "proposal_id": hashlib.sha256(
                    f"{session_id}|onboarding-channel|{channel_target}|{source_turn_id}".encode("utf-8")
                ).hexdigest()[:32],
                "expires_at": expires_at,
                "source_turn_id": source_turn_id,
            }
            channel_emitted = _emit_recommendation_attachment(session_id, channel)
        artifact_emitted = bool(artifact) and _emit_recommendation_attachment(session_id, artifact)
        agent_templates_emitted = 0
        for index, template in enumerate(welcome.get("agent_templates") or []):
            template_id = _text(template.get("template_id"), 160)
            agent_template = {
                "creation_type": AGENT_TEMPLATE_TYPE,
                "template_id": template_id,
                "suggested_name": _text(template.get("title"), 80),
                "reason": _text(template.get("reason"), 400),
                "confidence": 1.0,
                "dedup_key": _semantic_dedup_key(
                    template_id, AGENT_TEMPLATE_TYPE, _text(template.get("title"), 80)
                ),
                "proposal_id": hashlib.sha256(
                    f"{session_id}|onboarding-agent-template|{index}|{template_id}|{source_turn_id}".encode("utf-8")
                ).hexdigest()[:32],
                "expires_at": expires_at,
                "source_turn_id": source_turn_id,
            }
            if _emit_recommendation_attachment(session_id, agent_template):
                agent_templates_emitted += 1
        with _state_lock:
            state = _state_locked(session_id, time.monotonic())
            state["last_candidate"] = dict(task)
            state["last_proposal"] = dict(task)
            state["proposal_stage"] = "proposal_shown"
            state["candidate_turn"] = -10_000
            state["last_prompt_turn"] = int(state["turn"])
            state["last_delivery_turn"] = int(state["turn"])
        logger.info(
            "onboarding welcome recommendations emitted channel=%s artifact=%s agent_templates=%d task=1",
            channel_target if channel_emitted else "none",
            "app" if artifact_emitted else "none",
            agent_templates_emitted,
        )
        return response_text + "\n\n" + _recommendation_envelope(task)

    if not usable_response:
        with _state_lock:
            if state.get("proposal_stage") == "create_action_pending":
                state["proposal_stage"] = "proposal_shown"
        return response_text or ("\n" if stripped_forged_result else None)
    if turn_failed:
        with _state_lock:
            if state.get("proposal_stage") == "create_action_pending":
                state["proposal_stage"] = "proposal_shown"
            elif state.get("proposal_stage") == "proposal_shown":
                _discard_staged_proposal_locked(
                    session_id, state, release_claim=True
                )
        return response_text if stripped_forged_result else None
    with _state_lock:
        if state.get("proposal_stage") == "create_action_pending":
            _discard_staged_proposal_locked(
                session_id, state, release_claim=False
            )
    if _is_session_muted(session_id):
        return response_text if stripped_forged_result else None
    with _state_lock:
        state = _state_locked(session_id, now)
        proposal = state.get("last_proposal")
        current_turn = int(state["turn"])
        if (
            not isinstance(proposal, dict)
            or int(state["candidate_turn"]) != current_turn
            or int(state["last_delivery_turn"]) == current_turn
            # 候选只允许产生它的那一轮消费：state["turn"] 是共享计数器，同一
            # conversation 并发两轮时它认不出「这张卡是谁的」，会把 A 轮的卡
            # 挂到 B 轮的回复下面。
            or _delivery_turn_mismatch(proposal, turn_id)
        ):
            return response_text if stripped_forged_result else None

    if "<!--creation-recommendation:start " in response_text:
        return response_text if stripped_forged_result else None
    if _is_session_muted(session_id):
        return response_text if stripped_forged_result else None
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
        return response_text if stripped_forged_result else None
    with _state_lock:
        state = _state_locked(session_id, now)
        current = state.get("last_proposal")
        if (
            not isinstance(current, dict)
            or current.get("proposal_id") != proposal.get("proposal_id")
            or int(state["last_delivery_turn"]) == current_turn
        ):
            return response_text if stripped_forged_result else None
        current["action_receipts"] = bool(_receipt_transport(kwargs))
        proposal = dict(current)
        state["last_prompt_turn"] = current_turn
        state["last_delivery_turn"] = current_turn
        state["candidate_turn"] = -10_000
    if proposal.get("creation_type") in ATTACHMENT_DELIVERED_TYPES:
        # 连接/artifact 推荐走结构化 attachment 通道（channel.connect /
        # connector.connect / artifact.recommendation 卡），不追加文本信封；
        # 发射失败（无活跃流）静默降级，正文原样返回。
        # 位置有两个约束：① 在 _response_delivery_block_reason 闸之后——被判定
        # 「本轮没有实际交付」而抑制的提案同样不该出卡；② 在投递记账之后——
        # 出卡本身就是一次投递，跳过记账会让 last_delivery_turn 不推进，
        # 下一轮取不到 last_proposal，后续轮话术（指向卡片那句）整条丢失。
        emitted = _emit_recommendation_attachment(session_id, proposal)
        logger.info(
            "attachment recommendation %s type=%s target=%s turn=%s",
            "emitted" if emitted else "skipped (no active stream)",
            proposal.get("creation_type"),
            _text(proposal.get("target"), 80),
            current_turn,
        )
        return None
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
    current_invocation = _is_current_invocation(kwargs)
    if current_invocation and invocation is not None and invocation[2]:
        return json.dumps({"status": "not_proposed", "reason": invocation[2]})
    if not current_invocation and (
        _is_noninteractive(kwargs)
        or _is_unsupported_runtime(kwargs)
        or kwargs.get("structured_output")
    ):
        return json.dumps({"status": "not_proposed", "reason": "unsupported_runtime"})
    session_id = _session_key(kwargs)
    if not session_id:
        return json.dumps({"status": "invalid", "error": "missing_session_id"})
    result = _consider_candidate(
        session_id, args, time.monotonic(), _text(kwargs.get("turn_id"), 160)
    )
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
        # 这两个是进程级的，不清会在测试之间泄漏（同 turn_id 复用时表现成
        # 「闸门莫名已经在了」）。
        _creation_turn_quota.clear()
        _creation_turn_refs.clear()
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
    # 排在最后：flow 测试按索引取前两个 hook，新增注册不该挤动它们的位置。
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
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
