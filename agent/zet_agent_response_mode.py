"""Bind trusted high-risk skill execution to an exact App turn."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)

_TurnIdentity = tuple[str, object]

_TRUSTED_SKILL_SOURCE_LIMIT_BYTES = 512 * 1024
_TRUSTED_SKILL_SCAN_LIMIT = 512
_TRUSTED_INTEGRITY_MANIFEST_LIMIT_BYTES = 4 * 1024 * 1024
_TRUSTED_INTEGRITY_MANIFEST_PATH = "skills/.zettlab-integrity.json"
_TRUSTED_INTEGRITY_SCHEMA = "zettlab.presets.integrity.v1"
_TRUSTED_INTEGRITY_SIGNATURE_LIMIT_BYTES = 16 * 1024
_TRUSTED_INTEGRITY_SIGNATURE_PATH = "skills/.zettlab-integrity.sig.json"
_TRUSTED_INTEGRITY_SIGNATURE_SCHEMA = "zettlab.presets.integrity-signature.v1"
_TRUSTED_PRESETS_PUBLIC_KEYS_B64 = {
    "presets-cn-202605": "JuMytoCQjauAy3AAxvWIDu+To9FTTzAK0ZhKJqlkPxw=",
    "presets-intl-202605": "Bnjb1gl3hRQ2lsi+ntLaeNZzWzUUOEDDj7GG2UYavLA=",
}
_TRUSTED_PRESETS_DEV_KEY_ID_ENV = "ZETTLAB_PRESETS_DEV_KEY_ID"
_TRUSTED_PRESETS_DEV_PUBLIC_KEY_ENV = "ZETTLAB_PRESETS_DEV_PUBLIC_KEY_B64"
_TRUSTED_PRESETS_DEV_KEY_ID_PREFIX = "presets-dev-"
_TRUSTED_PRESETS_DEV_DIRECTORY_PREFIX = "dev-"
_ATTESTATION_FIELD = "_zet_agent_trusted_skill_attestation"
_ATTESTATION_TTL_SECONDS = 30.0
_ATTESTATION_MAX_ENTRIES = 64
_VIDEO_EDIT_SKILL_PATH = "skills/video-edit-workflow-mini/SKILL.md"
_VIDEO_EDIT_DIRECT_TOOLS = frozenset({"clarify", "terminal", "todo"})
_CAMERA_SKILL_PATH = "skills/camsnap/SKILL.md"
_CAMERA_DIRECT_TOOLS = frozenset({"terminal", "vision_analyze"})
_CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PRINTER3D_SKILL_PATHS = frozenset({
    "skills/printer3d/SKILL.md",
    "skills/printer3d-control/SKILL.md",
})
_PRINTER3D_DIRECT_TOOLS = frozenset({"terminal"})
_CAMERA_INTENT_RE = re.compile(
    r"(?:摄像头|镜头|camera).{0,48}(?:查看|看下|看看|获取|列出|截图|快照|画面|短视频|录像|统计|分析|识别|人数|多少人|有没有人|view|list|snap|snapshot|image|frame|clip|doctor|analy[sz]e|count|people|person)"
    r"|(?:查看|看下|看看|获取|列出|截图|快照|画面|短视频|录像|统计|分析|识别|人数|多少人|有没有人|view|list|snap|snapshot|image|frame|clip|doctor|analy[sz]e|count|people|person).{0,48}(?:摄像头|镜头|camera)",
    re.IGNORECASE | re.DOTALL,
)
# Inventory/status requests intentionally receive a narrower capability than
# media requests: they may list normalized state, but may not capture media.
_CAMERA_INVENTORY_INTENT_RE = re.compile(
    r"(?:摄像头|镜头|camera).{0,48}(?:检查|状态|连接情况|是否连接|是否可用|可用状态|check|status|connected|available|online)"
    r"|(?:检查|状态|连接情况|是否连接|是否可用|可用状态|check|status|connected|available|online).{0,48}(?:摄像头|镜头|camera)",
    re.IGNORECASE | re.DOTALL,
)
_HARDWARE_INVENTORY_INTENT_RE = re.compile(
    r"(?:(?:所有|全部|当前|整体|已连接(?:的)?)(?:硬件|设备|连接器).{0,32}(?:检查|查看|列出|有哪些|状态|连接情况|是否可用)"
    r"|(?:检查|查看|列出).{0,32}(?:所有|全部|当前|整体|已连接(?:的)?)(?:硬件|设备|连接器)"
    r"|(?:硬件|设备|连接器)(?:总览|列表|状态|连接情况|有哪些)"
    r"|(?:(?:all|current|connected)\s+(?:hardware|devices?|connectors?).{0,32}(?:check|show|list|status|available)"
    r"|(?:check|show|list).{0,32}(?:all|current|connected)\s+(?:hardware|devices?|connectors?)"
    r"|(?:hardware|devices?|connectors?)\s+(?:overview|list|status)))",
    re.IGNORECASE | re.DOTALL,
)
# A camera noun is optional only for a whole-message, high-confidence snapshot
# command. Anchoring prevents quoted instructions or surrounding document text
# from minting visual-data authority.
_CAMERA_DIRECT_SNAPSHOT_INTENT_RE = re.compile(
    r"^(?:(?:请|麻烦)(?:帮我)?|帮我|给我)?"
    r"(?:拍|抓取|获取|截取)(?:一张|张|一个|个)?(?:当前|最新)?快照"
    r"(?:吧|一下|看看)?[。！？!?]?$",
    re.IGNORECASE,
)
_CAMERA_CONTINUATION_INTENT_RE = re.compile(
    r"^(?:(?:请|麻烦)(?:帮我)?|帮我|给我)?"
    r"(?:再|重新|继续|接着)"
    r"(?:获取|查看|看下|看看|抓取|截取|分析|统计|识别)?"
    r"(?:一下|下)?(?:当前|最新|刚才|这张)?(?:的)?"
    r"(?:画面|图片|图像|快照|截图)"
    r"(?:[，,\s]*(?:并|然后|再)?(?:分析|统计|识别|看看|看下)?"
    r"(?:一下|下)?(?:当前)?(?:画面中?|图中)?(?:有|的)?(?:多少(?:个)?人|人数|有没有人|人员))?"
    r"(?:吧|一下)?[。！？!?，,\s]*$",
    re.IGNORECASE,
)
_PRINTER3D_INTENT_RE = re.compile(
    r"(?:3d\s*打印机|三维打印机|printer).{0,32}(?:查看|列出|状态|进度|暂停|继续|恢复|取消|list|status|progress|pause|resume|cancel)"
    r"|(?:查看|列出|状态|进度|暂停|继续|恢复|取消|list|status|progress|pause|resume|cancel).{0,32}(?:3d\s*打印机|三维打印机|printer)",
    re.IGNORECASE | re.DOTALL,
)
_HARDWARE_ENROLLMENT_FENCE = "zettlab-hardware-enrollment-intent"
_HARDWARE_ENROLLMENT_BLOCK_RE = re.compile(
    r"\s*```zettlab-hardware-enrollment-intent\s*\r?\n[\s\S]*?\r?\n```",
    re.IGNORECASE,
)
_LEGACY_HARDWARE_CAMERA_INTENT_BLOCK_RE = re.compile(
    r"\s*```zettlab-hardware-camera-intent\s*\r?\n[\s\S]*?\r?\n```",
    re.IGNORECASE,
)
_HARDWARE_ENROLLMENT_ACTION_RE = re.compile(
    r"(?:接入|连接|添加|安装|配对|发现|配置|设置)"
    r"|(?:\bconnect\b|\badd\b|\binstall\b|\benroll\b|\bdiscover\b|\bset\s*up\b|\bpair\b)"
    r"|(?:接続|追加|セットアップ|検出|登録)"
    r"|(?:연결|추가|설정|검색|등록)"
    r"|(?:verbinden|hinzuf[uü]gen|einrichten|installieren|erkennen)"
    r"|(?:connecter|ajouter|installer|configurer|d[ée]tecter)"
    r"|(?:conectar|a[ñn]adir|agregar|instalar|configurar|detectar)"
    r"|(?:collegare|connettere|aggiungere|installare|configurare|rilevare)",
    re.IGNORECASE,
)
_HARDWARE_ENROLLMENT_CONNECTION_STATE_RE = re.compile(
    r"(?:已(?:经)?|正在|正|当前|尚未|未|没有)连接(?:到|上|着|过|好|成功)?(?:的)?"
    r"|连接(?:中|的|状态|列表|信息|详情)"
    r"|\b(?:connected|connecting|connection)\b",
    re.IGNORECASE,
)
_HARDWARE_ENROLLMENT_META_OR_DIAG_RE = re.compile(
    r"(?:总结|翻译|解释|改写|这句话|示例|文档|代码|正则|失败|报错|错误|无法|不能)"
    r"|(?:summari[sz]e|translate|explain|rewrite|example|documentation|regex|failed|error|cannot|can't)",
    re.IGNORECASE,
)
_HARDWARE_ENROLLMENT_GENERIC_RE = re.compile(
    r"(?:硬件|设备|hardware|devices?|ハードウェア|デバイス|하드웨어|장치)",
    re.IGNORECASE,
)
_HARDWARE_ENROLLMENT_TYPE_RES = (
    (
        "camera",
        re.compile(
            r"(?:摄像头|攝像頭|局域网相机|局域網相機|camera|カメラ|카메라|kamera|cam[ée]ra|c[áa]mara|telecamera)",
            re.IGNORECASE,
        ),
    ),
    (
        "printer3d",
        re.compile(
            r"(?:3d\s*打印机|3d\s*印表機|三维打印机|三維印表機|3d\s*printer|3d\s*プリンター|3d\s*프린터|3d[-\s]*drucker|imprimante\s*3d|impresora\s*3d|stampante\s*3d)",
            re.IGNORECASE,
        ),
    ),
    (
        "pc_node",
        re.compile(
            r"(?:电脑节点|電腦節點|电脑端|電腦端|这台电脑|這台電腦|我的电脑|我的電腦|电脑|電腦|pc\s*node|computer\s*node|desktop\s*node|\bcomputer\b|\bdesktop\b|コンピューターノード|컴퓨터\s*노드)",
            re.IGNORECASE,
        ),
    ),
)
_VIDEO_EDIT_POLICY_VIOLATION_RETRIES = 2
_VIDEO_EDIT_RESUME_TTL_SECONDS = 3 * 60 * 60
_VIDEO_EDIT_RESUME_MAX_SESSIONS = 8
_CAMERA_RESUME_TTL_SECONDS = 5 * 60
_CAMERA_RESUME_MAX_SESSIONS = 8
_CAMERA_ANALYSIS_QUESTION_LIMIT = 4096
_VIDEO_EDIT_MEMORY_ACTIONS = frozenset(
    {"plan-forget", "plan-hard", "plan-migrate", "plan-reset", "plan-success"}
)
_VIDEO_FILE_SUFFIXES = (
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
)
_VIDEO_EDIT_CN_RE = re.compile(r"(?:视频)?(?:剪辑|剪片|剪成|成片)|做(?:个|一条)?\s*(?:vlog|视频)", re.IGNORECASE)
_VIDEO_EDIT_EN_RE = re.compile(
    r"(?:\b(?:edit|trim|cut|render)\b.{0,32}\b(?:video|clip|footage|movie|vlog)\b"
    r"|\b(?:video|clip|footage|movie|vlog)\b.{0,32}\b(?:edit|trim|cut|render)\b"
    r"|\bmake\b.{0,32}\b(?:video|vlog|movie)\b)",
    re.IGNORECASE | re.DOTALL,
)
_VIDEO_EDIT_CONTINUATION_CN_RE = re.compile(
    r"^(?:请|请帮我|帮我|让我们)?"
    r"(?:(?:继续|接着|恢复)"
    r"(?:(?:剪辑|处理|渲染|查询|检查|查看|轮询|下载|交付)"
    r"(?:这|这个|本次|当前|刚才|上次|原来|同一)?"
    r"(?:的)?"
    r"(?:\s*\d+\s*(?:段|个))?"
    r"(?:视频|素材|剪辑任务|任务|项目|成片)?"
    r"|(?:这|这个|本次|当前|刚才|上次|原来|同一)"
    r"(?:的)?"
    r"(?:\s*\d+\s*(?:段|个))?"
    r"(?:视频|素材|剪辑任务|任务|项目|成片))?"
    r"|(?:重新|再)(?:查询|检查|查看|轮询|下载|交付|恢复)"
    r"(?:这|这个|本次|当前|刚才|上次|原来|同一)?"
    r"(?:的)?"
    r"(?:\s*\d+\s*(?:段|个))?"
    r"(?:视频|素材|剪辑任务|任务|项目|成片|状态))"
    r"(?:[，,\s]*(?:不要|无需|别)(?:重新|再次)?"
    r"(?:上传|压缩|创建|新建)(?:新(?:的)?)?(?:素材|项目|任务)?"
    r"(?:(?:[或、和]|[，,]\s*也?\s*)"
    r"(?:(?:不要|无需|别))?(?:重新|再次)?"
    r"(?:上传|压缩|创建|新建)(?:新(?:的)?)?(?:素材|项目|任务)?)*"
    r")?"
    r"(?:吧|一下)?[。！？!?，,\s]*$",
    re.IGNORECASE,
)
_VIDEO_EDIT_CONTINUATION_EN_RE = re.compile(
    r"^(?:please\s+)?(?:continue|resume|keep\s+going)"
    r"(?:\s+(?:the|this|that|current|previous|same))?"
    r"(?:\s+(?:video|clip|footage|vlog|editing|render|download|delivery|project|task))*"
    r"[\s.!?,]*$",
    re.IGNORECASE,
)
_VIDEO_EDIT_SLASH_TOKEN_RE = re.compile(
    r"(?<!\S)/video-edit-workflow-mini(?!\S)",
    re.IGNORECASE,
)
_GATEWAY_MODEL_SWITCH_NOTE_RE = re.compile(
    r"^\s*\[Note: the model has changed and is now "
    r"[^\]\r\n]{1,128}\. Adjust your self-identification accordingly\.\]\s*",
    re.IGNORECASE,
)
_VIDEO_EDIT_DIRECT_CN_RE = re.compile(
    r"^(?:请|请帮我|帮我|我要|我想|想要|给我|麻烦)?\s*"
    r"(?:把\s*)?"
    r"(?:(?:这些|这批|这|当前|刚选的|上面的?)\s*)?"
    r"(?:(?:\d+|几)\s*(?:段|个|条)\s*)?"
    r"(?:视频|素材|片段)?\s*"
    r"(?:剪辑|剪片|混剪|剪成|做成)\s*"
    r"(?:(?:这些|这批|这|当前|刚选的|上面的?)\s*)?"
    r"(?:(?:\d+|几)\s*(?:段|个|条)\s*)?"
    r"(?:视频|素材|片段)?\s*"
    r"(?:成|为)?\s*"
    r"(?:一?(?:个|条|支)\s*)?"
    r"(?:(?:\d+\s*(?:秒|分钟)|竖屏|横屏|vlog|短视频|视频|成片|"
    r"高光(?:片段)?|集锦|日常(?:\s*vlog)?|旅行(?:\s*vlog)?|"
    r"自由混剪|电影感|轻松(?:氛围)?)"
    r"(?:[\s、，,]+(?:\d+\s*(?:秒|分钟)|竖屏|横屏|vlog|短视频|"
    r"视频|成片|高光(?:片段)?|集锦|日常(?:\s*vlog)?|"
    r"旅行(?:\s*vlog)?|自由混剪|电影感|轻松(?:氛围)?)){0,5})?"
    r"(?:吧|一下)?[。！？!?，,\s]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _PathSnapshot:
    relative_path: str
    fingerprint: tuple[int, ...]


@dataclass(frozen=True)
class _TrustedDirectSkillSnapshot:
    relative_path: str
    components: tuple[_PathSnapshot, ...]
    raw_sha256: str


@dataclass(frozen=True)
class _TrustedPresetsSnapshot:
    generation: str
    configured_root: str
    configured_root_fingerprint: tuple[int, ...]
    resolved_root: str
    resolved_root_fingerprint: tuple[int, ...]
    skills_root_fingerprint: tuple[int, ...]
    integrity_components: tuple[_PathSnapshot, ...]
    integrity_sha256: str
    integrity_signature_components: tuple[_PathSnapshot, ...]
    integrity_signature_sha256: str
    integrity_signature_key_id: str
    video_edit_script_digests: tuple[tuple[str, str], ...]
    skills: tuple[_TrustedDirectSkillSnapshot, ...]


@dataclass(frozen=True)
class _TrustedSkillReadEvidence:
    generation: str
    relative_path: str
    root_fingerprint: tuple[int, ...]
    file_fingerprint: tuple[int, ...]
    raw_sha256: str


@dataclass(frozen=True)
class _PendingSkillAttestation:
    expires_at: float
    result_sha256: str
    turn_identity: _TurnIdentity
    generation: str
    relative_path: str
    root_fingerprint: tuple[int, ...]
    file_fingerprint: tuple[int, ...]
    raw_sha256: str


@dataclass(frozen=True)
class _SkillDirectTaskContext:
    task_sha256: str
    turn_identity: _TurnIdentity | None
    video_edit_applicable: bool
    video_edit_explicit: bool = False
    camera_applicable: bool = False
    camera_explicit: bool = False
    camera_inventory_only: bool = False
    printer3d_applicable: bool = False
    printer3d_explicit: bool = False


@dataclass(frozen=True)
class _TrustedExecutionReceipt:
    agent_id: str = field(repr=False)
    action_token: str = field(repr=False)
    business_execution_token: str = field(repr=False)
    turn_id: str
    session_id: str


@dataclass(frozen=True)
class _SkillDirectScope:
    relative_path: str
    task_sha256: str
    turn_identity: _TurnIdentity
    allowed_tools: frozenset[str]
    execution_receipt: _TrustedExecutionReceipt | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    memory_payload_sha256: frozenset[str] = frozenset()
    # Camera actions are intentionally two-step: a request-bound ``list``
    # establishes the opaque IDs visible to this exact turn, then one of those
    # IDs may be used by snap/clip/doctor. Keeping the response-size-bounded
    # set in the capability prevents the model from guessing a display-name
    # suffix such as ``2`` and accidentally targeting a different device.
    camera_ids: frozenset[str] = frozenset()
    # Only the exact attachment returned by a successful request-bound snap
    # may be inspected. This keeps the camera capability from becoming a
    # general local-file vision grant.
    camera_attachment_paths: frozenset[str] = frozenset()
    # A fleet/status turn may only execute ``list``. This prevents a broad
    # health check from silently capturing media the user did not request.
    camera_inventory_only: bool = False
    command_format_retries: int = 1
    policy_violation_retries: int = _VIDEO_EDIT_POLICY_VIOLATION_RETRIES
    policy_exhausted: bool = False


@dataclass(frozen=True)
class _SkillDirectOperation:
    scope: _SkillDirectScope
    function_name: str
    may_authorize_memory: bool = False
    authorized_args_sha256: str = field(
        default="",
        repr=False,
        compare=False,
    )
    execution_claimed: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class _VideoEditResumeGrant:
    expires_at: float
    source_turn_id: str


@dataclass(frozen=True)
class _CameraResumeGrant:
    expires_at: float
    source_turn_id: str


_TRUSTED_PRESETS_SNAPSHOT: _TrustedPresetsSnapshot | None = None
_PENDING_ATTESTATIONS: OrderedDict[str, _PendingSkillAttestation] = OrderedDict()
_VideoEditResumeKey = tuple[str, str]
_VIDEO_EDIT_RESUME_SESSIONS: OrderedDict[
    _VideoEditResumeKey,
    _VideoEditResumeGrant | float,
] = OrderedDict()
_CameraResumeKey = tuple[str, str]
_CAMERA_RESUME_SESSIONS: OrderedDict[
    _CameraResumeKey,
    _CameraResumeGrant,
] = OrderedDict()
_ATTESTATION_LOCK = threading.Lock()
_SKILL_DIRECT_LOCK = threading.Lock()
_TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT: ContextVar[
    _TrustedExecutionReceipt | None
] = ContextVar("_TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT", default=None)


def _current_skill_direct_turn_identity() -> _TurnIdentity | None:
    """Read the server-minted identity for this exact request context."""
    try:
        from gateway.session_context import current_turn_identity

        return current_turn_identity()
    except Exception:
        return None


def _current_skill_direct_session_id() -> str:
    """Read the server-minted session key without consulting process env."""
    try:
        from gateway.session_context import get_session_env

        return str(get_session_env("HERMES_SESSION_KEY") or "").strip()
    except Exception:
        return ""


def _current_skill_direct_resume_key() -> _VideoEditResumeKey | None:
    """Bind resumable video-edit intent to the active profile and session."""
    session_id = _current_skill_direct_session_id()
    if not session_id:
        return None
    try:
        from hermes_constants import get_hermes_home

        profile_home = os.path.normcase(
            os.path.abspath(os.path.expanduser(str(get_hermes_home())))
        )
    except Exception:
        return None
    if not profile_home:
        return None
    return profile_home, session_id


def _camera_continuation_intent(normalized: str) -> bool:
    if not normalized or len(normalized) > 160:
        return False
    return bool(_CAMERA_CONTINUATION_INTENT_RE.fullmatch(normalized))


def _camera_resume_sessions_locked(
    *,
    now: float,
) -> OrderedDict[_CameraResumeKey, _CameraResumeGrant]:
    expired = [
        resume_key
        for resume_key, grant in _CAMERA_RESUME_SESSIONS.items()
        if grant.expires_at <= now
    ]
    for resume_key in expired:
        _CAMERA_RESUME_SESSIONS.pop(resume_key, None)
    while len(_CAMERA_RESUME_SESSIONS) > _CAMERA_RESUME_MAX_SESSIONS:
        _CAMERA_RESUME_SESSIONS.popitem(last=False)
    return _CAMERA_RESUME_SESSIONS


def _remember_camera_resume_locked(
    *,
    turn_identity: _TurnIdentity,
    now: float,
) -> None:
    resume_key = _current_skill_direct_resume_key()
    if resume_key is None:
        return
    sessions = _camera_resume_sessions_locked(now=now)
    sessions[resume_key] = _CameraResumeGrant(
        expires_at=now + _CAMERA_RESUME_TTL_SECONDS,
        source_turn_id=str(turn_identity[0] or "").strip(),
    )
    sessions.move_to_end(resume_key)
    while len(sessions) > _CAMERA_RESUME_MAX_SESSIONS:
        sessions.popitem(last=False)


def _video_edit_continuation_intent(normalized: str) -> bool:
    if not normalized or len(normalized) > 160:
        return False
    return bool(
        _VIDEO_EDIT_CONTINUATION_CN_RE.fullmatch(normalized)
        or _VIDEO_EDIT_CONTINUATION_EN_RE.fullmatch(normalized)
    )


def _video_edit_direct_command_intent(normalized: str) -> bool:
    if not normalized or len(normalized) > 96:
        return False
    return bool(_VIDEO_EDIT_DIRECT_CN_RE.fullmatch(normalized))


def _strip_gateway_model_switch_note(task_text: str) -> str:
    """Remove only the exact gateway-authored model identity preamble."""
    return _GATEWAY_MODEL_SWITCH_NOTE_RE.sub("", task_text, count=1)


def _video_edit_resume_sessions_locked(
    *,
    now: float,
) -> OrderedDict[_VideoEditResumeKey, _VideoEditResumeGrant | float]:
    def _expires_at(value: _VideoEditResumeGrant | float) -> float | None:
        if isinstance(value, _VideoEditResumeGrant):
            return value.expires_at
        if isinstance(value, (int, float)):
            return float(value)
        return None

    expired = [
        resume_key
        for resume_key, grant in _VIDEO_EDIT_RESUME_SESSIONS.items()
        if (_expires_at(grant) is None or _expires_at(grant) <= now)
    ]
    for resume_key in expired:
        _VIDEO_EDIT_RESUME_SESSIONS.pop(resume_key, None)
    while len(_VIDEO_EDIT_RESUME_SESSIONS) > _VIDEO_EDIT_RESUME_MAX_SESSIONS:
        _VIDEO_EDIT_RESUME_SESSIONS.popitem(last=False)
    return _VIDEO_EDIT_RESUME_SESSIONS


def _confirmed_video_edit_plan_resume(
    resume_key: _VideoEditResumeKey | None,
    resume_sessions: Mapping[
        _VideoEditResumeKey,
        _VideoEditResumeGrant | float,
    ],
) -> bool:
    if resume_key is None:
        return False
    grant = resume_sessions.get(resume_key)
    if not isinstance(grant, _VideoEditResumeGrant) or not grant.source_turn_id:
        return False
    try:
        from gateway.session_context import get_session_env

        status = str(get_session_env("HERMES_PLAN_ACK_STATUS") or "").strip()
        source_turn_id = str(
            get_session_env("HERMES_PLAN_ACK_TURN_ID") or ""
        ).strip()
    except Exception:
        return False
    return status == "confirmed" and source_turn_id == grant.source_turn_id


def _capture_trusted_execution_receipt(
    turn_identity: _TurnIdentity,
) -> _TrustedExecutionReceipt | None:
    """Freeze request-bound execution claims before later tool boundaries."""
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        secret_scope = current_secret_scope()
        multiplex_active = is_multiplex_active()
    except Exception:
        secret_scope = None
        multiplex_active = False

    def _profile_value(name: str) -> str:
        value = secret_scope.get(name) if secret_scope is not None else None
        if value is None and not multiplex_active:
            value = os.environ.get(name)
        return str(value or "").strip()

    try:
        from gateway.session_context import (
            business_execution_token,
            get_session_env,
        )

        business_token = business_execution_token()
        session_id = get_session_env("HERMES_SESSION_ID") or get_session_env(
            "HERMES_SESSION_KEY"
        )
    except Exception:
        business_token = ""
        session_id = ""

    receipt = _TrustedExecutionReceipt(
        agent_id=_profile_value("ZET_AGENT_ID"),
        action_token=_profile_value("ZETTLAB_AGENT_ACTION_TOKEN"),
        business_execution_token=str(business_token or "").strip(),
        turn_id=str(turn_identity[0] or "").strip(),
        session_id=str(session_id or "").strip(),
    )
    present = {
        "agent_id": bool(receipt.agent_id),
        "action_token": bool(receipt.action_token),
        "business_execution_token": bool(receipt.business_execution_token),
        "turn_id": bool(receipt.turn_id),
        "session_id": bool(receipt.session_id),
    }
    if not all(present.values()):
        logger.warning(
            "zet_agent: trusted video execution receipt incomplete: %s",
            present,
        )
        return None
    return receipt


def trusted_video_edit_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for the dedicated worker."""
    receipt = _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.get()
    if receipt is None:
        return {}
    return {
        "ZET_AGENT_ID": receipt.agent_id,
        "ZETTLAB_AGENT_ACTION_TOKEN": receipt.action_token,
        "ZETTLAB_BUSINESS_EXECUTION_TOKEN": receipt.business_execution_token,
        "HERMES_TURN_ID": receipt.turn_id,
        "HERMES_SESSION_KEY": receipt.session_id,
    }


def trusted_camera_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for the camsnap helper."""
    return trusted_video_edit_runtime_receipt()


def trusted_printer3d_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for signed printer helpers."""
    return trusted_video_edit_runtime_receipt()


def trusted_video_edit_manifest_digests() -> Mapping[str, str]:
    """Return signed helper digests captured before model-authored terminal use."""
    snapshot = _TRUSTED_PRESETS_SNAPSHOT
    if snapshot is None:
        return {}
    return dict(snapshot.video_edit_script_digests)


def _stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _component_is_safe(
    value: os.stat_result,
    *,
    allow_symlink: bool = False,
    allow_current_user: bool = False,
) -> bool:
    """Reject components writable by the model-controlled terminal identity."""
    is_symlink = stat.S_ISLNK(value.st_mode)
    if is_symlink != allow_symlink:
        return False
    if not is_symlink and stat.S_IMODE(value.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        return False

    if not hasattr(os, "geteuid"):
        return True
    euid = os.geteuid()
    if euid == 0:
        # The packaged device tree is root-owned. Root can still chmod/rewrite
        # it, so the immutable startup fingerprint below is the actual temporal
        # boundary; accepting any non-root owner would add another writer.
        return value.st_uid == 0
    if value.st_uid == euid:
        return allow_current_user
    return value.st_uid == 0


def _lstat_snapshot(
    path: Path,
    *,
    relative_path: str,
    expected_kind: str,
    allow_symlink: bool = False,
    allow_current_user: bool = False,
) -> _PathSnapshot:
    value = os.stat(path, follow_symlinks=False)
    if expected_kind == "dir" and not stat.S_ISDIR(value.st_mode):
        raise PermissionError(f"trusted path is not a directory: {relative_path}")
    if expected_kind == "file" and not stat.S_ISREG(value.st_mode):
        raise PermissionError(f"trusted path is not a regular file: {relative_path}")
    if expected_kind == "root" and not (
        stat.S_ISDIR(value.st_mode) or stat.S_ISLNK(value.st_mode)
    ):
        raise PermissionError("trusted presets root is not a directory or symlink")
    if not _component_is_safe(
        value,
        allow_symlink=allow_symlink and stat.S_ISLNK(value.st_mode),
        allow_current_user=allow_current_user,
    ):
        raise PermissionError(f"trusted path has unsafe ownership or mode: {relative_path}")
    return _PathSnapshot(relative_path, _stat_fingerprint(value))


def _read_stable_file(path: Path, *, max_bytes: int) -> tuple[bytes, tuple[int, ...]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    path_before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(path_before.st_mode):
        raise PermissionError("trusted skill source is not a regular file")

    fd = os.open(path, flags)
    try:
        opened_before = os.fstat(fd)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("trusted skill source exceeds size limit")
            chunks.append(chunk)
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)

    path_after = os.stat(path, follow_symlinks=False)
    fingerprints = {
        _stat_fingerprint(path_before),
        _stat_fingerprint(opened_before),
        _stat_fingerprint(opened_after),
        _stat_fingerprint(path_after),
    }
    if len(fingerprints) != 1 or not stat.S_ISREG(opened_after.st_mode):
        raise PermissionError("trusted skill source changed while being read")
    return b"".join(chunks), _stat_fingerprint(opened_after)


def _skill_component_snapshots(
    resolved_root: Path,
    relative_path: str,
    *,
    allow_current_user: bool,
) -> tuple[_PathSnapshot, ...]:
    relative = Path(relative_path)
    is_skill = relative.name == "SKILL.md"
    is_integrity_metadata = relative_path in {
        _TRUSTED_INTEGRITY_MANIFEST_PATH,
        _TRUSTED_INTEGRITY_SIGNATURE_PATH,
    }
    if (
        not relative.parts
        or relative.parts[0] != "skills"
        or not (is_skill or is_integrity_metadata)
    ):
        raise PermissionError("trusted response-mode skill has an invalid layout")

    snapshots: list[_PathSnapshot] = []
    current = resolved_root
    for index, part in enumerate(relative.parts):
        current = current / part
        snapshots.append(
            _lstat_snapshot(
                current,
                relative_path=str(Path(*relative.parts[: index + 1])),
                expected_kind="file" if index == len(relative.parts) - 1 else "dir",
                allow_current_user=allow_current_user,
            )
        )
    return tuple(snapshots)


def _parse_integrity_manifest(raw: bytes) -> Mapping[str, str]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid presets integrity manifest") from exc
    if not isinstance(value, dict) or value.get("schema") != _TRUSTED_INTEGRITY_SCHEMA:
        raise ValueError("unsupported presets integrity manifest schema")
    files = value.get("files")
    if not isinstance(files, dict):
        raise ValueError("presets integrity manifest files must be an object")
    for path, digest in files.items():
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("invalid presets integrity manifest entry")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid presets integrity digest")
    return files


def _verify_integrity_manifest_signature(
    manifest_raw: bytes,
    signature_raw: bytes,
    *,
    resolved_root: Path,
) -> str:
    try:
        value = json.loads(signature_raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid presets integrity signature") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != _TRUSTED_INTEGRITY_SIGNATURE_SCHEMA
        or set(value) != {"schema", "key_id", "signature"}
    ):
        raise ValueError("unsupported presets integrity signature schema")
    key_id = value.get("key_id")
    encoded_signature = value.get("signature")
    if not isinstance(key_id, str) or not isinstance(encoded_signature, str):
        raise ValueError("invalid presets integrity signature fields")
    encoded_public_key = _resolve_integrity_public_key(
        key_id,
        resolved_root=resolved_root,
    )
    if encoded_public_key is None:
        raise ValueError("unknown presets integrity signing key")
    try:
        public_key = base64.b64decode(encoded_public_key, validate=True)
        signature = base64.b64decode(encoded_signature, validate=True)
    except ValueError as exc:
        raise ValueError("invalid presets integrity signature encoding") from exc
    if len(public_key) != 32 or len(signature) != 64:
        raise ValueError("invalid presets integrity signature size")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            manifest_raw,
        )
    except (InvalidSignature, ValueError) as exc:
        raise PermissionError("presets integrity signature verification failed") from exc
    return key_id


def _resolve_integrity_public_key(
    key_id: str,
    *,
    resolved_root: Path,
) -> str | None:
    encoded_public_key = _TRUSTED_PRESETS_PUBLIC_KEYS_B64.get(key_id)
    if encoded_public_key is not None:
        return encoded_public_key

    if not resolved_root.name.startswith(
        _TRUSTED_PRESETS_DEV_DIRECTORY_PREFIX
    ):
        return None
    if not key_id.startswith(_TRUSTED_PRESETS_DEV_KEY_ID_PREFIX):
        return None

    configured_key_id = os.getenv(_TRUSTED_PRESETS_DEV_KEY_ID_ENV, "").strip()
    configured_public_key = os.getenv(
        _TRUSTED_PRESETS_DEV_PUBLIC_KEY_ENV,
        "",
    ).strip()
    if not configured_key_id and not configured_public_key:
        return None
    if not configured_key_id or not configured_public_key:
        raise ValueError("incomplete presets development signing key")
    if configured_key_id != key_id:
        return None
    return configured_public_key


def _capture_trusted_presets_snapshot(
    *,
    allow_current_user: bool = True,
) -> _TrustedPresetsSnapshot | None:
    """Capture the official video-edit skill before model-authored terminal use."""
    raw_root = os.getenv("ZETTLAB_PRESETS_DIR", "").strip()
    if not raw_root:
        return None

    try:
        configured_root = Path(
            os.path.expandvars(os.path.expanduser(raw_root))
        ).absolute()
        configured_value = os.stat(configured_root, follow_symlinks=False)
        configured_snapshot = _lstat_snapshot(
            configured_root,
            relative_path="<configured-root>",
            expected_kind="root",
            allow_symlink=stat.S_ISLNK(configured_value.st_mode),
            allow_current_user=allow_current_user,
        )
        resolved_root = configured_root.resolve(strict=True)
        resolved_snapshot = _lstat_snapshot(
            resolved_root,
            relative_path=".",
            expected_kind="dir",
            allow_current_user=allow_current_user,
        )
        skills_root = resolved_root / "skills"
        skills_snapshot = _lstat_snapshot(
            skills_root,
            relative_path="skills",
            expected_kind="dir",
            allow_current_user=allow_current_user,
        )
        integrity_path = resolved_root / _TRUSTED_INTEGRITY_MANIFEST_PATH
        integrity_before = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_MANIFEST_PATH,
            allow_current_user=allow_current_user,
        )
        integrity_raw, integrity_fingerprint = _read_stable_file(
            integrity_path,
            max_bytes=_TRUSTED_INTEGRITY_MANIFEST_LIMIT_BYTES,
        )
        integrity_after = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_MANIFEST_PATH,
            allow_current_user=allow_current_user,
        )
        if (
            integrity_before != integrity_after
            or integrity_before[-1].fingerprint != integrity_fingerprint
        ):
            raise PermissionError(
                "presets integrity manifest changed during startup"
            )
        integrity_signature_path = (
            resolved_root / _TRUSTED_INTEGRITY_SIGNATURE_PATH
        )
        integrity_signature_before = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_SIGNATURE_PATH,
            allow_current_user=allow_current_user,
        )
        integrity_signature_raw, integrity_signature_fingerprint = (
            _read_stable_file(
                integrity_signature_path,
                max_bytes=_TRUSTED_INTEGRITY_SIGNATURE_LIMIT_BYTES,
            )
        )
        integrity_signature_after = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_SIGNATURE_PATH,
            allow_current_user=allow_current_user,
        )
        if (
            integrity_signature_before != integrity_signature_after
            or integrity_signature_before[-1].fingerprint
            != integrity_signature_fingerprint
        ):
            raise PermissionError(
                "presets integrity signature changed during startup"
            )
        integrity_signature_key_id = _verify_integrity_manifest_signature(
            integrity_raw,
            integrity_signature_raw,
            resolved_root=resolved_root,
        )
        expected_hashes = _parse_integrity_manifest(integrity_raw)
        expected_video_edit_sha256 = expected_hashes.get(
            _VIDEO_EDIT_SKILL_PATH
        )
        if not expected_video_edit_sha256:
            raise ValueError(
                "official video-edit skill is absent from the release manifest"
            )
        expected_trusted_skill_hashes = {
            _VIDEO_EDIT_SKILL_PATH: expected_video_edit_sha256,
        }
        expected_camera_sha256 = expected_hashes.get(_CAMERA_SKILL_PATH)
        if expected_camera_sha256:
            expected_trusted_skill_hashes[_CAMERA_SKILL_PATH] = (
                expected_camera_sha256
            )
        for printer_skill_path in _PRINTER3D_SKILL_PATHS:
            expected_printer_sha256 = expected_hashes.get(printer_skill_path)
            if expected_printer_sha256:
                expected_trusted_skill_hashes[printer_skill_path] = expected_printer_sha256
        video_edit_scripts_root = Path(_VIDEO_EDIT_SKILL_PATH).parent / "scripts"
        video_edit_script_digests = tuple(sorted(
            (relative_path, digest)
            for relative_path, digest in expected_hashes.items()
            if Path(relative_path).parent == video_edit_scripts_root
            and Path(relative_path).suffix == ".py"
        ))

        trusted_skills: list[_TrustedDirectSkillSnapshot] = []
        scanned = 0
        from agent.skill_utils import is_excluded_skill_path

        for skill_md in skills_root.rglob("SKILL.md"):
            scanned += 1
            if scanned > _TRUSTED_SKILL_SCAN_LIMIT:
                raise ValueError("trusted skill scan limit exceeded")
            if is_excluded_skill_path(skill_md):
                continue
            relative_path = str(skill_md.relative_to(resolved_root))
            expected_skill_sha256 = expected_trusted_skill_hashes.get(relative_path)
            if not expected_skill_sha256:
                continue
            try:
                before = _skill_component_snapshots(
                    resolved_root,
                    relative_path,
                    allow_current_user=allow_current_user,
                )
                raw_source, file_fingerprint = _read_stable_file(
                    skill_md,
                    max_bytes=_TRUSTED_SKILL_SOURCE_LIMIT_BYTES,
                )
                after = _skill_component_snapshots(
                    resolved_root,
                    relative_path,
                    allow_current_user=allow_current_user,
                )
                if before != after or before[-1].fingerprint != file_fingerprint:
                    raise PermissionError("trusted skill path changed while being captured")
                raw_source.decode("utf-8")
                raw_sha256 = hashlib.sha256(raw_source).hexdigest()
                if raw_sha256 != expected_skill_sha256:
                    raise PermissionError(
                        "official trusted skill does not match release manifest"
                    )
            except (OSError, UnicodeError, ValueError, PermissionError):
                logger.warning(
                    "ignored unsafe official trusted skill during startup: %s",
                    relative_path,
                )
                continue
            trusted_skills.append(
                _TrustedDirectSkillSnapshot(
                    relative_path=relative_path,
                    components=before,
                    raw_sha256=raw_sha256,
                )
            )

        # Reject a root/current/skills replacement racing the startup scan.
        if configured_snapshot != _lstat_snapshot(
            configured_root,
            relative_path="<configured-root>",
            expected_kind="root",
            allow_symlink=stat.S_ISLNK(configured_value.st_mode),
            allow_current_user=allow_current_user,
        ):
            raise PermissionError("configured presets root changed during startup")
        if resolved_snapshot != _lstat_snapshot(
            resolved_root,
            relative_path=".",
            expected_kind="dir",
            allow_current_user=allow_current_user,
        ):
            raise PermissionError("resolved presets root changed during startup")
        if skills_snapshot != _lstat_snapshot(
            skills_root,
            relative_path="skills",
            expected_kind="dir",
            allow_current_user=allow_current_user,
        ):
            raise PermissionError("presets skills root changed during startup")

        return _TrustedPresetsSnapshot(
            generation=secrets.token_hex(16),
            configured_root=str(configured_root),
            configured_root_fingerprint=configured_snapshot.fingerprint,
            resolved_root=str(resolved_root),
            resolved_root_fingerprint=resolved_snapshot.fingerprint,
            skills_root_fingerprint=skills_snapshot.fingerprint,
            integrity_components=integrity_before,
            integrity_sha256=hashlib.sha256(integrity_raw).hexdigest(),
            integrity_signature_components=integrity_signature_before,
            integrity_signature_sha256=hashlib.sha256(
                integrity_signature_raw
            ).hexdigest(),
            integrity_signature_key_id=integrity_signature_key_id,
            video_edit_script_digests=video_edit_script_digests,
            skills=tuple(trusted_skills),
        )
    except (OSError, ValueError, PermissionError) as exc:
        logger.warning("official video-edit skill trust unavailable: %s", exc)
        return None


def _find_snapshot_skill(
    snapshot: _TrustedPresetsSnapshot,
    relative_path: str,
) -> _TrustedDirectSkillSnapshot | None:
    return next(
        (skill for skill in snapshot.skills if skill.relative_path == relative_path),
        None,
    )


def _current_root_matches(snapshot: _TrustedPresetsSnapshot) -> bool:
    try:
        configured_root = Path(snapshot.configured_root)
        if str(
            Path(
                os.path.expandvars(
                    os.path.expanduser(os.getenv("ZETTLAB_PRESETS_DIR", "").strip())
                )
            ).absolute()
        ) != snapshot.configured_root:
            return False
        configured = _stat_fingerprint(os.stat(configured_root, follow_symlinks=False))
        if configured != snapshot.configured_root_fingerprint:
            return False
        if str(configured_root.resolve(strict=True)) != snapshot.resolved_root:
            return False
        resolved_root = Path(snapshot.resolved_root)
        if _stat_fingerprint(os.stat(resolved_root, follow_symlinks=False)) != snapshot.resolved_root_fingerprint:
            return False
        if (
            _stat_fingerprint(
                os.stat(resolved_root / "skills", follow_symlinks=False)
            )
            != snapshot.skills_root_fingerprint
        ):
            return False
        integrity_components = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_MANIFEST_PATH,
            allow_current_user=True,
        )
        if integrity_components != snapshot.integrity_components:
            return False
        integrity_raw, integrity_fingerprint = _read_stable_file(
            resolved_root / _TRUSTED_INTEGRITY_MANIFEST_PATH,
            max_bytes=_TRUSTED_INTEGRITY_MANIFEST_LIMIT_BYTES,
        )
        if (
            integrity_fingerprint
            != snapshot.integrity_components[-1].fingerprint
            or hashlib.sha256(integrity_raw).hexdigest()
            != snapshot.integrity_sha256
        ):
            return False
        integrity_signature_components = _skill_component_snapshots(
            resolved_root,
            _TRUSTED_INTEGRITY_SIGNATURE_PATH,
            allow_current_user=True,
        )
        if (
            integrity_signature_components
            != snapshot.integrity_signature_components
        ):
            return False
        integrity_signature_raw, integrity_signature_fingerprint = (
            _read_stable_file(
                resolved_root / _TRUSTED_INTEGRITY_SIGNATURE_PATH,
                max_bytes=_TRUSTED_INTEGRITY_SIGNATURE_LIMIT_BYTES,
            )
        )
        return (
            integrity_signature_fingerprint
            == snapshot.integrity_signature_components[-1].fingerprint
            and hashlib.sha256(integrity_signature_raw).hexdigest()
            == snapshot.integrity_signature_sha256
        )
    except (OSError, ValueError, PermissionError):
        return False


def read_skill_source_with_trusted_execution_evidence(
    skill_md: Path,
) -> tuple[str | None, _TrustedSkillReadEvidence | None]:
    """Stable-read an eligible official SKILL.md and bind bytes to startup trust.

    ``(None, None)`` means the ordinary skill reader should continue, but no
    trusted-execution attestation may be issued for that result.
    """
    snapshot = _TRUSTED_PRESETS_SNAPSHOT
    if snapshot is None:
        return None, None

    try:
        candidate = Path(skill_md).absolute()
        relative_path: str | None = None
        for root in (
            Path(snapshot.configured_root) / "skills",
            Path(snapshot.resolved_root) / "skills",
        ):
            try:
                relative_path = str(Path("skills") / candidate.relative_to(root))
                break
            except ValueError:
                continue
        if relative_path is None:
            return None, None

        trusted_skill = _find_snapshot_skill(snapshot, relative_path)
        if trusted_skill is None:
            return None, None
        trusted_path = Path(snapshot.resolved_root) / relative_path
        if candidate.resolve(strict=True) != trusted_path:
            return None, None
        if not _current_root_matches(snapshot):
            raise PermissionError("trusted presets root no longer matches startup")

        before = _skill_component_snapshots(
            Path(snapshot.resolved_root),
            relative_path,
            allow_current_user=True,
        )
        if before != trusted_skill.components:
            raise PermissionError("trusted skill path no longer matches startup")
        raw_source, file_fingerprint = _read_stable_file(
            trusted_path,
            max_bytes=_TRUSTED_SKILL_SOURCE_LIMIT_BYTES,
        )
        after = _skill_component_snapshots(
            Path(snapshot.resolved_root),
            relative_path,
            allow_current_user=True,
        )
        raw_sha256 = hashlib.sha256(raw_source).hexdigest()
        if (
            before != after
            or after != trusted_skill.components
            or file_fingerprint != trusted_skill.components[-1].fingerprint
            or raw_sha256 != trusted_skill.raw_sha256
            or not _current_root_matches(snapshot)
        ):
            raise PermissionError("trusted skill bytes or path changed during skill_view")
        content = raw_source.decode("utf-8")
        return content, _TrustedSkillReadEvidence(
            generation=snapshot.generation,
            relative_path=relative_path,
            root_fingerprint=snapshot.resolved_root_fingerprint,
            file_fingerprint=file_fingerprint,
            raw_sha256=raw_sha256,
        )
    except (OSError, UnicodeError, ValueError, PermissionError) as exc:
        logger.warning("skill_view trusted-execution attestation rejected: %s", exc)
        return None, None


def _prune_attestations_locked(now: float) -> None:
    expired = [
        token
        for token, pending in _PENDING_ATTESTATIONS.items()
        if pending.expires_at <= now
    ]
    for token in expired:
        _PENDING_ATTESTATIONS.pop(token, None)


def serialize_skill_view_result(
    result: Mapping[str, Any],
    evidence: _TrustedSkillReadEvidence | None,
) -> str:
    """Serialize a skill result and attach a bounded one-shot internal proof."""
    plain_result = json.dumps(dict(result), ensure_ascii=False)
    if evidence is None:
        return plain_result
    snapshot = _TRUSTED_PRESETS_SNAPSHOT
    if snapshot is None or evidence.generation != snapshot.generation:
        return plain_result
    trusted_skill = _find_snapshot_skill(snapshot, evidence.relative_path)
    if trusted_skill is None or (
        evidence.root_fingerprint != snapshot.resolved_root_fingerprint
        or evidence.file_fingerprint != trusted_skill.components[-1].fingerprint
        or evidence.raw_sha256 != trusted_skill.raw_sha256
    ):
        return plain_result

    turn_identity = _current_skill_direct_turn_identity()
    if turn_identity is None:
        logger.warning("trusted-skill attestation missing turn identity; failing closed")
        return plain_result

    now = time.monotonic()
    with _ATTESTATION_LOCK:
        _prune_attestations_locked(now)
        if len(_PENDING_ATTESTATIONS) >= _ATTESTATION_MAX_ENTRIES:
            logger.warning("trusted-skill attestation capacity reached; failing closed")
            return plain_result
        token = secrets.token_urlsafe(32)
        while token in _PENDING_ATTESTATIONS:
            token = secrets.token_urlsafe(32)
        attested_result = dict(result)
        attested_result[_ATTESTATION_FIELD] = token
        serialized = json.dumps(attested_result, ensure_ascii=False)
        _PENDING_ATTESTATIONS[token] = _PendingSkillAttestation(
            expires_at=now + _ATTESTATION_TTL_SECONDS,
            result_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            turn_identity=turn_identity,
            generation=evidence.generation,
            relative_path=evidence.relative_path,
            root_fingerprint=evidence.root_fingerprint,
            file_fingerprint=evidence.file_fingerprint,
            raw_sha256=evidence.raw_sha256,
        )
        return serialized


def _consume_skill_attestation(
    token: str,
    serialized_result: str,
) -> _PendingSkillAttestation | None:
    now = time.monotonic()
    with _ATTESTATION_LOCK:
        _prune_attestations_locked(now)
        pending = _PENDING_ATTESTATIONS.pop(token, None)
    if pending is None or pending.expires_at <= now:
        return None
    if pending.result_sha256 != hashlib.sha256(serialized_result.encode("utf-8")).hexdigest():
        return None
    snapshot = _TRUSTED_PRESETS_SNAPSHOT
    if snapshot is None or pending.generation != snapshot.generation:
        return None
    trusted_skill = _find_snapshot_skill(snapshot, pending.relative_path)
    if trusted_skill is None or (
        pending.root_fingerprint != snapshot.resolved_root_fingerprint
        or pending.file_fingerprint != trusted_skill.components[-1].fingerprint
        or pending.raw_sha256 != trusted_skill.raw_sha256
    ):
        return None
    return pending


def _task_text_and_video_asset(user_message: Any) -> tuple[str, bool]:
    """Return bounded task text plus deterministic video-asset evidence."""
    parts: list[str] = []
    seen: set[int] = set()
    total = 0
    has_video_asset = False

    def _visit(value: Any, *, key: str = "", depth: int = 0) -> None:
        nonlocal total, has_video_asset
        if depth > 5 or total >= 64 * 1024 or value is None:
            return
        if isinstance(value, str):
            remaining = 64 * 1024 - total
            text = value[:remaining]
            total += len(text)
            parts.append(text)
            lowered = text.lower().split("?", 1)[0].split("#", 1)[0]
            key_lower = key.lower()
            if key_lower in {"mime_type", "mimetype", "content_type"}:
                has_video_asset = has_video_asset or lowered.startswith("video/")
            if key_lower == "type":
                has_video_asset = has_video_asset or lowered in {"video", "video_url"}
            if key_lower in {
                "file",
                "file_name",
                "filename",
                "name",
                "path",
                "url",
            }:
                has_video_asset = has_video_asset or (
                    lowered.endswith(_VIDEO_FILE_SUFFIXES)
                    or lowered.startswith("data:video/")
                )
            if "[file:" in lowered and any(suffix in lowered for suffix in _VIDEO_FILE_SUFFIXES):
                has_video_asset = True
            if "[video:" in lowered or "[视频:" in lowered:
                has_video_asset = True
            return
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            for child_key, child in value.items():
                _visit(child, key=str(child_key), depth=depth + 1)
            return
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in seen:
                return
            seen.add(identity)
            for child in value:
                _visit(child, key=key, depth=depth + 1)

    _visit(user_message)
    return "\n".join(parts), has_video_asset


def _skill_direct_task_context(
    agent: Any,
    user_message: Any,
    *,
    explicit_skill_slug: str = "",
    tool_execution_allowed: bool = True,
) -> _SkillDirectTaskContext:
    task_text, has_video_asset = _task_text_and_video_asset(user_message)
    task_text = _strip_gateway_model_switch_note(task_text)
    normalized = " ".join(task_text.lower().split())
    if not tool_execution_allowed:
        task_binding = f"tools:none\n{normalized}"
        return _SkillDirectTaskContext(
            task_sha256=hashlib.sha256(task_binding.encode("utf-8")).hexdigest(),
            turn_identity=_current_skill_direct_turn_identity(),
            video_edit_applicable=False,
            video_edit_explicit=False,
        )
    normalized_skill_slug = (
        explicit_skill_slug.strip().lstrip("/").lower()
        if isinstance(explicit_skill_slug, str)
        else ""
    )
    explicit_transport_selection = (
        normalized_skill_slug == "video-edit-workflow-mini"
    )
    camera_transport_selection = normalized_skill_slug == "camsnap"
    printer3d_transport_selection = normalized_skill_slug in {"printer3d", "printer3d-control"}
    # A slash token inside user-authored text is display/content, not a trusted
    # transport selection. Ignore the token itself for semantic intent while
    # preserving the remaining natural-language request.
    intent_normalized = " ".join(
        _VIDEO_EDIT_SLASH_TOKEN_RE.sub(" ", normalized).split()
    )
    has_edit_intent = bool(
        _VIDEO_EDIT_CN_RE.search(intent_normalized)
        or _VIDEO_EDIT_EN_RE.search(intent_normalized)
    )
    explicit = (
        explicit_transport_selection
        or (has_video_asset and has_edit_intent)
        or _video_edit_direct_command_intent(intent_normalized)
    )
    now = time.monotonic()
    resume_key = _current_skill_direct_resume_key()
    resume_sessions = _video_edit_resume_sessions_locked(now=now)
    continuation_intent = _video_edit_continuation_intent(intent_normalized)
    confirmed_plan_resume = _confirmed_video_edit_plan_resume(
        resume_key,
        resume_sessions,
    )
    resumed = bool(
        not explicit
        and (
            confirmed_plan_resume
            or (
                continuation_intent
                and (
                    has_edit_intent
                    or (resume_key is not None and resume_key in resume_sessions)
                )
            )
        )
    )
    if resumed and resume_key is not None and resume_key in resume_sessions:
        resume_sessions.move_to_end(resume_key)
    camera_resume_sessions = _camera_resume_sessions_locked(now=now)
    camera_resume_key = _current_skill_direct_resume_key()
    camera_continuation = _camera_continuation_intent(intent_normalized)
    camera_resumed = bool(
        not camera_transport_selection
        and camera_continuation
        and camera_resume_key is not None
        and camera_resume_key in camera_resume_sessions
    )
    if camera_resumed and camera_resume_key is not None:
        camera_resume_sessions.move_to_end(camera_resume_key)
    task_binding = (
        f"skill:{normalized_skill_slug}\n{normalized}"
        if explicit_transport_selection or camera_transport_selection or printer3d_transport_selection
        else normalized
    )
    hardware_inventory = bool(
        _HARDWARE_INVENTORY_INTENT_RE.search(intent_normalized)
    )
    return _SkillDirectTaskContext(
        task_sha256=hashlib.sha256(task_binding.encode("utf-8")).hexdigest(),
        turn_identity=_current_skill_direct_turn_identity(),
        video_edit_applicable=explicit or resumed,
        video_edit_explicit=explicit,
        camera_applicable=(
            camera_transport_selection
            or bool(_CAMERA_INTENT_RE.search(intent_normalized))
            or bool(_CAMERA_INVENTORY_INTENT_RE.search(intent_normalized))
            or hardware_inventory
            or bool(
                _CAMERA_DIRECT_SNAPSHOT_INTENT_RE.fullmatch(intent_normalized)
            )
            or camera_resumed
        ),
        camera_explicit=camera_transport_selection,
        camera_inventory_only=bool(
            not camera_transport_selection
            and (
                _CAMERA_INVENTORY_INTENT_RE.search(intent_normalized)
                or hardware_inventory
            )
            and not _CAMERA_INTENT_RE.search(intent_normalized)
        ),
        printer3d_applicable=(
            printer3d_transport_selection
            or bool(_PRINTER3D_INTENT_RE.search(normalized))
            or hardware_inventory
        ),
        printer3d_explicit=printer3d_transport_selection,
    )


def trusted_skill_allowed_tool_names(agent: Any) -> frozenset[str]:
    """Return the exact tool allowlist for an active trusted-skill scope."""
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        scope = getattr(agent, "_zet_agent_skill_direct_scope", None)
        if not isinstance(scope, _SkillDirectScope):
            return frozenset()
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            return frozenset()
        return scope.allowed_tools


def trusted_skill_scope_active(agent: Any) -> bool:
    """Return whether a proven, task-bound skill capability is active."""
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        scope = getattr(agent, "_zet_agent_skill_direct_scope", None)
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        return bool(
            isinstance(scope, _SkillDirectScope)
            and turn_identity is not None
            and scope.turn_identity == turn_identity
            and isinstance(task, _SkillDirectTaskContext)
            and task.turn_identity == turn_identity
            and task.task_sha256 == scope.task_sha256
            and (scope.allowed_tools or scope.policy_exhausted)
        )


def _video_edit_runtime_argv(
    function_args: Mapping[str, Any],
) -> list[str] | None:
    """Return argv only for a startup-anchored trusted video runtime command."""
    command = function_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        from tools.terminal_tool import _parse_video_edit_runtime_command

        parsed = _parse_video_edit_runtime_command(command)
    except Exception:
        return None
    argv = getattr(parsed, "argv", None)
    if not isinstance(argv, list) or len(argv) < 2:
        return None
    if not all(isinstance(value, str) for value in argv):
        return None
    return list(argv)


def _camera_runtime_argv(
    function_args: Mapping[str, Any],
) -> list[str] | None:
    """Return argv only for a startup-anchored trusted camsnap command."""
    command = function_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        from tools.terminal_tool import _parse_camera_runtime_command

        parsed = _parse_camera_runtime_command(command)
    except Exception:
        return None
    argv = getattr(parsed, "argv", None)
    if not isinstance(argv, list) or len(argv) < 3:
        return None
    if not all(isinstance(value, str) for value in argv):
        return None
    return list(argv)


def _camera_command_policy(
    function_args: Mapping[str, Any],
    *,
    camera_ids: frozenset[str] = frozenset(),
    inventory_only: bool = False,
) -> bool:
    if any(
        bool(function_args.get(field))
        for field in (
            "background",
            "force",
            "notify_on_complete",
            "pty",
            "watch_patterns",
        )
    ):
        return False
    from tools.runtime_workdir import AGENT_OUTPUT_WORKDIR

    workdir = function_args.get("workdir")
    if workdir not in (None, "", AGENT_OUTPUT_WORKDIR):
        return False
    argv = _camera_runtime_argv(function_args)
    if argv is None:
        return False
    arguments = argv[2:]
    if arguments == ["list"]:
        return True
    if inventory_only:
        return False
    return bool(
        len(arguments) >= 3
        and arguments[1] == "--camera-id"
        and arguments[2] in camera_ids
    )


def _camera_ids_from_terminal_result(
    result: Mapping[str, Any],
) -> frozenset[str] | None:
    """Extract one response-size-bounded, validated camera-ID snapshot."""
    output = result.get("output")
    if not isinstance(output, str) or len(output.encode("utf-8")) > 1024 * 1024:
        return None
    try:
        payload = json.loads(output)
    except (TypeError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("action") != "list"
        or data.get("status") != "ok"
        or not isinstance(cameras, list)
    ):
        return None
    camera_ids: set[str] = set()
    for camera in cameras:
        camera_id = camera.get("camera_id") if isinstance(camera, dict) else None
        if (
            not isinstance(camera_id, str)
            or _CAMERA_ID_RE.fullmatch(camera_id) is None
        ):
            return None
        camera_ids.add(camera_id)
    return frozenset(camera_ids)


def _trusted_camera_attachment_path(raw_path: Any) -> str | None:
    """Resolve one regular snapshot under the active profile output root."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    try:
        from tools.runtime_workdir import agent_output_dir

        output_root = agent_output_dir()
        if not output_root:
            return None
        candidate = os.path.abspath(raw_path.strip())
        root = os.path.realpath(output_root)
        resolved = os.path.realpath(candidate)
        if os.path.commonpath((root, resolved)) != root:
            return None
        value = os.lstat(candidate)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
            return None
        return resolved
    except (OSError, ValueError):
        return None


def _camera_attachment_path_from_terminal_result(
    result: Mapping[str, Any],
) -> str | None:
    """Extract the exact trusted attachment emitted by one successful snap."""
    output = result.get("output")
    if not isinstance(output, str) or len(output.encode("utf-8")) > 1024 * 1024:
        return None
    try:
        payload = json.loads(output)
    except (TypeError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("action") != "snap"
        or data.get("status") != "ok"
    ):
        return None
    return _trusted_camera_attachment_path(data.get("attachment_path"))


def _camera_vision_policy(
    function_args: Mapping[str, Any],
    *,
    attachment_paths: frozenset[str],
) -> bool:
    """Allow one bounded question about only the current trusted snapshot."""
    if set(function_args) != {"image_url", "question"}:
        return False
    image_url = function_args.get("image_url")
    question = function_args.get("question")
    trusted_image_path = _trusted_camera_attachment_path(image_url)
    return bool(
        isinstance(image_url, str)
        and image_url in attachment_paths
        and trusted_image_path == image_url
        and isinstance(question, str)
        and question.strip()
        and len(question) <= _CAMERA_ANALYSIS_QUESTION_LIMIT
    )


def _printer3d_runtime_argv(function_args: Mapping[str, Any]) -> list[str] | None:
    command = function_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        from tools.terminal_tool import _parse_printer3d_runtime_command

        parsed = _parse_printer3d_runtime_command(command)
    except Exception:
        return None
    argv = getattr(parsed, "argv", None)
    if not isinstance(argv, list) or len(argv) < 3 or not all(isinstance(value, str) for value in argv):
        return None
    return list(argv)


def _printer3d_command_policy(
    function_args: Mapping[str, Any],
    *,
    relative_path: str,
) -> bool:
    if any(
        bool(function_args.get(field))
        for field in ("background", "force", "notify_on_complete", "pty", "watch_patterns", "workdir")
    ):
        return False
    argv = _printer3d_runtime_argv(function_args)
    if argv is None:
        return False
    script_name = os.path.basename(argv[1])
    if relative_path == "skills/printer3d/SKILL.md":
        return script_name == "printer3d_connector.py"
    if relative_path == "skills/printer3d-control/SKILL.md":
        return script_name == "printer3d_control.py"
    return False


def _video_edit_command_policy(
    function_args: Mapping[str, Any],
) -> tuple[bool, bool]:
    if any(
        bool(function_args.get(field))
        for field in (
            "background",
            "force",
            "notify_on_complete",
            "pty",
            "watch_patterns",
            "workdir",
        )
    ):
        return False, False
    argv = _video_edit_runtime_argv(function_args)
    if argv is None:
        return False, False
    may_authorize_memory = bool(
        Path(argv[1]).name == "preference_resolver.py"
        and len(argv) >= 3
        and argv[2] in _VIDEO_EDIT_MEMORY_ACTIONS
    )
    return True, may_authorize_memory


def _recoverable_video_edit_command_format_error(
    function_name: str,
    function_args: Mapping[str, Any],
) -> bool:
    """Recognize a blocked trusted-helper attempt without executing any shell."""
    if function_name != "terminal":
        return False
    command = function_args.get("command")
    if not isinstance(command, str):
        return False
    try:
        from tools.terminal_tool import _VIDEO_EDIT_RUNTIME_SCRIPTS
    except Exception:
        return False
    return any(script_name in command for script_name in _VIDEO_EDIT_RUNTIME_SCRIPTS)


def _canonical_memory_payload_sha256(function_args: Mapping[str, Any]) -> str:
    if set(function_args) != {"operations", "target"}:
        return ""
    target = function_args.get("target")
    operations = function_args.get("operations")
    if target not in {"memory", "user"} or not isinstance(operations, list):
        return ""
    try:
        canonical = json.dumps(
            {"operations": operations, "target": target},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_tool_args_sha256(function_args: Mapping[str, Any]) -> str:
    """Freeze exact preflight args for the final registry dispatch check."""
    try:
        canonical = json.dumps(
            dict(function_args),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return ""
    if len(canonical.encode("utf-8")) > 256 * 1024:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalized_registry_tool_args(
    function_name: str,
    function_args: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the same schema-coerced args the registry will dispatch."""
    try:
        normalized = copy.deepcopy(dict(function_args))
        from model_tools import coerce_tool_args

        normalized = coerce_tool_args(function_name, normalized)
    except Exception as exc:
        logger.warning(
            "zet_agent: failed to normalize trusted %s args: %s",
            function_name,
            exc,
        )
        return None
    return normalized if isinstance(normalized, dict) else None


def _memory_payload_hashes_from_terminal_result(
    result: Mapping[str, Any],
) -> frozenset[str]:
    output = result.get("output")
    if not isinstance(output, str) or len(output) > 64 * 1024:
        return frozenset()
    try:
        payload = json.loads(output)
    except (TypeError, ValueError):
        return frozenset()
    operations = payload.get("operations") if isinstance(payload, dict) else None
    if not isinstance(operations, list) or not 1 <= len(operations) <= 4:
        return frozenset()

    grouped: dict[str, list[dict[str, Any]]] = {"memory": [], "user": []}
    for raw_operation in operations:
        if not isinstance(raw_operation, dict):
            return frozenset()
        operation = dict(raw_operation)
        target = operation.pop("target", None)
        action = operation.get("action")
        if target not in grouped or action not in {"add", "remove", "replace"}:
            return frozenset()
        if set(operation) - {"action", "content", "old_text"}:
            return frozenset()
        for field in ("content", "old_text"):
            value = operation.get(field)
            if value is not None and (
                not isinstance(value, str) or len(value) > 16 * 1024
            ):
                return frozenset()
        if action == "add" and not operation.get("content"):
            return frozenset()
        if action == "remove" and not operation.get("old_text"):
            return frozenset()
        if action == "replace" and not (
            operation.get("content") and operation.get("old_text")
        ):
            return frozenset()
        grouped[target].append(operation)

    hashes: set[str] = set()
    for target, target_operations in grouped.items():
        if not target_operations:
            continue
        digest = _canonical_memory_payload_sha256(
            {"operations": target_operations, "target": target}
        )
        if digest:
            hashes.add(digest)
    return frozenset(hashes)


def trusted_skill_operation_block_message(
    agent: Any,
    *,
    function_name: str,
    function_args: Mapping[str, Any],
) -> str | None:
    """Consume an active capability for one exact, pre-dispatch operation.

    A successful allowed operation may re-arm the scope through
    :func:`apply_trusted_skill_execution`. Any other operation revokes it
    before dispatch, so trusted helper authority cannot escape its task.
    """
    _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        scope = getattr(agent, "_zet_agent_skill_direct_scope", None)
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if isinstance(scope, _SkillDirectScope) and (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            agent._zet_agent_skill_direct_scope = None
            agent._zet_agent_skill_direct_operation = None
            logger.warning(
                "zet_agent: revoked trusted skill scope for mismatched task-local turn"
            )
            return (
                "The trusted skill execution scope belongs to another "
                "task-local turn. The scope was revoked before execution; "
                "reload the trusted skill for this turn."
            )
        if not isinstance(scope, _SkillDirectScope):
            operation = getattr(agent, "_zet_agent_skill_direct_operation", None)
            if isinstance(operation, _SkillDirectOperation):
                if (
                    turn_identity is None
                    or operation.scope.turn_identity != turn_identity
                ):
                    agent._zet_agent_skill_direct_operation = None
                    return (
                        "The trusted skill operation belongs to another "
                        "task-local turn. The scope was revoked before this "
                        "call; reload the trusted skill for this turn."
                    )
                # A provider violating the one-operation contract must not
                # race a second call past the consumed scope. This also clears
                # a failed/plugin-blocked operation on the next attempted call.
                agent._zet_agent_skill_direct_operation = None
                return (
                    "The trusted skill execution scope already has an "
                    "operation in flight or did not complete successfully. "
                    "The scope was revoked before this call; reload the trusted "
                    "skill for this turn."
                )
            if (
                function_name == "terminal"
                and _video_edit_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked trusted video runtime command without "
                    "a current trusted skill scope"
                )
                return (
                    "Trusted video-edit runtime commands require a current "
                    "request-bound scope minted by the trusted `skill_view` "
                    "result. Load the trusted skill and retry the exact operation."
                )
            if (
                function_name == "terminal"
                and _camera_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked camera runtime command without a current "
                    "trusted camsnap scope"
                )
                return (
                    "Trusted camera commands require a current request-bound "
                    "scope minted by the attested `camsnap` skill_view result. "
                    "Load that trusted skill and retry the exact operation."
                )
            if (
                function_name == "terminal"
                and _printer3d_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked printer3d runtime command without a current trusted scope"
                )
                return (
                    "Trusted 3D-printer commands require a current request-bound "
                    "scope minted by an attested printer skill. Load the matching "
                    "trusted skill and retry the exact operation."
                )
            return None

        if scope.policy_exhausted:
            return (
                "The trusted skill workflow exhausted its bounded command "
                "corrections. Do not call more trusted helpers; return a concise "
                "product error."
            )

        allowed = function_name in scope.allowed_tools
        operation_scope = scope
        may_authorize_memory = False
        authorized_args_sha256 = ""
        if allowed and function_name == "terminal":
            normalized_args = _normalized_registry_tool_args(
                function_name,
                function_args,
            )
            if normalized_args is None:
                allowed = False
            else:
                if scope.relative_path == _CAMERA_SKILL_PATH:
                    allowed = _camera_command_policy(
                        normalized_args,
                        camera_ids=scope.camera_ids,
                        inventory_only=scope.camera_inventory_only,
                    )
                elif scope.relative_path in _PRINTER3D_SKILL_PATHS:
                    allowed = _printer3d_command_policy(
                        normalized_args,
                        relative_path=scope.relative_path,
                    )
                else:
                    allowed, may_authorize_memory = _video_edit_command_policy(
                        normalized_args
                    )
                authorized_args_sha256 = _canonical_tool_args_sha256(
                    normalized_args
                )
            allowed = bool(
                allowed
                and scope.execution_receipt is not None
                and authorized_args_sha256
            )
        elif allowed and function_name == "vision_analyze":
            normalized_args = _normalized_registry_tool_args(
                function_name,
                function_args,
            )
            if normalized_args is None:
                allowed = False
            else:
                allowed = bool(
                    scope.relative_path == _CAMERA_SKILL_PATH
                    and scope.execution_receipt is not None
                    and _camera_vision_policy(
                        normalized_args,
                        attachment_paths=scope.camera_attachment_paths,
                    )
                )
                authorized_args_sha256 = _canonical_tool_args_sha256(
                    normalized_args
                )
                allowed = bool(allowed and authorized_args_sha256)
        elif allowed and function_name == "memory":
            memory_digest = _canonical_memory_payload_sha256(function_args)
            allowed = bool(memory_digest and memory_digest in scope.memory_payload_sha256)
            if allowed:
                authorized_args_sha256 = memory_digest
                remaining = scope.memory_payload_sha256 - {memory_digest}
                allowed_tools = scope.allowed_tools
                if not remaining:
                    allowed_tools = allowed_tools - {"memory"}
                operation_scope = replace(
                    scope,
                    allowed_tools=allowed_tools,
                    memory_payload_sha256=remaining,
                )
        if not allowed:
            agent._zet_agent_skill_direct_operation = None
            if (
                scope.command_format_retries > 0
                and _recoverable_video_edit_command_format_error(
                    function_name,
                    function_args,
                )
            ):
                agent._zet_agent_skill_direct_scope = replace(
                    scope,
                    command_format_retries=scope.command_format_retries - 1,
                )
                logger.warning(
                    "zet_agent: blocked malformed trusted video command and kept "
                    "one bounded trusted retry"
                )
                return (
                    "The trusted video-edit command was blocked before execution. "
                    "Retry only the helper as one foreground `python3` command. "
                    "Do not prepend `mkdir` or "
                    "`cd`, use shell operators or wrappers, set `workdir`, or start "
                    "a background/PTY process; trusted helpers create their own "
                    "bounded output directories."
                )
            if scope.policy_violation_retries > 0:
                agent._zet_agent_skill_direct_scope = replace(
                    scope,
                    policy_violation_retries=scope.policy_violation_retries - 1,
                )
                if scope.relative_path == _CAMERA_SKILL_PATH:
                    logger.warning(
                        "zet_agent: blocked out-of-policy trusted camera operation "
                        "and kept one bounded trusted retry"
                    )
                    if function_name == "vision_analyze":
                        return (
                            "The trusted camera analysis was blocked before "
                            "execution. Analyze only the exact attachment_path "
                            "returned by the current turn's successful snapshot, "
                            "with one bounded visual question. This is a "
                            "snapshot-binding error, not a Connector permission "
                            "failure."
                        )
                    return (
                        "The trusted camera operation was blocked before execution. "
                        "Run the pinned camera helper directly with no prerequisite "
                        "terminal command and no custom workdir. Start with `list`, "
                        "then use only a camera_id returned by that list. This is a "
                        "command-policy error, not a Connector permission failure."
                    )
                logger.warning(
                    "zet_agent: blocked out-of-policy trusted video operation and "
                    "kept one bounded trusted retry"
                )
                return (
                    "The trusted skill execution scope does not authorize "
                    f"`{function_name}` with these arguments. Retry only the exact "
                    "pinned helper as one foreground `python3` command; do not "
                    "switch to generic shell/file tools."
                )
            agent._zet_agent_skill_direct_scope = replace(
                scope,
                allowed_tools=frozenset(),
                policy_exhausted=True,
            )
            logger.warning(
                "zet_agent: exhausted bounded trusted %s corrections before "
                "out-of-policy tool %s",
                "camera" if scope.relative_path == _CAMERA_SKILL_PATH else "video",
                function_name or "<missing>",
            )
            return (
                "The trusted skill operation was rejected after bounded "
                "corrections. Do not call more trusted helpers; return a concise "
                "product error."
            )

        agent._zet_agent_skill_direct_scope = None
        agent._zet_agent_skill_direct_operation = _SkillDirectOperation(
            scope=operation_scope,
            function_name=function_name,
            may_authorize_memory=may_authorize_memory,
            authorized_args_sha256=authorized_args_sha256,
        )
        return None


def trusted_skill_operation_execution_block_message(
    agent: Any,
    *,
    function_name: str,
    function_args: Mapping[str, Any],
) -> str | None:
    """Revalidate the final trusted-memory payload after execution middleware."""
    if (
        (getattr(agent, "platform", "") or "") != "zet_agent"
        or function_name != "memory"
    ):
        return None

    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        operation = getattr(agent, "_zet_agent_skill_direct_operation", None)
        if not isinstance(operation, _SkillDirectOperation):
            if "memory" not in set(
                getattr(agent, "valid_tool_names", set()) or set()
            ):
                return (
                    "The scoped video-edit memory exception has no current "
                    "exact authorization. The write was blocked."
                )
            return None
        if operation.function_name != "memory":
            if "memory" not in set(
                getattr(agent, "valid_tool_names", set()) or set()
            ):
                return (
                    "The scoped video-edit memory exception belongs to another "
                    "operation. The write was blocked."
                )
            return None

        scope = operation.scope
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            agent._zet_agent_skill_direct_operation = None
            return (
                "The trusted video-edit memory operation belongs to another "
                "task-local turn. The operation was revoked before writing."
            )

        final_digest = _canonical_memory_payload_sha256(function_args)
        if (
            operation.execution_claimed
            or not final_digest
            or final_digest != operation.authorized_args_sha256
        ):
            agent._zet_agent_skill_direct_operation = None
            logger.warning(
                "zet_agent: blocked trusted memory payload changed after "
                "exact authorization"
            )
            return (
                "The trusted video-edit memory payload changed after exact "
                "authorization. The operation was revoked before writing."
            )

        agent._zet_agent_skill_direct_operation = replace(
            operation,
            execution_claimed=True,
        )
        return None


def _claim_trusted_terminal_dispatch(
    agent: Any,
    function_args: Mapping[str, Any],
) -> tuple[_TrustedExecutionReceipt | None, str | None]:
    """Claim one exact terminal dispatch after all plugin gates have run."""
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        operation = getattr(agent, "_zet_agent_skill_direct_operation", None)
        if not isinstance(operation, _SkillDirectOperation):
            if (
                _video_edit_runtime_argv(function_args) is not None
                or _camera_runtime_argv(function_args) is not None
                or _printer3d_runtime_argv(function_args) is not None
            ):
                return None, (
                    "Trusted runtime commands require a current "
                    "request-bound operation at final dispatch. Reload the "
                    "trusted skill and retry the exact operation."
                )
            return None, None

        if operation.function_name != "terminal" or operation.execution_claimed:
            agent._zet_agent_skill_direct_operation = None
            return None, (
                "The trusted terminal operation was already claimed "
                "or belongs to another tool. It was revoked before dispatch."
            )

        scope = operation.scope
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            agent._zet_agent_skill_direct_operation = None
            return None, (
                "The trusted terminal operation belongs to another "
                "task-local turn. It was revoked before dispatch."
            )

        normalized_args = _normalized_registry_tool_args("terminal", function_args)
        final_digest = (
            _canonical_tool_args_sha256(normalized_args)
            if normalized_args is not None
            else ""
        )
        if normalized_args is None:
            allowed, may_authorize_memory = False, False
        elif scope.relative_path == _CAMERA_SKILL_PATH:
            allowed = _camera_command_policy(
                normalized_args,
                camera_ids=scope.camera_ids,
                inventory_only=scope.camera_inventory_only,
            )
            may_authorize_memory = False
        elif scope.relative_path in _PRINTER3D_SKILL_PATHS:
            allowed = _printer3d_command_policy(
                normalized_args,
                relative_path=scope.relative_path,
            )
            may_authorize_memory = False
        else:
            allowed, may_authorize_memory = _video_edit_command_policy(
                normalized_args
            )
        receipt = scope.execution_receipt
        if (
            not allowed
            or not final_digest
            or final_digest != operation.authorized_args_sha256
            or may_authorize_memory != operation.may_authorize_memory
            or not isinstance(receipt, _TrustedExecutionReceipt)
        ):
            agent._zet_agent_skill_direct_operation = None
            logger.warning(
                "zet_agent: blocked trusted terminal args changed after exact "
                "authorization"
            )
            return None, (
                "The trusted terminal arguments changed after exact "
                "authorization. The operation was revoked before dispatch."
            )

        agent._zet_agent_skill_direct_operation = replace(
            operation,
            execution_claimed=True,
        )
        return receipt, None


def _claim_trusted_camera_vision_dispatch(
    agent: Any,
    function_args: Mapping[str, Any],
) -> str | None:
    """Revalidate one exact snapshot-analysis call after all middleware."""
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        operation = getattr(agent, "_zet_agent_skill_direct_operation", None)
        if not isinstance(operation, _SkillDirectOperation):
            return None
        if (
            operation.function_name != "vision_analyze"
            or operation.execution_claimed
        ):
            agent._zet_agent_skill_direct_operation = None
            return (
                "The trusted camera analysis operation was already claimed "
                "or belongs to another tool. It was revoked before dispatch."
            )

        scope = operation.scope
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or scope.relative_path != _CAMERA_SKILL_PATH
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            agent._zet_agent_skill_direct_operation = None
            return (
                "The trusted camera analysis operation belongs to another "
                "task-local turn. It was revoked before dispatch."
            )

        normalized_args = _normalized_registry_tool_args(
            "vision_analyze",
            function_args,
        )
        final_digest = (
            _canonical_tool_args_sha256(normalized_args)
            if normalized_args is not None
            else ""
        )
        if (
            normalized_args is None
            or not _camera_vision_policy(
                normalized_args,
                attachment_paths=scope.camera_attachment_paths,
            )
            or not final_digest
            or final_digest != operation.authorized_args_sha256
        ):
            agent._zet_agent_skill_direct_operation = None
            logger.warning(
                "zet_agent: blocked trusted camera analysis args changed after "
                "exact authorization"
            )
            return (
                "The trusted camera analysis arguments changed after exact "
                "authorization. The operation was revoked before dispatch."
            )

        agent._zet_agent_skill_direct_operation = replace(
            operation,
            execution_claimed=True,
        )
        return None


def dispatch_trusted_skill_operation(
    agent: Any,
    *,
    function_name: str,
    function_args: Mapping[str, Any],
    dispatch: Callable[[], Any],
) -> Any:
    """Run registry dispatch inside the narrow trusted-skill boundary.

    Plugin pre-hooks and execution middleware run before this function, while
    post/transform hooks run after it returns. The private receipt therefore
    exists only during the actual terminal registry handler. Terminal success
    is bound to the raw handler result before plugins can replace it, while a
    ``skill_view`` scope is activated later from the final displayed result.
    """
    _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
    receipt: _TrustedExecutionReceipt | None = None
    block_message: str | None = None
    if function_name == "terminal":
        receipt, block_message = _claim_trusted_terminal_dispatch(
            agent,
            function_args,
        )
    elif function_name == "vision_analyze":
        block_message = _claim_trusted_camera_vision_dispatch(
            agent,
            function_args,
        )

    if block_message is not None:
        result = json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": block_message,
                "video_edit_runtime_direct": False,
                "video_edit_runtime_blocked": True,
            },
            ensure_ascii=False,
        )
        if function_name != "skill_view":
            apply_trusted_skill_execution(
                agent,
                function_name=function_name,
                function_result=result,
            )
        return result

    if receipt is not None:
        _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(receipt)
    try:
        result = dispatch()
    except BaseException:
        _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
        if function_name != "skill_view":
            apply_trusted_skill_execution(
                agent,
                function_name=function_name,
                function_result=None,
            )
        raise
    finally:
        _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)

    if function_name != "skill_view":
        apply_trusted_skill_execution(
            agent,
            function_name=function_name,
            function_result=result,
        )
    return result


def _rearm_skill_direct_scope_after_success(
    agent: Any,
    *,
    function_name: str,
    function_result: Any,
) -> bool:
    _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
    turn_identity = _current_skill_direct_turn_identity()
    with _SKILL_DIRECT_LOCK:
        operation = getattr(agent, "_zet_agent_skill_direct_operation", None)
        agent._zet_agent_skill_direct_operation = None
        if not isinstance(operation, _SkillDirectOperation):
            return False
        if operation.function_name != function_name:
            return False

        scope = operation.scope
        task = getattr(agent, "_zet_agent_skill_direct_task", None)
        if (
            turn_identity is None
            or scope.turn_identity != turn_identity
            or not isinstance(task, _SkillDirectTaskContext)
            or task.turn_identity != turn_identity
            or task.task_sha256 != scope.task_sha256
        ):
            return False
        successful = function_name == "todo"
        if function_name == "clarify" and isinstance(function_result, str):
            try:
                clarify_result = json.loads(function_result)
            except (TypeError, ValueError):
                clarify_result = None
            user_response = (
                clarify_result.get("user_response")
                if isinstance(clarify_result, dict)
                else None
            )
            successful = bool(
                isinstance(clarify_result, dict)
                and not clarify_result.get("error")
                and isinstance(user_response, str)
                and user_response.strip()
            )
        if function_name == "memory" and isinstance(function_result, str):
            try:
                memory_result = json.loads(function_result)
            except (TypeError, ValueError):
                memory_result = None
            successful = bool(
                isinstance(memory_result, dict)
                and memory_result.get("success") is True
            )
        if function_name == "terminal" and isinstance(function_result, str):
            try:
                result = json.loads(function_result)
            except (TypeError, ValueError):
                result = None
            exit_code = result.get("exit_code") if isinstance(result, dict) else None
            if scope.relative_path == _CAMERA_SKILL_PATH:
                runtime_direct_field = "camera_runtime_direct"
                runtime_blocked_field = "camera_runtime_blocked"
            elif scope.relative_path in _PRINTER3D_SKILL_PATHS:
                runtime_direct_field = "printer3d_runtime_direct"
                runtime_blocked_field = "printer3d_runtime_blocked"
            else:
                runtime_direct_field = "video_edit_runtime_direct"
                runtime_blocked_field = "video_edit_runtime_blocked"
            successful = bool(
                isinstance(result, dict)
                and result.get(runtime_direct_field) is True
                and result.get(runtime_blocked_field) is not True
                and isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code == 0
                and not result.get("error")
            )
            memory_hashes = (
                _memory_payload_hashes_from_terminal_result(result)
                if (
                    successful
                    and scope.relative_path == _VIDEO_EDIT_SKILL_PATH
                    and operation.may_authorize_memory
                )
                else frozenset()
            )
            listed_camera_ids = (
                _camera_ids_from_terminal_result(result)
                if successful and scope.relative_path == _CAMERA_SKILL_PATH
                else None
            )
            camera_attachment_path = (
                _camera_attachment_path_from_terminal_result(result)
                if successful and scope.relative_path == _CAMERA_SKILL_PATH
                else None
            )
            allowed_tools = scope.allowed_tools - {"memory"}
            if memory_hashes:
                allowed_tools = allowed_tools | {"memory"}
            scope = replace(
                scope,
                allowed_tools=allowed_tools,
                memory_payload_sha256=memory_hashes,
                camera_ids=(
                    listed_camera_ids
                    if listed_camera_ids is not None
                    else scope.camera_ids
                ),
                camera_attachment_paths=(
                    frozenset({camera_attachment_path})
                    if camera_attachment_path is not None
                    else scope.camera_attachment_paths
                ),
                command_format_retries=1,
                policy_violation_retries=_VIDEO_EDIT_POLICY_VIOLATION_RETRIES,
                policy_exhausted=False,
            )
            if camera_attachment_path is not None:
                _remember_camera_resume_locked(
                    turn_identity=turn_identity,
                    now=time.monotonic(),
                )
        if not successful:
            return False

        agent._zet_agent_skill_direct_scope = scope
        return True


def request_response_mode(agent: Any) -> str:
    """Return the only structured response mode owned by the App plan flow."""
    mode = str(getattr(agent, "_zet_agent_response_mode", "") or "").strip().lower()
    return "plan" if mode == "plan" else ""


def _hardware_enrollment_requested_types(user_message: Any) -> tuple[str, ...]:
    """Recognize only short, direct hardware enrollment requests.

    Skill selection remains the primary semantic path. This bounded classifier
    is a delivery guard for models that answer with settings prose instead of
    the credential-free client intent promised by ``connector-setup``.
    """
    task_text, _ = _task_text_and_video_asset(user_message)
    normalized = " ".join(_strip_gateway_model_switch_note(task_text).split())
    # A connected-state noun phrase is not an enrollment action.  Remove it
    # before matching action verbs so requests such as “查看硬件连接中的电脑” or
    # “show connected computers” cannot append a setup card after a successful
    # PC-file operation.  Explicit actions such as “重新连接电脑” remain intact.
    action_text = _HARDWARE_ENROLLMENT_CONNECTION_STATE_RE.sub(" ", normalized)
    if (
        not normalized
        or len(normalized) > 320
        or not _HARDWARE_ENROLLMENT_ACTION_RE.search(action_text)
        or _HARDWARE_ENROLLMENT_META_OR_DIAG_RE.search(normalized)
    ):
        return ()

    positioned_types: list[tuple[int, str]] = []
    for hardware_type, pattern in _HARDWARE_ENROLLMENT_TYPE_RES:
        match = pattern.search(normalized)
        if match is not None:
            positioned_types.append((match.start(), hardware_type))
    if positioned_types:
        positioned_types.sort(key=lambda item: item[0])
        return tuple(hardware_type for _, hardware_type in positioned_types)
    if _HARDWARE_ENROLLMENT_GENERIC_RE.search(normalized):
        return ("camera", "printer3d", "pc_node")
    return ()


def ensure_hardware_enrollment_intent(
    agent: Any,
    *,
    user_message: Any,
    response_text: str,
    completed: bool,
    failed: bool,
    interrupted: bool,
    structured_output: bool,
) -> str:
    """Append a canonical, secret-free hardware enrollment intent when needed.

    The transform never discovers hardware or accepts addresses/credentials. It
    only gives first-party App/Web clients enough information to render the
    explicit user-confirmation card. Existing model-authored intent blocks are
    replaced so malformed or credential-bearing fields cannot suppress the
    trusted canonical card.
    """
    text = str(response_text or "")
    if (
        (getattr(agent, "platform", "") or "") != "zet_agent"
        or not completed
        or failed
        or interrupted
        or structured_output
        or not text.strip()
    ):
        return text
    visible_text = _HARDWARE_ENROLLMENT_BLOCK_RE.sub("", text)
    visible_text = _LEGACY_HARDWARE_CAMERA_INTENT_BLOCK_RE.sub("", visible_text)
    removed_model_intent = visible_text != text
    if removed_model_intent:
        visible_text = visible_text.strip()
    requested_types = _hardware_enrollment_requested_types(user_message)
    if not requested_types:
        # A model-authored setup block is not authoritative. Remove stale or
        # over-eager hardware cards from status/usage turns even when no new
        # canonical enrollment intent needs to be appended.
        return visible_text if removed_model_intent else text

    payload = json.dumps(
        {
            "schema_version": "1",
            "kind": "hardware",
            "requested_types": list(requested_types),
            "discovery_requested": True,
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        f"{visible_text}\n\n```{_HARDWARE_ENROLLMENT_FENCE}\n"
        f"{payload}\n```"
    )


def reset_trusted_skill_execution(
    agent: Any,
    user_message: Any = None,
    *,
    explicit_skill_slug: str = "",
    tool_execution_allowed: bool = True,
) -> None:
    """Clear trusted execution and bind eligibility to the new user task."""
    _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_scope = None
        agent._zet_agent_skill_direct_operation = None
        task = _skill_direct_task_context(
            agent,
            user_message,
            explicit_skill_slug=explicit_skill_slug,
            tool_execution_allowed=tool_execution_allowed,
        )
        if task.video_edit_explicit:
            resume_key = _current_skill_direct_resume_key()
            if resume_key is not None:
                sessions = _video_edit_resume_sessions_locked(
                    now=time.monotonic(),
                )
                source_turn_id = (
                    str(task.turn_identity[0] or "").strip()
                    if task.turn_identity is not None
                    else ""
                )
                sessions[resume_key] = _VideoEditResumeGrant(
                    expires_at=time.monotonic()
                    + _VIDEO_EDIT_RESUME_TTL_SECONDS,
                    source_turn_id=source_turn_id,
                )
                sessions.move_to_end(resume_key)
                while len(sessions) > _VIDEO_EDIT_RESUME_MAX_SESSIONS:
                    sessions.popitem(last=False)
        agent._zet_agent_skill_direct_task = task


def apply_trusted_skill_execution(
    agent: Any,
    *,
    function_name: str,
    function_result: Any,
) -> bool:
    """Activate or re-arm a least-scope trusted-skill execution capability.

    The JSON fields are display data, not trust inputs. Only ``skill_view`` can
    mint a proof after its exact raw bytes and path metadata match the immutable
    startup snapshot. The current user task and per-skill operation policy are
    separate trust inputs. Plan presentation remains owned by the App plan
    capability and is intentionally not changed here.
    """
    if function_name != "skill_view":
        return _rearm_skill_direct_scope_after_success(
            agent,
            function_name=function_name,
            function_result=function_result,
        )
    if function_name != "skill_view" or not isinstance(function_result, str):
        return False
    try:
        payload = json.loads(function_result)
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    token = payload.get(_ATTESTATION_FIELD)
    if not isinstance(token, str) or not token:
        return False
    pending = _consume_skill_attestation(token, function_result)
    if pending is None:
        return False

    if (getattr(agent, "platform", "") or "") != "zet_agent":
        return False

    if pending.relative_path not in {_VIDEO_EDIT_SKILL_PATH, _CAMERA_SKILL_PATH, *_PRINTER3D_SKILL_PATHS}:
        return False
    task = getattr(agent, "_zet_agent_skill_direct_task", None)
    task_matches_skill = bool(
        isinstance(task, _SkillDirectTaskContext)
        and (
            (
                pending.relative_path == _VIDEO_EDIT_SKILL_PATH
                and task.video_edit_applicable
            )
            or (
                pending.relative_path == _CAMERA_SKILL_PATH
                and task.camera_applicable
            )
            or (
                pending.relative_path in _PRINTER3D_SKILL_PATHS
                and task.printer3d_applicable
            )
        )
    )
    if not task_matches_skill:
        logger.warning(
            "zet_agent: trusted skill %s did not match the current user task",
            pending.relative_path,
        )
        return False
    current_turn_identity = _current_skill_direct_turn_identity()
    if (
        current_turn_identity is None
        or task.turn_identity != current_turn_identity
        or pending.turn_identity != current_turn_identity
    ):
        logger.warning(
            "zet_agent: trusted skill %s rejected for mismatched turn identity",
            pending.relative_path,
        )
        return False
    execution_receipt = _capture_trusted_execution_receipt(current_turn_identity)
    if execution_receipt is None:
        return False
    allowed_tools = (
        _CAMERA_DIRECT_TOOLS
        if pending.relative_path == _CAMERA_SKILL_PATH
        else _PRINTER3D_DIRECT_TOOLS
        if pending.relative_path in _PRINTER3D_SKILL_PATHS
        else _VIDEO_EDIT_DIRECT_TOOLS
    )

    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_operation = None
        agent._zet_agent_skill_direct_scope = _SkillDirectScope(
            relative_path=pending.relative_path,
            task_sha256=task.task_sha256,
            turn_identity=current_turn_identity,
            allowed_tools=allowed_tools,
            execution_receipt=execution_receipt,
            camera_inventory_only=bool(
                pending.relative_path == _CAMERA_SKILL_PATH
                and task.camera_inventory_only
            ),
        )
    logger.info(
        "zet_agent: trusted skill %s activated bounded execution scope",
        pending.relative_path,
    )
    return True


# Production gateways set ZETTLAB_PRESETS_DIR in the process environment before
# Python imports the agent. Capture before any model-authored terminal command.
_TRUSTED_PRESETS_SNAPSHOT = _capture_trusted_presets_snapshot()
