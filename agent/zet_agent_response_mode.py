# zettlab-overlay(H21-unowned): 响应模式与可信技能执行策略状态机整文件为 fork 新增，收敛时迁适配层; upstream: none
"""Bind trusted high-risk skill execution to an exact App turn."""

from __future__ import annotations

import base64
import copy
import hashlib
import ipaddress
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from agent.response_format import response_format_requires_structured_output

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
_CAMERA_SKILL_PATH = "skills/camsnap/SKILL.md"
# zettlab-overlay(camera-confirmation): expose the existing trusted client handoff; upstream: none
_CAMERA_DIRECT_TOOLS = frozenset({"terminal", "vision_analyze", "clarify"})
_CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_PRINTER3D_SKILL_PATHS = frozenset({
    "skills/printer3d/SKILL.md",
    "skills/printer3d-control/SKILL.md",
})
_PRINTER3D_DIRECT_TOOLS = frozenset({"terminal"})
_PLAUD_SKILL_PATH = "skills/plaud-recordings/SKILL.md"
_PLAUD_DIRECT_TOOLS = frozenset({"terminal"})
_SMART_HOME_SKILL_PATH = "skills/smart-home-light-control/SKILL.md"
_SMART_HOME_DIRECT_TOOLS = frozenset({"terminal"})
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
# A grant acknowledgement does not identify which media action to repeat.
# Admit a fresh inventory only; device/Agent/session authorization remains
# authoritative in CameraService. Whole-message matching excludes quotations
# and requests that append background monitoring or recording instructions.
_CAMERA_AUTHORIZATION_RETRY_RE = re.compile(
    r"(?:我)?(?:"
    r"(?:已经|已)?(?:授权|打开|开启)(?:了)?摄像头(?:权限|开关)?"
    r"|摄像头(?:权限|开关)?(?:已经|已)?(?:授权|打开|开启)(?:了)?"
    r")[，,。\s]*(?:请)?(?:再试|重试|重新试)(?:一下|下|一次)?"
    r"(?:刚才的操作)?[。！!\s]*",
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
_PLAUD_INTENT_RE = re.compile(
    r"(?:plaud|录音|录音笔|转写|逐字稿).{0,40}(?:获取|查看|列出|搜索|查找|读取|笔记|摘要|list|search|read|transcript|note)"
    r"|(?:获取|查看|列出|搜索|查找|读取|笔记|摘要|list|search|read|transcript|note).{0,40}(?:plaud|录音|录音笔|转写|逐字稿)",
    re.IGNORECASE | re.DOTALL,
)
_HARDWARE_ENROLLMENT_FENCE = "zettlab-hardware-enrollment-intent"
_CONNECTOR_ENROLLMENT_FENCE = "zettlab-connector-enrollment-intent"
_HARDWARE_ENROLLMENT_BLOCK_RE = re.compile(
    r"\s*```zettlab-hardware-enrollment-intent\s*\r?\n[\s\S]*?\r?\n```",
    re.IGNORECASE,
)
_PLAUD_CONTINUATION_INTENT_RE = re.compile(
    r"\s*(?:"
    r"(?:再|再次|重新)(?:试(?:一)?次|读取(?:一?下|一次)?|读(?:一?下|一次)?|查(?:一?下|一次)?)"
    r"|(?:retry|try again|read again|check again)"
    r")\s*[。！？.!?]*\s*",
    re.IGNORECASE,
)
_CONNECTOR_ENROLLMENT_BLOCK_RE = re.compile(
    r"\s*```zettlab-connector-enrollment-intent\s*\r?\n(?P<payload>[\s\S]*?)\r?\n```",
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
    # zettlab-overlay(connector-enrollment-guard): classify device status as diagnostics; upstream: none
    r"|(?:设备|硬件|摄像头|打印机|电脑|电视)(?:访问)?(?:权限|授权|状态|能力|可用性|连接)"
    r"|\b(?:connected|connecting|connection)\b",
    re.IGNORECASE,
)
# zettlab-overlay(connector-enrollment-guard): keep diagnostics out of setup intent; upstream: none
_HARDWARE_ENROLLMENT_DIAGNOSTIC_RE = re.compile(
    r"(?:检查|查看|查询|诊断|验证|列出|授权|权限|能力|可用|状态|区域)"
    r"|(?:\bdoctor\b|\blist\b|\bstatus\b|\bcapabilit(?:y|ies)\b|"
    r"\bpermission\b|\baccess\b|\bhealth\b|\binspect\b|\bverify\b)",
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
    (
        "tv",
        re.compile(
            r"(?:电视|電視|投屏设备|投屏裝置|投屏|tv|television|display|renderer|テレビ|TV|텔레비전)",
            re.IGNORECASE,
        ),
    ),
    (
        "voice_terminal",
        re.compile(
            r"(?:语音终端|語音終端|麦克风终端|麥克風終端|语音遥控器|語音遙控器|voice\s*terminal|microphone\s*terminal|voice\s*remote)",
            re.IGNORECASE,
        ),
    ),
)
_DISCOVERABLE_SUBNET_HARDWARE_TYPES = frozenset({"camera", "tv"})
_RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_IPV4_SCOPE_RE = re.compile(
    r"(?<![\d.])(?P<address>(?:\d{1,3}\.){3}\d{1,3})(?:/(?P<prefix>\d{1,2}))?(?![\d.])"
)
_IPV4_WILDCARD_SCOPE_RE = re.compile(
    r"(?<![\d.])(?P<base>(?:\d{1,3}\.){3})(?:x|\*)(?:/(?P<prefix>\d{1,2}))?(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)
_CURRENT_PRIVATE_NETWORK_RE = re.compile(
    r"(?:当前|本机|现在所在的?)(?:局域网|网段|子网)|(?:current|local)\s+(?:private\s+)?(?:network|subnet|lan)\b",
    re.IGNORECASE,
)
_LOCAL_PRIVATE_NETWORK_RE = re.compile(
    r"(?:局域网|區域網路|本地网络|本地網路)|\b(?:local\s+network|lan)\b",
    re.IGNORECASE,
)
_SUBNET_DISCOVERY_ACTION_RE = re.compile(
    r"(?:扫描|掃描|扫一下|掃一下|查找|搜索|搜尋|搜寻|发现|發現)"
    r"|\b(?:scan|discover|find|search)\b",
    re.IGNORECASE,
)
_SUBNET_UNSUPPORTED_TYPE_RE = re.compile(
    r"(?:打印机|印表機|プリンター|프린터)|\bprinters?\b",
    re.IGNORECASE,
)
_SUBNET_NON_HARDWARE_SCAN_RE = re.compile(
    r"(?:端口|埠|主机|主機|网关|網關)|\b(?:ports?|hosts?|gateway|nmap)\b",
    re.IGNORECASE,
)
_TRUSTED_POLICY_VIOLATION_RETRIES = 2
_OPAQUE_ACTION_TOKEN_MAX_BYTES = 4096
_CAMERA_RESUME_TTL_SECONDS = 5 * 60
_CAMERA_RESUME_MAX_SESSIONS = 8
_PLAUD_RESUME_TTL_SECONDS = 30 * 60
_PLAUD_RESUME_MAX_SESSIONS = 8
_CAMERA_ANALYSIS_QUESTION_LIMIT = 4096
_GATEWAY_MODEL_SWITCH_NOTE_RE = re.compile(
    r"^\s*\[Note: the model has changed and is now "
    r"[^\]\r\n]{1,128}\. Adjust your self-identification accordingly\.\]\s*",
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
    camera_applicable: bool = False
    camera_explicit: bool = False
    trusted_skill_slug: str = ""
    camera_inventory_only: bool = False
    printer3d_applicable: bool = False
    printer3d_explicit: bool = False
    plaud_applicable: bool = False
    plaud_explicit: bool = False
    smart_home_applicable: bool = False
    smart_home_explicit: bool = False


@dataclass(frozen=True)
class _TrustedExecutionReceipt:
    agent_id: str = field(repr=False)
    action_token: str = field(repr=False)
    hardware_execution_token: str = field(repr=False)
    turn_id: str
    session_id: str
    user_id: str = ""
    gateway_session_key: str = ""
    execution_policy: str = ""


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
    policy_violation_retries: int = _TRUSTED_POLICY_VIOLATION_RETRIES
    policy_exhausted: bool = False


@dataclass(frozen=True)
class _SkillDirectOperation:
    scope: _SkillDirectScope
    function_name: str
    authorized_args_sha256: str = field(
        default="",
        repr=False,
        compare=False,
    )
    execution_claimed: bool = field(default=False, repr=False, compare=False)


@dataclass(frozen=True)
class _CameraResumeGrant:
    expires_at: float
    source_turn_id: str


_TRUSTED_PRESETS_SNAPSHOT: _TrustedPresetsSnapshot | None = None
_PENDING_ATTESTATIONS: OrderedDict[str, _PendingSkillAttestation] = OrderedDict()
_CameraResumeKey = tuple[str, str]
_CAMERA_RESUME_SESSIONS: OrderedDict[
    _CameraResumeKey,
    _CameraResumeGrant,
] = OrderedDict()
_PLAUD_RESUME_SESSIONS: OrderedDict[
    _CameraResumeKey,
    _CameraResumeGrant,
] = OrderedDict()
_ATTESTATION_LOCK = threading.Lock()
_SKILL_DIRECT_LOCK = threading.Lock()
_TRUSTED_HARDWARE_RUNTIME_RECEIPT: ContextVar[
    _TrustedExecutionReceipt | None
] = ContextVar("_TRUSTED_HARDWARE_RUNTIME_RECEIPT", default=None)
_TRUSTED_SKILL_VIEW_FRESH_READ: ContextVar[bool] = ContextVar(
    "_TRUSTED_SKILL_VIEW_FRESH_READ",
    default=False,
)


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


def _current_resume_key() -> _CameraResumeKey | None:
    """Bind resumable hardware intent to the active profile and session."""
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
    resume_key = _current_resume_key()
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


def _plaud_resume_sessions_locked(
    *,
    now: float,
) -> OrderedDict[_CameraResumeKey, _CameraResumeGrant]:
    expired = [
        resume_key
        for resume_key, grant in _PLAUD_RESUME_SESSIONS.items()
        if grant.expires_at <= now
    ]
    for resume_key in expired:
        _PLAUD_RESUME_SESSIONS.pop(resume_key, None)
    while len(_PLAUD_RESUME_SESSIONS) > _PLAUD_RESUME_MAX_SESSIONS:
        _PLAUD_RESUME_SESSIONS.popitem(last=False)
    return _PLAUD_RESUME_SESSIONS


def _remember_plaud_resume_locked(
    *,
    turn_identity: _TurnIdentity,
    now: float,
) -> None:
    """Remember only that this session attested PLAUD, never its data."""
    resume_key = _current_resume_key()
    if resume_key is None:
        return
    sessions = _plaud_resume_sessions_locked(now=now)
    sessions[resume_key] = _CameraResumeGrant(
        expires_at=now + _PLAUD_RESUME_TTL_SECONDS,
        source_turn_id=str(turn_identity[0] or "").strip(),
    )
    sessions.move_to_end(resume_key)
    while len(sessions) > _PLAUD_RESUME_MAX_SESSIONS:
        sessions.popitem(last=False)


def _strip_gateway_model_switch_note(task_text: str) -> str:
    """Remove only the exact gateway-authored model identity preamble."""
    return _GATEWAY_MODEL_SWITCH_NOTE_RE.sub("", task_text, count=1)


def _capture_trusted_execution_receipt(
    turn_identity: _TurnIdentity,
    relative_path: str,
) -> _TrustedExecutionReceipt | None:
    """Freeze the existing hardware transport identity before a helper call.

    Video editing is implemented by ordinary Hermes plugin tools and never
    enters this receipt path. Camera/printer/PLAUD helpers still need the
    platform identity and hardware execution token because they address
    hardware or its authorized cloud data, not the cloud video renderer.
    """
    if relative_path not in {
        _CAMERA_SKILL_PATH,
        *_PRINTER3D_SKILL_PATHS,
        _PLAUD_SKILL_PATH,
        _SMART_HOME_SKILL_PATH,
    }:
        return None
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
            execution_session_key,
            execution_policy,
            get_session_env,
            hardware_execution_token,
            zettlab_auth_principal,
        )

        hardware_token = hardware_execution_token()
        bound_execution_policy = execution_policy()
        gateway_session_key = execution_session_key() or get_session_env(
            "HERMES_SESSION_KEY"
        )
        session_id = get_session_env("HERMES_SESSION_ID")
        if not session_id:
            session_id = gateway_session_key
        user_id = zettlab_auth_principal() or get_session_env("HERMES_SESSION_USER_ID")
    except Exception:
        hardware_token = ""
        bound_execution_policy = ""
        session_id = ""
        gateway_session_key = ""
        user_id = ""

    receipt = _TrustedExecutionReceipt(
        agent_id=_profile_value("ZET_AGENT_ID"),
        action_token=_profile_value("ZETTLAB_AGENT_ACTION_TOKEN"),
        hardware_execution_token=str(hardware_token or "").strip(),
        turn_id=str(turn_identity[0] or "").strip(),
        session_id=str(session_id or "").strip(),
        user_id=str(user_id or "").strip(),
        gateway_session_key=str(gateway_session_key or "").strip(),
        execution_policy=str(bound_execution_policy or "").strip().lower(),
    )
    require_hardware_token = relative_path != _CAMERA_SKILL_PATH
    present = {
        "agent_id": bool(receipt.agent_id),
        "turn_id": bool(receipt.turn_id),
    }
    present.update(
        {
            "action_token": bool(receipt.action_token),
            "hardware_execution_token": bool(receipt.hardware_execution_token) if require_hardware_token else True,
            "session_id": bool(receipt.session_id),
        }
    )
    if not all(present.values()):
        logger.warning(
            "zet_agent: trusted execution receipt incomplete: %s",
            present,
        )
        return None
    if (
        not _is_opaque_action_token(receipt.action_token)
        or (require_hardware_token and re.fullmatch(r"[0-9a-f]{64}", receipt.hardware_execution_token) is None)
    ):
        logger.warning("zet_agent: hardware execution receipt is malformed")
        return None
    return receipt


def _is_opaque_action_token(value: str) -> bool:
    """Validate a profile action capability without imposing token syntax."""
    if not isinstance(value, str) or not value:
        return False
    try:
        if len(value.encode("utf-8")) > _OPAQUE_ACTION_TOKEN_MAX_BYTES:
            return False
    except UnicodeError:
        return False
    return not any(
        char.isspace() or ord(char) < 0x20 or ord(char) == 0x7F
        for char in value
    )


def _trusted_skill_path_for_slug(skill_slug: str) -> str:
    """Derive the canonical signed skill path from an explicit slug."""
    normalized = str(skill_slug or "").strip().lstrip("/").lower()
    if not normalized or re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", normalized) is None:
        return ""
    return f"skills/{normalized}/SKILL.md"


def _trusted_runtime_receipt(require_hardware_token: bool) -> Mapping[str, str]:
    """Return the private one-operation receipt for the camsnap helper."""
    receipt = _TRUSTED_HARDWARE_RUNTIME_RECEIPT.get()
    if (
        receipt is None
        or not _is_opaque_action_token(receipt.action_token)
        or (require_hardware_token and re.fullmatch(r"[0-9a-f]{64}", receipt.hardware_execution_token) is None)
        or not receipt.session_id
    ):
        return {}
    values = {
        "ZET_AGENT_ID": receipt.agent_id,
        "ZETTLAB_AGENT_ACTION_TOKEN": receipt.action_token,
        "HERMES_TURN_ID": receipt.turn_id,
        "HERMES_SESSION_KEY": receipt.session_id,
        "ZETTLAB_USER_ID": receipt.user_id,
    }
    if receipt.hardware_execution_token:
        values["ZETTLAB_HARDWARE_EXECUTION_TOKEN"] = receipt.hardware_execution_token
    return values


def trusted_camera_runtime_receipt() -> Mapping[str, str]:
    """Return camera context; the hardware execution bearer is optional."""
    return _trusted_runtime_receipt(require_hardware_token=False)


def trusted_printer3d_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for signed printer helpers."""
    return _trusted_runtime_receipt(require_hardware_token=False)


def trusted_plaud_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for the PLAUD helper."""
    return _trusted_runtime_receipt(require_hardware_token=False)


def trusted_smart_home_runtime_receipt() -> Mapping[str, str]:
    """Return the private one-operation receipt for the light helper."""
    return _trusted_runtime_receipt(require_hardware_token=False)


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
    """Capture signed hardware skills before model-authored terminal use."""
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
        expected_trusted_skill_hashes: dict[str, str] = {}
        expected_camera_sha256 = expected_hashes.get(_CAMERA_SKILL_PATH)
        if expected_camera_sha256:
            expected_trusted_skill_hashes[_CAMERA_SKILL_PATH] = (
                expected_camera_sha256
            )
        for printer_skill_path in _PRINTER3D_SKILL_PATHS:
            expected_printer_sha256 = expected_hashes.get(printer_skill_path)
            if expected_printer_sha256:
                expected_trusted_skill_hashes[printer_skill_path] = expected_printer_sha256
        expected_plaud_sha256 = expected_hashes.get(_PLAUD_SKILL_PATH)
        if expected_plaud_sha256:
            expected_trusted_skill_hashes[_PLAUD_SKILL_PATH] = expected_plaud_sha256
        expected_smart_home_sha256 = expected_hashes.get(_SMART_HOME_SKILL_PATH)
        if expected_smart_home_sha256:
            expected_trusted_skill_hashes[_SMART_HOME_SKILL_PATH] = expected_smart_home_sha256

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
            skills=tuple(trusted_skills),
        )
    except (OSError, ValueError, PermissionError) as exc:
        logger.warning("signed hardware skill trust unavailable: %s", exc)
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


def _task_text(user_message: Any) -> str:
    """Return bounded user task text for hardware-intent classification."""
    parts: list[str] = []
    seen: set[int] = set()
    total = 0

    def _visit(value: Any, *, key: str = "", depth: int = 0) -> None:
        nonlocal total
        if depth > 5 or total >= 64 * 1024 or value is None:
            return
        if isinstance(value, str):
            remaining = 64 * 1024 - total
            text = value[:remaining]
            total += len(text)
            parts.append(text)
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
    return "\n".join(parts)


def _skill_direct_task_context(
    agent: Any,
    user_message: Any,
    *,
    explicit_skill_slug: str = "",
    tool_execution_allowed: bool = True,
) -> _SkillDirectTaskContext:
    del agent
    task_text = _strip_gateway_model_switch_note(_task_text(user_message))
    normalized = " ".join(task_text.lower().split())
    if not tool_execution_allowed:
        normalized = f"tools:none\n{normalized}"

    normalized_skill_slug = (
        explicit_skill_slug.strip().lstrip("/").lower()
        if isinstance(explicit_skill_slug, str)
        else ""
    )
    camera_transport_selection = normalized_skill_slug == "camsnap"
    printer3d_transport_selection = normalized_skill_slug in {
        "printer3d",
        "printer3d-control",
    }
    plaud_transport_selection = normalized_skill_slug == "plaud-recordings"
    smart_home_transport_selection = normalized_skill_slug == "smart-home-light-control"
    camera_resume_sessions = _camera_resume_sessions_locked(now=time.monotonic())
    camera_resume_key = _current_resume_key()
    camera_continuation = _camera_continuation_intent(normalized)
    camera_resumed = bool(
        not camera_transport_selection
        and camera_continuation
        and camera_resume_key is not None
        and camera_resume_key in camera_resume_sessions
    )
    if camera_resumed and camera_resume_key is not None:
        camera_resume_sessions.move_to_end(camera_resume_key)
    plaud_resume_sessions = _plaud_resume_sessions_locked(now=time.monotonic())
    plaud_resume_key = _current_resume_key()
    plaud_continuation = bool(
        not plaud_transport_selection
        and _PLAUD_CONTINUATION_INTENT_RE.fullmatch(normalized)
        and plaud_resume_key is not None
        and plaud_resume_key in plaud_resume_sessions
    )
    if plaud_continuation and plaud_resume_key is not None:
        plaud_resume_sessions.move_to_end(plaud_resume_key)

    task_binding = (
        f"skill:{normalized_skill_slug}\n{normalized}"
        if (
            camera_transport_selection
            or printer3d_transport_selection
            or plaud_transport_selection
            or smart_home_transport_selection
        )
        else normalized
    )
    hardware_inventory = bool(_HARDWARE_INVENTORY_INTENT_RE.search(normalized))
    camera_authorization_retry = bool(
        _CAMERA_AUTHORIZATION_RETRY_RE.fullmatch(normalized)
    )
    return _SkillDirectTaskContext(
        task_sha256=hashlib.sha256(task_binding.encode("utf-8")).hexdigest(),
        turn_identity=_current_skill_direct_turn_identity(),
        camera_applicable=(
            camera_transport_selection
            or bool(_CAMERA_INTENT_RE.search(normalized))
            or bool(_CAMERA_INVENTORY_INTENT_RE.search(normalized))
            or hardware_inventory
            or camera_authorization_retry
            or bool(_CAMERA_DIRECT_SNAPSHOT_INTENT_RE.fullmatch(normalized))
            or camera_resumed
        ),
        camera_explicit=camera_transport_selection,
        trusted_skill_slug=normalized_skill_slug,
        camera_inventory_only=bool(
            not camera_transport_selection
            and (
                _CAMERA_INVENTORY_INTENT_RE.search(normalized)
                or hardware_inventory
                or camera_authorization_retry
            )
            and not _CAMERA_INTENT_RE.search(normalized)
        ),
        printer3d_applicable=(
            printer3d_transport_selection
            or bool(_PRINTER3D_INTENT_RE.search(normalized))
            or hardware_inventory
        ),
        printer3d_explicit=printer3d_transport_selection,
        plaud_applicable=(
            plaud_transport_selection
            or bool(_PLAUD_INTENT_RE.search(normalized))
            or plaud_continuation
        ),
        plaud_explicit=plaud_transport_selection,
        smart_home_applicable=smart_home_transport_selection,
        smart_home_explicit=smart_home_transport_selection,
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
            # A bounded policy failure is a terminal capability state, not an
            # active scope.  Treating ``policy_exhausted`` as active lets the
            # request skip the fresh attested ``skill_view`` bootstrap and can
            # leak a stale provider tool list into the next retry.
            and bool(scope.allowed_tools)
        )


def _trusted_skill_view_refresh_required(
    agent: Any,
    function_args: Mapping[str, Any],
) -> bool:
    """Require a new signed read only while rebuilding a trusted scope."""
    if (getattr(agent, "platform", "") or "") != "zet_agent":
        return False
    if function_args.get("file_path") not in (None, ""):
        return False
    requested_path = _trusted_skill_path_for_slug(function_args.get("name", ""))
    if not requested_path or trusted_skill_scope_active(agent):
        return False

    task = getattr(agent, "_zet_agent_skill_direct_task", None)
    turn_identity = _current_skill_direct_turn_identity()
    if (
        not isinstance(task, _SkillDirectTaskContext)
        or turn_identity is None
        or task.turn_identity != turn_identity
    ):
        return False

    task_paths = set()
    if task.camera_applicable:
        task_paths.add(_CAMERA_SKILL_PATH)
    if task.printer3d_applicable:
        task_paths.update(_PRINTER3D_SKILL_PATHS)
    if task.plaud_applicable:
        task_paths.add(_PLAUD_SKILL_PATH)
    if task.smart_home_applicable:
        task_paths.add(_SMART_HOME_SKILL_PATH)
    if requested_path not in task_paths:
        return False

    if getattr(agent, "_zet_agent_execution_policy", "") == "silent_automation":
        return (
            _trusted_skill_path_for_slug(task.trusted_skill_slug)
            == requested_path
        )
    return True


def trusted_skill_view_fresh_read_required() -> bool:
    """Expose the dispatch-local signed-read requirement to ``skill_view``."""
    return _TRUSTED_SKILL_VIEW_FRESH_READ.get()


def _activate_execution_policy_tools(
    agent: Any,
    allowed_tools: frozenset[str],
) -> None:
    """Restore only the intersection of policy and attested-skill tools."""
    if getattr(agent, "_zet_agent_execution_policy", "") != "silent_automation":
        return
    policy_tools = list(
        getattr(agent, "_zet_agent_execution_policy_tools", ()) or ()
    )
    policy_names = set(
        getattr(
            agent,
            "_zet_agent_execution_policy_valid_tool_names",
            (),
        )
        or ()
    )

    def _tool_name(tool: Any) -> str:
        if not isinstance(tool, dict):
            return ""
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return str(tool.get("name") or "")

    scoped_names = policy_names & set(allowed_tools)
    agent.tools = [
        copy.deepcopy(tool)
        for tool in policy_tools
        if _tool_name(tool) in scoped_names
    ]
    agent.valid_tool_names = scoped_names


def _activate_trusted_skill_scope(
    agent: Any,
    *,
    relative_path: str,
    attested_turn_identity: _TurnIdentity,
) -> bool:
    """Activate the existing request-bound scope after a trusted byte read."""
    if (getattr(agent, "platform", "") or "") != "zet_agent":
        return False
    if relative_path not in {
        _CAMERA_SKILL_PATH,
        *_PRINTER3D_SKILL_PATHS,
        _PLAUD_SKILL_PATH,
        _SMART_HOME_SKILL_PATH,
    }:
        return False

    task = getattr(agent, "_zet_agent_skill_direct_task", None)
    task_matches_skill = bool(
        isinstance(task, _SkillDirectTaskContext)
        and (
            (
                relative_path == _CAMERA_SKILL_PATH
                and task.camera_applicable
            )
            or (
                relative_path in _PRINTER3D_SKILL_PATHS
                and task.printer3d_applicable
            )
            or (
                relative_path == _PLAUD_SKILL_PATH
                and task.plaud_applicable
            )
            or (
                relative_path == _SMART_HOME_SKILL_PATH
                and task.smart_home_applicable
            )
        )
    )
    if not task_matches_skill:
        logger.warning(
            "zet_agent: trusted skill %s did not match the current user task",
            relative_path,
        )
        return False

    current_turn_identity = _current_skill_direct_turn_identity()
    if (
        current_turn_identity is None
        or task.turn_identity != current_turn_identity
        or attested_turn_identity != current_turn_identity
    ):
        logger.warning(
            "zet_agent: trusted skill %s rejected for mismatched turn identity",
            relative_path,
        )
        return False
    if (
        getattr(agent, "_zet_agent_execution_policy", "")
        == "silent_automation"
        and _trusted_skill_path_for_slug(
            getattr(task, "trusted_skill_slug", "")
        )
        != relative_path
    ):
        logger.warning(
            "zet_agent: silent skill %s does not match the trusted slug %r",
            relative_path,
            getattr(task, "trusted_skill_slug", ""),
        )
        return False

    execution_receipt = _capture_trusted_execution_receipt(
        current_turn_identity,
        relative_path,
    )
    if execution_receipt is None:
        return False
    allowed_tools = (
        _CAMERA_DIRECT_TOOLS
        if relative_path == _CAMERA_SKILL_PATH
        else _PRINTER3D_DIRECT_TOOLS
        if relative_path in _PRINTER3D_SKILL_PATHS
        else _PLAUD_DIRECT_TOOLS
        if relative_path == _PLAUD_SKILL_PATH
        else _SMART_HOME_DIRECT_TOOLS
    )

    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_operation = None
        agent._zet_agent_skill_direct_scope = _SkillDirectScope(
            relative_path=relative_path,
            task_sha256=task.task_sha256,
            turn_identity=current_turn_identity,
            allowed_tools=allowed_tools,
            execution_receipt=execution_receipt,
            camera_inventory_only=bool(
                relative_path == _CAMERA_SKILL_PATH
                and task.camera_inventory_only
            ),
        )
        _activate_execution_policy_tools(agent, allowed_tools)
        if relative_path == _PLAUD_SKILL_PATH:
            _remember_plaud_resume_locked(
                turn_identity=current_turn_identity,
                now=time.monotonic(),
            )
    logger.info(
        "zet_agent: trusted skill %s activated bounded execution scope",
        relative_path,
    )
    return True


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
    # The shared terminal parser also recognizes scheduled semantic helpers.
    # Those use Cron execution identity and device policy authorization, not a
    # Chat camsnap receipt. Never classify them as manual camera operations.
    if Path(argv[1]).name != "camera_connector.py":
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
    # zettlab-overlay(ac432-history): Include scoped history image artifacts; upstream: none
    """Extract the exact trusted camera attachment under the active output root."""
    output = result.get("output")
    if not isinstance(output, str) or len(output.encode("utf-8")) > 1024 * 1024:
        return None
    try:
        payload = json.loads(output)
    except (TypeError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    # zettlab-overlay(ac432-history): Classify camera artifacts in the adapter; upstream: none
    from gateway.platforms.zet_agent_camera_arguments import camera_result_has_image
    if not camera_result_has_image(data):
        return None
    return _trusted_camera_attachment_path(data.get("attachment_path"))


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


def _plaud_runtime_argv(function_args: Mapping[str, Any]) -> list[str] | None:
    command = function_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        from tools.terminal_tool import _parse_plaud_runtime_command

        parsed = _parse_plaud_runtime_command(command)
    except Exception:
        return None
    argv = getattr(parsed, "argv", None)
    if (
        not isinstance(argv, list)
        or len(argv) < 3
        or not all(isinstance(value, str) for value in argv)
    ):
        return None
    return list(argv)


def _plaud_command_policy(function_args: Mapping[str, Any]) -> bool:
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
        return False
    argv = _plaud_runtime_argv(function_args)
    return bool(
        argv is not None and os.path.basename(argv[1]) == "plaud_connector.py"
    )


def _smart_home_runtime_argv(function_args: Mapping[str, Any]) -> list[str] | None:
    command = function_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    try:
        from tools.terminal_tool import _parse_smart_home_runtime_command
        parsed = _parse_smart_home_runtime_command(command)
    except Exception:
        return None
    argv = getattr(parsed, "argv", None)
    if not isinstance(argv, list) or len(argv) < 3 or not all(isinstance(value, str) for value in argv):
        return None
    return list(argv)


def _smart_home_command_policy(function_args: Mapping[str, Any]) -> bool:
    if any(bool(function_args.get(field)) for field in ("background", "force", "notify_on_complete", "pty", "watch_patterns", "workdir")):
        return False
    return _smart_home_runtime_argv(function_args) is not None


def _silent_skill_view_scope_block_message(
    agent: Any,
    function_args: Mapping[str, Any],
) -> str | None:
    """Fail closed before a silent turn can read an unrelated skill."""
    if getattr(agent, "_zet_agent_execution_policy", "") != "silent_automation":
        return None

    task = getattr(agent, "_zet_agent_skill_direct_task", None)
    turn_identity = _current_skill_direct_turn_identity()
    expected_path = (
        _trusted_skill_path_for_slug(task.trusted_skill_slug)
        if isinstance(task, _SkillDirectTaskContext)
        else ""
    )
    requested_path = _trusted_skill_path_for_slug(function_args.get("name", ""))
    if (
        expected_path
        and requested_path == expected_path
        and turn_identity is not None
        and task.turn_identity == turn_identity
    ):
        return None

    logger.warning(
        "zet_agent: blocked silent skill_view outside the transport-selected scope"
    )
    return (
        "Silent automation may load only the transport-selected signed skill "
        "for this request-bound turn. Do not inspect or load another skill."
    )


class CameraTaskScopeMissing(str):
    """String-compatible policy rejection with additive machine-readable facts."""

    code = "camera_task_scope_missing"
    authorization_status = "not_checked"


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
    _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
    if function_name == "skill_view":
        silent_scope_block = _silent_skill_view_scope_block_message(
            agent,
            function_args,
        )
        if silent_scope_block is not None:
            return silent_scope_block
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
                and getattr(agent, "_zet_agent_execution_policy", "")
                == "silent_automation"
            ):
                logger.warning(
                    "zet_agent: blocked silent terminal without an attested skill scope"
                )
                return (
                    "Silent automation terminal access requires a current "
                    "request-bound scope minted by an attested `skill_view` result."
                )
            if (
                function_name == "terminal"
                and _camera_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked camera runtime command without a current "
                    "trusted camsnap scope"
                )
                return CameraTaskScopeMissing(
                    "Trusted camera commands require a current request-bound "
                    "scope minted by the attested `camsnap` skill_view result. "
                    "The command was not dispatched; device and Chat grants "
                    "were not checked. Do not claim that camera permission is "
                    "disabled or has not synced. Load the trusted skill for "
                    "the current camera task. If it cannot establish scope, "
                    "request an explicit camera operation instead of retrying "
                    "the same blocked command."
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
            if (
                function_name == "terminal"
                and _plaud_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked PLAUD runtime command without a current trusted scope"
                )
                return (
                    "Trusted PLAUD commands require a current request-bound scope "
                    "minted by the attested `plaud-recordings` skill. Load that "
                    "trusted skill and retry the exact operation."
                )
            if (
                function_name == "terminal"
                and _smart_home_runtime_argv(function_args) is not None
            ):
                logger.warning(
                    "zet_agent: blocked smart-home runtime command without a current trusted scope"
                )
                return (
                    "Trusted smart-home commands require a current request-bound "
                    "scope minted by the attested `smart-home-light-control` skill. "
                    "Load that trusted skill and retry the exact operation."
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
        authorized_args_sha256 = ""
        # zettlab-overlay(camera-confirmation): validate proposals without granting recording authority; upstream: none
        if allowed and function_name == "clarify" and scope.relative_path == _CAMERA_SKILL_PATH:
            from gateway.platforms.zet_agent_camera_chat_intent import camera_confirmation_allowed
            allowed = not scope.camera_inventory_only and camera_confirmation_allowed(function_args, scope.camera_ids)
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
                elif scope.relative_path == _PLAUD_SKILL_PATH:
                    allowed = _plaud_command_policy(normalized_args)
                elif scope.relative_path == _SMART_HOME_SKILL_PATH:
                    allowed = _smart_home_command_policy(normalized_args)
                else:
                    allowed = False
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
        if not allowed:
            agent._zet_agent_skill_direct_operation = None
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
                    "zet_agent: blocked out-of-policy trusted hardware operation and "
                    "kept one bounded trusted retry"
                )
                return (
                    "The trusted skill execution scope does not authorize "
                    f"`{function_name}` with these arguments. Retry only the "
                    "pinned hardware helper with its documented arguments."
                )
            agent._zet_agent_skill_direct_scope = replace(
                scope,
                allowed_tools=frozenset(),
                policy_exhausted=True,
            )
            logger.warning(
                "zet_agent: exhausted bounded trusted %s corrections before "
                "out-of-policy tool %s",
                "camera" if scope.relative_path == _CAMERA_SKILL_PATH else "printer",
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
            authorized_args_sha256=authorized_args_sha256,
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
                _camera_runtime_argv(function_args) is not None
                or _printer3d_runtime_argv(function_args) is not None
                or _plaud_runtime_argv(function_args) is not None
                or _smart_home_runtime_argv(function_args) is not None
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
            allowed = False
        elif scope.relative_path == _CAMERA_SKILL_PATH:
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
        elif scope.relative_path == _PLAUD_SKILL_PATH:
            allowed = _plaud_command_policy(normalized_args)
        elif scope.relative_path == _SMART_HOME_SKILL_PATH:
            allowed = _smart_home_command_policy(normalized_args)
        else:
            allowed = False
        receipt = scope.execution_receipt
        if (
            not allowed
            or not final_digest
            or final_digest != operation.authorized_args_sha256
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
    _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
    if function_name == "skill_view":
        silent_scope_block = _silent_skill_view_scope_block_message(
            agent,
            function_args,
        )
        if silent_scope_block is not None:
            return json.dumps(
                {
                    "success": False,
                    "error": silent_scope_block,
                    "trusted_skill_scope_blocked": True,
                },
                ensure_ascii=False,
            )
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
                "trusted_runtime_direct": False,
                "trusted_runtime_blocked": True,
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
        _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(receipt)
    fresh_read_required = (
        function_name == "skill_view"
        and _trusted_skill_view_refresh_required(agent, function_args)
    )
    if fresh_read_required:
        logger.info(
            "zet_agent: refreshing signed trusted skill view for a new scope"
        )
    fresh_read_token = _TRUSTED_SKILL_VIEW_FRESH_READ.set(fresh_read_required)
    try:
        result = dispatch()
    except BaseException:
        _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
        if function_name != "skill_view":
            apply_trusted_skill_execution(
                agent,
                function_name=function_name,
                function_result=None,
            )
        raise
    finally:
        _TRUSTED_SKILL_VIEW_FRESH_READ.reset(fresh_read_token)
        _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)

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
    _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
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
        successful = False
        if function_name == "todo":
            successful = True
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
            elif scope.relative_path == _PLAUD_SKILL_PATH:
                runtime_direct_field = "plaud_runtime_direct"
                runtime_blocked_field = "plaud_runtime_blocked"
            elif scope.relative_path == _SMART_HOME_SKILL_PATH:
                runtime_direct_field = "smart_home_runtime_direct"
                runtime_blocked_field = "smart_home_runtime_blocked"
            else:
                return False
            successful = bool(
                isinstance(result, dict)
                and result.get(runtime_direct_field) is True
                and result.get(runtime_blocked_field) is not True
                and isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code == 0
                and not result.get("error")
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
            scope = replace(
                scope,
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
                policy_violation_retries=_TRUSTED_POLICY_VIOLATION_RETRIES,
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


def _private_hardware_discovery_scope(user_message: Any) -> str:
    """Return one normalized RFC1918 /24-/30 scope from the user message."""
    task_text = _task_text(user_message)
    matches = [
        (match, False) for match in _IPV4_SCOPE_RE.finditer(task_text)
    ] + [
        (match, True) for match in _IPV4_WILDCARD_SCOPE_RE.finditer(task_text)
    ]
    if len(matches) != 1:
        return ""
    match, wildcard = matches[0]
    prefix = match.group("prefix") or "24"
    if wildcard and prefix != "24":
        return ""
    address = f"{match.group('base')}0" if wildcard else match.group("address")
    try:
        network = ipaddress.ip_network(
            f"{address}/{prefix}",
            strict=False,
        )
    except ValueError:
        return ""
    if (
        network.version != 4
        or network.prefixlen < 24
        or network.prefixlen > 30
        or not any(network.subnet_of(private) for private in _RFC1918_NETWORKS)
    ):
        return ""
    return str(network)


def _has_explicit_hardware_discovery_scope(task_text: str) -> bool:
    return bool(
        _IPV4_SCOPE_RE.search(task_text)
        or _IPV4_WILDCARD_SCOPE_RE.search(task_text)
    )


def _subnet_hardware_discovery_request(
    user_message: Any,
) -> tuple[tuple[str, ...], str, bool] | None:
    """Recognize one explicit, bounded private-network discovery request."""
    task_text = _task_text(user_message)
    normalized = " ".join(_strip_gateway_model_switch_note(task_text).split())
    if (
        not normalized
        or len(normalized) > 320
        or not _SUBNET_DISCOVERY_ACTION_RE.search(normalized)
        or _HARDWARE_ENROLLMENT_META_OR_DIAG_RE.search(normalized)
    ):
        return None

    network_scope = _private_hardware_discovery_scope(user_message)
    if _has_explicit_hardware_discovery_scope(normalized) and not network_scope:
        return None
    current_network = not network_scope and bool(
        _CURRENT_PRIVATE_NETWORK_RE.search(normalized)
        or _LOCAL_PRIVATE_NETWORK_RE.search(normalized)
    )
    if not network_scope and not current_network:
        return None
    if (
        _SUBNET_UNSUPPORTED_TYPE_RE.search(normalized)
        or _SUBNET_NON_HARDWARE_SCAN_RE.search(normalized)
    ):
        return None

    positioned_types: list[tuple[int, str]] = []
    for hardware_type, pattern in _HARDWARE_ENROLLMENT_TYPE_RES:
        match = pattern.search(normalized)
        if match is not None:
            positioned_types.append((match.start(), hardware_type))
    positioned_types.sort(key=lambda item: item[0])
    requested_types = tuple(hardware_type for _, hardware_type in positioned_types)
    if not requested_types:
        requested_types = ("camera", "tv")
    if any(
        hardware_type not in _DISCOVERABLE_SUBNET_HARDWARE_TYPES
        for hardware_type in requested_types
    ):
        return None
    return requested_types, network_scope, current_network


def _blocked_subnet_hardware_discovery_request(user_message: Any) -> bool:
    """Return whether a scan-shaped request must stop before Agent tools."""
    task_text = _task_text(user_message)
    normalized = " ".join(_strip_gateway_model_switch_note(task_text).split())
    if (
        not normalized
        or len(normalized) > 320
        or not _SUBNET_DISCOVERY_ACTION_RE.search(normalized)
        or _HARDWARE_ENROLLMENT_META_OR_DIAG_RE.search(normalized)
        or not (
            _has_explicit_hardware_discovery_scope(normalized)
            or _CURRENT_PRIVATE_NETWORK_RE.search(normalized)
            or _LOCAL_PRIVATE_NETWORK_RE.search(normalized)
        )
    ):
        return False
    if (
        (
            _has_explicit_hardware_discovery_scope(normalized)
            and not _private_hardware_discovery_scope(user_message)
        )
        or _SUBNET_UNSUPPORTED_TYPE_RE.search(normalized)
        or _SUBNET_NON_HARDWARE_SCAN_RE.search(normalized)
    ):
        return True
    return any(
        pattern.search(normalized)
        and hardware_type not in _DISCOVERABLE_SUBNET_HARDWARE_TYPES
        for hardware_type, pattern in _HARDWARE_ENROLLMENT_TYPE_RES
    )


def _hardware_enrollment_requested_types(
    user_message: Any,
    *,
    subnet_scoped: bool = False,
) -> tuple[str, ...]:
    """Recognize only short, direct hardware enrollment requests.

    Skill selection remains the primary semantic path. This bounded classifier
    is a delivery guard for models that answer with settings prose instead of
    the credential-free client intent promised by ``connector-setup``.
    """
    task_text = _task_text(user_message)
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
    # zettlab-overlay(connector-enrollment-guard): preserve explicit connect intent; upstream: none
    # Diagnostic/status requests may mention a noun phrase such as “设备连接”
    # or “摄像头权限”. Those are not a request to enroll a connector. Keep an
    # explicit connection verb actionable, but reject diagnostic-only turns.
    if (
        _HARDWARE_ENROLLMENT_DIAGNOSTIC_RE.search(normalized)
        and not _HARDWARE_ENROLLMENT_ACTION_RE.search(action_text)
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
        if subnet_scoped:
            return ("camera", "tv")
        return ("camera", "printer3d", "pc_node", "tv")
    return ()


def _strip_model_hardware_enrollment_blocks(text: str) -> str:
    """Remove hardware-only setup blocks while preserving protocol intents."""
    stripped = _HARDWARE_ENROLLMENT_BLOCK_RE.sub("", text)
    stripped = _LEGACY_HARDWARE_CAMERA_INTENT_BLOCK_RE.sub("", stripped)

    def remove_hardware_v2(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group("payload"))
        except (TypeError, ValueError):
            return match.group(0)
        if not isinstance(payload, dict) or payload.get("kind") != "connector_enrollment":
            return match.group(0)
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return match.group(0)
        known_hardware = {"camera", "printer3d", "pc_node", "tv", "voice_terminal"}
        if not all(
            isinstance(item, dict) and item.get("resource_kind") in known_hardware
            for item in items
        ):
            return match.group(0)
        return ""

    return _CONNECTOR_ENROLLMENT_BLOCK_RE.sub(remove_hardware_v2, stripped)


def _append_hardware_enrollment_intent(
    visible_text: str,
    requested_types: tuple[str, ...],
    *,
    network_scope: str = "",
    current_network: bool = False,
) -> str:
    intent: dict[str, Any] = {
        "schema_version": "2",
        "kind": "connector_enrollment",
        "items": [
            {"resource_kind": hardware_type}
            for hardware_type in requested_types
        ],
        "setup_requested": True,
    }
    if network_scope:
        intent["network_scope"] = {"cidr": network_scope}
    elif current_network:
        intent["network_scope"] = {"mode": "current"}
    payload = json.dumps(intent, ensure_ascii=False, indent=2)
    return (
        f"{visible_text}\n\n```{_CONNECTOR_ENROLLMENT_FENCE}\n"
        f"{payload}\n```"
    )


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
    """Append one canonical, secret-free connector enrollment intent when needed.

    The transform never discovers hardware or accepts addresses/credentials. It
    only gives first-party App/Web clients enough information to render the
    explicit user-confirmation card. Existing model-authored intent blocks are
    replaced so malformed or credential-bearing fields cannot suppress the
    trusted canonical card.
    """
    text = str(response_text or "")
    if getattr(agent, "_zettlab_connector_direct_input", False):
        # Composer-capable clients use the trusted clarify handoff. Do not
        # append a second legacy enrollment card after that interaction.
        return _strip_model_hardware_enrollment_blocks(text).strip()
    if (
        (getattr(agent, "platform", "") or "") != "zet_agent"
        or not completed
        or failed
        or interrupted
        or structured_output
        or not text.strip()
    ):
        return text
    visible_text = _strip_model_hardware_enrollment_blocks(text)
    removed_model_intent = visible_text != text
    if removed_model_intent:
        visible_text = visible_text.strip()
    if _blocked_subnet_hardware_discovery_request(user_message):
        return visible_text if removed_model_intent else text
    task_text = _task_text(user_message)
    subnet_request = _subnet_hardware_discovery_request(user_message)
    if subnet_request is not None:
        requested_types, network_scope, current_network = subnet_request
    else:
        network_scope = _private_hardware_discovery_scope(user_message)
        current_network = not network_scope and bool(
            _CURRENT_PRIVATE_NETWORK_RE.search(task_text)
        )
        requested_types = _hardware_enrollment_requested_types(
            user_message,
            subnet_scoped=bool(network_scope) or current_network,
        )
    if _has_explicit_hardware_discovery_scope(task_text) and not network_scope:
        # An invalid, public, oversized or ambiguous range must not degrade to
        # broad unscoped discovery.
        return visible_text if removed_model_intent else text
    if requested_types and _CONNECTOR_ENROLLMENT_BLOCK_RE.search(visible_text):
        # A mixed hardware + protocol V2 intent is outside this hardware-only
        # canonicalizer. Preserve that single V2 block and suppress the legacy
        # fallback instead of dropping protocol items or producing two cards.
        return visible_text if removed_model_intent else text
    if not requested_types:
        # A model-authored setup block is not authoritative. Remove stale or
        # over-eager hardware cards from status/usage turns even when no new
        # canonical enrollment intent needs to be appended.
        return visible_text if removed_model_intent else text

    if (network_scope or current_network) and any(
        hardware_type not in _DISCOVERABLE_SUBNET_HARDWARE_TYPES
        for hardware_type in requested_types
    ):
        # Do not silently convert a printer/PC/pairing request into a camera/TV
        # scan. The model can explain that those types use trusted manual or
        # account/pairing discovery instead.
        return visible_text if removed_model_intent else text

    return _append_hardware_enrollment_intent(
        visible_text,
        requested_types,
        network_scope=network_scope,
        current_network=current_network,
    )


def hardware_enrollment_preflight_response(
    agent: Any,
    user_message: Any,
) -> str:
    """Return an immediate client card before the provider/tool loop."""
    if (getattr(agent, "platform", "") or "") != "zet_agent":
        return ""
    subnet_request = _subnet_hardware_discovery_request(user_message)
    if subnet_request is None:
        if _blocked_subnet_hardware_discovery_request(user_message):
            task_text = _task_text(user_message)
            if re.search(r"[\u3400-\u9fff]", task_text):
                return (
                    "该请求不会执行 shell、nmap 或端口扫描。请使用私有 /24–/30 "
                    "网段；当前网段发现仅支持 ONVIF 摄像头和 DLNA 电视。"
                )
            return (
                "This request will not run shell, nmap, or a port scan. Use a "
                "private /24–/30 subnet; discovery currently supports ONVIF "
                "cameras and DLNA TVs."
            )
        return ""
    if response_format_requires_structured_output(
        (getattr(agent, "request_overrides", None) or {}).get("response_format")
    ):
        return ""
    requested_types, network_scope, current_network = subnet_request
    task_text = _task_text(user_message)
    is_chinese = bool(re.search(r"[\u3400-\u9fff]", task_text))
    intro = (
        "点击卡片中的“发现附近设备”开始扫描；结果会显示在卡片内，且不会自动连接。"
        if is_chinese
        else (
            "Select “Discover nearby devices” to scan. Results stay in the "
            "card and are not connected automatically."
        )
    )
    return _append_hardware_enrollment_intent(
        intro,
        requested_types,
        network_scope=network_scope,
        current_network=current_network,
    )


def reset_trusted_skill_execution(
    agent: Any,
    user_message: Any = None,
    *,
    explicit_skill_slug: str = "",
    tool_execution_allowed: bool = True,
) -> None:
    """Clear trusted execution and bind eligibility to the new user task."""
    _TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_scope = None
        agent._zet_agent_skill_direct_operation = None
        task = _skill_direct_task_context(
            agent,
            user_message,
            explicit_skill_slug=explicit_skill_slug,
            tool_execution_allowed=tool_execution_allowed,
        )
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

    return _activate_trusted_skill_scope(
        agent,
        relative_path=pending.relative_path,
        attested_turn_identity=pending.turn_identity,
    )


# Production gateways set ZETTLAB_PRESETS_DIR in the process environment before
# Python imports the agent. Capture before any model-authored terminal command.
_TRUSTED_PRESETS_SNAPSHOT = _capture_trusted_presets_snapshot()
