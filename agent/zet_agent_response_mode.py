"""Bind trusted video-edit skill execution to an exact App turn."""

from __future__ import annotations

import base64
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
from typing import Any, Mapping

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
_VIDEO_EDIT_POLICY_VIOLATION_RETRIES = 2
_VIDEO_EDIT_RESUME_TTL_SECONDS = 3 * 60 * 60
_VIDEO_EDIT_RESUME_MAX_SESSIONS = 8
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
_VIDEO_EDIT_SLASH_RE = re.compile(
    r"^/video-edit-workflow-mini(?:\s+.{0,192})?$",
    re.IGNORECASE | re.DOTALL,
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
    command_format_retries: int = 1
    policy_violation_retries: int = _VIDEO_EDIT_POLICY_VIOLATION_RETRIES
    policy_exhausted: bool = False


@dataclass(frozen=True)
class _SkillDirectOperation:
    scope: _SkillDirectScope
    function_name: str
    may_authorize_memory: bool = False


_TRUSTED_PRESETS_SNAPSHOT: _TrustedPresetsSnapshot | None = None
_PENDING_ATTESTATIONS: OrderedDict[str, _PendingSkillAttestation] = OrderedDict()
_VIDEO_EDIT_RESUME_SESSIONS: OrderedDict[str, float] = OrderedDict()
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


def _video_edit_continuation_intent(normalized: str) -> bool:
    if not normalized or len(normalized) > 160:
        return False
    return bool(
        _VIDEO_EDIT_CONTINUATION_CN_RE.fullmatch(normalized)
        or _VIDEO_EDIT_CONTINUATION_EN_RE.fullmatch(normalized)
    )


def _video_edit_slash_intent(normalized: str) -> bool:
    if not normalized or len(normalized) > 224:
        return False
    return bool(_VIDEO_EDIT_SLASH_RE.fullmatch(normalized))


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
) -> OrderedDict[str, float]:
    expired = [
        session_id
        for session_id, expires_at in _VIDEO_EDIT_RESUME_SESSIONS.items()
        if not isinstance(expires_at, (int, float)) or expires_at <= now
    ]
    for session_id in expired:
        _VIDEO_EDIT_RESUME_SESSIONS.pop(session_id, None)
    while len(_VIDEO_EDIT_RESUME_SESSIONS) > _VIDEO_EDIT_RESUME_MAX_SESSIONS:
        _VIDEO_EDIT_RESUME_SESSIONS.popitem(last=False)
    return _VIDEO_EDIT_RESUME_SESSIONS


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
            if relative_path != _VIDEO_EDIT_SKILL_PATH:
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
                if raw_sha256 != expected_video_edit_sha256:
                    raise PermissionError(
                        "official video-edit skill does not match release manifest"
                    )
            except (OSError, UnicodeError, ValueError, PermissionError):
                logger.warning(
                    "ignored unsafe official video-edit skill during startup: %s",
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


def _skill_direct_task_context(agent: Any, user_message: Any) -> _SkillDirectTaskContext:
    task_text, has_video_asset = _task_text_and_video_asset(user_message)
    task_text = _strip_gateway_model_switch_note(task_text)
    normalized = " ".join(task_text.lower().split())
    has_edit_intent = bool(
        _VIDEO_EDIT_CN_RE.search(normalized) or _VIDEO_EDIT_EN_RE.search(normalized)
    )
    explicit = (
        (has_video_asset and has_edit_intent)
        or _video_edit_slash_intent(normalized)
        or _video_edit_direct_command_intent(normalized)
    )
    now = time.monotonic()
    session_id = _current_skill_direct_session_id()
    resume_sessions = _video_edit_resume_sessions_locked(now=now)
    continuation_intent = _video_edit_continuation_intent(normalized)
    resumed = bool(
        not explicit
        and continuation_intent
        and (
            has_edit_intent
            or (session_id and session_id in resume_sessions)
        )
    )
    if resumed and session_id in resume_sessions:
        resume_sessions.move_to_end(session_id)
    return _SkillDirectTaskContext(
        task_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        turn_identity=_current_skill_direct_turn_identity(),
        video_edit_applicable=explicit or resumed,
        video_edit_explicit=explicit,
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
        target = operation.get("target")
        action = operation.get("action")
        if target not in grouped or action not in {"add", "remove", "replace"}:
            return frozenset()
        if set(operation) - {"action", "content", "old_text", "target"}:
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
                "The trusted video-edit execution scope belongs to another "
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
                        "The trusted video-edit operation belongs to another "
                        "task-local turn. The scope was revoked before this "
                        "call; reload the trusted skill for this turn."
                    )
                # A provider violating the one-operation contract must not
                # race a second call past the consumed scope. This also clears
                # a failed/plugin-blocked operation on the next attempted call.
                agent._zet_agent_skill_direct_operation = None
                return (
                    "The trusted video-edit execution scope already has an "
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
            return None

        if scope.policy_exhausted:
            return (
                "The trusted video-edit workflow exhausted its bounded command "
                "corrections. Do not call more trusted helpers; return a concise "
                "product error."
            )

        allowed = function_name in scope.allowed_tools
        operation_scope = scope
        may_authorize_memory = False
        if allowed and function_name == "terminal":
            allowed, may_authorize_memory = _video_edit_command_policy(function_args)
            allowed = allowed and scope.execution_receipt is not None
        elif allowed and function_name == "memory":
            memory_digest = _canonical_memory_payload_sha256(function_args)
            allowed = bool(memory_digest and memory_digest in scope.memory_payload_sha256)
            if allowed:
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
                logger.warning(
                    "zet_agent: blocked out-of-policy trusted video operation and "
                    "kept one bounded trusted retry"
                )
                return (
                    "The trusted video-edit execution scope does not authorize "
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
                "zet_agent: exhausted bounded trusted video corrections before "
                "out-of-policy tool %s",
                function_name or "<missing>",
            )
            return (
                "The trusted video-edit operation was rejected after bounded "
                "corrections. Do not call more trusted helpers; return a concise "
                "product error."
            )

        agent._zet_agent_skill_direct_scope = None
        agent._zet_agent_skill_direct_operation = _SkillDirectOperation(
            scope=operation_scope,
            function_name=function_name,
            may_authorize_memory=may_authorize_memory,
        )
        if function_name == "terminal":
            _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(scope.execution_receipt)
        return None


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
            successful = bool(
                isinstance(result, dict)
                and result.get("video_edit_runtime_direct") is True
                and result.get("video_edit_runtime_blocked") is not True
                and isinstance(exit_code, int)
                and not isinstance(exit_code, bool)
                and exit_code == 0
                and not result.get("error")
            )
            memory_hashes = (
                _memory_payload_hashes_from_terminal_result(result)
                if successful and operation.may_authorize_memory
                else frozenset()
            )
            allowed_tools = scope.allowed_tools - {"memory"}
            if memory_hashes:
                allowed_tools = allowed_tools | {"memory"}
            scope = replace(
                scope,
                allowed_tools=allowed_tools,
                memory_payload_sha256=memory_hashes,
                command_format_retries=1,
                policy_violation_retries=_VIDEO_EDIT_POLICY_VIOLATION_RETRIES,
                policy_exhausted=False,
            )
        if not successful:
            return False

        agent._zet_agent_skill_direct_scope = scope
        return True


def request_response_mode(agent: Any) -> str:
    """Return the only structured response mode owned by the App plan flow."""
    mode = str(getattr(agent, "_zet_agent_response_mode", "") or "").strip().lower()
    return "plan" if mode == "plan" else ""


def reset_trusted_skill_execution(agent: Any, user_message: Any = None) -> None:
    """Clear trusted execution and bind eligibility to the new user task."""
    _TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(None)
    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_scope = None
        agent._zet_agent_skill_direct_operation = None
        task = _skill_direct_task_context(
            agent,
            user_message,
        )
        if task.video_edit_explicit:
            session_id = _current_skill_direct_session_id()
            if session_id:
                sessions = _video_edit_resume_sessions_locked(
                    now=time.monotonic(),
                )
                sessions[session_id] = (
                    time.monotonic() + _VIDEO_EDIT_RESUME_TTL_SECONDS
                )
                sessions.move_to_end(session_id)
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

    if pending.relative_path != _VIDEO_EDIT_SKILL_PATH:
        return False
    task = getattr(agent, "_zet_agent_skill_direct_task", None)
    if not isinstance(task, _SkillDirectTaskContext) or not task.video_edit_applicable:
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

    with _SKILL_DIRECT_LOCK:
        agent._zet_agent_skill_direct_operation = None
        agent._zet_agent_skill_direct_scope = _SkillDirectScope(
            relative_path=pending.relative_path,
            task_sha256=task.task_sha256,
            turn_identity=current_turn_identity,
            allowed_tools=_VIDEO_EDIT_DIRECT_TOOLS,
            execution_receipt=execution_receipt,
        )
    logger.info(
        "zet_agent: trusted skill %s activated bounded execution scope",
        pending.relative_path,
    )
    return True


# Production gateways set ZETTLAB_PRESETS_DIR in the process environment before
# Python imports the agent. Capture before any model-authored terminal command.
_TRUSTED_PRESETS_SNAPSHOT = _capture_trusted_presets_snapshot()
