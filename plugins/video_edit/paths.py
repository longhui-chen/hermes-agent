"""Profile-scoped paths and bounded filesystem validation for video editing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RAW_ROOTS = ("/volume1/subvol/data", "/volume1/data")
_MAX_PATH_BYTES = 4096


class VideoPathError(ValueError):
    pass


def safe_id(value: Any, *, fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text or not _PROFILE_RE.fullmatch(text):
        return fallback
    return text


def agent_id_from_kwargs(kwargs: dict | None = None) -> str:
    kwargs = kwargs or {}
    for key in ("agent_id", "profile_id", "agent"):
        value = kwargs.get(key)
        if isinstance(value, str) and _PROFILE_RE.fullmatch(value.strip()):
            return value.strip()
    try:
        from agent.secret_scope import get_secret

        for key in ("ZET_AGENT_ID", "ZETTLAB_AGENT_ID", "AGENT_ID"):
            value = str(get_secret(key, "") or "").strip()
            if _PROFILE_RE.fullmatch(value):
                return value
    except Exception:
        pass
    try:
        from hermes_cli.profiles import get_active_profile_name

        return safe_id(get_active_profile_name())
    except Exception:
        return "default"


def task_id_from_kwargs(kwargs: dict | None = None) -> str:
    kwargs = kwargs or {}
    for key in ("task_id", "turn_id", "session_id"):
        value = str(kwargs.get(key) or "").strip()
        if value:
            return value[:256]
    try:
        from gateway.session_context import zettlab_turn_id

        value = zettlab_turn_id()
        if value:
            return value[:256]
    except Exception:
        pass
    return "interactive"


def state_root() -> Path:
    raw_home = Path(get_hermes_home())
    if raw_home.is_symlink():
        raise VideoPathError("Hermes state root is a symlink")
    root = raw_home.resolve() / "video_edit"
    if root.exists() and root.is_symlink():
        raise VideoPathError("video edit state root is a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise VideoPathError("video edit state root is a symlink")
    return root


def state_path(name: str, agent_id: str) -> Path:
    if not re.fullmatch(r"[a-z_]{1,48}\.json", name):
        raise VideoPathError("invalid video edit state file")
    profile_root = state_root() / safe_id(agent_id)
    if profile_root.is_symlink():
        raise VideoPathError("video edit profile state root is a symlink")
    profile_root.mkdir(mode=0o700, parents=False, exist_ok=True)
    return profile_root / name


def _under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def validate_input_file(raw: str, agent_id: str) -> Path:
    if not isinstance(raw, str) or len(raw.encode()) > _MAX_PATH_BYTES:
        raise VideoPathError("input path is invalid")
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        raise VideoPathError("input path must be absolute")
    try:
        if candidate.is_symlink():
            raise VideoPathError("input symlink is not allowed")
        resolved = candidate.resolve(strict=True)
        stat_result = resolved.stat()
    except (OSError, ValueError) as exc:
        raise VideoPathError("input file is unavailable") from exc
    if not resolved.is_file() or stat_result.st_size <= 0:
        raise VideoPathError("input file is unavailable")
    raw_ok = any(_under(resolved, Path(root).resolve()) for root in _RAW_ROOTS)
    output = output_root(agent_id)
    if not raw_ok and not _under(resolved, output):
        raise VideoPathError("input path is outside the media workspace")
    return resolved


def output_root(agent_id: str) -> Path:
    try:
        from tools.runtime_workdir import agent_output_dir

        raw = agent_output_dir()
    except Exception:
        raw = None
    if not raw:
        raw = os.environ.get("ZET_AGENT_OUTPUT_DIR", "")
    if not raw or not os.path.isabs(raw):
        raise VideoPathError("agent output directory is unavailable")
    raw_root = Path(raw)
    if raw_root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    root = raw_root.resolve()
    if root.exists() and root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    if root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    # agent-specific buckets keep concurrent profiles from sharing artifacts.
    bucket = root / safe_id(agent_id)
    bucket.mkdir(mode=0o750, parents=True, exist_ok=True)
    return bucket


def result_path(
    agent_id: str,
    filename: str,
    *,
    allow_existing: bool = False,
    session_id: str = "",
) -> Path:
    name = Path(str(filename or "video-edit.mp4").strip()).name
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
        raise VideoPathError("invalid result filename")
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    root = output_root(agent_id)
    if session_id:
        bucket = safe_id(session_id, fallback="")
        if not bucket:
            raise VideoPathError("result session is invalid")
        root = root / bucket
        root.mkdir(mode=0o750, parents=False, exist_ok=True)
        if root.is_symlink():
            raise VideoPathError("result session directory is a symlink")
    candidate = root / name
    if candidate.is_symlink():
        raise VideoPathError("result path is unavailable")
    target = candidate.resolve()
    if not _under(target, root):
        raise VideoPathError("result path is unavailable")
    if target.exists() and (not allow_existing or not target.is_file()):
        raise VideoPathError("result path is unavailable")
    return target
