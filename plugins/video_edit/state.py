"""Small, durable, profile-scoped workflow store for video edits."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from plugins.video_edit.paths import safe_id, state_path

MAX_WORKFLOWS = 64
WORKFLOW_TTL_SECONDS = 90 * 24 * 60 * 60
# Keep one workflow bounded while allowing the plugin to hide provider-sized
# upload batches from the model. Fifty references bound the durable manifest;
# provider count and byte limits split it further without increasing peak prep.
MAX_FILES = 50
MAX_PROACTIVE_FILES = 8
MAX_WORKFLOW_BYTES = 2 * 1024 * 1024
REPORT_LOCK_BUCKETS = 16
UPLOAD_LOCK_BUCKETS = 16


class WorkflowError(ValueError):
    pass


def workflow_id(task_id: str, agent_id: str) -> str:
    digest = hashlib.sha256(f"{agent_id}\x00{task_id}".encode()).hexdigest()[:24]
    return f"vew_{digest}"


@contextlib.contextmanager
def _workflow_operation_lock(
    workflow: str,
    agent_id: str,
    *,
    name: str,
    buckets: int,
) -> Iterator[None]:
    """Lock one fixed per-profile workflow bucket without following links."""

    workflow = str(workflow or "").strip()
    if not workflow or len(workflow) > 128 or not name or buckets < 1:
        raise WorkflowError("invalid video workflow")
    expected_agent_id = safe_id(agent_id)
    profile_root = state_path("workflows.json", expected_agent_id).parent
    digest = hashlib.sha256(workflow.encode()).digest()
    bucket = int.from_bytes(digest[:2], "big") % buckets
    lock_path = profile_root / f".{name}-{bucket:02d}.lock"
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if os.name != "posix" or no_follow is None:
        raise WorkflowError("video workflow lock is unavailable")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    flags |= no_follow
    fd = -1
    try:
        fd = os.open(lock_path, flags, 0o600)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise WorkflowError("video workflow lock is unavailable")
        os.fchmod(fd, 0o600)
    except (OSError, WorkflowError) as exc:
        if fd >= 0:
            os.close(fd)
        raise WorkflowError("video workflow lock is unavailable") from exc
    with os.fdopen(fd, "r+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise WorkflowError("video workflow lock is unavailable") from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def report_lock(workflow: str, agent_id: str) -> Iterator[None]:
    """Serialize report delivery for one bounded workflow hash bucket."""

    with _workflow_operation_lock(
        workflow,
        agent_id,
        name="report",
        buckets=REPORT_LOCK_BUCKETS,
    ):
        yield


@contextlib.contextmanager
def upload_lock(workflow: str, agent_id: str) -> Iterator[None]:
    """Serialize media preparation and upload for one workflow hash bucket."""

    with _workflow_operation_lock(
        workflow,
        agent_id,
        name="upload",
        buckets=UPLOAD_LOCK_BUCKETS,
    ):
        yield


def _empty() -> dict[str, Any]:
    return {"version": 1, "updated_at": int(time.time()), "workflows": {}}


def _read(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_WORKFLOW_BYTES:
            raise WorkflowError("video workflow state file is too large")
    except FileNotFoundError:
        return _empty()
    except OSError as exc:
        raise WorkflowError("video workflow state is unreadable") from exc
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError) as exc:
        raise WorkflowError("video workflow state is unreadable") from exc
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("workflows"), dict):
        raise WorkflowError("video workflow state is invalid")
    return data


def _prune(data: dict[str, Any]) -> None:
    now = int(time.time())
    workflows = data["workflows"]
    stale = []
    for key, value in workflows.items():
        if not isinstance(value, dict) or now - int(value.get("updated_at", 0) or 0) > WORKFLOW_TTL_SECONDS:
            stale.append(key)
    for key in stale:
        workflows.pop(key, None)
    if len(workflows) > MAX_WORKFLOWS:
        ordered = sorted(workflows.items(), key=lambda item: int(item[1].get("updated_at", 0) or 0))
        for key, _ in ordered[: len(workflows) - MAX_WORKFLOWS]:
            workflows.pop(key, None)


def _write(path: Path, data: dict[str, Any]) -> None:
    data["updated_at"] = int(time.time())
    _prune(data)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > 2 * 1024 * 1024:
        raise WorkflowError("video workflow state exceeds size limit")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _locked(agent_id: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    path = state_path("workflows.json", agent_id)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(mode=0o600, exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        data = _read(path)
        _prune(data)
        try:
            yield path, data
        finally:
            _write(path, data)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def get(workflow: str, agent_id: str) -> dict[str, Any] | None:
    workflow = str(workflow or "").strip()
    if not workflow:
        return None
    expected_agent_id = safe_id(agent_id)
    with _locked(expected_agent_id) as (_, data):
        value = data["workflows"].get(workflow)
        if not isinstance(value, dict):
            return None
        if value.get("agent_id") != expected_agent_id:
            raise WorkflowError("video workflow owner is invalid")
        return dict(value)


def update(workflow: str, agent_id: str, patch: dict[str, Any], *, create: bool = True) -> dict[str, Any]:
    workflow = str(workflow or "").strip()
    if not workflow or len(workflow) > 128 or not isinstance(patch, dict):
        raise WorkflowError("invalid video workflow")
    expected_agent_id = safe_id(agent_id)
    with _locked(expected_agent_id) as (_, data):
        current = data["workflows"].get(workflow)
        if current is None:
            if not create:
                raise WorkflowError("video workflow not found")
            current = {
                "workflow_id": workflow,
                "agent_id": expected_agent_id,
                "created_at": int(time.time()),
            }
        if not isinstance(current, dict):
            raise WorkflowError("video workflow is invalid")
        if current.get("agent_id") != expected_agent_id:
            raise WorkflowError("video workflow owner is invalid")
        current.update(patch)
        current["workflow_id"] = workflow
        current["agent_id"] = expected_agent_id
        current["updated_at"] = int(time.time())
        data["workflows"][workflow] = current
        return dict(current)


def create_or_validate_identity(
    workflow: str,
    agent_id: str,
    identity: dict[str, Any],
    initial: dict[str, Any],
    *,
    legacy_identity: dict[str, Any] | None = None,
    legacy_requires_non_proactive: bool = False,
) -> dict[str, Any]:
    """Create once, validate identity, or atomically adopt a proven legacy entry."""

    workflow = str(workflow or "").strip()
    if (
        not workflow
        or len(workflow) > 128
        or not isinstance(identity, dict)
        or not identity
        or not isinstance(initial, dict)
        or set(identity).intersection(initial)
        or (
            legacy_identity is not None
            and (
                not isinstance(legacy_identity, dict)
                or not legacy_identity
                or set(identity).intersection(legacy_identity)
            )
        )
        or not isinstance(legacy_requires_non_proactive, bool)
    ):
        raise WorkflowError("invalid video workflow identity")
    expected_agent_id = safe_id(agent_id)
    with _locked(expected_agent_id) as (_, data):
        current = data["workflows"].get(workflow)
        if current is None:
            current = {
                "workflow_id": workflow,
                "agent_id": expected_agent_id,
                "created_at": int(time.time()),
                **identity,
                **initial,
            }
        if not isinstance(current, dict):
            raise WorkflowError("video workflow is invalid")
        if current.get("agent_id") != expected_agent_id:
            raise WorkflowError("video workflow owner is invalid")
        if any(current.get(key) != value for key, value in identity.items()):
            if (
                legacy_identity is None
                or any(key in current for key in identity)
                or any(
                    current.get(key) != value
                    for key, value in legacy_identity.items()
                )
                or (
                    legacy_requires_non_proactive
                    and current.get("proactive") not in (None, False)
                )
            ):
                raise WorkflowError("video workflow identity changed")
            # Older interactive checkpoints predate the immutable request
            # marker. Only bind one after their existing durable values prove
            # the same request; preserve every upload/project field verbatim.
            current.update(identity)
        current["workflow_id"] = workflow
        current["agent_id"] = expected_agent_id
        current["updated_at"] = int(time.time())
        data["workflows"][workflow] = current
        return dict(current)
