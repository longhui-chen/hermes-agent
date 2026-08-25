"""Small, durable, profile-scoped workflow store for video edits."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from plugins.video_edit.paths import safe_id, state_path

MAX_WORKFLOWS = 64
WORKFLOW_TTL_SECONDS = 90 * 24 * 60 * 60
MAX_FILES = 8
MAX_WORKFLOW_BYTES = 2 * 1024 * 1024


class WorkflowError(ValueError):
    pass


def workflow_id(task_id: str, agent_id: str) -> str:
    digest = hashlib.sha256(f"{agent_id}\x00{task_id}".encode()).hexdigest()[:24]
    return f"vew_{digest}"


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
