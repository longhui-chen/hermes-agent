"""Persistent, bounded desktop-pet creation lifecycle for Zet Agent."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

_TASK_TTL_SECONDS = 60 * 60
_MAX_TASKS = 16
_MAX_INFLIGHT_TASKS = 4
_MAX_ACTIVE_CANDIDATES = 3
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16_000_000
_MAX_EXPORT_BYTES = 32 * 1024 * 1024
_MAX_MANIFEST_BYTES = 256 * 1024
_MAX_CONCEPT_CHARS = 2_000
_MAX_STYLE_CHARS = 120
_MAX_INSTRUCTION_CHARS = 1_000
_MAX_NAME_CHARS = 120
_MAX_DESCRIPTION_CHARS = 1_000
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_CANDIDATE_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_FORMAT_METADATA = {
    "kind": "desktop-pet-package",
    "format": "petdex-8x9",
    "format_version": 1,
    "columns": 8,
    "rows": 9,
    "frame_width": 192,
    "frame_height": 208,
    "width": 1536,
    "height": 1872,
}

_state_lock = threading.RLock()
_inflight: dict[str, threading.Event] = {}


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _ok(action: str, **payload: Any) -> str:
    return _json({"success": True, "action": action, **payload})


def _error(action: str, message: str, *, code: str = "invalid_request") -> str:
    return _json({"success": False, "action": action, "error": message, "code": code})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_text(value: Any, *, label: str, maximum: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ValueError(f"{label} is required")
    if len(text) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return text


def _scope(task_id: str | None, session_id: str | None) -> tuple[str, str] | None:
    task = str(task_id or "").strip()
    session = str(session_id or "").strip()
    return (task, session) if task and session else None


def _session_short_id(session_id: str) -> str:
    candidate = str(session_id or "").strip().rsplit(":", 1)[-1]
    return candidate if _SAFE_ID_RE.fullmatch(candidate) else ""


def _session_agent_id(session_id: str) -> str:
    parts = str(session_id or "").strip().split(":")
    if len(parts) < 4:
        return ""
    candidate = parts[-2].strip()
    return candidate if _SAFE_ID_RE.fullmatch(candidate) else ""


def _scoped_env(name: str) -> str:
    from agent.secret_scope import get_secret

    return str(get_secret(name, "") or "").strip()


def _agent_id(session_id: str) -> str:
    scoped = _scoped_env("ZET_AGENT_ID")
    contextual = _session_agent_id(session_id)
    if scoped and not _SAFE_ID_RE.fullmatch(scoped):
        scoped = ""
    if scoped and contextual and scoped.casefold() != contextual.casefold():
        return ""
    return scoped or contextual


def _task_root() -> Path:
    root = get_hermes_home() / "cache" / "desktop-pet-creator"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("desktop-pet task storage is unavailable")
    return root.resolve(strict=True)


def _task_dir(token: str) -> Path:
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError("task token is invalid or expired")
    root = _task_root()
    candidate = root / token
    if candidate.is_symlink():
        raise ValueError("task token is invalid or expired")
    resolved = candidate.resolve(strict=False)
    if resolved.parent != root:
        raise ValueError("task token is invalid or expired")
    return resolved


def _manifest_path(token: str) -> Path:
    return _task_dir(token) / "manifest.json"


def _write_manifest(manifest: dict[str, Any]) -> None:
    token = str(manifest.get("token") or "")
    directory = _task_dir(token)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("desktop-pet task storage is unavailable")
    manifest["updated_epoch"] = time.time()
    manifest["updated_at"] = _now_iso()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise ValueError("desktop-pet manifest exceeds maximum size")
    destination = directory / "manifest.json"
    partial = directory / "manifest.json.part"
    partial.unlink(missing_ok=True)
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _load_manifest(token: str, *, task_id: str, session_id: str) -> dict[str, Any]:
    path = _manifest_path(token)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("task token is invalid or expired")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("task token is invalid or expired") from exc
    if not isinstance(manifest, dict):
        raise ValueError("task token is invalid or expired")
    if not (
        hmac.compare_digest(str(manifest.get("task_id") or ""), task_id)
        and hmac.compare_digest(str(manifest.get("session_id") or ""), session_id)
    ):
        raise ValueError("task token is invalid or expired")
    manifest["updated_epoch"] = time.time()
    manifest["updated_at"] = _now_iso()
    return manifest


def _new_manifest(
    *, token: str, task_id: str, session_id: str, concept: str, style: str
) -> dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "token": token,
        "task_id": task_id,
        "session_id": session_id,
        "concept": concept,
        "style": style,
        "status": "drafting",
        "candidates": [],
        "selected_candidate_id": "",
        "created_epoch": now,
        "updated_epoch": now,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "last_error": "",
    }


def _prune_tasks() -> None:
    root = _task_root()
    with _state_lock:
        active = set(_inflight)
    retained: list[tuple[float, Path]] = []
    cutoff = time.time() - _TASK_TTL_SECONDS
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink() or child.name in active:
            continue
        manifest_path = child / "manifest.json"
        updated = child.stat().st_mtime
        try:
            if manifest_path.is_file() and manifest_path.stat().st_size <= _MAX_MANIFEST_BYTES:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                updated = float(data.get("updated_epoch") or updated)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        if updated < cutoff:
            shutil.rmtree(child, ignore_errors=True)
        else:
            retained.append((updated, child))
    retained.sort(key=lambda item: item[0], reverse=True)
    for _updated, child in retained[_MAX_TASKS:]:
        shutil.rmtree(child, ignore_errors=True)


def _reserve(token: str) -> threading.Event:
    with _state_lock:
        if token in _inflight:
            raise RuntimeError("desktop-pet task is already running")
        if len(_inflight) >= _MAX_INFLIGHT_TASKS:
            raise RuntimeError("too many desktop-pet tasks are running")
        event = threading.Event()
        _inflight[token] = event
        return event


def _release(token: str) -> None:
    with _state_lock:
        _inflight.pop(token, None)


def _cancel_event(token: str) -> threading.Event | None:
    with _state_lock:
        return _inflight.get(token)


def _resolve_output_dir(session_id: str, *, create: bool) -> Path | None:
    agent_id = _agent_id(session_id)
    short_id = _session_short_id(session_id)
    if not agent_id or not short_id:
        return None

    roots: list[Path] = []
    configured = _scoped_env("ZET_AGENT_OUTPUT_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(
        [
            Path("/data/agents/data") / agent_id / "output",
            Path("/volume1/agents/data") / agent_id / "output",
            Path("/volume1/subvol/agents/data") / agent_id / "output",
        ]
    )
    for root in roots:
        try:
            if not root.is_absolute() or root.is_symlink() or not root.is_dir():
                continue
            resolved_root = root.resolve(strict=True)
            destination = resolved_root / short_id
            if destination.is_symlink():
                continue
            if create:
                destination.mkdir(mode=0o755, exist_ok=True)
            if not destination.is_dir():
                continue
            resolved = destination.resolve(strict=True)
            if resolved.parent == resolved_root:
                return resolved
        except OSError:
            continue
    return None


def _inspect_image(path: Path, *, max_bytes: int = _MAX_IMAGE_BYTES) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("generated image is not a regular file")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError("generated image exceeds maximum size")
    from PIL import Image

    with Image.open(path) as opened:
        width, height = opened.size
        if str(opened.format or "").upper() not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("generated image must be PNG, JPEG, or WebP")
        if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
            raise ValueError("generated image exceeds decoded pixel limit")
        opened.verify()


def _atomic_png_copy(source: Path, destination: Path) -> Path:
    _inspect_image(source)
    if destination.resolve(strict=False).parent != destination.parent.resolve(strict=True):
        raise ValueError("unsafe generated image destination")
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    from PIL import Image

    try:
        with Image.open(source) as opened:
            with partial.open("xb") as handle:
                opened.convert("RGBA").save(handle, format="PNG")
                handle.flush()
                os.fsync(handle.fileno())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    _inspect_image(destination)
    return destination


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "path": candidate["path"],
        "status": candidate["status"],
        "created_at": candidate["created_at"],
    }


def _active_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        return []
    return [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("status") == "ready"
    ][-_MAX_ACTIVE_CANDIDATES:]


def _find_candidate(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError("candidate_id is invalid")
    for candidate in manifest.get("candidates") or []:
        if (
            isinstance(candidate, dict)
            and candidate.get("candidate_id") == candidate_id
            and candidate.get("status") == "ready"
        ):
            return candidate
    raise ValueError("candidate_id does not belong to this task")


def _private_candidate_path(manifest: dict[str, Any], candidate: dict[str, Any]) -> Path:
    root = _task_dir(str(manifest["token"]))
    path = Path(str(candidate.get("private_path") or ""))
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("candidate image is unavailable")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("candidate image is unavailable") from exc
    _inspect_image(resolved)
    return resolved


def _publish_generated_candidates(
    sources: list[Path],
    *,
    manifest: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    token = str(manifest["token"])
    directory = _task_dir(token)
    private_dir = directory / "candidates"
    private_dir.mkdir(mode=0o700, exist_ok=True)
    created: list[dict[str, Any]] = []
    created_paths: list[Path] = []
    try:
        for source in sources[:_MAX_ACTIVE_CANDIDATES]:
            candidate_id = str(uuid.uuid4())
            private_path = private_dir / f"{candidate_id}.png"
            public_path = output_dir / f"desktop-pet-{token[:8]}-{candidate_id}.png"
            _atomic_png_copy(Path(source), private_path)
            created_paths.append(private_path)
            _atomic_png_copy(private_path, public_path)
            created_paths.append(public_path)
            created.append(
                {
                    "candidate_id": candidate_id,
                    "private_path": str(private_path),
                    "path": str(public_path),
                    "status": "ready",
                    "created_at": _now_iso(),
                }
            )
        return created
    except Exception:
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise


def _merge_candidates(
    manifest: dict[str, Any], new_candidates: list[dict[str, Any]]
) -> None:
    candidates = [
        item for item in manifest.get("candidates") or [] if isinstance(item, dict)
    ]
    candidates.extend(new_candidates)
    active = [item for item in candidates if item.get("status") == "ready"]
    keep_ids = {
        item.get("candidate_id") for item in active[-_MAX_ACTIVE_CANDIDATES:]
    }
    for item in candidates:
        if item.get("status") == "ready" and item.get("candidate_id") not in keep_ids:
            item["status"] = "archived"
    manifest["candidates"] = candidates[-(_MAX_ACTIVE_CANDIDATES * 4) :]
    selected = str(manifest.get("selected_candidate_id") or "")
    if selected and selected not in keep_ids:
        manifest["selected_candidate_id"] = ""


def _generate_candidates(
    *,
    manifest: dict[str, Any],
    count: int,
    references: list[str | Path] | None,
    concept: str,
    style: str,
    cancel: threading.Event,
) -> list[dict[str, Any]]:
    from agent.pet.generate import generate_base_drafts

    output_dir = _resolve_output_dir(str(manifest["session_id"]), create=True)
    if output_dir is None:
        raise ValueError("current Agent Computer session output directory is unavailable")
    sources = generate_base_drafts(
        concept,
        n=count,
        style=style,
        reference_images=references,
        is_cancelled=cancel.is_set,
    )
    if cancel.is_set():
        raise RuntimeError("desktop-pet generation was cancelled")
    if not sources:
        raise RuntimeError("image generation produced no usable candidates")
    return _publish_generated_candidates(
        [Path(path) for path in sources], manifest=manifest, output_dir=output_dir
    )


def _manifest_result(manifest: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "token": manifest["token"],
        "status": manifest.get("status") or "",
        "drafts": [_candidate_public(item) for item in _active_candidates(manifest)],
        "selected_candidate_id": manifest.get("selected_candidate_id") or None,
        "last_error": manifest.get("last_error") or None,
    }
    preview = manifest.get("preview")
    if isinstance(preview, dict):
        result["preview"] = preview
    export = manifest.get("export")
    if isinstance(export, dict):
        result["export"] = export
    return result


def _status(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    if token:
        manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
        _write_manifest(manifest)
        return _ok("status", **_manifest_result(manifest))

    from agent.pet.generate.imagegen import list_sprite_providers

    output_dir = _resolve_output_dir(session_id, create=True)
    providers = list_sprite_providers()
    return _ok(
        "status",
        available=bool(providers),
        providers=providers,
        output_capable=output_dir is not None,
        current_turn_image_available=bool(_current_turn_reference()),
        max_candidates=_MAX_ACTIVE_CANDIDATES,
        actions=[
            "draft",
            "regenerate",
            "refine",
            "select",
            "hatch",
            "cancel",
            "export",
        ],
    )


def _current_turn_reference() -> str:
    from gateway.session_context import current_turn_reference_image

    return current_turn_reference_image()


def _draft(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    _prune_tasks()
    concept = _validate_text(
        args.get("concept"), label="concept", maximum=_MAX_CONCEPT_CHARS, required=True
    )
    style = _validate_text(
        args.get("style") or "auto", label="style", maximum=_MAX_STYLE_CHARS
    )
    count = args.get("count", _MAX_ACTIVE_CANDIDATES)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= _MAX_ACTIVE_CANDIDATES:
        raise ValueError(f"count must be between 1 and {_MAX_ACTIVE_CANDIDATES}")
    if _resolve_output_dir(session_id, create=True) is None:
        raise ValueError("current Agent Computer session output directory is unavailable")

    token = secrets.token_hex(16)
    directory = _task_dir(token)
    directory.mkdir(mode=0o700)
    manifest = _new_manifest(
        token=token,
        task_id=task_id,
        session_id=session_id,
        concept=concept,
        style=style,
    )
    _write_manifest(manifest)
    cancel = _reserve(token)
    try:
        reference = _current_turn_reference()
        generated = _generate_candidates(
            manifest=manifest,
            count=count,
            references=[reference] if reference else None,
            concept=concept,
            style=style,
            cancel=cancel,
        )
        _merge_candidates(manifest, generated)
        manifest["status"] = "drafted"
        manifest["reference_used"] = bool(reference)
        manifest["last_error"] = ""
        _write_manifest(manifest)
        return _ok("draft", **_manifest_result(manifest))
    except Exception as exc:
        manifest["status"] = "cancelled" if cancel.is_set() else "failed"
        manifest["last_error"] = str(exc)[:500]
        _write_manifest(manifest)
        raise
    finally:
        _release(token)


def _regenerate(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    count = args.get("count", _MAX_ACTIVE_CANDIDATES)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= _MAX_ACTIVE_CANDIDATES:
        raise ValueError(f"count must be between 1 and {_MAX_ACTIVE_CANDIDATES}")
    cancel = _reserve(token)
    manifest["status"] = "drafting"
    _write_manifest(manifest)
    try:
        reference = _current_turn_reference()
        generated = _generate_candidates(
            manifest=manifest,
            count=count,
            references=[reference] if reference else None,
            concept=str(manifest["concept"]),
            style=str(manifest["style"]),
            cancel=cancel,
        )
        _merge_candidates(manifest, generated)
        manifest["status"] = "drafted"
        manifest["reference_used"] = bool(reference)
        manifest["last_error"] = ""
        _write_manifest(manifest)
        return _ok(
            "regenerate",
            token=token,
            status="drafted",
            drafts=[_candidate_public(item) for item in generated],
        )
    except Exception as exc:
        manifest["status"] = "drafted" if _active_candidates(manifest) else "failed"
        manifest["last_error"] = str(exc)[:500]
        _write_manifest(manifest)
        raise
    finally:
        _release(token)


def _refine(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    candidate_id = str(args.get("candidate_id") or "").strip()
    instruction = _validate_text(
        args.get("instruction"),
        label="instruction",
        maximum=_MAX_INSTRUCTION_CHARS,
        required=True,
    )
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    candidate = _find_candidate(manifest, candidate_id)
    reference = _private_candidate_path(manifest, candidate)
    cancel = _reserve(token)
    manifest["status"] = "refining"
    _write_manifest(manifest)
    try:
        generated = _generate_candidates(
            manifest=manifest,
            count=1,
            references=[reference],
            concept=f"{manifest['concept']}\nRefinement: {instruction}",
            style=str(manifest["style"]),
            cancel=cancel,
        )
        _merge_candidates(manifest, generated)
        manifest["status"] = "drafted"
        manifest["last_error"] = ""
        _write_manifest(manifest)
        return _ok(
            "refine",
            token=token,
            status="drafted",
            drafts=[_candidate_public(item) for item in generated],
        )
    except Exception as exc:
        manifest["status"] = "drafted"
        manifest["last_error"] = str(exc)[:500]
        _write_manifest(manifest)
        raise
    finally:
        _release(token)


def _select(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    candidate_id = str(args.get("candidate_id") or "").strip()
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    _find_candidate(manifest, candidate_id)
    manifest["selected_candidate_id"] = candidate_id
    manifest["status"] = "selected"
    _write_manifest(manifest)
    return _ok(
        "select",
        token=token,
        status="selected",
        selected_candidate_id=candidate_id,
    )


def _preview_metadata(path: Path, *, candidate_id: str) -> dict[str, Any]:
    _inspect_image(path, max_bytes=_MAX_EXPORT_BYTES)
    return {
        "candidate_id": candidate_id,
        "path": str(path),
        "filename": path.name,
        "bytes": path.stat().st_size,
    }


def _hatch(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    candidate_id = str(
        args.get("candidate_id") or ""
    ).strip()
    name = _validate_text(
        args.get("name"), label="name", maximum=_MAX_NAME_CHARS, required=True
    )
    description = _validate_text(
        args.get("description"),
        label="description",
        maximum=_MAX_DESCRIPTION_CHARS,
    )
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    if not candidate_id:
        candidate_id = str(manifest.get("selected_candidate_id") or "")
    candidate = _find_candidate(manifest, candidate_id)
    base = _private_candidate_path(manifest, candidate)
    output_dir = _resolve_output_dir(session_id, create=True)
    if output_dir is None:
        raise ValueError("current Agent Computer session output directory is unavailable")

    cancel = _reserve(token)
    task_dir = _task_dir(token)
    staging_root = task_dir / "hatched"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(mode=0o700)
    manifest["status"] = "hatching"
    manifest["selected_candidate_id"] = candidate_id
    _write_manifest(manifest)
    try:
        from agent.pet.generate import hatch_pet

        result = hatch_pet(
            base_image=base,
            slug=name,
            display_name=name,
            description=description,
            concept=str(manifest["concept"]),
            style=str(manifest["style"]),
            is_cancelled=cancel.is_set,
            staging_dir=staging_root,
        )
        if cancel.is_set():
            raise RuntimeError("desktop-pet hatch was cancelled")
        preview = output_dir / f"{result.slug}-{token[:8]}-preview.webp"
        source = Path(result.spritesheet)
        _inspect_image(source, max_bytes=_MAX_EXPORT_BYTES)
        partial = preview.with_name(preview.name + ".part")
        partial.unlink(missing_ok=True)
        try:
            with source.open("rb") as reader, partial.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            partial.replace(preview)
        finally:
            partial.unlink(missing_ok=True)
        manifest["status"] = "hatched"
        manifest["slug"] = result.slug
        manifest["display_name"] = result.display_name
        manifest["description"] = description
        manifest["pet_dir"] = str(source.parent)
        manifest["spritesheet_path"] = str(source)
        manifest["states"] = result.states
        manifest["preview"] = _preview_metadata(preview, candidate_id=candidate_id)
        manifest["last_error"] = ""
        _write_manifest(manifest)
        return _ok(
            "hatch",
            token=token,
            status="hatched",
            slug=result.slug,
            states=result.states,
            preview=manifest["preview"],
        )
    except Exception as exc:
        manifest["status"] = "drafted"
        manifest["last_error"] = str(exc)[:500]
        _write_manifest(manifest)
        raise
    finally:
        _release(token)


def _unique_export_path(directory: Path, slug: str) -> Path:
    destination = directory / f"{slug}.desktop-pet.zip"
    suffix = 2
    while destination.exists():
        destination = directory / f"{slug}-{suffix}.desktop-pet.zip"
        suffix += 1
    return destination


def _validated_pet_files(manifest: dict[str, Any]) -> tuple[str, Path, Path]:
    slug = str(manifest.get("slug") or "")
    if not _SAFE_ID_RE.fullmatch(slug):
        raise ValueError("hatched pet is unavailable")
    task_dir = _task_dir(str(manifest["token"]))
    pet_dir = Path(str(manifest.get("pet_dir") or ""))
    if not pet_dir.is_absolute() or pet_dir.is_symlink():
        raise ValueError("hatched pet is unavailable")
    pet_dir = pet_dir.resolve(strict=True)
    try:
        pet_dir.relative_to(task_dir)
    except ValueError as exc:
        raise ValueError("hatched pet is unavailable") from exc
    metadata = pet_dir / "pet.json"
    spritesheet = pet_dir / "spritesheet.webp"
    if metadata.is_symlink() or spritesheet.is_symlink():
        raise ValueError("hatched pet is unavailable")
    if not metadata.is_file() or metadata.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("hatched pet metadata is unavailable")
    _inspect_image(spritesheet, max_bytes=_MAX_EXPORT_BYTES)
    parsed = json.loads(metadata.read_text(encoding="utf-8"))
    if (
        not isinstance(parsed, dict)
        or parsed.get("id") != slug
        or parsed.get("spritesheetPath") != "spritesheet.webp"
    ):
        raise ValueError("hatched pet metadata is invalid")
    return slug, metadata, spritesheet


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            digest.update(chunk)
    return {
        "path": str(path),
        "filename": path.name,
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _export(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    existing = manifest.get("export")
    if manifest.get("status") == "exported" and isinstance(existing, dict):
        path = Path(str(existing.get("path") or ""))
        if path.is_file() and path.name.endswith(".desktop-pet.zip"):
            return _ok("export", token=token, status="exported", **existing)
    if manifest.get("status") != "hatched":
        raise ValueError("a successfully hatched pet is required before export")

    slug, metadata, spritesheet = _validated_pet_files(manifest)
    output_dir = _resolve_output_dir(session_id, create=True)
    if output_dir is None:
        raise ValueError("current Agent Computer session output directory is unavailable")
    destination = _unique_export_path(output_dir, slug)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(metadata, arcname=f"{slug}/pet.json")
            archive.write(spritesheet, arcname=f"{slug}/spritesheet.webp")
        if partial.stat().st_size <= 0 or partial.stat().st_size > _MAX_EXPORT_BYTES:
            raise ValueError("desktop-pet package exceeds maximum size")
        with zipfile.ZipFile(partial) as archive:
            if set(archive.namelist()) != {
                f"{slug}/pet.json",
                f"{slug}/spritesheet.webp",
            }:
                raise ValueError("desktop-pet package has invalid contents")
            if any(info.file_size > _MAX_EXPORT_BYTES for info in archive.infolist()):
                raise ValueError("desktop-pet package has an oversized entry")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)

    exported = {**_FORMAT_METADATA, **_file_metadata(destination)}
    manifest["status"] = "exported"
    manifest["export"] = exported
    manifest["last_error"] = ""
    _write_manifest(manifest)
    return _ok("export", token=token, status="exported", **exported)


def _cancel(args: dict[str, Any], *, task_id: str, session_id: str) -> str:
    token = str(args.get("token") or "").strip()
    manifest = _load_manifest(token, task_id=task_id, session_id=session_id)
    event = _cancel_event(token)
    if event is not None:
        event.set()
        status = "cancelling"
    else:
        status = "cancelled"
    manifest["status"] = status
    _write_manifest(manifest)
    return _ok("cancel", token=token, status=status)


def desktop_pet_creator(
    args: dict[str, Any],
    *,
    task_id: str | None = None,
    session_id: str | None = None,
) -> str:
    action = str(args.get("action") or "").strip().lower()
    supported = {
        "status",
        "draft",
        "regenerate",
        "refine",
        "select",
        "hatch",
        "cancel",
        "export",
    }
    if action not in supported:
        return _error(action or "unknown", "unsupported desktop-pet action")
    scope = _scope(task_id, session_id)
    if scope is None:
        return _error(
            action,
            "task and session context are required",
            code="missing_scope",
        )
    task, session = scope
    try:
        if action == "status":
            return _status(args, task_id=task, session_id=session)
        if action == "draft":
            return _draft(args, task_id=task, session_id=session)
        if action == "regenerate":
            return _regenerate(args, task_id=task, session_id=session)
        if action == "refine":
            return _refine(args, task_id=task, session_id=session)
        if action == "select":
            return _select(args, task_id=task, session_id=session)
        if action == "hatch":
            return _hatch(args, task_id=task, session_id=session)
        if action == "cancel":
            return _cancel(args, task_id=task, session_id=session)
        return _export(args, task_id=task, session_id=session)
    except ValueError as exc:
        return _error(action, str(exc))
    except RuntimeError as exc:
        return _error(action, str(exc), code="operation_failed")
    except Exception as exc:
        logger.exception("desktop_pet_creator %s failed: %s", action, exc)
        return _error(
            action,
            f"{action} failed; check Hermes logs for details",
            code="operation_failed",
        )


def _reset_state_for_tests() -> None:
    with _state_lock:
        events = list(_inflight.values())
        _inflight.clear()
    for event in events:
        event.set()


DESKTOP_PET_CREATOR_SCHEMA = {
    "name": "desktop_pet_creator",
    "description": (
        "Create an original animated desktop pet with persistent candidates. "
        "Call status first, then draft, refine/regenerate, select/hatch, and export. "
        "The current turn's single attached image is used automatically; the tool "
        "never accepts file paths and never installs or replaces the active pet."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "status",
                    "draft",
                    "regenerate",
                    "refine",
                    "select",
                    "hatch",
                    "cancel",
                    "export",
                ],
            },
            "token": {"type": "string", "description": "Task token returned by draft."},
            "concept": {"type": "string", "description": "Original desktop-pet concept."},
            "style": {"type": "string", "description": "Optional visual style."},
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_ACTIVE_CANDIDATES,
            },
            "candidate_id": {
                "type": "string",
                "description": "Stable candidate UUID returned by draft/refine/regenerate.",
            },
            "instruction": {
                "type": "string",
                "description": "Requested visual change for refine.",
            },
            "name": {"type": "string", "description": "Display name used when hatching."},
            "description": {"type": "string"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


registry.register(
    name="desktop_pet_creator",
    toolset="desktop_pet",
    schema=DESKTOP_PET_CREATOR_SCHEMA,
    handler=lambda args, **kw: desktop_pet_creator(
        args,
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
    ),
    max_result_size_chars=20_000,
)
