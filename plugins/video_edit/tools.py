"""Atomic tool handlers for the video_edit plugin."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from tools.registry import tool_error, tool_result

from plugins.video_edit import client, normalizer, preferences, schemas, state
from plugins.video_edit.paths import (
    MAX_TASK_ID_LENGTH,
    VideoPathError,
    agent_id_from_kwargs,
    result_path,
    safe_id,
    task_id_from_kwargs,
    validate_output_file,
    validate_input_file,
)


logger = logging.getLogger(__name__)


class _WorkflowUnavailable(state.WorkflowError):
    """The supplied workflow cannot be safely resumed in this profile."""


def _ok(value: dict[str, Any]) -> str:
    return tool_result(value)


def _fail(message: str, **fields: Any) -> str:
    return tool_error(message, **fields)


def _failure(tool_name: str, reason_code: str, message: str, **fields: Any) -> str:
    contract = schemas.error_contract(tool_name, reason_code)
    return _fail(
        message,
        code=contract["code"],
        reason_code=contract["reason_code"],
        retryable=contract["retryable"],
        next=contract["next_tool"],
        recovery=contract["recovery"],
        **fields,
    )


def _business_fail(
    tool_name: str,
    message: str,
    exc: Exception,
    **fields: Any,
) -> str:
    raw_status = getattr(exc, "status", getattr(exc, "code", 0))
    status = raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else 0
    if status in {401, 403}:
        return _failure(
            tool_name,
            "service_admission_failed",
            "video service admission failed",
            **fields,
        )
    if isinstance(exc, _WorkflowUnavailable):
        return _failure(tool_name, "workflow_unavailable", str(exc), **fields)
    if isinstance(exc, (state.WorkflowError, preferences.PreferenceError)):
        return _failure(
            tool_name,
            "local_state_unavailable",
            "local video state is unavailable",
            **fields,
        )
    if isinstance(exc, VideoPathError):
        return _failure(tool_name, "invalid_arguments", str(exc), **fields)
    if isinstance(exc, normalizer.NormalizeError):
        return _failure(
            tool_name,
            "media_preparation_failed",
            "video media preparation failed",
            **fields,
        )
    if isinstance(exc, client.VideoClientError):
        if (
            bool(getattr(exc, "transient", False))
            or status in {408, 425, 429}
            or status >= 500
        ):
            reason_code = "transient_failure"
            rendered = message
        else:
            reason_code = "service_request_rejected"
            rendered = "video service rejected the request"
        logger.warning(
            "video service call failed tool=%s status=%s transient=%s",
            tool_name,
            status,
            reason_code == "transient_failure",
        )
        return _failure(tool_name, reason_code, rendered, **fields)

    safe_types = (
        TimeoutError,
    )
    detail = str(exc) if isinstance(exc, safe_types) else ""
    logger.warning(
        "video tool failed tool=%s exception_type=%s",
        tool_name,
        type(exc).__name__,
    )
    rendered = f"{message}: {detail}" if detail else message
    return _failure(tool_name, "transient_failure", rendered, **fields)


def _terminal_fail(
    tool_name: str,
    reason_code: str,
    message: str,
    **fields: Any,
) -> str:
    return _failure(tool_name, reason_code, message, **fields)


def _video_tool(tool_name: str):
    """Apply the shared pure-help and schema-validation entry boundary."""

    def decorate(handler):
        @functools.wraps(handler)
        def guarded(args: dict, **kwargs: Any) -> str:
            # This literal-True check must remain the first handler action.
            # Help reads only the call arguments and static schema metadata.
            if isinstance(args, dict) and args.get("help") is True:
                return _ok(schemas.render_tool_help(tool_name, args))

            issues = schemas.validate_tool_arguments(tool_name, args)
            if issues:
                return _failure(
                    tool_name,
                    "invalid_arguments",
                    "invalid video tool arguments",
                    tool=tool_name,
                    issues=issues,
                )
            return handler(args, **kwargs)

        return guarded

    return decorate


def _workflow_or_error(raw: Any, agent_id: str) -> dict[str, Any]:
    try:
        value = state.get(str(raw or "").strip(), agent_id)
    except state.WorkflowError as exc:
        raise _WorkflowUnavailable("video workflow is unavailable") from exc
    if value is None:
        raise _WorkflowUnavailable("video workflow not found")
    return value


def _workflow_stage_reason(entry: dict[str, Any]) -> str:
    raw_source_paths = entry.get("source_paths") or []
    raw_object_keys = entry.get("object_keys") or []
    if not isinstance(raw_source_paths, list) or not isinstance(raw_object_keys, list):
        return "upload_incomplete"
    source_paths = [
        value.strip()
        for value in raw_source_paths
        if isinstance(value, str) and value.strip()
    ]
    object_keys = [
        value.strip()
        for value in raw_object_keys
        if isinstance(value, str) and value.strip()
    ]
    if (
        not source_paths
        or len(source_paths) != len(raw_source_paths)
        or len(object_keys) != len(raw_object_keys)
        or len(object_keys) != len(source_paths)
    ):
        return "upload_incomplete"
    if not str(entry.get("project_id") or "").strip():
        return "project_not_created"
    project = entry.get("project") if isinstance(entry.get("project"), dict) else {}
    status = str(project.get("status") or entry.get("status") or "").strip().lower()
    if status in {"failed", "cancelled", "error"}:
        return "project_terminal"
    result_url = str(entry.get("result_url") or project.get("result_url") or "").strip()
    if status != "completed" or not result_url:
        return "project_not_completed"
    if not str(entry.get("output_path") or "").strip():
        return "result_not_downloaded"
    return "delivered"


_STAGE_ERROR_MESSAGES = {
    "upload_incomplete": "video asset upload is incomplete",
    "project_not_created": "video project has not been created",
    "project_not_completed": "video project is not completed",
    "project_terminal": "video project failed",
    "result_not_downloaded": "video result has not been downloaded",
}


def _stage_fail(tool_name: str, reason_code: str, **fields: Any) -> str:
    return _failure(
        tool_name,
        reason_code,
        _STAGE_ERROR_MESSAGES[reason_code],
        **fields,
    )


def _scene(args: dict[str, Any]) -> str:
    return str(args.get("scene") or "general").strip()[:64] or "general"


def _interactive_request_identity(
    args: dict[str, Any], task_id: str
) -> dict[str, Any]:
    """Build the immutable, normalized request identity for one edit task."""

    return {
        "kind": "interactive",
        "task_id": task_id,
        "scene": _scene(args),
        "preferences": preferences._clean_preferences(args.get("preferences")),
        "silent": bool(args.get("silent")),
    }


def _proactive_session_id(entry: dict[str, Any]) -> str:
    """Return the server-owned output bucket for a proactive workflow."""
    if not entry.get("proactive"):
        return ""
    trigger_id = safe_id(entry.get("proactive_trigger_id"), fallback="")
    if not trigger_id or not trigger_id.startswith("pvm-"):
        raise _WorkflowUnavailable("proactive workflow checkpoint is unavailable")
    return "proactive-" + trigger_id


def _checkpoint_target(
    raw_path: str,
    agent_id: str,
    *,
    session_id: str = "",
) -> Path:
    """Validate/rehome an existing checkpoint into its required session bucket."""
    name = Path(raw_path).name
    target = result_path(agent_id, name, allow_existing=True, session_id=session_id)
    if not raw_path:
        return target
    candidate = Path(raw_path)
    if not candidate.exists():
        # A pending path may legitimately not have been created yet. The
        # sanitized basename above is the only part we carry forward.
        return target
    try:
        source = validate_output_file(str(candidate), agent_id, session_id=session_id)
    except VideoPathError as exc:
        raise state.WorkflowError("video result checkpoint is invalid") from exc
    if source == target:
        return target
    if target.exists():
        source_evidence = client.file_evidence(source)
        target_evidence = client.file_evidence(target)
        if source_evidence["sha256"] != target_evidence["sha256"]:
            raise state.WorkflowError("video result checkpoint conflicts with session artifact")
        source.unlink()
        return target
    os.replace(source, target)
    return target


def _recoverable_checkpoint_target(
    raw_path: str,
    agent_id: str,
    *,
    session_id: str = "",
) -> Path:
    """Keep a damaged in-bucket artifact recoverable without trusting its path."""
    try:
        return _checkpoint_target(raw_path, agent_id, session_id=session_id)
    except (VideoPathError, state.WorkflowError, client.VideoClientError) as exc:
        try:
            expected = result_path(
                agent_id,
                Path(raw_path).name,
                allow_existing=True,
                session_id=session_id,
            )
        except VideoPathError as path_exc:
            raise _WorkflowUnavailable(
                "video result checkpoint is unavailable"
            ) from path_exc
        if Path(os.path.abspath(raw_path)) != expected:
            raise _WorkflowUnavailable(
                "video result checkpoint is unavailable"
            ) from exc
        return expected


@_video_tool("video_edit_preferences_resolve")
def handle_preferences_resolve(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        task_id = str(args.get("task_id") or task_id_from_kwargs(kwargs)).strip()[
            :MAX_TASK_ID_LENGTH
        ]
        workflow = state.workflow_id(task_id, agent_id)
        resolved = preferences.resolve(
            agent_id,
            _scene(args),
            args.get("preferences"),
            silent=bool(args.get("silent")),
        )
        try:
            entry = state.create_or_validate_identity(
                workflow,
                agent_id,
                {"resolve_request": _interactive_request_identity(args, task_id)},
                {
                    "task_id": task_id,
                    "scene": resolved["scene"],
                    "preferences": resolved["preferences"],
                    "preference_sources": resolved["sources"],
                    "memory_hit": resolved["memory_hit"],
                    "status": "preferences_resolved",
                },
                legacy_identity={
                    "task_id": task_id,
                    "scene": resolved["scene"],
                    "preferences": resolved["preferences"],
                },
                legacy_requires_non_proactive=True,
            )
        except state.WorkflowError as exc:
            raise _WorkflowUnavailable("video workflow identity changed") from exc
        persisted_preferences = entry.get("preferences")
        persisted_sources = entry.get("preference_sources")
        if not isinstance(persisted_preferences, dict) or not isinstance(
            persisted_sources, dict
        ):
            raise _WorkflowUnavailable("video workflow identity is unavailable")
        memory_hit = entry.get("memory_hit")
        if not isinstance(memory_hit, bool):
            memory_hit = any(
                value != "default" for value in persisted_sources.values()
            )
        return _ok({
            "ok": True,
            "workflow_id": workflow,
            "scene": entry.get("scene"),
            "preferences": persisted_preferences,
            "sources": persisted_sources,
            "memory_hit": memory_hit,
            "next": "video_edit_upload_assets",
        })
    except Exception as exc:
        return _business_fail(
            "video_edit_preferences_resolve",
            "video preference resolution failed",
            exc,
        )


@_video_tool("video_edit_preferences_update")
def handle_preferences_update(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        result = preferences.update(
            agent_id, str(args.get("scope") or ""), _scene(args), str(args.get("kind") or ""),
            str(args.get("action") or ""), args.get("preferences"),
        )
        return _ok(result)
    except Exception as exc:
        return _business_fail(
            "video_edit_preferences_update",
            "video preference update failed",
            exc,
        )


@_video_tool("video_edit_preferences_record_success")
def handle_preferences_record_success(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        return _ok(preferences.record_success(agent_id, _scene(args), args.get("preferences"), args.get("confirmed_fields")))
    except Exception as exc:
        return _business_fail(
            "video_edit_preferences_record_success",
            "video preference memory write failed",
            exc,
        )


def _files_for_upload(entry: dict[str, Any], args: dict[str, Any]) -> list[Path]:
    raw_files = args.get("files")
    persisted_files = entry.get("source_paths")
    has_persisted_selection = isinstance(persisted_files, list) and bool(persisted_files)
    if has_persisted_selection:
        # A workflow is a durable continuation boundary. Once the first
        # selection is recorded, later turns must use that exact selection;
        # an explicit re-edit gets a new task/workflow instead of silently
        # replacing the source set behind an existing cloud project.
        if isinstance(raw_files, list) and raw_files:
            requested = [str(value).strip() for value in raw_files]
            if requested != [str(value).strip() for value in persisted_files]:
                raise _WorkflowUnavailable(
                    "workflow source selection changed; start a new edit"
                )
        raw_files = persisted_files
    if not isinstance(raw_files, list) or not raw_files or len(raw_files) > state.MAX_FILES:
        raise VideoPathError("video file count must be between 1 and 8")
    agent_id = str(entry.get("agent_id") or "")
    try:
        return [validate_input_file(str(value), agent_id) for value in raw_files]
    except VideoPathError as exc:
        if has_persisted_selection:
            raise _WorkflowUnavailable(
                "workflow media is unavailable; start a new edit"
            ) from exc
        raise


def _source_fingerprint(files: list[Path]) -> str:
    """Identify the selected source set without reading multi-GB media twice."""
    digest = hashlib.sha256()
    for path in files:
        stat_result = path.stat()
        digest.update(str(path).encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update(str(stat_result.st_dev).encode())
        digest.update(b"\x00")
        digest.update(str(stat_result.st_ino).encode())
        digest.update(b"\x00")
        digest.update(str(stat_result.st_size).encode())
        digest.update(b"\x00")
        digest.update(str(stat_result.st_mtime_ns).encode())
        digest.update(b"\x00")
        # The media admission helper pins this inode with a temporary hard link,
        # which changes ctime without changing the selected media. Device/inode,
        # size, and mtime remain stable while still rejecting path replacement.
    return digest.hexdigest()


def _upload_batches(files: list[Path]) -> list[list[Path]]:
    """Split uploads by both file count and bounded request bytes."""
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for path in files:
        size = path.stat().st_size
        if current and (len(current) >= 3 or current_bytes + size > 512 * 1024 * 1024):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += size
    if current:
        batches.append(current)
    return batches


def _should_normalize(entry: dict[str, Any], files: list[Path], args: dict[str, Any]) -> bool:
    sizes = [path.stat().st_size for path in files]
    if any(size > client.MAX_UPLOAD_BYTES for size in sizes):
        return True
    requested = args.get("normalize")
    if isinstance(requested, bool):
        return requested
    preference = str((entry.get("preferences") or {}).get("upload_preference") or "").strip()
    if preference == "normalized":
        return True
    if preference == "raw_direct":
        return False
    return len(files) >= 8 or sum(sizes) > 300 * 1024 * 1024


def _checkpoint_normalize_strategy(entry: dict[str, Any]) -> bool:
    normalize = entry.get("normalize")
    if type(normalize) is not bool:
        raise _WorkflowUnavailable("video upload checkpoint is invalid")
    return normalize


def _normalizer_generation_for_upload(
    entry: dict[str, Any],
    *,
    normalize: bool,
    source_checkpointed: bool,
) -> str:
    if not normalize:
        return ""
    raw_persisted = entry.get("normalizer_generation")
    if raw_persisted is None:
        persisted = ""
    elif isinstance(raw_persisted, str):
        persisted = raw_persisted.strip()
    else:
        raise _WorkflowUnavailable("video upload checkpoint is invalid")
    # A new workflow must be able to try normalization even when the optional
    # presets helper is unavailable.  The helper identity is recorded only
    # after normalization succeeds below.  Existing normalized checkpoints,
    # however, must remain pinned to the exact helper generation used for the
    # accepted uploads.
    if not source_checkpointed:
        return ""
    try:
        current = normalizer.generation()
    except normalizer.NormalizeError as exc:
        raise _WorkflowUnavailable(
            "video media preparation is unavailable; start a new edit"
        ) from exc
    if not persisted or persisted != current:
        raise _WorkflowUnavailable(
            "video media preparation changed; start a new edit"
        )
    return current


def _refresh_result_url_checkpoint(
    workflow_id: str,
    agent_id: str,
    project_id: str,
    stale_url: str,
) -> tuple[str, str]:
    """Poll the existing project once and persist only its refreshed result state."""
    project = client.poll_project(project_id, timeout=120.0, agent_id=agent_id)
    status = str(project.get("status") or "").strip().lower()
    refreshed_url = str(project.get("result_url") or "").strip()
    refresh_required = not (
        status == "completed" and refreshed_url and refreshed_url != stale_url
    )
    patch: dict[str, Any] = {
        "project": project,
        "status": status or "polling",
        "result_url_refresh_required": refresh_required,
    }
    if status == "completed" and refreshed_url:
        patch["result_url"] = refreshed_url
    state.update(workflow_id, agent_id, patch)
    return status, refreshed_url


def _handle_upload_assets_locked(
    args: dict[str, Any], agent_id: str, workflow_id: str
) -> str:
    normalized: list[Path] = []
    try:
        entry = _workflow_or_error(workflow_id, agent_id)
        raw_existing = entry.get("object_keys")
        if raw_existing is None:
            raw_existing = []
        if not isinstance(raw_existing, list):
            raise _WorkflowUnavailable("video upload checkpoint is invalid")
        existing = [
            value.strip()
            for value in raw_existing
            if isinstance(value, str) and value.strip()
        ]
        if len(existing) != len(raw_existing):
            raise _WorkflowUnavailable("video upload checkpoint is invalid")

        if existing:
            raw_persisted = entry.get("source_paths")
            if not isinstance(raw_persisted, list):
                raise _WorkflowUnavailable("video upload checkpoint is invalid")
            persisted = [
                value.strip()
                for value in raw_persisted
                if isinstance(value, str) and value.strip()
            ]
            if (
                not persisted
                or len(persisted) != len(raw_persisted)
                or len(persisted) > state.MAX_FILES
                or len(existing) > len(persisted)
            ):
                raise _WorkflowUnavailable("video upload checkpoint is invalid")

            raw_requested = args.get("files")
            if isinstance(raw_requested, list) and raw_requested:
                requested = [str(value).strip() for value in raw_requested]
                if requested != persisted:
                    raise _WorkflowUnavailable(
                        "workflow source selection changed; start a new edit"
                    )

            if len(existing) == len(persisted):
                normalize = _checkpoint_normalize_strategy(entry)
                return _ok({"ok": True, "workflow_id": workflow_id, "uploaded": len(existing), "reused": True, "strategy": "normalized" if normalize else "raw_direct", "next": "video_edit_create_project"})

        files = _files_for_upload(entry, args)
        source_fingerprint = _source_fingerprint(files)
        previous_fingerprint = str(entry.get("source_fingerprint") or "").strip()
        if previous_fingerprint != source_fingerprint and (
            previous_fingerprint or entry.get("object_keys")
        ):
            raise _WorkflowUnavailable(
                "workflow media changed; start an explicit re-edit with a new task_id"
            )
        normalize = (
            _checkpoint_normalize_strategy(entry)
            if previous_fingerprint
            else _should_normalize(entry, files, args)
        )
        normalizer_generation = _normalizer_generation_for_upload(
            entry,
            normalize=normalize,
            source_checkpointed=bool(previous_fingerprint),
        )
        # Keep the private source checkpoint for retries, but never return it
        # to the model on proactive runs.
        state.update(workflow_id, agent_id, {
            "source_paths": [str(path) for path in files],
            "source_names": [path.name for path in files],
            "source_fingerprint": source_fingerprint,
            "normalize": normalize,
            "normalizer_generation": normalizer_generation,
            "status": "uploading",
        })
        if len(existing) != len(raw_existing) or len(existing) > len(files):
            raise _WorkflowUnavailable("video upload checkpoint is invalid")
        upload_files = files
        if normalize:
            try:
                upload_files = normalized = normalizer.normalize_files(files, workflow_id)
                # Read the helper identity after the output has been produced.
                # This keeps an unavailable optional helper from blocking the
                # first attempt before the raw-direct fallback can run.
                normalizer_generation = normalizer.generation()
                state.update(workflow_id, agent_id, {
                    "normalizer_generation": normalizer_generation,
                })
            except normalizer.NormalizeError:
                if existing or any(
                    path.stat().st_size > client.MAX_UPLOAD_BYTES for path in files
                ):
                    raise
                # Hardware normalization is an optimization. When direct
                # upload remains inside the provider contract, fall back
                # without asking the user or creating a second workflow.
                normalize = False
                upload_files = files
                state.update(workflow_id, agent_id, {
                    "normalize": False,
                    "normalizer_generation": "",
                    "normalization_fallback": "raw_direct",
                    "status": "uploading",
                })
        uploaded = list(existing)
        for batch in _upload_batches(upload_files[len(existing):]):
            body = client.upload(
                batch,
                agent_id=agent_id,
                replay_scope=source_fingerprint,
            )
            batch_keys = client.extract_upload_keys(body)
            if len(batch_keys) != len(batch):
                raise client.VideoClientError("video upload response does not match the requested batch")
            uploaded.extend(batch_keys)
            state.update(
                workflow_id,
                agent_id,
                {
                    "object_keys": uploaded,
                    "uploaded_count": len(uploaded),
                    "status": (
                        "assets_uploaded"
                        if len(uploaded) >= len(upload_files)
                        else "uploading"
                    ),
                },
            )
            # Normalized intermediates are disposable as soon as their upload
            # batch has been accepted; the durable workflow keeps only keys.
            if normalize:
                normalizer.cleanup(batch, workflow_id)
                normalized = [path for path in normalized if path not in batch]
        return _ok({"ok": True, "workflow_id": workflow_id, "uploaded": len(uploaded), "strategy": "normalized" if normalize else "raw_direct", "next": "video_edit_create_project"})
    except Exception as exc:
        return _business_fail(
            "video_edit_upload_assets",
            "video asset upload failed",
            exc,
        )
    finally:
        if normalized:
            normalizer.cleanup(normalized, str(args.get("workflow_id") or ""))


@_video_tool("video_edit_upload_assets")
def handle_upload_assets(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        try:
            # Keep one lock across checkpoint reads/writes, preparation,
            # provider upload, and cleanup. The normalizer uses deterministic
            # workspace filenames, so releasing it earlier would reintroduce
            # an overwrite race for this workflow.
            with state.upload_lock(workflow_id, agent_id):
                return _handle_upload_assets_locked(args, agent_id, workflow_id)
        except state.WorkflowError as exc:
            raise _WorkflowUnavailable("video upload workflow is unavailable") from exc
    except Exception as exc:
        return _business_fail(
            "video_edit_upload_assets",
            "video asset upload failed",
            exc,
        )


@_video_tool("video_edit_create_project")
def handle_create_project(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        if _workflow_stage_reason(entry) == "upload_incomplete":
            return _stage_fail(
                "video_edit_create_project",
                "upload_incomplete",
                workflow_id=workflow_id,
            )
        object_keys = [str(key).strip() for key in entry.get("object_keys") or []]
        existing = str(entry.get("project_id") or "").strip()
        if existing:
            return _ok({"ok": True, "workflow_id": workflow_id, "project_id": existing, "reused": True, "next": "video_edit_wait_project"})
        project = client.create_project(
            object_keys,
            dict(entry.get("preferences") or {}),
            user_prompt=str(args.get("user_prompt") or ""),
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
        project_id = str(project.get("project_id") or "").strip()
        if not project_id:
            raise client.VideoClientError("video project id is missing")
        state.update(workflow_id, agent_id, {"project_id": project_id, "project": project, "status": "project_created"})
        return _ok({"ok": True, "workflow_id": workflow_id, "project_id": project_id, "next": "video_edit_wait_project"})
    except Exception as exc:
        return _business_fail(
            "video_edit_create_project",
            "video project creation failed",
            exc,
        )


@_video_tool("video_edit_wait_project")
def handle_wait_project(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        raw_project = entry.get("project")
        project = raw_project if isinstance(raw_project, dict) else {}
        stage_reason = _workflow_stage_reason(entry)
        if stage_reason in {"upload_incomplete", "project_not_created"}:
            return _stage_fail(
                "video_edit_wait_project",
                stage_reason,
                workflow_id=workflow_id,
            )
        project_id = str(entry.get("project_id") or "").strip()
        if stage_reason == "project_terminal":
            return _terminal_fail(
                "video_edit_wait_project",
                "project_terminal",
                "video project failed",
                workflow_id=workflow_id,
                project_id=project_id,
                status=str(project.get("status") or entry.get("status") or ""),
            )
        if stage_reason in {"result_not_downloaded", "delivered"}:
            return _ok(
                {
                    "ok": True,
                    "workflow_id": workflow_id,
                    "project_id": project_id,
                    "status": "completed",
                    "continue_required": False,
                    "reused": True,
                    "next": "video_edit_download_result",
                }
            )
        deadline = time.monotonic() + max(15, min(480, int(args.get("max_wait_seconds") or 120)))
        while True:
            project = client.poll_project(project_id, timeout=min(120, max(15, deadline - time.monotonic())), agent_id=agent_id)
            status = str(project.get("status") or "").strip().lower()
            state.update(workflow_id, agent_id, {"project": project, "status": status or "polling"})
            if status == "completed":
                result_url = str(project.get("result_url") or "").strip()
                if not result_url:
                    return _stage_fail(
                        "video_edit_wait_project",
                        "project_not_completed",
                        workflow_id=workflow_id,
                        project_id=project_id,
                    )
                state.update(
                    workflow_id,
                    agent_id,
                    {
                        "result_url": result_url,
                        "result_url_refresh_required": False,
                        "status": "completed",
                    },
                )
                # The signed provider URL is private plugin state.  The model
                # only needs the durable workflow handle for the next atomic
                # tool, so never echo the URL into conversation history.
                return _ok({"ok": True, "workflow_id": workflow_id, "project_id": project_id, "status": status, "continue_required": False, "next": "video_edit_download_result"})
            if status in {"failed", "cancelled", "error"}:
                return _terminal_fail(
                    "video_edit_wait_project",
                    "project_terminal",
                    "video project failed",
                    workflow_id=workflow_id,
                    project_id=project_id,
                    status=status,
                )
            if time.monotonic() >= deadline:
                return _ok({"ok": True, "workflow_id": workflow_id, "project_id": project_id, "status": status or "processing", "continue_required": True, "next": "video_edit_wait_project"})
            time.sleep(min(10, max(1, deadline - time.monotonic())))
    except Exception as exc:
        return _business_fail(
            "video_edit_wait_project",
            "video project polling failed",
            exc,
        )


@_video_tool("video_edit_download_result")
def handle_download_result(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        raw_project = entry.get("project")
        project = raw_project if isinstance(raw_project, dict) else {}
        stage_reason = _workflow_stage_reason(entry)
        if stage_reason in {
            "upload_incomplete",
            "project_not_created",
            "project_not_completed",
            "project_terminal",
        }:
            return _stage_fail(
                "video_edit_download_result",
                stage_reason,
                workflow_id=workflow_id,
            )
        result_url = str(entry.get("result_url") or "").strip()
        if not result_url:
            result_url = str(project.get("result_url") or "").strip()
        if not result_url:
            return _stage_fail(
                "video_edit_download_result",
                "project_not_completed",
                workflow_id=workflow_id,
            )
        output_session = _proactive_session_id(entry)
        existing = str(entry.get("output_path") or "").strip()
        pending = str(entry.get("pending_output_path") or "").strip()
        checkpoint = existing or pending
        if checkpoint:
            target = _recoverable_checkpoint_target(
                checkpoint,
                agent_id,
                session_id=output_session,
            )
        else:
            target = result_path(
                agent_id,
                str(args.get("filename") or f"{workflow_id}.mp4"),
                session_id=output_session,
            )
        if existing and target.is_file():
            try:
                evidence = client.file_evidence(target)
            except client.VideoClientError:
                pass
            else:
                if str(target) != existing:
                    state.update(
                        workflow_id,
                        agent_id,
                        {"output_path": evidence["path"]},
                    )
                return _ok({
                    "ok": True,
                    "workflow_id": workflow_id,
                    "output": evidence["path"],
                    "size": evidence["size"],
                    "sha256": evidence["sha256"],
                    "reused": True,
                    "next": (
                        "video_edit_proactive_report"
                        if entry.get("proactive")
                        else "video_edit_preferences_record_success"
                    ),
                })
        state.update(
            workflow_id,
            agent_id,
            {
                "pending_output_path": str(target),
                "output_path": "",
                "status": "downloading",
            },
        )
        recovered = not existing and target.is_file()
        evidence: dict[str, Any] | None = None
        if recovered:
            try:
                evidence = client.file_evidence(target)
            except client.VideoClientError:
                recovered = False
        if evidence is None:
            refresh_used = False
            if entry.get("result_url_refresh_required"):
                status, refreshed_url = _refresh_result_url_checkpoint(
                    workflow_id,
                    agent_id,
                    str(entry.get("project_id") or "").strip(),
                    result_url,
                )
                refresh_used = True
                if status in {"failed", "cancelled", "error"}:
                    return _terminal_fail(
                        "video_edit_download_result",
                        "project_terminal",
                        "video project failed",
                        workflow_id=workflow_id,
                        project_id=str(entry.get("project_id") or "").strip(),
                        status=status,
                    )
                if status != "completed":
                    return _stage_fail(
                        "video_edit_download_result",
                        "project_not_completed",
                        workflow_id=workflow_id,
                    )
                if not refreshed_url or refreshed_url == result_url:
                    return _failure(
                        "video_edit_download_result",
                        "transient_failure",
                        "video result URL refresh is pending",
                        workflow_id=workflow_id,
                    )
                result_url = refreshed_url
            try:
                evidence = client.download(result_url, target)
            except client.ResultURLUnavailable as exc:
                state.update(
                    workflow_id,
                    agent_id,
                    {"result_url_refresh_required": True},
                )
                if refresh_used:
                    raise client.VideoClientError(
                        "video result URL refresh is pending",
                        transient=True,
                    ) from exc
                status, refreshed_url = _refresh_result_url_checkpoint(
                    workflow_id,
                    agent_id,
                    str(entry.get("project_id") or "").strip(),
                    result_url,
                )
                if status in {"failed", "cancelled", "error"}:
                    return _terminal_fail(
                        "video_edit_download_result",
                        "project_terminal",
                        "video project failed",
                        workflow_id=workflow_id,
                        project_id=str(entry.get("project_id") or "").strip(),
                        status=status,
                    )
                if status != "completed":
                    return _stage_fail(
                        "video_edit_download_result",
                        "project_not_completed",
                        workflow_id=workflow_id,
                    )
                if not refreshed_url or refreshed_url == result_url:
                    return _failure(
                        "video_edit_download_result",
                        "transient_failure",
                        "video result URL refresh is pending",
                        workflow_id=workflow_id,
                    )
                result_url = refreshed_url
                try:
                    evidence = client.download(result_url, target)
                except client.ResultURLUnavailable as retry_exc:
                    state.update(
                        workflow_id,
                        agent_id,
                        {"result_url_refresh_required": True},
                    )
                    raise client.VideoClientError(
                        "video result URL refresh is pending",
                        transient=True,
                    ) from retry_exc
        state.update(workflow_id, agent_id, {
            "pending_output_path": "", "output_path": evidence["path"],
            "output": evidence,
            "result_url_refresh_required": False,
            "status": "delivered",
        })
        return _ok({
            "ok": True, "workflow_id": workflow_id, "output": evidence["path"],
            "size": evidence["size"], "sha256": evidence["sha256"],
            "recovered": recovered,
            "next": (
                "video_edit_proactive_report"
                if entry.get("proactive")
                else "video_edit_preferences_record_success"
            ),
        })
    except Exception as exc:
        return _business_fail(
            "video_edit_download_result",
            "video result download failed",
            exc,
        )


@_video_tool("video_edit_proactive_resolve")
def handle_proactive_resolve(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        manifest_id = str(args.get("manifest_id") or "").strip()
        task_id = str(args.get("task_id") or task_id_from_kwargs(kwargs)).strip()[
            :MAX_TASK_ID_LENGTH
        ]
        body = client.proactive_resolve(manifest_id, agent_id=agent_id)
        payload = body.get("data") if isinstance(body, dict) and isinstance(body.get("data"), dict) else body
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            raise _WorkflowUnavailable("proactive workflow manifest is unavailable")
        trigger_id = safe_id(payload.get("trigger_id"), fallback="")
        if not trigger_id or not trigger_id.startswith("pvm-"):
            raise _WorkflowUnavailable("proactive workflow manifest is unavailable")
        workflow = state.workflow_id(task_id, agent_id)
        scene = str(payload.get("scene") or "general")
        resolved = preferences.resolve(agent_id, scene, {}, silent=True)
        files = payload["files"]
        if not 2 <= len(files) <= state.MAX_FILES:
            raise _WorkflowUnavailable("proactive workflow manifest is unavailable")
        if any(
            not isinstance(item, dict)
            or not str(item.get("path") or "").strip()
            for item in files
        ):
            raise _WorkflowUnavailable("proactive workflow manifest is unavailable")
        paths = [str(item["path"]).strip() for item in files]
        try:
            state.create_or_validate_identity(
                workflow,
                agent_id,
                {
                    "task_id": task_id,
                    "manifest_id": manifest_id,
                    "proactive_trigger_id": trigger_id,
                    "source_paths": paths,
                    "scene": scene,
                    "proactive": True,
                },
                {
                    "preferences": resolved["preferences"],
                    "preference_sources": resolved["sources"],
                    "status": "proactive_resolved",
                },
            )
        except state.WorkflowError as exc:
            raise _WorkflowUnavailable(
                "proactive workflow identity changed"
            ) from exc
        return _ok({"ok": True, "workflow_id": workflow, "manifest_id": manifest_id, "file_count": len(paths), "silent": True, "next": "video_edit_upload_assets"})
    except Exception as exc:
        return _business_fail(
            "video_edit_proactive_resolve",
            "weekly video manifest resolve failed",
            exc,
        )


def _report_proactive_result(workflow_id: str, agent_id: str) -> str:
    entry = _workflow_or_error(workflow_id, agent_id)
    if not entry.get("proactive"):
        return _terminal_fail(
            "video_edit_proactive_report",
            "not_proactive_workflow",
            "workflow is not proactive",
            workflow_id=workflow_id,
        )
    if entry.get("reported"):
        return _ok({"ok": True, "workflow_id": workflow_id, "reported": True, "reused": True})
    stage_reason = _workflow_stage_reason(entry)
    if stage_reason != "delivered":
        return _stage_fail(
            "video_edit_proactive_report",
            stage_reason,
            workflow_id=workflow_id,
        )
    manifest_id = str(entry.get("manifest_id") or "").strip()
    output = str(entry.get("output_path") or "").strip()
    if not manifest_id:
        return _failure(
            "video_edit_proactive_report",
            "workflow_unavailable",
            "proactive workflow checkpoint is unavailable",
            workflow_id=workflow_id,
        )
    try:
        target = _recoverable_checkpoint_target(
            output,
            agent_id,
            session_id=_proactive_session_id(entry),
        )
        evidence = client.file_evidence(target)
    except _WorkflowUnavailable:
        raise
    except client.VideoClientError:
        return _stage_fail(
            "video_edit_proactive_report",
            "result_not_downloaded",
            workflow_id=workflow_id,
        )
    result = client.proactive_report(
        manifest_id,
        evidence["path"],
        agent_id=agent_id,
    )
    state.update(workflow_id, agent_id, {"reported": True, "status": "reported"})
    return _ok({"ok": True, "workflow_id": workflow_id, "reported": True, "result": result})


@_video_tool("video_edit_proactive_report")
def handle_proactive_report(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        try:
            with state.report_lock(workflow_id, agent_id):
                return _report_proactive_result(workflow_id, agent_id)
        except state.WorkflowError as exc:
            raise _WorkflowUnavailable(
                "proactive workflow checkpoint is unavailable"
            ) from exc
    except Exception as exc:
        return _business_fail(
            "video_edit_proactive_report",
            "weekly video result report failed",
            exc,
        )


HANDLERS = {
    "video_edit_preferences_resolve": handle_preferences_resolve,
    "video_edit_preferences_update": handle_preferences_update,
    "video_edit_preferences_record_success": handle_preferences_record_success,
    "video_edit_upload_assets": handle_upload_assets,
    "video_edit_create_project": handle_create_project,
    "video_edit_wait_project": handle_wait_project,
    "video_edit_download_result": handle_download_result,
    "video_edit_proactive_resolve": handle_proactive_resolve,
    "video_edit_proactive_report": handle_proactive_report,
}
