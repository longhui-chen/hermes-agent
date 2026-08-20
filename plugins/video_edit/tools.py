"""Atomic tool handlers for the video_edit plugin."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tools.registry import tool_error, tool_result

from plugins.video_edit import client, normalizer, preferences, state
from plugins.video_edit.paths import VideoPathError, agent_id_from_kwargs, result_path, task_id_from_kwargs, validate_input_file


def _ok(value: dict[str, Any]) -> str:
    return tool_result(value)


def _fail(message: str, **fields: Any) -> str:
    return tool_error(message, **fields)


def _workflow_or_error(raw: Any, agent_id: str) -> dict[str, Any]:
    value = state.get(str(raw or "").strip(), agent_id)
    if value is None:
        raise state.WorkflowError("video workflow not found")
    return value


def _scene(args: dict[str, Any]) -> str:
    return str(args.get("scene") or "general").strip()[:64] or "general"


def handle_preferences_resolve(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        task_id = str(args.get("task_id") or task_id_from_kwargs(kwargs)).strip()[:256]
        workflow = state.workflow_id(task_id, agent_id)
        resolved = preferences.resolve(agent_id, _scene(args), args.get("preferences"), silent=bool(args.get("silent")))
        entry = state.update(workflow, agent_id, {
            "task_id": task_id,
            "scene": resolved["scene"],
            "preferences": resolved["preferences"],
            "preference_sources": resolved["sources"],
            "status": "preferences_resolved",
        })
        return _ok({
            "ok": True,
            "workflow_id": workflow,
            "scene": entry.get("scene"),
            "preferences": resolved["preferences"],
            "sources": resolved["sources"],
            "memory_hit": resolved["memory_hit"],
            "next": "video_edit_upload_assets",
        })
    except Exception as exc:
        return _fail(f"video preference resolution failed: {exc}")


def handle_preferences_update(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        result = preferences.update(
            agent_id, str(args.get("scope") or ""), _scene(args), str(args.get("kind") or ""),
            str(args.get("action") or ""), args.get("preferences"),
        )
        return _ok(result)
    except Exception as exc:
        return _fail(f"video preference update failed: {exc}")


def handle_preferences_record_success(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        return _ok(preferences.record_success(agent_id, _scene(args), args.get("preferences"), args.get("confirmed_fields")))
    except Exception as exc:
        return _fail(f"video preference memory write failed: {exc}")


def _files_for_upload(entry: dict[str, Any], args: dict[str, Any]) -> list[Path]:
    raw_files = args.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raw_files = entry.get("source_paths")
    if not isinstance(raw_files, list) or not raw_files or len(raw_files) > state.MAX_FILES:
        raise VideoPathError("video file count must be between 1 and 8")
    agent_id = str(entry.get("agent_id") or "")
    return [validate_input_file(str(value), agent_id) for value in raw_files]


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


def handle_upload_assets(args: dict, **kwargs: Any) -> str:
    normalized: list[Path] = []
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        files = _files_for_upload(entry, args)
        normalize = _should_normalize(entry, files, args)
        # Keep the private source checkpoint for retries, but never return it
        # to the model on proactive runs.
        state.update(workflow_id, agent_id, {
            "source_paths": [str(path) for path in files],
            "source_names": [path.name for path in files],
            "normalize": normalize,
            "status": "uploading",
        })
        existing = list(entry.get("object_keys") or [])
        if len(existing) >= len(files):
            return _ok({"ok": True, "workflow_id": workflow_id, "uploaded": len(existing), "reused": True, "strategy": "normalized" if normalize else "raw_direct", "next": "video_edit_create_project"})
        upload_files = files
        if normalize:
            try:
                upload_files = normalized = normalizer.normalize_files(files, workflow_id)
            except normalizer.NormalizeError:
                if any(path.stat().st_size > client.MAX_UPLOAD_BYTES for path in files):
                    raise
                # Hardware normalization is an optimization. When direct
                # upload remains inside the provider contract, fall back
                # without asking the user or creating a second workflow.
                normalize = False
                upload_files = files
                state.update(workflow_id, agent_id, {
                    "normalize": False,
                    "normalization_fallback": "raw_direct",
                    "status": "uploading",
                })
        uploaded = list(existing)
        for batch in _upload_batches(upload_files[len(existing):]):
            body = client.upload(batch, agent_id=agent_id)
            uploaded.extend(client.extract_upload_keys(body))
            state.update(workflow_id, agent_id, {"object_keys": uploaded, "uploaded_count": len(uploaded), "status": "assets_uploaded"})
            # Normalized intermediates are disposable as soon as their upload
            # batch has been accepted; the durable workflow keeps only keys.
            if normalize:
                normalizer.cleanup(batch, workflow_id)
                normalized = [path for path in normalized if path not in batch]
        return _ok({"ok": True, "workflow_id": workflow_id, "uploaded": len(uploaded), "strategy": "normalized" if normalize else "raw_direct", "next": "video_edit_create_project"})
    except Exception as exc:
        return _fail(f"video asset upload failed: {exc}")
    finally:
        if normalized:
            normalizer.cleanup(normalized, str(args.get("workflow_id") or ""))


def handle_create_project(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        object_keys = [str(key).strip() for key in entry.get("object_keys") or [] if str(key).strip()]
        if not object_keys:
            raise state.WorkflowError("video assets have not been uploaded")
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
        return _fail(f"video project creation failed: {exc}")


def handle_wait_project(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        project_id = str(entry.get("project_id") or "").strip()
        if not project_id:
            raise state.WorkflowError("video project has not been created")
        deadline = time.monotonic() + max(15, min(480, int(args.get("max_wait_seconds") or 120)))
        project = dict(entry.get("project") or {})
        while True:
            project = client.poll_project(project_id, timeout=min(120, max(15, deadline - time.monotonic())), agent_id=agent_id)
            status = str(project.get("status") or "").strip().lower()
            state.update(workflow_id, agent_id, {"project": project, "status": status or "polling"})
            if status == "completed":
                result_url = str(project.get("result_url") or "").strip()
                if not result_url:
                    raise client.VideoClientError("completed video project has no result URL")
                state.update(workflow_id, agent_id, {"result_url": result_url, "status": "completed"})
                # The signed provider URL is private plugin state.  The model
                # only needs the durable workflow handle for the next atomic
                # tool, so never echo the URL into conversation history.
                return _ok({"ok": True, "workflow_id": workflow_id, "project_id": project_id, "status": status, "continue_required": False, "next": "video_edit_download_result"})
            if status in {"failed", "cancelled", "error"}:
                return _fail("video project failed", workflow_id=workflow_id, project_id=project_id, status=status)
            if time.monotonic() >= deadline:
                return _ok({"ok": True, "workflow_id": workflow_id, "project_id": project_id, "status": status or "processing", "continue_required": True, "next": "video_edit_wait_project"})
            time.sleep(min(10, max(1, deadline - time.monotonic())))
    except Exception as exc:
        return _fail(f"video project polling failed: {exc}")


def handle_download_result(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        result_url = str(entry.get("result_url") or "").strip()
        if not result_url:
            project = dict(entry.get("project") or {})
            result_url = str(project.get("result_url") or "").strip()
        if not result_url:
            raise state.WorkflowError("video project is not completed")
        existing = str(entry.get("output_path") or "").strip()
        if existing:
            # Never trust a persisted path merely because it exists. Revalidate
            # it against the current agent's output bucket before reusing a
            # completed artifact; this keeps profile state and produced files
            # aligned after a restart or manual state-file replacement.
            existing_target = result_path(agent_id, Path(existing).name, allow_existing=True)
            if str(existing_target) != existing:
                raise state.WorkflowError("video result checkpoint is invalid")
            evidence = client.file_evidence(existing_target)
            return _ok({
                "ok": True,
                "workflow_id": workflow_id,
                "output": evidence["path"],
                "size": evidence["size"],
                "sha256": evidence["sha256"],
                "reused": True,
                "next": "video_edit_preferences_record_success",
            })
        pending = str(entry.get("pending_output_path") or "").strip()
        if pending:
            target = result_path(agent_id, Path(pending).name, allow_existing=True)
            if str(target) != pending:
                raise state.WorkflowError("video result checkpoint is invalid")
        else:
            target = result_path(agent_id, str(args.get("filename") or f"{workflow_id}.mp4"))
            state.update(workflow_id, agent_id, {"pending_output_path": str(target), "status": "downloading"})
        recovered = target.is_file()
        evidence = client.file_evidence(target) if recovered else client.download(result_url, target)
        state.update(workflow_id, agent_id, {
            "pending_output_path": "", "output_path": evidence["path"],
            "output": evidence, "status": "delivered",
        })
        return _ok({
            "ok": True, "workflow_id": workflow_id, "output": evidence["path"],
            "size": evidence["size"], "sha256": evidence["sha256"],
            "recovered": recovered, "next": "video_edit_preferences_record_success",
        })
    except Exception as exc:
        return _fail(f"video result download failed: {exc}")


def handle_proactive_resolve(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        manifest_id = str(args.get("manifest_id") or "").strip()
        task_id = str(args.get("task_id") or task_id_from_kwargs(kwargs)).strip()[:256]
        body = client.proactive_resolve(manifest_id, agent_id=agent_id)
        payload = body.get("data") if isinstance(body, dict) and isinstance(body.get("data"), dict) else body
        if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
            raise client.VideoClientError("proactive manifest contains no files")
        workflow = state.workflow_id(task_id, agent_id)
        scene = str(payload.get("scene") or "general")
        resolved = preferences.resolve(agent_id, scene, {}, silent=True)
        files = payload["files"][:state.MAX_FILES]
        paths = [str(item.get("path") or "") for item in files if isinstance(item, dict) and str(item.get("path") or "").strip()]
        if not paths:
            raise client.VideoClientError("proactive manifest contains no usable files")
        state.update(workflow, agent_id, {
            "task_id": task_id, "scene": scene,
            "manifest_id": manifest_id, "source_paths": paths,
            "preferences": resolved["preferences"], "preference_sources": resolved["sources"],
            "status": "proactive_resolved", "proactive": True,
        })
        return _ok({"ok": True, "workflow_id": workflow, "manifest_id": manifest_id, "file_count": len(paths), "silent": True, "next": "video_edit_upload_assets"})
    except Exception as exc:
        return _fail(f"weekly video manifest resolve failed: {exc}")


def handle_proactive_report(args: dict, **kwargs: Any) -> str:
    try:
        agent_id = agent_id_from_kwargs(kwargs)
        workflow_id = str(args.get("workflow_id") or "").strip()
        entry = _workflow_or_error(workflow_id, agent_id)
        if not entry.get("proactive"):
            raise state.WorkflowError("workflow is not proactive")
        if entry.get("reported"):
            return _ok({"ok": True, "workflow_id": workflow_id, "reported": True, "reused": True})
        manifest_id = str(entry.get("manifest_id") or "").strip()
        output = str(entry.get("output_path") or "").strip()
        if not manifest_id or not output:
            raise state.WorkflowError("proactive result is not ready")
        result = client.proactive_report(manifest_id, output, agent_id=agent_id)
        state.update(workflow_id, agent_id, {"reported": True, "status": "reported"})
        return _ok({"ok": True, "workflow_id": workflow_id, "reported": True, "result": result})
    except Exception as exc:
        return _fail(f"weekly video result report failed: {exc}")


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
