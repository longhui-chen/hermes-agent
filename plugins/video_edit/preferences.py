"""Hard/soft video preference memory owned by the Hermes plugin."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from plugins.video_edit.paths import state_path

DEFAULTS = {
    "style": "freestyle",
    "aspect_ratio": "9:16",
    "duration": 60,
    "decision_mode": "auto",
    "editing_directives": [],
}
VALID_FIELDS = frozenset({"style", "aspect_ratio", "duration", "decision_mode", "editing_directives", "upload_preference", "user_prompt"})
VALID_DIRECTIVES = frozenset({
    "energetic_pacing", "relaxed_pacing", "chronological_story", "highlights_first",
    "keep_original_audio", "add_background_music", "no_background_music", "add_captions",
    "no_captions", "preserve_dialogue",
})


class PreferenceError(ValueError):
    pass


def _empty() -> dict[str, Any]:
    return {"version": 1, "global": {"hard": {}, "soft": {}}, "scenes": {}}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty()
    except (OSError, ValueError) as exc:
        raise PreferenceError("video preference memory is unreadable") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise PreferenceError("video preference memory is invalid")
    value.setdefault("global", {"hard": {}, "soft": {}})
    value.setdefault("scenes", {})
    return value


def _clean_preferences(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in VALID_FIELDS:
            continue
        if key == "editing_directives":
            if not isinstance(value, list):
                continue
            values = [str(item).strip() for item in value[:3] if str(item).strip() in VALID_DIRECTIVES]
            out[key] = values
        elif key == "duration":
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= 3600:
                out[key] = number
        elif key == "aspect_ratio" and str(value) in {"9:16", "16:9", "1:1"}:
            out[key] = str(value)
        elif key == "decision_mode" and str(value) in {"auto", "balanced", "precise"}:
            out[key] = str(value)
        elif key == "upload_preference" and str(value) in {"raw_direct", "normalized"}:
            out[key] = str(value)
        elif key in {"style", "user_prompt"} and isinstance(value, str) and value.strip():
            out[key] = value.strip()[:512]
    return out


def _write(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def _target(data: dict[str, Any], scope: str, scene: str) -> dict[str, Any]:
    if scope == "global":
        return data["global"]
    if scope != "scene" or not scene:
        raise PreferenceError("scene scope requires a scene")
    return data["scenes"].setdefault(scene, {"hard": {}, "soft": {}})


def resolve(agent_id: str, scene: str, explicit: Any, *, silent: bool = False) -> dict[str, Any]:
    scene = str(scene or "general").strip()[:64] or "general"
    path = state_path("preferences.json", agent_id)
    lock = path.with_suffix(".lock")
    lock.touch(mode=0o600, exist_ok=True)
    with lock.open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        data = _read(path)
        explicit_clean = _clean_preferences(explicit)
        scene_data = data["scenes"].get(scene, {}) if isinstance(data.get("scenes"), dict) else {}
        global_data = data.get("global", {})
        result = dict(DEFAULTS)
        sources = {key: "default" for key in result}
        for key, value in (global_data.get("soft", {}) or {}).items():
            if key in VALID_FIELDS:
                result[key], sources[key] = value, "memory_soft"
        for key, value in (scene_data.get("soft", {}) or {}).items():
            if key in VALID_FIELDS:
                result[key], sources[key] = value, "scene_soft"
        for key, value in (global_data.get("hard", {}) or {}).items():
            if key in VALID_FIELDS:
                result[key], sources[key] = value, "global_hard"
        for key, value in (scene_data.get("hard", {}) or {}).items():
            if key in VALID_FIELDS:
                result[key], sources[key] = value, "scene_hard"
        for key, value in explicit_clean.items():
            result[key], sources[key] = value, "explicit"
        if not silent:
            # The model is free to continue with defaults; this is metadata,
            # never a gate that asks for an authorization/confirmation turn.
            result.setdefault("decision_mode", "auto")
        return {"scene": scene, "preferences": result, "sources": sources, "memory_hit": any(v != "default" for v in sources.values())}


def update(agent_id: str, scope: str, scene: str, kind: str, action: str, preferences: Any) -> dict[str, Any]:
    if scope not in {"global", "scene"} or kind not in {"hard", "soft"} or action not in {"set", "forget"}:
        raise PreferenceError("invalid preference update")
    path = state_path("preferences.json", agent_id)
    lock = path.with_suffix(".lock")
    lock.touch(mode=0o600, exist_ok=True)
    with lock.open("r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        data = _read(path)
        target = _target(data, scope, str(scene or "").strip()[:64])
        if action == "forget":
            for key in _clean_preferences(preferences).keys() or VALID_FIELDS:
                target[kind].pop(key, None)
        else:
            target[kind].update(_clean_preferences(preferences))
        data["updated_at"] = int(time.time())
        _write(path, data)
        return {"ok": True, "scope": scope, "kind": kind, "action": action, "scene": str(scene or "general")}


def record_success(agent_id: str, scene: str, preferences: Any, confirmed_fields: Any) -> dict[str, Any]:
    fields = {str(item).strip() for item in (confirmed_fields if isinstance(confirmed_fields, list) else [])}
    values = _clean_preferences(preferences)
    if fields:
        values = {key: value for key, value in values.items() if key in fields}
    values.pop("user_prompt", None)
    if not values:
        return {"ok": True, "recorded": []}
    return update(agent_id, "scene", str(scene or "general"), "soft", "set", values) | {"recorded": sorted(values)}
