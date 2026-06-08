"""Session model override blobs written by the Zettlab local-server."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

OVERRIDES_FILENAME = "session_model_overrides.json"
RUNTIME_KEYS = ("model", "provider", "api_key", "base_url", "api_mode")


def overrides_path(home: Path | None = None) -> Path:
    """Return the profile-local override JSON path."""
    return (home or get_hermes_home()) / OVERRIDES_FILENAME


def load_session_model_overrides(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load profile-local session model overrides.

    The file is owned by local-server and is intentionally treated as an
    opaque control-plane blob. Hermes only consumes runtime keys it already
    understands and ignores malformed entries.
    """
    target = path or overrides_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("session-model-overrides: read failed for %s: %s", target, exc)
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("session-model-overrides: invalid json in %s: %s", target, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("session-model-overrides: expected object in %s", target)
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for session_id, override in data.items():
        if not isinstance(session_id, str) or not session_id:
            continue
        if not isinstance(override, dict):
            continue
        model = override.get("model")
        if not isinstance(model, str) or not model:
            continue

        cleaned: Dict[str, Any] = {"model": model}
        for key in RUNTIME_KEYS[1:]:
            value = override.get(key)
            if value is not None:
                cleaned[key] = value

        # Preserve non-runtime metadata for introspection while runtime code
        # continues to whitelist what it applies.
        for key, value in override.items():
            if key not in cleaned:
                cleaned[key] = value
        out[session_id] = cleaned
    return out
