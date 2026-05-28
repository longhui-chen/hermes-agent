"""Per-session "last seen effective model" state, owned by Hermes.

Used by the zet_agent platform to inject a one-shot model-identity note when
a session's effective model changes (session override or agent-level config
default). Unlike ``session_model_overrides.json`` (written by local-server),
this file is Hermes-owned: we both read and write it, and it persists the
per-session last-seen model across gateway restarts so a model switch is still
announced on the session's next turn even if the process bounced in between.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SEEN_FILENAME = "session_seen_models.json"


def seen_path(home: Path | None = None) -> Path:
    """Return the profile-local seen-models JSON path."""
    return (home or get_hermes_home()) / SEEN_FILENAME


def load_seen_models(path: Path | None = None) -> Dict[str, str]:
    """Load the persisted ``{session_id: model}`` map. Tolerant of a missing
    or malformed file (returns an empty map)."""
    target = path or seen_path()
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("session-seen-models: read failed for %s: %s", target, exc)
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("session-seen-models: invalid json in %s: %s", target, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: v
        for k, v in data.items()
        if isinstance(k, str) and k and isinstance(v, str) and v
    }


def save_seen_models(models: Dict[str, str], path: Path | None = None) -> None:
    """Atomically persist the ``{session_id: model}`` map (tmp write + rename).

    The tmp file carries a pid+uuid suffix so concurrent writers never share
    one tmp path (a fixed name would let one writer ``os.replace`` a tmp the
    other is still writing — a truncated file could become the live one).
    """
    target = path or seen_path()
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(models, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("session-seen-models: write failed for %s: %s", target, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
