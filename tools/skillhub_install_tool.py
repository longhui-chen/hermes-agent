"""Native, approval-bound installation of one external SkillHub candidate."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Optional

from rich.console import Console

from tools.registry import registry, tool_error


MAX_IDENTIFIER_CHARS = 1000
PENDING_INTENT_TTL_SECONDS = 600
MAX_PENDING_INTENTS = 64


@dataclass(frozen=True)
class _PendingInstallIntent:
    session_id: str
    identifier: str
    created_turn_id: str
    candidate: dict[str, Any]
    expires_at: float


_pending_intents: OrderedDict[
    tuple[str, str], _PendingInstallIntent
] = OrderedDict()
_pending_intents_lock = threading.RLock()

_NEGATIVE_INTENT_RE = re.compile(
    r"(?:\b(?:no|cancel|deny|reject|do\s+not|don't)\b|"
    r"不(?:要|想|用)?(?:安装|装)|别装|取消|拒绝)",
    re.IGNORECASE,
)
_AFFIRMATIVE_RE = re.compile(
    r"(?:\b(?:yes|confirm|proceed|go\s+ahead|install\s+(?:it|this))\b|"
    r"确认(?:安装)?|同意安装|继续安装|安装(?:它|这个|该技能)|就装)",
    re.IGNORECASE,
)


def _validate_exact_identifier(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    identifier = value.strip()
    if (
        not identifier
        or len(identifier) > MAX_IDENTIFIER_CHARS
        or identifier.startswith("-")
        or identifier.lower().startswith(("http://", "https://"))
        or identifier.lower().startswith("official/")
        or "/" not in identifier
        or any(ord(char) < 32 or ord(char) == 127 for char in identifier)
    ):
        return None
    return identifier


def _prune_pending_intents(now: float) -> None:
    expired = [
        key
        for key, pending in _pending_intents.items()
        if pending.expires_at <= now
    ]
    for key in expired:
        _pending_intents.pop(key, None)
    while len(_pending_intents) > MAX_PENDING_INTENTS:
        _pending_intents.popitem(last=False)


def _store_pending_intent(
    *,
    session_id: str,
    turn_id: str,
    identifier: str,
    candidate: dict[str, Any],
) -> None:
    now = time.monotonic()
    key = (session_id, identifier)
    with _pending_intents_lock:
        _prune_pending_intents(now)
        _pending_intents[key] = _PendingInstallIntent(
            session_id=session_id,
            identifier=identifier,
            created_turn_id=turn_id,
            candidate=dict(candidate),
            expires_at=now + PENDING_INTENT_TTL_SECONDS,
        )
        _pending_intents.move_to_end(key)
        _prune_pending_intents(now)


def _get_pending_intent(
    session_id: str,
    identifier: str,
) -> Optional[_PendingInstallIntent]:
    now = time.monotonic()
    key = (session_id, identifier)
    with _pending_intents_lock:
        _prune_pending_intents(now)
        pending = _pending_intents.get(key)
        if pending is not None:
            _pending_intents.move_to_end(key)
        return pending


def _consume_pending_intent(session_id: str, identifier: str) -> None:
    with _pending_intents_lock:
        _pending_intents.pop((session_id, identifier), None)


def _has_direct_install_intent(message: str, identifier: str) -> bool:
    if not message or identifier not in message:
        return False
    if _NEGATIVE_INTENT_RE.search(message):
        return False
    exact = re.escape(identifier)
    direct_request = re.compile(
        rf"^\s*(?:"
        rf"(?:please\s+|can\s+you\s+|could\s+you\s+|"
        rf"i\s+(?:want|would\s+like)\s+(?:you\s+)?to\s+|"
        rf"go\s+ahead\s+and\s+)?"
        rf"(?:install|confirm\s+install(?:ation)?\s+of)"
        rf"(?:\s+(?:the\s+)?skill)?\s+{exact}\b"
        rf"(?:\s+(?:please|for\s+me|now))?"
        rf"|(?:请|麻烦|帮我|给我|直接|现在|立即|就|"
        rf"我(?:要|想|确认要))?"
        rf"(?:下载并安装|安装|装上)\s*(?:这个|该)?(?:\s*技能)?\s*{exact}"
        rf"|{exact}\s*(?:请)?(?:直接|现在|立即)?(?:安装|装上)"
        rf")\s*(?:吧|[.!?。！？])?\s*$",
        re.IGNORECASE,
    )
    return direct_request.search(message) is not None


def _confirms_pending_intent(
    *,
    pending: _PendingInstallIntent,
    turn_id: str,
    user_message: str,
    previous_assistant_message: str,
) -> bool:
    if not turn_id or turn_id == pending.created_turn_id:
        return False
    if pending.identifier not in previous_assistant_message:
        return False
    source_url = pending.candidate.get("source_url")
    if not isinstance(source_url, str) or source_url not in previous_assistant_message:
        return False
    if _NEGATIVE_INTENT_RE.search(user_message or ""):
        return False
    return _AFFIRMATIVE_RE.search(user_message or "") is not None


def _read_lock_entries() -> Optional[dict[str, dict[str, Any]]]:
    """Read authoritative install state; None means reconciliation is unknown."""
    from tools.skills_hub import LOCK_FILE

    path = Path(LOCK_FILE)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    installed = payload.get("installed")
    if not isinstance(installed, dict):
        return None
    return {
        name: entry
        for name, entry in installed.items()
        if isinstance(name, str) and isinstance(entry, dict)
    }


def _matching_entries(
    entries: Optional[dict[str, dict[str, Any]]],
    identifier: str,
) -> Optional[dict[str, dict[str, Any]]]:
    if entries is None:
        return None
    return {
        name: entry
        for name, entry in entries.items()
        if entry.get("identifier") == identifier
    }


def _verify_entries_on_disk(
    entries: Optional[dict[str, dict[str, Any]]],
) -> Optional[dict[str, dict[str, Any]]]:
    """Return entries only when their installed directories and hashes agree."""
    if entries is None:
        return None
    from tools.skills_guard import content_hash
    from tools.skills_hub import SKILLS_DIR

    skills_root = Path(SKILLS_DIR).resolve()
    for entry in entries.values():
        install_path = entry.get("install_path")
        expected_hash = entry.get("content_hash")
        if not isinstance(install_path, str) or not isinstance(expected_hash, str):
            return None
        candidate = (skills_root / install_path).resolve()
        if (
            not candidate.is_relative_to(skills_root)
            or not candidate.is_dir()
            or content_hash(candidate) != expected_hash
        ):
            return None
    return entries


def skillhub_install(
    identifier: Any,
    *,
    session_id: str = "",
    turn_id: str = "",
    user_message: str = "",
    previous_assistant_message: str = "",
) -> str:
    """Prepare or install one exact candidate from runtime-bound user intent."""
    exact_identifier = _validate_exact_identifier(identifier)
    if exact_identifier is None:
        return tool_error(
            "identifier must be an exact non-official, non-URL registry "
            "identifier containing '/' and no control characters"
        )
    if not session_id or not turn_id:
        return tool_error(
            "skillhub_install requires live session and turn context; no "
            "installation was attempted"
        )

    before = _verify_entries_on_disk(
        _matching_entries(_read_lock_entries(), exact_identifier)
    )
    if before is None:
        return json.dumps(
            {
                "status": "unknown_reconcile",
                "identifier": exact_identifier,
                "message": "Could not read the SkillHub lock before installation; no mutation was attempted.",
            },
            ensure_ascii=False,
        )
    if before:
        skill_name, entry = next(iter(before.items()))
        return json.dumps(
            {
                "status": "already_installed",
                "identifier": exact_identifier,
                "skill_name": skill_name,
                "source": entry.get("source"),
                "content_hash": entry.get("content_hash"),
                "install_path": entry.get("install_path"),
            },
            ensure_ascii=False,
        )

    pending = _get_pending_intent(session_id, exact_identifier)
    direct_intent = _has_direct_install_intent(user_message, exact_identifier)
    pending_confirmed = (
        pending is not None
        and _confirms_pending_intent(
            pending=pending,
            turn_id=turn_id,
            user_message=user_message,
            previous_assistant_message=previous_assistant_message,
        )
    )
    intent_confirmed = direct_intent or pending_confirmed
    expected_candidate = pending.candidate if pending is not None else None

    if pending is not None and _NEGATIVE_INTENT_RE.search(user_message or ""):
        _consume_pending_intent(session_id, exact_identifier)
        return json.dumps(
            {
                "status": "confirmation_declined",
                "identifier": exact_identifier,
                "message": (
                    "The pending install intent was cancelled; no mutation "
                    "was attempted."
                ),
            },
            ensure_ascii=False,
        )
    if intent_confirmed and pending is not None:
        _consume_pending_intent(session_id, exact_identifier)

    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=120)
    from hermes_cli.skills_hub import do_agent_install

    try:
        flow_result = do_agent_install(
            exact_identifier,
            console=console,
            intent_confirmed=intent_confirmed,
            expected_candidate=expected_candidate,
        )
    except Exception as exc:
        output = stream.getvalue().strip()
        return json.dumps(
            {
                "status": "unknown_reconcile",
                "identifier": exact_identifier,
                "message": (
                    "The install flow raised after it started; filesystem and "
                    "lock state may be partially changed. Do not retry until "
                    "state is reconciled."
                ),
                "installer_error": f"{type(exc).__name__}: {exc}",
                "installer_output": output,
            },
            ensure_ascii=False,
        )
    output = stream.getvalue().strip()

    if isinstance(flow_result, dict) and flow_result.get("status") in {
        "confirmation_required",
        "candidate_changed",
    }:
        candidate = flow_result.get("candidate")
        if isinstance(candidate, dict):
            _store_pending_intent(
                session_id=session_id,
                turn_id=turn_id,
                identifier=exact_identifier,
                candidate=candidate,
            )
        return json.dumps(
            {
                **flow_result,
                "message": (
                    "Disclose this exact candidate and source to the user, then "
                    "end the turn. A later user-authored confirmation can "
                    "authorize installation."
                ),
                "installer_output": output,
            },
            ensure_ascii=False,
        )
    if isinstance(flow_result, dict) and flow_result.get("status") == "risk_denied":
        return json.dumps(
            {**flow_result, "installer_output": output},
            ensure_ascii=False,
        )

    after = _verify_entries_on_disk(
        _matching_entries(_read_lock_entries(), exact_identifier)
    )
    if after is None:
        return json.dumps(
            {
                "status": "unknown_reconcile",
                "identifier": exact_identifier,
                "message": (
                    "The install flow returned, but authoritative lock state "
                    "could not be read. Do not retry until state is reconciled."
                ),
                "installer_output": output,
            },
            ensure_ascii=False,
        )

    if not after:
        return json.dumps(
            {
                "status": "failed_before_effect",
                "identifier": exact_identifier,
                "installer_output": output,
            },
            ensure_ascii=False,
        )

    skill_name, entry = next(iter(after.items()))
    return json.dumps(
        {
            "status": "installed",
            "identifier": exact_identifier,
            "skill_name": skill_name,
            "source": entry.get("source"),
            "trust_level": entry.get("trust_level"),
            "scan_verdict": entry.get("scan_verdict"),
            "content_hash": entry.get("content_hash"),
            "install_path": entry.get("install_path"),
            "installer_output": output,
        },
        ensure_ascii=False,
    )


SKILLHUB_INSTALL_SCHEMA = {
    "name": "skillhub_install",
    "description": (
        "Prepare or install one exact external skill registry candidate. "
        "Without a runtime-verifiable user install choice, it scans and returns "
        "a candidate that must be disclosed and confirmed in a later user turn. "
        "A safe confirmed candidate installs without another prompt; scan "
        "findings trigger a one-operation risk decision. Never use for ZettLab "
        "catalog items."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "identifier": {
                "type": "string",
                "description": (
                    "Exact non-URL identifier returned by external skill search "
                    "and already disclosed to the user."
                ),
            }
        },
        "required": ["identifier"],
    },
}


registry.register(
    name="skillhub_install",
    toolset="skills",
    schema=SKILLHUB_INSTALL_SCHEMA,
    handler=lambda args, **kw: skillhub_install(
        args.get("identifier"),
        session_id=str(kw.get("session_id") or ""),
        turn_id=str(kw.get("turn_id") or ""),
        user_message=str(kw.get("user_task") or ""),
        previous_assistant_message=str(
            kw.get("previous_assistant_message") or ""
        ),
    ),
    emoji="🧩",
)
