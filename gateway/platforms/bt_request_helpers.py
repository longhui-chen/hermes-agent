"""Side-effect-free request metadata and trusted input parsing helpers."""

import base64
import binascii
import re
from typing import Any, Dict, List

MAX_CANONICAL_FINAL_TURN_ID_LEN = 200

MAX_TURN_ID_LEN = MAX_CANONICAL_FINAL_TURN_ID_LEN

_HARDWARE_EXECUTION_TOKEN_HEADER = "X-Zettlab-Hardware-Execution-Token"

_VIDEO_EDIT_SKILL_SLUGS = frozenset({
    "video-edit-workflow-mini",
    "video-edit-workflow",
    "video-edit",
    "video_edit",
})

_CURRENT_TURN_IMAGE_MAX_BYTES = 5 * 1024 * 1024

_CURRENT_TURN_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp"})

def _extract_turn_id(body: Dict[str, Any]) -> str:
    """Extract metadata.turn_id (zettlab local-server's per-turn correlation
    token) so the NAS agent-search fallback can echo it back as the
    X-Zettlab-Turn-Id header. Reject only what would corrupt that header
    (whitespace / control chars); don't restrict the charset further —
    local-server accepts any trimmed token, so a stricter filter would silently
    drop valid ids and lose the precise-turn pinning."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("turn_id", metadata.get("turnId", ""))
    tid = str(raw or "").strip()
    if not tid or any(c.isspace() or ord(c) < 0x20 for c in tid):
        return ""
    # The id is echoed onto every extension frame of the turn (HERMES_TURN_ID →
    # zet_agent `_stamp_extension_turn_id`), so an oversized value would be
    # re-serialised hundreds of times per streamed turn. Anything past the cap
    # is dropped rather than truncated (a truncated id would silently
    # mis-correlate); the cap is the canonical-final one so both paths agree.
    if len(tid) > MAX_TURN_ID_LEN:
        return ""
    return tid

def _extract_connector_route_capability(body: Dict[str, Any]) -> str:
    """Extract local-server's opaque per-turn Connector routing capability.

    The fixed 32-byte base64url shape keeps malformed or oversized metadata
    out of the dedicated runner environment. It is transport-only and never
    becomes a general session variable or model-visible instruction.
    """
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("connector_route_capability")
    if not isinstance(raw, str):
        return ""
    capability = raw.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{43}", capability) is None:
        return ""
    return capability

def _extract_skill_slug(body: Dict[str, Any]) -> str:
    """Extract metadata.skill_slug — the App quick-pick's EXPLICIT skill
    invocation signal (ZET fork).

    The client owns the text↔selection UX (it drops the field when the user
    edits the inserted "/<slug>" token away); the server NEVER sniffs message
    text for slash commands — in-band signaling is ambiguous ("/<skill> 是什么"
    would fire the skill) and this explicit field is the only trigger.
    Absent/malformed → no skill. A leading slash is tolerated and stripped so
    the client may send either "deep-research" or "/deep-research"."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("skill_slug", metadata.get("skillSlug", ""))
    slug = str(raw or "").strip().lstrip("/")
    if not slug or any(c.isspace() or ord(c) < 0x20 for c in slug):
        return ""
    return slug

def _extract_connector_policy_disabled_skills(body: Dict[str, Any]) -> tuple[str, ...]:
    """Read the machine-authored Chat visibility overlay from metadata."""
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ()
    raw = metadata.get("connector_policy_disabled_skills")
    if not isinstance(raw, list) or len(raw) > 2048:
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        skill = str(value or "").strip()
        if re.fullmatch(r"[a-z][a-z0-9_-]{1,127}", skill) and skill not in seen:
            seen.add(skill)
            out.append(skill)
    return tuple(out)

def _extract_hardware_execution_token(request: Any) -> str:
    """Relay the dedicated capability only to trusted hardware helpers."""
    if request is None:
        return ""
    token = str(
        request.headers.get(_HARDWARE_EXECUTION_TOKEN_HEADER, "") or ""
    ).strip()
    return token if re.fullmatch(r"[0-9a-f]{64}", token) is not None else ""

def _extract_creation_action_receipt_transport(body: Dict[str, Any]) -> str:
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("creation_action_receipt_transport")
    if raw == "canonical_final_v1":
        return "canonical_final_v1"
    return ""

def _strip_skill_display_token(user_message: Any, skill_slug: str) -> Any:
    """Remove only the App quick-pick token from a string user task."""
    if not isinstance(user_message, str) or not skill_slug:
        return user_message
    token = "/" + skill_slug
    task_text = re.sub(
        r"(?<!\S)" + re.escape(token) + r"(?!\S)",
        "",
        user_message,
    )
    return "\n".join(
        line for line in (value.rstrip() for value in task_text.splitlines()) if line
    ).strip()

def _trusted_skill_task_message(user_message: Any, skill_slug: str) -> Any:
    """Preserve user-authored task text separately from transport selection."""
    return _strip_skill_display_token(user_message, skill_slug)

def _extract_requested_execution_policy(body: Dict[str, Any]) -> str:
    metadata = body.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    raw = metadata.get("execution_policy", metadata.get("executionPolicy", ""))
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()

def _extract_current_turn_reference_image(content: Any) -> str:
    """Return one bounded data image from the current normalized user turn.

    This is intentionally fail-closed without rejecting the surrounding chat:
    remote URLs, malformed bytes, unsupported formats, and turns containing a
    second image simply do not grant the desktop-pet tool image access.
    """
    if not isinstance(content, list):
        return ""
    image_urls: List[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        image_ref = part.get("image_url")
        value = image_ref.get("url") if isinstance(image_ref, dict) else None
        if isinstance(value, str) and value.strip():
            image_urls.append(value.strip())
    if len(image_urls) != 1:
        return ""

    value = image_urls[0]
    header, separator, encoded = value.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        return ""
    declared_mime = header[len("data:") : -len(";base64")].lower()
    if declared_mime not in _CURRENT_TURN_IMAGE_MIMES or not encoded:
        return ""
    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    if len(encoded) % 4 != 0:
        return ""
    decoded_size = len(encoded) // 4 * 3 - padding
    if decoded_size <= 0 or decoded_size > _CURRENT_TURN_IMAGE_MAX_BYTES:
        return ""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return ""
    if declared_mime == "image/png":
        valid_magic = raw.startswith(b"\x89PNG\r\n\x1a\n")
    elif declared_mime == "image/jpeg":
        valid_magic = raw.startswith(b"\xff\xd8\xff")
    else:
        valid_magic = len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP"
    return value if valid_magic else ""

def _is_video_edit_skill_slug(slug: str) -> bool:
    normalized = str(slug or "").strip().lower().strip("/")
    return normalized in _VIDEO_EDIT_SKILL_SLUGS
