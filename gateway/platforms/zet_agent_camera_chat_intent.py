"""Credential-free camera Chat proposals for the three monitoring layers."""

import re

_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

CAMERA_LIVE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["camera_id"],
    "properties": {"camera_id": {"type": "string", "maxLength": 128}},
    "description": "Camera only: request a short-lived live viewer lease. The client opens and stops it; no recording is implied.",
}

CAMERA_RECORDING_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["camera_id"],
    "properties": {
        "camera_id": {"type": "string", "maxLength": 128},
        "retention_days": {"type": "integer", "enum": [0, 1, 7, 30]},
    },
    "description": "Camera only: propose continuous recording. The trusted client must prepare and require a second confirmation before recording starts.",
}


def _validate_ref(value: object) -> bool:
    return isinstance(value, str) and bool(_REF.fullmatch(value))


def normalize_camera_live_setup(value: dict) -> dict:
    if set(value) != {"resource_kind", "live"} or value.get("resource_kind") != "camera":
        raise ValueError("camera_live_invalid")
    raw = value["live"]
    if not isinstance(raw, dict) or set(raw) != {"camera_id"} or not _validate_ref(raw.get("camera_id")):
        raise ValueError("camera_live_invalid")
    return {"resource_kind": "camera", "live": {"camera_id": raw["camera_id"]}}


def normalize_camera_recording_setup(value: dict) -> dict:
    if set(value) != {"resource_kind", "recording"} or value.get("resource_kind") != "camera":
        raise ValueError("camera_recording_invalid")
    raw = value["recording"]
    if not isinstance(raw, dict) or set(raw) - {"camera_id", "retention_days"} or not _validate_ref(raw.get("camera_id")):
        raise ValueError("camera_recording_invalid")
    if "retention_days" in raw and raw["retention_days"] not in (0, 1, 7, 30):
        raise ValueError("camera_recording_invalid")
    return {"resource_kind": "camera", "recording": dict(raw)}


def camera_confirmation_allowed(arguments, camera_ids) -> bool:
    """Allow only a credential-free proposal for a discovered camera."""
    from gateway.platforms.zet_agent_connector_setup_intent import normalize_connector_setup

    if set(arguments) - {"question", "connector_setup", "choices", "multi_select"}:
        return False
    if arguments.get("choices") or arguments.get("multi_select"):
        return False
    try:
        intent = normalize_connector_setup(arguments.get("connector_setup"))
    except (ValueError, TypeError):
        return False
    if intent.get("resource_kind") != "camera":
        return False
    proposal = next((intent[key] for key in ("recording", "live", "observation") if key in intent), None)
    return isinstance(proposal, dict) and proposal.get("camera_id") in camera_ids
