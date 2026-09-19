"""Credential-free camera confirmation intent on the legacy clarify wire.

Software Connector creation uses ordinary Chat and connector_chat_create.
The legacy wire is retained only for camera observation/live/recording consent.
"""

import json

# zettlab-overlay(ac432-observation-intent): delegate camera proposal validation to adapter; upstream: none
from gateway.platforms.zet_agent_camera_observation_intent import CAMERA_OBSERVATION_SCHEMA, normalize_camera_observation_setup
# zettlab-overlay(ac432-chat-camera-layers): keep all camera Chat proposals credential-free; upstream: none
from gateway.platforms.zet_agent_camera_chat_intent import CAMERA_LIVE_SCHEMA, CAMERA_RECORDING_SCHEMA, normalize_camera_live_setup, normalize_camera_recording_setup

import re

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

CONNECTOR_SETUP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        # zettlab-overlay(ac432-observation-intent): optional camera proposal schema only; upstream: none
        "observation": CAMERA_OBSERVATION_SCHEMA,
        "live": CAMERA_LIVE_SCHEMA,
        "recording": CAMERA_RECORDING_SCHEMA,
        "resource_kind": {"type": "string", "enum": ["camera"]},
    },
    "required": ["resource_kind"],
    "description": "Camera consent only: exactly one observation, live or recording proposal. Software Connector setup must use ordinary Chat messages and connector_chat_create, never this clarify path.",
}


def normalize_connector_setup(value: object) -> dict:
    """Reject the retired software/hardware setup path; preserve camera consent."""
    if not isinstance(value, dict) or value.get("resource_kind") != "camera":
        raise ValueError("connector_setup_invalid")
    proposals = [key for key in ("observation", "live", "recording") if key in value]
    if len(proposals) != 1:
        raise ValueError("connector_setup_invalid")
    if proposals[0] == "observation":
        return normalize_camera_observation_setup(value)
    if proposals[0] == "live":
        return normalize_camera_live_setup(value)
    return normalize_camera_recording_setup(value)


def connector_setup_result(raw: object) -> str:
    """Only bounded status reaches the model; never return free-form answers."""
    try:
        payload = json.loads(raw) if isinstance(raw, str) and len(raw) <= 512 else None
    except (TypeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or set(payload) - {"status", "target_id"}:
        return json.dumps({"status": "cancelled"})
    if payload.get("status") not in ("submitted", "cancelled", "failed"):
        return json.dumps({"status": "cancelled"})
    result = {"status": payload["status"]}
    if payload.get("target_id"):
        if not isinstance(payload["target_id"], str) or not _ID.fullmatch(payload["target_id"]):
            return json.dumps({"status": "cancelled"})
        result["target_id"] = payload["target_id"]
    if result["status"] == "submitted":
        result["next_step"] = "This camera confirmation receipt contains no credentials and is not a grant. Verify the current camera state before claiming success or resuming the task."
    return json.dumps(result)
