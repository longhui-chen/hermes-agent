"""Credential-free finite observation proposal; no mutations or consent."""

import re

_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
CAMERA_OBSERVATION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["camera_id", "duration_seconds", "subject_kind", "predicate"],
    "properties": {
        "camera_id": {"type": "string", "maxLength": 128},
        "duration_seconds": {"type": "integer", "minimum": 1, "maximum": 2147483647},
        "subject_kind": {"type": "string", "enum": ["person", "object"]},
        "predicate": {"type": "string", "enum": ["appears", "disappears", "enters_zone", "leaves_zone", "lingers"]},
        "subject_ref": {"type": "string", "maxLength": 128},
        "zone_id": {"type": "string", "maxLength": 128},
        "min_duration_seconds": {"type": "integer", "minimum": 0, "maximum": 2147483647},
    },
    "description": "Camera only: propose a finite future observation for explicit client confirmation. Never substitute for a historical request. References are not identity evidence; omit unknown optional references, never guess names. Owner, device, Memo, consent, credentials and jobs are client-owned, not proposal fields.",
}


def normalize_camera_observation_setup(value: dict) -> dict:
    if set(value) != {"resource_kind", "observation"} or value.get("resource_kind") != "camera":
        raise ValueError("camera_observation_invalid")
    raw = value["observation"]
    if not isinstance(raw, dict) or set(raw) - set(CAMERA_OBSERVATION_SCHEMA["properties"]):
        raise ValueError("camera_observation_invalid")
    duration = raw.get("duration_seconds")
    minimum = raw.get("min_duration_seconds", 0)
    if type(duration) is not int or not 0 < duration <= 2147483647 or type(minimum) is not int or not 0 <= minimum <= duration:
        raise ValueError("camera_observation_invalid")
    for key in ("camera_id", "subject_ref", "zone_id"):
        if key == "camera_id" or key in raw:
            if not isinstance(raw.get(key), str) or not _REF.fullmatch(raw[key]):
                raise ValueError("camera_observation_invalid")
    if raw.get("subject_kind") not in ("person", "object") or raw.get("predicate") not in CAMERA_OBSERVATION_SCHEMA["properties"]["predicate"]["enum"]:
        raise ValueError("camera_observation_invalid")
    if raw["predicate"] in ("enters_zone", "leaves_zone") and not raw.get("zone_id"):
        raise ValueError("camera_observation_invalid")
    return {"resource_kind": "camera", "observation": dict(raw)}
