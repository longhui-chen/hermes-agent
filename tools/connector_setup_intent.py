"""Credential-free intent validation for the trusted Memo composer handoff.

This module never installs, creates, authorizes or invokes a Connector. The
existing clarify interaction owns delivery, cancellation and task continuation.
"""

import json
import re
from urllib.parse import urlsplit

_KINDS = ("custom_api", "custom_mcp", "saas", "camera", "printer3d", "tv", "pc_node")
_FIELDS = {"resource_kind", "template_id", "provider_id", "url", "auth_kind"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

CONNECTOR_SETUP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resource_kind": {"type": "string", "enum": list(_KINDS)},
        "template_id": {"type": "string", "maxLength": 128},
        "provider_id": {"type": "string", "maxLength": 128},
        "url": {"type": "string", "maxLength": 2048},
        "auth_kind": {"type": "string", "enum": ["none", "bearer", "basic", "header"]},
    },
    "required": ["resource_kind"],
    "description": (
        "Optional Memo connector handoff. Ask the trusted client to collect only "
        "missing configuration in the current composer. Never put passwords, "
        "tokens, hardware addresses or serial numbers here. Custom API requires "
        "a published template_id; SaaS requires provider_id. Only remote MCP may "
        "include a credential-free URL. The client must support this interaction; "
        "otherwise the tool fails without asking the user to enter credentials. "
        "Do not ask for a secret in prose before this handoff."
    ),
}


def normalize_connector_setup(value: object) -> dict:
    """Reject unknown fields rather than forwarding arbitrary configuration."""
    if not isinstance(value, dict) or set(value) - _FIELDS:
        raise ValueError("connector_setup_invalid")
    if any(not isinstance(item, str) for item in value.values()):
        raise ValueError("connector_setup_invalid")
    kind = value.get("resource_kind")
    if kind not in _KINDS:
        raise ValueError("connector_setup_invalid")
    result = {"resource_kind": kind}
    allowed = {"resource_kind"}
    if kind == "custom_api":
        allowed.add("template_id")
        if not _ID.fullmatch(value.get("template_id", "")):
            raise ValueError("connector_setup_template_required")
        result["template_id"] = value["template_id"]
    elif kind == "saas":
        allowed.add("provider_id")
        if not _ID.fullmatch(value.get("provider_id", "")):
            raise ValueError("connector_setup_provider_required")
        result["provider_id"] = value["provider_id"]
    elif kind == "custom_mcp":
        allowed.update(("url", "auth_kind"))
        if "url" in value:
            url = value["url"]
            try:
                parts = urlsplit(url)
                valid = (0 < len(url) <= 2048 and parts.scheme == "https" and parts.hostname
                         and not parts.username and not parts.password and not parts.query
                         and not parts.fragment and not any(c.isspace() for c in url))
                _ = parts.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("connector_setup_url_invalid")
            result["url"] = url
        if "auth_kind" in value:
            if value["auth_kind"] not in ("none", "bearer", "basic", "header"):
                raise ValueError("connector_setup_auth_invalid")
            result["auth_kind"] = value["auth_kind"]
    if set(value) - allowed:
        raise ValueError("connector_setup_invalid")
    return result


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
        result["next_step"] = "Verify current connector availability and session authorization before claiming success or resuming the original authorized task. This is a client submission receipt, not a grant."
    return json.dumps(result)
