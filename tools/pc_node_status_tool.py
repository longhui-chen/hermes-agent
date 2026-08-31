"""Read-only status for the PC node authorized to the current Chat."""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env, zettlab_browser_session_token
from tools.registry import registry

_MAX_RESPONSE_BYTES = 64 << 10
_TIMEOUT_SECONDS = 10

PC_NODE_STATUS_SCHEMA = {
    "name": "pc_node_status",
    "description": (
        "Check whether the desktop PC node authorized for this Chat is currently "
        "online and which normalized capability groups are ready. Use this for PC "
        "connection/status inventory requests. It reads no files, captures no "
        "screen, and performs no computer action."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _endpoint() -> str:
    raw = str(get_secret("ZETTLAB_BROWSER_ACTION_URL", "") or "").strip()
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        trusted = host == "localhost" or ipaddress.ip_address(host).is_loopback
        if parsed.scheme != "http" or not parsed.port or not trusted:
            return ""
    except (ValueError, TypeError):
        return ""
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        "/api/v1/internal/pc/action",
        "",
        "",
    ))


def _session_id() -> str:
    try:
        value = str(
            get_session_env("HERMES_SESSION_KEY", "")
            or get_session_env("HERMES_SESSION_ID", "")
            or ""
        ).strip()
    except Exception:
        return ""
    return value if value.startswith("zettlab:") else ""


def _session_token() -> str:
    try:
        return str(zettlab_browser_session_token() or "").strip()
    except Exception:
        return ""


def _check_pc_node_status() -> bool:
    return bool(
        _endpoint()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
        and _session_token()
        and _session_id()
    )


_check_pc_node_status._profile_scope_sensitive = True  # type: ignore[attr-defined]


def pc_node_status_tool(_: dict[str, Any], **__: Any) -> str:
    try:
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(
                _endpoint(),
                headers={
                    "Content-Type": "application/json",
                    "X-Zettlab-Agent-Action-Token": str(
                        get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or ""
                    ).strip(),
                    "X-Zettlab-Browser-Session-Token": _session_token(),
                },
                json={"session_id": _session_id(), "action": "status"},
                timeout=_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        return json.dumps(
            {
                "success": False,
                "code": "pc_status_unavailable",
                "error": str(exc)[:512],
            },
            ensure_ascii=False,
        )
    if len(response.content) > _MAX_RESPONSE_BYTES:
        return json.dumps({"success": False, "code": "pc_response_too_large"})
    try:
        payload = response.json()
    except ValueError:
        return json.dumps({"success": False, "code": "invalid_pc_response"})
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="pc_node_status",
    toolset="zettlab_pc",
    schema=PC_NODE_STATUS_SCHEMA,
    handler=pc_node_status_tool,
    check_fn=_check_pc_node_status,
    emoji="🖥️",
    # Connection inventory is part of the trusted-client control plane. Keep
    # this schema directly visible once the Chat-bound PC toolset is selected;
    # otherwise a status request would first depend on Tool Search, which can
    # itself be unavailable before the PC authorization state is known.
    defer_to_tool_search=False,
)
