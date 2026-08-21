"""Read-only Provider discovery for the authorized Coding Agent Host."""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env, zettlab_browser_session_token
from tools.registry import registry

_SCHEMA = {
    "name": "coding_agent_host",
    "description": "Discover Coding Agent providers on the authorized computer Host. This is read-only.",
    "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["providers"]}}, "required": ["action"], "additionalProperties": False},
}


def _endpoint() -> str:
    raw = str(get_secret("ZETTLAB_BROWSER_ACTION_URL", "") or "").strip()
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or not parsed.port or not ipaddress.ip_address(host).is_loopback:
            return ""
    except (ValueError, TypeError):
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/internal/pc/action", "", ""))


def _enabled() -> bool:
    return bool(_endpoint() and get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") and zettlab_browser_session_token())


_enabled._profile_scope_sensitive = True  # type: ignore[attr-defined]


def coding_agent_host(action: str = "providers", **_: Any) -> str:
    if action != "providers":
        return json.dumps({"success": False, "code": "invalid_action"})
    session_id = str(get_session_env("HERMES_SESSION_KEY", "") or get_session_env("HERMES_SESSION_ID", "") or "")
    token = str(zettlab_browser_session_token() or "")
    if not session_id.startswith("zettlab:") or not token:
        return json.dumps({"success": False, "code": "pc_task_identity_missing"})
    try:
        response = requests.post(
            _endpoint(),
            headers={"X-Zettlab-Agent-Action-Token": str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "")), "X-Zettlab-Browser-Session": token},
            json={"session_id": session_id, "action": "coding-agent.providers", "params": {}},
            timeout=10,
        )
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False)
    except (requests.RequestException, ValueError) as exc:
        return json.dumps({"success": False, "code": "coding_agent_host_unavailable", "error": str(exc)[:256]})


registry.register(name="coding_agent_host", toolset="zettlab_pc", schema=_SCHEMA, handler=coding_agent_host, check_fn=_enabled, defer_to_tool_search=False)
