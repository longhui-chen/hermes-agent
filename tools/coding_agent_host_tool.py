"""Session-scoped Coding Agent control for the authorized Coding Agent Host."""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env, zettlab_browser_session_token
from tools.registry import registry

_ACTIONS = {"providers", "session_create", "session_send", "timeline_read", "session_cancel", "permission_respond"}
_SCHEMA = {
    "name": "coding_agent_host",
    "description": "Use a verified Coding Agent provider on the authorized computer Host. Workspace is always a Host-owned alias.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": sorted(_ACTIONS)},
        "provider_id": {"type": "string", "enum": ["codex", "claude_code", "paseo"]},
        "workspace_alias": {"type": "string", "maxLength": 64},
        "session_id": {"type": "string", "maxLength": 128},
        "text": {"type": "string", "maxLength": 65536},
        "permission_id": {"type": "string", "maxLength": 128},
        "decision": {"type": "string", "enum": ["allow_once", "deny"]},
        "cursor": {"type": "string", "maxLength": 256},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
    }, "required": ["action"], "additionalProperties": False},
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


def coding_agent_host(action: str = "providers", provider_id: str = "", workspace_alias: str = "", session_id: str = "", text: str = "", permission_id: str = "", decision: str = "", cursor: str = "", limit: int = 50, **_: Any) -> str:
    if action not in _ACTIONS:
        return json.dumps({"success": False, "code": "invalid_action"})
    target_session_id = session_id
    chat_session_id = str(get_session_env("HERMES_SESSION_KEY", "") or get_session_env("HERMES_SESSION_ID", "") or "")
    token = str(zettlab_browser_session_token() or "")
    if not chat_session_id.startswith("zettlab:") or not token:
        return json.dumps({"success": False, "code": "pc_task_identity_missing"})
    try:
        wire_action = {
            "providers": "coding-agent.providers",
            "session_create": "coding-agent.session.create",
            "session_send": "coding-agent.session.send",
            "timeline_read": "coding-agent.timeline.read",
            "session_cancel": "coding-agent.session.cancel",
            "permission_respond": "coding-agent.permission.respond",
        }[action]
        params = {
            "provider_id": provider_id, "workspace_alias": workspace_alias, "session_id": session_id,
            "text": text, "permission_id": permission_id, "decision": decision, "cursor": cursor, "limit": limit,
        }
        params = {key: value for key, value in params.items() if value not in ("", None)}
        response = requests.post(
            _endpoint(),
            headers={"X-Zettlab-Agent-Action-Token": str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "")), "X-Zettlab-Browser-Session": token},
            json={"session_id": chat_session_id, "action": wire_action, "params": params},
            timeout=10,
        )
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False)
    except (requests.RequestException, ValueError) as exc:
        return json.dumps({"success": False, "code": "coding_agent_host_unavailable", "error": str(exc)[:256]})


registry.register(name="coding_agent_host", toolset="zettlab_pc", schema=_SCHEMA, handler=coding_agent_host, check_fn=_enabled, defer_to_tool_search=False)
