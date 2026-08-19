"""Session-scoped access to a user-authorized directory on a connected PC."""

from __future__ import annotations

import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env, zettlab_browser_session_token
from tools.registry import registry

_MAX_RESPONSE_BYTES = 512 << 10
_TIMEOUT_SECONDS = 35

PC_FILE_SCHEMA = {
    "name": "pc_file",
    "description": (
        "List or read files inside the directory the user explicitly authorized "
        "on their connected desktop computer. If the user asks to view or list "
        "computer files without naming a subdirectory, immediately list path '.'; "
        "do not ask them for a directory path. Paths are always relative to the "
        "authorized directory; never ask for or invent an absolute local path."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "read"]},
            "path": {
                "type": "string",
                "description": (
                    "Path relative to the authorized directory. Use '.' for the "
                    "authorized root whenever the user did not name a subdirectory."
                ),
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "offset": {"type": "integer", "minimum": 0},
        },
        "required": ["action"],
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
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/internal/pc/action", "", ""))


def _session_id() -> str:
    try:
        value = str(get_session_env("HERMES_SESSION_KEY", "") or get_session_env("HERMES_SESSION_ID", "") or "").strip()
    except Exception:
        return ""
    return value if value.startswith("zettlab:") else ""


def _session_token() -> str:
    try:
        return str(zettlab_browser_session_token() or "").strip()
    except Exception:
        return ""


def _check_pc_file() -> bool:
    return bool(
        _endpoint()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
        and _session_token()
        and _session_id()
    )


_check_pc_file._profile_scope_sensitive = True  # type: ignore[attr-defined]


def pc_file_tool(args: dict[str, Any], **_: Any) -> str:
    action = args.get("action")
    if action not in {"list", "read"}:
        return json.dumps({"success": False, "code": "invalid_action"})
    params: dict[str, Any] = {"path": args.get("path", ".")}
    if "limit" in args:
        params["limit"] = args["limit"]
    if "offset" in args:
        params["offset"] = args["offset"]
    try:
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(
                _endpoint(),
                headers={
                    "Content-Type": "application/json",
                    "X-Zettlab-Agent-Action-Token": str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip(),
                    "X-Zettlab-Browser-Session-Token": _session_token(),
                },
                json={
                    "session_id": _session_id(),
                    "action": f"file.{action}",
                    "params": params,
                },
                timeout=_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        return json.dumps({"success": False, "code": "pc_host_unavailable", "error": str(exc)[:512]}, ensure_ascii=False)
    if len(response.content) > _MAX_RESPONSE_BYTES:
        return json.dumps({"success": False, "code": "pc_response_too_large"})
    try:
        payload = response.json()
    except ValueError:
        return json.dumps({"success": False, "code": "invalid_pc_response"})
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="pc_file",
    toolset="zettlab_pc",
    schema=PC_FILE_SCHEMA,
    handler=pc_file_tool,
    check_fn=_check_pc_file,
    emoji="🖥️",
)
