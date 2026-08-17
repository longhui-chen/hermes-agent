"""Session-scoped semantic Computer Use on a locally approved desktop."""

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
_ACTIONS = {"launch", "snapshot", "focus", "invoke", "set_value", "scroll", "keystroke"}

PC_UI_SCHEMA = {
    "name": "pc_ui",
    "description": (
        "Use semantic Computer Use on the connected desktop after the user approves it locally. "
        "If the application has no visible window, launch it first. Start every interaction with snapshot once a window exists, and name the application exactly; "
        "for example use Google Chrome rather than Browser or 浏览器. Reuse only the pid, window_id and element "
        "indices returned by that fresh snapshot; never invent selectors, paths, shell commands, "
        "scripts or CDP requests. Secure controls are hidden and sensitive actions may require "
        "another confirmation on the computer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(_ACTIONS)},
            "app": {
                "type": "string",
                "maxLength": 256,
                "description": "Exact application name. Required for launch and snapshot.",
            },
            "pid": {"type": "integer", "minimum": 1},
            "window_id": {"type": "integer", "minimum": 1},
            "element": {
                "type": "integer",
                "minimum": 1,
                "description": "Element index from the latest snapshot for this Chat session.",
            },
            "value": {"type": "string", "maxLength": 8192},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 20},
            "key": {
                "type": "string",
                "maxLength": 64,
                "description": "One key or a plus-separated hotkey such as CMD+L.",
            },
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


def _check_pc_ui() -> bool:
    return bool(
        _endpoint()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
        and _session_token()
        and _session_id()
    )


_check_pc_ui._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _params(args: dict[str, Any], action: str) -> dict[str, Any] | None:
    allowed = {
        "launch": ("app",),
        "snapshot": ("app",),
        "focus": ("pid", "window_id"),
        "invoke": ("pid", "window_id", "element"),
        "set_value": ("pid", "window_id", "element", "value"),
        "scroll": ("pid", "window_id", "element", "direction", "amount"),
        "keystroke": ("pid", "window_id", "key"),
    }[action]
    required = {
        "launch": {"app"},
        "snapshot": {"app"},
        "focus": {"pid", "window_id"},
        "invoke": {"pid", "window_id", "element"},
        "set_value": {"pid", "window_id", "element", "value"},
        "scroll": {"pid", "window_id", "direction"},
        "keystroke": {"pid", "window_id", "key"},
    }[action]
    if any(name not in args for name in required):
        return None
    return {name: args[name] for name in allowed if name in args}


def pc_ui_tool(args: dict[str, Any], **_: Any) -> str:
    action = args.get("action")
    if action not in _ACTIONS:
        return json.dumps({"success": False, "code": "invalid_action"})
    params = _params(args, action)
    if params is None:
        return json.dumps({"success": False, "code": "invalid_parameters"})
    wire_action = "set-value" if action == "set_value" else action
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
                json={
                    "session_id": _session_id(),
                    "action": f"ui.{wire_action}",
                    "params": params,
                },
                timeout=_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        return json.dumps(
            {"success": False, "code": "pc_host_unavailable", "error": str(exc)[:512]},
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
    name="pc_ui",
    toolset="zettlab_pc",
    schema=PC_UI_SCHEMA,
    handler=pc_ui_tool,
    check_fn=_check_pc_ui,
    emoji="🖥️",
    # An explicit desktop-control request must not depend on Tool Search to
    # discover the trusted-client capability. The session token and local Host
    # still enforce whether an operation can actually execute.
    defer_to_tool_search=False,
)
