"""Session-scoped full control of SSH connections trusted in Zettlab Memo."""

from __future__ import annotations

import base64
import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env
from tools.registry import registry

_MAX_RESPONSE_BYTES = 1 << 20
_DEFAULT_TIMEOUT_SECONDS = 65

SSH_CONTROL_SCHEMA = {
    "name": "ssh_control",
    "description": (
        "Use an SSH connection that the user trusted for this Agent and Chat. "
        "It provides unrestricted remote shell and file read/write within the SSH account's permissions. "
        "List connections first when no connection id is known."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_connections",
                    "shell_exec",
                    "file_list",
                    "file_read",
                    "file_write",
                    "file_mkdir",
                    "file_rename",
                    "file_remove",
                ],
            },
            "connection_id": {
                "type": "string",
                "description": "Opaque id returned by list_connections.",
            },
            "command": {
                "type": "string",
                "description": "Complete remote shell command for shell_exec.",
            },
            "path": {
                "type": "string",
                "description": "Absolute or relative path on the remote SSH host.",
            },
            "destination": {
                "type": "string",
                "description": "Destination path for file_rename.",
            },
            "content": {
                "type": "string",
                "description": "UTF-8 file content for file_write. Existing content is overwritten.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _endpoint() -> str:
    raw = str(get_secret("ZETTLAB_LOCAL_SERVER_URL", "") or "").strip()
    if not raw:
        raw = str(get_secret("ZETTLAB_BROWSER_ACTION_URL", "") or "").strip()
    if not raw:
        raw = "http://127.0.0.1:19090"
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
        trusted = host == "localhost" or ipaddress.ip_address(host).is_loopback
        if parsed.scheme != "http" or parsed.username or parsed.password or not parsed.port or not trusted:
            return ""
    except (TypeError, ValueError):
        return ""
    return urlunsplit((parsed.scheme, parsed.netloc, "/api/v1/agent/hardware/protocol/actions", "", ""))


def _session_value(name: str) -> str:
    try:
        return str(get_session_env(name, "") or "").strip()
    except Exception:
        return ""


def _runtime_context() -> tuple[str, str, str]:
    action_token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    session_id = _session_value("HERMES_SESSION_ID") or _session_value("HERMES_SESSION_KEY")
    turn_id = _session_value("HERMES_TURN_ID")
    return action_token, session_id, turn_id


def _check_ssh_control() -> bool:
    action_token, session_id, turn_id = _runtime_context()
    # Discovery establishes that a trusted Chat turn can ask for SSH work; it
    # must not consume the separate execution capability that authorizes an
    # operation. Dispatch remains fail-closed below and the local-server
    # revalidates the owner/Agent/session/turn grant before any connection data
    # or remote operation is returned.
    return bool(_endpoint() and action_token and session_id and turn_id)


_check_ssh_control._session_scope_sensitive = True  # type: ignore[attr-defined]


def _payload(args: dict[str, Any]) -> dict[str, Any]:
    action = args.get("action")
    action_map = {
        "list_connections": "connection.list",
        "shell_exec": "shell.exec",
        "file_list": "file.list",
        "file_read": "file.read",
        "file_write": "file.write",
        "file_mkdir": "file.mkdir",
        "file_rename": "file.rename",
        "file_remove": "file.remove",
    }
    remote_action = action_map.get(action)
    if remote_action is None:
        raise ValueError("unsupported SSH action")
    payload: dict[str, Any] = {"action": remote_action}
    for key in ("connection_id", "command", "path", "destination", "timeout_seconds"):
        if key in args:
            payload[key] = args[key]
    if action == "file_write":
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("file_write requires content")
        payload["content_base64"] = base64.b64encode(content.encode("utf-8")).decode("ascii")
    return payload


def ssh_control_tool(args: dict[str, Any], **_: Any) -> str:
    try:
        payload = _payload(args)
    except (TypeError, ValueError) as exc:
        return json.dumps({"success": False, "code": "invalid_action", "error": str(exc)}, ensure_ascii=False)
    action_token, session_id, turn_id = _runtime_context()
    if not all((action_token, session_id, turn_id)):
        return json.dumps({"success": False, "code": "ssh_authorization_unavailable"})
    timeout_seconds = payload.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS - 5)
    request_timeout = min(max(int(timeout_seconds) + 5, 10), 605)
    try:
        with requests.Session() as client:
            client.trust_env = False
            response = client.post(
                _endpoint(),
                headers={
                    "Content-Type": "application/json",
                    "X-Zettlab-Agent-Action-Token": action_token,
                    "X-Hermes-Session-Id": session_id,
                    "X-Hermes-Turn-Id": turn_id,
                },
                json=payload,
                timeout=request_timeout,
            )
    except requests.RequestException as exc:
        return json.dumps({"success": False, "code": "ssh_service_unavailable", "error": str(exc)[:512]}, ensure_ascii=False)
    if len(response.content) > _MAX_RESPONSE_BYTES:
        return json.dumps({"success": False, "code": "ssh_response_too_large"})
    try:
        body = response.json()
    except ValueError:
        return json.dumps({"success": False, "code": "invalid_ssh_response"})
    if response.status_code >= 400:
        return json.dumps({"success": False, **(body if isinstance(body, dict) else {})}, ensure_ascii=False)
    if isinstance(body, dict) and isinstance(body.get("data"), dict):
        body = body["data"]
    return json.dumps({"success": True, "result": body}, ensure_ascii=False)


registry.register(
    name="ssh_control",
    toolset="zettlab_ssh",
    schema=SSH_CONTROL_SCHEMA,
    handler=ssh_control_tool,
    check_fn=_check_ssh_control,
    emoji="🖧",
    # This is a trusted-client capability, not an optional integration the
    # model should have to rediscover.  When the current turn carries the
    # session grant, keep the native schema in the model-facing tool list so
    # ordinary SSH requests cannot incorrectly conclude that no tool exists.
    defer_to_tool_search=False,
)
