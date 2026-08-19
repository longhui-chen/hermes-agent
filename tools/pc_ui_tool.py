"""Session-scoped semantic Computer Use on a locally approved desktop."""

from __future__ import annotations

import ipaddress
import base64
import binascii
import hashlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from agent.secret_scope import get_secret
from gateway.session_context import get_session_env, zettlab_browser_session_token
from tools.registry import registry

_MAX_RESPONSE_BYTES = 6 << 20
_MAX_SCREENSHOT_BYTES = 4 << 20
_TIMEOUT_SECONDS = 35
_ACTIONS = {
    "list_apps",
    "list_windows",
    "launch",
    "snapshot",
    "desktop_snapshot",
    "focus",
    "invoke",
    "click",
    "move_cursor",
    "drag",
    "type_text",
    "set_value",
    "scroll",
    "keystroke",
    "invoke_menu",
    "verify",
    "zoom",
    "set_window_frame",
    "clipboard_read",
    "clipboard_write",
    "kill_app",
    "complete_task",
}
_READ_ACTIONS = {
    "list_apps", "list_windows", "snapshot", "desktop_snapshot", "verify", "zoom", "clipboard_read", "complete_task",
}
_TASK_CONTROL_FIELDS = {"snapshot_revision", "user_input_epoch", "postcondition"}

PC_UI_SCHEMA = {
    "name": "pc_ui",
    "description": (
        "Use semantic Computer Use on the connected desktop after the user approves it locally. "
        "Use list_apps first when the application name is uncertain, then pass the exact returned name to launch or snapshot. "
        "If the application has no visible window, launch it first. Use list_windows to distinguish the main window from dialogs or notifications, then snapshot the chosen exact pid and window_id. "
        "Request include_screenshot=true when the accessibility tree is incomplete, then use coordinates only against that exact fresh screenshot. "
        "Start every interaction with snapshot once a window exists. Reuse only the pid, window_id and element "
        "indices returned by that fresh snapshot; never invent selectors, paths, shell commands, "
        "scripts or CDP requests. Secure controls are hidden and sensitive actions may require "
        "another confirmation on the computer. When snapshot returns snapshot_revision and "
        "user_input_epoch, copy both into every following mutation and provide a bounded, "
        "observable postcondition. A mutation is successful only after the Host re-observes "
        "the exact window and proves that postcondition. If a result includes task metadata, call complete_task once "
        "after the requested desktop task is fully finished; it closes only the current runtime task lease and never "
        "performs a desktop mutation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(_ACTIONS)},
            "app": {
                "type": "string",
                "maxLength": 256,
                "description": "Exact name returned by list_apps. Required for list_windows, launch and snapshot.",
            },
            "pid": {"type": "integer", "minimum": 1},
            "window_id": {
                "type": "integer",
                "minimum": 1,
                "description": "Exact visible window identity returned by list_windows. pid and window_id may be supplied to snapshot together.",
            },
            "element": {
                "type": "integer",
                "minimum": 1,
                "description": "Element index from the latest snapshot for this Chat session.",
            },
            "value": {"type": "string", "maxLength": 8192},
            "text": {"type": "string", "maxLength": 8192},
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 50},
            "by": {"type": "string", "enum": ["line", "page"]},
            "key": {
                "type": "string",
                "maxLength": 64,
                "description": "One key or a plus-separated hotkey such as CMD+L.",
            },
            "include_screenshot": {"type": "boolean"},
            "include_hidden": {"type": "boolean"},
            "include_text": {"type": "boolean"},
            "query": {"type": "string", "maxLength": 256},
            "scope": {"type": "string", "enum": ["window", "desktop"]},
            "x": {"type": "number", "minimum": 0},
            "y": {"type": "number", "minimum": 0},
            "x1": {"type": "number", "minimum": 0},
            "y1": {"type": "number", "minimum": 0},
            "x2": {"type": "number", "minimum": 0},
            "y2": {"type": "number", "minimum": 0},
            "from_x": {"type": "number", "minimum": 0},
            "from_y": {"type": "number", "minimum": 0},
            "to_x": {"type": "number", "minimum": 0},
            "to_y": {"type": "number", "minimum": 0},
            "from_zoom": {"type": "boolean"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "count": {"type": "integer", "enum": [1, 2]},
            "ax_action": {"type": "string", "enum": ["press", "show_menu", "pick", "confirm", "cancel", "open"]},
            "delivery_mode": {"type": "string", "enum": ["background", "foreground"]},
            "modifiers": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "delay_ms": {"type": "integer", "minimum": 0, "maximum": 200},
            "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
            "steps": {"type": "integer", "minimum": 1, "maximum": 200},
            "path": {"type": "array", "items": {"type": "string", "maxLength": 200}, "minItems": 1, "maxItems": 16},
            "expect": {"type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 8},
            "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
            "stable_samples": {"type": "integer", "minimum": 1, "maximum": 5},
            "width": {"type": "number", "minimum": 1},
            "height": {"type": "number", "minimum": 1},
            "snapshot_revision": {
                "type": "integer", "minimum": 1,
                "description": "Exact revision returned by the latest snapshot for this task.",
            },
            "user_input_epoch": {
                "type": "integer", "minimum": 0,
                "description": "Input epoch returned by the latest snapshot. A user takeover invalidates it.",
            },
            "postcondition": {
                "type": "array", "items": {"type": "object"}, "minItems": 1, "maxItems": 8,
                "description": "Deterministic predicates the Host must verify after a mutation.",
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
        "list_apps": (),
        "list_windows": ("app", "include_hidden"),
        "launch": ("app",),
        "snapshot": ("app", "pid", "window_id", "include_screenshot", "include_hidden", "query"),
        "desktop_snapshot": (),
        "focus": ("pid", "window_id"),
        "invoke": ("pid", "window_id", "element"),
        "click": ("pid", "window_id", "scope", "element", "x", "y", "button", "count", "ax_action", "delivery_mode", "modifiers", "from_zoom"),
        "move_cursor": ("scope", "x", "y"),
        "drag": ("pid", "window_id", "scope", "from_x", "from_y", "to_x", "to_y", "button", "duration_ms", "steps", "delivery_mode", "modifiers", "from_zoom"),
        "type_text": ("pid", "window_id", "scope", "element", "x", "y", "text", "delay_ms", "delivery_mode"),
        "set_value": ("pid", "window_id", "element", "value"),
        "scroll": ("pid", "window_id", "scope", "element", "x", "y", "direction", "amount", "by", "delivery_mode"),
        "keystroke": ("pid", "window_id", "scope", "element", "x", "y", "key", "delivery_mode"),
        "invoke_menu": ("pid", "window_id", "path"),
        "verify": ("pid", "window_id", "expect", "timeout_ms", "stable_samples", "include_screenshot"),
        "zoom": ("pid", "window_id", "x1", "y1", "x2", "y2"),
        "set_window_frame": ("pid", "window_id", "x", "y", "width", "height"),
        "clipboard_read": ("include_text",),
        "clipboard_write": ("text",),
        "kill_app": ("pid",),
        "complete_task": (),
    }[action]
    required = {
        "list_apps": set(),
        "list_windows": {"app"},
        "launch": {"app"},
        "snapshot": {"app"},
        "desktop_snapshot": set(),
        "focus": {"pid", "window_id"},
        "invoke": {"pid", "window_id", "element"},
        "click": set(),
        "move_cursor": {"scope", "x", "y"},
        "drag": {"from_x", "from_y", "to_x", "to_y"},
        "type_text": {"text"},
        "set_value": {"pid", "window_id", "element", "value"},
        "scroll": {"direction"},
        "keystroke": {"key"},
        "invoke_menu": {"pid", "window_id", "path"},
        "verify": {"pid", "window_id", "expect"},
        "zoom": {"pid", "window_id", "x1", "y1", "x2", "y2"},
        "set_window_frame": {"pid", "window_id", "x", "y", "width", "height"},
        "clipboard_read": set(),
        "clipboard_write": {"text"},
        "kill_app": {"pid"},
        "complete_task": set(),
    }[action]
    if any(name not in args for name in required):
        return None
    if any(name not in allowed and name != "action" and name not in _TASK_CONTROL_FIELDS for name in args):
        return None
    if action == "snapshot":
        has_pid = "pid" in args
        has_window_id = "window_id" in args
        if has_pid != has_window_id:
            return None
        if args.get("include_hidden") is True and not has_pid:
            return None
    scope = args.get("scope", "window")
    if scope not in {"window", "desktop"}:
        return None
    if action in {"click", "move_cursor", "drag", "type_text", "scroll", "keystroke"}:
        if scope == "desktop":
            if any(name in args for name in ("pid", "window_id", "element")):
                return None
            rejected = {
                "click": {"ax_action", "delivery_mode", "modifiers", "from_zoom"},
                "drag": {"delivery_mode", "modifiers", "from_zoom"},
            }.get(action, {"delivery_mode"})
            if any(name in args for name in rejected):
                return None
        elif any(name not in args for name in ("pid", "window_id")):
            return None
    if action == "click" and "element" not in args and not all(name in args for name in ("x", "y")):
        return None
    if action == "click" and "element" in args and "x" in args:
        return None
    if action == "click" and "element" in args and args.get("count") == 2:
        if args.get("button", "left") != "left" or any(
            name in args for name in ("ax_action", "modifiers", "from_zoom")
        ):
            return None
    if action == "move_cursor" and scope != "desktop":
        return None
    for x_name, y_name in (("x", "y"),):
        if (x_name in args) != (y_name in args):
            return None
    if action in {"type_text", "scroll", "keystroke"}:
        if "element" in args and "x" in args:
            return None
    if action == "drag":
        for x_name, y_name in (("from_x", "from_y"), ("to_x", "to_y")):
            if (x_name in args) != (y_name in args):
                return None
    return {name: args[name] for name in allowed if name in args}


def _wire_task_id(task_id: str) -> str:
    return "task-" + hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:40]


def _target_fingerprint(action: str, params: dict[str, Any]) -> str:
    # Never hash user text or clipboard values into a durable identifier. The
    # exact window/control/coordinates are sufficient to bind receipt replay.
    safe = {
        key: params[key]
        for key in (
            "app", "pid", "window_id", "element", "scope", "x", "y",
            "from_x", "from_y", "to_x", "to_y", "path", "key",
        )
        if key in params
    }
    encoded = json.dumps(
        {"action": action, "target": safe},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _task_control(
    args: dict[str, Any],
    action: str,
    params: dict[str, Any],
    task_id: str,
    tool_call_id: str,
) -> dict[str, Any] | None:
    task_id = str(task_id or "").strip()
    if not task_id:
        return None
    envelope: dict[str, Any] = {"task_id": _wire_task_id(task_id)}
    if action not in _READ_ACTIONS:
        tool_call_id = str(tool_call_id or "").strip()
        if not tool_call_id:
            raise ValueError("missing tool call identity for desktop mutation")
        material = f"{task_id}\x00{tool_call_id}\x00{action}\x00{_target_fingerprint(action, params)}"
        envelope["operation_id"] = "op-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:48]
    for name in ("snapshot_revision", "user_input_epoch"):
        value = args.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            envelope[name] = value
    postcondition = args.get("postcondition")
    if isinstance(postcondition, list):
        envelope["postcondition"] = postcondition
    return envelope


def pc_ui_tool(
    args: dict[str, Any],
    task_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> Any:
    action = args.get("action")
    if action not in _ACTIONS:
        return json.dumps({"success": False, "code": "invalid_action"})
    params = _params(args, action)
    if params is None:
        return json.dumps({"success": False, "code": "invalid_parameters"})
    try:
        task_control = _task_control(args, action, params, task_id, tool_call_id)
    except ValueError:
        return json.dumps({"success": False, "code": "pc_task_identity_missing"})
    wire_action = {
        "list_apps": "list-apps",
        "list_windows": "list-windows",
        "desktop_snapshot": "desktop-snapshot",
        "move_cursor": "move-cursor",
        "type_text": "type-text",
        "set_value": "set-value",
        "invoke_menu": "invoke-menu",
        "set_window_frame": "set-window-frame",
        "clipboard_read": "clipboard-read",
        "clipboard_write": "clipboard-write",
        "kill_app": "kill-app",
        "complete_task": "task-complete",
    }.get(action, action)
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
                    **({"task_control": task_control} if task_control else {}),
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
    result = payload.get("result") if isinstance(payload, dict) else None
    if isinstance(result, dict):
        screenshot_b64 = result.pop("screenshot_b64", None)
        screenshot_mime = result.pop("screenshot_mime_type", None)
        if isinstance(screenshot_b64, str) and screenshot_mime in {"image/png", "image/jpeg"}:
            try:
                image_bytes = base64.b64decode(screenshot_b64, validate=True)
            except (ValueError, TypeError, binascii.Error):
                return json.dumps({"success": False, "code": "invalid_pc_screenshot"})
            if len(image_bytes) > _MAX_SCREENSHOT_BYTES:
                return json.dumps({"success": False, "code": "pc_screenshot_too_large"})
            summary = json.dumps(payload, ensure_ascii=False)
            return {
                "_multimodal": True,
                "content": [
                    {"type": "text", "text": summary},
                    {"type": "image_url", "image_url": {"url": f"data:{screenshot_mime};base64,{screenshot_b64}"}},
                ],
                "text_summary": summary,
            }
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
