"""Action-level adapter for Zettlab's Electron-owned desktop browser.

The browser never exposes a CDP URL to Hermes.  Hermes sends a bounded action
to local-server's loopback broker; the broker validates the per-agent action
token plus the stable zettlab user/agent session scope before forwarding it to
the authenticated PC Browser Host.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from typing import Any, Optional

import requests


_URL_ENV = "ZETTLAB_DESKTOP_BROWSER_HOST_URL"
_TOKEN_ENV = "ZETTLAB_AGENT_ACTION_TOKEN"
_MAX_RESPONSE_BYTES = 8 << 20
_MAX_ACTIVE_SESSIONS = 64
_TIMEOUT_SECONDS = 35

_active_sessions: "OrderedDict[str, None]" = OrderedDict()
_active_lock = threading.Lock()


def _endpoint() -> str:
    return os.environ.get(_URL_ENV, "").strip()


def is_desktop_browser_configured() -> bool:
    return bool(_endpoint() and os.environ.get(_TOKEN_ENV, "").strip())


def _session_id() -> str:
    try:
        from gateway.session_context import get_session_env

        value = (
            get_session_env("HERMES_SESSION_KEY", "")
            or get_session_env("HERMES_SESSION_ID", "")
        )
    except Exception:
        value = os.environ.get("HERMES_SESSION_KEY", "") or os.environ.get(
            "HERMES_SESSION_ID", ""
        )
    value = str(value or "").strip()
    return value if value.startswith("zettlab:") else ""


def _mark_active(session_id: str) -> None:
    with _active_lock:
        _active_sessions.pop(session_id, None)
        _active_sessions[session_id] = None
        while len(_active_sessions) > _MAX_ACTIVE_SESSIONS:
            _active_sessions.popitem(last=False)


def _drop_active(session_id: str) -> None:
    with _active_lock:
        _active_sessions.pop(session_id, None)


def has_desktop_browser_session(task_id: Optional[str] = None) -> bool:
    del task_id  # Session ownership is task-local ContextVar state, not model task ids.
    session_id = _session_id()
    if not session_id:
        return False
    with _active_lock:
        return session_id in _active_sessions


def _call(action: str, params: Optional[dict[str, Any]] = None) -> tuple[str, dict[str, Any]]:
    endpoint = _endpoint()
    token = os.environ.get(_TOKEN_ENV, "").strip()
    session_id = _session_id()
    if not endpoint or not token or not session_id:
        return "unavailable", {
            "success": False,
            "code": "browser_host_unavailable",
            "error": "Desktop browser host is not available in this session.",
        }
    try:
        with requests.Session() as client:
            # This port is always a loopback-only local-server endpoint.  User
            # HTTP(S)_PROXY settings must never reroute the action token or
            # Browser Host traffic through an external proxy.
            client.trust_env = False
            response = client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "X-Zettlab-Agent-Action-Token": token,
                },
                json={
                    "session_id": session_id,
                    "action": action,
                    "params": params or {},
                },
                timeout=_TIMEOUT_SECONDS,
            )
    except requests.RequestException as exc:
        return "unavailable", {
            "success": False,
            "code": "browser_host_unavailable",
            "error": f"Desktop browser host request failed: {exc}",
        }
    if len(response.content) > _MAX_RESPONSE_BYTES:
        return "error", {
            "success": False,
            "code": "browser_host_response_too_large",
            "error": "Desktop browser host response exceeded 8 MiB.",
        }
    try:
        payload = response.json()
    except ValueError:
        return "error", {
            "success": False,
            "code": "invalid_browser_host_response",
            "error": "Desktop browser host returned invalid JSON.",
        }
    if not isinstance(payload, dict):
        return "error", {
            "success": False,
            "code": "invalid_browser_host_response",
            "error": "Desktop browser host returned an invalid response.",
        }
    if response.status_code == 503 and payload.get("code") == "browser_host_unavailable":
        return "unavailable", payload
    if response.status_code >= 400:
        payload.setdefault("success", False)
        payload.setdefault("error", f"Desktop browser host HTTP {response.status_code}")
        return "error", payload
    if payload.get("success") is not True:
        payload.setdefault("success", False)
        return "error", payload
    result = payload.get("result")
    if not isinstance(result, dict):
        result = {}
    result = {"success": True, **result}
    return "ok", result


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def desktop_browser_navigate(url: str, task_id: Optional[str] = None) -> Optional[str]:
    del task_id
    status, payload = _call("navigate", {"url": url})
    if status == "unavailable":
        # No PC host is connected: preserve the existing Camofox/local-browser
        # fallback instead of turning an additive capability into an outage.
        return None
    if status == "ok":
        _mark_active(_session_id())
    return _as_json(payload)


def desktop_browser_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None,
) -> str:
    del task_id, user_task
    _, payload = _call("snapshot", {"full": bool(full)})
    return _as_json(payload)


def desktop_browser_click(ref: str, task_id: Optional[str] = None) -> str:
    del task_id
    _, payload = _call("click", {"ref": ref})
    return _as_json(payload)


def desktop_browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    del task_id
    _, payload = _call("type", {"ref": ref, "text": text})
    from agent.display import (
        redact_browser_typed_text_for_display,
        redact_tool_args_for_display,
    )

    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})[
        "text"
    ]
    if payload.get("success"):
        payload["typed"] = display_text
        payload.setdefault("element", ref.lstrip("@"))
    payload = redact_browser_typed_text_for_display(payload, text)
    return _as_json(payload)


def desktop_browser_screenshot(task_id: Optional[str] = None) -> str:
    del task_id
    _, payload = _call("screenshot")
    return _as_json(payload)


def desktop_browser_close(task_id: Optional[str] = None) -> str:
    del task_id
    session_id = _session_id()
    if not session_id:
        return _as_json({"success": True, "closed": True})
    try:
        _, payload = _call("close")
        return _as_json(payload)
    finally:
        _drop_active(session_id)


def desktop_browser_unsupported(action: str) -> str:
    return _as_json({
        "success": False,
        "code": "desktop_browser_action_not_supported",
        "error": (
            f"{action} is not supported by the desktop browser MVP. "
            "Use navigate, snapshot, click, or type."
        ),
    })


def _reset_desktop_browser_state_for_tests() -> None:
    with _active_lock:
        _active_sessions.clear()
