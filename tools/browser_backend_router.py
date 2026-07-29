"""Client for local-server's authoritative browser backend router.

Hermes exposes one browser tool family to the model. For Zettlab sessions this
module asks local-server whether the session is bound to the Electron browser or
to device-side Camofox. Hermes executes Camofox's existing adapter only when the
router explicitly delegates the action; it never keeps a second backend-choice
cache of its own.
"""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit

import requests


_URL_ENV = "ZETTLAB_BROWSER_ACTION_URL"
_TOKEN_ENV = "ZETTLAB_AGENT_ACTION_TOKEN"
_MAX_RESPONSE_BYTES = 8 << 20
_TIMEOUT_SECONDS = 35
# Availability probes gate tool advertisement, so they must stay snappy even
# when local-server is wedged.
_HOST_STATUS_TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class BrowserRoute:
    """One routed action.

    ``backend`` is ``desktop`` or ``camofox`` on a valid router response and
    ``error`` when the authoritative router rejected or could not serve it.
    ``result`` is already serialized for direct tool output when present.
    """

    backend: str
    result: Optional[str] = None


def _runtime_value(name: str) -> str:
    """Read the active profile's managed browser configuration."""
    from agent.secret_scope import get_secret

    return str(get_secret(name, "") or "").strip()


def _trusted_loopback_endpoint(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme != "http" or not port:
        return False
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _endpoint() -> str:
    value = _runtime_value(_URL_ENV)
    return value if _trusted_loopback_endpoint(value) else ""


def _action_token() -> str:
    return _runtime_value(_TOKEN_ENV)


def is_managed_browser_configured() -> bool:
    return bool(_endpoint() and _action_token())


def is_desktop_host_online() -> bool:
    """Whether local-server currently has a PC Browser Host connected.

    local-server injects the router endpoint whenever it runs, so
    configuration alone must not advertise the browser tools. This probe sends
    the lightweight ``host_status`` action, which the broker answers locally
    (no PC round-trip, no session binding side effects) with
    ``{"ok": true, "result": {"host_online": bool}}``. Anything else — network
    error, non-200, or an older local-server that rejects the unknown action —
    reads as offline so callers fall back to the Camofox/local checks.
    """
    endpoint = _endpoint()
    token = _action_token()
    session_id = _session_id()
    if not endpoint or not token or not session_id:
        return False
    try:
        with requests.Session() as client:
            # Same loopback-only endpoint as route_browser_action; user proxy
            # settings must never receive the per-agent action token.
            client.trust_env = False
            response = client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "X-Zettlab-Agent-Action-Token": token,
                },
                json={
                    "session_id": session_id,
                    "action": "host_status",
                    "params": {},
                },
                timeout=_HOST_STATUS_TIMEOUT_SECONDS,
            )
        if response.status_code != 200:
            return False
        payload = response.json()
    except Exception:
        return False
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return False
    result = payload.get("result")
    return isinstance(result, dict) and result.get("host_online") is True


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


def _as_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _error(code: str, message: str) -> BrowserRoute:
    return BrowserRoute("error", _as_json({
        "success": False,
        "code": code,
        "error": message,
    }))


def route_browser_action(
    action: str,
    params: Optional[dict[str, Any]] = None,
) -> Optional[BrowserRoute]:
    """Route one browser action, or return ``None`` outside managed sessions."""
    endpoint = _endpoint()
    token = _action_token()
    session_id = _session_id()
    if not endpoint or not token or not session_id:
        return None
    try:
        with requests.Session() as client:
            # This is a loopback-only local-server endpoint. User proxy settings
            # must never receive the per-agent action token.
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
        return _error(
            "browser_router_unavailable",
            f"Managed browser router request failed: {exc}",
        )
    if len(response.content) > _MAX_RESPONSE_BYTES:
        return _error(
            "browser_router_response_too_large",
            "Managed browser router response exceeded 8 MiB.",
        )
    try:
        payload = response.json()
    except ValueError:
        return _error(
            "invalid_browser_router_response",
            "Managed browser router returned invalid JSON.",
        )
    if not isinstance(payload, dict):
        return _error(
            "invalid_browser_router_response",
            "Managed browser router returned an invalid response.",
        )

    backend = payload.get("backend")
    if (
        response.status_code < 400
        and payload.get("success") is True
        and backend == "camofox"
        and payload.get("delegated") is True
    ):
        return BrowserRoute("camofox")
    if response.status_code < 400 and payload.get("success") is True and backend == "desktop":
        result = payload.get("result")
        if not isinstance(result, dict):
            result = {}
        return BrowserRoute("desktop", _as_json({"success": True, **result}))

    payload.setdefault("success", False)
    if response.status_code >= 400:
        payload.setdefault("error", f"Managed browser router HTTP {response.status_code}")
    return BrowserRoute("error", _as_json(payload))
