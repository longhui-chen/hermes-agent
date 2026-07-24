"""Camofox browser backend — local anti-detection browser via REST API.

Camofox-browser is a self-hosted Node.js server wrapping Camoufox (Firefox
fork with C++ fingerprint spoofing).  It exposes a REST API that maps 1:1
to our browser tool interface: accessibility snapshots with element refs,
click/type/scroll by ref, screenshots, etc.

When ``CAMOFOX_URL`` is set (e.g. ``http://localhost:9377``), the browser
tools route through this module instead of the ``agent-browser`` CLI.

The service is managed by the device's local-server. Set ``CAMOFOX_URL`` to
the managed proxy endpoint; Hermes does not start or install Camofox itself.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import uuid
from typing import Any, Dict, Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

from hermes_cli.config import cfg_get, load_config, read_raw_config


_HANDBACK_SENSITIVE_CONTROL = re.compile(
    r"(?ix)"
    r"\b(?:password|passcode|one[- ]?time(?:\s+(?:password|code))?|otp|"
    r"verification(?:\s+code)?|security\s+code|\d+[- ]digit\s+code|pin|"
    r"card\s+number|credit\s+card|debit\s+card|cvv|cvc|social\s+security|"
    r"ssn|passport|tax\s+id)\b|密码|验证码|银行卡|身份证"
)
_HANDBACK_EDITABLE_CONTROL = re.compile(
    r"(?i)^\s*(?:-\s*)?(?:textbox|searchbox|combobox|listbox|option|spinbutton|slider|checkbox|radio|switch)\b"
)
_HANDBACK_VALUE_ATTRIBUTE = re.compile(
    r"(?i)\bvalue=(?:\"[^\"]*\"|'[^']*'|\S+)"
)


def _redact_handback_page_state(value: str) -> str:
    """Apply mandatory privacy filtering to state captured after human control."""
    from tools.browser_tool import _redact_browser_output

    redacted = _redact_browser_output(value)
    lines = []
    for line in redacted.splitlines():
        if _HANDBACK_EDITABLE_CONTROL.search(line) or _HANDBACK_SENSITIVE_CONTROL.search(line):
            lines.append("[REDACTED sensitive form control]")
        else:
            lines.append(_HANDBACK_VALUE_ATTRIBUTE.sub("value=\"[REDACTED]\"", line))
    return "\n".join(lines)


def _redact_handback_url(value: str) -> str:
    """Return only a credential-free browser location after human control."""
    from agent.redact import redact_cdp_url

    redacted = redact_cdp_url(value)
    parts = urlsplit(redacted)
    hostname = parts.hostname
    if not hostname:
        return ""
    authority = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        return ""
    if port is not None:
        authority = f"{authority}:{port}"
    return urlunsplit((parts.scheme, authority, "", "", ""))


def _set_handback_privacy_filter(session: Dict[str, Any], enabled: bool) -> None:
    """Persist handback privacy filtering for subsequent reads of this tab."""
    with _session_lock(session):
        session["privacy_filter_after_handback"] = enabled


def _handback_privacy_filter_enabled(session: Dict[str, Any]) -> bool:
    """Return whether raw page reads are blocked after human control."""
    with _session_lock(session):
        return bool(session.get("privacy_filter_after_handback"))


def _filter_page_state_after_handback(session: Dict[str, Any], value: str) -> str:
    """Filter page state while a human-mutated page remains current."""
    return _redact_handback_page_state(value) if _handback_privacy_filter_enabled(session) else value


def _unsafe_handback_url(value: str) -> bool:
    """Fail closed under the browser's metadata, private, and DNS policy."""
    from tools.browser_tool import (
        _is_always_blocked_url,
        _is_safe_url,
        _url_is_private,
    )

    # _is_safe_url rejects DNS failures, malformed URLs, and private targets by
    # default. _url_is_private is deliberately retained as an unconditional
    # floor because _is_safe_url honors HERMES_ALLOW_PRIVATE_URLS/config opt-outs,
    # which must never weaken handback evidence validation.
    return (
        _is_always_blocked_url(value)
        or _url_is_private(value)
        or not _is_safe_url(value)
    )


def _valid_resume_url_origin(value: str) -> bool:
    """Require a credential-free HTTP(S) origin without path/query/fragment."""
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return False
        if parts.username is not None or parts.password is not None:
            return False
        if parts.path not in {"", "/"} or parts.query or parts.fragment:
            return False
        # Accessing port validates malformed authorities such as ``host:bad``.
        parts.port
        return True
    except ValueError:
        return False


def _exact_session_tab(tabs: Any, session: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return only the tab bound to this exact managed browser session."""
    if not isinstance(tabs, list):
        return None
    return next(
        (
            tab
            for tab in tabs
            if isinstance(tab, dict)
            and tab.get("tabId") == session.get("tab_id")
            and tab.get("listItemId") == session.get("session_key")
        ),
        None,
    )


from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_SNAPSHOT_MAX_CHARS = 80_000  # camofox paginates at this limit
_vnc_url: Optional[str] = None  # single-profile cache from /health response
_vnc_url_checked = False  # only probe once per single-profile process

# Cached command timeout from config (resolved lazily, like browser_tool)
_cached_cmd_timeout: Optional[int] = None
_cmd_timeout_resolved = False

_SAFE_HTTP_ERROR_STRING_FIELDS = {
    "error": 256,
    "message": 1_000,
    "detail": 2_000,
    "phase": 128,
    "takeover_session_id": 256,
    "takeoverSessionId": 256,
    "resume_token": 512,
    "resumeToken": 512,
}


class CamofoxHTTPError(requests.HTTPError):
    """HTTP failure carrying only fields safe to expose to the Agent."""

    def __init__(self, response: requests.Response, payload: Dict[str, Any]):
        super().__init__(f"HTTP {response.status_code}", response=response)
        self.payload = payload


def _get_command_timeout() -> int:
    """Return ``browser.command_timeout`` from config, falling back to 30s.

    Mirrors :func:`tools.browser_tool._get_command_timeout` so both the
    local browser path and the Camofox path honour the same config knob.
    Result is cached after the first call.
    """
    global _cached_cmd_timeout, _cmd_timeout_resolved
    if _cmd_timeout_resolved:
        return _cached_cmd_timeout  # type: ignore[return-value]

    _cmd_timeout_resolved = True
    result = _DEFAULT_TIMEOUT
    try:
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # floor at 5s
    except Exception as exc:
        logger.debug("Could not read browser.command_timeout: %s", exc)
    _cached_cmd_timeout = result
    return result


def _runtime_value(name: str, default: str = "") -> str:
    """Resolve per-profile browser runtime configuration safely.

    ``get_secret`` transparently falls back to ``os.environ`` in the legacy
    single-profile process. In multiplex mode it reads only the active
    profile's context-local scope and fails closed when that scope is missing.
    """
    from agent.secret_scope import get_secret

    value = get_secret(name, default)
    return str(value or default)


def _auth_headers() -> Dict[str, str]:
    """Return the configured Camofox authentication header.

    A local-server proxy authenticates Hermes with the per-agent action token
    using a dedicated header. Direct Camofox connections retain the upstream
    bearer-key contract for backwards compatibility.
    """
    auth_mode = _runtime_value("CAMOFOX_AUTH_MODE").strip().lower()
    if auth_mode == "zettlab_action_token":
        token = _runtime_value("ZETTLAB_AGENT_ACTION_TOKEN").strip()
        if not token:
            raise RuntimeError(
                "CAMOFOX_AUTH_MODE=zettlab_action_token requires "
                "ZETTLAB_AGENT_ACTION_TOKEN"
            )
        return {"X-Zettlab-Agent-Action-Token": token}

    key = _runtime_value("CAMOFOX_API_KEY").strip()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def get_camofox_url() -> str:
    """Return the configured Camofox server URL, or empty string."""
    return _runtime_value("CAMOFOX_URL").rstrip("/")


def _config_cdp_url() -> str:
    """Persistent ``browser.cdp_url`` from config.yaml, or empty string.

    Read here (instead of importing ``browser_tool._get_cdp_override`` to avoid
    a circular import) so Camofox can yield to a config-based CDP override the
    same way it already yields to the ``BROWSER_CDP_URL`` env override.
    """
    try:
        from hermes_cli.config import read_raw_config

        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception:
        pass
    return ""


def is_camofox_mode() -> bool:
    """True when Camofox backend is configured and no CDP override is active.

    A CDP override takes priority over Camofox so the browser tools operate on
    the real CDP browser (and a CDP backend is treated as non-local for SSRF
    checks) instead of being silently routed to Camofox. The override may come
    from the ``BROWSER_CDP_URL`` env var (set by ``/browser connect``) OR a
    persistent ``browser.cdp_url`` in config.yaml — both are honored, matching
    ``browser_tool._get_cdp_override()``'s precedence. (Previously only the env
    var suppressed Camofox, so ``CAMOFOX_URL`` + a config CDP override still
    routed navigation through Camofox.)
    """
    if _runtime_value("BROWSER_CDP_URL").strip():
        return False
    if _config_cdp_url():
        return False
    return bool(get_camofox_url())


def check_camofox_available() -> bool:
    """Verify the Camofox server is reachable."""
    global _vnc_url, _vnc_url_checked
    url = get_camofox_url()
    if not url:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=5, headers=_auth_headers())
        if resp.status_code == 200 and not _vnc_url_checked and not _local_server_managed():
            try:
                data = resp.json()
                vnc_port = data.get("vncPort")
                if isinstance(vnc_port, int) and 1 <= vnc_port <= 65535:
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    host = parsed.hostname or "localhost"
                    _vnc_url = f"http://{host}:{vnc_port}"
            except (ValueError, KeyError):
                pass
            _vnc_url_checked = True
        return resp.status_code == 200
    except Exception:
        return False


def get_vnc_url() -> Optional[str]:
    """Return the VNC URL if the Camofox server exposes one, or None."""
    from agent.secret_scope import is_multiplex_active

    # A process-global VNC endpoint cannot be safely attributed to one profile
    # in a multiplex gateway. Managed deployments expose their authenticated
    # viewer through local-server instead.
    if _local_server_managed() or is_multiplex_active():
        return None
    if not _vnc_url_checked:
        check_camofox_available()
    return _vnc_url


def _get_camofox_config() -> Dict[str, Any]:
    """Return the ``browser.camofox`` config block, or an empty dict."""
    try:
        camofox_cfg = load_config().get("browser", {}).get("camofox", {})
    except Exception as exc:
        logger.warning("camofox config check failed, defaulting to disabled: %s", exc)
        return {}
    return camofox_cfg if isinstance(camofox_cfg, dict) else {}


def _managed_persistence_enabled() -> bool:
    """Return whether Hermes-managed persistence is enabled for Camofox.

    When enabled, sessions use a stable profile-scoped userId so the
    Camofox server can map it to a persistent browser profile directory.
    When disabled (default), each session gets a random userId (ephemeral).

    Controlled by ``browser.camofox.managed_persistence`` in config.yaml.
    """
    return bool(_get_camofox_config().get("managed_persistence"))


def _local_server_managed() -> bool:
    """Return whether local-server owns Camofox process/profile lifecycle."""
    return _env_flag("CAMOFOX_MANAGED_BY_LOCAL_SERVER") is True


def _camofox_identity_override(task_id: Optional[str], camofox_cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Return an externally configured Camofox identity, if one is set.

    Integrations that own the visible Camofox browser can set a shared user ID
    so Hermes operates in the same browser profile instead of creating a
    separate private session.
    """
    user_id = _runtime_value("CAMOFOX_USER_ID").strip() or str(camofox_cfg.get("user_id") or "").strip()
    if not user_id:
        return None

    session_key = (
        _runtime_value("CAMOFOX_SESSION_KEY").strip()
        or str(camofox_cfg.get("session_key") or "").strip()
        or get_camofox_identity(task_id)["session_key"]
    )
    return {"user_id": user_id, "session_key": session_key}


def _env_flag(name: str) -> Optional[bool]:
    raw = _runtime_value(name).strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.debug("Ignoring invalid boolean env %s=%r", name, raw)
    return None


def _adopt_existing_tab_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether Hermes should recover an existing Camofox tab ID."""
    if _local_server_managed():
        return True
    env_value = _env_flag("CAMOFOX_ADOPT_EXISTING_TAB")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("adopt_existing_tab"))


def _loopback_rewrite_enabled(camofox_cfg: Dict[str, Any]) -> bool:
    """Return whether loopback navigation URLs should be rewritten for Docker.

    ``CAMOFOX_URL`` itself often points at a host-published Docker port such as
    ``http://127.0.0.1:9377``.  That is correct for Hermes talking to the
    Camofox control API, but a page URL like ``http://127.0.0.1:3000`` is opened
    by the browser *inside* the Docker container.  In that context loopback
    points at the container, not the host running the web app.

    The rewrite is opt-in because non-Docker Camofox installs run the browser on
    the host, where loopback URLs are already correct.
    """
    env_value = _env_flag("CAMOFOX_REWRITE_LOOPBACK_URLS")
    if env_value is not None:
        return env_value
    return bool(camofox_cfg.get("rewrite_loopback_urls"))


def _loopback_rewrite_host(camofox_cfg: Dict[str, Any]) -> str:
    """Return the host alias used when rewriting loopback page URLs."""
    return (
        _runtime_value("CAMOFOX_LOOPBACK_HOST_ALIAS").strip()
        or str(camofox_cfg.get("loopback_host_alias") or "").strip()
        or "host.docker.internal"
    )


def _is_loopback_hostname(hostname: Optional[str]) -> bool:
    """Return True for localhost/127.0.0.0/8/::1-style hostnames."""
    if not hostname:
        return False
    host = hostname.strip().strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _rewrite_loopback_url_for_camofox(url: str) -> tuple[str, Optional[Dict[str, str]]]:
    """Rewrite loopback page URLs for Docker-hosted Camofox, if configured.

    Returns ``(rewritten_url, metadata)``.  ``metadata`` is present only when a
    rewrite happened so the tool result can disclose the change to the model.
    """
    camofox_cfg = _get_camofox_config()
    if not _loopback_rewrite_enabled(camofox_cfg):
        return url, None

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url, None

    if parsed.scheme not in {"http", "https"} or not _is_loopback_hostname(parsed.hostname):
        return url, None

    alias = _loopback_rewrite_host(camofox_cfg)
    if not alias:
        return url, None

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    host_part = f"[{alias}]" if ":" in alias and not alias.startswith("[") else alias
    port_part = f":{parsed.port}" if parsed.port else ""
    rewritten = urlunsplit(
        SplitResult(parsed.scheme, f"{userinfo}{host_part}{port_part}", parsed.path, parsed.query, parsed.fragment)
    )
    return rewritten, {
        "from": parsed.hostname or "",
        "to": alias,
        "original_url": url,
        "rewritten_url": rewritten,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
# Maps a profile-scoped identity plus task to its process-local tab state.
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()


def _session_cache_key(task_id: str, identity: Dict[str, str]) -> str:
    return f"{identity['user_id']}\x00{identity['session_key']}\x00{task_id}"


def _session_lock(session: Dict[str, Any]) -> threading.Lock:
    with _sessions_lock:
        lock = session.get("_lock")
        if lock is None:
            lock = threading.Lock()
            session["_lock"] = lock
        return lock


def _adopt_existing_tab(session: Dict[str, Any]) -> Dict[str, Any]:
    """Attach process-local state to an already-open managed Camofox tab.

    Some integrations own the visible Camofox tab outside Hermes. Gateway
    restarts can leave this module's in-memory session cache empty even though
    Camofox still has that tab, so rehydrate tab_id before creating a new tab.
    """
    if session.get("tab_id") or not session.get("adopt_existing_tab"):
        return session

    if not get_camofox_url():
        return session

    try:
        tabs = _get("/tabs", params={"userId": session["user_id"]}, timeout=5).get("tabs", [])
    except Exception as exc:
        logger.debug("Camofox tab adoption failed for %s: %s", session.get("user_id"), exc)
        return session

    if not isinstance(tabs, list) or not tabs:
        return session

    session_key = session.get("session_key")
    matching_tabs = [
        tab
        for tab in tabs
        if isinstance(tab, dict) and tab.get("listItemId") == session_key
    ]
    latest = matching_tabs[-1] if matching_tabs else None
    tab_id = latest.get("tabId") if isinstance(latest, dict) else None
    if isinstance(tab_id, str) and tab_id:
        session["tab_id"] = tab_id
        logger.debug("Adopted existing Camofox tab %s for %s", tab_id, session.get("user_id"))

    return session


def _get_session(task_id: Optional[str]) -> Dict[str, Any]:
    """Get or create a camofox session for the given task.

    When managed persistence is enabled, uses a deterministic userId
    derived from the Hermes profile so the Camofox server can map it
    to the same persistent browser profile across restarts.
    """
    task_id = task_id or "default"
    camofox_cfg = _get_camofox_config()
    profile_identity = get_camofox_identity(task_id)
    identity_override = _camofox_identity_override(task_id, camofox_cfg)
    cache_identity = identity_override or profile_identity
    cache_key = _session_cache_key(task_id, cache_identity)
    with _sessions_lock:
        if cache_key in _sessions:
            session = _sessions[cache_key]
        else:
            if identity_override:
                session = {
                    "user_id": identity_override["user_id"],
                    "tab_id": None,
                    "session_key": identity_override["session_key"],
                    "managed": True,
                    "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                    "privacy_filter_after_handback": False,
                    "_lock": threading.Lock(),
                }
            elif _local_server_managed() or bool(camofox_cfg.get("managed_persistence")):
                session = {
                    "user_id": profile_identity["user_id"],
                    "tab_id": None,
                    "session_key": profile_identity["session_key"],
                    "managed": True,
                    "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                    "privacy_filter_after_handback": False,
                    "_lock": threading.Lock(),
                }
            else:
                session = {
                    "user_id": f"hermes_{uuid.uuid4().hex[:10]}",
                    "tab_id": None,
                    "session_key": profile_identity["session_key"],
                    "managed": False,
                    "adopt_existing_tab": False,
                    "privacy_filter_after_handback": False,
                    "_lock": threading.Lock(),
                }
            _sessions[cache_key] = session

    with _session_lock(session):
        return _adopt_existing_tab(session)


def _ensure_tab(task_id: Optional[str], url: Optional[str] = None) -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed."""
    session = _get_session(task_id)
    with _session_lock(session):
        if session["tab_id"]:
            return session
        body = {
            "userId": session["user_id"],
            "listItemId": session["session_key"],
        }
        if url is not None:
            body["url"] = url
        data = _post("/tabs", body)
        session["tab_id"] = data.get("tabId")
        return session


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info."""
    task_id = task_id or "default"
    with _sessions_lock:
        camofox_cfg = _get_camofox_config()
        identity = _camofox_identity_override(task_id, camofox_cfg) or get_camofox_identity(task_id)
        return _sessions.pop(_session_cache_key(task_id, identity), None)


def _release_local_server_lease() -> None:
    """Best-effort release of the profile's long-lived Agent runtime lease."""
    try:
        _post("/_zettlab/release", {}, timeout=5)
    except Exception as exc:
        logger.debug("Camofox local-server lease release failed: %s", exc)


def _takeover_ui_hint(session: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Build the opaque identifiers the App needs to claim this exact tab."""
    if not _local_server_managed():
        return None
    agent_id = _runtime_value("ZET_AGENT_ID").strip()
    browser_session_id = str(session.get("session_key") or "").strip()
    tab_id = str(session.get("tab_id") or "").strip()
    if not agent_id or not browser_session_id or not tab_id:
        return None
    return {
        "type": "takeover_browser",
        "agent_id": agent_id,
        "browser_session_id": browser_session_id,
        "tab_id": tab_id,
    }


def camofox_soft_cleanup(task_id: Optional[str] = None) -> bool:
    """Release the in-memory session without destroying the server-side context.

    When managed persistence is enabled the browser profile (and its cookies)
    must survive across agent tasks.  This helper drops only the local tracking
    entry and returns ``True``.  When managed persistence is *not* enabled it
    does nothing and returns ``False`` so the caller can fall back to
    :func:`camofox_close`.
    """
    camofox_cfg = _get_camofox_config()
    if (
        _local_server_managed()
        or bool(camofox_cfg.get("managed_persistence"))
        or _camofox_identity_override(task_id, camofox_cfg)
    ):
        _drop_session(task_id)
        if _local_server_managed():
            _release_local_server_lease()
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True
    return False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _safe_http_error_payload(resp: requests.Response) -> Dict[str, Any]:
    """Build an Agent-safe error without exposing arbitrary response text."""
    payload: Dict[str, Any] = {"success": False}
    try:
        body = resp.json()
    except (TypeError, ValueError):
        body = None

    if isinstance(body, dict):
        for field, limit in _SAFE_HTTP_ERROR_STRING_FIELDS.items():
            value = body.get(field)
            if isinstance(value, str) and value:
                payload[field] = value[:limit]
        if isinstance(body.get("retryable"), bool):
            payload["retryable"] = body["retryable"]

    if "error" not in payload:
        payload["error"] = f"HTTP {resp.status_code}"
    return payload


def _raise_for_status(resp: requests.Response) -> None:
    if 200 <= resp.status_code < 400:
        return
    raise CamofoxHTTPError(resp, _safe_http_error_payload(resp))


def _post(path: str, body: dict, timeout: Optional[int] = None) -> dict:
    """POST JSON to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.post(url, json=body, timeout=timeout, headers=_auth_headers())
    _raise_for_status(resp)
    return resp.json()


def _get(path: str, params: dict = None, timeout: Optional[int] = None) -> dict:
    """GET from camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout, headers=_auth_headers())
    _raise_for_status(resp)
    return resp.json()


def _get_raw(path: str, params: dict = None, timeout: Optional[int] = None) -> requests.Response:
    """GET from camofox and return raw response (for binary data)."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.get(url, params=params, timeout=timeout, headers=_auth_headers())
    _raise_for_status(resp)
    return resp


def _delete(path: str, body: dict = None, timeout: Optional[int] = None) -> dict:
    """DELETE to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.delete(url, json=body, timeout=timeout, headers=_auth_headers())
    _raise_for_status(resp)
    return resp.json()


def _control_error_payload(exc: BaseException) -> Optional[Dict[str, Any]]:
    """Return a validated local-server control error envelope, if present."""
    if not isinstance(exc, requests.HTTPError) or exc.response is None:
        return None
    try:
        payload = exc.response.json()
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    status = exc.response.status_code
    code = payload.get("error")
    if (status, code) not in {
        (423, "browser_human_controlled"),
        (409, "browser_resnapshot_required"),
    }:
        return None
    return payload


def _retryable_control_result(
    exc: BaseException,
    session: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Translate takeover/handback HTTP responses into retryable tool JSON.

    A handback invalidates accessibility refs. On the 409 transition response,
    Hermes captures a fresh snapshot and acknowledges the resume token before
    asking the model to retry. The blocked action itself is never replayed.
    """
    payload = _control_error_payload(exc)
    if payload is None:
        return None

    code = payload["error"]
    result: Dict[str, Any] = {
        "success": False,
        "error": code,
        "retryable": True,
    }
    message = payload.get("message")
    if isinstance(message, str) and message:
        result["message"] = message[:500]
    retry_after_ms = payload.get("retry_after_ms")
    if isinstance(retry_after_ms, int) and not isinstance(retry_after_ms, bool):
        result["retry_after_ms"] = max(0, min(retry_after_ms, 60_000))
    takeover_session_id = payload.get("takeover_session_id")
    if isinstance(takeover_session_id, str) and takeover_session_id:
        result["takeover_session_id"] = takeover_session_id[:256]

    if code != "browser_resnapshot_required":
        return json.dumps(result)

    result["resnapshot_completed"] = False
    result["resume_acknowledged"] = False
    if not session or not session.get("tab_id") or not session.get("user_id"):
        return json.dumps(result)

    # Human-entered values can remain in the current accessibility tree after
    # handback. Keep filtering every later read until Hermes explicitly leaves
    # this page or clears the local session.
    _set_handback_privacy_filter(session, True)

    try:
        before_tabs = _get(
            "/tabs",
            params={"userId": session["user_id"]},
            timeout=5,
        ).get("tabs", [])
        before_tab = _exact_session_tab(before_tabs, session)
        resume_origin = before_tab.get("resumeUrlOrigin") if before_tab else None
        pending_url = before_tab.get("url") if before_tab else None
        if (
            not isinstance(pending_url, str)
            or pending_url != ""
            or not isinstance(resume_origin, str)
            or not _valid_resume_url_origin(resume_origin.strip())
            or _unsafe_handback_url(resume_origin.strip())
        ):
            result.update({
                "error": "browser_handback_origin_blocked",
                "message": "Browser handback lacks a safe resume origin. Close the browser session before retrying.",
                "retryable": False,
            })
            return json.dumps(result)

        snapshot_data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )
        snapshot = snapshot_data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""

        after_tabs = _get(
            "/tabs",
            params={"userId": session["user_id"]},
            timeout=5,
        ).get("tabs", [])
        after_tab = _exact_session_tab(after_tabs, session)
        current_url = after_tab.get("url") if after_tab else None
        if (
            not isinstance(current_url, str)
            or not current_url.strip()
            or _unsafe_handback_url(current_url.strip())
        ):
            result.update({
                "error": "browser_handback_url_blocked",
                "message": "Browser handback ended on an unsafe or unverifiable URL. Close the browser session before retrying.",
                "retryable": False,
            })
            return json.dumps(result)

        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            snapshot = _truncate_snapshot(snapshot)
        result["snapshot"] = _filter_page_state_after_handback(session, snapshot)
        result["element_count"] = snapshot_data.get("refsCount", 0)
        result["resnapshot_completed"] = True
        result["url"] = _redact_handback_url(current_url.strip())
        result["title"] = "[REDACTED after human control]"
    except Exception as snapshot_exc:
        logger.warning("Camofox handback resnapshot failed: %s", snapshot_exc)
        return json.dumps(result)

    resume_token = payload.get("resume_token")
    if not isinstance(resume_token, str) or not resume_token:
        return json.dumps(result)

    try:
        _post(
            "/_zettlab/control/resume/ack",
            {
                "userId": session["user_id"],
                "tabId": session["tab_id"],
                "resumeToken": resume_token,
            },
        )
        result["resume_acknowledged"] = True
    except Exception as ack_exc:
        logger.warning("Camofox handback acknowledgement failed: %s", ack_exc)
    return json.dumps(result)


def _tool_error_from_exception(
    exc: BaseException,
    *,
    session: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    retryable = _retryable_control_result(exc, session)
    if retryable is not None:
        payload = json.loads(retryable)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(exc, CamofoxHTTPError):
        payload = dict(exc.payload)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(exc, requests.ConnectionError):
        payload = {
            "success": False,
            "error": "browser_runtime_unavailable",
            "message": (
                "Managed local browser service is unavailable. "
                "Retry later or check the device browser environment."
            ),
            "retryable": True,
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
    return tool_error(f"{prefix}{exc}", success=False, **(extra or {}))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _navigation_tab_context(session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return enough opaque tab identity for takeover after navigation fails."""
    if not session or not session.get("tab_id"):
        return {}
    context: Dict[str, Any] = {"tabId": session["tab_id"]}
    takeover_hint = _takeover_ui_hint(session)
    if takeover_hint:
        context["ui_hint"] = takeover_hint
    return context


def camofox_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via Camofox."""
    session: Optional[Dict[str, Any]] = None
    try:
        browser_url, rewrite_info = _rewrite_loopback_url_for_camofox(url)
        # Camofox 1.13 rejects non-http(s) values such as ``about:blank`` when
        # ``url`` is present on tab creation. Omitting it creates the blank tab
        # and gives Hermes a stable tabId before navigation can time out.
        session = _ensure_tab(task_id)
        try:
            data = _post(
                f"/tabs/{session['tab_id']}/navigate",
                {"userId": session["user_id"], "url": browser_url},
                timeout=60,
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning(
                    "Camofox tab %s returned 404 — tab was garbage collected. "
                    "Creating a fresh tab.",
                    session["tab_id"],
                )
                session["tab_id"] = None
                session = _ensure_tab(task_id)
                data = _post(
                    f"/tabs/{session['tab_id']}/navigate",
                    {"userId": session["user_id"], "url": browser_url},
                    timeout=60,
                )
            else:
                raise
        _set_handback_privacy_filter(session, False)
        result = {
            "success": True,
            "url": data.get("url", browser_url),
            "title": data.get("title", ""),
            "tabId": session["tab_id"],
        }
        takeover_hint = _takeover_ui_hint(session)
        if takeover_hint:
            result["ui_hint"] = takeover_hint
        if rewrite_info:
            result["requested_url"] = url
            result["url_rewrite"] = rewrite_info
            result["warning"] = (
                "Rewrote loopback URL for Docker-hosted Camofox: "
                f"{rewrite_info['from']} -> {rewrite_info['to']}"
            )
        vnc = get_vnc_url()
        if vnc:
            result["vnc_url"] = vnc
            result["vnc_hint"] = (
                "Browser is visible via VNC. "
                "Share this link with the user so they can watch the browser live."
            )

        # Auto-take a compact snapshot so the model can act immediately
        try:
            snap_data = _get(
                f"/tabs/{session['tab_id']}/snapshot",
                params={"userId": session["user_id"]},
            )
            snapshot_text = snap_data.get("snapshot", "")
            from tools.browser_tool import (
                SNAPSHOT_SUMMARIZE_THRESHOLD,
                _truncate_snapshot,
            )
            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = _truncate_snapshot(snapshot_text)
            result["snapshot"] = snapshot_text
            result["element_count"] = snap_data.get("refsCount", 0)
        except Exception:
            pass  # Navigation succeeded; snapshot is a bonus

        return json.dumps(result)
    except requests.HTTPError as e:
        return _tool_error_from_exception(
            e,
            session=session,
            prefix="Navigation failed: ",
            extra=_navigation_tab_context(session),
        )
    except Exception as e:
        return _tool_error_from_exception(
            e,
            session=session,
            extra=_navigation_tab_context(session),
        )


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None,
                     user_task: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )

        snapshot = data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""
        snapshot = _filter_page_state_after_handback(session, snapshot)
        refs_count = data.get("refsCount", 0)

        # Apply same summarization logic as the main browser tool
        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _extract_relevant_content,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            if user_task:
                snapshot = _extract_relevant_content(snapshot, user_task)
            else:
                snapshot = _truncate_snapshot(snapshot)

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        # Strip @ prefix if present (our tool convention)
        clean_ref = ref.lstrip("@")

        data = _post(
            f"/tabs/{session['tab_id']}/click",
            {"userId": session["user_id"], "ref": clean_ref},
        )
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": data.get("url", ""),
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        clean_ref = ref.lstrip("@")

        _post(
            f"/tabs/{session['tab_id']}/type",
            {"userId": session["user_id"], "ref": clean_ref, "text": text},
        )
        from agent.display import (
            redact_browser_typed_text_for_display,
            redact_tool_args_for_display,
        )

        display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]

        response = {
            "success": True,
            # Match browser_tool.browser_type: run typed text through the
            # secret-pattern redactor so API keys / tokens don't leak into
            # tool progress or chat history.  The raw text is still typed into
            # the page; only the returned display value is redacted.
            "typed": display_text,
            "element": clean_ref,
        }
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response)
    except Exception as e:
        from agent.display import redact_browser_typed_text_for_display

        failure = json.loads(
            _tool_error_from_exception(e, session=locals().get("session"))
        )
        failure = redact_browser_typed_text_for_display(failure, text)
        return json.dumps(failure, ensure_ascii=False)


def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        _post(
            f"/tabs/{session['tab_id']}/scroll",
            {"userId": session["user_id"], "direction": direction},
        )
        return json.dumps({"success": True, "scrolled": direction})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        data = _post(
            f"/tabs/{session['tab_id']}/back",
            {"userId": session["user_id"]},
        )
        return json.dumps({"success": True, "url": data.get("url", "")})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        _post(
            f"/tabs/{session['tab_id']}/press",
            {"userId": session["user_id"], "key": key},
        )
        return json.dumps({"success": True, "pressed": key})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = _drop_session(task_id)
        if not session:
            return json.dumps({"success": True, "closed": True})

        if _local_server_managed():
            _release_local_server_lease()
            return json.dumps({
                "success": True,
                "closed": False,
                "released": True,
            })

        _delete(
            f"/sessions/{session['user_id']}",
        )
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        return json.dumps({"success": True, "closed": True, "warning": str(e)})


def camofox_get_images(task_id: Optional[str] = None) -> str:
    """Get images on the current page via Camofox.

    Extracts image information from the accessibility tree snapshot,
    since Camofox does not expose a dedicated /images endpoint.
    """
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        import re

        data = _get(
            f"/tabs/{session['tab_id']}/snapshot",
            params={"userId": session["user_id"]},
        )
        snapshot = data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""
        snapshot = _filter_page_state_after_handback(session, snapshot)

        # Parse img elements from the accessibility tree.
        # Format: img "alt text" or img "alt text" [eN]
        # URLs appear on /url: lines following img entries
        images = []
        lines = snapshot.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("- img ", "img ")):
                alt_match = re.search(r'img\s+"([^"]*)"', stripped)
                alt = alt_match.group(1) if alt_match else ""
                # Look for URL on the next line
                src = ""
                if i + 1 < len(lines):
                    url_match = re.search(r'/url:\s*(\S+)', lines[i + 1].strip())
                    if url_match:
                        src = url_match.group(1)
                if alt or src:
                    images.append({"src": src, "alt": alt})

        return json.dumps({
            "success": True,
            "images": images,
            "count": len(images),
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_vision(question: str, annotate: bool = False,
                   task_id: Optional[str] = None) -> str:
    """Take a screenshot and analyze it with vision AI via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)
        if _handback_privacy_filter_enabled(session):
            return tool_error(
                "Browser vision is blocked after human control until the Agent navigates to a new page or closes the session.",
                success=False,
            )

        # Get screenshot as binary PNG
        resp = _get_raw(
            f"/tabs/{session['tab_id']}/screenshot",
            params={"userId": session["user_id"]},
        )

        # Save screenshot to cache
        from hermes_constants import get_hermes_home
        screenshots_dir = get_hermes_home() / "browser_screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = str(screenshots_dir / f"browser_screenshot_{uuid.uuid4().hex[:8]}.png")

        with open(screenshot_path, "wb") as f:
            f.write(resp.content)

        # Encode for vision LLM
        img_b64 = base64.b64encode(resp.content).decode("utf-8")

        # Also get annotated snapshot if requested
        annotation_context = ""
        if annotate:
            try:
                snap_data = _get(
                    f"/tabs/{session['tab_id']}/snapshot",
                    params={"userId": session["user_id"]},
                )
                snapshot = snap_data.get("snapshot", "")
                if not isinstance(snapshot, str):
                    snapshot = ""
                snapshot = _filter_page_state_after_handback(session, snapshot)
                annotation_context = f"\n\nAccessibility tree (element refs for interaction):\n{snapshot[:3000]}"
            except Exception:
                pass

        # Redact secrets from annotation context before sending to vision LLM.
        # The screenshot image itself cannot be redacted, but at least the
        # text-based accessibility tree snippet won't leak secret values.
        from agent.redact import redact_sensitive_text
        annotation_context = redact_sensitive_text(annotation_context)

        # Send to vision LLM
        from agent.auxiliary_client import call_llm

        vision_prompt = (
            f"Analyze this browser screenshot and answer: {question}"
            f"{annotation_context}"
        )

        try:
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vision_timeout = float(_vision_cfg.get("timeout", 120))
            _vision_temperature = float(_vision_cfg.get("temperature", 0.1))
        except Exception:
            _vision_timeout = 120.0
            _vision_temperature = 0.1

        response = call_llm(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }],
            task="vision",
            temperature=_vision_temperature,
            timeout=_vision_timeout,
        )
        analysis = (response.choices[0].message.content or "").strip() if response.choices else ""

        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text
        analysis = redact_sensitive_text(analysis)

        return json.dumps({
            "success": True,
            "analysis": analysis,
            "screenshot_path": screenshot_path,
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))


def camofox_console(clear: bool = False, task_id: Optional[str] = None) -> str:
    """Get console output — limited support in Camofox.

    Camofox does not expose browser console logs via its REST API.
    Returns an empty result with a note.
    """
    return json.dumps({
        "success": True,
        "console_messages": [],
        "js_errors": [],
        "total_messages": 0,
        "total_errors": 0,
        "note": "Console log capture is not available with the Camofox backend. "
                "Use browser_snapshot or browser_vision to inspect page state.",
    })
