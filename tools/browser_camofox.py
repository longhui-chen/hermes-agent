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

import atexit
import base64
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
import uuid
from typing import Any, Dict, Optional
from urllib.parse import SplitResult, quote, urlsplit, urlunsplit

import requests

from hermes_cli.config import cfg_get, load_config, read_raw_config


# Camofox tab IDs are opaque handles; anything outside this alphabet cannot be
# a legitimate ID and must never reach a request path.
_TAB_ID_PATTERN = re.compile(r"\A[A-Za-z0-9_.:-]{1,128}\Z")

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
# The optional bracketed group is what makes an IPv6 authority match at all:
# without it `https://[2001:db8::1]/cb?code=…` is skipped entirely (brackets are
# excluded from the tail so a URL inside markdown/parentheses is not swallowed),
# and the whole query would reach the model verbatim.
_HANDBACK_URL = re.compile(
    # userinfo may precede a bracketed IPv6 authority, and both are optional.
    # Without allowing that combination the match stops at the "@" and the rest
    # of the URL — path, query, OAuth code — is left in the text verbatim.
    # The brackets themselves delimit the authority, so anything up to the
    # closing one is accepted — an RFC 6874 zone identifier (`[fe80::1%25eth0]`)
    # contains letters outside the hex alphabet and would otherwise stop the
    # match at the scheme, leaving the path and query in the text.
    # Parentheses are legal in a path (`/(S(secret))/callback`), so excluding
    # them left the sensitive tail in the text. They are accepted here and the
    # whole match is replaced by the origin; over-matching a trailing delimiter
    # from surrounding prose costs a bracket, under-matching costs a token.
    r"(?i)\bhttps?://(?:[^\s\"'<>\[\]/@]*@)?(?:\[[^\]\s]+\])?[^\s\"'<>]*"
)


def _url_origin_only(url: str) -> str:
    """Reduce a URL to its origin.

    Pages a human just controlled routinely carry session material in URL
    paths and queries (OAuth codes, reset tokens, pre-signed links), and the
    general redaction policy deliberately preserves web URL queries — so the
    handback privacy filter must drop everything past the origin itself.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "[REDACTED URL]"
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "[REDACTED URL]"
    # Rebuild from hostname (+ port) only. netloc would keep any
    # ``user:password@`` userinfo, which is exactly the kind of credential a
    # human may have typed into a basic-auth URL during handback.
    host = parsed.hostname
    if ":" in host:
        # urlsplit strips the brackets off an IPv6 literal; put them back or
        # the rebuilt origin is not a valid URL.
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    authority = f"{host}:{port}" if port is not None else host
    return f"{parsed.scheme}://{authority}/"


def _reduce_urls_to_origin(value: str) -> str:
    return _HANDBACK_URL.sub(lambda match: _url_origin_only(match.group(0)), value)


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
    return _reduce_urls_to_origin("\n".join(lines))


_EPOCH_HEADER = "X-Zettlab-Browser-Epoch"


def _session_epoch_header(session: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Declare the page epoch this session last synchronized with."""
    if not isinstance(session, dict):
        return {}
    epoch = session.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool):
        return {}
    return {_EPOCH_HEADER: str(epoch)}


def _adopt_session_epoch(session: Optional[Dict[str, Any]], epoch: Any) -> None:
    """Record the server-reported page epoch on the session.

    The epoch increments whenever human control of the page ends, so a change
    means the page content is no longer what the Agent last saw and any values
    a human typed may still be present. Adopting a changed epoch therefore
    also enables the handback privacy filter; the filter clears when the Agent
    navigates away. Keys are written without the session lock because single
    key access is atomic and callers may already hold the lock.
    """
    if not isinstance(session, dict) or not isinstance(epoch, int) or isinstance(epoch, bool):
        return
    previous = session.get("epoch")
    if previous is not None and previous != epoch:
        session["privacy_filter_after_handback"] = True
        # The epoch only moves when a human took the tab and gave it back, so
        # the document every outstanding ref describes is the one they left
        # behind. Turns sharing this physical tab keep their own session entry
        # and their own stamp, so invalidate by document rather than by
        # clearing this session's — otherwise a concurrent turn would still
        # act on refs from before the takeover.
    session["epoch"] = epoch


def _adopt_epoch_from_response(
    session: Optional[Dict[str, Any]],
    resp: "requests.Response",
    *,
    tab_operation: bool = False,
) -> bool:
    """Adopt the epoch header a managed local-server proxy adds to responses.

    On a managed deployment the epoch is the only thing that tells this process
    a human touched the page. If a tab operation comes back without a usable
    one — a local-server too old to send it, a proxy that drops it, a garbled
    value — the handback filter would silently never engage and the next read
    would hand over whatever the human typed. Once this session has seen a
    valid epoch, a later tab response without one is a protocol failure, and
    the safe reading of it is "assume the page changed".
    """
    before = bool(session.get("privacy_filter_after_handback")) if isinstance(session, dict) else False
    status = getattr(resp, "status_code", None)
    succeeded = isinstance(status, int) and 200 <= status < 300
    # Only a response that describes a completed operation may move the epoch.
    # This runs before _raise_for_status, so an older or hostile proxy putting a
    # header on its own 409 browser_epoch_stale would otherwise hand the Agent
    # the very epoch that refusal was protecting — and if the recovery snapshot
    # then failed, the next press or back would carry it and be accepted.
    value = resp.headers.get(_EPOCH_HEADER) if (resp is not None and succeeded) else None
    if value is not None:
        try:
            _adopt_session_epoch(session, int(str(value).strip()))
            if isinstance(session, dict) and not before and session.get("privacy_filter_after_handback"):
                _response_facts.started_handback = True
            return True
        except (TypeError, ValueError):
            pass
    if not tab_operation or not isinstance(session, dict):
        return False
    if not succeeded:
        # Error envelopes are generated before dispatch and carry no page data,
        # so a missing header there says nothing about the page.
        return False
    if not session.get("local_server_managed"):
        return False
    logger.warning("Camofox managed tab response carried no usable %s header", _EPOCH_HEADER)
    session["privacy_filter_after_handback"] = True
    _response_facts.started_handback = True
    return False


def _set_handback_privacy_filter(session: Dict[str, Any], enabled: bool) -> None:
    """Persist handback privacy filtering for subsequent reads of this tab."""
    with _session_lock(session):
        session["privacy_filter_after_handback"] = enabled


def _handback_privacy_filter_enabled(session: Dict[str, Any]) -> bool:
    """Return whether raw page reads are blocked after human control."""
    with _session_lock(session):
        return bool(session.get("privacy_filter_after_handback"))


def _filter_page_state_after_handback(
    session: Dict[str, Any], value: str, filtered_at_request: bool = False
) -> str:
    """Filter page state while a human-mutated page remains current.

    ``filtered_at_request`` carries the state from when the read was issued.
    Concurrent turns share one session dict, so a navigate finishing in between
    could otherwise clear the flag and let a capture taken under the filter
    through unredacted.
    """
    if filtered_at_request or _handback_privacy_filter_enabled(session):
        return _redact_handback_page_state(value)
    return value


def _filter_url_after_handback(session: Dict[str, Any], url: Any, revealed: bool = False) -> Any:
    """Reduce an operation-result URL to its origin while the filter is active.

    Click/back results report the page URL the human left behind; without this
    the origin-only policy applied to snapshots could be bypassed by reading
    the same URL from an action result.
    """
    if not isinstance(url, str) or not url:
        return url
    if not revealed and not _handback_privacy_filter_enabled(session):
        return url
    return _url_origin_only(url)


from tools.browser_camofox_state import get_camofox_identity
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30  # fallback when config is unreadable
_TAB_CREATION_TIMEOUT_FLOOR = 60  # managed Camofox may need a cold start
_SNAPSHOT_MAX_CHARS = 80_000  # camofox paginates at this limit
_vnc_url: Optional[str] = None  # single-profile cache from /health response
_vnc_url_checked = False  # only probe once per single-profile process

# Cached command timeout from config (resolved lazily, like browser_tool)
_cached_cmd_timeout: Optional[int] = None
_cmd_timeout_resolved = False

# Every request below carries X-Zettlab-Agent-Action-Token to a loopback
# endpoint. requests would otherwise honour HTTP_PROXY / ALL_PROXY from the
# environment whenever NO_PROXY does not cover loopback, handing this device's
# Agent authority to whatever host that proxy points at. Validating the URL is
# not enough — the transport has to refuse the proxy as well.
_NO_ENV_PROXIES = {"http": None, "https": None, "all": None}


_SAFE_HTTP_ERROR_STRING_FIELDS = {
    "error": 256,
    "message": 1_000,
    "detail": 2_000,
    "phase": 128,
    "takeover_session_id": 256,
    "takeoverSessionId": 256,
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


def _is_trusted_action_token_endpoint(url: str) -> bool:
    """Whether ``url`` is the local-server proxy the action token belongs to."""
    if not url:
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme != "http":
        # The proxy is plain HTTP on loopback; anything else means the value
        # was repointed somewhere this credential does not belong.
        return False
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        return True
    # Everything else must be a literal loopback address. A prefix test would
    # accept names like `127.attacker.example`, which resolve wherever their
    # owner points them.
    import ipaddress

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
        # The token grants local Agent authority, so it is only ever sent to
        # the local-server proxy on loopback. If the profile's .env is
        # misconfigured or overwritten with an external address, fail closed
        # rather than hand the credential to whoever answers.
        if not _is_trusted_action_token_endpoint(get_camofox_url()):
            raise RuntimeError(
                "CAMOFOX_AUTH_MODE=zettlab_action_token requires a loopback "
                "CAMOFOX_URL; refusing to send the Agent action token to "
                "a non-local endpoint"
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
        # Never follow a redirect: requests keeps custom headers across hops, so
        # a misconfigured or compromised proxy answering /health with a 30x
        # would receive the Agent action token at another origin.
        resp = requests.get(
            f"{url}/health", timeout=5, headers=_auth_headers(), allow_redirects=False,
            proxies=_NO_ENV_PROXIES,
        )
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


def _session_cache_key(task_id: str, identity: Dict[str, str], owner: str = "") -> str:
    """Key a cached session by who it belongs to as well as what it is.

    ``owner`` is the credential-derived profile identity. Without it, two
    profiles in one multiplex gateway that were given the same explicit
    CAMOFOX_USER_ID and session key collide on a single entry: they are
    isolated by different action tokens and talk to different runtimes, but
    they would share a tab id, an epoch and a handback privacy flag, so one
    profile would drive the other's tab and could clear its privacy state.
    """
    return f"{owner}\x00{identity['user_id']}\x00{identity['session_key']}\x00{task_id}"


def _validated_tab_id(raw: Any) -> Optional[str]:
    """Return ``raw`` if it is a well-formed opaque Camofox tab ID, else None.

    The ID comes back from the runtime and is then interpolated into
    ``/tabs/<id>/...`` request paths that carry
    ``X-Zettlab-Agent-Action-Token``. A malformed or hostile value containing
    ``/``, ``..``, ``?`` or ``#`` could re-target those privileged requests at
    other local-server routes, so only the documented opaque form is accepted.
    """
    if not isinstance(raw, str):
        return None
    tab_id = raw.strip()
    if not _TAB_ID_PATTERN.match(tab_id):
        return None
    # `.` and `..` are made of allowed characters but are path segments, not
    # identifiers: quote() leaves dots alone, so `/tabs/../snapshot` would
    # survive to whatever normalizes the path next.
    if set(tab_id) <= {"."}:
        return None
    return tab_id


def _tab_path(session: Dict[str, Any], suffix: str = "") -> str:
    """Build a ``/tabs/<id>`` path with the ID encoded as a single segment."""
    tab_id = _validated_tab_id(session.get("tab_id"))
    if tab_id is None:
        raise ValueError("browser tab id is missing or malformed")
    return f"/tabs/{quote(tab_id, safe='')}{suffix}"


def _session_lock(session: Dict[str, Any]) -> threading.Lock:
    with _sessions_lock:
        lock = session.get("_lock")
        if lock is None:
            lock = threading.Lock()
            session["_lock"] = lock
        return lock


# Last epoch this process observed per tab. It deliberately outlives the
# per-turn session cache so an ordinary multi-turn continuation can be told
# apart from a gateway restart; bounded because a stale entry is only ever a
# missed filter-suppression, never a leak.
_MAX_REMEMBERED_TAB_EPOCHS = 256
_remembered_tab_epochs: Dict[str, tuple] = {}


def _tab_epoch_memory_key(session: Dict[str, Any], tab_id: str) -> str:
    # Keyed by the credential-derived owner as well, like the session cache and
    # the document registry. Two profiles in one multiplex gateway can be given
    # the same explicit identity and be handed the same tab id by their own
    # runtimes; sharing this record would let one profile's "filter was off"
    # become the other's trusted state and clear a live handback filter.
    return (
        f"{session.get('release_owner') or ''}\x00{session.get('user_id')}"
        f"\x00{session.get('session_key')}\x00{tab_id}"
    )


def _remembered_tab_state(session: Dict[str, Any], tab_id: str) -> Optional[tuple]:
    with _sessions_lock:
        return _remembered_tab_epochs.get(_tab_epoch_memory_key(session, tab_id))


def _remember_tab_epoch(session: Optional[Dict[str, Any]]) -> None:
    """Record the epoch a still-owned tab was last seen at, with its filter.

    The privacy flag travels with the epoch because it does not follow from it:
    a handback detected during this turn leaves the filter on at an epoch the
    server also reports, so remembering the epoch alone would let the next
    turn's adoption conclude "nothing happened" and clear the filter over a
    page the human just typed into.
    """
    if not isinstance(session, dict):
        return
    tab_id = session.get("tab_id")
    epoch = session.get("epoch")
    if not tab_id or not isinstance(epoch, int) or isinstance(epoch, bool):
        return
    filtered = bool(session.get("privacy_filter_after_handback"))
    with _sessions_lock:
        if len(_remembered_tab_epochs) >= _MAX_REMEMBERED_TAB_EPOCHS:
            _remembered_tab_epochs.clear()
        _remembered_tab_epochs[_tab_epoch_memory_key(session, tab_id)] = (epoch, filtered)


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
    tab_id = _validated_tab_id(latest.get("tabId")) if isinstance(latest, dict) else None
    if tab_id:
        session["tab_id"] = tab_id
        adopted_epoch = latest.get("epoch")
        _adopt_session_epoch(session, adopted_epoch)
        # The in-process session cache is dropped at the end of every turn by
        # cleanup_task_resources, so most adoptions are an ordinary multi-turn
        # continuation, not a gateway restart. Filtering those would blank out
        # every form control and block vision/eval from the second turn on.
        # Compare against the last epoch this process saw for the tab instead:
        # unchanged means no handback happened, anything else (including no
        # record at all, i.e. a genuine restart) filters until the Agent
        # navigates.
        remembered = _remembered_tab_state(session, tab_id)
        recognized = (
            remembered is not None
            and isinstance(adopted_epoch, int)
            and not isinstance(adopted_epoch, bool)
            and remembered[0] == adopted_epoch
        )
        # Recognized means this process saw the tab at exactly this epoch last
        # turn, so restore the filter state it had then — which stays on when
        # that turn ended mid-handback. Anything else (moved epoch, or no
        # record at all, i.e. a genuine restart) filters until the Agent
        # navigates away.
        session["privacy_filter_after_handback"] = remembered[1] if recognized else True
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
    # Captured here because lifecycle cleanup runs outside this request's
    # profile and secret scope, where these reads fail closed: the probe would
    # skip releasing the local-server lease, and the release call itself could
    # no longer resolve its endpoint or credential.
    local_server_managed = _local_server_managed()
    camofox_base = get_camofox_url()
    try:
        scoped_headers = _auth_headers()
    except Exception:
        scoped_headers = {}
    release_url = f"{camofox_base}/_zettlab/release" if local_server_managed else ""
    release_headers = dict(scoped_headers) if local_server_managed else {}
    # Direct Camofox sessions are torn down with DELETE /sessions/<user_id>,
    # and that runs from the scope-less maintenance thread too, so its endpoint
    # and credential have to be captured here like the lease context is.
    delete_base = "" if local_server_managed else camofox_base
    delete_headers = {} if local_server_managed else dict(scoped_headers)
    # The lease is per profile, and in multiplex every profile shares the same
    # loopback CAMOFOX_URL — only the credential differs. Group holders by the
    # profile identity plus a digest of that credential, never by URL.
    release_owner = _release_owner_key(cache_identity.get("user_id", ""), scoped_headers, camofox_base)
    # Computed before the cache lookup: the credential digest inside it is what
    # separates two profiles that were handed the same explicit identity.
    cache_key = _session_cache_key(task_id, cache_identity, release_owner)
    now = time.monotonic()
    overflowing = False
    created_entry = False
    with _held_owner_lock(release_owner), _sessions_lock:
        idle = _prune_idle_sessions_locked(now)
        if cache_key in _sessions:
            session = _sessions[cache_key]
            session["last_used_at"] = now
            # The gateway can rotate the action token or repoint CAMOFOX_URL
            # while this entry lives. Browser requests would pick the new values
            # up, but teardown uses what was captured at creation — so refresh
            # it here or the release/delete goes to the old endpoint with the
            # old credential and the current runtime loses its only handle.
            if release_owner and session.get("release_owner") != release_owner:
                logger.debug("Camofox session %s adopting rotated teardown context", task_id)
            session["release_url"] = release_url
            session["release_headers"] = release_headers
            session["release_owner"] = release_owner
            session["delete_base"] = delete_base
            session["delete_headers"] = delete_headers
            session["local_server_managed"] = local_server_managed
        else:
            if identity_override:
                session = {
                    "user_id": identity_override["user_id"],
                    "tab_id": None,
                    "session_key": identity_override["session_key"],
                    "managed": True,
                    "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                    "privacy_filter_after_handback": False,
                    "epoch": None,
                    "task_id": task_id,
                    "local_server_managed": local_server_managed,
                    "release_url": release_url,
                    "release_headers": release_headers,
                    "release_owner": release_owner,
                    "delete_base": delete_base,
                    "delete_headers": delete_headers,
                    "last_used_at": now,
                    "_lock": threading.Lock(),
                }
            elif local_server_managed or bool(camofox_cfg.get("managed_persistence")):
                session = {
                    "user_id": profile_identity["user_id"],
                    "tab_id": None,
                    "session_key": profile_identity["session_key"],
                    "managed": True,
                    "adopt_existing_tab": _adopt_existing_tab_enabled(camofox_cfg),
                    "privacy_filter_after_handback": False,
                    "epoch": None,
                    "task_id": task_id,
                    "local_server_managed": local_server_managed,
                    "release_url": release_url,
                    "release_headers": release_headers,
                    "release_owner": release_owner,
                    "delete_base": delete_base,
                    "delete_headers": delete_headers,
                    "last_used_at": now,
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
                    "epoch": None,
                    "task_id": task_id,
                    "local_server_managed": local_server_managed,
                    "release_url": release_url,
                    "release_headers": release_headers,
                    "release_owner": release_owner,
                    "delete_base": delete_base,
                    "delete_headers": delete_headers,
                    "last_used_at": now,
                    "_lock": threading.Lock(),
                }
            _sessions[cache_key] = session
            created_entry = True
        # The reference is taken here, under the same lock that admitted the
        # entry, and released by the caller. Handing the entry back unreferenced
        # left a window in which another _get_session could pick it as the one
        # idle entry, evict it and tear it down while this call was still
        # creating a tab on it.
        session["in_flight"] = int(session.get("in_flight") or 0) + 1
        session["last_used_at"] = now
        idle.extend(_evict_surplus_sessions_locked())
        # Backpressure, not a soft cap: if nothing could be evicted the cache is
        # full of work in flight, and admitting more would grow the cache and the
        # browser's tabs without bound. The caller sees a retryable failure.
        # Only a new entry can push the cache past its ceiling, and only a new
        # entry is refused: a turn coming back to a session that is already
        # tracked must not be turned away because its neighbours are busy.
        if created_entry and len(_sessions) > _MAX_TRACKED_SESSIONS:
            session["in_flight"] = max(0, int(session.get("in_flight") or 0) - 1)
            _sessions.pop(cache_key, None)
            overflowing = True
    for expired in idle:
        _teardown_session(expired)
    # Deliberately at most one, and only what is already due. A local-server
    # that stopped answering leaves entries whose attempt can take seconds
    # each, and draining the whole ready list here would put that wait in front
    # of an ordinary browser call. Retrying is the maintenance worker's job.
    _run_pending_teardowns(max_items=1)
    # A tracked session must be reclaimable even if this process never calls
    # into the browser again.
    _ensure_maintenance_worker()
    if overflowing:
        raise CamofoxSessionsBusy(
            "every tracked browser session is in use; retry when one finishes"
        )

    try:
        with _session_lock(session):
            return _adopt_existing_tab(session)
    except BaseException:
        # The caller never receives the session, so it can never release it.
        _end_session_call(session)
        raise


def _browser_identity_key(session: Dict[str, Any]) -> str:
    """The identity Camofox itself keys a tab by: profile plus list item.

    Deliberately not the task id. Concurrent turns under one session key — a
    parent and its subagent, say — resolve to the same browser identity but
    get their own cache entry and their own lock, so serializing on the task
    would let both create a tab under the same listItemId. Adoption can then
    only guess which one is "the" tab.
    """
    return f"{session.get('release_owner') or ''}\x00{session.get('user_id')}\x00{session.get('session_key')}"


def _ensure_tab(task_id: Optional[str], url: Optional[str] = None) -> Dict[str, Any]:
    """Ensure a tab exists for the session, creating one if needed.

    Returns an entry that already holds its cache reference — ownership passes
    to the caller, which must release it. A failure here never reaches the
    caller, so this releases it itself: an entry left referenced is skipped by
    both the idle sweep and eviction forever, and enough of them make every
    later call fail with browser_sessions_busy.
    """
    session = _get_session(task_id)
    try:
        if session["tab_id"]:
            return session
        # Serialized by browser identity, so a concurrent turn sharing it
        # cannot create a second tab for the same listItemId.
        with _held_owner_lock(_browser_identity_key(session)), _session_lock(session):
            if session["tab_id"]:
                return session
            # Another turn may have created it while this one waited; adopt
            # rather than duplicate.
            _adopt_existing_tab(session)
            if session["tab_id"]:
                return session
            body = {
                "userId": session["user_id"],
                "listItemId": session["session_key"],
            }
            if url is not None:
                body["url"] = url
            data = _post(
                "/tabs",
                body,
                timeout=max(_get_command_timeout(), _TAB_CREATION_TIMEOUT_FLOOR),
                session=session,
            )
            tab_id = _validated_tab_id(data.get("tabId"))
            if tab_id is None:
                raise ValueError("browser runtime returned a malformed tab id")
            session["tab_id"] = tab_id
            return session
    except BaseException:
        _end_session_call(session)
        raise


# A session whose profile scope is gone can only be reclaimed on a timer: the
# owner is unknowable at that point, and guessing by task id alone would reach
# into another profile's live browser state. The bound keeps _sessions and the
# local-server leases behind it from growing for the life of the process.
# One window, deliberately. Whether anyone is still using the browser — an
# Agent mid-call or a human mid-takeover — is not observable from this process,
# and every approximation of it tried here (turn-end release, a grace window,
# tiers keyed off takeover hints and refusals) was wrong in a different way.
# local-server owns the lease and knows who holds the tab; until it reports
# that, this stays a plain timer.
_SESSION_IDLE_TTL_SECONDS = 30 * 60


def _session_key_for_task_locked(task_id: str) -> Optional[str]:
    """The cache key this task owns, or None when it cannot be established.

    Only the identity-derived key is authoritative. Cleanup callers (idle
    reaper, ``/new``, shutdown) run outside the request's profile home and
    secret scope, where ``get_secret`` fails closed — and a task id is not
    owner-scoped, since the API's ``session_id`` becomes the effective task id
    and two profiles can legitimately carry the same one. Returning None there
    is deliberate: :func:`_prune_idle_sessions_locked` reclaims the entry with
    the release context captured while the scope still existed.
    Callers must hold ``_sessions_lock``.
    """
    try:
        camofox_cfg = _get_camofox_config()
        identity = _camofox_identity_override(task_id, camofox_cfg) or get_camofox_identity(task_id)
        try:
            owner = _release_owner_key(identity.get("user_id", ""), _auth_headers(), get_camofox_url())
        except Exception:
            owner = ""
    except Exception:
        return None
    preferred = _session_cache_key(task_id, identity, owner)
    return preferred if preferred in _sessions else None


# A hard ceiling as well as a TTL: a client rotating session keys inside the
# idle window would otherwise grow _sessions, its owner locks and the
# local-server tabs behind them without bound, and time-based expiry cannot
# bound memory on a 2 GB device.
_MAX_TRACKED_SESSIONS = 64


@contextmanager
def _capture_guard(session: Optional[Dict[str, Any]]):
    """Hold the tab identity across a read whose content must be judged.

    Unconditional, and that is the point. The capture and the URL check that
    decides whether the Agent may read it have to describe the same moment: a
    concurrent turn's navigate landing between them would let content captured
    on a page the Agent may not read pass, because the tab had since moved to
    one it may. Deciding by the filter state before the request cannot work —
    the first post-handback capture is the response that turns the filter on,
    which is exactly the capture that needs protecting.

    Only the HTTP capture is inside: vision's model round trip, which can run
    for minutes, happens outside it.
    """
    if not isinstance(session, dict):
        yield
        return
    with _held_owner_lock(_browser_identity_key(session)):
        # Re-checked here, inside the critical section. The caller's check ran
        # before this lock, and another turn holding it can navigate the shared
        # tab onto a refused page in between — an ordinary redirect turns no
        # filter on, so nothing downstream would notice.
        yield


@contextmanager
def _session_operation(session: Optional[Dict[str, Any]]):
    """Hold a session in use for a whole tool call, not just one request.

    A time window cannot do this job: a client rotating session keys keeps
    every entry inside any window, which turns the ceiling into no ceiling at
    all. Only an explicit reference says "someone is working with this", and it
    has to span the gaps inside a composite operation — vision takes a
    screenshot, calls a model for up to two minutes, then takes an annotation
    snapshot.

    The reference itself is taken by :func:`_get_session` under the cache lock,
    so there is no unreferenced moment between admission and use. This adopts
    that reference and releases it; every _get_session must be paired with one.
    """
    try:
        yield session
    finally:
        _end_session_call(session)


def _begin_session_call(session: Optional[Dict[str, Any]]) -> None:
    if not isinstance(session, dict):
        return
    with _sessions_lock:
        session["in_flight"] = int(session.get("in_flight") or 0) + 1
        session["last_used_at"] = time.monotonic()


def _end_session_call(session: Optional[Dict[str, Any]]) -> None:
    if not isinstance(session, dict):
        return
    with _sessions_lock:
        session["in_flight"] = max(0, int(session.get("in_flight") or 0) - 1)
        session["last_used_at"] = time.monotonic()


def _evict_surplus_sessions_locked(protect_key: str = "") -> list:
    """Drop the least recently used entries once past the ceiling.

    Returns them so the caller can tear them down outside the lock — evicting
    the tracking without releasing the browser state behind it would just move
    the leak somewhere less visible.

    ``protect_key`` names the entry the current call is about to use. It stands
    in for the reference that call has not been able to take yet.
    """
    surplus = len(_sessions) - _MAX_TRACKED_SESSIONS
    if surplus <= 0:
        return []
    by_age = sorted(_sessions.items(), key=lambda kv: float(kv[1].get("last_used_at") or 0.0))
    evicted = []
    for key, session in by_age:
        if len(evicted) >= surplus:
            break
        if protect_key and key == protect_key:
            continue
        # Evicting a session mid-operation would close the tab under a running
        # tool call — worse than being briefly over the ceiling.
        if int(session.get("in_flight") or 0) > 0:
            continue
        _sessions.pop(key, None)
        evicted.append(session)
    if not evicted:
        logger.warning(
            "Camofox session cache is over its %d-entry ceiling but every entry is in use",
            _MAX_TRACKED_SESSIONS,
        )
        return []
    logger.warning(
        "Camofox session cache exceeded %d entries; reclaimed %d idle",
        _MAX_TRACKED_SESSIONS, len(evicted),
    )
    return evicted

def _prune_idle_sessions_locked(now: float) -> list:
    """Drop sessions untouched past the idle TTL, returning them for release.

    Callers must hold ``_sessions_lock`` and must release the returned
    sessions' leases outside it.
    """
    expired = [
        key
        for key, session in _sessions.items()
        if int(session.get("in_flight") or 0) == 0
        and now - float(session.get("last_used_at") or 0.0) > _SESSION_IDLE_TTL_SECONDS
    ]
    dropped = []
    for key in expired:
        session = _sessions.pop(key, None)
        if session is not None:
            logger.debug("Camofox reclaimed idle session %s", session.get("task_id"))
            dropped.append(session)
    return dropped


def _peek_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the tracked session for task_id without removing it."""
    task_id = task_id or "default"
    with _sessions_lock:
        key = _session_key_for_task_locked(task_id)
        return _sessions.get(key) if key else None


def has_camofox_session(task_id: Optional[str] = None) -> bool:
    """Whether this process still tracks Camofox state this task owns."""
    return _peek_session(task_id) is not None


def _drop_session(task_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Remove and return session info, reclaiming idle sessions on the way."""
    task_id = task_id or "default"
    with _sessions_lock:
        key = _session_key_for_task_locked(task_id)
        dropped = _sessions.pop(key, None) if key else None
        idle = _prune_idle_sessions_locked(time.monotonic())
    for session in idle:
        _teardown_session(session)
    return dropped


# Releases that could not be delivered yet. The session they belonged to is
# already gone, so this is the only remaining handle on that lease; the entries
# are tiny (url + headers) and capped.
_MAX_PENDING_LEASE_RELEASES = 256
_pending_lease_releases: list = []


# Reclaim runs on its own clock: a turn whose cleanup happens after the profile
# scope is gone cannot drop its own session, and a failed teardown is often the
# last browser use of the process.
_MAINTENANCE_TICK_SECONDS = 30
# Quiet window before a shared profile's runtime is actually released.
_RELEASE_GRACE_SECONDS = 60
_MAX_PENDING_RELEASE_ATTEMPTS = 8
_maintenance_worker: Optional[threading.Thread] = None


def _run_pending_teardowns(
    force: bool = False,
    only_url: str = "",
    only_owner: str = "",
    max_items: int = 0,
    deadline: float = 0.0,
) -> None:
    """Drain scheduled teardowns.

    Releases are deliberately deferred: a profile's runtime is shared, and the
    number of turns currently using it is not observable from here (a turn calls
    _get_session many times but cleans up once, and two turns can share a task
    id). Waiting for a quiet window and re-checking under the owner lock is what
    makes "the last user is gone" true rather than guessed.
    """
    now = time.monotonic()
    with _sessions_lock:
        ready = [
            e for e in _pending_lease_releases
            if (
                (e["url"] == only_url and e.get("owner", "") == only_owner)
                if only_url
                else (force or e["ready_at"] <= now)
            )
        ]
        if max_items > 0:
            ready = ready[:max_items]
        for entry in ready:
            _pending_lease_releases.remove(entry)
    for entry in ready:
        if deadline and time.monotonic() >= deadline:
            with _sessions_lock:
                _pending_lease_releases.append(entry)
            continue
        owner = entry.get("owner") or ""
        with _held_owner_lock(owner):
            if entry["kind"] == "release" and owner and _profile_still_in_use(owner):
                logger.debug("Camofox dropped a scheduled release: the profile is in use again")
                continue
            if _attempt_teardown(entry):
                continue
        attempts = entry["attempts"] + 1
        if attempts >= _MAX_PENDING_RELEASE_ATTEMPTS:
            logger.warning(
                "Camofox gave up on a %s for the local browser; the server side "
                "reclaims it when its own timeout expires",
                entry["kind"],
            )
            continue
        entry["attempts"] = attempts
        entry["ready_at"] = time.monotonic() + _MAINTENANCE_TICK_SECONDS
        with _sessions_lock:
            if len(_pending_lease_releases) < _MAX_PENDING_LEASE_RELEASES:
                _pending_lease_releases.append(entry)


def _attempt_teardown(entry: Dict[str, Any]) -> bool:
    """Run one teardown request. True when nothing is left to chase."""
    url, headers = entry["url"], entry["headers"]
    try:
        if entry["kind"] == "delete":
            resp = requests.delete(url, timeout=5, headers=headers, allow_redirects=False,
                                   proxies=_NO_ENV_PROXIES)
        else:
            resp = requests.post(url, json={}, timeout=5, headers=headers, allow_redirects=False,
                                 proxies=_NO_ENV_PROXIES)
        _raise_for_status(resp)
        return True
    except Exception as exc:
        if _is_retryable_read_error(exc):
            return False
        # A definitive answer (already gone, unknown id) ends the chase.
        logger.debug("Camofox %s rejected: %s", entry["kind"], exc)
        return True


def _queue_pending_teardown(
    kind: str,
    url: str,
    headers: Dict[str, str],
    owner: str = "",
    delay: float = 0.0,
) -> None:
    with _sessions_lock:
        # Merge by target: repeat teardowns of the same thing are the same
        # work, so the queue is bounded by distinct profiles rather than by
        # events. Dropping the oldest entry to stay under a cap would throw
        # away the only handle that can close a runtime.
        for existing in _pending_lease_releases:
            # Owner is part of the identity: in multiplex every profile posts
            # to the same /_zettlab/release and differs only by credential, so
            # merging on the URL alone would overwrite another profile's only
            # release credential.
            if existing["kind"] == kind and existing["url"] == url and existing.get("owner", "") == owner:
                existing["headers"] = dict(headers)
                existing["ready_at"] = min(existing["ready_at"], time.monotonic() + delay)
                break
        else:
            if len(_pending_lease_releases) >= _MAX_PENDING_LEASE_RELEASES:
                # Nothing may be discarded silently; make the overflow visible
                # and keep the newest, which is the one still reachable.
                dropped = _pending_lease_releases.pop(0)
                logger.error(
                    "Camofox teardown queue is full (%d distinct targets); dropping a queued %s. "
                    "The server side reclaims it when its own timeout expires.",
                    _MAX_PENDING_LEASE_RELEASES, dropped["kind"],
                )
            _pending_lease_releases.append({
                "kind": kind,
                "url": url,
                "headers": dict(headers),
                "owner": owner,
                "attempts": 0,
                "ready_at": time.monotonic() + delay,
            })
    _ensure_maintenance_worker()


def _queue_pending_lease_release(url: str, headers: Dict[str, str], owner: str = "") -> None:
    _queue_pending_teardown("release", url, headers, owner)


def _ensure_maintenance_worker() -> None:
    """Start the reclaim thread unless one is already running.

    The worker clears the global under the same lock in which it decides there
    is nothing left to do, so an enqueue racing its exit either lands before
    that check (the worker keeps going) or after it (this call sees None and
    starts a replacement). Neither ordering can drop the work.
    """
    global _maintenance_worker
    with _sessions_lock:
        if _maintenance_worker is not None:
            return
        worker = threading.Thread(
            target=_run_maintenance,
            name="camofox-maintenance",
            daemon=True,
        )
        _maintenance_worker = worker
    _ensure_shutdown_hook()
    worker.start()


_shutdown_hook_registered = False

# What a normal exit may spend releasing browser state. A daemon worker is
# killed outright at interpreter exit, so without this every restart leaves the
# profile's browser child or the direct Camofox session running until the
# server side times out — on a 2 GB device that is a resident process nobody
# asked for.
_SHUTDOWN_DRAIN_BUDGET_SECONDS = 10.0


def _ensure_shutdown_hook() -> None:
    global _shutdown_hook_registered
    with _sessions_lock:
        if _shutdown_hook_registered:
            return
        _shutdown_hook_registered = True
    atexit.register(shutdown_camofox_sessions)


def shutdown_camofox_sessions() -> None:
    """Release every tracked Camofox session, within a bounded budget.

    Called from atexit, and safe to call directly from a gateway's own
    shutdown. Sessions are taken atomically so a concurrent caller cannot
    resurrect one halfway through, and the drain is forced rather than waiting
    out the quiet window nobody will be here for.
    """
    deadline = time.monotonic() + _SHUTDOWN_DRAIN_BUDGET_SECONDS
    with _sessions_lock:
        taken = list(_sessions.values())
        _sessions.clear()
    for session in taken:
        if time.monotonic() >= deadline:
            logger.warning("Camofox shutdown budget spent; %d session(s) left to the server side", len(taken))
            break
        try:
            _teardown_session(session)
        except Exception as exc:
            logger.debug("Camofox shutdown teardown failed: %s", exc)
    try:
        _run_pending_teardowns(force=True, deadline=deadline)
    except Exception as exc:
        logger.debug("Camofox shutdown drain failed: %s", exc)


def _run_maintenance() -> None:
    global _maintenance_worker
    while True:
        time.sleep(_MAINTENANCE_TICK_SECONDS)
        with _sessions_lock:
            idle = _prune_idle_sessions_locked(time.monotonic())
        for session in idle:
            _teardown_session(session)
        _run_pending_teardowns()
        with _sessions_lock:
            if not _sessions and not _pending_lease_releases:
                _maintenance_worker = None
                return


def _release_owner_key(user_id: str, headers: Dict[str, str], base_url: str = "") -> str:
    """Stable per-profile identity for lease ownership.

    Three things make it: the profile, the credential and the endpoint. The
    credential is hashed rather than stored a second time; the digest only has
    to distinguish profiles inside this process.

    The endpoint belongs in it because a tab id, an epoch and a ref generation
    mean nothing outside the runtime that issued them. A profile whose
    CAMOFOX_URL is repointed while its entry is alive would otherwise keep
    hitting the cached tab state against a different runtime — 404s at best,
    the wrong page if the new one happens to reuse the id. Two multiplex
    profiles pointed at different endpoints with the same credential collide
    the same way.

    This key feeds the session cache key, the tab identity lock and the
    per-tab document record, so naming the endpoint here separates all of them
    at once.
    """
    import hashlib

    credential = ""
    for key in ("X-Zettlab-Agent-Action-Token", "Authorization"):
        value = headers.get(key)
        if value:
            credential = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            break
    endpoint = _normalized_endpoint(base_url)
    return f"{user_id}\x00{credential}\x00{endpoint}"


def _normalized_endpoint(base_url: str) -> str:
    """Compare endpoints by what they address, not by how they were spelled."""
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw.rstrip("/").lower()
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    path = (parsed.path or "").rstrip("/")
    return f"{scheme}://{host}:{port}{path}"


_MAX_OWNER_LOCKS = 256
_owner_locks: Dict[str, threading.RLock] = {}
# Owners with a lock currently reserved or held; never evicted.
_owner_lock_refs: Dict[str, int] = {}


@contextmanager
def _held_owner_lock(owner: str):
    """Acquire the owner lock and keep it un-evictable for the duration.

    Returning a bare lock left a window: between the lookup and the caller's
    acquire, an eviction could remove it as unheld and unreferenced, and the
    next registration would build a second lock for the same owner — two
    critical sections where there must be one.
    """
    lock = _owner_lock(owner, reserve=True)
    try:
        with lock:
            yield lock
    finally:
        with _sessions_lock:
            count = _owner_lock_refs.get(owner, 0) - 1
            if count > 0:
                _owner_lock_refs[owner] = count
            else:
                _owner_lock_refs.pop(owner, None)


def _owner_lock(owner: str, reserve: bool = False) -> threading.RLock:
    """Serialize session registration against lease release for one profile.

    Without it the "am I the last holder" answer can go stale between the check
    and the request: another turn registers and starts the runtime, and this
    release tears it down under them. Always acquired before ``_sessions_lock``.
    """
    with _sessions_lock:
        if reserve:
            _owner_lock_refs[owner] = _owner_lock_refs.get(owner, 0) + 1
        lock = _owner_locks.get(owner)
        if lock is not None:
            return lock
        if len(_owner_locks) >= _MAX_OWNER_LOCKS:
            # Evict only locks nobody holds and no session refers to. Clearing
            # the map wholesale could hand the same owner a second lock while
            # the first is still held, which is exactly the serialization this
            # exists to provide.
            live = {session.get("release_owner") for session in _sessions.values()}
            live.update(_owner_lock_refs)
            for key in [k for k in _owner_locks if k not in live]:
                candidate = _owner_locks[key]
                if candidate.acquire(blocking=False):
                    candidate.release()
                    del _owner_locks[key]
        lock = threading.RLock()
        _owner_locks[owner] = lock
        return lock


def _profile_still_in_use(release_owner: str, exclude_key: str = "") -> bool:
    """Whether another tracked session still holds this profile's runtime.

    ``/_zettlab/release`` carries only the profile action token — no task or
    lease id — so it tears down the runtime for the whole profile. Two turns on
    the same profile can each own a session, and releasing when the first one
    ends would pull the runtime out from under the other, or out from under a
    human mid-takeover.
    """
    if not release_owner:
        return False
    with _sessions_lock:
        for key, session in _sessions.items():
            if key == exclude_key:
                continue
            if session.get("release_owner") == release_owner:
                return True
    return False


def _teardown_session(session: Optional[Dict[str, Any]]) -> None:
    """Release or close a dropped session according to how it was created.

    A local-server-managed session holds a shared runtime lease; a direct
    Camofox session owns a throwaway server-side session instead, and the only
    thing that frees it is DELETE /sessions/<user_id>.
    """
    if not isinstance(session, dict):
        return
    # The record may only go when the tab it names is really gone. A direct
    # session owns its server-side session outright, so DELETE takes the tab
    # with it. A managed one only gives back a shared runtime lease — and may
    # not even do that, if another turn is still holding the profile — so its
    # tab outlives this call. Dropping a blocked record there would let the
    # next adoption of the same listItemId read the page this one was refused
    # for: an ordinary redirect leaves the privacy filter off, so nothing else
    # would stop it.
    if session.get("local_server_managed"):
        owner = str(session.get("release_owner") or "")
        # The check and the release are one critical section: otherwise another
        # turn can register and start the runtime in between, and this release
        # tears it down under them.
        with _held_owner_lock(owner):
            if _profile_still_in_use(owner):
                logger.debug("Camofox lease kept: another session still holds this profile")
                return
            _release_local_server_lease(session)
        return
    if session.get("managed"):
        # Managed persistence without local-server: the profile must survive.
        return
    user_id = str(session.get("user_id") or "")
    base = str(session.get("delete_base") or "")
    if not user_id or not base:
        return
    headers = session.get("delete_headers")
    headers = dict(headers) if isinstance(headers, dict) else {}
    # Queued like the lease release: this also runs from the scope-less
    # maintenance thread, and a transient failure must not lose the only handle
    # that can close the server-side session.
    _queue_pending_teardown(
        "delete",
        f"{base}/sessions/{quote(user_id, safe='')}",
        headers,
        owner=str(session.get("release_owner") or ""),
    )


def _release_local_server_lease(session: Optional[Dict[str, Any]] = None) -> None:
    """Schedule the release of this profile's long-lived Agent runtime lease.

    Scheduled, not immediate: the endpoint frees the runtime for the whole
    profile, and how many turns are still using it is not knowable here. The
    maintenance thread runs it after a quiet window and re-checks holders under
    the owner lock, so a turn that is still working — or a human mid-takeover —
    keeps its browser.

    The endpoint and credential come from the session because teardown reaches
    this from the idle reaper, ``/new`` and shutdown, where re-reading them
    raises ``UnscopedSecretError``.
    """
    url = ""
    headers: Dict[str, str] = {}
    owner = ""
    if isinstance(session, dict):
        url = str(session.get("release_url") or "")
        owner = str(session.get("release_owner") or "")
        captured = session.get("release_headers")
        if isinstance(captured, dict):
            headers = dict(captured)
    if not url:
        try:
            url = f"{get_camofox_url()}/_zettlab/release"
            headers = _auth_headers()
        except Exception as exc:
            logger.debug("Camofox local-server lease release unresolvable: %s", exc)
            return
    _queue_pending_teardown("release", url, headers, owner=owner, delay=_RELEASE_GRACE_SECONDS)




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
    session = _peek_session(task_id)
    if session is not None:
        # Decide from the session's own creation-time state. Cleanup runs from
        # the idle reaper, ``/new`` and shutdown, all outside the request's
        # profile home and secret scope, where recomputing managed-ness fails
        # closed and would strand the entry plus its local-server lease.
        if not session.get("managed"):
            return False
        # The tab survives this cleanup (that is the point of the soft path),
        # so carry its epoch forward: the next turn re-adopts it and must be
        # able to tell "same page, no handback" from a genuine restart.
        _remember_tab_epoch(session)
        if session.get("local_server_managed"):
            # Deliberately no release here, and the entry stays. How long a
            # turn will keep using the browser is not observable from this
            # process — a vision call runs for minutes, a human takeover for
            # longer — so any timer started at turn end is a guess. The entry's
            # last-use stamp is the one real signal, and the idle sweep acts on
            # that. Precise, immediate release needs a lease id the
            # local-server can refcount; tracked as follow-up.
            logger.debug("Camofox soft cleanup for task %s (lease left to the idle sweep)", task_id)
            return True
        _drop_session(task_id)
        logger.debug("Camofox soft cleanup for task %s (managed persistence)", task_id)
        return True

    # Nothing is tracked for this task, so it never took a browser lease and
    # must not release one: cleanup_task_resources() runs at the end of every
    # turn, and releasing here would tear down the runtime out from under a
    # concurrent turn on the same profile that is actually using it.
    try:
        camofox_cfg = _get_camofox_config()
        return bool(
            _local_server_managed()
            or camofox_cfg.get("managed_persistence")
            or _camofox_identity_override(task_id, camofox_cfg)
        )
    except Exception:
        # Scope-less cleanup path; with no tracked session there is nothing to
        # tear down either way.
        return True


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
    if "retryable" not in payload:
        # A proxy-generated 502/503 carries no JSON envelope, but a cold start
        # or a brief overload is not a terminal answer. Mutations are never
        # replayed automatically; this only tells the Agent it may try again.
        status = resp.status_code
        if isinstance(status, int) and (status in (408, 429) or 500 <= status < 600):
            payload["retryable"] = True
    return payload


def _raise_for_status(resp: requests.Response) -> None:
    # Only 2xx is success. The internal local-server proxy is a fixed loopback
    # endpoint that never legitimately redirects, so a 3xx means misconfig or
    # compromise; reject it rather than treating it as OK (requests is also
    # called with allow_redirects=False so the action token is never resent to
    # a redirect target).
    if 200 <= resp.status_code < 300:
        return
    raise CamofoxHTTPError(resp, _safe_http_error_payload(resp))


def _request_headers(session: Optional[Dict[str, Any]]) -> Dict[str, str]:
    return {**_auth_headers(), **_session_epoch_header(session)}


# Facts about the most recent response on this thread. A turn runs its browser
# calls synchronously, so "the last response on this thread" is precisely "this
# call's response" — unlike anything stored on the shared session dict.
_response_facts = threading.local()


def _last_response_epoch_verified() -> bool:
    return bool(getattr(_response_facts, "epoch_verified", False))


def _last_response_started_handback() -> bool:
    """Whether the response just received is the one that revealed a handback.

    The shared flag it sets can be cleared again by a concurrent turn before
    this response's caller gets to look at it, so the caller has to remember
    what its own response reported.
    """
    return bool(getattr(_response_facts, "started_handback", False))


def _is_tab_operation(path: str) -> bool:
    """Whether this path targets one specific tab, i.e. carries an epoch."""
    return path.startswith("/tabs/")


# Bounded retry for reads only. A cold camofox start or a loaded device makes
# the proxy briefly answer 502/503 or time out; without this a single blip ends
# the tool call. Mutations (click/type/navigate) are never replayed — their side
# effects are not idempotent — so they surface the retryable degradation to the
# Agent instead.
_READ_RETRY_ATTEMPTS = 3
_READ_RETRY_BACKOFF_SECONDS = 0.25
_RETRYABLE_READ_STATUSES = frozenset({502, 503, 504})


def _is_retryable_read_error(exc: BaseException) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    resp = getattr(exc, "response", None)
    if resp is None:
        return False
    status = getattr(resp, "status_code", None)
    if not isinstance(status, int):
        return False
    # Every 5xx is the server failing to answer, not an answer. Treating an
    # internal error as definitive would drop a teardown whose runtime is still
    # alive. 408/429 are explicit "come back" replies.
    return status in _RETRYABLE_READ_STATUSES or status in (408, 429) or 500 <= status < 600


def _retry_read(operation):
    """Run an idempotent read with bounded exponential backoff."""
    import time

    last_exc: Optional[BaseException] = None
    for attempt in range(_READ_RETRY_ATTEMPTS):
        try:
            return operation()
        except Exception as exc:
            if not _is_retryable_read_error(exc) or attempt == _READ_RETRY_ATTEMPTS - 1:
                raise
            last_exc = exc
            time.sleep(_READ_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    raise last_exc  # pragma: no cover - loop always returns or raises above


def _post(path: str, body: dict, timeout: Optional[int] = None, session: Optional[Dict[str, Any]] = None) -> dict:
    """POST JSON to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    _response_facts.started_handback = False
    _begin_session_call(session)
    try:
        resp = requests.post(url, json=body, timeout=timeout, headers=_request_headers(session),
                             allow_redirects=False, proxies=_NO_ENV_PROXIES)
    finally:
        _end_session_call(session)
    # POST /tabs establishes the baseline epoch for the new tab. Without it a
    # later response's epoch looks like the first one ever seen and is taken as
    # a safe baseline, so a takeover between creation and the first read would
    # go unnoticed.
    # Response-local, not session state: concurrent turns share the session
    # dict, so another response landing in between could flip a shared flag
    # before this caller reads it. A thread-local is exactly the scope of one
    # synchronous request/response pair.
    _response_facts.epoch_verified = _adopt_epoch_from_response(
        session, resp, tab_operation=_is_tab_operation(path) or path == "/tabs"
    )
    _raise_for_status(resp)
    return resp.json()


def _get(path: str, params: dict = None, timeout: Optional[int] = None, session: Optional[Dict[str, Any]] = None) -> dict:
    """GET from camofox and return parsed response."""
    return _get_raw(path, params=params, timeout=timeout, session=session).json()


def _get_raw(path: str, params: dict = None, timeout: Optional[int] = None, session: Optional[Dict[str, Any]] = None) -> requests.Response:
    """GET from camofox and return raw response (for binary data)."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"

    # Reset once for the whole call, not per attempt: an earlier attempt can
    # carry the epoch that reveals a handback and still fail retryably, and
    # that fact has to reach the caller.
    _response_facts.started_handback = False

    def _once() -> requests.Response:
        _begin_session_call(session)
        try:
            resp = requests.get(url, params=params, timeout=timeout, headers=_request_headers(session),
                                allow_redirects=False, proxies=_NO_ENV_PROXIES)
        finally:
            _end_session_call(session)
        _adopt_epoch_from_response(session, resp, tab_operation=_is_tab_operation(path))
        _raise_for_status(resp)
        return resp

    return _retry_read(_once)


def _delete(path: str, body: dict = None, timeout: Optional[int] = None, session: Optional[Dict[str, Any]] = None) -> dict:
    """DELETE to camofox and return parsed response."""
    if timeout is None:
        timeout = _get_command_timeout()
    url = f"{get_camofox_url()}{path}"
    resp = requests.delete(url, json=body, timeout=timeout, headers=_request_headers(session),
                           allow_redirects=False, proxies=_NO_ENV_PROXIES)
    _adopt_epoch_from_response(session, resp)
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
    if (status, code) != (409, "browser_epoch_stale"):
        return None
    return payload


def _current_tab_url(session: Dict[str, Any]) -> str:
    """The proxy's view of where this tab currently is, or "" when unknown.

    Read from ``/tabs`` rather than the page itself: the local-server proxy
    owns that field for controlled tabs, so it is not something the page or a
    compromised runtime can dictate.
    """
    tab_id = session.get("tab_id")
    if not tab_id:
        return ""
    try:
        listed = _get("/tabs", params={"userId": session["user_id"]}, timeout=5, session=session)
    except Exception as exc:
        logger.debug("Camofox tab url lookup failed: %s", exc)
        return ""
    tabs = listed.get("tabs") if isinstance(listed, dict) else None
    if not isinstance(tabs, list):
        return ""
    for candidate in tabs:
        if isinstance(candidate, dict) and candidate.get("tabId") == tab_id:
            url = candidate.get("url")
            return url if isinstance(url, str) else ""
    return ""


def _document_identity(url: Any) -> str:
    """Everything but the fragment: what identifies the loaded document."""
    if not isinstance(url, str) or not url:
        return ""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return ""
    return urlunsplit(SplitResult(parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _left_handback_document(session: Dict[str, Any], document_before: str, landed_url: Any) -> bool:
    """Whether a navigation actually left the page the human handed back.

    A fragment-only navigation keeps the same document — the DOM and every
    value the human typed into it survive — and does not advance the handback
    epoch, so "the epoch stood still" is not enough to clear the filter. When
    the document identity cannot be established, keep filtering.
    """
    if not _handback_privacy_filter_enabled(session):
        return True
    if not document_before:
        return False
    landed = _document_identity(landed_url)
    if not landed:
        return False
    return landed != document_before


def _handback_page_readable(session: Dict[str, Any], snapshot_data: Any = None) -> bool:
    """Whether a post-handback page may be read at all.

    Redaction is not enough on its own: the human may have left the tab on
    cloud metadata or an intranet host, which the Agent is not allowed to read
    in any form. Only consulted while the handback filter is on, so the extra
    lookup costs nothing on the normal path. Fails closed when the URL cannot
    be established.

    The capture's own URL decides when it carries one: that is the page the
    content actually came from. Camofox does not always send it, and a /tabs
    lookup only describes where the tab is *now* — which a concurrent navigate
    could make a different page. Callers therefore run the capture and this
    check inside :func:`_capture_guard`, which holds the tab identity for the
    whole read so nothing in this process can move the tab in between.
    """
    if isinstance(snapshot_data, dict):
        captured = snapshot_data.get("url")
        if isinstance(captured, str) and captured and not _recovery_target_allowed(captured):
            return False
    landed_url = _current_tab_url(session)
    if not landed_url:
        return False
    return _recovery_target_allowed(landed_url)


def _blocked_handback_page_error() -> str:
    return tool_error(
        "The human left the browser on a page this Agent is not allowed to read "
        "(cloud metadata or a private-network address). Navigate to an allowed "
        "page before continuing.",
        success=False,
    )


def _recovery_target_allowed(url: str) -> bool:
    """Whether the Agent may read the page the human handed back.

    Two separate policies have to hold. The SSRF guards keep the Agent off
    cloud metadata and private-network addresses, and the configured website
    policy (``security.website_blocklist``) says which sites this deployment
    refuses to hand to a model at all — browser_navigate() enforces the latter,
    so a page reached by human takeover instead of by navigation must not slip
    past it.

    Fails closed: if the guards cannot be imported or a check raises, the page
    is treated as off limits.
    """
    try:
        from tools.browser_tool import _is_always_blocked_url, _is_safe_url

        if _is_always_blocked_url(url) or not _is_safe_url(url):
            return False
    except Exception:
        return False
    try:
        from tools.website_policy import check_website_access

        return check_website_access(url) is None
    except ImportError:
        # Same fail-open as browser_tool's own import guard: a deployment
        # without the policy module has no blocklist to enforce.
        return True
    except Exception:
        return False


def _retryable_control_result(
    exc: BaseException,
    session: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Translate the epoch-staleness conflict into a retryable tool result.

    A human controlled this page since the Agent last looked, so its element
    refs are stale. Recovery is stateless and idempotent: adopt the current
    epoch from the conflict payload, take one privacy-filtered snapshot (the
    proxy always admits snapshots), and ask the model to retry with fresh
    refs. There is no handshake to complete, so nothing here can strand the
    session; a failed snapshot simply leaves the result retryable.
    """
    payload = _control_error_payload(exc)
    if payload is None:
        return None

    result: Dict[str, Any] = {
        "success": False,
        "error": "browser_epoch_stale",
        "retryable": True,
        "resnapshot_completed": False,
        "message": (
            "Page state changed while a human controlled the browser. "
            "A fresh privacy-filtered snapshot is included; retry using its refs."
        ),
    }
    takeover_session_id = payload.get("takeover_session_id")
    if isinstance(takeover_session_id, str) and takeover_session_id:
        result["takeover_session_id"] = takeover_session_id[:256]

    if not session or not session.get("tab_id") or not session.get("user_id"):
        return json.dumps(result)

    # Human-entered values can remain in the page state. Filter every later
    # read until the Agent explicitly leaves this page. The refusal also means
    # a human held this tab, so whatever refs the Agent still has describe the
    # page from before that — invalidate them here rather than relying on an
    # epoch header this error envelope may not carry.
    _set_handback_privacy_filter(session, True)
    # Deliberately not adopting any epoch the refusal itself carries. Copying it
    # into the next request would clear the barrier without the snapshot that
    # carries the human's page state — which is the whole point of the barrier.
    # The recovery snapshot below reports the epoch through the response header
    # like every other read, and if it fails this session keeps its old epoch
    # and is refused again.

    # Where the human left the page is not where the Agent may follow. The
    # deleted resume handshake used to validate this before acking; without an
    # equivalent here, a handback on 169.254.169.254 or an intranet page would
    # hand its contents to the model through the recovery snapshot, straight
    # past the guard browser_navigate applies to the Agent's own navigations.
    if not _handback_page_readable(session):
        result["message"] = (
            "The human left the browser on a page this Agent is not allowed to read "
            "(cloud metadata or a private-network address). Page state was not captured. "
            "Navigate to an allowed page before continuing."
        )
        result["blocked_page"] = True
        return json.dumps(result)

    try:
        # Sampled before the capture and registered after, exactly like every
        # other snapshot: the refs this recovery hands back are the ones the
        # message tells the Agent to retry with, so they have to be stamped or
        # the next call refuses them — and they must not be stamped onto a
        # generation the capture does not describe.
        with _capture_guard(session):
            snapshot_data = _get(
                _tab_path(session, "/snapshot"),
                params={"userId": session["user_id"]},
                session=session,
            )
            # Re-check after the response: the human can take over again
            # between the pre-check and this reply, and a snapshot is always
            # admitted, so nothing else would stop the new page from coming
            # back.
            if not _handback_page_readable(session, snapshot_data):
                result["message"] = (
                    "The human left the browser on a page this Agent is not allowed to read "
                    "(cloud metadata or a private-network address). Page state was not captured. "
                    "Navigate to an allowed page before continuing."
                )
                result["blocked_page"] = True
                return json.dumps(result)

        snapshot = snapshot_data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""

        from tools.browser_tool import (
            SNAPSHOT_SUMMARIZE_THRESHOLD,
            _truncate_snapshot,
        )

        if len(snapshot) > SNAPSHOT_SUMMARIZE_THRESHOLD:
            snapshot = _truncate_snapshot(snapshot)
        result["snapshot"] = _redact_handback_page_state(snapshot)
        result["element_count"] = snapshot_data.get("refsCount", 0)
        result["resnapshot_completed"] = True
    except Exception as snapshot_exc:
        logger.warning("Camofox post-handback snapshot failed: %s", snapshot_exc)
    return json.dumps(result)


def _epoch_moved_result(session: Optional[Dict[str, Any]]) -> str:
    payload = {
        "success": False,
        "error": "browser_epoch_stale",
        "message": (
            "Page state changed while this action waited for the browser tab. "
            "Take a fresh snapshot and retry using its refs."
        ),
        "retryable": True,
    }
    if isinstance(session, dict) and session.get("epoch") is not None:
        payload["epoch"] = session["epoch"]
    return json.dumps(payload, ensure_ascii=False)


def _tool_error_from_exception(
    exc: BaseException,
    *,
    session: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    if isinstance(exc, CamofoxEpochMoved):
        return _epoch_moved_result(session)
    if isinstance(exc, CamofoxSessionsBusy):
        # Backpressure, not a failure of this request: the cache is full of work
        # in flight, and one of those calls finishing frees a slot.
        payload: Dict[str, Any] = {
            "success": False,
            "error": "browser_sessions_busy",
            "message": (
                "Every tracked browser session is in use. Retry shortly, or close a "
                "browser session that is no longer needed."
            ),
            "retryable": True,
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(exc, CamofoxEpochUnavailable):
        payload = {
            "success": False,
            "error": "browser_epoch_unavailable",
            "message": (
                "This browser session has no page-state baseline yet, so writes are "
                "refused. Take a snapshot first; if the device's local-server does not "
                "report page state, the browser is read-only."
            ),
            "retryable": True,
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(exc, CamofoxEvaluateBlocked):
        payload = {
            "success": False,
            "error": (
                "Browser evaluation is blocked after human control until the "
                "Agent navigates to a new page or closes the session."
            ),
        }
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)
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
    # Timeout is a sibling of ConnectionError here, not a hard failure: a cold
    # camofox start or a loaded device makes the proxy slow, and reporting that
    # as unrecoverable turns a transient hiccup into a dead tool call.
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
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

class CamofoxEpochMoved(Exception):
    """Raised when the page moved on while a mutation waited for the tab."""




class CamofoxSessionsBusy(Exception):
    """Raised when every tracked browser session is in use."""


class CamofoxEpochUnavailable(Exception):
    """Raised when a managed session has no epoch baseline to mutate against."""


class CamofoxEvaluateBlocked(Exception):
    """Raised when arbitrary JavaScript is refused because a human held the tab."""




# Document generation per physical tab. The epoch contract only advances on a
# human handback, so it cannot see an ordinary navigate — and turns that share
# a HERMES_SESSION_KEY share the tab while keeping their own session entry. A
# parent's ``e1`` would otherwise still validate after its subagent navigated,
# and land on whatever element reuses that ref on the new page.


def _mutating_tab_call(session: Dict[str, Any], path_suffix: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Run an operation that changes the document, holding the identity lock.

    Turns that share a browser identity share the physical tab, so a click or a
    back landing between another turn's navigate and the snapshot describing
    where it landed would make that turn report one page and hand back another
    page's refs. Reads are deliberately not serialized here: a snapshot of
    whatever the tab currently shows is inherent to sharing it, and queueing
    long reads such as vision behind every mutation would cost more than it
    buys.
    """
    # The refs in this call describe the page as it was when the Agent last
    # read it. Another turn can advance the epoch while this one waits for the
    # tab, and the request header is built from the shared session — so without
    # this check the proxy would see a current epoch attached to stale refs and
    # accept them, clicking or typing on a page the human just handed back.
    # A managed session with no epoch yet has no handback protocol behind it:
    # the request goes out with no X-Zettlab-Browser-Epoch, so nothing on either
    # side would stop a stale ref — or arbitrary JavaScript — from acting on a
    # page a human is in the middle of. Filtering the response cannot undo a
    # write that already happened, so the mutation is refused instead. A read
    # (snapshot) establishes the baseline, and against a local-server too old to
    # send the header the browser tool degrades to reads only.
    if session.get("local_server_managed") and session.get("epoch") is None:
        raise CamofoxEpochUnavailable(
            "this browser session has no page-state baseline yet; take a snapshot first"
        )
    observed_epoch = session.get("epoch")
    with _held_owner_lock(_browser_identity_key(session)):
        if session.get("epoch") != observed_epoch:
            raise CamofoxEpochMoved(
                "the page changed while this operation waited for the browser tab"
            )
        # Re-checked here, not only by the caller: evaluate runs arbitrary
        # JavaScript, so between a caller's check and this lock another turn's
        # response can turn the filter on and this call would then run against
        # the page a human is holding. Discarding the result afterwards cannot
        # undo a location.href, a DOM write or a request the script made.
        if path_suffix == "/evaluate" and _handback_privacy_filter_enabled(session):
            raise CamofoxEvaluateBlocked(
                "browser evaluation is blocked after human control until the Agent "
                "navigates to a new page or closes the session"
            )
        return _post(_tab_path(session, path_suffix), body, session=session)


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
        with _session_operation(session):
            return _navigate_locked(session, task_id, url, browser_url, rewrite_info)
    except requests.HTTPError as e:
        return _tool_error_from_exception(
            e, session=session, prefix="Navigation failed: ", extra=_navigation_tab_context(session),
        )
    except Exception as e:
        return _tool_error_from_exception(
            e, session=session, prefix="Navigation failed: ", extra=_navigation_tab_context(session),
        )


def _navigate_locked(
    session: Dict[str, Any],
    task_id: Optional[str],
    url: str,
    browser_url: str,
    rewrite_info: Any,
) -> str:
    try:
        # Turns that share a browser identity share the physical tab, so the
        # navigate and the snapshot that describes where it landed have to be
        # one unit: otherwise this call returns its own url and title with the
        # other turn's page state, and the refs point at the wrong document.
        # The lock is reentrant, so _ensure_tab above having taken it is fine.
        with _held_owner_lock(_browser_identity_key(session)):
            return _navigate_within_identity(session, task_id, url, browser_url, rewrite_info)
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
            prefix="Navigation failed: ",
            extra=_navigation_tab_context(session),
        )


def _navigate_within_identity(
    session: Dict[str, Any],
    task_id: Optional[str],
    url: str,
    browser_url: str,
    rewrite_info: Any,
) -> str:
    rebound_session: Optional[Dict[str, Any]] = None
    try:
        # An epoch advance during this call means a human took over and handed
        # back while the navigation was in flight, so the page the Agent is
        # about to read is not the one it asked for. Clearing the filter
        # unconditionally below would let those values through, and the fresh
        # epoch means the server will not flag the next read as stale either.
        epoch_before_navigate = session.get("epoch")
        # Declared before the request goes out, not after it returns: from this
        # moment the document every outstanding ref describes is on its way out,
        # and a concurrent turn's click must fail rather than race the landing.
        navigate_filtered_at_request = _handback_privacy_filter_enabled(session)
        # Only meaningful while the filter is on, and it costs a round trip, so
        # it is not taken on the normal path.
        document_before_navigate = (
            _document_identity(_current_tab_url(session))
            if _handback_privacy_filter_enabled(session)
            else ""
        )
        try:
            data = _post(
                _tab_path(session, "/navigate"),
                {"userId": session["user_id"], "url": browser_url},
                timeout=60,
                session=session,
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
                # _ensure_tab returns an entry that already holds its own
                # reference, and the caller's _session_operation only owns the
                # one this call started with — so this replacement has to be
                # released here.
                rebound_session = session
                data = _post(
                    _tab_path(session, "/navigate"),
                    {"userId": session["user_id"], "url": browser_url},
                    timeout=60,
                    session=session,
                )
            else:
                raise

        # Three things must hold before the page counts as left behind: this
        # response actually carried a verified epoch (a protocol downgrade must
        # not read as "nothing happened"), the epoch did not move, and the
        # document identity really changed.
        # The epoch requirement only applies where the protocol exists: a
        # direct Camofox session has no epoch to verify.
        epoch_protocol_ok = _last_response_epoch_verified() or not session.get("local_server_managed")
        if (
            epoch_protocol_ok
            and session.get("epoch") == epoch_before_navigate
            and _left_handback_document(session, document_before_navigate, data.get("url", browser_url))
        ):
            _set_handback_privacy_filter(session, False)
        # A handback that landed mid-navigation leaves the filter on, and the
        # page the human ended on is reported here: an OAuth callback, a reset
        # link or any URL with personal query parameters would otherwise reach
        # the model in full, with the title alongside it.
        landed_url = data.get("url", browser_url)
        landed_title = data.get("title", "")
        # Same three signals the reads use: the state when the request went
        # out, whether this response is the one that revealed the handback, and
        # the state now. A concurrent navigate can clear the shared flag before
        # this line runs.
        landed_handback_revealed = navigate_filtered_at_request or _last_response_started_handback()
        if landed_handback_revealed or _handback_privacy_filter_enabled(session):
            landed_url = _filter_url_after_handback(session, landed_url, True)
            landed_title = "[REDACTED]" if landed_title else landed_title
        result = {
            "success": True,
            "url": landed_url,
            "title": landed_title,
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

        # Auto-take a compact snapshot so the model can act immediately, unless
        # the tab is still on a page a refused navigation left it on: this
        # response reported no landing URL, so nothing has proven it moved.
        try:
            snapshot_filtered_at_request = _handback_privacy_filter_enabled(session)
            # No _capture_guard here: _navigate_locked already holds this tab's
            # identity for the whole navigate, so this capture and the
            # readability check below are inside the same critical section the
            # guard would take. The lock is reentrant, so adding one would be
            # harmless — just redundant.
            snap_data = _get(
                _tab_path(session, "/snapshot"),
                params={"userId": session["user_id"]},
                session=session,
            )
            snapshot_text = snap_data.get("snapshot", "")
            from tools.browser_tool import (
                SNAPSHOT_SUMMARIZE_THRESHOLD,
                _truncate_snapshot,
            )
            if len(snapshot_text) > SNAPSHOT_SUMMARIZE_THRESHOLD:
                snapshot_text = _truncate_snapshot(snapshot_text)
            # Same rule as camofox_snapshot(): the epoch that enables the
            # filter arrives with this very response, so a human who took over
            # and handed back between the navigate and the snapshot must not
            # have what they typed land in the tool result — and if they left
            # the tab somewhere the Agent may not read at all, redaction is not
            # enough, the snapshot is dropped.
            snapshot_handback_revealed = snapshot_filtered_at_request or _last_response_started_handback()
            if (snapshot_handback_revealed or _handback_privacy_filter_enabled(session)) and not _handback_page_readable(session, snap_data):
                result["snapshot_withheld"] = True
                result["warning"] = (
                    "A human took over and left the browser on a page this Agent is not "
                    "allowed to read. Page state was not captured."
                )
            else:
                result["snapshot"] = _filter_page_state_after_handback(session, snapshot_text, snapshot_handback_revealed)
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
    finally:
        if rebound_session is not None:
            _end_session_call(rebound_session)


def camofox_snapshot(full: bool = False, task_id: Optional[str] = None,
                     user_task: Optional[str] = None) -> str:
    """Get accessibility tree snapshot from Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        filtered_at_request = _handback_privacy_filter_enabled(session)
        # Held across the capture and the readability check when the filter is
        # already on, so the two describe the same page.
        with _capture_guard(session):
            data = _get(
                _tab_path(session, "/snapshot"),
                params={"userId": session["user_id"]},
                session=session,
            )
            handback_revealed = filtered_at_request or _last_response_started_handback()
            if (handback_revealed or _handback_privacy_filter_enabled(session)) and not _handback_page_readable(session, data):
                return _blocked_handback_page_error()

        # The response is what advances the epoch, so the filter can only be
        # known to be on at this point — the guard has to run here, not before
        # the request. The request-time state is carried alongside it: a
        # concurrent navigate could clear the flag while this read was in
        # flight, and the capture was still taken under it.
        snapshot = data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""
        snapshot = _filter_page_state_after_handback(session, snapshot, handback_revealed)
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

        # The refs in this snapshot are only meaningful for the document it
        # describes; record which version that is so a mutation carrying them
        # can be refused after somebody else navigates the shared tab.

        return json.dumps({
            "success": True,
            "snapshot": snapshot,
            "element_count": refs_count,
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        # Strip @ prefix if present (our tool convention)
        clean_ref = ref.lstrip("@")

        filtered_at_request = _handback_privacy_filter_enabled(session)
        data = _mutating_tab_call(session, "/click", {"userId": session["user_id"], "ref": clean_ref})
        # The result reports where the click landed, which is where the human
        # is if one took over. Judged on the request-time state and this
        # response's own fact, not just the shared flag a concurrent turn can
        # clear.
        result_handback_revealed = filtered_at_request or _last_response_started_handback()
        # The page this landed on was compared against the snapshot's inside
        # _mutating_tab_call, while it still held the tab identity: doing it
        # here would leave a window for another turn to take the lock and act
        # on refs this click had already invalidated.
        return json.dumps({
            "success": True,
            "clicked": clean_ref,
            "url": _filter_url_after_handback(session, data.get("url", ""), result_handback_revealed),
        })
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        clean_ref = ref.lstrip("@")

        _mutating_tab_call(session, "/type", {"userId": session["user_id"], "ref": clean_ref, "text": text})
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
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        _mutating_tab_call(session, "/scroll", {"userId": session["user_id"], "direction": direction})
        return json.dumps({"success": True, "scrolled": direction})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_back(task_id: Optional[str] = None) -> str:
    """Navigate back via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        filtered_at_request = _handback_privacy_filter_enabled(session)
        data = _mutating_tab_call(session, "/back", {"userId": session["user_id"]})
        result_handback_revealed = filtered_at_request or _last_response_started_handback()
        return json.dumps({"success": True, "url": _filter_url_after_handback(session, data.get("url", ""), result_handback_revealed)})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via Camofox."""
    try:
        session = _get_session(task_id)
        if not session["tab_id"]:
            return tool_error("No browser session. Call browser_navigate first.", success=False)

        data = _mutating_tab_call(session, "/press", {"userId": session["user_id"], "key": key})
        # Enter on a form submits it, and the page that answers is a different
        # document — compared against the snapshot's URL inside
        # _mutating_tab_call, under the tab identity lock.
        _ = data
        return json.dumps({"success": True, "pressed": key})
    except Exception as e:
        return _tool_error_from_exception(e, session=locals().get("session"))
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


def camofox_close(task_id: Optional[str] = None) -> str:
    """Close the browser session via Camofox."""
    try:
        session = _drop_session(task_id)
        if not session:
            return json.dumps({"success": True, "closed": True})

        # Prefer the flag captured when the session was created: teardown can
        # run without the profile scope the live probe needs.
        if session.get("local_server_managed") or _local_server_managed():
            owner = str(session.get("release_owner") or "")
            with _held_owner_lock(owner):
                if _profile_still_in_use(owner):
                    return json.dumps({"success": True, "closed": False, "released": False})
                _release_local_server_lease(session)
            return json.dumps({
                "success": True,
                "closed": False,
                "released": True,
            })

        # Queued with the context captured at creation, and retried: the local
        # state is already gone, so a transient failure here would otherwise
        # leave the server-side session running with nothing able to close it.
        base = str(session.get("delete_base") or "")
        own_owner = str(session.get("release_owner") or "")
        own_url = f"{base}/sessions/{quote(str(session.get('user_id') or ''), safe='')}"
        _teardown_session(session)
        # Only this session's own teardown is forced, and "own" includes the
        # credential: in multiplex two profiles can share a CAMOFOX_URL and an
        # explicit CAMOFOX_USER_ID and differ only by Authorization, so matching
        # on the URL alone would fire the other profile's delete under this
        # one's quiet window. A blanket drain would do the same to every
        # profile, one of which may be mid-takeover.
        _run_pending_teardowns(only_url=own_url, only_owner=own_owner)
        with _sessions_lock:
            closed = not any(
                entry["kind"] == "delete"
                and entry["url"] == own_url
                and entry.get("owner", "") == own_owner
                for entry in _pending_lease_releases
            )
        return json.dumps({"success": True, "closed": closed})
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

        images_filtered_at_request = _handback_privacy_filter_enabled(session)
        with _capture_guard(session):
            data = _get(
                _tab_path(session, "/snapshot"),
                params={"userId": session["user_id"]},
                session=session,
            )
            images_handback_revealed = images_filtered_at_request or _last_response_started_handback()
            if (images_handback_revealed or _handback_privacy_filter_enabled(session)) and not _handback_page_readable(session, data):
                return _blocked_handback_page_error()
        # Image alt/src are page content too: redaction only strips form values
        # and secret-shaped text, so intranet metadata would pass straight
        # through. Same guard as camofox_snapshot, applied after the response
        # because that is what turns the filter on.
        snapshot = data.get("snapshot", "")
        if not isinstance(snapshot, str):
            snapshot = ""
        snapshot = _filter_page_state_after_handback(session, snapshot, images_handback_revealed)

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
    finally:
        # _get_session hands back a referenced entry; release it whichever
        # way this call ends.
        _end_session_call(locals().get("session"))


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
        # The reference _get_session took covers the whole call: the model round
        # trip between the screenshot and the annotation snapshot can run for
        # minutes, and the session is in use throughout. Released in the finally
        # below, which must not be conditional — the early returns above (no
        # tab, filter engaged) are ordinary outcomes, not exemptions.

        # Get screenshot as binary PNG
        screenshot_filtered_at_request = _handback_privacy_filter_enabled(session)
        # A screenshot is a capture like any other — it just returns pixels
        # instead of a tree — so it takes the same critical section, which also
        # re-checks the blocked-landing state under the lock.
        with _capture_guard(session):
            resp = _get_raw(
                _tab_path(session, "/screenshot"),
                params={"userId": session["user_id"]},
                session=session,
            )
            # Judged on the state when the capture was issued as well as now:
            # the epoch that turns the filter on arrives with this very
            # response, and a concurrent navigate could clear the shared flag
            # before this check — either way the image is of the human's screen.
            if screenshot_filtered_at_request or _last_response_started_handback() or _handback_privacy_filter_enabled(session):
                return tool_error(
                    "Browser vision is blocked after human control until the Agent navigates to a new page or closes the session.",
                    success=False,
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
                annotation_filtered_at_request = _handback_privacy_filter_enabled(session)
                with _capture_guard(session):
                    snap_data = _get(
                        _tab_path(session, "/snapshot"),
                        params={"userId": session["user_id"]},
                        session=session,
                    )
                    annotation_handback_revealed = annotation_filtered_at_request or _last_response_started_handback()
                    if (annotation_handback_revealed or _handback_privacy_filter_enabled(session)) and not _handback_page_readable(session, snap_data):
                        return _blocked_handback_page_error()
                # A takeover can land between the screenshot and this call, and
                # the filter it turns on only strips form values — ordinary
                # intranet text would still reach the vision model.
                snapshot = snap_data.get("snapshot", "")
                if not isinstance(snapshot, str):
                    snapshot = ""
                snapshot = _filter_page_state_after_handback(session, snapshot, annotation_handback_revealed)
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
    finally:
        _end_session_call(locals().get("session"))


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
