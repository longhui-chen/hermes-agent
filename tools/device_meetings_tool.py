"""Read meeting summaries already stored by the local Zettlab device."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import get_secret
from tools.loopback_transport import is_trusted_loopback_http, urlopen_hardened
from tools.registry import registry


logger = logging.getLogger(__name__)

_TOKEN = "X-Zettlab-Agent-Action-Token"
_LIST = "/api/v1/internal/meetings"
_GET = "/api/v1/internal/meetings/"
_REQUEST_TIMEOUT_SECONDS = 8
_MAX_RESPONSE_BYTES = 512 * 1024
_MAX_ATTEMPTS = 2
_MAX_OFFSET = 100_000
_RETRYABLE_HTTP_STATUS = frozenset({502, 503, 504})

_ERROR_MESSAGES = {
    "unavailable": "Device meeting bridge is unavailable.",
    "invalid_response": "Device meeting bridge returned an invalid response.",
    "response_too_large": "Device meeting response exceeded the size limit.",
}

SCHEMA = {
    "name": "device_meetings",
    "description": (
        "Read stored device meetings. Use action list or get; "
        "meeting_id must come from list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "get"],
            },
            "meeting_id": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "offset": {"type": "integer", "minimum": 0, "maximum": _MAX_OFFSET},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def _secret(name: str) -> str:
    try:
        return str(get_secret(name, "") or "").strip()
    except Exception:
        return ""


def _base() -> str | None:
    raw = _secret("ZET_CHAT_APPEND_URL")
    try:
        parts = urlsplit(raw)
        # Accessing port validates malformed bracketed/overflow ports.
        _ = parts.port
    except ValueError:
        return None
    if (
        not parts.netloc
        or not is_trusted_loopback_http(parts)
        or parts.username is not None
        or parts.password is not None
    ):
        return None
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _available() -> bool:
    return bool(_base() and _secret("ZETTLAB_AGENT_ACTION_TOKEN"))


# Do not reuse a cached availability result across multiplexed profiles.
_available._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _error_result(code: str) -> str:
    message = _ERROR_MESSAGES.get(code, _ERROR_MESSAGES["unavailable"])
    return json.dumps(
        {"error": {"code": code, "message": message}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _integer_arg(args: dict[str, Any], name: str, default: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _request_path(args: dict[str, Any]) -> str | None:
    action = str(args.get("action", "")).strip()
    if action == "list":
        limit = max(1, min(20, _integer_arg(args, "limit", 20)))
        offset = max(0, min(_MAX_OFFSET, _integer_arg(args, "offset", 0)))
        return f"{_LIST}?limit={limit}&offset={offset}"
    if action == "get":
        meeting_id = str(args.get("meeting_id", "")).strip()
        if meeting_id:
            return _GET + urllib.parse.quote(meeting_id, safe="")
    return None


def _decode_response(raw: bytes) -> str:
    if len(raw) > _MAX_RESPONSE_BYTES:
        logger.warning("device meetings response exceeded %d bytes", _MAX_RESPONSE_BYTES)
        return _error_result("response_too_large")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        logger.warning("device meetings response was not valid JSON")
        return _error_result("invalid_response")

    # Current local-server responses use {code, data}. Treat the envelope as a
    # transport contract and pass only its data to the model. Older devices
    # returned the payload directly, so valid bare JSON remains supported.
    if isinstance(parsed, dict) and "code" in parsed:
        code = parsed.get("code")
        if not isinstance(code, int) or isinstance(code, bool):
            return _error_result("invalid_response")
        if code != 200:
            return _error_result("unavailable")
        if "data" not in parsed:
            return _error_result("invalid_response")
        parsed = parsed["data"]

    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError):
        logger.warning("device meetings response could not be serialized")
        return _error_result("invalid_response")


def _read_response(request: urllib.request.Request) -> str:
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urlopen_hardened(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                return _decode_response(raw)
            if not isinstance(status, int) or not 200 <= status < 300:
                if status in _RETRYABLE_HTTP_STATUS and attempt + 1 < _MAX_ATTEMPTS:
                    continue
                return _error_result("unavailable")
            return _decode_response(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_HTTP_STATUS and attempt + 1 < _MAX_ATTEMPTS:
                continue
            logger.debug("device meetings HTTP request failed: %s", type(exc).__name__)
            return _error_result("unavailable")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < _MAX_ATTEMPTS:
                continue
            logger.debug("device meetings request failed: %s", type(exc).__name__)
            return _error_result("unavailable")
    return _error_result("unavailable")


def device_meetings_tool(args: Any, **_kw: Any) -> str:
    try:
        normalized_args = args if isinstance(args, dict) else {}
        base = _base()
        token = _secret("ZETTLAB_AGENT_ACTION_TOKEN")
        if not base or not token:
            return _error_result("unavailable")

        path = _request_path(normalized_args)
        if path is None:
            return _error_result("invalid_response")

        request = urllib.request.Request(
            base + path,
            headers={_TOKEN: token, "Accept": "application/json"},
        )
        return _read_response(request)
    except Exception as exc:
        # Keep provider details out of the model-facing result. The exception
        # type is enough for local diagnostics and cannot contain credentials.
        logger.debug("device meetings tool failed: %s", type(exc).__name__)
        return _error_result("unavailable")


registry.register(
    name="device_meetings",
    toolset="zettlab_skill_runtime",
    schema=SCHEMA,
    handler=device_meetings_tool,
    check_fn=_available,
    emoji="🗒️",
    max_result_size_chars=60000,
)
