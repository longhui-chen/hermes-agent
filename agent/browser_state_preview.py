"""Bounded, presentation-safe browser state projected to live chat clients.

Browser tools return rich results for the model, including raw accessibility
trees and device-local paths.  Those results are deliberately not a client
contract.  This module is the single seam that turns a successful browser tool
result into a small, versioned preview suitable for an authenticated UI.
"""

from __future__ import annotations

from collections.abc import Mapping
import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlsplit

from agent.redact import redact_sensitive_text


MAX_PREVIEW_BYTES = 8 * 1024
MAX_ELEMENTS = 24
MAX_LABEL_CHARS = 160
MAX_SUMMARY_CHARS = 512
MAX_TITLE_CHARS = 160
MAX_PATH_CHARS = 512
MAX_SESSION_ID_CHARS = 256
MAX_SNAPSHOT_SCAN_CHARS = 64 * 1024
MAX_URL_SCAN_CHARS = 4096

_TOOL_SOURCES = {
    "browser_navigate": "navigate",
    "browser_snapshot": "snapshot",
    "browser_vision": "vision",
    "browser_click": "action_result",
    "browser_back": "action_result",
}

_ROLE_MAP = {
    "heading": "heading",
    "button": "button",
    "link": "link",
    "textbox": "textbox",
    "alert": "alert",
    "checkbox": "other",
    "radio": "other",
    "combobox": "other",
    "menuitem": "other",
    "option": "other",
    "switch": "other",
    "tab": "other",
}
_ELEMENT_RE = re.compile(
    r"^\s*(?:-\s*)?"
    r"(heading|button|link|textbox|alert|checkbox|radio|combobox|menuitem|option|switch|tab)"
    r'\b(?:\s+"((?:[^"\\]|\\.)*)")?',
    re.IGNORECASE,
)
_HIDDEN_STATE_RE = re.compile(
    r"\[[^\]]*\bhidden\b[^\]]*\]|aria-hidden\s*=\s*true", re.IGNORECASE
)
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_SAFE_STATES = (
    "disabled",
    "checked",
    "unchecked",
    "expanded",
    "collapsed",
    "selected",
    "required",
    "readonly",
    "pressed",
)


def _clean_text(value: Any, max_chars: int) -> tuple[str, bool]:
    if not isinstance(value, str) or not value:
        return "", False
    # Bound work before running the shared redactor.  The omitted suffix can
    # never enter the preview, so it does not need to be inspected.
    scan_limit = max(max_chars * 8, max_chars)
    truncated = len(value) > scan_limit
    value = value[:scan_limit]
    value = redact_sensitive_text(value, force=True, redact_url_credentials=True)
    value = " ".join(value.split())
    if len(value) > max_chars:
        value = value[:max_chars].rstrip()
        truncated = True
    return value, truncated


def _safe_url(value: Any) -> tuple[dict[str, str] | None, bool]:
    if not isinstance(value, str) or not value:
        return None, False
    truncated = len(value) > MAX_URL_SCAN_CHARS
    value = value[:MAX_URL_SCAN_CHARS]
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return None, truncated
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except (TypeError, ValueError):
        return None, truncated
    if parsed.scheme.lower() not in {"http", "https"} or not hostname:
        return None, truncated

    hostname = hostname.lower()
    if len(hostname) > 253:
        return None, True
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if re.fullmatch(r"[a-z0-9.-]+", hostname) is None:
            return None, truncated

    path, path_truncated = _clean_text(parsed.path or "/", MAX_PATH_CHARS)
    truncated = truncated or path_truncated
    result = {"hostname": hostname}
    if path:
        result["path"] = path
    # userinfo, port, query and fragment are intentionally never copied.
    return result, truncated


def _decode_label(value: str) -> str:
    # Accessibility snapshots use JSON-like escapes inside quoted names.  A
    # failed decode degrades to the literal text; it never makes raw lines part
    # of the preview.
    try:
        decoded = json.loads(f'"{value}"')
    except (json.JSONDecodeError, TypeError):
        decoded = value.replace(r"\"", '"').replace(r"\\", "\\")
    return decoded if isinstance(decoded, str) else str(decoded)


def _safe_state(line: str) -> str:
    states: list[str] = []
    for bracket in _BRACKET_RE.findall(line):
        normalized = bracket.lower().replace("_", "-")
        for state in _SAFE_STATES:
            if re.search(rf"(?:^|[\s,;]){re.escape(state)}(?:$|[\s,;=])", normalized):
                if state not in states:
                    states.append(state)
    return ", ".join(states)


def _snapshot_elements(snapshot: Any) -> tuple[list[dict[str, str]], int, bool]:
    if not isinstance(snapshot, str) or not snapshot:
        return [], 0, False

    truncated = len(snapshot) > MAX_SNAPSHOT_SCAN_CHARS
    snapshot = snapshot[:MAX_SNAPSHOT_SCAN_CHARS]
    elements: list[dict[str, str]] = []
    matched_count = 0
    for line in snapshot.splitlines():
        if _HIDDEN_STATE_RE.search(line):
            continue
        match = _ELEMENT_RE.match(line)
        if not match:
            continue
        matched_count += 1
        if len(elements) >= MAX_ELEMENTS:
            truncated = True
            continue

        source_role = match.group(1).lower()
        element: dict[str, str] = {"role": _ROLE_MAP[source_role]}
        raw_label = match.group(2)
        if raw_label:
            label, label_truncated = _clean_text(
                _decode_label(raw_label), MAX_LABEL_CHARS
            )
            truncated = truncated or label_truncated
            if label:
                element["label"] = label
        state = _safe_state(line)
        if state:
            element["state"] = state[:MAX_LABEL_CHARS]
        elements.append(element)
    return elements, matched_count, truncated


def _bounded_count(value: Any, fallback: int) -> tuple[int, bool]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return fallback, False
    if value > 1_000_000:
        return 1_000_000, True
    return value, False


def _serialized_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _fit_byte_budget(payload: dict[str, Any]) -> dict[str, Any]:
    elements = payload.get("elements")
    while (
        isinstance(elements, list)
        and elements
        and _serialized_size(payload) > MAX_PREVIEW_BYTES
    ):
        elements.pop()
        payload["truncated"] = True
    if isinstance(elements, list) and not elements:
        payload.pop("elements", None)

    # The character caps normally fit after element trimming.  These final
    # fallbacks make the byte ceiling absolute even for four-byte Unicode.
    for field in ("summary", "title", "browserSessionId"):
        if _serialized_size(payload) <= MAX_PREVIEW_BYTES:
            break
        if field in payload:
            payload.pop(field, None)
            payload["truncated"] = True
    if _serialized_size(payload) > MAX_PREVIEW_BYTES:
        url = payload.get("url")
        if isinstance(url, dict):
            url.pop("path", None)
            payload["truncated"] = True
    return payload


def project_browser_state_preview(
    tool_name: str,
    output: Mapping[str, Any],
    *,
    browser_session_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a bounded ``BrowserStatePreviewV1`` or ``None``.

    The function is intentionally fail-closed: only known core browser tools,
    successful results and allowlisted fields can produce a preview.
    """
    source = _TOOL_SOURCES.get(tool_name)
    if source is None or output.get("success") is not True:
        return None

    preview: dict[str, Any] = {"version": 1, "source": source}
    truncated = False

    if browser_session_id:
        session_id, session_truncated = _clean_text(
            browser_session_id, MAX_SESSION_ID_CHARS
        )
        truncated = truncated or session_truncated
        if session_id:
            preview["browserSessionId"] = session_id

    safe_url, url_truncated = _safe_url(output.get("url"))
    truncated = truncated or url_truncated
    if safe_url:
        preview["url"] = safe_url

    title, title_truncated = _clean_text(output.get("title"), MAX_TITLE_CHARS)
    truncated = truncated or title_truncated
    if title:
        preview["title"] = title

    if source in {"navigate", "snapshot"}:
        elements, matched_count, elements_truncated = _snapshot_elements(
            output.get("snapshot")
        )
        truncated = truncated or elements_truncated
        if elements:
            preview["elements"] = elements
            first_summary = next(
                (
                    item.get("label", "")
                    for item in elements
                    if item["role"] in {"heading", "alert"} and item.get("label")
                ),
                "",
            )
            if first_summary:
                preview["summary"] = first_summary[:MAX_SUMMARY_CHARS]
        element_count, count_truncated = _bounded_count(
            output.get("element_count"), matched_count
        )
        truncated = truncated or count_truncated or element_count > len(elements)
        preview["elementCount"] = element_count

    if source == "vision":
        summary, summary_truncated = _clean_text(
            output.get("analysis"), MAX_SUMMARY_CHARS
        )
        truncated = truncated or summary_truncated
        if summary:
            preview["summary"] = summary

    # An action result without a safe landing URL does not describe page state.
    if source == "action_result" and "url" not in preview:
        return None
    if source == "vision" and "summary" not in preview:
        return None

    if truncated:
        preview["truncated"] = True
    return _fit_byte_budget(preview)
