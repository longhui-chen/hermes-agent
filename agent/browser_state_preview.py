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
from urllib.parse import unquote, urlsplit

from agent.redact import redact_sensitive_text


MAX_PREVIEW_BYTES = 8 * 1024
MAX_ELEMENTS = 24
MAX_LABEL_CHARS = 160
MAX_SUMMARY_CHARS = 512
MAX_TITLE_CHARS = 160
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
_ROLE_PRIORITY = {
    "alert": 0,
    "heading": 1,
    "textbox": 2,
    "button": 3,
    "link": 4,
    "other": 5,
}
_ELEMENT_RE = re.compile(
    r"^\s*(?:-\s*)?"
    r"(?P<role>[a-z][a-z0-9_-]*)"
    r'\b(?:\s+"(?P<label>(?:[^"\\]|\\.)*)")?',
    re.IGNORECASE,
)
_HIDDEN_STATE_RE = re.compile(
    r"\[[^\]]*\bhidden\b[^\]]*\]|aria-hidden\s*=\s*true", re.IGNORECASE
)
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_EMBEDDED_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_EMBEDDED_FILE_URL_RE = re.compile(
    r"\bfile:(?://[^\s/\"'<>]*)?/[^\s\"'<>]+", re.IGNORECASE
)
_LOCAL_PATH_RE = re.compile(
    r"(^|[^a-z0-9/])(?:/[^\s\"'<>]+|[a-z]:\\[^\s\"'<>]+)",
    re.IGNORECASE,
)
_URL_TRAILING_PUNCTUATION = ".,;:!?)]}"
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


def _hostname_only_embedded_urls(value: str) -> tuple[str, bool]:
    """Replace absolute HTTP(S) URLs in presentation text with hostnames.

    Titles and accessibility labels are free text, so a URL can bypass the
    structured ``url`` field entirely.  Paths are never presentation-safe:
    magic links and signed resources routinely place credentials there.  URL-
    like text that cannot be parsed fails closed instead of preserving it.
    """
    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        token = match.group(0)
        trailing = ""
        while token and token[-1] in _URL_TRAILING_PUNCTUATION:
            trailing = token[-1] + trailing
            token = token[:-1]
        safe_url, _ = _safe_url(token)
        changed = True
        if safe_url is None:
            return f"[REDACTED]{trailing}"
        return f"{safe_url['hostname']}{trailing}"

    return _EMBEDDED_HTTP_URL_RE.sub(_replace, value), changed


def _presentation_safe_paths_and_urls(value: str) -> tuple[str, bool]:
    """Apply bounded raw/encoded URL and local-path presentation policy."""
    candidate = value
    decoded = value
    decoded_rounds = 0
    for _ in range(2):
        try:
            next_value = unquote(decoded, errors="strict")
        except UnicodeError:
            return "[REDACTED]", True
        if next_value == decoded:
            break
        decoded = next_value
        decoded_rounds += 1

    if decoded_rounds and (
        _EMBEDDED_HTTP_URL_RE.search(decoded)
        or _EMBEDDED_FILE_URL_RE.search(decoded)
        or _LOCAL_PATH_RE.search(decoded)
    ):
        candidate = decoded

    candidate, urls_redacted = _hostname_only_embedded_urls(candidate)
    if _EMBEDDED_FILE_URL_RE.search(candidate) or _LOCAL_PATH_RE.search(candidate):
        return "[REDACTED]", True
    return candidate, urls_redacted or candidate != value


def _clean_text(value: Any, max_chars: int) -> tuple[str, bool]:
    if not isinstance(value, str) or not value:
        return "", False
    # Bound work before running the shared redactor.  The omitted suffix can
    # never enter the preview, so it does not need to be inspected.
    scan_limit = max(max_chars * 8, max_chars)
    truncated = len(value) > scan_limit
    value = value[:scan_limit]
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # Browser-controlled text can contain lone UTF-16 surrogates.  Replace
        # them before redaction/serialization so preview failure cannot escape
        # into the tool-completion path.
        value = value.encode("utf-8", errors="replace").decode("utf-8")
        truncated = True
    value, unsafe_content_removed = _presentation_safe_paths_and_urls(value)
    truncated = truncated or unsafe_content_removed
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

    # Every path segment is untrusted presentation data: magic links, password
    # resets, invitations and signed resources routinely put credentials there.
    # Keep only site identity; userinfo, port, path, query and fragment are
    # intentionally never copied, regardless of encoding.
    return {"hostname": hostname}, truncated


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
    for bracket in _BRACKET_RE.findall(line):
        normalized = bracket.lower().replace("_", "-")
        for state in _SAFE_STATES:
            if re.search(rf"(?:^|[\s,;]){re.escape(state)}(?:$|[\s,;=])", normalized):
                return state
    return ""


def _snapshot_elements(snapshot: Any) -> tuple[list[dict[str, str]], int, bool]:
    if not isinstance(snapshot, str) or not snapshot:
        return [], 0, False

    truncated = len(snapshot) > MAX_SNAPSHOT_SCAN_CHARS
    snapshot = snapshot[:MAX_SNAPSHOT_SCAN_CHARS]
    candidates: list[dict[str, str]] = []
    matched_count = 0
    for line in snapshot.splitlines():
        match = _ELEMENT_RE.match(line)
        if not match:
            continue
        attributes = line[match.end() :]
        if _HIDDEN_STATE_RE.search(attributes):
            continue
        matched_count += 1

        source_role = match.group("role").lower()
        element: dict[str, str] = {"role": _ROLE_MAP.get(source_role, "other")}
        raw_label = match.group("label")
        if raw_label:
            label, label_truncated = _clean_text(
                _decode_label(raw_label), MAX_LABEL_CHARS
            )
            truncated = truncated or label_truncated
            if label:
                element["label"] = label
        state = _safe_state(attributes)
        if state:
            element["state"] = state[:MAX_LABEL_CHARS]
        candidates.append(element)

    if len(candidates) > MAX_ELEMENTS:
        truncated = True
        selected_indices = sorted(
            sorted(
                range(len(candidates)),
                key=lambda index: (
                    _ROLE_PRIORITY[candidates[index]["role"]],
                    index,
                ),
            )[:MAX_ELEMENTS]
        )
        elements = [candidates[index] for index in selected_indices]
    else:
        elements = candidates
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
        drop_index = max(
            range(len(elements)),
            key=lambda index: (
                _ROLE_PRIORITY.get(elements[index].get("role", "other"), 5),
                index,
            ),
        )
        elements.pop(drop_index)
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
    native_vision = (
        source == "vision"
        and output.get("success") is not False
        and output.get("_multimodal") is True
        and isinstance(output.get("text_summary"), str)
    )
    if source is None or (output.get("success") is not True and not native_vision):
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
        raw_summary = output.get("analysis")
        if native_vision:
            # browser_vision appends the device-local screenshot path to the
            # model-facing fallback.  Only the bounded prose before that marker
            # is eligible for the UI projection; content/meta are never read.
            raw_summary = output.get("text_summary", "").split(
                "Screenshot path:", 1
            )[0].rstrip()
        summary, summary_truncated = _clean_text(
            raw_summary, MAX_SUMMARY_CHARS
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
