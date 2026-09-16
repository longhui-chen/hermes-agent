"""Bounded, redacted tool-call display projections for BT-T1.

These pure helpers serve the producer boundary; clients receive bounded,
redacted summaries instead of raw arguments/output.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

from agent.redact import redact_sensitive_text

ARGS_MAX_BYTES = 256
SUMMARY_MAX_BYTES = 4 * 1024

_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:authorization\s*:\s*)(?:basic|bearer)\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:token|api[_-]?key|access[_-]?key|secret|password|cookie)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)([\"'](?:token|api[_-]?key|access[_-]?key|secret|password|cookie)[\"']\s*:\s*[\"'])[^\"']*([\"'])"), r"\1[REDACTED]\2"),
    (re.compile(r"(?:/Users/[^\s/'\"]+|/home/[^\s/'\"]+|/private/tmp/[^\s/'\"]+)"), "[PRIVATE_PATH]"),
)


def redact(text: str) -> str:
    """Redact bounded text using the IM policy plus display-only private paths."""
    if len(text) > SUMMARY_MAX_BYTES:
        return "[TRUNCATED]"
    try:
        text.encode("utf-8", "strict")
        result = redact_sensitive_text(text, force=True)
        for pattern, replacement in _SECRET_PATTERNS:
            result = pattern.sub(replacement, result)
        return result
    except Exception:
        # Display is optional; a failed sanitizer must never expose its input.
        return "[INVALID_TEXT]"


_SECRET_KEYS = re.compile(r"(?i)(token|api[_-]?key|access[_-]?key|secret|password|cookie|authorization|credential|private[_-]?key|session[_-]?secret)")


def _safe_text(value: Any, limit: int) -> tuple[str, bool]:
    """Bound the entire traversal, including keys, before serialization.

    Limits are per projection, not per level: at most 256 nodes and `limit`
    source characters are examined. Oversized atoms are omitted rather than
    cut mid-secret before redaction. No state survives this call.
    """
    remaining_nodes = 256
    remaining_chars = limit
    cut = False
    seen: set[int] = set()

    def clean(item: Any, depth: int = 0) -> Any:
        nonlocal remaining_nodes, remaining_chars, cut
        if remaining_nodes <= 0 or depth > 16:
            cut = True
            return "[TRUNCATED]"
        remaining_nodes -= 1
        if isinstance(item, str):
            if len(item) > remaining_chars:
                cut = True
                return "[TRUNCATED]"
            remaining_chars -= len(item)
            return redact(item)
        if isinstance(item, (Mapping, list, tuple)):
            if id(item) in seen:
                cut = True
                return "[CYCLE]"
            seen.add(id(item))
            result = {} if isinstance(item, Mapping) else []
            entries = item.items() if isinstance(item, Mapping) else item
            count = 0
            for entry in islice(entries, 256):
                count += 1
                if remaining_nodes <= 0:
                    cut = True
                    break
                if isinstance(item, Mapping):
                    key, child = entry
                    if not isinstance(key, str) or len(key) > remaining_chars:
                        cut = True
                        break
                    remaining_chars -= len(key)
                    safe_key = redact(key)
                    if _SECRET_KEYS.search(key):
                        remaining_nodes -= 1
                        result[safe_key] = "[REDACTED]"
                    else:
                        result[safe_key] = clean(child, depth + 1)
                else:
                    result.append(clean(entry, depth + 1))
            if count < len(item):
                cut = True
            seen.remove(id(item))
            return result
        if item is None or isinstance(item, (bool, float)):
            return item
        if isinstance(item, int) and item.bit_length() <= 64:
            return item
        cut = True
        return "[UNSUPPORTED]"

    try:
        safe = clean(value)
        if isinstance(safe, str):
            return safe, cut
        # All strings and containers are now bounded and redacted. JSON escaping
        # can enlarge this bounded representation; final UTF-8 clipping follows.
        return json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False), cut
    except Exception:
        return "[INVALID_TEXT]", True


def _truncate_utf8(text: str, limit: int) -> tuple[str, bool, int]:
    raw = text.encode("utf-8", "strict")
    if len(raw) <= limit:
        return text, False, len(raw)
    result = raw[:limit].decode("utf-8", "ignore")
    return result, True, len(result.encode("utf-8"))


def _display_text(value: Any, limit: int) -> tuple[str, bool, int]:
    safe, cut = _safe_text(value, limit)
    text, clipped, size = _truncate_utf8(safe, limit)
    return text, cut or clipped, size


def source_from_registration(tool_id: str, registration: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Project a registry entry into the bounded source object."""
    row = registration or {}
    kind = row.get("kind") or "builtin"
    if not isinstance(kind, str) or kind not in {"builtin", "mcp", "skill", "connector"}:
        kind = "builtin"
    # MCP grouping identity is the registered server, never the tool's id.
    source_id = (row.get("server") or row.get("id") or tool_id) if kind == "mcp" else (row.get("id") or tool_id)
    label = row.get("server_label") or row.get("label") or row.get("name") or source_id
    return {"kind": kind, "id": _display_text(source_id, SUMMARY_MAX_BYTES)[0], "label": _display_text(label, SUMMARY_MAX_BYTES)[0]}


def args_summary(arguments: Any) -> dict[str, Any]:
    text, truncated, _ = _display_text(arguments, ARGS_MAX_BYTES)
    return {"args_summary": text, "truncated": truncated}


def result_display(output: Any = None, *, error: Any = None, content_type: str | None = None) -> dict[str, Any]:
    is_error = error is not None and not (isinstance(error, str) and error == "")
    value = error if is_error else output
    # Preserve scalar text conventions without invoking arbitrary __str__.
    if value is None:
        value = ""
    elif isinstance(value, bool):
        value = "True" if value else "False"
    summary, truncated, size = _display_text(value, SUMMARY_MAX_BYTES)
    if is_error:
        kind = "error"
    elif content_type in {"text", "markdown", "json", "error"}:
        kind = content_type
    elif isinstance(output, (Mapping, list, tuple)):
        kind = "json"
    else:
        kind = "markdown" if "```" in summary else "text"
    return {"summary": summary, "content_type": kind, "truncated": truncated, "bytes": size}


def build_tool_start_display(tool_id: str, arguments: Any, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"source": source_from_registration(tool_id, registration), "display": args_summary(arguments)}


def build_tool_result_display(output: Any = None, *, error: Any = None, content_type: str | None = None) -> dict[str, Any]:
    return {"display": result_display(output, error=error, content_type=content_type)}
