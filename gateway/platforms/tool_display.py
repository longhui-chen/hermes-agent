"""Bounded, redacted tool-call display projections for BT-T1.

This module is deliberately pure and unconnected. T2 will call it at the
producer boundary; clients receive only these bounded summaries and never raw
arguments/output.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from itertools import islice
from typing import Any

ARGS_MAX_BYTES = 256
SUMMARY_MAX_BYTES = 4 * 1024

_SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:token|api[_-]?key|access[_-]?key|secret|password|cookie)\s*[:=]\s*)[^\s,;]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+"), r"\1[REDACTED]"),
    (re.compile(r"(?:/Users/[^\s/'\"]+|/home/[^\s/'\"]+|/private/tmp/[^\s/'\"]+)"), "[PRIVATE_PATH]"),
)


def redact(text: str) -> str:
    """Redact credentials and private local paths before persistence."""
    try:
        result = str(text)
        result.encode("utf-8")
    except UnicodeError:
        return "[INVALID_TEXT]"
    for pattern, replacement in _SECRET_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


_SECRET_KEYS = re.compile(r"(?i)(token|api[_-]?key|access[_-]?key|secret|password|cookie)")


def _redact_value(value: Any, *, max_chars: int = SUMMARY_MAX_BYTES * 4) -> Any:
    if isinstance(value, Mapping):
        return {str(key): "[REDACTED]" if _SECRET_KEYS.search(str(key)) else _redact_value(item, max_chars=max_chars) for key, item in islice(value.items(), 256)}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, max_chars=max_chars) for item in islice(value, 256)]
    if isinstance(value, str):
        return redact(value)[:max_chars]
    return value


def _truncate_utf8(text: str, limit: int) -> tuple[str, bool, int]:
    source = text if len(text) <= limit * 4 else text[: limit * 4]
    clean = redact(source)
    try:
        raw = clean.encode("utf-8")
    except UnicodeError:
        clean = "[INVALID_TEXT]"
        raw = clean.encode("utf-8")
    exact = len(raw)
    if exact <= limit:
        return clean, False, exact
    # Binary search a character boundary without ever emitting a partial codepoint.
    lo, hi = 0, len(clean)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(clean[:mid].encode("utf-8")) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return clean[:lo], True, len(clean[:lo].encode("utf-8"))


def _safe_json(value: Any, *, max_chars: int = SUMMARY_MAX_BYTES * 4) -> str:
    value = _redact_value(value, max_chars=max_chars)
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:max_chars]
    except (TypeError, ValueError):
        return str(value)


def source_from_registration(tool_id: str, registration: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Project a registry entry into the bounded source object."""
    row = registration or {}
    kind = str(row.get("kind") or "builtin")
    if kind not in {"builtin", "mcp", "skill", "connector"}:
        kind = "builtin"
    source_id = str(row.get("id") or row.get("server") or tool_id)
    label = str(row.get("label") or row.get("server_label") or row.get("name") or source_id)
    return {"kind": kind, "id": source_id, "label": label}


def args_summary(arguments: Any) -> dict[str, Any]:
    text, truncated, _ = _truncate_utf8(_safe_json(arguments, max_chars=ARGS_MAX_BYTES * 4), ARGS_MAX_BYTES)
    return {"args_summary": text, "truncated": truncated}


def result_display(output: Any = None, *, error: Any = None, content_type: str | None = None) -> dict[str, Any]:
    is_error = error not in (None, "")
    if is_error:
        kind = "error"
        value = str(error)
    elif content_type in {"text", "markdown", "json", "error"}:
        kind = content_type
        value = _safe_json(output, max_chars=SUMMARY_MAX_BYTES * 4)
    elif isinstance(output, (Mapping, list, tuple)):
        kind = "json"
        value = _safe_json(output)
    else:
        value = "" if output is None else str(output)
        kind = "markdown" if "```" in value else "text"
    summary, truncated, _ = _truncate_utf8(value, SUMMARY_MAX_BYTES)
    return {"summary": summary, "content_type": "error" if is_error else kind, "truncated": truncated, "bytes": len(summary.encode("utf-8"))}


def build_tool_start_display(tool_id: str, arguments: Any, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"source": source_from_registration(tool_id, registration), "display": args_summary(arguments)}


def build_tool_result_display(output: Any = None, *, error: Any = None, content_type: str | None = None) -> dict[str, Any]:
    return {"display": result_display(output, error=error, content_type=content_type)}
