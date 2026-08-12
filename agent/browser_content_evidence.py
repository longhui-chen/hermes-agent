"""Bounded browser content evidence projected to authenticated chat clients.

The browser tool result is the model-facing source of truth.  This module
extracts only allowlisted text from that already-redacted result and turns it
into a small, versioned payload that downstream clients can persist and render
as ordinary text.  It never reads the page, screenshot, or spill file again.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from agent.browser_state_preview import (
    _decode_label,
    _presentation_safe_paths_and_urls,
)
from agent.redact import redact_sensitive_text


MAX_EVIDENCE_BYTES = 48 * 1024
MAX_BLOCKS = 128
MAX_BLOCK_CHARS = 2_000
MAX_SOURCE_SCAN_CHARS = 256 * 1024

_SNAPSHOT_LINE_RE = re.compile(
    r"^\s*(?:-\s*)?(?P<role>[a-z][a-z0-9_-]*)"
    r'\b(?:\s+"(?P<label>(?:[^"\\]|\\.)*)")?',
    re.IGNORECASE,
)
_SNAPSHOT_TRAILING_REF_RE = re.compile(
    r"\s+\[(?:ref|id)=[^\]]+\]\s*:?\s*$", re.IGNORECASE
)
_HIDDEN_STATE_RE = re.compile(
    r"\[[^\]]*\bhidden\b[^\]]*\]|aria-hidden\s*=\s*true", re.IGNORECASE
)
_SOURCE_TRUNCATION_RE = re.compile(
    r"^\s*\[\.\.\..*(?:truncated|summarized|full snapshot|read_file).*\]\s*$",
    re.IGNORECASE,
)
_SCREENSHOT_PATH_RE = re.compile(r"^\s*screenshot path\s*:", re.IGNORECASE)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
_MARKDOWN_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+)$")
_MARKDOWN_QUOTE_RE = re.compile(r"^\s*>\s?(?P<text>.*)$")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)

_SNAPSHOT_KIND_BY_ROLE = {
    "heading": "heading",
    "paragraph": "paragraph",
    "text": "paragraph",
    "statictext": "paragraph",
    "listitem": "list_item",
    "list-item": "list_item",
    "list_item": "list_item",
    "blockquote": "quote",
    "quote": "quote",
    "row": "table_row",
}
_SENSITIVE_SNAPSHOT_ROLES = {
    "textbox",
    "searchbox",
    "input",
    "password",
    "combobox",
    "spinbutton",
}
_PROVENANCE_BY_TOOL = {
    "browser_navigate": {"page_text"},
    "browser_snapshot": {"page_text", "task_extraction"},
}
_TRUNCATION_REASON_ORDER = ("source", "sanitization", "step_budget")
_BLOCK_KIND_PRIORITY = {
    "other": 0,
    "paragraph": 1,
    "list_item": 2,
    "quote": 2,
    "table_row": 2,
    "heading": 3,
}


def _clean_block_text(
    value: Any,
    *,
    max_chars: int = MAX_BLOCK_CHARS,
) -> tuple[str, bool, bool]:
    """Return ``(text, sanitized, step_truncated)`` for one block."""
    if not isinstance(value, str) or not value:
        return "", False, False
    scan_limit = max_chars * 8
    step_truncated = len(value) > scan_limit
    value = value[:scan_limit]
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        value = value.encode("utf-8", errors="replace").decode("utf-8")
        step_truncated = True

    value, presentation_changed = _presentation_safe_paths_and_urls(value)
    redacted = redact_sensitive_text(
        value,
        force=True,
        redact_url_credentials=True,
    )
    sanitized = presentation_changed or redacted != value
    normalized = " ".join(redacted.split())
    if not normalized or normalized == "[REDACTED]":
        return "", True, step_truncated
    if len(normalized) > max_chars:
        normalized = normalized[:max_chars].rstrip()
        step_truncated = True
    return normalized, sanitized, step_truncated


def _snapshot_blocks(value: str) -> tuple[list[tuple[str, str]], bool]:
    blocks: list[tuple[str, str]] = []
    source_truncated = False
    for line in value.splitlines():
        if _SOURCE_TRUNCATION_RE.match(line):
            source_truncated = True
            continue
        match = _SNAPSHOT_LINE_RE.match(line)
        if match is None or _HIDDEN_STATE_RE.search(line[match.end() :]):
            continue
        role = match.group("role").lower()
        if role in _SENSITIVE_SNAPSHOT_ROLES:
            continue
        raw_label = match.group("label")
        if raw_label is not None:
            label = _decode_label(raw_label)
        else:
            remainder = line[match.end() :].strip()
            if not remainder.startswith(":"):
                continue
            label = _SNAPSHOT_TRAILING_REF_RE.sub("", remainder[1:]).strip()
        if not label:
            continue
        kind = _SNAPSHOT_KIND_BY_ROLE.get(role, "other")
        blocks.append((kind, label))
    return blocks, source_truncated


def _append_deduplicated_block(
    blocks: list[dict[str, str]], kind: str, text: str
) -> int:
    """Append one block and return the number of semantic duplicates removed.

    Accessibility snapshots commonly expose the same label through a semantic
    role (for example ``heading``) and an interactive role (for example
    ``link``). Keep the richer representation regardless of which one appears
    first. Table rows also contain their child cells, so child labels that are
    already present in a retained row are redundant evidence.
    """
    normalized = text.casefold()
    for index, block in enumerate(blocks):
        if block["text"].casefold() != normalized:
            continue
        if _BLOCK_KIND_PRIORITY[kind] > _BLOCK_KIND_PRIORITY[block["kind"]]:
            blocks[index] = {"kind": kind, "text": text}
        return 1

    if kind == "other":
        if any(
            block["kind"] == "table_row" and normalized in block["text"].casefold()
            for block in blocks
        ):
            return 1
    elif kind == "table_row":
        redundant_indexes = [
            index
            for index, block in enumerate(blocks)
            if block["kind"] == "other" and block["text"].casefold() in normalized
        ]
        for index in reversed(redundant_indexes):
            blocks.pop(index)
        blocks.append({"kind": kind, "text": text})
        return len(redundant_indexes)

    blocks.append({"kind": kind, "text": text})
    return 0


def _plain_text_blocks(value: str) -> tuple[list[tuple[str, str]], bool]:
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    source_truncated = False

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(paragraph)))
            paragraph.clear()

    for line in value.splitlines():
        if _SOURCE_TRUNCATION_RE.match(line) or _SCREENSHOT_PATH_RE.match(line):
            flush_paragraph()
            source_truncated = True
            continue
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            flush_paragraph()
            blocks.append(("heading", match.group("text")))
            continue
        match = _MARKDOWN_LIST_RE.match(line)
        if match:
            flush_paragraph()
            blocks.append(("list_item", match.group("text")))
            continue
        match = _MARKDOWN_QUOTE_RE.match(line)
        if match:
            flush_paragraph()
            blocks.append(("quote", match.group("text")))
            continue
        if "|" in stripped and not _MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            flush_paragraph()
            blocks.append(("table_row", stripped.strip("|").strip()))
            continue
        if _MARKDOWN_TABLE_SEPARATOR_RE.match(line):
            continue
        paragraph.append(stripped)
    flush_paragraph()
    return blocks, source_truncated


def _serialized_size(payload: Mapping[str, Any]) -> int:
    return len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _fit_byte_budget(
    payload: dict[str, Any], reasons: set[str]
) -> dict[str, Any] | None:
    blocks = payload["blocks"]
    while blocks and _serialized_size(payload) > MAX_EVIDENCE_BYTES:
        blocks.pop()
        reasons.add("step_budget")
        payload["truncated"] = True
        payload["truncationReasons"] = [
            reason for reason in _TRUNCATION_REASON_ORDER if reason in reasons
        ]
    return (
        payload if blocks and _serialized_size(payload) <= MAX_EVIDENCE_BYTES else None
    )


def project_browser_content_evidence(
    tool_name: str,
    output: Mapping[str, Any],
    *,
    browser_session_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a bounded ``BrowserContentEvidenceV1`` or ``None``.

    Snapshot provenance must be stamped by the producer.  The projector never
    guesses that model-generated extraction text is page text.
    """
    if output.get("success") is not True:
        return None

    source_value: Any
    provenance: str
    source_truncated = output.get("_browser_content_source_truncated") is True
    if tool_name == "browser_vision":
        provenance = "vision_analysis"
        source_value = output.get("analysis")
        if not isinstance(source_value, str):
            return None
        candidates, marker_truncated = _plain_text_blocks(
            source_value[:MAX_SOURCE_SCAN_CHARS]
        )
    else:
        provenance_value = output.get("_browser_content_provenance")
        if provenance_value not in _PROVENANCE_BY_TOOL.get(tool_name, set()):
            return None
        provenance = str(provenance_value)
        source_value = output.get("snapshot")
        if not isinstance(source_value, str):
            return None
        bounded_source = source_value[:MAX_SOURCE_SCAN_CHARS]
        if provenance == "page_text":
            candidates, marker_truncated = _snapshot_blocks(bounded_source)
        else:
            candidates, marker_truncated = _plain_text_blocks(bounded_source)

    source_truncated = (
        source_truncated
        or marker_truncated
        or len(source_value) > MAX_SOURCE_SCAN_CHARS
    )
    original_block_count = len(candidates)
    if original_block_count == 0:
        return None

    reasons: set[str] = set()
    if source_truncated:
        reasons.add("source")
    blocks: list[dict[str, str]] = []
    duplicate_count = 0
    for kind, raw_text in candidates:
        text, sanitized, block_truncated = _clean_block_text(raw_text)
        if sanitized:
            reasons.add("sanitization")
        if block_truncated:
            reasons.add("step_budget")
        if not text:
            continue
        duplicate_delta = _append_deduplicated_block(blocks, kind, text)
        duplicate_count += duplicate_delta
        if duplicate_delta:
            continue
        if len(blocks) > MAX_BLOCKS:
            blocks.pop()
            reasons.add("step_budget")
            break
    if not blocks:
        return None

    payload: dict[str, Any] = {
        "version": 1,
        "provenance": provenance,
        "blocks": blocks,
        "originalBlockCount": original_block_count,
    }
    if browser_session_id:
        session_id, sanitized, session_truncated = _clean_block_text(
            browser_session_id,
            max_chars=256,
        )
        if session_id:
            payload["browserSessionId"] = session_id
        if sanitized:
            reasons.add("sanitization")
        if session_truncated:
            reasons.add("step_budget")
    if duplicate_count:
        payload["duplicateBlockCount"] = duplicate_count
    if reasons:
        payload["truncated"] = True
        payload["truncationReasons"] = [
            reason for reason in _TRUNCATION_REASON_ORDER if reason in reasons
        ]
    return _fit_byte_budget(payload, reasons)
