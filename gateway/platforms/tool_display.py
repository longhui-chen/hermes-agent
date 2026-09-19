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
from urllib.parse import urlsplit

from agent.redact import redact_sensitive_text

ARGS_MAX_BYTES = 256
SUMMARY_MAX_BYTES = 1024
_SOURCE_MAX_BYTES = 4 * 1024

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
    if len(text) > _SOURCE_MAX_BYTES:
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
    return {"kind": kind, "id": _display_text(source_id, _SOURCE_MAX_BYTES)[0], "label": _display_text(label, _SOURCE_MAX_BYTES)[0]}


def _terminal_args_summary(command: str) -> tuple[str, bool]:
    """Scan only the display budget; never materialize every input line."""
    first = ""
    current: list[str] = []
    count = 0
    scanned = 0
    for char in islice(command, ARGS_MAX_BYTES + 1):
        scanned += 1
        if char in {"\r", "\n"}:
            if current:
                count += 1
                if not first:
                    first = "".join(current)
                current.clear()
            continue
        if len(current) < ARGS_MAX_BYTES:
            current.append(char)
    if current:
        count += 1
        if not first:
            first = "".join(current)
    if scanned > ARGS_MAX_BYTES:
        return "[TRUNCATED]", True
    if count == 0:
        return command, False
    return first if count == 1 else f"{first} + {count - 1}", False


def _args_summary_source(tool_id: str, arguments: Any) -> tuple[Any, bool]:
    """Select one human-useful field per known tool before redaction."""
    if not isinstance(arguments, Mapping):
        # 同 D1：非字典的 arguments 只有本身是字符串时才算「人读短语」，
        # list / 数字等一律空着，绝不落到 `_display_text` 渲染成 JSON。
        return (arguments.strip() if isinstance(arguments, str) else ""), False
    if tool_id == "read_file":
        path = arguments.get("path")
        if isinstance(path, str):
            return path.replace("\\", "/").rsplit("/", 1)[-1], False
    elif tool_id == "terminal":
        command = arguments.get("command")
        if isinstance(command, str):
            return _terminal_args_summary(command)
    elif tool_id.startswith("browser_"):
        url = arguments.get("url")
        if isinstance(url, str):
            try:
                host = urlsplit(url).hostname
            except ValueError:
                host = ""
            if host:
                return host, False
    elif tool_id in {"search_files", "nas_search"}:
        for key in ("query", "pattern"):
            value = arguments.get(key)
            if isinstance(value, str):
                return value, False
    # D1 兜底：其它工具（skill / delegation / 自定义 MCP …）取**第一个非空字符串型
    # 参数值**；一个都没有就返回空串，客户端据此不显示 chip。
    #
    # ⛔ 不要把整个 `arguments` 丢下去 —— 它会被 `_display_text` 渲染成原始 JSON 摆到
    # 用户面前（AC-1379：skill 行显示 `{"file_path":"","name":"command-execution"}`）。
    # 「做了什么」要么是人读短语，要么空着，没有第三种。
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        # 敏感键一律跳过，用既有的 `_SECRET_KEYS` 判据（HR9：这条判断只有一处）。
        # 按键名脱敏只在遍历 mapping 时生效，直接返回裸串会绕过它。
        if isinstance(key, str) and _SECRET_KEYS.search(key):
            continue
        text = value.strip()
        if not text:
            continue
        # 路径型字段与 read_file 同口径取 basename，别把整条绝对路径摆上屏。
        if key in {"path", "file_path", "filepath"}:
            return text.replace("\\", "/").rsplit("/", 1)[-1], False
        return text, False
    return "", False


def args_summary(arguments: Any, *, tool_id: str = "") -> dict[str, Any]:
    source, source_cut = _args_summary_source(tool_id, arguments)
    text, truncated, _ = _display_text(source, ARGS_MAX_BYTES)
    return {"args_summary": text, "truncated": source_cut or truncated}


# Bound JSON parsing and line inspection independently of the emitted summary.
_PARSE_MAX_CHARS = 64 * 1024
_LINE_WINDOW_CHARS = 4096


def _structured_prefix(value: str) -> tuple[bool, bool]:
    """Scan leading whitespace only within the JSON admission budget.

    Exhausting the window is an omission, never permission to treat the unseen
    suffix as display text. Do not lstrip/copy an unbounded original result.
    """
    for char in islice(value, _PARSE_MAX_CHARS):
        if not char.isspace():
            return char in {"{", "["}, False
    return False, len(value) > _PARSE_MAX_CHARS


def _decode_result(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        structured, omitted = _structured_prefix(value)
        if omitted:
            return None, True
        if not structured:
            return value, False
        if len(value) > _PARSE_MAX_CHARS:
            return None, True
        try:
            return json.loads(value), False
        except json.JSONDecodeError:
            # A bracket prefix alone is not JSON: logs and Markdown links are
            # ordinary text. Preserve them through the bounded sanitizer.
            return value, False
        except (ValueError, RecursionError):
            return None, True
    return value, False


def _flat_text(value: Any) -> tuple[str, bool]:
    """Select human text, never serialize a result tree back into JSON."""
    pieces: list[str] = []
    nodes = 0
    cut = False
    seen: set[int] = set()

    def visit(item: Any, depth: int = 0) -> None:
        nonlocal nodes, cut
        nodes += 1
        if nodes > 64 or depth > 8 or sum(map(len, pieces)) >= 1024:
            cut = True
            return
        if isinstance(item, str):
            structured, omitted = _structured_prefix(item)
            if omitted:
                cut = True
                return
            if structured:
                decoded, omitted = _decode_result(item)
                cut |= omitted
                if decoded is not None and not isinstance(decoded, str):
                    visit(decoded, depth + 1)
                    return
                if decoded is None:
                    return
            # Avoid cutting credential atoms before redaction. Inspect one
            # bounded atom; oversized strings safely omit the remainder.
            cut |= len(item) > 1024
            pieces.append(redact(item[:1024]))
        elif isinstance(item, Mapping):
            if id(item) in seen:
                cut = True
                return
            seen.add(id(item))
            for key, child in islice(item.items(), 64):
                if nodes >= 64:
                    cut = True
                    break
                if not isinstance(key, str) or len(key) > 256:
                    cut = True
                    continue
                if _SECRET_KEYS.search(key):
                    continue
                visit(child, depth + 1)
            cut |= len(item) > 64
            seen.remove(id(item))
        elif isinstance(item, (list, tuple)):
            if id(item) in seen:
                cut = True
                return
            seen.add(id(item))
            for child in islice(item, 64):
                if nodes >= 64:
                    cut = True
                    break
                visit(child, depth + 1)
            cut |= len(item) > 64
            seen.remove(id(item))
        elif item is None:
            return
        elif isinstance(item, (bool, float)) or isinstance(item, int) and item.bit_length() <= 64:
            pieces.append(str(item))
        else:
            cut = True

    visit(value)
    text, clipped, _ = _truncate_utf8(" • ".join(pieces), 512)
    return text, cut or clipped


def _preferred_text(value: Any) -> tuple[str, bool]:
    if isinstance(value, Mapping):
        for field in ("summary", "message", "error", "text"):
            selected = value.get(field)
            if selected is not None and selected != "":
                return _flat_text(selected)
    return _flat_text(value)


def _lines(value: Any, count: int, *, tail: bool = False) -> tuple[list[str], bool]:
    if not isinstance(value, str):
        text, cut = _preferred_text(value)
        return [text], cut
    structured, omitted = _structured_prefix(value)
    if omitted:
        return ["Structured result omitted"], True
    if structured:
        decoded, cut = _decode_result(value)
        if not isinstance(decoded, str):
            text, clipped = _preferred_text(decoded)
            return [text or "Structured result"], cut or clipped
    cut = len(value) > _LINE_WINDOW_CHARS
    window = value[-_LINE_WINDOW_CHARS:] if tail else value[:_LINE_WINDOW_CHARS]
    lines = window.splitlines()
    if cut:
        # Drop the partial boundary line, so a sliced credential cannot lose
        # its label and escape the sanitizer.
        lines = lines[1:] if tail else lines[:-1]
    cut |= len(lines) > count
    selected = lines[-count:] if tail else lines[:count]
    result = []
    for line in selected:
        if len(line) > SUMMARY_MAX_BYTES:
            result.append("[TRUNCATED]")
            cut = True
        else:
            result.append(redact(line))
    return result, cut


def _derive_summary(value: Any, tool_id: str, arguments: Any) -> tuple[str, bool]:
    row = value if isinstance(value, Mapping) else {}
    args = arguments if isinstance(arguments, Mapping) else {}
    codex_command = tool_id == "exec_command"
    codex_patch = tool_id == "apply_patch"
    tool_id = {"exec_command": "terminal", "apply_patch": "patch"}.get(tool_id, tool_id)
    if codex_command and isinstance(value, str):
        exit_match = re.match(r"\[exit (-?\d{1,12})\]\n", value[:64])
        row = {"exit_code": exit_match.group(1) if exit_match else "unavailable",
               "output": value}
    if tool_id == "terminal":
        code, cut = _flat_text(row.get("exit_code", "unknown"))
        lines, clipped = _lines(row.get("output", row.get("stdout", value)), 20, tail=True)
        stderr, stderr_cut = _lines(row.get("stderr", ""), 20, tail=True)
        combined = (lines + stderr)[-20:]
        return "Exit code: " + code + "\n" + "\n".join(combined), cut or clipped or stderr_cut or len(lines + stderr) > 20
    if tool_id == "read_file":
        path, cut = _flat_text(args.get("path", row.get("path", "File")))
        total, a = _flat_text(row.get("total_lines", "unknown"))
        size, b = _flat_text(row.get("file_size", "unknown"))
        lines, c = _lines(row.get("content", ""), 5)
        return f"File: {path}\nLines: {total}; Bytes: {size}\n" + "\n".join(lines), cut or a or b or c
    if tool_id == "patch":
        if codex_patch:
            changes = args.get("changes", [])
            paths = [change.get("path", "") for change in islice(changes, 64) if isinstance(change, Mapping)] if isinstance(changes, (list, tuple)) else []
            names, cut = _flat_text(paths)
            status, clipped = _preferred_text(value)
            # Codex completion omits the diff. Do not fabricate +0/-0 counts.
            return f"Files: {names or 'unavailable'}\nChanges: counts unavailable\n{status}", cut or clipped
        files = [row.get(key, []) for key in ("files_modified", "files_created", "files_deleted")]
        names, cut = _flat_text(files if any(files) else args.get("path", "File"))
        diff = row.get("diff", "")
        if not isinstance(diff, str) or len(diff) > _PARSE_MAX_CHARS:
            return f"Files: {names}\nChanges: counts unavailable", True
        added = removed = 0
        # No full split/copy of the diff: inspect at most the bounded source.
        import io
        for line in io.StringIO(diff):
            added += line.startswith("+") and not line.startswith("+++")
            removed += line.startswith("-") and not line.startswith("---")
        return f"Files: {names}\nChanges: +{added} -{removed}", cut
    if tool_id in {"search", "search_files", "nas_search"}:
        total, cut = _flat_text(row.get("total_count", row.get("total", "unknown")))
        titles = []
        matches = row.get("matches", row.get("results", row.get("files", [])))
        if isinstance(matches, (list, tuple)):
            cut |= len(matches) > 3
            for match in islice(matches, 3):
                title = match.get("title", match.get("path", match.get("content", ""))) if isinstance(match, Mapping) else match
                text, clipped = _flat_text(title)
                titles.append(text)
                cut |= clipped
        if not titles and isinstance(row.get("matches_text"), str):
            lines, clipped = _lines(row["matches_text"], 12)
            path = ""
            for line in lines:
                if line.startswith("  "):
                    titles.append(f"{path}: {line.strip()}")
                    if len(titles) == 3:
                        break
                else:
                    path = line
            cut |= clipped or len(lines) > len(titles) + 1

        return f"Matches: {total}\n" + "\n".join(titles), cut
    if tool_id == "todo" and isinstance(row.get("summary"), Mapping):
        counts = row["summary"]
        fields = []
        cut = False
        for field in ("total", "completed", "pending", "in_progress", "cancelled"):
            text, clipped = _flat_text(counts.get(field, 0))
            fields.append(f"{field}: {text}")
            cut |= clipped
        action = "write" if args.get("todos") is not None else "read"
        return f"Action: {action} tasks\n" + "; ".join(fields), cut
    if tool_id == "skill_view" and row.get("name"):
        name, cut = _flat_text(row["name"])
        status, clipped = _flat_text(row.get("readiness_status", "loaded"))
        return f"Action: view skill {name}\nResult: {status}", cut or clipped
    if tool_id == "app_host" and isinstance(row.get("data"), Mapping):
        data = row["data"]
        selected = next((data[key] for key in ("summary", "message", "error", "text", "outcome", "state") if data.get(key) is not None), "Completed" if row.get("ok") else "Result unavailable")
        text, cut = _preferred_text(selected)
    else:
        text, cut = _preferred_text(value)
    if tool_id in {"app_host", "skill_view", "todo"}:
        action, clipped = _flat_text(args.get("action", tool_id))
        return f"Action: {action}\nResult: {text}", cut or clipped
    if isinstance(value, (bool, int, float)):
        return text, cut
    return "Result: " + (text or "Structured result"), cut


def result_display(output: Any = None, *, error: Any = None, content_type: str | None = None,
                   tool_id: str = "", arguments: Any = None) -> dict[str, Any]:
    is_error = error is not None and not (isinstance(error, str) and error == "")
    value, omitted = _decode_result(error if is_error else output)
    if isinstance(value, Mapping):
        is_error |= bool(value.get("error"))
    if is_error:
        text, cut = _preferred_text(value)
        summary = "Error: " + (text or "Tool failed")
    else:
        summary, cut = _derive_summary(value, tool_id, arguments)
    summary, clipped, _ = _truncate_utf8(summary, SUMMARY_MAX_BYTES)
    kind = "error" if is_error or content_type == "error" else "markdown" if content_type == "markdown" else "text"
    display = {"summary": summary, "content_type": kind, "truncated": omitted or cut or clipped}
    original = error if error is not None and error != "" else output
    if isinstance(original, str):
        try:
            # D1 requires original byte size. Measure without a whole-result
            # allocation/encode; this is O(n) counting with constant memory.
            display["bytes"] = sum(len(original[i:i + 1024].encode("utf-8", "strict"))
                                   for i in range(0, len(original), 1024))
        except UnicodeError:
            pass  # No valid UTF-8 source size exists; optional bytes is absent.
    # A mapping has no original wire encoding. Do not invent one for counting.
    return display


def build_tool_start_display(tool_id: str, arguments: Any, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"source": source_from_registration(tool_id, registration), "display": args_summary(arguments, tool_id=tool_id)}


def build_tool_result_display(output: Any = None, *, error: Any = None, content_type: str | None = None, tool_id: str = "", arguments: Any = None) -> dict[str, Any]:
    return {"display": result_display(output, error=error, content_type=content_type, tool_id=tool_id, arguments=arguments)}
