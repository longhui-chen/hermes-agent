"""Parse one systemd EnvironmentFile and emit NUL-delimited key/value pairs."""

from __future__ import annotations

import re
import sys
from pathlib import Path


_VALID_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_OUTSIDE_WHITESPACE = " \t"
_NEWLINES = "\n\r"
_MAX_ENV_FILE_BYTES = 64 * 1024


def _skip_to_next_line(text: str, index: int) -> int:
    newline_indexes = [
        found
        for marker in _NEWLINES
        if (found := text.find(marker, index)) >= 0
    ]
    if not newline_indexes:
        return len(text)
    newline = min(newline_indexes)
    if text[newline : newline + 2] == "\r\n":
        return newline + 2
    return newline + 1


def _skip_comment(text: str, index: int) -> int:
    escaped = False
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            if char in _NEWLINES:
                index += 1
            else:
                index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if char in _NEWLINES:
            return _skip_to_next_line(text, index)
        index += 1
    return index


def _parse_value(text: str, index: int) -> tuple[str, int]:
    value: list[str] = []
    significant_length = 0
    state = "pre"
    while index < len(text):
        char = text[index]

        if state == "pre":
            if char in _OUTSIDE_WHITESPACE:
                index += 1
                continue
            if char in _NEWLINES:
                return "".join(value), _skip_to_next_line(text, index)
            if char == "'":
                state = "single"
                index += 1
                continue
            if char == '"':
                state = "double"
                index += 1
                continue
            state = "unquoted"
            continue

        if state == "unquoted":
            if char in _NEWLINES:
                return "".join(value[:significant_length]), _skip_to_next_line(
                    text, index
                )
            if char == "\\":
                significant_length = len(value)
                if index + 1 >= len(text):
                    return "".join(value), len(text)
                escaped = text[index + 1]
                index += 2
                if escaped in _NEWLINES:
                    continue
                value.append(escaped)
                significant_length = len(value)
                continue
            value.append(char)
            index += 1
            if char not in _OUTSIDE_WHITESPACE:
                significant_length = len(value)
            continue

        if state == "single":
            if char == "'":
                state = "pre"
                significant_length = len(value)
                index += 1
                continue
            value.append(char)
            significant_length = len(value)
            index += 1
            continue

        if state == "double":
            if char == '"':
                state = "pre"
                significant_length = len(value)
                index += 1
                continue
            if char == "\\":
                if index + 1 >= len(text):
                    return "".join(value), len(text)
                escaped = text[index + 1]
                index += 2
                if escaped == "\n":
                    continue
                if escaped in {'"', "\\", "`", "$"}:
                    value.append(escaped)
                else:
                    value.extend(("\\", escaped))
                significant_length = len(value)
                continue
            value.append(char)
            significant_length = len(value)
            index += 1
            continue

    if state == "unquoted":
        return "".join(value[:significant_length]), index
    return "".join(value), index


def _parse_environment_file(
    text: str,
) -> tuple[dict[str, str], list[tuple[str, int, int]]]:
    if "\0" in text or "\ufeff" in text:
        raise ValueError("NUL and byte-order-mark characters are not allowed")
    if any(
        0xFDD0 <= ord(char) <= 0xFDEF or ord(char) & 0xFFFF in {0xFFFE, 0xFFFF}
        for char in text
    ):
        raise ValueError("Unicode noncharacters are not allowed")

    values: dict[str, str] = {}
    assignments: list[tuple[str, int, int]] = []
    index = 0
    while index < len(text):
        record_start = index
        while index < len(text) and text[index] in _OUTSIDE_WHITESPACE:
            index += 1
        if index >= len(text):
            break
        if text[index] in _NEWLINES:
            index = _skip_to_next_line(text, index)
            continue
        if text[index] in {"#", ";"}:
            index = _skip_comment(text, index)
            continue

        separator = text.find("=", index)
        newline_indexes = [
            found
            for marker in _NEWLINES
            if (found := text.find(marker, index)) >= 0
        ]
        newline = min(newline_indexes) if newline_indexes else -1
        if separator < 0 or (newline >= 0 and newline < separator):
            index = _skip_to_next_line(text, index)
            continue
        key = text[index:separator].strip(_OUTSIDE_WHITESPACE)
        index = separator + 1
        while index < len(text) and text[index] in _OUTSIDE_WHITESPACE:
            index += 1
        value, index = _parse_value(text, index)
        if not _VALID_KEY.fullmatch(key):
            continue
        values[key] = value
        assignments.append((key, record_start, index))
    return values, assignments


def parse_environment_file(text: str) -> dict[str, str]:
    values, _assignments = _parse_environment_file(text)
    return values


def filter_environment_file(text: str, excluded_keys: set[str]) -> str:
    _values, assignments = _parse_environment_file(text)
    output: list[str] = []
    cursor = 0
    for key, start, end in assignments:
        if key not in excluded_keys:
            continue
        output.append(text[cursor:start])
        cursor = end
    output.append(text[cursor:])
    filtered = "".join(output)
    if filtered and not filtered.endswith(tuple(_NEWLINES)):
        filtered += "\n"
    return filtered


def main() -> int:
    filter_mode = len(sys.argv) >= 4 and sys.argv[1] == "--filter-excluding"
    if not filter_mode and len(sys.argv) != 2:
        print(
            f"usage: {Path(sys.argv[0]).name} "
            "[--filter-excluding ENV_FILE KEY ...] ENV_FILE",
            file=sys.stderr,
        )
        return 2

    path = Path(sys.argv[2] if filter_mode else sys.argv[1])
    try:
        with path.open("rb") as env_file:
            raw = env_file.read(_MAX_ENV_FILE_BYTES + 1)
        if len(raw) > _MAX_ENV_FILE_BYTES:
            raise ValueError(
                f"file exceeds {_MAX_ENV_FILE_BYTES}-byte safety limit"
            )
        text = raw.decode("utf-8")
        values, _assignments = _parse_environment_file(text)
    except Exception as exc:
        print(f"invalid environment file {path}: {exc}", file=sys.stderr)
        return 1

    if filter_mode:
        excluded_keys = set(sys.argv[3:])
        if not excluded_keys or any(
            not _VALID_KEY.fullmatch(key) for key in excluded_keys
        ):
            print("invalid excluded environment key", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(
            filter_environment_file(text, excluded_keys).encode("utf-8")
        )
        return 0

    output = sys.stdout.buffer
    for key, value in values.items():
        output.write(key.encode("utf-8") + b"\0")
        output.write(value.encode("utf-8") + b"\0")
    output.write(b"\0\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
