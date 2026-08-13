"""Portable validation for trusted deep-memory transport identity headers."""

from typing import Any


def bounded_identity_header(value: Any, max_bytes: int = 1024) -> str:
    text = str(value or "").strip()
    if (
        not text
        or "\x00" in text
        or len(text.encode("utf-8")) > max_bytes
        or any(ord(char) < 0x20 for char in text)
    ):
        return ""
    return text
