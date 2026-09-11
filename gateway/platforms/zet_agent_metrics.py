"""Bounded Hermes-owned interaction counters."""
from __future__ import annotations

import threading
from collections import Counter
from typing import Any

_LOCK = threading.Lock()
_COUNTERS: Counter[str] = Counter()
_TERMINAL_STATES = frozenset({"expired", "cancelled", "runtime_lost"})
_REJECTION_REASONS = frozenset({"caller_inactive", "other"})


def increment(name: str, label: str | None = None) -> None:
    key = name if label is None else f"{name}{{{label}}}"
    with _LOCK:
        _COUNTERS[key] += 1


def interaction_opened() -> None:
    increment("interaction_opened")


def interaction_answered() -> None:
    increment("interaction_answered")


def interaction_terminal(state: str) -> None:
    increment("interaction_terminal", state if state in _TERMINAL_STATES else "other")


def clarify_rejected(reason: str) -> None:
    increment("clarify_rejected", reason if reason in _REJECTION_REASONS else "other")


def snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_COUNTERS)


def reset_for_tests() -> None:
    with _LOCK:
        _COUNTERS.clear()
