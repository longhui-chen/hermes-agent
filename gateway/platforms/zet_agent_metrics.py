"""Bounded Hermes-owned interaction counters."""
from __future__ import annotations

import threading
from collections import Counter

_LOCK = threading.Lock()
_COUNTERS: Counter[str] = Counter()
_METRIC_NAMES = frozenset({"interaction_opened", "interaction_answered", "interaction_terminal", "clarify_rejected", "item_frame_dropped_backlog"})
_TERMINAL_STATES = frozenset({"expired", "cancelled", "runtime_lost"})
_REJECTION_REASONS = frozenset({"caller_inactive", "other"})


def _increment(name: str, label: str | None = None) -> None:
    if name not in _METRIC_NAMES:
        return
    if name == "interaction_terminal":
        key = f"{name}{{source=hermes,state={label or 'other'}}}"
    elif name == "clarify_rejected":
        key = f"{name}{{reason={label or 'other'}}}"
    elif name == "item_frame_dropped_backlog":
        key = f"{name}{{kind={label or 'other'}}}"
    else:
        key = name
    with _LOCK:
        _COUNTERS[key] += 1


def interaction_opened() -> None:
    _increment("interaction_opened")


def interaction_answered() -> None:
    _increment("interaction_answered")


def interaction_terminal(state: str) -> None:
    _increment("interaction_terminal", state if state in _TERMINAL_STATES else "other")


def clarify_rejected(reason: str) -> None:
    _increment("clarify_rejected", reason if reason in _REJECTION_REASONS else "other")


def item_frame_dropped_backlog(kind: str) -> None:
    _increment("item_frame_dropped_backlog", kind if kind in {"attachment", "subagent"} else "other")


def snapshot() -> dict[str, int]:
    with _LOCK:
        return dict(_COUNTERS)


def reset_for_tests() -> None:
    with _LOCK:
        _COUNTERS.clear()
