"""Bounded Hermes-owned interaction counters."""
from __future__ import annotations

import threading
from collections import Counter

_LOCK = threading.Lock()
_COUNTERS: Counter[str] = Counter()
_ITEM_FRAME_NAMES = frozenset({"item_frame_stranded", "item_frame_unregistered", "item_frame_unclassified"})
# Zero values are exported too, so one health sample can verify all T6 buckets.
_ITEM_FRAME_DEFAULTS = dict.fromkeys(
    (*sorted(_ITEM_FRAME_NAMES), *(f"item_frame_dropped_backlog{{kind={kind}}}"
                                 for kind in ("attachment", "subagent", "other"))),
    0,
)
_METRIC_NAMES = _ITEM_FRAME_NAMES | frozenset({"interaction_opened", "interaction_answered", "interaction_terminal", "clarify_rejected", "item_frame_dropped_backlog"})
_TERMINAL_STATES = frozenset({"expired", "cancelled", "runtime_lost"})
_REJECTION_REASONS = frozenset({"caller_inactive", "other"})


def _increment(name: str, label: str | None = None, count: int = 1) -> None:
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
        _COUNTERS[key] += count


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


def item_frame_count(name: str, count: int = 1) -> None:
    """Aggregate fixed BT buckets across requests; never label user/profile data."""
    if name in _ITEM_FRAME_NAMES and isinstance(count, int) and count > 0:
        _increment(name, count=count)


def snapshot() -> dict[str, int]:
    with _LOCK:
        return _ITEM_FRAME_DEFAULTS | dict(_COUNTERS)


def reset_for_tests() -> None:
    with _LOCK:
        _COUNTERS.clear()
