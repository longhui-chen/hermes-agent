# zettlab-overlay(H20-unowned): prestream 观测整文件为 fork 新增，收敛时迁出稳定层; upstream: none
"""Bounded, fail-open timing summary for one public streaming turn.

The observer intentionally owns only scalar milestones for one call stack.  It
never retains token/chunk payloads and never changes the stream it observes.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Callable, Optional


EVENT_NAME = "hermes.prestream.turn"
MAX_CORRELATION_ID_LENGTH = 128
_SEMANTIC_KINDS = frozenset({"reasoning", "content", "tool_start", "attachment"})
_PROVIDER_SEMANTIC_KINDS = frozenset({"reasoning", "content", "tool_start"})
_PROVIDER_DISPATCH_SCOPES = frozenset({"physical", "composite"})
_AGENT_SHELL_OUTCOMES = frozenset(
    {"created", "runtime_cache_hit", "onboarding_cache_hit", "error"}
)
_LOGGER = logging.getLogger(__name__)
PRESTREAM_TIMING_CONTEXT: ContextVar[Optional["PrestreamTiming"]] = ContextVar(
    "agent_prestream_timing", default=None
)


@dataclass(frozen=True, slots=True)
class PrestreamSemanticEvent:
    """Timestamp token created when a public semantic event is observed."""

    kind: str
    observed_at: Optional[float]


def _bounded_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or len(text) > MAX_CORRELATION_ID_LENGTH:
        return ""
    if any(ord(char) < 0x20 for char in text):
        return ""
    return text


def _duration_ms(start: Optional[float], end: Optional[float]) -> Optional[int]:
    if start is None or end is None or end < start:
        return None
    return int(round((end - start) * 1000))


def observe_provider_dispatch(*, scope: str = "physical") -> None:
    """Record one provider attempt on the current request, if any."""
    try:
        timing = PRESTREAM_TIMING_CONTEXT.get()
        if timing is not None:
            timing.provider_dispatch_started(scope=scope)
    except Exception:
        return


class PrestreamTiming:
    """Collect and emit exactly one bounded summary for a streaming turn."""

    __slots__ = (
        "_agent_init_finished_at",
        "_agent_init_outcome",
        "_agent_init_started_at",
        "_agent_shell_finished_at",
        "_agent_shell_outcome",
        "_agent_shell_started_at",
        "_clock",
        "_emitted",
        "_executor_queued_at",
        "_executor_started_at",
        "_explicit_skill",
        "_history_count",
        "_history_outcome",
        "_history_ready_at",
        "_history_source",
        "_ingress_at",
        "_lock",
        "_logger",
        "_observed_semantic_at",
        "_observed_semantic_kind",
        "_provider_dispatch_count",
        "_provider_dispatch_scope",
        "_provider_first_dispatch_at",
        "_provider_semantic_at",
        "_session_id",
        "_skill_finished_at",
        "_skill_outcome",
        "_skill_started_at",
        "_turn_id",
    )

    def __init__(
        self,
        *,
        logger: Any = _LOGGER,
        clock: Callable[[], float] = time.monotonic,
        session_id: object = "",
        turn_id: object = "",
        explicit_skill: bool = False,
        ingress_at: Optional[float] = None,
    ) -> None:
        self._logger = logger
        self._clock = clock
        self._lock = threading.Lock()
        self._session_id = _bounded_id(session_id)
        self._turn_id = _bounded_id(turn_id)
        self._explicit_skill = bool(explicit_skill)
        self._emitted = False
        self._history_source = ""
        self._history_outcome = ""
        self._history_count: Optional[int] = None
        self._history_ready_at: Optional[float] = None
        self._observed_semantic_kind = ""
        self._observed_semantic_at: Optional[float] = None
        self._provider_dispatch_count = 0
        self._provider_dispatch_scope = ""
        self._provider_first_dispatch_at: Optional[float] = None
        self._provider_semantic_at: Optional[float] = None
        self._skill_started_at: Optional[float] = None
        self._skill_finished_at: Optional[float] = None
        self._skill_outcome = ""
        self._executor_queued_at: Optional[float] = None
        self._executor_started_at: Optional[float] = None
        self._agent_init_started_at: Optional[float] = None
        self._agent_init_finished_at: Optional[float] = None
        self._agent_init_outcome = ""
        self._agent_shell_started_at: Optional[float] = None
        self._agent_shell_finished_at: Optional[float] = None
        self._agent_shell_outcome = ""
        self._ingress_at = ingress_at if ingress_at is not None else self._now()

    def _now(self) -> Optional[float]:
        try:
            return float(self._clock())
        except Exception:
            return None

    def history_ready(
        self,
        *,
        source: str,
        count: int,
        observed_at: Optional[float] = None,
    ) -> None:
        try:
            normalized_source = source if source in {"request", "session_db", "silent"} else "unknown"
            normalized_count = max(0, int(count))
            with self._lock:
                self._history_source = normalized_source
                self._history_outcome = (
                    "session_db_success"
                    if normalized_source == "session_db"
                    else "request"
                )
                self._history_count = normalized_count
                self._history_ready_at = (
                    observed_at if observed_at is not None else self._now()
                )
        except Exception:
            return

    def history_failed(self, *, source: str, outcome: str) -> None:
        try:
            normalized_source = source if source == "session_db" else "unknown"
            normalized_outcome = (
                outcome if outcome == "session_db_error" else "error"
            )
            with self._lock:
                self._history_source = normalized_source
                self._history_outcome = normalized_outcome
                self._history_count = None
                self._history_ready_at = None
        except Exception:
            return

    def skill_expand_started(self) -> None:
        self._set_timestamp("_skill_started_at")

    def skill_expand_finished(self) -> None:
        self.skill_expand_settled()
        self.skill_expand_completed("success")

    def skill_expand_settled(self) -> None:
        self._set_timestamp("_skill_finished_at")

    def skill_expand_completed(self, outcome: str) -> None:
        try:
            normalized = outcome if outcome in {"success", "error", "cancelled"} else "error"
            with self._lock:
                if self._skill_outcome in {"error", "cancelled"} and normalized == "success":
                    return
                self._skill_outcome = normalized
                if normalized != "success":
                    self._skill_finished_at = None
        except Exception:
            return

    def executor_queued(self) -> None:
        self._set_timestamp("_executor_queued_at")

    def executor_started(self) -> None:
        self._set_timestamp("_executor_started_at")

    def agent_init_started(self) -> None:
        self._set_timestamp("_agent_init_started_at")

    def agent_init_finished(self, outcome: str = "success") -> None:
        try:
            normalized = outcome if outcome in {"success", "error"} else "error"
            observed = self._now() if normalized == "success" else None
            with self._lock:
                self._agent_init_outcome = normalized
                self._agent_init_finished_at = observed
        except Exception:
            return

    def agent_shell_started(self) -> None:
        """Mark the cache lookup / constructor boundary inside agent init."""
        self._set_timestamp("_agent_shell_started_at")

    def agent_shell_finished(self, outcome: str) -> None:
        """Classify one bounded shell creation or cache-hit outcome."""
        try:
            normalized = outcome if outcome in _AGENT_SHELL_OUTCOMES else "error"
            observed = self._now() if normalized != "error" else None
            with self._lock:
                self._agent_shell_outcome = normalized
                self._agent_shell_finished_at = observed
        except Exception:
            return

    def provider_dispatch_started(self, *, scope: str = "physical") -> None:
        """Record one provider-bound execution and retain only its first start."""
        try:
            normalized_scope = (
                scope if scope in _PROVIDER_DISPATCH_SCOPES else "physical"
            )
            observed = self._now()
            with self._lock:
                if (
                    self._emitted
                    or self._provider_semantic_at is not None
                ):
                    return
                self._provider_dispatch_count += 1
                if not self._provider_dispatch_scope:
                    self._provider_dispatch_scope = normalized_scope
                elif self._provider_dispatch_scope != normalized_scope:
                    self._provider_dispatch_scope = "mixed"
                if self._provider_first_dispatch_at is None:
                    self._provider_first_dispatch_at = observed
        except Exception:
            return

    def _set_timestamp(self, name: str) -> None:
        try:
            observed = self._now()
            with self._lock:
                setattr(self, name, observed)
        except Exception:
            return

    def semantic_observed(self, kind: object) -> Optional[PrestreamSemanticEvent]:
        try:
            normalized = str(kind or "")
            if normalized not in _SEMANTIC_KINDS:
                return None
            observed_at = self._now()
            if observed_at is None:
                return None
            return PrestreamSemanticEvent(normalized, observed_at)
        except Exception:
            return None

    def observe_queued_semantic(self, kind: object) -> None:
        """Remember only the first queued semantic milestone for this turn."""
        try:
            normalized = str(kind or "")
            if normalized not in _SEMANTIC_KINDS:
                return
            observed_at = self._now()
            with self._lock:
                if self._emitted:
                    return
                if (
                    normalized in _PROVIDER_SEMANTIC_KINDS
                    and self._provider_semantic_at is None
                ):
                    self._provider_semantic_at = observed_at
                if not self._observed_semantic_kind:
                    self._observed_semantic_kind = normalized
                    self._observed_semantic_at = observed_at
        except Exception:
            return

    def semantic_classified(self, kind: object) -> Optional[PrestreamSemanticEvent]:
        """Classify a semantic write when its earlier observation is unknown."""
        try:
            normalized = str(kind or "")
            if normalized not in _SEMANTIC_KINDS:
                return None
            with self._lock:
                observed_at = (
                    self._observed_semantic_at
                    if normalized == self._observed_semantic_kind
                    else None
                )
            return PrestreamSemanticEvent(normalized, observed_at)
        except Exception:
            return None

    def public_write_completed(
        self, event: Optional[PrestreamSemanticEvent]
    ) -> None:
        try:
            if event is None or event.kind not in _SEMANTIC_KINDS:
                return
            completed_at = self._now()
            self._emit_once(
                first_event_kind=event.kind,
                first_public_at=completed_at,
                semantic_observed_at=event.observed_at,
            )
        except Exception:
            return

    def terminal_write_completed(self) -> None:
        try:
            self._emit_once(
                first_event_kind="none",
                first_public_at=None,
                semantic_observed_at=None,
            )
        except Exception:
            return

    def _emit_once(
        self,
        *,
        first_event_kind: str,
        first_public_at: Optional[float],
        semantic_observed_at: Optional[float],
    ) -> None:
        with self._lock:
            if self._emitted:
                return
            self._emitted = True
            payload = self._payload(
                first_event_kind=first_event_kind,
                first_public_at=first_public_at,
                semantic_observed_at=semantic_observed_at,
            )
        try:
            self._logger.info(
                "%s %s",
                EVENT_NAME,
                json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
            )
        except Exception:
            return

    def _payload(
        self,
        *,
        first_event_kind: str,
        first_public_at: Optional[float],
        semantic_observed_at: Optional[float],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "explicit_skill": self._explicit_skill,
            "first_event_kind": first_event_kind,
        }
        if self._session_id:
            payload["session_id"] = self._session_id
        if self._turn_id:
            payload["turn_id"] = self._turn_id
        if self._history_source:
            payload["history_source"] = self._history_source
        if self._history_outcome:
            payload["history_outcome"] = self._history_outcome
        if self._history_count is not None:
            payload["history_count"] = self._history_count
        if self._skill_outcome:
            payload["skill_expand_outcome"] = self._skill_outcome
        if self._agent_init_outcome:
            payload["agent_init_outcome"] = self._agent_init_outcome
        if self._agent_shell_outcome:
            payload["agent_shell_outcome"] = self._agent_shell_outcome
        if self._provider_dispatch_count:
            payload["provider_dispatch_count"] = self._provider_dispatch_count
            payload["provider_dispatch_scope"] = self._provider_dispatch_scope

        durations = {
            "ingress_to_history_ready_ms": _duration_ms(self._ingress_at, self._history_ready_at),
            "skill_expand_ms": (
                _duration_ms(self._skill_started_at, self._skill_finished_at)
                if self._skill_outcome == "success"
                else None
            ),
            "executor_queue_ms": _duration_ms(self._executor_queued_at, self._executor_started_at),
            "agent_init_ms": _duration_ms(self._agent_init_started_at, self._agent_init_finished_at),
            "agent_prepare_ms": _duration_ms(
                self._agent_init_started_at, self._agent_shell_started_at
            ),
            "agent_shell_ms": (
                _duration_ms(
                    self._agent_shell_started_at, self._agent_shell_finished_at
                )
                if self._agent_shell_outcome != "error"
                else None
            ),
            "agent_post_bind_ms": _duration_ms(
                self._agent_shell_finished_at, self._agent_init_finished_at
            ),
            "ingress_to_provider_dispatch_ms": _duration_ms(
                self._ingress_at, self._provider_first_dispatch_at
            ),
            "provider_wait_ms": _duration_ms(
                self._provider_first_dispatch_at, self._provider_semantic_at
            ),
            "ingress_to_first_public_ms": _duration_ms(self._ingress_at, first_public_at),
            "semantic_to_sse_write_ms": _duration_ms(semantic_observed_at, first_public_at),
        }
        for field, value in durations.items():
            if value is not None:
                payload[field] = value

        missing: list[str] = []
        if durations["ingress_to_history_ready_ms"] is None:
            missing.append(
                f"history_ready:{self._history_outcome}"
                if self._history_outcome
                else "history_ready"
            )
        if self._explicit_skill and durations["skill_expand_ms"] is None:
            missing.append(
                f"skill_expand:{self._skill_outcome}"
                if self._skill_outcome
                else "skill_expand"
            )
        if durations["executor_queue_ms"] is None:
            missing.append("executor_queue")
        if durations["agent_init_ms"] is None:
            missing.append(
                f"agent_init:{self._agent_init_outcome}"
                if self._agent_init_outcome
                else "agent_init"
            )
        if self._agent_shell_started_at is not None:
            if durations["agent_prepare_ms"] is None:
                missing.append("agent_prepare")
            if durations["agent_shell_ms"] is None:
                missing.append(
                    f"agent_shell:{self._agent_shell_outcome}"
                    if self._agent_shell_outcome
                    else "agent_shell"
                )
            if (
                self._agent_init_outcome == "success"
                and durations["agent_post_bind_ms"] is None
            ):
                missing.append("agent_post_bind")
        if self._provider_dispatch_count:
            if durations["ingress_to_provider_dispatch_ms"] is None:
                missing.append("provider_dispatch")
            if durations["provider_wait_ms"] is None:
                missing.append("provider_first_semantic")
        if first_event_kind == "none":
            missing.append("first_public_semantic")
        elif durations["ingress_to_first_public_ms"] is None:
            missing.append("ingress_to_first_public")
        if first_event_kind != "none" and durations["semantic_to_sse_write_ms"] is None:
            missing.append("semantic_to_sse_write")

        payload["timing_status"] = "partial" if missing else "complete"
        if missing:
            payload["missing_stages"] = missing
        return payload
