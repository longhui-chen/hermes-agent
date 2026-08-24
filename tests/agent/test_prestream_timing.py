import json
import logging
from typing import Any

import pytest

from agent.prestream_timing import PrestreamTiming


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance_ms(self, milliseconds: int) -> None:
        self.now += milliseconds / 1000


class _CapturingLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.records.append(message % args)


def _payload(logger: _CapturingLogger) -> dict[str, Any]:
    assert len(logger.records) == 1
    prefix, raw = logger.records[0].split(" ", 1)
    assert prefix == "hermes.prestream.turn"
    return json.loads(raw)


@pytest.mark.parametrize(
    "kind", ["reasoning", "content", "tool_start", "attachment"]
)
def test_first_semantic_write_emits_once_for_supported_kinds(kind: str) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(
        logger=logger,
        clock=clock,
        session_id="session-1",
        turn_id="turn-1",
        explicit_skill=False,
    )
    timing.history_ready(source="request", count=2)
    timing.executor_queued()
    timing.executor_started()
    timing.agent_init_started()
    timing.agent_init_finished()

    semantic = timing.semantic_observed(kind)
    clock.advance_ms(7)
    timing.public_write_completed(semantic)
    timing.public_write_completed(timing.semantic_observed("content"))
    timing.terminal_write_completed()

    payload = _payload(logger)
    assert payload["first_event_kind"] == kind
    assert payload["semantic_to_sse_write_ms"] == 7
    assert payload["timing_status"] == "complete"


def test_nonsemantic_writes_do_not_complete_summary() -> None:
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger)

    for kind in ("role", "keepalive", "usage", "provider_metadata", None):
        timing.public_write_completed(timing.semantic_observed(kind))

    assert logger.records == []


def test_terminal_without_semantic_emits_none_once() -> None:
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, session_id="session-1")

    timing.terminal_write_completed()
    timing.terminal_write_completed()

    payload = _payload(logger)
    assert payload["first_event_kind"] == "none"
    assert payload["timing_status"] == "partial"
    assert "first_public_semantic" in payload["missing_stages"]
    assert "ingress_to_first_public_ms" not in payload
    assert "semantic_to_sse_write_ms" not in payload


def test_ten_thousand_chunks_keep_one_summary_and_constant_state() -> None:
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger)

    for _ in range(10_000):
        timing.public_write_completed(timing.semantic_observed("content"))

    assert len(logger.records) == 1
    assert not hasattr(timing, "events")
    assert not hasattr(timing, "chunks")


def test_stage_delays_only_populate_their_own_duration_fields() -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(
        logger=logger,
        clock=clock,
        explicit_skill=True,
    )

    clock.advance_ms(5)
    timing.history_ready(source="session_db", count=3)
    timing.skill_expand_started()
    clock.advance_ms(7)
    timing.skill_expand_finished()
    timing.executor_queued()
    clock.advance_ms(11)
    timing.executor_started()
    timing.agent_init_started()
    clock.advance_ms(13)
    timing.agent_init_finished()
    semantic = timing.semantic_observed("content")
    clock.advance_ms(17)
    timing.public_write_completed(semantic)

    payload = _payload(logger)
    assert payload["ingress_to_history_ready_ms"] == 5
    assert payload["skill_expand_ms"] == 7
    assert payload["skill_expand_outcome"] == "success"
    assert payload["executor_queue_ms"] == 11
    assert payload["agent_init_ms"] == 13
    assert payload["ingress_to_first_public_ms"] == 53
    assert payload["semantic_to_sse_write_ms"] == 17
    assert payload["history_source"] == "session_db"
    assert payload["history_count"] == 3
    assert payload["explicit_skill"] is True


def test_history_and_agent_init_delay_changes_are_isolated() -> None:
    def render(*, history_ms: int, init_ms: int) -> dict[str, Any]:
        clock = _Clock()
        logger = _CapturingLogger()
        timing = PrestreamTiming(logger=logger, clock=clock)
        clock.advance_ms(history_ms)
        timing.history_ready(source="request", count=1)
        timing.executor_queued()
        timing.executor_started()
        timing.agent_init_started()
        clock.advance_ms(init_ms)
        timing.agent_init_finished()
        timing.public_write_completed(timing.semantic_observed("content"))
        return _payload(logger)

    slow_history = render(history_ms=40, init_ms=5)
    slow_init = render(history_ms=5, init_ms=40)

    assert slow_history["ingress_to_history_ready_ms"] == 40
    assert slow_history["agent_init_ms"] == 5
    assert slow_init["ingress_to_history_ready_ms"] == 5
    assert slow_init["agent_init_ms"] == 40


def test_unobserved_stages_are_omitted_and_marked_partial_never_zero_filled() -> None:
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, explicit_skill=True)

    timing.public_write_completed(timing.semantic_observed("content"))

    payload = _payload(logger)
    assert payload["timing_status"] == "partial"
    assert set(payload["missing_stages"]) >= {
        "history_ready",
        "skill_expand",
        "executor_queue",
        "agent_init",
    }
    for field in (
        "ingress_to_history_ready_ms",
        "skill_expand_ms",
        "executor_queue_ms",
        "agent_init_ms",
    ):
        assert field not in payload


def test_history_session_db_error_has_outcome_without_success_duration() -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, clock=clock)
    clock.advance_ms(23)

    timing.history_failed(source="session_db", outcome="session_db_error")
    timing.public_write_completed(timing.semantic_observed("content"))

    payload = _payload(logger)
    assert payload["history_source"] == "session_db"
    assert payload["history_outcome"] == "session_db_error"
    assert "ingress_to_history_ready_ms" not in payload
    assert "history_count" not in payload
    assert "history_ready:session_db_error" in payload["missing_stages"]


def test_cancelled_skill_waits_for_worker_settlement_without_success_duration() -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, clock=clock, explicit_skill=True)
    timing.skill_expand_started()
    clock.advance_ms(19)
    timing.skill_expand_completed("cancelled")

    timing.public_write_completed(timing.semantic_observed("content"))
    payload = _payload(logger)
    assert payload["skill_expand_outcome"] == "cancelled"
    assert "skill_expand_ms" not in payload
    assert "skill_expand:cancelled" in payload["missing_stages"]

    timing.skill_expand_settled()
    assert len(logger.records) == 1


def test_skill_expand_error_has_outcome_without_success_duration() -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, clock=clock, explicit_skill=True)
    timing.skill_expand_started()
    clock.advance_ms(11)
    timing.skill_expand_settled()
    timing.skill_expand_completed("error")

    timing.terminal_write_completed()

    payload = _payload(logger)
    assert payload["skill_expand_outcome"] == "error"
    assert "skill_expand_ms" not in payload
    assert "skill_expand:error" in payload["missing_stages"]


def test_agent_init_error_has_outcome_without_success_duration() -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(logger=logger, clock=clock)
    timing.agent_init_started()
    clock.advance_ms(31)
    timing.agent_init_finished("error")

    timing.terminal_write_completed()

    payload = _payload(logger)
    assert payload["agent_init_outcome"] == "error"
    assert "agent_init_ms" not in payload
    assert "agent_init:error" in payload["missing_stages"]


def test_correlation_ids_are_bounded_and_sensitive_values_are_never_logged() -> None:
    logger = _CapturingLogger()
    timing = PrestreamTiming(
        logger=logger,
        session_id="s" * 129,
        turn_id="turn-safe",
        explicit_skill=False,
    )

    timing.public_write_completed(timing.semantic_observed("content"))

    rendered = logger.records[0]
    payload = _payload(logger)
    assert "session_id" not in payload
    assert payload["turn_id"] == "turn-safe"
    for forbidden in (
        "user_message",
        "preview",
        "reasoning_text",
        "tool_args",
        "tool_result",
        "api_key",
        "provider",
        "model",
        "url",
        "owner",
    ):
        assert forbidden not in rendered.lower()


def test_logger_and_clock_failures_are_fail_open() -> None:
    class _FailingLogger:
        def info(self, *_args: object) -> None:
            raise RuntimeError("logging unavailable")

    calls = 0

    def flaky_clock() -> float:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("clock unavailable")
        return 1.0

    timing = PrestreamTiming(logger=_FailingLogger(), clock=flaky_clock)

    timing.history_ready(source="request", count=1)
    timing.public_write_completed(timing.semantic_observed("content"))
    timing.terminal_write_completed()


def test_real_zettos_formatter_contains_searchable_event_without_secret() -> None:
    from agent.redact import RedactingFormatter
    from hermes_logging import ZettosJSONFormatter

    secret = "sk-secret-must-not-appear"
    stream = __import__("io").StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        ZettosJSONFormatter(
            RedactingFormatter("%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s")
        )
    )
    logger = logging.getLogger("agent.prestream_timing.formatter_test")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        timing = PrestreamTiming(logger=logger, session_id="session-safe")
        timing.public_write_completed(timing.semantic_observed("content"))
    finally:
        logger.handlers = []
        logger.propagate = True

    record = json.loads(stream.getvalue())
    msg = record["Attributes"]["msg"]
    assert msg.startswith("hermes.prestream.turn ")
    assert secret not in stream.getvalue()
