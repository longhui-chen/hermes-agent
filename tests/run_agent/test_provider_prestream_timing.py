from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agent.prestream_timing import PRESTREAM_TIMING_CONTEXT, PrestreamTiming


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
    return json.loads(logger.records[0].split(" ", 1)[1])


def _response(text: str = "answer") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                    tool_calls=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="test-model",
    )


def _stream_chunk(*, content: str | None = None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        model="test-model",
        usage=None,
    )


def _build_agent(tmp_path, monkeypatch, *, streaming: bool):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="test-model",
        api_key="sk-dummy",
        base_url="https://example.invalid/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_tool_loading=True,
        config_context_length=256_000,
        platform="zet_agent",
        stream_delta_callback=(lambda _delta: None) if streaming else None,
    )
    agent._disable_streaming = not streaming
    return agent


@pytest.mark.parametrize("streaming", [False, True])
def test_conversation_loop_observes_provider_dispatch_and_first_semantic(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(clock=clock, logger=logger)
    agent = _build_agent(tmp_path, monkeypatch, streaming=streaming)
    if streaming:
        agent.stream_delta_callback = (
            lambda delta: timing.observe_queued_semantic("content")
            if delta
            else None
        )
    request_client = MagicMock()

    def _create(**_kwargs):
        clock.advance_ms(41)
        if streaming:
            return iter([
                _stream_chunk(content="answer"),
                _stream_chunk(finish_reason="stop"),
            ])
        return _response()

    request_client.chat.completions.create.side_effect = _create
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **_kwargs: request_client,
    )
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_args, **_kwargs: None)

    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        result = agent.run_conversation("private prompt")
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)
    assert result["final_response"] == "answer"

    timing.observe_queued_semantic("content")
    semantic = timing.semantic_observed("content")
    timing.public_write_completed(semantic)
    payload = _payload(logger)
    assert payload["provider_dispatch_count"] == 1
    assert payload["provider_wait_ms"] == 41


def test_invalid_nonstream_response_counts_retry_before_first_semantic(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(clock=clock, logger=logger)
    agent = _build_agent(tmp_path, monkeypatch, streaming=False)
    responses = [SimpleNamespace(choices=[]), _response()]
    request_client = MagicMock()

    def _create(**_kwargs):
        clock.advance_ms(23)
        return responses.pop(0)

    request_client.chat.completions.create.side_effect = _create
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **_kwargs: request_client,
    )
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "agent.conversation_loop.jittered_backoff",
        lambda *_args, **_kwargs: 0.0,
    )

    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        result = agent.run_conversation("private prompt")
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)
    assert result["final_response"] == "answer"
    timing.observe_queued_semantic("content")
    timing.public_write_completed(timing.semantic_observed("content"))
    payload = _payload(logger)
    assert payload["provider_dispatch_count"] == 2
    assert payload["provider_wait_ms"] == 46


def test_internal_stream_retry_counts_each_physical_provider_attempt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(clock=clock, logger=logger)
    agent = _build_agent(tmp_path, monkeypatch, streaming=True)
    attempts = 0

    def _create(**_kwargs):
        nonlocal attempts
        attempts += 1
        clock.advance_ms(11)
        if attempts == 1:
            raise httpx.ReadTimeout("retry", request=httpx.Request("POST", "https://example.invalid"))
        return iter([
            _stream_chunk(content="answer"),
            _stream_chunk(finish_reason="stop"),
        ])

    request_client = MagicMock()
    request_client.chat.completions.create.side_effect = _create
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **_kwargs: request_client,
    )
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    agent.stream_delta_callback = (
        lambda delta: timing.observe_queued_semantic("content")
        if delta
        else None
    )

    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        response = agent._interruptible_streaming_api_call({})
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)

    assert response.choices[0].message.content == "answer"
    timing.public_write_completed(timing.semantic_classified("content"))
    payload = _payload(logger)
    assert payload["provider_dispatch_count"] == 2
    assert payload["provider_wait_ms"] == 22


def test_codex_stream_retry_counts_each_physical_provider_attempt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(clock=clock, logger=logger)
    agent = _build_agent(tmp_path, monkeypatch, streaming=True)
    agent.api_mode = "codex_responses"
    attempts = 0
    completed = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="answer")],
            )
        ],
        usage=None,
        model="test-model",
    )

    def _create(**_kwargs):
        nonlocal attempts
        attempts += 1
        clock.advance_ms(17)
        if attempts == 1:
            raise httpx.ConnectError(
                "retry",
                request=httpx.Request("POST", "https://example.invalid"),
            )
        return iter(
            [
                SimpleNamespace(type="response.output_text.delta", delta="answer"),
                SimpleNamespace(type="response.completed", response=completed),
            ]
        )

    agent.client = SimpleNamespace(responses=SimpleNamespace(create=_create))
    agent.stream_delta_callback = (
        lambda delta: timing.observe_queued_semantic("content")
        if delta
        else None
    )

    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        response = agent._run_codex_stream({"model": "test-model", "input": []})
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)

    assert response.status == "completed"
    timing.public_write_completed(timing.semantic_classified("content"))
    payload = _payload(logger)
    assert payload["provider_dispatch_count"] == 2
    assert payload["provider_wait_ms"] == 34


def test_moa_facade_marks_composite_dispatch_scope(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    logger = _CapturingLogger()
    timing = PrestreamTiming(clock=clock, logger=logger)
    agent = _build_agent(tmp_path, monkeypatch, streaming=False)
    agent.provider = "moa"
    agent.client = MagicMock()

    def _create(**_kwargs):
        clock.advance_ms(37)
        return _response()

    agent.client.chat.completions.create.side_effect = _create
    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        response = agent._interruptible_api_call({})
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)

    assert response.choices[0].message.content == "answer"
    timing.observe_queued_semantic("content")
    timing.public_write_completed(timing.semantic_classified("content"))
    payload = _payload(logger)
    assert payload["provider_dispatch_scope"] == "composite"
    assert payload["provider_dispatch_count"] == 1
    assert payload["provider_wait_ms"] == 37


def test_provider_timing_observer_failure_never_changes_the_turn(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _build_agent(tmp_path, monkeypatch, streaming=False)

    class _FailingTiming(PrestreamTiming):
        def provider_dispatch_started(self) -> None:
            raise RuntimeError("observer unavailable")

    request_client = MagicMock()
    request_client.chat.completions.create.return_value = _response()
    monkeypatch.setattr(
        agent,
        "_create_request_openai_client",
        lambda **_kwargs: request_client,
    )
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *_args, **_kwargs: None)

    token = PRESTREAM_TIMING_CONTEXT.set(_FailingTiming())
    try:
        result = agent.run_conversation("private prompt")
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)

    assert result["final_response"] == "answer"
