import asyncio
import json
import logging
import queue
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from agent.prestream_timing import PrestreamSemanticEvent, PrestreamTiming
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    return request


def _response() -> tuple[AsyncMock, list[bytes]]:
    chunks: list[bytes] = []
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()
    response.write = AsyncMock(side_effect=lambda payload: chunks.append(payload))
    return response, chunks


async def _completed_agent(final_response: str = "") -> tuple[dict, dict]:
    return (
        {"final_response": final_response, "completed": True},
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


class _CountingTerminalTiming(PrestreamTiming):
    __slots__ = ("terminal_calls",)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.terminal_calls = 0

    def terminal_write_completed(self) -> None:
        self.terminal_calls += 1
        super().terminal_write_completed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("queue_item", "expected_kind"),
    [
        ("answer", "content"),
        (("__tool_progress__", {"type": "reasoning.delta", "text": "private reasoning"}), "reasoning"),
        (("__tool_progress__", {"toolCallId": "call-1", "status": "running"}), "tool_start"),
    ],
)
async def test_writer_summarizes_first_public_semantic_after_successful_write(
    caplog: pytest.LogCaptureFixture,
    queue_item: object,
    expected_kind: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    stream_q: queue.Queue = queue.Queue()
    stream_q.put(queue_item)
    stream_q.put(None)
    response, chunks = _response()
    timing = PrestreamTiming(session_id="session-1", turn_id="turn-1")

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        await adapter._write_sse_chat_completion(
            _request(),
            "cmpl-1",
            "model-hidden",
            1,
            stream_q,
            asyncio.create_task(_completed_agent()),
            prestream_timing=timing,
        )

    summaries = [r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn ")]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].split(" ", 1)[1])
    assert payload["first_event_kind"] == expected_kind
    wire = b"".join(chunks)
    assert wire.startswith(b"data: ")
    assert wire.endswith(b"data: [DONE]\n\n")
    assert b"model-hidden" in wire
    assert "model-hidden" not in summaries[0]
    assert "private reasoning" not in summaries[0]


@pytest.mark.asyncio
async def test_role_usage_and_terminal_without_semantic_emit_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    stream_q: queue.Queue = queue.Queue()
    stream_q.put(("__tool_progress__", {
        "type": "provider.metadata",
        "status": "running",
        "provider": "must-not-be-logged",
    }))
    stream_q.put(None)
    response, _ = _response()
    timing = PrestreamTiming()

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
        await adapter._write_sse_chat_completion(
            _request(),
            "cmpl-empty",
            "model-hidden",
            1,
            stream_q,
            asyncio.create_task(_completed_agent()),
            prestream_timing=timing,
        )

    summaries = [r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn ")]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].split(" ", 1)[1])
    assert payload["first_event_kind"] == "none"
    assert "must-not-be-logged" not in summaries[0]


@pytest.mark.asyncio
async def test_observer_failure_does_not_change_sse_bytes() -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))

    async def render(timing: PrestreamTiming | None) -> bytes:
        stream_q: queue.Queue = queue.Queue()
        stream_q.put("answer")
        stream_q.put(None)
        response, chunks = _response()
        with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
            await adapter._write_sse_chat_completion(
                _request(),
                "cmpl-same",
                "same-model",
                1,
                stream_q,
                asyncio.create_task(_completed_agent()),
                prestream_timing=timing,
            )
        return b"".join(chunks)

    baseline = await render(None)
    class _FailingTiming(PrestreamTiming):
        def public_write_completed(
            self, event: PrestreamSemanticEvent | None
        ) -> None:
            del event
            raise RuntimeError("observer failed")

        def terminal_write_completed(self) -> None:
            raise RuntimeError("observer failed")

    timing = _FailingTiming()
    observed = await render(timing)

    assert observed == baseline


@pytest.mark.asyncio
async def test_direct_final_response_records_semantic_to_successful_sse_write(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    stream_q: queue.Queue = queue.Queue()
    stream_q.put(None)
    now = 10.0

    def clock() -> float:
        return now

    timing = _CountingTerminalTiming(clock=clock)
    chunks: list[bytes] = []
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()

    async def _write(payload: bytes) -> None:
        nonlocal now
        chunks.append(payload)
        if b'"content": "direct answer"' in payload:
            now += 0.037

    response.write = AsyncMock(side_effect=_write)
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")

    with patch(
        "gateway.platforms.api_server.web.StreamResponse", return_value=response
    ):
        await adapter._write_sse_chat_completion(
            _request(),
            "cmpl-direct",
            "model-hidden",
            1,
            stream_q,
            asyncio.create_task(_completed_agent("direct answer")),
            prestream_timing=timing,
        )

    summaries = [
        record.message
        for record in caplog.records
        if record.message.startswith("hermes.prestream.turn ")
    ]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].split(" ", 1)[1])
    assert payload["first_event_kind"] == "content"
    assert payload["semantic_to_sse_write_ms"] == 37
    assert timing.terminal_calls == 1
    assert b"direct answer" in b"".join(chunks)
    assert "direct answer" not in summaries[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write_error", "cancelled"])
async def test_direct_final_response_failed_write_completes_terminal_once(
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    stream_q: queue.Queue = queue.Queue()
    stream_q.put(None)
    timing = _CountingTerminalTiming()
    response = AsyncMock(spec=web.StreamResponse)
    response.prepare = AsyncMock()

    async def _write(payload: bytes) -> None:
        if b'"content": "direct answer"' not in payload:
            return
        if failure == "cancelled":
            raise asyncio.CancelledError
        raise OSError("client disconnected")

    response.write = AsyncMock(side_effect=_write)
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")

    async def _render() -> None:
        with patch(
            "gateway.platforms.api_server.web.StreamResponse", return_value=response
        ):
            await adapter._write_sse_chat_completion(
                _request(),
                f"cmpl-direct-{failure}",
                "model-hidden",
                1,
                stream_q,
                asyncio.create_task(_completed_agent("direct answer")),
                prestream_timing=timing,
            )

    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await _render()
    else:
        await _render()

    summaries = [
        record.message
        for record in caplog.records
        if record.message.startswith("hermes.prestream.turn ")
    ]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].split(" ", 1)[1])
    assert payload["first_event_kind"] == "none"
    assert "semantic_to_sse_write_ms" not in payload
    assert timing.terminal_calls == 1


@pytest.mark.asyncio
async def test_zet_style_writer_override_keeps_one_turn_one_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, token="test"))
    assert adapter.platform == Platform.ZET_AGENT
    runner_timings: list[PrestreamTiming] = []

    async def _run_agent(**kwargs):
        runner_timings.append(kwargs["prestream_timing"])
        kwargs["stream_delta_callback"]("answer")
        return (
            {"final_response": "answer", "completed": True},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    from aiohttp.test_utils import TestClient, TestServer

    with patch.object(adapter, "_run_agent", side_effect=_run_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "must-not-be-logged",
                    "messages": [
                        {"role": "user", "content": "private-user-secret"}
                    ],
                    "stream": True,
                    "metadata": {"turn_id": "turn-1"},
                },
            )
            assert response.status == 200
            assert "answer" in await response.text()

    summaries = [r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn ")]
    assert len(summaries) == 1
    assert len(runner_timings) == 1
    assert json.loads(summaries[0].split(" ", 1)[1])["first_event_kind"] == "content"
    assert "private-user-secret" not in summaries[0]
    assert "must-not-be-logged" not in summaries[0]


@pytest.mark.asyncio
async def test_run_agent_records_executor_queue_and_agent_init_boundaries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.api_server import _prestream_timing_context

    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {"final_response": "ok"}
    fake_agent.session_prompt_tokens = 0
    fake_agent.session_completion_tokens = 0
    fake_agent.session_total_tokens = 0

    timing = PrestreamTiming()
    timing.history_ready(source="request", count=0)
    timing.executor_queued()
    token = _prestream_timing_context.set(timing)
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    try:
        def _slow_create_agent(**_kwargs):
            time.sleep(0.02)
            return fake_agent

        with patch.object(adapter, "_create_agent", side_effect=_slow_create_agent):
            await adapter._run_agent(
                user_message="private user text",
                conversation_history=[],
                session_id="session-1",
            )
    finally:
        _prestream_timing_context.reset(token)

    timing.public_write_completed(timing.semantic_observed("content"))
    summary = next(
        record.message
        for record in caplog.records
        if record.message.startswith("hermes.prestream.turn ")
    )
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["executor_queue_ms"] >= 0
    assert payload["agent_init_ms"] >= 15
    assert "private user text" not in summary


@pytest.mark.asyncio
async def test_run_agent_explicit_timing_reaches_executor_and_zet_creation_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.api_server import _prestream_timing_context

    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {"final_response": "ok"}
    fake_agent.session_prompt_tokens = 0
    fake_agent.session_completion_tokens = 0
    fake_agent.session_total_tokens = 0
    timing = PrestreamTiming()
    timing.history_ready(source="request", count=0)
    timing.executor_queued()
    seen_in_executor: list[PrestreamTiming | None] = []

    def _create_agent(**_kwargs):
        seen_in_executor.append(_prestream_timing_context.get())
        return fake_agent

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    assert _prestream_timing_context.get() is None
    with patch.object(adapter, "_create_agent", side_effect=_create_agent):
        await adapter._run_agent(
            user_message="private user text",
            conversation_history=[],
            session_id="session-1",
            prestream_timing=timing,
        )

    assert seen_in_executor == [timing]
    assert _prestream_timing_context.get() is None
    timing.public_write_completed(timing.semantic_observed("content"))
    summary = next(
        record.message
        for record in caplog.records
        if record.message.startswith("hermes.prestream.turn ")
    )
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["executor_queue_ms"] >= 0
    assert payload["agent_init_ms"] >= 0


@pytest.mark.asyncio
async def test_run_agent_init_exception_records_error_without_success_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.api_server import _prestream_timing_context

    adapter = APIServerAdapter(PlatformConfig(enabled=True, token="test"))
    timing = PrestreamTiming()
    timing.history_ready(source="request", count=0)
    timing.executor_queued()
    token = _prestream_timing_context.set(timing)
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    try:
        with patch.object(adapter, "_create_agent", side_effect=RuntimeError("init failed")):
            with pytest.raises(RuntimeError, match="init failed"):
                await adapter._run_agent(
                    user_message="private",
                    conversation_history=[],
                    session_id="session-1",
                )
    finally:
        _prestream_timing_context.reset(token)

    timing.terminal_write_completed()
    summary = next(r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn "))
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["agent_init_outcome"] == "error"
    assert "agent_init_ms" not in payload


@pytest.mark.asyncio
async def test_session_db_history_error_is_not_reported_as_ready(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))

    async def _history_error():
        raise RuntimeError("db unavailable")

    async def _run_agent(**kwargs):
        kwargs["stream_delta_callback"]("answer")
        return (
            {"final_response": "answer", "completed": True},
            {"input_tokens": 0, "output_tokens": 1, "total_tokens": 1},
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    from aiohttp.test_utils import TestClient, TestServer

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    with (
        patch.object(adapter, "_ensure_session_db_async", side_effect=_history_error),
        patch.object(adapter, "_run_agent", side_effect=_run_agent),
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Hermes-Session-Id": "session-1",
                },
                json={
                    "model": "hidden",
                    "messages": [{"role": "user", "content": "private"}],
                    "stream": True,
                },
            )
            assert response.status == 200
            await response.read()

    summary = next(r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn "))
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["history_outcome"] == "session_db_error"
    assert "ingress_to_history_ready_ms" not in payload


@pytest.mark.asyncio
async def test_real_zet_reasoning_callback_preserves_queue_to_write_duration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.api_server import _prestream_timing_context
    from gateway.platforms.zet_agent import ZetAgentAdapter

    now = 10.0

    def clock() -> float:
        return now

    class _Agent:
        def __init__(self, **kwargs):
            self.model = kwargs.get("model")
            self.provider = kwargs.get("provider")

    monkeypatch.setattr("run_agent.AIAgent", _Agent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "test", "model": "test/model", "api_key": "secret"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "test/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 10)
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", lambda: {})
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", lambda: None)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test"}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    stream_q: queue.Queue = queue.Queue()

    def _delta(value: str) -> None:
        stream_q.put(value)

    timing = PrestreamTiming(clock=clock)
    token = _prestream_timing_context.set(timing)
    try:
        agent = adapter._create_agent(
            session_id="session-1",
            stream_delta_callback=_delta,
        )
    finally:
        _prestream_timing_context.reset(token)

    agent.reasoning_callback("private chain of thought")
    now += 0.037
    stream_q.put(None)
    response, _ = _response()
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    writer_token = _prestream_timing_context.set(timing)
    try:
        with patch("gateway.platforms.api_server.web.StreamResponse", return_value=response):
            await adapter._write_sse_chat_completion(
                _request(),
                "cmpl-reasoning",
                "hidden-model",
                1,
                stream_q,
                asyncio.create_task(_completed_agent()),
            )
    finally:
        _prestream_timing_context.reset(writer_token)

    summary = next(r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn "))
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["first_event_kind"] == "reasoning"
    assert payload["semantic_to_sse_write_ms"] == 37
    assert "private chain of thought" not in summary


@pytest.mark.asyncio
async def test_cancelled_skill_worker_settles_timing_only_when_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test"}))
    started = threading.Event()
    release = threading.Event()
    settled = threading.Event()
    timing = PrestreamTiming(explicit_skill=True)
    timing.skill_expand_started()

    def _slow_expand(*_args, **_kwargs):
        started.set()
        release.wait(5)
        return "expanded"

    monkeypatch.setattr(
        adapter,
        "_expand_inbound_skill_invocation_blocking",
        _slow_expand,
    )

    def _on_settled() -> None:
        timing.skill_expand_settled()
        settled.set()

    task = asyncio.create_task(
        adapter._expand_inbound_skill_invocation(
            "task",
            "deep-research",
            on_settled=_on_settled,
        )
    )
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    timing.skill_expand_completed("cancelled")
    assert not settled.is_set()

    release.set()
    await asyncio.to_thread(settled.wait, 5)
    assert settled.is_set()


@pytest.mark.asyncio
async def test_fail_open_skill_build_is_still_classified_as_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from gateway.platforms.api_server import _prestream_timing_context
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test"}))
    monkeypatch.setattr("agent.skill_commands.scan_skill_commands", lambda: {})
    timing = PrestreamTiming(explicit_skill=True)
    timing.skill_expand_started()
    token = _prestream_timing_context.set(timing)
    try:
        result = await adapter._expand_inbound_skill_invocation(
            "task",
            "missing-skill",
            on_settled=timing.skill_expand_settled,
        )
    finally:
        _prestream_timing_context.reset(token)

    assert result == "task"
    timing.skill_expand_completed("success")
    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    timing.terminal_write_completed()
    summary = next(r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn "))
    payload = json.loads(summary.split(" ", 1)[1])
    assert payload["skill_expand_outcome"] == "error"
    assert "skill_expand_ms" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["missing", "build_error"])
async def test_handler_skill_fail_open_keeps_error_outcome_without_manual_context(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure_mode: str,
) -> None:
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    if failure_mode == "missing":
        monkeypatch.setattr("agent.skill_commands.scan_skill_commands", lambda: {})
    else:
        monkeypatch.setattr(
            "agent.skill_commands.scan_skill_commands",
            lambda: {"/broken": {"name": "broken"}},
        )

        def _build_error(*_args, **_kwargs):
            raise RuntimeError("broken skill")

        monkeypatch.setattr(
            "agent.skill_commands.build_skill_invocation_message",
            _build_error,
        )

    async def _run_agent(**kwargs):
        kwargs["stream_delta_callback"]("answer")
        return (
            {"final_response": "answer", "completed": True},
            {"input_tokens": 0, "output_tokens": 1, "total_tokens": 1},
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    from aiohttp.test_utils import TestClient, TestServer

    caplog.set_level(logging.INFO, logger="agent.prestream_timing")
    with patch.object(adapter, "_run_agent", side_effect=_run_agent):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "hidden",
                    "messages": [{"role": "user", "content": "private task"}],
                    "metadata": {"skill_slug": "broken"},
                    "stream": True,
                },
            )
            assert response.status == 200
            await response.read()

    summaries = [r.message for r in caplog.records if r.message.startswith("hermes.prestream.turn ")]
    assert len(summaries) == 1
    payload = json.loads(summaries[0].split(" ", 1)[1])
    assert payload["skill_expand_outcome"] == "error"
    assert "skill_expand_ms" not in payload
    assert "skill_expand:error" in payload["missing_stages"]
