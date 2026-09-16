from __future__ import annotations

import json
import queue
import random
import threading
import uuid

from agent.agent_runtime_helpers import (
    drain_steer_for_next_api_call,
    reclaim_tail_steer,
)
from gateway.platforms.zet_agent import _SteerProducer
from run_agent import AIAgent


def _agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_steer = []
    agent._pending_steer_lock = threading.Lock()
    agent._interrupt_requested = False
    agent._turn_last_steer_text = None
    agent._steer_merged_db_rows = []
    agent._persist_disabled = True
    agent.quiet_mode = True
    return agent


def _bind(agent, turn_id, stream_q, accepted=None, terminal=None):
    if agent._pending_steer is None:
        agent._pending_steer = []
    producer = _SteerProducer(
        agent,
        turn_id=turn_id,
        stream_q=stream_q,
        binding_token=object(),
        accepted_sender=accepted or (lambda _payload: None),
        terminal_sender=terminal or (lambda _payload: None),
        stream_backlog_max=2000,
    )
    agent._steer_admission_hook = producer
    agent._zettlab_admit_steer = producer
    return producer


def _encoded_size(item: tuple[str, str]) -> int:
    steer_id, text = item
    return len(
        json.dumps(
            {"steer_id": steer_id, "text": text},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_id_admission_is_atomic_and_uuid7_canonical():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    accepted = []
    _bind(agent, "turn-1", stream_q, accepted.append)

    result = agent._zettlab_admit_steer("顺便检查日志", "turn-1")

    assert result["accepted"] is True
    steer_id = result["steer_id"]
    parsed = uuid.UUID(steer_id)
    assert parsed.version == 7
    assert str(parsed) == steer_id
    assert len(steer_id.encode("ascii")) == 36
    assert agent._pending_steer == [(steer_id, "顺便检查日志")]
    assert accepted == [
        {
            "type": "steer_accepted",
            "turn_id": "turn-1",
            "steer_id": steer_id,
            "text": "顺便检查日志",
        }
    ]


def test_admission_rejects_stale_binding_and_rolls_back_enqueue_failure():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    _bind(agent, "turn-1", stream_q, lambda _payload: (_ for _ in ()).throw(RuntimeError("closed")))

    stale = agent._zettlab_admit_steer("old turn", "turn-0")
    failed = agent._zettlab_admit_steer("current turn", "turn-1")

    assert stale == {"accepted": False, "reason": "upstream_rejected"}
    assert failed == {"accepted": False, "reason": "stream_enqueue_failed"}
    assert agent._pending_steer == []
    assert agent._steer_admission_hook.pending_bytes == 0


def test_rejection_rolls_back_existing_request_usage_reservation():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    rollbacks = []
    agent._steer_usage_rollback = lambda: rollbacks.append("released")
    _bind(agent, "turn-1", stream_q)

    assert agent._zettlab_admit_steer("stale", "turn-0")["accepted"] is False
    assert rollbacks == ["released"]


def test_admission_enforces_text_item_and_byte_limits():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    _bind(agent, "turn-1", stream_q)

    for idx in range(64):
        result = agent._zettlab_admit_steer(f"note-{idx}", "turn-1")
        assert result["accepted"] is True

    assert len(agent._pending_steer) == 64
    assert agent._zettlab_admit_steer("overflow", "turn-1") == {"accepted": False, "reason": "pending_limit"}

    oversized = "界" * ((64 * 1024 // 3) + 1)
    other = _agent()
    _bind(other, "turn-2", stream_q)
    assert other._zettlab_admit_steer(oversized, "turn-2") == {"accepted": False, "reason": "steer_text_malformed"}


def test_fifo_join_happens_only_at_model_feed_and_reclaim_preserves_ids():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    _bind(agent, "turn-1", stream_q)
    first = agent._zettlab_admit_steer("first", "turn-1")["steer_id"]
    second = agent._zettlab_admit_steer("second", "turn-1")["steer_id"]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "call-1"}]

    drain_steer_for_next_api_call(agent, messages)

    assert messages[-1]["content"].endswith("first\nsecond")
    assert agent._pending_steer == []
    assert agent._steer_admission_hook.inflight == [(first, "first"), (second, "second")]

    reclaim_tail_steer(agent, messages)

    assert agent._pending_steer == [(first, "first"), (second, "second")]
    assert agent._steer_admission_hook.inflight == []
    assert agent._steer_admission_hook.pending_bytes == sum(
        _encoded_size(item) for item in agent._pending_steer
    )


def test_provider_entered_steer_is_not_reclaimed_as_unconsumed():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    _bind(agent, "wire-turn", stream_q)
    agent._zettlab_admit_steer("already sent", "wire-turn")
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    agent._steer_admission_hook.on_provider_entered()
    before = [dict(messages[0]), dict(messages[-1])]
    reclaim_tail_steer(agent, messages)

    assert messages == before
    assert agent._pending_steer == []


def test_legacy_steer_uses_adapter_hook_and_codex_is_rejected():
    agent = _agent()
    stream_q = queue.Queue()
    producer = _bind(agent, "wire-turn", stream_q)
    accepted = agent.steer("legacy correction")
    assert accepted is True
    assert producer.inflight == []
    agent.api_mode = "codex_app_server"
    result = agent._zettlab_admit_steer("codex correction", "wire-turn")
    assert result == {"accepted": False, "reason": "upstream_rejected"}


def test_random_admit_consume_close_never_exceeds_pending_envelope():
    rng = random.Random(20260909)
    for _case in range(40):
        agent = _agent()
        stream_q = queue.Queue()
        token = object()
        frames = []
        turn_id = "turn-random-0"
        _bind(agent, turn_id, stream_q, frames.append, frames.append)
        messages = [{"role": "tool", "content": "x", "tool_call_id": "c"}]

        for step in range(250):
            operation = rng.choice(("admit", "consume", "reclaim", "close", "stale"))
            if operation == "admit":
                agent._zettlab_admit_steer(
                    f"{step}-" + ("界" * rng.randrange(0, 80)), turn_id
                )
            elif operation == "consume":
                drain_steer_for_next_api_call(agent, messages)
                agent._steer_admission_hook.on_provider_entered()
            elif operation == "reclaim":
                if agent._steer_admission_hook.inflight:
                    reclaim_tail_steer(agent, messages)
            elif operation == "close":
                agent._drain_pending_steer(close=True)
                assert agent._pending_steer == []
                assert agent._steer_admission_hook.pending_bytes == 0
                turn_id = f"turn-random-{step + 1}"
                stream_q = queue.Queue()
                token = object()
                _bind(agent, turn_id, stream_q, frames.append, frames.append)
                messages = [{"role": "tool", "content": "x", "tool_call_id": "c"}]
            else:
                assert agent._zettlab_admit_steer("replayed old request", "stale-turn")["accepted"] is False

            assert len(agent._pending_steer) <= 64
            assert agent._steer_admission_hook.pending_bytes == sum(
                _encoded_size(item)
                for item in agent._pending_steer
                if item[0]
            )
            assert agent._steer_admission_hook.pending_bytes <= 64 * 1024
        accepted_ids = {frame["steer_id"] for frame in frames if frame["type"] == "steer_accepted"}
        dropped_ids = [frame["steer_id"] for frame in frames if frame["type"] == "steer_dropped"]
        assert set(dropped_ids) <= accepted_ids
        assert len(dropped_ids) == len(set(dropped_ids))
