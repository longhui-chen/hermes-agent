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
from run_agent import AIAgent


def _agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_steer = []
    agent._pending_steer_lock = threading.Lock()
    agent._steer_closed = True
    agent._steer_binding_turn_id = ""
    agent._steer_stream_q = None
    agent._steer_binding_token = None
    agent._steer_pending_bytes = 0
    agent._steer_inflight_batch = []
    agent._steer_provider_entered = False
    agent._steer_terminal_handback = []
    agent._interrupt_requested = False
    agent._turn_last_steer_text = None
    agent._steer_merged_db_rows = []
    agent._persist_disabled = True
    agent.quiet_mode = True
    return agent


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
    agent._bind_steer_turn("turn-1", stream_q, token, accepted.append)

    result = agent._admit_steer(
        "顺便检查日志",
        turn_id="turn-1",
        stream_q=stream_q,
        binding_token=token,
    )

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
    agent._bind_steer_turn(
        "turn-1",
        stream_q,
        token,
        lambda _payload: (_ for _ in ()).throw(RuntimeError("closed")),
    )

    stale = agent._admit_steer(
        "old turn",
        turn_id="turn-0",
        stream_q=stream_q,
        binding_token=token,
    )
    failed = agent._admit_steer(
        "current turn",
        turn_id="turn-1",
        stream_q=stream_q,
        binding_token=token,
    )

    assert stale == {"accepted": False, "reason": "upstream_rejected"}
    assert failed == {"accepted": False, "reason": "stream_enqueue_failed"}
    assert agent._pending_steer == []
    assert agent._steer_pending_bytes == 0


def test_rejection_rolls_back_existing_request_usage_reservation():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    rollbacks = []
    agent._steer_usage_rollback = lambda: rollbacks.append("released")
    agent._bind_steer_turn("turn-1", stream_q, token, lambda _payload: None)

    assert agent._admit_steer(
        "stale",
        turn_id="turn-0",
        stream_q=stream_q,
        binding_token=token,
    )["accepted"] is False
    assert rollbacks == ["released"]


def test_admission_enforces_text_item_and_byte_limits():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    agent._bind_steer_turn("turn-1", stream_q, token, lambda _payload: None)

    for idx in range(64):
        result = agent._admit_steer(
            f"note-{idx}",
            turn_id="turn-1",
            stream_q=stream_q,
            binding_token=token,
        )
        assert result["accepted"] is True

    assert len(agent._pending_steer) == 64
    assert agent._admit_steer(
        "overflow",
        turn_id="turn-1",
        stream_q=stream_q,
        binding_token=token,
    ) == {"accepted": False, "reason": "pending_limit"}

    oversized = "界" * ((64 * 1024 // 3) + 1)
    other = _agent()
    other._bind_steer_turn("turn-2", stream_q, token, lambda _payload: None)
    assert other._admit_steer(
        oversized,
        turn_id="turn-2",
        stream_q=stream_q,
        binding_token=token,
    ) == {"accepted": False, "reason": "steer_text_malformed"}


def test_fifo_join_happens_only_at_model_feed_and_reclaim_preserves_ids():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    agent._bind_steer_turn("turn-1", stream_q, token, lambda _payload: None)
    first = agent._admit_steer(
        "first",
        turn_id="turn-1",
        stream_q=stream_q,
        binding_token=token,
    )["steer_id"]
    second = agent._admit_steer(
        "second",
        turn_id="turn-1",
        stream_q=stream_q,
        binding_token=token,
    )["steer_id"]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "call-1"}]

    drain_steer_for_next_api_call(agent, messages)

    assert messages[-1]["content"].endswith("first\nsecond")
    assert agent._pending_steer == []
    assert agent._steer_inflight_batch == [(first, "first"), (second, "second")]

    reclaim_tail_steer(agent, messages)

    assert agent._pending_steer == [(first, "first"), (second, "second")]
    assert agent._steer_inflight_batch == []
    assert agent._steer_pending_bytes == sum(
        _encoded_size(item) for item in agent._pending_steer
    )


def test_provider_entered_steer_is_not_reclaimed_as_unconsumed():
    agent = _agent()
    stream_q = queue.Queue()
    token = object()
    agent._bind_steer_turn("wire-turn", stream_q, token, lambda _payload: None)
    agent._admit_steer("already sent", turn_id="wire-turn", stream_q=stream_q, binding_token=token)
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    agent._mark_steer_batch_provider_entered()
    before = [dict(messages[0]), dict(messages[-1])]
    reclaim_tail_steer(agent, messages)

    assert messages == before
    assert agent._pending_steer == []


def test_random_admit_consume_close_never_exceeds_pending_envelope():
    rng = random.Random(20260909)
    for _case in range(40):
        agent = _agent()
        stream_q = queue.Queue()
        token = object()
        frames = []
        agent._bind_steer_turn("turn-random", stream_q, token, frames.append)
        messages = [{"role": "tool", "content": "x", "tool_call_id": "c"}]

        for step in range(250):
            operation = rng.choice(("admit", "consume", "reclaim"))
            if operation == "admit":
                agent._admit_steer(
                    f"{step}-" + ("界" * rng.randrange(0, 80)),
                    turn_id="turn-random",
                    stream_q=stream_q,
                    binding_token=token,
                )
            elif operation == "consume":
                drain_steer_for_next_api_call(agent, messages)
                agent._mark_steer_batch_provider_entered()
            else:
                if agent._steer_inflight_batch:
                    reclaim_tail_steer(agent, messages)

            assert len(agent._pending_steer) <= 64
            assert agent._steer_pending_bytes == sum(
                _encoded_size(item)
                for item in agent._pending_steer
                if item[0]
            )
            assert agent._steer_pending_bytes <= 64 * 1024
