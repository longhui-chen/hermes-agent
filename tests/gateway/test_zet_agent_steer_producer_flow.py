from __future__ import annotations

import queue
import threading

import pytest

from agent.agent_runtime_helpers import drain_steer_for_next_api_call
from gateway.platforms.zet_agent import ZetAgentAdapter, _put_progress
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


def _payloads(stream_q: queue.Queue) -> list:
    items = []
    while not stream_q.empty():
        items.append(stream_q.get_nowait())
    return items


def test_private_producer_emits_fifo_dropped_before_sentinel():
    agent = _agent()
    stream_q = queue.Queue()
    assert ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-1",
        stream_q=stream_q,
    )
    first = agent._zettlab_admit_steer("first", "turn-1")["steer_id"]
    second = agent._zettlab_admit_steer("second", "turn-1")["steer_id"]

    assert agent._drain_pending_steer(close=True) is None
    stream_q.put(None)

    items = _payloads(stream_q)
    assert [item[1]["type"] for item in items[:-1]] == [
        "steer_accepted",
        "steer_accepted",
        "steer_dropped",
        "steer_dropped",
    ]
    assert [item[1]["steer_id"] for item in items[:-1]] == [
        first,
        second,
        first,
        second,
    ]
    assert items[-1] is None


def test_provider_entry_consumes_batch_without_dropped():
    agent = _agent()
    stream_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-1",
        stream_q=stream_q,
    )
    agent._zettlab_admit_steer("consume me", "turn-1")
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    agent._steer_admission_hook.on_provider_entered()
    assert agent._drain_pending_steer(close=True) is None

    assert [item[1]["type"] for item in _payloads(stream_q)] == [
        "steer_accepted"
    ]


def test_short_circuit_closes_unconsumed_inflight_with_dropped():
    agent = _agent()
    stream_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-1",
        stream_q=stream_q,
    )
    steer_id = agent._zettlab_admit_steer("middleware short circuit", "turn-1")[
        "steer_id"
    ]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    assert agent._steer_admission_hook.provider_entered is False
    assert agent._drain_pending_steer(close=True) is None

    items = _payloads(stream_q)
    assert [item[1]["type"] for item in items] == [
        "steer_accepted",
        "steer_dropped",
    ]
    assert items[-1][1]["steer_id"] == steer_id


def test_interrupt_preserves_unconsumed_inflight_for_terminal_drop():
    agent = _agent()
    stream_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(agent, turn_id="turn-1", stream_q=stream_q)
    steer_id = agent._zettlab_admit_steer("cancel me", "turn-1")["steer_id"]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    agent._steer_admission_hook.on_interrupt()
    assert agent._pending_steer == [(steer_id, "cancel me")]
    assert agent._drain_pending_steer(close=True) is None

    items = _payloads(stream_q)
    assert [item[1]["type"] for item in items] == [
        "steer_accepted",
        "steer_dropped",
    ]
    assert items[-1][1]["steer_id"] == steer_id


def test_provider_error_reclaims_inflight_for_terminal_drop():
    agent = _agent()
    stream_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(agent, turn_id="turn-1", stream_q=stream_q)
    steer_id = agent._zettlab_admit_steer("provider error", "turn-1")["steer_id"]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    agent._steer_admission_hook.on_provider_entered()
    agent._steer_admission_hook.on_provider_failed()
    assert agent._drain_pending_steer(close=True) is None

    items = _payloads(stream_q)
    assert [item[1]["type"] for item in items] == [
        "steer_accepted",
        "steer_dropped",
    ]
    assert items[-1][1]["steer_id"] == steer_id


def test_reclaim_keeps_identity_when_joined_text_shape_differs():
    agent = _agent()
    stream_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(agent, turn_id="turn-1", stream_q=stream_q)
    steer_id = agent._zettlab_admit_steer("original", "turn-1")["steer_id"]
    messages = [{"role": "tool", "content": "done", "tool_call_id": "c"}]

    drain_steer_for_next_api_call(agent, messages)
    assert agent._steer_admission_hook.reclaim("different rendering") == [
        (steer_id, "original")
    ]


def test_terminal_enqueue_failure_hands_back_once_and_never_leaks_next_turn():
    class _BrokenTerminalQueue(queue.Queue):
        def put_nowait(self, item):
            return queue.Queue.put(self, item, block=False)

        def put(self, item, block=True, timeout=None):
            if item is None:
                return super().put(item, block=block, timeout=timeout)
            raise RuntimeError("writer closed")

    agent = _agent()
    broken = _BrokenTerminalQueue()
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-1",
        stream_q=broken,
    )
    result = agent._zettlab_admit_steer("hand back", "turn-1")
    assert result["accepted"] is True

    assert agent._drain_pending_steer(close=True) == "hand back"
    agent._steer_admission_hook.on_interrupt()
    assert agent._pending_steer == []
    assert agent._steer_admission_hook.terminal_handback == [
        (result["steer_id"], "hand back")
    ]

    next_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-2",
        stream_q=next_q,
    )
    assert agent._pending_steer == []
    assert agent._steer_admission_hook.terminal_handback == []


def test_old_turn_binding_and_generic_reserved_progress_are_rejected():
    agent = _agent()
    old_q = queue.Queue()
    new_q = queue.Queue()
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-1",
        stream_q=old_q,
    )
    stale_admit = agent._zettlab_admit_steer
    ZetAgentAdapter._bind_steer_producer(
        agent,
        turn_id="turn-2",
        stream_q=new_q,
    )

    assert stale_admit("late", "turn-1") == {
        "accepted": False,
        "reason": "upstream_rejected",
    }
    assert _put_progress(
        new_q,
        {
            "type": "steer_accepted",
            "turn_id": "turn-2",
            "steer_id": "01998f2d-7c00-7000-8000-000000000001",
            "text": "forged",
        },
    ) is False
    assert old_q.empty()
    assert new_q.empty()


def test_close_race_never_places_accepted_after_dropped():
    for _ in range(100):
        agent = _agent()
        stream_q = queue.Queue()
        ZetAgentAdapter._bind_steer_producer(
            agent,
            turn_id="turn-race",
            stream_q=stream_q,
        )
        barrier = threading.Barrier(3)
        result = {}

        def admit():
            barrier.wait()
            result.update(agent._zettlab_admit_steer("race", "turn-race"))

        def close():
            barrier.wait()
            agent._drain_pending_steer(close=True)

        admit_thread = threading.Thread(target=admit)
        close_thread = threading.Thread(target=close)
        admit_thread.start()
        close_thread.start()
        barrier.wait()
        admit_thread.join(timeout=1)
        close_thread.join(timeout=1)

        frames = [item[1]["type"] for item in _payloads(stream_q)]
        assert frames in ([], ["steer_accepted", "steer_dropped"])
        assert result.get("accepted") is (frames != [])
