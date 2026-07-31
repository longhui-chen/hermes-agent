import queue
import threading

import pytest

from gateway.platforms.zet_agent import ZetAgentAdapter


class _Goals:
    def __init__(self):
        self.pending = []

    def on_interaction_pending(self, session_id):
        self.pending.append(session_id)


def _adapter():
    adapter = object.__new__(ZetAgentAdapter)
    adapter._pending_lock = threading.Lock()
    adapter._pending_approval = {}
    adapter._approval_stream_queues = {}
    goals = _Goals()
    adapter._goals = lambda: goals
    return adapter, goals


def test_live_approval_projection_emits_and_advances_fifo():
    adapter, goals = _adapter()
    stream = queue.Queue()
    notify = adapter._make_approval_cb(stream, "session-a")
    first = {"approval_id": "a" * 24, "command": "first"}
    second = {"approval_id": "b" * 24, "command": "second"}

    notify(first)
    notify(second)

    assert stream.get_nowait()[1]["approval_id"] == first["approval_id"]
    with pytest.raises(queue.Empty):
        stream.get_nowait()
    assert adapter._approval_projection_head("session-a")["approval_id"] == first["approval_id"]
    assert goals.pending == ["session-a"]

    assert adapter._remove_approval_projection("session-a", first["approval_id"])
    assert stream.get_nowait()[1]["approval_id"] == second["approval_id"]
    assert adapter._approval_projection_head("session-a")["approval_id"] == second["approval_id"]
    assert not adapter._remove_approval_projection("session-a", second["approval_id"])
    assert adapter._approval_projection_head("session-a") is None


def test_resolving_non_head_approval_keeps_visible_head():
    adapter, _goals = _adapter()
    stream = queue.Queue()
    notify = adapter._make_approval_cb(stream, "session-a")
    first = {"approval_id": "a" * 24, "command": "first"}
    second = {"approval_id": "b" * 24, "command": "second"}
    notify(first)
    notify(second)
    stream.get_nowait()

    assert adapter._remove_approval_projection("session-a", second["approval_id"])
    assert adapter._approval_projection_head("session-a")["approval_id"] == first["approval_id"]
    with pytest.raises(queue.Empty):
        stream.get_nowait()
