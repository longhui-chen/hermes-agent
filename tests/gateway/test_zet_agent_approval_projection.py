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
    scoped_key = adapter._active_turn_key("session-a")

    assert stream.get_nowait()[1]["approval_id"] == first["approval_id"]
    with pytest.raises(queue.Empty):
        stream.get_nowait()
    assert adapter._approval_projection_head(scoped_key)["approval_id"] == first["approval_id"]
    assert goals.pending == ["session-a"]

    assert adapter._remove_approval_projection(scoped_key, first["approval_id"])
    assert stream.get_nowait()[1]["approval_id"] == second["approval_id"]
    assert adapter._approval_projection_head(scoped_key)["approval_id"] == second["approval_id"]
    assert not adapter._remove_approval_projection(scoped_key, second["approval_id"])
    assert adapter._approval_projection_head(scoped_key) is None


def test_resolving_non_head_approval_keeps_visible_head():
    adapter, _goals = _adapter()
    stream = queue.Queue()
    notify = adapter._make_approval_cb(stream, "session-a")
    first = {"approval_id": "a" * 24, "command": "first"}
    second = {"approval_id": "b" * 24, "command": "second"}
    notify(first)
    notify(second)
    scoped_key = adapter._active_turn_key("session-a")
    stream.get_nowait()

    assert adapter._remove_approval_projection(scoped_key, second["approval_id"])
    assert adapter._approval_projection_head(scoped_key)["approval_id"] == first["approval_id"]
    with pytest.raises(queue.Empty):
        stream.get_nowait()


def test_projection_isolated_for_same_session_across_profiles(monkeypatch, tmp_path):
    adapter, _goals = _adapter()
    active_home = [tmp_path / "main"]
    for home in (tmp_path / "main", tmp_path / "coder"):
        home.mkdir()
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: active_home[0]
    )
    main_stream = queue.Queue()
    main_notify = adapter._make_approval_cb(main_stream, "shared-session")
    active_home[0] = tmp_path / "coder"
    coder_stream = queue.Queue()
    coder_notify = adapter._make_approval_cb(coder_stream, "shared-session")

    main_notify({"approval_id": "m" * 24, "command": "main command"})
    coder_notify({"approval_id": "c" * 24, "command": "coder command"})

    main_key = f"{tmp_path / 'main'}|shared-session"
    coder_key = f"{tmp_path / 'coder'}|shared-session"
    assert adapter._approval_projection_head(main_key)["command"] == "main command"
    assert adapter._approval_projection_head(coder_key)["command"] == "coder command"
    assert main_stream.get_nowait()[1]["command"] == "main command"
    assert coder_stream.get_nowait()[1]["command"] == "coder command"

    assert not adapter._remove_approval_projection(main_key, "m" * 24)
    assert adapter._approval_projection_head(main_key) is None
    assert adapter._approval_projection_head(coder_key)["command"] == "coder command"


def test_timeout_cleanup_advances_projection_fifo():
    adapter, _goals = _adapter()
    stream = queue.Queue()
    notify = adapter._make_approval_cb(stream, "session-a")
    cleanup_first = notify({"approval_id": "a" * 24, "command": "first"})
    cleanup_second = notify({"approval_id": "b" * 24, "command": "second"})
    scoped_key = adapter._active_turn_key("session-a")
    stream.get_nowait()

    cleanup_first()
    assert stream.get_nowait()[1]["command"] == "second"
    assert adapter._approval_projection_head(scoped_key)["command"] == "second"
    cleanup_second()
    assert adapter._approval_projection_head(scoped_key) is None


def test_projection_payload_and_capacity_are_bounded(monkeypatch):
    adapter, _goals = _adapter()
    adapter._APPROVAL_PROJECTION_MAX_PER_SESSION = 2
    adapter._APPROVAL_PROJECTION_MAX_GLOBAL = 2
    adapter._APPROVAL_PROJECTION_MAX_BYTES = 32_000
    stream = queue.Queue()
    notify = adapter._make_approval_cb(stream, "session-a")
    notify({"approval_id": "a" * 24, "command": "x" * 100_000})
    notify({"approval_id": "b" * 24, "command": "second"})
    scoped_key = adapter._active_turn_key("session-a")
    head = adapter._approval_projection_head(scoped_key)
    assert len(head["command"]) < 5000
    assert len(head["payload_fingerprint"]) == 64
    with pytest.raises(RuntimeError, match="session limit"):
        notify({"approval_id": "c" * 24, "command": "third"})
    assert len(adapter._pending_approval[scoped_key]) == 2
