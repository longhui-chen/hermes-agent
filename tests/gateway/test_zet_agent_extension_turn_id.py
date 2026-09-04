"""chat-ui contract §1.1: extension frames carry the bound turn id.

Unit coverage for ``_stamp_extension_turn_id`` / ``_put_progress`` and a flow
check that the static title push goes through the single emit point.
"""

import queue

from gateway.platforms import zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter, _put_progress, _stamp_extension_turn_id


def _bind_turn(monkeypatch, turn_id):
    def fake_get_session_env(name, default=""):
        if name == "HERMES_TURN_ID":
            return turn_id
        return default

    monkeypatch.setattr("gateway.session_context.get_session_env", fake_get_session_env)


def test_stamp_adds_turn_id_when_turn_is_bound(monkeypatch):
    _bind_turn(monkeypatch, "turn-1")
    payload = {"type": "hermes.todo", "todos": [], "summary": {}}
    assert _stamp_extension_turn_id(payload) is payload
    assert payload["turn_id"] == "turn-1"


def test_stamp_keeps_existing_turn_id(monkeypatch):
    _bind_turn(monkeypatch, "turn-1")
    payload = {"type": "hermes.clarify", "turn_id": "turn-0"}
    _stamp_extension_turn_id(payload)
    assert payload["turn_id"] == "turn-0"


def test_stamp_leaves_frame_alone_outside_a_turn(monkeypatch):
    _bind_turn(monkeypatch, "")
    payload = {"type": "conversation.title", "title": "t"}
    _stamp_extension_turn_id(payload)
    assert "turn_id" not in payload


def test_stamp_never_raises_when_session_context_is_unavailable(monkeypatch):
    def boom(name, default=""):
        raise RuntimeError("no context")

    monkeypatch.setattr("gateway.session_context.get_session_env", boom)
    payload = {"type": "reasoning.delta", "text": "x"}
    assert _stamp_extension_turn_id(payload) is payload
    assert "turn_id" not in payload


def test_put_progress_wraps_frame_on_the_progress_lane(monkeypatch):
    _bind_turn(monkeypatch, "turn-2")
    q = queue.Queue()
    _put_progress(q, {"type": "reasoning.delta", "text": "x"})
    tag, frame = q.get_nowait()
    assert tag == "__tool_progress__"
    assert frame == {"type": "reasoning.delta", "text": "x", "turn_id": "turn-2"}


def test_title_push_flows_through_single_emit_point(monkeypatch):
    _bind_turn(monkeypatch, "turn-3")
    q = queue.Queue()
    ZetAgentAdapter._push_title(q, "周末公园 vlog")
    tag, frame = q.get_nowait()
    assert tag == "__tool_progress__"
    assert frame == {"type": "conversation.title", "title": "周末公园 vlog", "turn_id": "turn-3"}


def test_adapter_has_no_raw_progress_puts():
    """Every extension frame must go through _put_progress (guard for future edits)."""
    import inspect

    src = inspect.getsource(zet_agent)
    # Exactly one raw push may exist: the body of _put_progress itself.
    assert src.count('stream_q.put(("__tool_progress__"') == 1
