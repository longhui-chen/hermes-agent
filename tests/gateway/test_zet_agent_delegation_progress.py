"""hermes.delegation.progress SSE bridge (delegation-app-foundation).

Covers ZetAgentAdapter._make_delegation_progress_cb: the parent
tool_progress_callback that forwards delegate_task child lifecycle events
onto the ``__tool_progress__`` SSE lane, and its contract with
delegate_tool's ``_build_child_progress_callback`` relay (the two halves
must agree on signature and kwargs or the App banner goes dark silently).
"""

import queue
import threading

from unittest.mock import MagicMock

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


def _drain(q):
    frames = []
    while True:
        try:
            frames.append(q.get_nowait())
        except queue.Empty:
            return frames


def _payloads(q):
    out = []
    for lane, payload in _drain(q):
        assert lane == "__tool_progress__"
        out.append(payload)
    return out


def _cb_and_queue():
    q = queue.Queue()
    return ZetAgentAdapter._make_delegation_progress_cb(q), q


def test_forwards_status_level_events_with_identity():
    cb, q = _cb_and_queue()
    cb(
        "subagent.start",
        None,
        "Research topic A",
        None,
        task_index=0,
        task_count=3,
        goal="Research topic A",
        subagent_id="sa_1",
        child_session_id="child-sess-1",
        tool_count=0,
    )
    cb(
        "subagent.tool",
        "web_search",
        "quantum computing",
        None,
        task_index=0,
        task_count=3,
        goal="Research topic A",
        subagent_id="sa_1",
        tool_count=1,
    )
    cb(
        "subagent.complete",
        None,
        "done",
        None,
        task_index=0,
        task_count=3,
        goal="Research topic A",
        subagent_id="sa_1",
        status="completed",
        duration_seconds=12.5,
    )

    payloads = _payloads(q)
    assert [p["event"] for p in payloads] == [
        "subagent.start",
        "subagent.tool",
        "subagent.complete",
    ]
    for p in payloads:
        assert p["type"] == "hermes.delegation.progress"
        assert p["kind"] == "delegation"
        assert p["task_index"] == 0
        assert p["task_count"] == 3
        assert p["goal"] == "Research topic A"
        assert p["subagent_id"] == "sa_1"
    assert payloads[0]["child_session_id"] == "child-sess-1"
    assert payloads[1]["tool"] == "web_search"
    assert payloads[1]["tool_count"] == 1
    assert payloads[2]["status"] == "completed"
    assert payloads[2]["duration_seconds"] == 12.5


def test_drops_chatty_and_unknown_events():
    cb, q = _cb_and_queue()
    cb("subagent.text", None, "long streamed prose...", None, task_index=0)
    cb("subagent.thinking", None, "hmm", None, task_index=0)
    cb("something.else", "tool", "x", None)
    assert _payloads(q) == []


def test_normalises_nested_orchestrator_passthrough():
    # Nested pass-through arrives as ("subagent_progress", summary) with the
    # summary in the tool_name positional slot.
    cb, q = _cb_and_queue()
    cb("subagent_progress", "[1] 🔀 web_search, read_file", None, None, task_index=1)
    payloads = _payloads(q)
    assert len(payloads) == 1
    assert payloads[0]["event"] == "subagent.progress"
    assert payloads[0]["preview"] == "[1] 🔀 web_search, read_file"
    assert "tool" not in payloads[0]


def test_preview_is_truncated():
    cb, q = _cb_and_queue()
    cb("subagent.progress", None, "x" * 500, None, task_index=0)
    payloads = _payloads(q)
    assert len(payloads[0]["preview"]) == ZetAgentAdapter._DELEGATION_PREVIEW_MAX + 1
    assert payloads[0]["preview"].endswith("…")


def test_never_raises_when_queue_fails():
    class _BoomQueue:
        def put(self, item):
            raise RuntimeError("queue closed")

    cb = ZetAgentAdapter._make_delegation_progress_cb(_BoomQueue())
    cb("subagent.start", None, "g", None, task_index=0)  # must not raise


def test_contract_with_delegate_tool_child_relay():
    """End-to-end: delegate_tool's relay drives the bridge.

    Builds the real child progress callback with a parent whose
    tool_progress_callback is this bridge, fires child lifecycle events the
    way tool_executor/delegate_tool do, and asserts structured frames land on
    the SSE lane. Catches signature drift between the two halves.
    """
    from tools.delegate_tool import _build_child_progress_callback

    bridge, q = _cb_and_queue()
    parent = MagicMock()
    parent._delegate_spinner = None
    parent.tool_progress_callback = bridge

    child_cb = _build_child_progress_callback(
        task_index=1,
        goal="Fix the build",
        parent_agent=parent,
        task_count=2,
        subagent_id="sa_42",
        depth=1,
    )
    assert child_cb is not None  # wiring the bridge is what enables the relay

    child_cb("subagent.start", preview="Fix the build")
    for i in range(5):  # one full batch of tool starts → one progress frame
        child_cb("tool.started", f"tool_{i}", "arg preview")
    child_cb("subagent.complete", preview="done", status="completed")

    payloads = _payloads(q)
    events = [p["event"] for p in payloads]
    # 5 tool_start → 5 subagent.tool frames + 1 batched subagent.progress
    assert events.count("subagent.start") == 1
    assert events.count("subagent.tool") == 5
    assert events.count("subagent.progress") == 1
    assert events.count("subagent.complete") == 1
    for p in payloads:
        assert p["task_index"] == 1
        assert p["task_count"] == 2
        assert p["goal"] == "Fix the build"
        assert p["subagent_id"] == "sa_42"
    batch = [p for p in payloads if p["event"] == "subagent.progress"][0]
    assert "tool_0" in batch["preview"] and "tool_4" in batch["preview"]


def test_backlog_cap_stops_writes_to_dead_queue():
    """Background children keep the callback after the SSE writer exits —
    the backlog must be bounded (HR#1), not grow with task lifetime."""
    cb, q = _cb_and_queue()
    cap = ZetAgentAdapter._DELEGATION_PROGRESS_BACKLOG_MAX
    for _ in range(cap + 50):
        cb("subagent.tool", "terminal", "ls")
    assert q.qsize() <= cap + 1  # one in-flight put may land past the check
