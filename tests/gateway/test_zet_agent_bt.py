from queue import Queue

from gateway.platforms.zet_agent_bt import bind_item_callbacks


def _frames(queue):
    return [value[1] for value in list(queue.queue)]


def test_callbacks_emit_item_lifecycle_and_display():
    queue = Queue()
    seen = []
    reasoning, start, complete = bind_item_callbacks(
        stream_q=queue, turn_id="turn-1", reasoning=lambda text: seen.append(text),
        tool_start=lambda *args: seen.append(("start", args)),
        tool_complete=lambda *args: seen.append(("complete", args)),
    )
    reasoning("hello")
    start("call-1", "search", {"token": "secret", "q": "x"})
    complete("call-1", "search", {}, "ok")
    frames = _frames(queue)
    assert frames[0]["type"] == "item.started"
    assert any(frame.get("type") == "tool.start" for frame in frames)
    assert any(frame.get("type") == "tool.result" for frame in frames)
    assert "[REDACTED]" in next(frame for frame in frames if frame.get("type") == "tool.start")["display"]["args_summary"]
    assert seen[0] == "hello"


def test_callbacks_preserve_legacy_callbacks():
    queue = Queue()
    calls = []
    _, start, complete = bind_item_callbacks(
        stream_q=queue, turn_id=None, reasoning=None,
        tool_start=lambda *args: calls.append(("start", args)),
        tool_complete=lambda *args: calls.append(("complete", args)),
    )
    start("c", "tool", {})
    complete("c", "tool", {}, {"ok": True})
    assert [name for name, _ in calls] == ["start", "complete"]
