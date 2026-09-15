from queue import Queue

from gateway.platforms.zet_agent_bt import bind_item_callbacks


def _frames(queue):
    return [value[1] for value in list(queue.queue)]


def test_callbacks_emit_item_lifecycle_and_display():
    queue = Queue()
    seen = []
    reasoning, start, complete, finish, _ = bind_item_callbacks(
        stream_q=queue, turn_id="turn-1", reasoning=lambda text: seen.append(text),
        tool_start=lambda *args: seen.append(("start", args)),
        tool_complete=lambda *args: seen.append(("complete", args)),
    )
    reasoning("hello")
    start("call-1", "search", {"token": "secret", "q": "x"})
    complete("call-1", "search", {}, "ok")
    finish()
    frames = _frames(queue)
    assert frames[0]["type"] == "item.started"
    assert any(frame.get("type") == "tool.start" for frame in frames)
    assert any(frame.get("type") == "tool.result" for frame in frames)
    assert [frame["type"] for frame in frames].count("item.started") >= 2
    assert [frame["type"] for frame in frames].count("item.completed") >= 1
    assert "[REDACTED]" in next(frame for frame in frames if frame.get("type") == "tool.start")["display"]["args_summary"]
    assert seen[0] == "hello"


def test_finish_closes_reasoning_item():
    queue = Queue()
    reasoning, _, _, finish, _ = bind_item_callbacks(
        stream_q=queue, turn_id=None, reasoning=None, tool_start=None, tool_complete=None
    )
    reasoning("only answer")
    finish()
    assert any(frame.get("type") == "item.completed" for frame in _frames(queue))


def test_error_result_is_error_and_redacts_authorization_key():
    queue = Queue()
    _, _, complete, _, _ = bind_item_callbacks(
        stream_q=queue, turn_id=None, reasoning=None, tool_start=None, tool_complete=None
    )
    complete("c", "tool", {}, {"status": "error", "authorization": "Bearer secret"})
    result = next(frame for frame in _frames(queue) if frame.get("type") == "tool.result")
    assert result["display"]["content_type"] == "error"
    assert "secret" not in result["display"]["summary"]


def test_callbacks_preserve_legacy_callbacks():
    queue = Queue()
    calls = []
    _, start, complete, _, _ = bind_item_callbacks(
        stream_q=queue, turn_id=None, reasoning=None,
        tool_start=lambda *args: calls.append(("start", args)),
        tool_complete=lambda *args: calls.append(("complete", args)),
    )
    start("c", "tool", {})
    complete("c", "tool", {}, {"ok": True})
    assert [name for name, _ in calls] == ["start", "complete"]


def test_transform_attaches_item_identity_without_duplicate_text():
    queue = Queue()
    _, _, _, _, transform = bind_item_callbacks(
        stream_q=queue, turn_id="turn-1", reasoning=None,
        tool_start=None, tool_complete=None,
    )
    emitted = transform("hello")
    assert len([item for item in emitted if isinstance(item, str)]) == 1
    content = next(item for item in emitted if isinstance(item, str))
    assert content == "hello"
    assert isinstance(content.item_id, str)
    assert isinstance(content.index, int)
