"""BT adapter hooks kept outside the upstream gateway kernel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .item_sequencer import ItemSequencer
from .tool_display import build_tool_result_display, build_tool_start_display


class BTContentDelta(str):
    """A text delta carrying the sequencer identity for the SSE writer."""

    def __new__(cls, value: str, *, item_id: str, index: int):
        result = super().__new__(cls, value)
        result.item_id = item_id
        result.index = index
        return result


def bind_item_callbacks(
    *, stream_q: Any, turn_id: str | None, reasoning: Callable[..., Any] | None,
    tool_start: Callable[..., Any] | None, tool_complete: Callable[..., Any] | None,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any], Callable[[], None], Callable[..., list[Any]]]:
    sequencer = ItemSequencer(turn_id=turn_id)
    finished = False

    def emit(frame: Any) -> None:
        for item in sequencer.process(frame):
            if isinstance(item, dict):
                stream_q.put(("__tool_progress__", item))

    def transform(delta: Any) -> list[Any]:
        if delta is None:
            return []
        frames = sequencer.process(delta)
        item = sequencer.open_items.get("text")
        content = (
            BTContentDelta(delta, item_id=item.item_id, index=item.index)
            if isinstance(delta, str) and item is not None
            else delta
        )
        return [content] + [("__tool_progress__", frame) for frame in frames if isinstance(frame, dict)]

    def on_reasoning(text: Any) -> None:
        if text:
            emit({"type": "reasoning.delta", "text": text})
        if reasoning:
            reasoning(text)

    def on_start(call_id: Any, name: Any, args: Any) -> None:
        emit({"type": "tool.start", "toolCallId": call_id, **build_tool_start_display(str(name), args)})
        if tool_start:
            tool_start(call_id, name, args)

    def on_complete(call_id: Any, name: Any, args: Any, result: Any) -> None:
        error = result if isinstance(result, BaseException) else None
        if isinstance(result, Mapping) and (result.get("error") or result.get("status") == "error"):
            error = result.get("error") or result
        emit({"type": "tool.result", "toolCallId": call_id, **build_tool_result_display(result, error=error)})
        if tool_complete:
            tool_complete(call_id, name, args, result)

    def finish() -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        for kind in ("reasoning", "text", "tool"):
            for frame in sequencer._close(kind):
                stream_q.put(("__tool_progress__", frame))

    return on_reasoning, on_start, on_complete, finish, transform
