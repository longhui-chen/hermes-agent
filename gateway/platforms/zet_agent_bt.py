"""Request-local BT projection at the SSE writer, after queue ordering."""
from __future__ import annotations

from contextvars import ContextVar
import json
from pathlib import Path
from typing import Any

from .item_sequencer import ItemSequencer
from .tool_display import build_tool_result_display, build_tool_start_display

projection_context: ContextVar[WriterProjection] = ContextVar("bt_writer_projection")


class BTContentDelta(str):
    def __new__(cls, value: str, item_id: str, index: int):
        result = super().__new__(cls, value)
        result.wire_fields = {"hermes": {"item_id": item_id, "index": index}}
        return result


def identity_fields() -> dict[str, str]:
    """The vendored producer manifest is the runtime identity registry."""
    manifest = json.loads((Path(__file__).parents[2] / "schemas/chat-ui.manifest.json").read_text())
    return {name: shape["identity_field"] for name, shape in manifest["hermes"]["frame_shapes"].items()
            if "identity_field" in shape}


class WriterProjection:
    def __init__(self, identities: dict[str, str] | None = None):
        self.sequencer = ItemSequencer(identity_fields=identity_fields() if identities is None else identities)

    def project(self, item: Any) -> list[Any]:
        if isinstance(item, str):
            return self._project_text(item)
        if not isinstance(item, tuple) or len(item) != 2:
            self.sequencer._count("item_frame_unclassified")
            return [item]
        tag, payload = item
        if tag != "__tool_progress__" or not isinstance(payload, dict):
            return [item]
        # Tool progress has no type discriminator in the registered wire dialect.
        tool = not payload.get("type") and isinstance(payload.get("tool"), str)
        frame = dict(payload)
        if tool:
            frame["type"] = "tool.result" if frame.get("status") == "completed" else "tool.start"
        frames = self.sequencer.process(frame)
        if frames == [frame] and "item_id" not in frame and "index" not in frame:
            return [item]  # rejected producer input retains the existing fallback
        result = []
        for projected in frames:
            if tool and projected.get("type") in {"tool.start", "tool.result"}:
                projected = dict(projected)
                projected.pop("type")
            result.append((tag, projected))
        return result

    def _project_text(self, text: str) -> list[Any]:
        result = []
        for frame in self.sequencer.process(text):
            if isinstance(frame, str):
                result.append(frame)
            elif frame.get("type") == "text.delta":
                result.append(BTContentDelta(frame["text"], frame["item_id"], frame["index"]))
            else:
                result.append(("__tool_progress__", frame))
        return result

    def finish(self, result: dict[str, Any]) -> list[Any]:
        frames = []
        if result.get("canonical_response_required") or result.get("response_transformed"):
            frames.extend(self.sequencer.complete_canonical(result.get("final_response") or ""))
        for kind in ("reasoning", "text"):
            frames.extend(self.sequencer._close(kind))
        return [("__tool_progress__", frame) for frame in frames]

    def count_stranded(self, count: int) -> None:
        if count:
            self.sequencer.counters["item_frame_stranded"] = count


def tool_callbacks(stream_q, timing=None):
    """Only build bounded payloads here; identity is assigned when drained."""
    from gateway.platforms.api_server import _tool_completion_payload
    from tools.registry import registry

    def start(call_id, name, args):
        if not isinstance(call_id, str) or not isinstance(name, str) or name.startswith("_"):
            return
        entry = registry.get_entry(name)
        registration = {"kind": "builtin", "id": name, "label": name}
        if entry and entry.toolset.startswith("mcp-"):
            server = entry.toolset.removeprefix("mcp-")
            registration = {"kind": "mcp", "server": server, "server_label": server}
        display = build_tool_start_display(name, args, registration)
        stream_q.put(("__tool_progress__", {"tool": name, "toolCallId": call_id, "status": "running", **display}))
        if timing:
            timing.observe_queued_semantic("tool_start")

    def complete(call_id, name, args, result):
        if not isinstance(call_id, str) or not isinstance(name, str) or name.startswith("_"):
            return
        payload = _tool_completion_payload(call_id, name, result)
        error = result if payload.get("outcome") == "error" else None
        payload.update(build_tool_result_display(result, error=error))
        stream_q.put(("__tool_progress__", payload))

    return start, complete
