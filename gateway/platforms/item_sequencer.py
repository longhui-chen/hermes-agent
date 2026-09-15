"""Pure, unconnected timeline item sequencing for the BT contract.

The adapter owns this state machine; the SSE writer will call it in T2.  Keeping
it independent of the writer makes the ordering and bounded snapshot invariants
executable before any runtime wiring is enabled.
"""
from __future__ import annotations

import time
import threading
import uuid
import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

UUID7_RE = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
MAX_SNAPSHOT_BYTES = 256 * 1024
MAX_INDEX = 2**31
MAX_VERSION = 2**31
MAX_ITEMS = 2048
MAX_IDENTITY_BYTES = 512


class AdmissionError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

# UUIDv7 has a 48-bit millisecond timestamp, a 12-bit monotonic counter and a
# 62-bit random tail.  The lock is deliberately process-local: a sequencer is
# scoped to one request/profile and no cross-turn state is retained.
_last_uuid_ms = -1
_uuid_counter = 0
_uuid_lock = threading.Lock()


def uuid7(*, now_ms: int | None = None) -> str:
    """Return a lowercase, canonical UUIDv7 with same-ms monotonic ordering."""
    global _last_uuid_ms, _uuid_counter
    ms = int(time.time_ns() // 1_000_000 if now_ms is None else now_ms)
    if not 0 <= ms < 1 << 48:
        raise ValueError("UUIDv7 timestamp out of range")
    with _uuid_lock:
        ms = max(ms, _last_uuid_ms)
        counter = _uuid_counter + 1 if ms == _last_uuid_ms else 0
        if counter > 0xFFF:
            ms, counter = ms + 1, 0
        if ms >= 1 << 48:
            raise ValueError("UUIDv7 timestamp exhausted")
        _last_uuid_ms, _uuid_counter = ms, counter
        rand = uuid.uuid4().int & ((1 << 62) - 1)
        value = (ms << 80) | (0x7 << 76) | (counter << 64) | (0x2 << 62) | rand
    return str(uuid.UUID(int=value))


# Descriptive aliases keep call sites readable while the public helper remains
# compatible with tests and the future adapter hook.
generate_uuid7 = uuid7
new_uuid7 = uuid7


def _utf8_size(value: str) -> int:
    if len(value) > MAX_SNAPSHOT_BYTES:
        return MAX_SNAPSHOT_BYTES + 1
    return len(value.encode("utf-8"))


def _frame_data(frame: Mapping[str, Any]) -> MutableMapping[str, Any]:
    data = frame.get("data")
    if isinstance(data, MutableMapping):
        return dict(data)
    return dict(frame)


def _put_frame_data(frame: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(frame)
    if isinstance(frame.get("data"), Mapping):
        out["data"] = dict(data)
        return out
    out.update(data)
    return out


def _text_from(frame: Mapping[str, Any]) -> str | None:
    data = _frame_data(frame)
    for key in ("text", "content", "delta"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


@dataclass
class _Item:
    item_id: str
    index: int
    kind: str
    identity: str
    version: int = 0
    chunks: list[str] = field(default_factory=list)
    text_bytes: int = 0
    snapshot_omitted: bool = False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


@dataclass
class ItemSequencer:
    """Deterministically add item identity and ordering fields to frames.

    ``process`` returns a list because opening/closing a reasoning or text item
    emits lifecycle frames around the original delta.  No method performs I/O
    or depends on the gateway, so callers can replay arbitrary interleavings.
    """

    max_snapshot_bytes: int = MAX_SNAPSHOT_BYTES
    max_items: int = MAX_ITEMS
    identity_fields: Mapping[str, str] = field(default_factory=dict)
    turn_id: str | None = None
    next_index: int = 0
    items: dict[str, _Item] = field(default_factory=dict)
    open_items: dict[str, _Item] = field(default_factory=dict)
    _todo: _Item | None = None
    _delegation: dict[str, _Item] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.max_snapshot_bytes <= MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot limit outside BT envelope")
        if not 0 < self.max_items <= MAX_ITEMS:
            raise ValueError("item limit outside BT envelope")

    def _count(self, name: str) -> None:
        if name.startswith("item_frame_unregistered:"):
            name = "item_frame_unregistered"
        self.counters[name] = self.counters.get(name, 0) + 1

    def _reject(self, reason: str) -> None:
        self._count(f"item_frame_rejected{{reason:{reason}}}")
        raise AdmissionError(reason)

    def _validate_text(self, value: Any) -> str:
        if not isinstance(value, str):
            self._reject("text_invalid")
        if len(value) > self.max_snapshot_bytes:
            self._reject("text_oversize")
        try:
            encoded = value.encode("utf-8")
        except UnicodeError:
            self._reject("text_invalid")
        if len(encoded) > self.max_snapshot_bytes:
            self._reject("text_oversize")
        return value

    def _validate_canonical(self, value: Any) -> str:
        if not isinstance(value, str):
            self._reject("text_invalid")
        if len(value) > self.max_snapshot_bytes:
            return value
        try:
            if len(value.encode("utf-8")) > self.max_snapshot_bytes:
                return value
        except UnicodeError:
            self._reject("text_invalid")
        return value

    def _identity_ok(self, identity: str) -> None:
        try:
            size = len(identity.encode("utf-8"))
        except UnicodeError:
            self._reject("identity_missing")
        if size > MAX_IDENTITY_BYTES:
            self._reject("identity_oversize")

    def _capacity(self, identities: list[str]) -> None:
        new_count = sum(1 for identity in identities if identity not in self.items)
        if self.next_index + new_count > self.max_items:
            self._reject("capacity")

    def _new(self, kind: str, identity: str) -> _Item:
        if len(identity.encode("utf-8")) > MAX_IDENTITY_BYTES:
            self._reject("identity_oversize")
        if self.next_index >= self.max_items:
            self._reject("capacity")
        item = _Item(uuid7(), self.next_index, kind, identity)
        self.next_index += 1
        self.items[identity] = item
        return item

    def _ensure(self, kind: str, identity: str) -> tuple[_Item, bool]:
        item = self.items.get(identity)
        if item is not None:
            if item.kind != kind:
                self._count("item_identity_collision")
                raise ValueError("item identity reused for another kind")
            return item, False
        return self._new(kind, identity), True

    def _lifecycle(self, kind: str, item: _Item, completed: bool = False, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "item.completed" if completed else "item.started",
            "kind": kind,
            "item_id": item.item_id,
            "index": item.index,
        }
        if self.turn_id:
            payload["turn_id"] = self.turn_id
        now_ms = int(time.time() * 1000)
        payload["finished_at" if completed else "started_at"] = now_ms
        payload.update(extra)
        return payload

    def _append(self, item: _Item, text: str | None) -> None:
        if not text or item.snapshot_omitted:
            return
        try:
            delta_bytes = len(text.encode("utf-8"))
        except UnicodeError:
            self._reject("text_invalid")
        if item.text_bytes + delta_bytes > self.max_snapshot_bytes:
            item.chunks = []
            item.text_bytes = 0
            item.snapshot_omitted = True
            self._count("snapshot_omitted")
            return
        item.chunks.append(text)
        item.text_bytes += delta_bytes

    def _close(self, kind: str, *, snapshot: str | None = None, canonical: bool = False) -> list[dict[str, Any]]:
        item = self.open_items.pop(kind, None)
        if item is None:
            return []
        value = snapshot if snapshot is not None else item.text
        omitted = item.snapshot_omitted if snapshot is None else False
        if value is not None and (len(value) > self.max_snapshot_bytes or len(value.encode("utf-8")) > self.max_snapshot_bytes):
            value = None
            omitted = True
            item.snapshot_omitted = True
        payload: dict[str, Any] = self._lifecycle(
            kind,
            item,
            completed=True,
        )
        if value is not None and not omitted:
            payload["text"] = value
        if omitted:
            payload["snapshot_omitted"] = True
        if canonical:
            payload["canonical"] = True
        # Closed snapshots travel in the output only. Keeping them on all
        # historical keys would multiply the per-open-item budget by 2048.
        item.chunks = []
        item.text_bytes = 0
        return [payload]

    def _switch(self, kind: str) -> list[dict[str, Any]]:
        other = "reasoning" if kind == "text" else "text"
        if kind not in self.open_items and self.next_index >= self.max_items:
            self._reject("capacity")
        out = self._close(other)
        if kind not in self.open_items:
            identity = f"{kind}:{len([i for i in self.items.values() if i.kind == kind])}"
            item, created = self._ensure(kind, identity)
            self.open_items[kind] = item
            if created:
                out.append(self._lifecycle(kind, item))
        return out

    def _attach(self, frame: Mapping[str, Any], item: _Item, *, version: int | None = None) -> dict[str, Any]:
        data = _frame_data(frame)
        data["item_id"] = item.item_id
        data["index"] = item.index
        if version is not None:
            data["v"] = version
        return _put_frame_data(frame, data)

    def _extension_identity(self, frame_type: str, data: Mapping[str, Any]) -> str | None:
        path = self.identity_fields.get(frame_type)
        if path == "fixed":
            return f"fixed:{frame_type}"
        if not path:
            return None
        value: Any = data
        for part in path.split("."):
            if not isinstance(value, Mapping):
                return None
            value = value.get(part)
        if value in (None, ""):
            return None
        # The path is the identity namespace. This intentionally lets
        # steer_accepted and steer_dropped share one row when they carry the
        # same steer_id, while unrelated fields cannot collide.
        return f"{path}:{value}"

    def _admit(self, frame: Mapping[str, Any] | str) -> Mapping[str, Any]:
        """Read-only admission pass. No sequencer state is mutated here."""
        if isinstance(frame, str):
            frame = {"type": "text.delta", "text": frame}
        if not isinstance(frame, Mapping):
            self._reject("text_invalid")
        frame_type = frame.get("type")
        data = _frame_data(frame)
        if frame_type in ("reasoning.delta", "text.delta"):
            value = _text_from(frame)
            self._validate_text(value)
            kind = "reasoning" if frame_type == "reasoning.delta" else "text"
            identity = next((i.identity for i in self.items.values() if i.kind == kind and self.open_items.get(kind) is i), None)
            if identity is None:
                identity = f"{kind}:{len([i for i in self.items.values() if i.kind == kind])}"
            self._identity_ok(identity)
            self._capacity([identity])
            if "text" not in data:
                return _put_frame_data(frame, {**data, "text": value})
            return frame
        if frame_type in ("tool.start", "tool.result") or "toolCallId" in data or "tool_call_id" in data:
            tool_id = data.get("toolCallId") or data.get("tool_call_id") or data.get("call_id")
            if not isinstance(tool_id, str) or not tool_id:
                self._reject("identity_missing")
            identity = f"tool:{tool_id}"
            self._identity_ok(identity)
            if frame_type == "tool.start" or data.get("status") == "running":
                self._capacity([identity])
            return frame
        if frame_type in ("hermes.todo", "todo.update"):
            self._capacity(["todo:turn"])
            return frame
        if frame_type in ("hermes.delegation.progress", "delegation.status"):
            child = data.get("subagent_id")
            if not isinstance(child, str) or not child:
                self._reject("identity_missing")
            identity = f"subagent:{child}"
            self._identity_ok(identity)
            self._capacity([identity])
            return frame
        if frame_type in ("item.started", "item.completed"):
            if not isinstance(data.get("item_id"), str) or not isinstance(data.get("index"), int):
                self._reject("identity_missing")
            return frame
        identity = self._extension_identity(str(frame_type), data)
        if identity is None and frame_type in self.identity_fields:
            self._reject("identity_missing")
        if identity is None:
            self._reject("unregistered")
        self._identity_ok(identity)
        self._capacity([identity])
        return frame

    def _process(self, frame: Mapping[str, Any] | str | Any) -> list[Any]:
        """Process one frame and return lifecycle frame(s) plus the frame."""
        if isinstance(frame, str):
            frame = {"type": "text.delta", "text": frame}
        if not isinstance(frame, Mapping):
            self._count("item_frame_unclassified")
            return [frame]
        frame_type = frame.get("type")
        data = _frame_data(frame)
        out: list[dict[str, Any]] = []

        if frame_type in ("reasoning.delta", "text.delta"):
            kind = "reasoning" if frame_type == "reasoning.delta" else "text"
            text = _text_from(frame)
            if text is None:
                self._count("item_frame_rejected")
                return [dict(frame)]
            # Normalize accepted provider aliases into the frozen wire shape.
            if "text" not in data:
                frame = _put_frame_data(frame, {**data, "text": text})
                data = _frame_data(frame)
            try:
                text.encode("utf-8")
            except UnicodeError:
                self._count("item_frame_rejected")
                return [frame]
            out.extend(self._switch(kind))
            item = self.open_items[kind]
            self._append(item, text)
            out.append(self._attach(frame, item))
            return out

        if frame_type in ("tool.start", "tool.result") or "toolCallId" in data or "tool_call_id" in data:
            tool_id = data.get("toolCallId") or data.get("tool_call_id") or data.get("call_id")
            if not isinstance(tool_id, str) or not tool_id:
                self._count("item_frame_rejected")
                return [dict(frame)]
            if frame_type == "tool.start" or data.get("status") == "running":
                if self.next_index >= self.max_items and not any(i.kind == "tool" and i.identity == f"tool:{tool_id}" for i in self.items.values()):
                    self._count("item_index_limit")
                    return [dict(frame)]
            if frame_type == "tool.start" or data.get("status") == "running":
                out.extend(self._close("reasoning"))
                out.extend(self._close("text"))
            identity = f"tool:{tool_id}"
            item, _ = self._ensure("tool", identity)
            out.append(self._attach(frame, item))
            return out

        if frame_type in ("hermes.todo", "todo.update"):
            if self._todo is None:
                self._todo, _ = self._ensure("todo", "todo:turn")
            if self._todo.version + 1 >= MAX_VERSION:
                raise ValueError("item version exhausted")
            self._todo.version += 1
            out.append(self._attach(frame, self._todo, version=self._todo.version))
            return out

        if frame_type in ("hermes.delegation.progress", "delegation.status"):
            child = data.get("subagent_id")
            if not isinstance(child, str) or not child:
                raise ValueError("subagent_id is required")
            identity = f"subagent:{child}"
            item = self._delegation.get(identity)
            if item is None:
                item, _ = self._ensure("subagent", identity)
                self._delegation[identity] = item
            if item.version + 1 >= MAX_VERSION:
                raise ValueError("item version exhausted")
            item.version += 1
            out.append(self._attach(frame, item, version=item.version))
            return out

        if frame_type in ("item.started", "item.completed"):
            item_id = data.get("item_id")
            index = data.get("index")
            if not isinstance(item_id, str) or not isinstance(index, int):
                self._count("item_frame_rejected")
                return []
            out.append(dict(frame))
            return out

        identity = self._extension_identity(str(frame_type), data)
        if identity is None and frame_type in self.identity_fields:
            self._count("item_frame_rejected")
            return [dict(frame)]
        if identity is None:
            self._count(f"item_frame_unregistered:{frame_type}")
            return [dict(frame)]
        item, _ = self._ensure("extension", identity)
        # Extension frames carry ordering only, never item identity.
        out.append(_put_frame_data(frame, {**data, "index": item.index}))
        return out

    def process(self, frame: Mapping[str, Any] | str | Any) -> list[Any]:
        """Process one frame, rejecting malformed/budget-exceeding input."""
        try:
            admitted = self._admit(frame)
            return self._process(admitted)
        except AdmissionError:
            return [frame]
        except (TypeError, ValueError):
            self._count("item_frame_rejected{reason:text_invalid}")
            return [frame]

    def process_many(self, frames: list[Any]) -> list[Any]:
        out: list[Any] = []
        for frame in frames:
            out.extend(self.process(frame))
        return out

    sequence = process
    feed = process

    def open_item(self, kind: str, identity: str) -> dict[str, Any]:
        item, _ = self._ensure(kind, identity)
        self.open_items[kind] = item
        return self._lifecycle(kind, item)

    def close_item(self, kind: str, *, snapshot: str | None = None) -> dict[str, Any] | None:
        frames = self._close(kind, snapshot=snapshot)
        return frames[0] if frames else None

    def complete_canonical(self, text: str) -> list[dict[str, Any]]:
        """Close current text with canonical text, opening a new item if needed."""
        self._validate_canonical(text)
        if "text" not in self.open_items and self.next_index >= self.max_items:
            self._count("item_index_limit")
            return []
        out = self._close("reasoning")
        if "text" in self.open_items:
            out.extend(self._close("text", snapshot=text, canonical=True))
        else:
            item, _ = self._ensure("text", f"text:canonical:{len(self.items)}")
            self.open_items["text"] = item
            out.append(self._lifecycle("text", item))
            out.extend(self._close("text", snapshot=text, canonical=True))
        return out

    def snapshot(self, item_id: str) -> str | None:
        return next((i.text for i in self.items.values() if i.item_id == item_id and not i.snapshot_omitted), None)
