"""Randomized BT-T1 invariants for the unconnected ItemSequencer."""
from __future__ import annotations

import random

from gateway.platforms.item_sequencer import MAX_SNAPSHOT_BYTES, ItemSequencer, uuid7


def _data(frame):
    return frame.get("data", frame)


def _assert_indices(frames):
    seen = {}
    maximum = -1
    for frame in frames:
        data = _data(frame)
        if "index" not in data:
            continue
        key = data.get("item_id") or (data.get("type"), data.get("call_id"), data.get("subagent_id"), data.get("index"))
        index = data["index"]
        if key in seen:
            assert index == seen[key]
        else:
            assert index > maximum
            seen[key] = index
            maximum = index
    assert len(set(seen.values())) <= 2048


def test_random_interleaving_preserves_i1_and_versions():
    for seed in range(40):
        rng = random.Random(seed)
        seq = ItemSequencer(identity_fields={"hermes.attachment": "attachment.id", "context.compaction": "fixed"})
        frames = []
        for _ in range(80):
            choice = rng.randrange(7)
            if choice == 0:
                frames.extend(seq.process({"type": "reasoning.delta", "text": "思考"}))
            elif choice == 1:
                frames.extend(seq.process({"type": "text.delta", "text": "答复"}))
            elif choice == 2:
                frames.extend(seq.process({"type": "tool.start", "call_id": f"c{rng.randrange(8)}"}))
            elif choice == 3:
                frames.extend(seq.process({"type": "tool.result", "call_id": f"c{rng.randrange(8)}"}))
            elif choice == 4:
                frames.extend(seq.process({"type": "hermes.todo", "todos": []}))
            elif choice == 5:
                frames.extend(seq.process({"type": "hermes.delegation.progress", "subagent_id": f"s{rng.randrange(3)}"}))
            else:
                frames.extend(seq.process({"type": "hermes.attachment", "attachment": {"id": f"a{rng.randrange(3)}"}}))
        _assert_indices(frames)
        for item in seq._todo, *seq._delegation.values():
            if item:
                assert item.version >= 1


def test_text_can_resume_after_delegation_without_new_item():
    seq = ItemSequencer()
    frames = seq.process_many([
        {"type": "text.delta", "text": "a"},
        {"type": "hermes.delegation.progress", "subagent_id": "child"},
        {"type": "text.delta", "text": "b"},
    ])
    text = [_data(f) for f in frames if f.get("type") == "text.delta"]
    assert text[0]["item_id"] == text[1]["item_id"]
    assert text[0]["index"] == text[1]["index"]


def test_utf8_snapshot_limit_is_enforced_at_append_and_stays_bounded():
    seq = ItemSequencer()
    frames = seq.process({"type": "text.delta", "text": "🙂" * (MAX_SNAPSHOT_BYTES // 4 + 2)})
    assert frames
    assert not seq.open_items
    assert seq.counters.get("item_frame_rejected{reason:text_oversize}") == 1
    assert seq.counters.get("item_frame_rejected{reason:text_oversize}") == 1


def test_uuid7_is_canonical_and_monotonic_same_millisecond():
    values = [uuid7(now_ms=1700000000000) for _ in range(100)]
    assert values == sorted(values)
    assert all(len(value) == 36 and value[14] == "7" for value in values)


def test_canonical_after_tool_opens_distinct_text_item():
    seq = ItemSequencer()
    seq.process({"type": "text.delta", "text": "interim"})
    seq.process({"type": "tool.start", "call_id": "tool"})
    out = seq.complete_canonical("final")
    completed = [f for f in out if f.get("type") == "item.completed"]
    assert completed[-1]["canonical"] is True
    assert completed[-1]["kind"] == "text"


def test_registered_extension_identity_is_shared_across_frame_variants():
    seq = ItemSequencer(identity_fields={"steer_accepted": "steer_id", "steer_dropped": "steer_id"})
    frames = seq.process_many([
        {"type": "steer_accepted", "steer_id": "s"},
        {"type": "steer_dropped", "steer_id": "s"},
    ])
    assert frames[0]["index"] == frames[1]["index"]


def test_canonical_limit_rejects_before_state_change():
    seq = ItemSequencer()
    seq.process({"type": "text.delta", "text": "hello"})
    assert seq.complete_canonical("x" * (MAX_SNAPSHOT_BYTES + 1)) == []
    assert "text" in seq.open_items
    assert seq.counters["item_frame_rejected{reason:text_oversize}"] == 1


def test_near_2048_identity_envelope_rejects_the_2049th_key():
    seq = ItemSequencer(identity_fields={"hermes.attachment": "attachment.id"})
    emitted = []
    for i in range(2048):
        emitted.extend(seq.process({"type": "hermes.attachment", "attachment": {"id": f"a{i}"}}))
    assert seq.next_index == 2048
    rejected = seq.process({"type": "hermes.attachment", "attachment": {"id": "overflow"}})
    assert rejected == [{"type": "hermes.attachment", "attachment": {"id": "overflow"}}]
    assert seq.counters["item_frame_rejected{reason:capacity}"] == 1


def test_missing_registered_identity_is_rejected_without_consuming_index():
    seq = ItemSequencer(identity_fields={"hermes.attachment": "attachment.id"})
    frame = {"type": "hermes.attachment", "attachment": {}}
    assert seq.process(frame) == [frame]
    assert seq.next_index == 0
    assert seq.counters["item_frame_rejected{reason:identity_missing}"] == 1


def test_index_exhaustion_does_not_close_open_text_or_raise():
    seq = ItemSequencer(max_items=1)
    first = seq.process({"type": "text.delta", "text": "open"})
    second = {"type": "reasoning.delta", "text": "late"}
    assert seq.process(second) == [second]
    assert "text" in seq.open_items
    assert first[0]["type"] == "item.started"


def test_canonical_after_index_exhaustion_uses_existing_open_item_or_fallback():
    seq = ItemSequencer(max_items=1)
    seq.process({"type": "tool.start", "call_id": "tool"})
    assert seq.complete_canonical("final") == []
    assert seq.counters["item_frame_rejected{reason:capacity}"] == 1


def test_tool_without_call_id_is_rejected_before_closing_text():
    seq = ItemSequencer(max_items=2)
    seq.process({"type": "text.delta", "text": "open"})
    frame = {"type": "tool.start", "name": "bad"}
    assert seq.process(frame) == [frame]
    assert "text" in seq.open_items


def test_invalid_unicode_is_rejected_without_opening_item():
    seq = ItemSequencer()
    frame = {"type": "text.delta", "text": "bad\ud800"}
    assert seq.process(frame) == [frame]
    assert not seq.open_items


def test_random_rejections_are_atomic_and_use_fixed_reason_buckets():
    cases = [
        ("capacity", lambda s: (setattr(s, "next_index", s.max_items), s.process({"type": "text.delta", "text": "x"}))),
        ("identity_missing", lambda s: s.process({"type": "tool.start", "name": "x"})),
        ("identity_oversize", lambda s: s.process({"type": "tool.start", "call_id": "x" * 600})),
        ("text_invalid", lambda s: s.process({"type": "text.delta", "text": "bad\ud800"})),
        ("text_oversize", lambda s: s.process({"type": "text.delta", "text": "x" * (MAX_SNAPSHOT_BYTES + 1)})),
        ("unregistered", lambda s: s.process({"type": "new.extension", "id": "x"})),
    ]
    for reason, operation in cases:
        seq = ItemSequencer()
        before = __import__("copy").deepcopy(seq.__dict__)
        operation(seq)
        after = __import__("copy").deepcopy(seq.__dict__)
        if reason != "capacity":
            assert after["next_index"] == before["next_index"]
            assert after["items"] == before["items"]
        assert seq.counters.get(f"item_frame_rejected{{reason:{reason}}}") == 1


def test_incremental_utf8_encoding_is_linear_for_single_character_deltas():
    class CountingText(str):
        calls = 0
        def encode(self, *args, **kwargs):
            type(self).calls += 1
            return super().encode(*args, **kwargs)
    seq = ItemSequencer()
    for _ in range(1000):
        seq.process({"type": "text.delta", "text": CountingText("x")})
    assert CountingText.calls <= 4200


def test_invalid_text_alias_is_normalized_to_string_text_field():
    seq = ItemSequencer()
    output = seq.process({"type": "text.delta", "text": None, "content": "alias"})
    delta = next(frame for frame in output if frame.get("type") == "text.delta")
    assert delta["text"] == "alias"
