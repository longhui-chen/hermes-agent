"""Producer conformance: real frame constructors vs the mirrored golden set.

The manifest test proves every ``payload.type`` literal has a golden; this
file proves the frames those producers actually build stay inside the golden
field sets, and that the state enum a producer emits is the one the contract
document freezes (chat-ui-b0-1-guard-manifests, design D3).
"""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms import zet_agent as zet_agent_module
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import clear_turn_vars, set_turn_vars
from tests.gateway import chat_ui_contract as cuc


@pytest.fixture(scope="module")
def golden():
    return cuc.load_snapshot()["payload"]["golden"]["hermes"]


def _adapter() -> ZetAgentAdapter:
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(
            on_interaction_pending=lambda _sid, **_kwargs: None,
            on_interaction_resolved=lambda _sid: None,
            note_compaction_rotation=lambda _old, _new: None,
        ),
    )
    return adapter


def _take(q: "queue.Queue") -> dict:
    lane, frame = q.get_nowait()
    assert lane == "__tool_progress__"
    return frame


def _assert_within(frame: dict, golden_frame: dict, name: str, path: str = "") -> None:
    """Every key the producer emits exists in the golden — recursively for nested
    objects the golden documents as objects (output, connector_error, browser*)."""
    extra = set(frame) - set(golden_frame)
    assert extra == set(), f"{name}{path}: producer emitted {sorted(extra)} not in golden/hermes/{name}.json"
    for key, value in frame.items():
        if key == "attachment":
            continue  # the attachment envelope is checked against the ChatAttachmentWire table, not one golden sample
        g = golden_frame.get(key)
        if isinstance(value, dict) and isinstance(g, dict) and g:
            _assert_within(value, g, name, f"{path}.{key}")


def _attachment_wire_table() -> dict:
    return cuc.load_snapshot()["payload"]["fields"]["attachment_wire"]


def _assert_attachment_wire(att: dict) -> None:
    table = _attachment_wire_table()
    extra = set(att) - set(table)
    assert extra == set(), f"attachment carries {sorted(extra)} outside ChatAttachmentWire"
    for f, spec in table.items():
        if spec["required"]:
            assert f in att, f"attachment lacks required ChatAttachmentWire field {f}"


# ---------- context.compaction: the state enum is what the producer sends ----------

COMPACTION_STATES = ("started", "succeeded", "failed")


@pytest.mark.parametrize("state", COMPACTION_STATES)
def test_compaction_status_callback_forwards_producer_states(golden, state):
    adapter = _adapter()
    q: "queue.Queue" = queue.Queue()
    status = adapter._make_status_cb(q)
    # The exact shapes agent/conversation_compression.py emits (started / failed / succeeded).
    payload = {"state": state, "message": "compacting", "old_session_id": "sess_old"}
    if state == "started":
        payload.update({"before_messages": 42, "before_tokens": 18000})
    elif state == "failed":
        payload.update({"error": "boom"})
    else:
        payload.update({"new_session_id": "sess_new", "before_messages": 42, "after_messages": 12, "before_tokens": 18000, "after_tokens": 6000})
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        status("context.compaction", payload)
    finally:
        clear_turn_vars(tokens)
    frame = _take(q)
    assert frame["type"] == "context.compaction"
    assert frame["state"] == state
    assert frame["turn_id"] == "turn-1"
    _assert_within(frame, golden["context.compaction"], "context.compaction")


def test_compaction_doc_enum_matches_producer_states():
    """The frozen doc lists exactly the states the producer emits."""
    text = cuc.CONTRACT_DOC.read_text(encoding="utf-8")
    section = text.split("### `context.compaction`", 1)[1].split("### ", 1)[0]
    for state in COMPACTION_STATES:
        assert f"`{state}`" in section, f"doc does not list compaction state {state}"
    for stale in ("`start`", "`running`", "`done`"):
        assert stale not in section, f"doc still lists a state the producer never emits: {stale}"


def test_non_compaction_status_is_not_a_frame():
    adapter = _adapter()
    q: "queue.Queue" = queue.Queue()
    adapter._make_status_cb(q)("lifecycle", "thinking")
    assert q.empty()


# ---------- tool lifecycle frames ----------

def test_tool_completion_success_stays_within_golden(golden):
    frame = api_server._tool_completion_payload("call_1", "web_search", json.dumps({"results": []}))
    assert frame["status"] == "completed" and frame["outcome"] == "success"
    _assert_within(frame, golden["tool-frame.completed"], "tool-frame.completed")


def test_tool_completion_error_uses_snake_case_connector_error(golden):
    result = {
        "error": "connector returned 401",
        "connector_error": {"provider": "google_drive", "status": 401, "code": "unauthorized", "message": "token expired", "nextAction": "reconnect"},
    }
    frame = api_server._tool_completion_payload("call_2", "connector_call", json.dumps(result))
    assert frame["outcome"] == "error"
    assert frame["connector_error"]["nextAction"] == "reconnect"
    assert "connectorError" not in frame
    assert frame["statusCode"] == 401 and frame["provider"] == "google_drive" and frame["errorCode"] == "unauthorized"
    _assert_within(frame, golden["tool-frame.error"], "tool-frame.error")


def test_tool_completion_media_output_stays_within_golden(golden):
    frame = api_server._tool_completion_payload("call_3", "image_generate", json.dumps({"success": True, "host_image": "/tmp/x.png"}))
    assert frame["output"]["success"] is True
    _assert_within(frame, golden["tool-frame.completed"], "tool-frame.completed")


# ---------- single emit point: title / steer_dropped / reasoning ----------

@pytest.mark.parametrize("frame_in,name", [
    ({"type": "conversation.title", "title": "旅行计划"}, "conversation.title"),
    ({"type": "steer_dropped", "text": "leftover"}, "steer_dropped"),
    ({"type": "reasoning.delta", "text": "thinking"}, "reasoning.delta"),
])
def test_put_progress_frames_stay_within_golden(golden, frame_in, name):
    q: "queue.Queue" = queue.Queue()
    tokens = set_turn_vars(turn_id="turn-9")
    try:
        zet_agent_module._put_progress(q, frame_in)
    finally:
        clear_turn_vars(tokens)
    frame = _take(q)
    assert frame["turn_id"] == "turn-9"
    _assert_within(frame, golden[name], name)


# ---------- hermes.attachment: the real emitters ----------

def test_plugin_attachment_emitter_frames_stay_within_golden(golden):
    adapter = _adapter()
    q: "queue.Queue" = queue.Queue()
    emit = adapter._build_attachment_emitter(q)
    att = {"id": "cg-1", "category": "interactive", "kind": "connector.connect", "v": 1, "state": "active",
           "payload": {"provider": "notion", "blocking": False}, "actions": [{"id": "connect", "style": "primary"}]}
    tokens = set_turn_vars(turn_id="turn-a")
    try:
        assert emit(att) is True
    finally:
        clear_turn_vars(tokens)
    frame = _take(q)
    assert frame["turn_id"] == "turn-a"
    _assert_within(frame, golden["hermes.attachment"], "hermes.attachment")
    _assert_attachment_wire(frame["attachment"])
    # detached copy: mutating the plugin's dict after emit must not reach the queued frame
    att["payload"]["provider"] = "changed"
    assert frame["attachment"]["payload"]["provider"] == "notion"


def test_plugin_attachment_emitter_rejects_oversized_and_non_dict(golden):
    adapter = _adapter()
    q: "queue.Queue" = queue.Queue()
    emit = adapter._build_attachment_emitter(q)
    assert emit("not-a-dict") is False
    huge = {"id": "x", "kind": "memory.saved", "v": 1, "state": "active", "payload": {"blob": "x" * (adapter._ATTACHMENT_MAX_BYTES + 1)}}
    assert emit(huge) is False
    assert q.empty()


@pytest.mark.parametrize("pusher,kind", [("_push_memory_citations", "memory.citations"), ("_push_memory_saved", "memory.saved")])
def test_memory_attachment_pushers_stay_within_golden(golden, pusher, kind):
    q: "queue.Queue" = queue.Queue()
    items = [{"id": "m1", "source": "notes/a.md", "excerpt": "…"}]
    tokens = set_turn_vars(turn_id="turn-m")
    try:
        assert getattr(ZetAgentAdapter, pusher)(q, "turn-m", "sess", items) is True
    finally:
        clear_turn_vars(tokens)
    frame = _take(q)
    assert frame["attachment"]["kind"] == kind
    _assert_within(frame, golden["hermes.attachment"], "hermes.attachment")
    _assert_attachment_wire(frame["attachment"])
    assert frame["attachment"]["kind"] in golden_attachment_kinds()


def golden_attachment_kinds() -> set:
    return set(cuc.load_snapshot()["payload"]["golden"]["attachments"])


# ---------- hermes.delegation.progress: the real callback, every relayed field ----------

def test_delegation_progress_callback_forwards_every_identity_field(golden):
    q: "queue.Queue" = queue.Queue()
    cb = ZetAgentAdapter._make_delegation_progress_cb(q)
    identity = {f: (1 if f in ("task_index", "task_count", "depth", "tool_count") else 2.5 if f == "duration_seconds" else "v")
                for f in ZetAgentAdapter._DELEGATION_PROGRESS_FIELDS}
    tokens = set_turn_vars(turn_id="turn-d")
    try:
        cb("subagent.tool", "terminal", "ls", None, **identity)
        cb("subagent_progress", "summary text", None, None, **identity)
        cb("not.a.delegation.event", "x", "y", None, **identity)
    finally:
        clear_turn_vars(tokens)
    first = _take(q)
    second = _take(q)
    assert q.empty(), "unknown events must not be forwarded"
    for frame in (first, second):
        assert frame["type"] == "hermes.delegation.progress" and frame["turn_id"] == "turn-d"
        _assert_within(frame, golden["hermes.delegation.progress"], "hermes.delegation.progress")
    assert set(ZetAgentAdapter._DELEGATION_PROGRESS_FIELDS) <= set(golden["hermes.delegation.progress"]), "every relayed identity field is frozen in the golden"
    assert second["event"] == "subagent.progress" and second["preview"] == "summary text"


# ---------- event: hermes.error ----------

@pytest.mark.parametrize("recoverable", [True, False])
def test_chat_stream_error_payload_stays_within_golden(golden, recoverable):
    result = {"completed": False, "partial": True, "failed": False, "error": "boom",
              "provider_error": {"code": "rate_limited", "reason": "quota", "provider": "p", "model": "m", "status_code": 429,
                                 "provider_error_code": "429", "provider_message": "slow down", "recoverable": recoverable}}
    frame = api_server._chat_stream_error_payload(result, "error")
    assert frame is not None and frame["recoverable"] is recoverable and frame["code"] == "rate_limited"
    _assert_within(frame, golden["hermes-error"], "hermes-error")
    truncated = api_server._chat_stream_error_payload({"completed": True, "partial": True}, "length")
    assert truncated["code"] == "output_truncated"
    _assert_within(truncated, golden["hermes-error"], "hermes-error")


def test_tool_completion_failed_media_output_stays_within_golden(golden):
    frame = api_server._tool_completion_payload("call_9", "image_generate", json.dumps({"success": False, "error": "quota"}))
    assert frame["outcome"] == "error" and frame["output"] == {"success": False}
    _assert_within(frame, golden["tool-frame.error"], "tool-frame.error")
