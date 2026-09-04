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


def _assert_within(frame: dict, golden_frame: dict, name: str) -> None:
    extra = set(frame) - set(golden_frame)
    assert extra == set(), f"{name}: producer emitted {sorted(extra)} not in golden/hermes/{name}.json"


# ---------- context.compaction: the state enum is what the producer sends ----------

COMPACTION_STATES = ("started", "succeeded", "failed")


@pytest.mark.parametrize("state", COMPACTION_STATES)
def test_compaction_status_callback_forwards_producer_states(golden, state):
    adapter = _adapter()
    q: "queue.Queue" = queue.Queue()
    status = adapter._make_status_cb(q)
    payload = {"state": state, "message": "compacting"}
    if state == "succeeded":
        payload.update({"old_session_id": "sess_old", "new_session_id": "sess_new"})
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
