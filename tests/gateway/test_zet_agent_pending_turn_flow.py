import queue
import threading
from collections import OrderedDict
from types import SimpleNamespace

from gateway.platforms import zet_agent as zet_agent_module
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import clear_turn_vars, set_turn_vars
from tools import approval


def _adapter_for_pending_callbacks() -> ZetAgentAdapter:
    adapter = object.__new__(ZetAgentAdapter)
    adapter._pending_lock = threading.Lock()
    adapter._pending_approval = {}
    adapter._pending_clarify = {}
    adapter._pending_mirror_meta = OrderedDict()
    adapter._pending_mirror_bytes = 0
    adapter._clarify_state_lock = threading.Lock()
    adapter._clarify_queues = {}
    adapter._delivery_lock = threading.Lock()
    adapter._interaction_deliveries = OrderedDict()
    adapter._published_interaction_generations = OrderedDict()
    adapter._committed_interaction_generations = {}
    adapter._provisional_interaction_generations = {}
    adapter._durable_session_digests = OrderedDict()
    adapter._durable_source_pin_count = 0
    adapter._prepared_raw_bytes = 0
    adapter._goals = lambda: SimpleNamespace(
        on_interaction_pending=lambda _sid, **_kwargs: None
    )
    return adapter


def test_pending_interaction_flow_carries_exact_turn_id(monkeypatch):
    adapter = _adapter_for_pending_callbacks()
    stream_q = queue.Queue()
    monkeypatch.setattr(zet_agent_module, "CLARIFY_RESPONSE_TIMEOUT", 0.01)
    tokens = set_turn_vars(turn_id="turn-video-42")
    approval_data = {
        "command": "render",
        "description": "Render video",
    }
    queue_key = adapter._interaction_queue_key("session-1")
    approval.enqueue_gateway_approval(queue_key, approval_data)
    try:
        adapter._make_approval_cb(stream_q, "session-1")(approval_data)
        clarify_result = adapter._make_clarify_cb(stream_q, "session-1")(
            "Choose quality", ["high", "medium"]
        )
    finally:
        clear_turn_vars(tokens)
        approval.resolve_gateway_approval(queue_key, "deny")

    assert clarify_result == ""
    approval_event = stream_q.get_nowait()[1]
    clarify_event = stream_q.get_nowait()[1]
    assert approval_event["turn_id"] == "turn-video-42"
    assert clarify_event["turn_id"] == "turn-video-42"
    assert approval_event["interaction_delivery_version"] == 1
    assert clarify_event["interaction_delivery_version"] == 1
