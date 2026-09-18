"""B1 Hermes producer conformance against the pinned root golden snapshot."""

import json
from pathlib import Path

import pytest

from gateway.platforms.zet_agent import _clarify_sentinel


SNAPSHOT = Path(__file__).parents[2] / "schemas" / "chat-ui-golden.snapshot.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(SNAPSHOT.read_text())["payload"]["golden"]["hermes"]


@pytest.mark.parametrize(
    "kind, state, reason",
    [
        (kind, state, reason)
        for kind in ("clarify", "approval")
        for state, reason in (
            ("expired", "timeout"),
            ("cancelled", "turn_interrupted"),
            ("cancelled", "session_reset"),
            ("cancelled", "delivery_failed"),
        )
    ],
)
def test_terminal_variant_is_registered_and_bounded(golden, kind, state, reason):
    key = f"hermes.{kind}.{state}.{reason}"
    assert key in golden
    payload = golden[key]
    assert payload["type"] == f"hermes.{kind}"
    assert payload["state"] == state
    assert payload["state_reason"] == reason
    sentinel = _clarify_sentinel(kind, "a" * 64, state, reason, "clarify could not be delivered")
    assert len(sentinel.encode("utf-8")) <= 160



def test_real_producers_and_writer_share_identity_order(golden, monkeypatch):
    """Callbacks enqueue raw frames; only the drain assigns stable identities."""
    from queue import Queue
    from types import SimpleNamespace
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from gateway.platforms.zet_agent_bt import WriterProjection, tool_callbacks

    monkeypatch.setattr("gateway.session_context.get_session_env", lambda name, default="": "turn-proof" if name == "HERMES_TURN_ID" else default)
    adapter = object.__new__(ZetAgentAdapter)
    queue = Queue()
    queue.put(("__tool_progress__", {"type": "reasoning.delta", "text": "考虑"}))
    queue.put("先说明。")
    start, complete = tool_callbacks(queue)
    start("call-proof", "search_files", {"query": "public"})
    todo = adapter._make_todo_emit_cb(queue)
    todo([], {})
    todo([], {"done": 0})
    delegation = adapter._make_delegation_progress_cb(queue)
    delegation("subagent.start", subagent_id="child-proof", status="running")
    delegation("subagent.complete", subagent_id="child-proof", status="done")
    assert adapter._build_attachment_emitter(queue)({"id": "attachment-proof", "kind": "memory.citations", "v": 1, "state": "active", "payload": {"items": []}})
    adapter._push_title(queue, "标题")
    adapter._make_plan_emit_cb(queue, SimpleNamespace())("计划", [], "plan-proof")
    adapter._make_status_cb(queue)("context.compaction", {"state": "started", "message": "压缩"})
    complete("call-proof", "search_files", {}, {"success": True})
    queue.put("答复。")

    projection = WriterProjection(turn_id="turn-proof")
    emitted = []
    while not queue.empty():
        raw = queue.get_nowait()
        if isinstance(raw, tuple):
            assert "index" not in raw[1]
        emitted.extend(projection.project(raw))
    emitted.extend(projection.finish({"completed": True}))

    indices, versions = {}, {}
    seen_types = set()
    tool_indices = []
    for value in emitted:
        if isinstance(value, str):
            payload = value.wire_fields["hermes"]
            assert set(payload) == set(golden["content-chunk"]["hermes"])
            key = payload["item_id"]
        else:
            _, payload = value
            kind = payload.get("type")
            seen_types.add(kind)
            sample = kind or ("tool-frame.completed" if payload["status"] == "completed" else "tool-frame.running")
            assert set(payload) <= set(golden[sample]), (sample, set(payload) - set(golden[sample]))
            key = payload.get("item_id") or kind
            if kind == "hermes.attachment":
                key = payload["attachment"]["id"]
            if kind is None:
                tool_indices.append(payload["index"])
            if "v" in payload:
                assert payload["v"] > versions.get(key, 0)
                versions[key] = payload["v"]
        if key in indices:
            assert payload["index"] == indices[key]
        else:
            assert payload["index"] > max(indices.values(), default=-1)
            indices[key] = payload["index"]
    assert tool_indices[0] == tool_indices[1]
    assert {"reasoning.delta", "item.started", "item.completed", "hermes.todo", "hermes.delegation.progress", "hermes.attachment", "conversation.title", "hermes.plan", "context.compaction"} <= seen_types
    assert "subagent_id" in ZetAgentAdapter._DELEGATION_PROGRESS_FIELDS


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_adapter_finally_counts_post_sentinel_frames(monkeypatch, abort):
    from queue import Queue
    from types import SimpleNamespace
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from gateway.platforms.zet_agent_bt import projection_context

    adapter = object.__new__(ZetAgentAdapter)
    monkeypatch.setattr(adapter, "_register_active_session_turn", lambda *args: None)
    monkeypatch.setattr(adapter, "_clear_active_session_turn", lambda *args: None)
    queue = Queue()
    queue.put(None)
    queue.put(("__tool_progress__", {"type": "conversation.title", "title": "late"}))
    captured = []

    async def drain(*args, **kwargs):
        captured.append(projection_context.get())
        assert queue.get_nowait() is None
        if abort:
            raise ConnectionResetError("reader left")

    monkeypatch.setattr(APIServerAdapter, "_write_sse_chat_completion", drain)
    try:
        await adapter._write_sse_chat_completion(SimpleNamespace(), "id", "model", 1, queue, None)
    except ConnectionResetError:
        assert abort
    assert captured[0].sequencer.counters["item_frame_stranded"] == 1
    with pytest.raises(LookupError):
        projection_context.get()


def test_unregistered_and_unclassified_fallbacks_are_counted():
    from gateway.platforms.zet_agent_bt import WriterProjection
    projection = WriterProjection()
    unknown = ("__tool_progress__", {"type": "future.extension"})
    malformed = ("__tool_progress__", [])
    assert projection.project(unknown) == [unknown]
    assert projection.project(malformed) == [malformed]
    assert projection.sequencer.counters["item_frame_unregistered"] == 1
    assert projection.sequencer.counters["item_frame_unclassified"] == 1
    assert projection.sequencer.next_index == 0


def test_actual_compaction_producer_calls_cover_registered_shape(monkeypatch, golden):
    """Execute the three upstream status calls through the real adapter callback."""
    import ast
    from queue import Queue
    from types import SimpleNamespace
    from unittest.mock import Mock
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from gateway.platforms.zet_agent_bt import WriterProjection

    monkeypatch.setattr("gateway.session_context.get_session_env", lambda name, default="": "turn-proof" if name == "HERMES_TURN_ID" else default)
    adapter = object.__new__(ZetAgentAdapter)
    monkeypatch.setattr(adapter, "_register_interaction_route_alias", Mock())
    monkeypatch.setattr(adapter, "_goals", lambda: SimpleNamespace(note_compaction_rotation=Mock()))
    queue = Queue()
    producer = SimpleNamespace(session_id="new-session", _emit_structured_status=adapter._make_status_cb(queue))
    env = {"agent": producer, "_old_session_id": "old-session", "_pre_msg_count": 42,
           "approx_tokens": 18000, "compressed": ["summary"], "_compressed_est": 100,
           "_compress_exc": ValueError("unavailable")}
    tree = ast.parse((SNAPSHOT.parents[1] / "agent/conversation_compression.py").read_text())
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_emit_structured_status" and len(call.args) == 2
                and isinstance(call.args[0], ast.Constant) and call.args[0].value == "context.compaction"):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=call)], type_ignores=[])), "<compaction-producer>", "exec"), env)
    projection = WriterProjection()
    frames = [projection.project(queue.get_nowait())[0][1] for _ in range(queue.qsize())]
    assert {frame["state"] for frame in frames} == {"started", "failed", "succeeded"}
    assert {frame["index"] for frame in frames} == {0}
    assert set().union(*(set(frame) for frame in frames)) == set(golden["context.compaction"])


def test_randomized_registered_emitters_preserve_identity_indices(golden):
    import copy
    import random
    from queue import Queue
    from gateway.platforms.zet_agent import _put_progress, _put_named_steer_progress, _STEER_SOURCE_TOKEN
    from gateway.platforms.zet_agent_bt import WriterProjection, identity_fields
    from gateway.platforms.item_sequencer import uuid7

    registry = identity_fields()
    rng = random.Random(473)
    for _ in range(12):
        kinds = list(registry) * 3
        rng.shuffle(kinds)
        projection = WriterProjection()
        queue = Queue()
        seen = {}
        steer_id = uuid7()
        for kind in kinds:
            payload = copy.deepcopy(golden[kind])
            payload.pop("index")
            if kind in {"steer_accepted", "steer_dropped"}:
                payload.update(steer_id=steer_id, text="修改要求")
                _put_named_steer_progress(queue, payload, source_token=_STEER_SOURCE_TOKEN, nonblocking=False)
            else:
                assert _put_progress(queue, payload)
            tag, emitted = projection.project(queue.get_nowait())[0]
            assert tag == "__tool_progress__"
            assert set(emitted) <= set(golden[kind])
            path = registry[kind]
            value = kind
            if path != "fixed":
                value = emitted
                for component in path.split("."):
                    value = value[component]
                assert isinstance(value, str) and value
            identity = (path, value)
            if identity in seen:
                assert emitted["index"] == seen[identity]
            else:
                assert emitted["index"] > max(seen.values(), default=-1)
                seen[identity] = emitted["index"]
        assert projection.sequencer.counters == {}


def test_saturated_producers_reject_and_count_without_enqueuing():
    from unittest.mock import Mock
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from gateway.platforms import zet_agent_metrics

    adapter = object.__new__(ZetAgentAdapter)
    queue = Mock()
    queue.qsize.return_value = 2001
    before = zet_agent_metrics.snapshot()
    assert not adapter._build_attachment_emitter(queue)({"id": "overload", "kind": "memory.citations"})
    adapter._make_delegation_progress_cb(queue)("subagent.start", subagent_id="child")
    queue.put.assert_not_called()
    after = zet_agent_metrics.snapshot()
    for kind in ("attachment", "subagent"):
        key = f"item_frame_dropped_backlog{{kind={kind}}}"
        assert after[key] == before.get(key, 0) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", [False, True])
async def test_bt_counters_reach_existing_health_from_request_writers(monkeypatch, abort):
    """Normal/failed writers export process totals without retaining profile IDs."""
    from queue import Queue
    from types import SimpleNamespace
    from unittest.mock import Mock
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms import api_server, zet_agent_metrics as metrics
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from gateway.platforms.zet_agent_bt import projection_context

    metrics.reset_for_tests()
    adapter = object.__new__(ZetAgentAdapter)
    monkeypatch.setattr(adapter, "_register_active_session_turn", lambda *args: None)
    monkeypatch.setattr(adapter, "_clear_active_session_turn", lambda *args: None)
    projections = []

    async def drain(*args, **kwargs):
        projection = projection_context.get()
        projections.append(projection)
        projection.project("hello")
        projection.project(17)
        projection.project(("__tool_progress__", {"type": "unknown.secret-profile"}))
        queue = args[5]
        assert queue.get_nowait() is None
        if abort:
            raise ConnectionResetError("reader left")

    monkeypatch.setattr(api_server.APIServerAdapter, "_write_sse_chat_completion", drain)
    for _ in range(2):
        queue = Queue()
        queue.put(None)
        queue.put(("__tool_progress__", {"type": "conversation.title", "title": "late"}))
        try:
            await adapter._write_sse_chat_completion(SimpleNamespace(), "id", "model", 1, queue, None)
        except ConnectionResetError:
            assert abort
    assert projections[0] is not projections[1]
    assert all(p.sequencer.next_index == 1 for p in projections)
    assert all(p.sequencer.counters["item_frame_unregistered"] == 1 for p in projections)
    queue = Mock()
    queue.qsize.return_value = 2001
    assert not adapter._build_attachment_emitter(queue)({"id": "overload", "kind": "memory.citations"})
    adapter._make_delegation_progress_cb(queue)("subagent.start", subagent_id="child")
    queue.put.assert_not_called()

    monkeypatch.setattr(adapter, "_check_auth", lambda request: None)
    monkeypatch.setattr(adapter, "_readiness_work_counts", lambda: (0, 0, 0))
    monkeypatch.setattr("gateway.status.read_runtime_status", lambda: {})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "test/model")
    monkeypatch.setattr(api_server, "collect_runtime_readiness", lambda **kwargs: {"status": "ok"})
    route_table = adapter._http_route_table()
    assert ("GET", "/health/detailed", adapter._handle_health_detailed) in route_table
    assert not any(path == "/internal/hermes/health/detailed" for _, path, _ in route_table)
    app = web.Application()
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/health/detailed")
        assert response.status == 200
        body = await response.json()
        assert body["version"]
        actual = body["interaction_metrics"]
    assert actual == {
        "item_frame_stranded": 2,
        "item_frame_unregistered": 2,
        "item_frame_unclassified": 2,
        "item_frame_dropped_backlog{kind=attachment}": 1,
        "item_frame_dropped_backlog{kind=subagent}": 1,
        "item_frame_dropped_backlog{kind=other}": 0,
    }
