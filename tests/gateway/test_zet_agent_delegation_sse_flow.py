"""Delegation progress → chat-completions SSE wire framing (flow test).

The unit tests in ``test_zet_agent_delegation_progress.py`` stop at the
in-memory ``stream_q``; this flow test covers the remaining in-repo glue on
the hot path (HR#4): the REAL ``_make_delegation_progress_cb`` payload, the
REAL ``_sniff_stream_q`` discovery used by ``_create_agent``'s 3d wiring, and
the REAL ``/v1/chat/completions`` SSE writer — asserting the exact wire
framing (``event: hermes.tool.progress`` + ``type=hermes.delegation.progress``
payload fields) that local-server's translator consumes. Cross-repo, the
same field names are pinned by local-server's translator tests and the App's
converter tests; the device end-to-end pass lives in the release validation
matrix.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import patch

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import ZetAgentAdapter


def _wire_adapter():
    return ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))


def _app(adapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _progress_payloads(body: str):
    """Collect hermes.delegation.progress payloads from the SSE body."""
    out = []
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "event: hermes.tool.progress":
            continue
        for follow in lines[i + 1: i + 4]:
            if follow.startswith("data: "):
                payload = json.loads(follow[len("data: "):])
                if payload.get("type") == "hermes.delegation.progress":
                    out.append(payload)
                break
    return out


@pytest.mark.asyncio
async def test_delegation_progress_reaches_sse_wire_flow():
    adapter = _wire_adapter()
    app = _app(adapter)
    async with TestClient(TestServer(app)) as cli:

        async def _mock_run_agent(**kwargs):
            # Same discovery _create_agent's 3d wiring performs, against the
            # REAL callbacks the chat-completions handler passed in.
            stream_q = adapter._sniff_stream_q(
                kwargs.get("tool_start_callback"),
                kwargs.get("tool_complete_callback"),
                kwargs.get("stream_delta_callback"),
            )
            assert stream_q is not None, "sniff must find the handler's stream_q"
            cb = ZetAgentAdapter._make_delegation_progress_cb(stream_q)
            cb(
                "subagent.start",
                subagent_id="sub_1",
                task_index=0,
                goal="research X",
                status="running",
                child_session_id="zettlab:u1:agentB:a2a-agentA-cafe0001",
            )
            cb(
                "subagent.complete",
                subagent_id="sub_1",
                task_index=0,
                status="completed",
                duration_seconds=1.5,
            )
            delta = kwargs.get("stream_delta_callback")
            if delta:
                delta("done.")
            return (
                {"final_response": "done.", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
            resp = await cli.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "go"}],
                    "stream": True,
                },
            )
            assert resp.status == 200
            body = await resp.text()

    events = _progress_payloads(body)
    assert len(events) == 2, f"expected start+complete on the wire, got {events}"
    start, complete = events
    # Field names ARE the contract local-server translates into chatproto
    # delegation events — a rename here must fail this test, not ship dark.
    assert start["event"] == "subagent.start"
    assert start["kind"] == "delegation"
    assert start["subagent_id"] == "sub_1"
    assert start["task_index"] == 0
    assert start["goal"] == "research X"
    assert start["status"] == "running"
    assert start["child_session_id"] == "zettlab:u1:agentB:a2a-agentA-cafe0001"
    assert complete["event"] == "subagent.complete"
    assert complete["status"] == "completed"
    assert complete["duration_seconds"] == 1.5
