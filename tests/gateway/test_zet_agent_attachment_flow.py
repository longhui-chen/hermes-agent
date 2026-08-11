"""Generic chat attachment emit and action ingress flow coverage."""

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _adapter() -> ZetAgentAdapter:
    return ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))


def _attachment():
    return {
        "id": "att-1",
        "kind": "channel.connect",
        "v": 1,
        "state": "active",
        "payload": {"channel_kind": "feishu"},
        "actions": [{"id": "connect", "style": "primary"}],
        "expires_at": 1785920400000,
    }


def _attachment_payloads(body: str):
    payloads = []
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "event: hermes.tool.progress":
            continue
        for follow in lines[index + 1:index + 4]:
            if follow.startswith("data: "):
                payload = json.loads(follow[len("data: "):])
                if payload.get("type") == "hermes.attachment":
                    payloads.append(payload)
                break
    return payloads


@pytest.mark.asyncio
async def test_plugin_attachment_emit_reaches_real_sse_boundary_flow():
    adapter = _adapter()
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)

    async def _fake_base_run(_self, **kwargs):
        context = PluginContext(
            PluginManifest(name="attachment-test", source="test"),
            PluginManager(),
        )
        # asyncio.to_thread copies ContextVars, matching the real base adapter's
        # copy_context() + executor boundary.
        assert await asyncio.to_thread(context.emit_attachment, _attachment())
        delta = kwargs.get("stream_delta_callback")
        if delta:
            delta("done")
        return (
            {"final_response": "done", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    async with TestClient(TestServer(app)) as client:
        with patch.object(APIServerAdapter, "_run_agent", new=_fake_base_run):
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": "go"}],
                    "stream": True,
                },
            )
            assert response.status == 200
            body = await response.text()

    assert _attachment_payloads(body) == [{
        "type": "hermes.attachment",
        "attachment": _attachment(),
    }]
    context = PluginContext(
        PluginManifest(name="attachment-test", source="test"),
        PluginManager(),
    )
    assert context.emit_attachment(_attachment()) is False


@pytest.mark.asyncio
async def test_attachment_action_endpoint_validates_and_dispatches_hook(monkeypatch):
    adapter = _adapter()
    app = web.Application()
    app.router.add_post(
        "/v1/sessions/{session_id}/attachment/action",
        adapter._handle_attachment_action,
    )
    received = []

    def _capture_hook(name, **kwargs):
        received.append((name, kwargs))
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _capture_hook)
    headers = {"Authorization": "Bearer test-key"}
    async with TestClient(TestServer(app)) as client:
        invalid = await client.post(
            "/v1/sessions/session-1/attachment/action",
            headers=headers,
            json={
                "attachment_id": "att-1",
                "action_id": "connect",
                "action_token": "token-1",
                "payload": [],
            },
        )
        assert invalid.status == 400

        accepted = await client.post(
            "/v1/sessions/session-1/attachment/action",
            headers=headers,
            json={
                "attachment_id": "att-1",
                "action_id": "connect",
                "action_token": "token-1",
                "turn_id": "turn-1",
                "payload": {"source": "card"},
            },
        )
        assert accepted.status == 202
        assert await accepted.json() == {"accepted": True}
        await asyncio.wait_for(adapter._attachment_action_queue.join(), timeout=1)

    await adapter.cancel_background_tasks()
    assert received == [(
        "attachment_action",
        {
            "session_id": "session-1",
            "attachment_id": "att-1",
            "action_id": "connect",
            "action_token": "token-1",
            "turn_id": "turn-1",
            "payload": {"source": "card"},
            "profile_name": "default",
        },
    )]


@pytest.mark.asyncio
async def test_attachment_action_queue_saturation_is_bounded(monkeypatch):
    adapter = _adapter()
    queue = asyncio.Queue(maxsize=1)
    queue.put_nowait(("default", {}))
    adapter._attachment_action_queue = queue
    app = web.Application()
    app.router.add_post(
        "/v1/sessions/{session_id}/attachment/action",
        adapter._handle_attachment_action,
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/sessions/session-1/attachment/action",
            headers={"Authorization": "Bearer test-key"},
            json={
                "attachment_id": "att-1",
                "action_id": "connect",
                "action_token": "token-1",
            },
        )
    assert response.status == 503
    assert not adapter._background_tasks
