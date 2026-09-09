import json
import queue
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import ZetAgentAdapter
from tools.clarify_tool import clarify_tool
from agent.agent_runtime_helpers import invoke_tool
from tests.run_agent.test_tool_call_guardrail_runtime import _make_agent


def adapter():
    result = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    result._goals = lambda: SimpleNamespace(on_interaction_pending=lambda *a, **k: None, on_interaction_resolved=lambda *a, **k: None)
    return result


def test_old_client_fails_before_publishing_an_input_request():
    runtime = adapter()
    stream = queue.Queue()
    callback = runtime._make_clarify_cb(stream, "s")
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera"}, callback=callback)
    assert "connector_setup_unavailable" in result
    assert stream.empty()
    assert not runtime._clarify_queues


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "submitted"])
async def test_setup_uses_existing_pending_and_response_flow(status):
    runtime = adapter()
    stream = queue.Queue()
    callback = runtime._make_clarify_cb(stream, "s", connector_input_capable=True)
    app = web.Application()
    app.router.add_get('/v1/sessions/{session_id}/pending', runtime._handle_pending)
    app.router.add_post('/v1/sessions/{session_id}/clarify/respond', runtime._handle_clarify_respond)
    headers = {"Authorization": "Bearer test-key"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        agent = _make_agent("clarify")
        agent.clarify_callback = callback
        future = pool.submit(invoke_tool, agent, "clarify", {"question": "ignored", "connector_setup": {"resource_kind": "tv"}}, "setup-flow")
        event = stream.get(timeout=5)
        # Progress queue envelope is owned by the adapter, not model prose.
        payload = event[1] if isinstance(event, tuple) else event
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload["connector_setup"] == {"resource_kind": "tv"}
        async with TestClient(TestServer(app)) as client:
            pending = await (await client.get('/v1/sessions/s/pending', headers=headers)).json()
            assert pending["clarify"]["connector_setup"] == {"resource_kind": "tv"}
            response = await client.post('/v1/sessions/s/clarify/respond', headers=headers, json={
                "clarify_id": payload["clarify_id"], "response": json.dumps({"status": status}),
            })
            assert response.status == 200
        result = json.loads(future.result(timeout=5))
        assert result["status"] == status
        assert "user_response" not in result
        if status == "submitted":
            assert "contains no credentials" in result["next_step"]


@pytest.mark.parametrize('capable', [False, True])
@pytest.mark.parametrize('question', ['请在安全连接卡片的受保护输入框中填写新 PAT 并保存。', '请在安全连接卡中完成公司 Jira 的连接配置。'])
def test_ordinary_clarify_cannot_impersonate_a_secure_input_flow(capable, question):
    runtime = adapter()
    stream = queue.Queue()
    callback = runtime._make_clarify_cb(stream, 's', connector_input_capable=capable)
    result = clarify_tool(
        question,
        callback=callback,
    )
    assert 'connector_setup_required' in result
    assert stream.empty()
    assert not runtime._clarify_queues
