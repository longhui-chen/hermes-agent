import asyncio
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
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera", "live": {"camera_id": "cam-1"}}, callback=callback)
    assert "connector_setup_unavailable" in result
    assert stream.empty()
    assert not runtime._clarify_queues


@pytest.mark.asyncio
async def test_camera_presentation_capability_is_request_local_and_resets_on_failure(monkeypatch):
    from gateway.platforms.zet_agent import _zettlab_camera_observation_input_capable

    runtime = adapter()

    async def handle(request, _handler):
        expected = request.headers.get("X-Zettlab-Camera-Observation-Input") == "1"
        assert _zettlab_camera_observation_input_capable.get() is expected
        await asyncio.sleep(0)
        assert _zettlab_camera_observation_input_capable.get() is expected
        if request.headers.get("Fail"):
            raise RuntimeError("request_failed")
        return web.Response()

    monkeypatch.setattr(runtime, "_handle_with_zettlab_identity", handle)
    await asyncio.gather(*[
        runtime._handle_chat_completions(SimpleNamespace(headers=headers))
        for headers in [{}, {"X-Zettlab-Camera-Observation-Input": "1"}, {"X-Zettlab-Camera-Observation-Input": "true"}]
    ])
    with pytest.raises(RuntimeError, match="request_failed"):
        await runtime._handle_chat_completions(SimpleNamespace(headers={"X-Zettlab-Camera-Observation-Input": "1", "Fail": "1"}))
    assert _zettlab_camera_observation_input_capable.get() is False


def test_observation_publication_stays_closed_until_confirmation_consumers_are_ready():
    runtime = adapter()
    stream = queue.Queue()
    callback = runtime._make_clarify_cb(stream, "s", connector_input_capable=True)
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera", "observation": {
        "camera_id": "cam-1", "duration_seconds": 60, "subject_kind": "person", "predicate": "appears",
    }}, callback=callback)
    assert "connector_setup_unavailable" in result
    with pytest.raises(ValueError, match="camera_observation_input_unavailable"):
        callback("ignored", None, connector_setup={"resource_kind": "camera", "observation": {
            "camera_id": "cam-1", "duration_seconds": 60, "subject_kind": "person", "predicate": "appears",
        }})
    assert stream.empty()
    assert not runtime._clarify_queues


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "submitted"])
@pytest.mark.parametrize("observation", [False, True, "recording"])
async def test_setup_uses_existing_pending_and_response_flow(status, observation):
    runtime = adapter()
    stream = queue.Queue()
    callback = runtime._make_clarify_cb(stream, "s", connector_input_capable=True, camera_observation_input_capable=observation)
    setup = {"resource_kind": "camera", "observation": {
        "camera_id": "cam-1", "duration_seconds": 60, "subject_kind": "person", "predicate": "appears",
    }} if observation else {"resource_kind": "camera", "live": {"camera_id": "cam-1"}}
    if observation == "recording":
        setup = {"resource_kind": "camera", "recording": {"camera_id": "cam-1"}}
    app = web.Application()
    app.router.add_get('/v1/sessions/{session_id}/pending', runtime._handle_pending)
    app.router.add_post('/v1/sessions/{session_id}/clarify/respond', runtime._handle_clarify_respond)
    headers = {"Authorization": "Bearer test-key"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        agent = _make_agent("clarify")
        agent.clarify_callback = callback
        def invoke_with_camera_scope():
            from gateway.session_context import set_turn_vars, clear_turn_vars
            import agent.zet_agent_response_mode as mode

            tokens = set_turn_vars(turn_id="confirmation-flow")
            try:
                if observation == "recording":
                    agent.platform = "zet_agent"
                    task = mode._skill_direct_task_context(agent, "为摄像头准备持续录像")
                    agent._zet_agent_skill_direct_task = task
                    agent._zet_agent_skill_direct_scope = mode._SkillDirectScope(
                        relative_path=mode._CAMERA_SKILL_PATH,
                        task_sha256=task.task_sha256,
                        turn_identity=task.turn_identity,
                        allowed_tools=mode._CAMERA_DIRECT_TOOLS,
                        camera_ids=frozenset({"cam-1"}),
                    )
                return invoke_tool(agent, "clarify", {"question": "ignored", "connector_setup": setup}, "setup-flow")
            finally:
                clear_turn_vars(tokens)

        future = pool.submit(invoke_with_camera_scope)
        event = stream.get(timeout=5)
        # Progress queue envelope is owned by the adapter, not model prose.
        payload = event[1] if isinstance(event, tuple) else event
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload["connector_setup"] == setup
        async with TestClient(TestServer(app)) as client:
            pending = await (await client.get('/v1/sessions/s/pending', headers=headers)).json()
            assert pending["clarify"]["connector_setup"] == setup
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
    assert 'connector_chat_turn_required' in result
    assert stream.empty()
    assert not runtime._clarify_queues
