import hashlib
import asyncio
import json
import threading
import textwrap
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import zettlab_connector_route_capability
from tools import terminal_tool as terminal_tool_module


def _api_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _write_connector_runtime(tmp_path):
    script = (
        tmp_path
        / "presets"
        / "skills"
        / "linear"
        / "scripts"
        / "connector_runtime.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        textwrap.dedent(
            """
            import hashlib
            import json
            import os

            print(json.dumps({
                "runner_session_key": hashlib.sha256(os.environ.get("HERMES_SESSION_KEY", "").encode()).hexdigest(),
                "stable_session": hashlib.sha256(os.environ.get("ZETTLAB_CONNECTOR_SESSION_ID", "").encode()).hexdigest(),
                "connector_action_runtime": os.environ.get(
                    "ZETTLAB_CONNECTOR_ACTION_RUNTIME", ""
                ),
            }))
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return script


def _connector_command() -> str:
    return (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/'
        'connector_runtime.py" inspect'
    )


@pytest.mark.asyncio
async def test_http_capabilities_remain_isolated_through_direct_runner(
    monkeypatch,
    tmp_path,
):
    """HTTP metadata stays task-local until the dedicated runner boundary."""
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_CONNECTOR_RUNTIME_ROOT_ANCHOR",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_sensitive_runtime_boundary",
        lambda: True,
    )

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "flow-test-key"})
    )
    overlap = threading.Barrier(2)

    class _Agent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, *, user_message, conversation_history, task_id):
            overlap.wait(timeout=5)
            result = json.loads(
                terminal_tool_module.terminal_tool(
                    _connector_command(),
                    task_id=f"route-capability-{user_message}",
                )
            )
            assert result["connector_runtime_direct"] is True
            return {"final_response": result["output"]}

    async with TestClient(TestServer(_api_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=lambda **_: _Agent()):
            async def _post(label: str, capability: str, session_key: str):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer flow-test-key",
                        "X-Hermes-Session-Key": session_key,
                    },
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": label}],
                        "metadata": {
                            "connector_route_capability": capability,
                        },
                    },
                )
                assert response.status == 200
                body = await response.json()
                output = body["choices"][0]["message"]["content"]
                return json.loads(output)

            capability_a = "A" * 43
            capability_b = "B" * 43
            result_a, result_b = await asyncio.gather(
                _post("transcript-A", capability_a, "route-session-A"),
                _post("transcript-B", capability_b, "route-session-B"),
            )

    expected_runtime = "skills/linear/scripts/connector_runtime.py"
    assert result_a == {
        "runner_session_key": hashlib.sha256(capability_a.encode()).hexdigest(),
        "stable_session": hashlib.sha256(b"route-session-A").hexdigest(),
        "connector_action_runtime": expected_runtime,
    }
    assert result_b == {
        "runner_session_key": hashlib.sha256(capability_b.encode()).hexdigest(),
        "stable_session": hashlib.sha256(b"route-session-B").hexdigest(),
        "connector_action_runtime": expected_runtime,
    }
    combined = json.dumps([result_a, result_b])
    assert "transcript-A" not in combined
    assert "transcript-B" not in combined
    assert "route-session-A" not in combined
    assert "route-session-B" not in combined
    assert zettlab_connector_route_capability() == ""


@pytest.mark.asyncio
async def test_capability_is_cleared_after_agent_exception(monkeypatch):
    """A failed turn must not retain its opaque capability."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    observed: list[str] = []
    observed_lock = threading.Lock()

    from gateway import session_context

    original_setter = session_context.set_zettlab_connector_route_capability

    def _recording_setter(capability: str) -> None:
        original_setter(capability)
        with observed_lock:
            observed.append(zettlab_connector_route_capability())

    class _FailingAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, **_):
            assert zettlab_connector_route_capability() == "E" * 43
            raise RuntimeError("expected test failure")

    monkeypatch.setattr(
        session_context,
        "set_zettlab_connector_route_capability",
        _recording_setter,
    )
    with patch.object(adapter, "_create_agent", return_value=_FailingAgent()):
        with pytest.raises(RuntimeError, match="expected test failure"):
            await adapter._run_agent(
                user_message="exception",
                conversation_history=[],
                connector_route_capability="E" * 43,
            )

    assert observed == ["E" * 43, ""]
    assert zettlab_connector_route_capability() == ""


@pytest.mark.asyncio
async def test_capability_is_cleared_when_awaiting_turn_is_cancelled(monkeypatch):
    """Cancellation cannot leave the worker's task-local capability behind."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    entered = threading.Event()
    release = threading.Event()
    cleared = threading.Event()

    from gateway import session_context

    original_setter = session_context.set_zettlab_connector_route_capability

    def _recording_setter(capability: str) -> None:
        original_setter(capability)
        if capability == "":
            cleared.set()

    class _BlockingAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, **_):
            assert zettlab_connector_route_capability() == "C" * 43
            entered.set()
            assert release.wait(timeout=5)
            return {"final_response": "released"}

    monkeypatch.setattr(
        session_context,
        "set_zettlab_connector_route_capability",
        _recording_setter,
    )
    with patch.object(adapter, "_create_agent", return_value=_BlockingAgent()):
        task = asyncio.create_task(
            adapter._run_agent(
                user_message="cancel",
                conversation_history=[],
                connector_route_capability="C" * 43,
            )
        )
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(100):
            if cleared.is_set():
                break
            await asyncio.sleep(0.01)

    assert cleared.is_set()
    assert zettlab_connector_route_capability() == ""
