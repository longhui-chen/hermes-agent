"""Phase 1: zet_agent `/p/<profile>` API routes for local-server mux mode."""

import asyncio
import json
import queue
import threading
from pathlib import Path

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import ZetAgentAdapter


def _make_adapter() -> ZetAgentAdapter:
    return ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))


def _add_prefixed_zet_agent_routes(app: web.Application, adapter: ZetAgentAdapter) -> None:
    app["api_server_adapter"] = adapter
    adapter._register_profile_api_routes(
        app.router,
        chat_handler=adapter._diagnostic_chat_completions,
    )
    app.router.add_post(
        "/p/{profile}/v1/model/switch",
        adapter._profile_handler(adapter._handle_model_switch),
    )
    app.router.add_post(
        "/p/{profile}/v1/skills/reload",
        adapter._profile_handler(adapter._handle_skills_reload),
    )
    app.router.add_post(
        "/p/{profile}/v1/connectors/reload",
        adapter._profile_handler(adapter._handle_connectors_reload),
    )
    app.router.add_post(
        "/p/{profile}/v1/profile/reload",
        adapter._profile_handler(adapter._handle_profile_reload),
    )
    app.router.add_post(
        "/p/{profile}/v1/runtime/reset",
        adapter._profile_handler(adapter._handle_runtime_reset),
    )
    app.router.add_post(
        "/p/{profile}/v1/profile/unload",
        adapter._profile_handler(adapter._handle_profile_unload),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/model/switch",
        adapter._profile_handler(adapter._handle_session_model_switch),
    )
    app.router.add_delete(
        "/p/{profile}/v1/sessions/{session_id}/model",
        adapter._profile_handler(adapter._handle_session_model_clear),
    )
    app.router.add_get(
        "/p/{profile}/v1/sessions/{session_id}/pending",
        adapter._profile_handler(adapter._handle_pending),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/approval/respond",
        adapter._profile_handler(adapter._handle_approval_respond),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/clarify/respond",
        adapter._profile_handler(adapter._handle_clarify_respond),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/interrupt",
        adapter._profile_handler(adapter._handle_session_interrupt),
    )


@pytest.mark.asyncio
async def test_clarify_id_is_stable_across_stream_pending_and_exact_response():
    """The same Hermes-generated id must drive live, reconnect and response.

    This exercises the real callback -> HTTP route path. A wrong id must not
    consume the pending callback; the right id unblocks exactly that callback.
    """
    adapter = _make_adapter()
    app = web.Application()
    app.router.add_get(
        "/v1/sessions/{session_id}/pending", adapter._handle_pending,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/clarify/respond", adapter._handle_clarify_respond,
    )
    stream_q: queue.Queue = queue.Queue()
    answered = []
    ask = adapter._make_clarify_cb(stream_q, "sid-clarify-id")
    thread = threading.Thread(
        target=lambda: answered.append(ask("Choose a runtime", ["A", "B"])),
        daemon=True,
    )
    thread.start()
    event_name, streamed = stream_q.get(timeout=1)
    assert event_name == "__tool_progress__"
    clarify_id = streamed.get("clarify_id")
    assert isinstance(clarify_id, str) and len(clarify_id) == 32

    async with TestClient(TestServer(app)) as cli:
        pending = await cli.get(
            "/v1/sessions/sid-clarify-id/pending",
            headers={"Authorization": "Bearer test-key"},
        )
        pending_data = await pending.json()
        assert pending.status == 200
        assert pending_data["clarify"]["clarify_id"] == clarify_id

        wrong = await cli.post(
            "/v1/sessions/sid-clarify-id/clarify/respond",
            json={"clarify_id": "not-the-live-card", "response": "wrong"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert wrong.status == 404
        assert thread.is_alive(), "a mismatched id must not consume FIFO state"

        resolved = await cli.post(
            "/v1/sessions/sid-clarify-id/clarify/respond",
            json={"clarify_id": clarify_id, "response": "B"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert resolved.status == 200
        assert await resolved.json() == {"resolved": 1}

    thread.join(timeout=1)
    assert not thread.is_alive()
    assert answered == ["B"]


@pytest.mark.asyncio
async def test_exact_clarify_response_keeps_pending_projection_on_fifo_head():
    """Resolving a later exact id cannot replace /pending's oldest card."""
    adapter = _make_adapter()
    app = web.Application()
    app.router.add_get(
        "/v1/sessions/{session_id}/pending", adapter._handle_pending,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/clarify/respond", adapter._handle_clarify_respond,
    )
    stream_q: queue.Queue = queue.Queue()
    ask = adapter._make_clarify_cb(stream_q, "sid-two-clarifies")
    answers = []
    def ask_and_record(question):
        answers.append((question, ask(question, None)))

    first = threading.Thread(target=lambda: ask_and_record("first"), daemon=True)
    second = threading.Thread(target=lambda: ask_and_record("second"), daemon=True)
    first.start()
    second.start()
    _, first_payload = stream_q.get(timeout=1)
    _, second_payload = stream_q.get(timeout=1)

    headers = {"Authorization": "Bearer test-key"}
    async with TestClient(TestServer(app)) as cli:
        pending = await cli.get("/v1/sessions/sid-two-clarifies/pending", headers=headers)
        assert (await pending.json())["clarify"]["clarify_id"] == first_payload["clarify_id"]

        later = await cli.post(
            "/v1/sessions/sid-two-clarifies/clarify/respond",
            json={"clarify_id": second_payload["clarify_id"], "response": "later answer"},
            headers=headers,
        )
        assert later.status == 200
        pending_after_later = await cli.get("/v1/sessions/sid-two-clarifies/pending", headers=headers)
        assert (await pending_after_later.json())["clarify"]["clarify_id"] == first_payload["clarify_id"]

        first_reply = await cli.post(
            "/v1/sessions/sid-two-clarifies/clarify/respond",
            json={"clarify_id": first_payload["clarify_id"], "response": "first answer"},
            headers=headers,
        )
        assert first_reply.status == 200

    first.join(timeout=1)
    second.join(timeout=1)
    assert not first.is_alive() and not second.is_alive()
    assert dict(answers) == {
        first_payload["question"]: "first answer",
        second_payload["question"]: "later answer",
    }


@pytest.mark.asyncio
async def test_prefixed_profiles_isolate_same_named_clarify_session(profile_homes):
    """A profile-local card cannot be read, answered, or interrupted by another profile."""
    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    same_session = "same-session-id"
    main_stream: queue.Queue = queue.Queue()
    coder_stream: queue.Queue = queue.Queue()
    main_answers = []
    coder_answers = []

    # This is the same attachment-time scope used by the real chat-completion
    # path. The callback later runs on a worker thread, so it proves we
    # captured profile identity instead of consulting a thread-local value.
    with adapter._profile_api_scope("main"):
        ask_main = adapter._make_clarify_cb(main_stream, same_session)
    with adapter._profile_api_scope("coder"):
        ask_coder = adapter._make_clarify_cb(coder_stream, same_session)

    main_thread = threading.Thread(target=lambda: main_answers.append(ask_main("main card", None)), daemon=True)
    coder_thread = threading.Thread(target=lambda: coder_answers.append(ask_coder("coder card", None)), daemon=True)
    main_thread.start()
    coder_thread.start()
    _, main_payload = main_stream.get(timeout=1)
    _, coder_payload = coder_stream.get(timeout=1)
    assert main_payload["clarify_id"] != coder_payload["clarify_id"]

    headers = {"Authorization": "Bearer test-key"}
    async with TestClient(TestServer(app)) as cli:
        main_pending = await cli.get(f"/p/main/v1/sessions/{same_session}/pending", headers=headers)
        coder_pending = await cli.get(f"/p/coder/v1/sessions/{same_session}/pending", headers=headers)
        assert (await main_pending.json())["clarify"]["question"] == "main card"
        assert (await coder_pending.json())["clarify"]["question"] == "coder card"

        # A coder response carrying main's id must not cross the profile
        # boundary or wake either callback.
        crossed = await cli.post(
            f"/p/coder/v1/sessions/{same_session}/clarify/respond",
            json={"clarify_id": main_payload["clarify_id"], "response": "wrong profile"},
            headers=headers,
        )
        assert crossed.status == 404
        assert main_thread.is_alive() and coder_thread.is_alive()

        # Interrupt uses the same scoped key as pending/respond. It may
        # unblock coder's clarify but must leave main's same-named session
        # untouched; timeout/push-failure cleanup reuses this exact discard
        # helper and key shape.
        coder_interrupt = await cli.post(
            f"/p/coder/v1/sessions/{same_session}/interrupt",
            headers=headers,
        )
        assert coder_interrupt.status == 200
        coder_thread.join(timeout=1)
        assert not coder_thread.is_alive()
        assert coder_answers == [""]
        main_still_pending = await cli.get(f"/p/main/v1/sessions/{same_session}/pending", headers=headers)
        assert (await main_still_pending.json())["clarify"]["clarify_id"] == main_payload["clarify_id"]

        main_reply = await cli.post(
            f"/p/main/v1/sessions/{same_session}/clarify/respond",
            json={"clarify_id": main_payload["clarify_id"], "response": "main answer"},
            headers=headers,
        )
        assert main_reply.status == 200

    main_thread.join(timeout=1)
    assert not main_thread.is_alive() and not coder_thread.is_alive()
    assert main_answers == ["main answer"]
    assert coder_answers == [""]


@pytest.fixture
def profile_homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    default_home = root
    coder_home = root / "profiles" / "coder"
    default_home.mkdir(parents=True)
    coder_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", default_home), ("coder", coder_home)],
    )
    return {"main": default_home, "coder": coder_home}


@pytest.mark.asyncio
async def test_prefixed_main_health_is_registered(profile_homes):
    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/p/main/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_prefixed_models_route_is_registered(profile_homes):
    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/p/coder/v1/models",
            headers={"Authorization": "Bearer test-key"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "hermes-agent"


@pytest.mark.asyncio
async def test_prefixed_chat_hits_handler_inside_profile_scope(profile_homes, monkeypatch):
    seen = []
    adapter = _make_adapter()

    async def fake_run_agent(**kwargs):
        from hermes_constants import get_hermes_home
        seen.append((get_hermes_home(), kwargs["user_message"]))
        return (
            {
                "final_response": "scoped",
                "session_id": kwargs["session_id"],
                "completed": True,
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(adapter, "_run_agent", fake_run_agent)
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/v1/chat/completions",
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "hello coder"}],
            },
            headers={"Authorization": "Bearer test-key"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "scoped"

    assert seen == [(profile_homes["coder"], "hello coder")]


@pytest.mark.asyncio
async def test_prefixed_chat_scope_reaches_agent_executor(profile_homes, monkeypatch):
    """The agent is created in an executor thread, so profile context must cross it."""
    seen = []
    adapter = _make_adapter()

    class FakeAgent:
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_id = "sid"

        def run_conversation(self, **_kwargs):
            return {"final_response": "ok", "completed": True}

    def fake_create_agent(**_kwargs):
        from hermes_constants import get_hermes_home

        seen.append(get_hermes_home())
        return FakeAgent()

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    with adapter._profile_api_scope("coder"):
        result, usage = await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="sid",
        )

    assert result["final_response"] == "ok"
    assert usage["total_tokens"] == 2
    assert seen == [profile_homes["coder"]]


@pytest.mark.asyncio
async def test_prefixed_model_switch_writes_scoped_profile_config(profile_homes):
    for home in profile_homes.values():
        (home / "config.yaml").write_text(
            yaml.safe_dump({"model": {"default": "old"}}),
            encoding="utf-8",
        )

    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        main_resp = await cli.post(
            "/p/main/v1/model/switch",
            json={"model": "main-model", "provider": "custom"},
            headers={"Authorization": "Bearer test-key"},
        )
        coder_resp = await cli.post(
            "/p/coder/v1/model/switch",
            json={"model": "coder-model", "provider": "custom"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert main_resp.status == 200
    assert coder_resp.status == 200
    main_cfg = yaml.safe_load((profile_homes["main"] / "config.yaml").read_text())
    coder_cfg = yaml.safe_load((profile_homes["coder"] / "config.yaml").read_text())
    assert main_cfg["model"]["default"] == "main-model"
    assert coder_cfg["model"]["default"] == "coder-model"


@pytest.mark.asyncio
async def test_prefixed_jobs_use_scoped_profile_home(profile_homes, monkeypatch):
    import gateway.platforms.api_server as api_server

    seen = []

    def fake_list(include_disabled=False):
        from hermes_constants import get_hermes_home
        seen.append((get_hermes_home(), include_disabled))
        return []

    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    monkeypatch.setattr(api_server, "_cron_list", fake_list)

    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp_main = await cli.get(
            "/p/main/api/jobs?include_disabled=true",
            headers={"Authorization": "Bearer test-key"},
        )
        resp_coder = await cli.get(
            "/p/coder/api/jobs",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp_main.status == 200
    assert resp_coder.status == 200
    assert seen == [
        (profile_homes["main"], True),
        (profile_homes["coder"], False),
    ]


@pytest.mark.asyncio
async def test_prefixed_cron_fire_uses_scoped_profile_home(profile_homes, monkeypatch):
    import gateway.platforms.api_server as api_server

    seen = []

    class SpyProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            from hermes_constants import get_hermes_home
            seen.append((get_hermes_home(), job_id))
            return True

    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: SpyProvider())
    monkeypatch.setattr(
        "plugins.cron.chronos.verify.get_fire_verifier",
        lambda: (lambda **_kwargs: {"purpose": "cron_fire"}),
    )

    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/api/cron/fire",
            json={"job_id": "nightly"},
            headers={"Authorization": "Bearer fire-token"},
        )

    assert resp.status == 202
    for _ in range(50):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == [(profile_homes["coder"], "nightly")]


@pytest.mark.asyncio
async def test_prefixed_profile_unload_calls_targeted_runner(profile_homes):
    calls = []

    class FakeRunner:
        async def unload_profile_runtime(self, profile):
            calls.append(profile)
            return {"evicted_sessions": 2, "disconnected_adapters": 1}

    adapter = _make_adapter()
    adapter.gateway_runner = FakeRunner()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": "Bearer test-key"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert calls == ["coder"]
    assert data["evicted_sessions"] == 2
    assert data["disconnected_adapters"] == 1


@pytest.mark.asyncio
async def test_prefixed_profile_unload_blocks_active_sessions(profile_homes):
    class FakeRunner:
        async def unload_profile_runtime(self, profile):
            return {
                "blocked": True,
                "active_sessions": 1,
                "evicted_sessions": 0,
                "disconnected_adapters": 0,
            }

    adapter = _make_adapter()
    adapter.gateway_runner = FakeRunner()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": "Bearer test-key"},
        )
        data = await resp.json()

    assert resp.status == 409
    assert data["unloaded"] is False
    assert data["active_sessions"] == 1


@pytest.mark.asyncio
async def test_gateway_profile_unload_blocks_pending_sentinel():
    from gateway.run import _AGENT_PENDING_SENTINEL, GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {"agent:coder:api_server:dm:sid": _AGENT_PENDING_SENTINEL}

    result = await runner.unload_profile_runtime("coder")

    assert result["blocked"] is True
    assert result["active_sessions"] == 1


@pytest.mark.asyncio
async def test_prefixed_jobs_read_scoped_cron_store(profile_homes, monkeypatch):
    import cron.jobs as cron_jobs
    import gateway.platforms.api_server as api_server

    def write_jobs(home: Path, job_id: str, name: str) -> None:
        cron_dir = home / "cron"
        cron_dir.mkdir(parents=True, exist_ok=True)
        (cron_dir / "jobs.json").write_text(
            json.dumps(
                {
                    "jobs": [
                        {
                            "id": job_id,
                            "name": name,
                            "prompt": "ping",
                            "enabled": True,
                            "schedule": {
                                "kind": "interval",
                                "minutes": 10,
                                "display": "every 10m",
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    write_jobs(profile_homes["main"], "main-job", "main job")
    write_jobs(profile_homes["coder"], "coder-job", "coder job")

    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    monkeypatch.setattr(api_server, "_cron_list", cron_jobs.list_jobs)

    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        main_resp = await cli.get(
            "/p/main/api/jobs?include_disabled=true",
            headers={"Authorization": "Bearer test-key"},
        )
        coder_resp = await cli.get(
            "/p/coder/api/jobs?include_disabled=true",
            headers={"Authorization": "Bearer test-key"},
        )
        main_data = await main_resp.json()
        coder_data = await coder_resp.json()

    assert main_resp.status == 200
    assert coder_resp.status == 200
    assert [job["id"] for job in main_data["jobs"]] == ["main-job"]
    assert [job["id"] for job in coder_data["jobs"]] == ["coder-job"]


@pytest.mark.asyncio
async def test_prefixed_control_routes_registered(profile_homes):
    adapter = _make_adapter()
    adapter.gateway_runner = type(
        "Runner",
        (),
        {
            "_session_model_overrides": {},
            "_evict_cached_agent": lambda self, sid: None,
            "invalidate_all_cached_agents": lambda self: 0,
        },
    )()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    registered = {resource.canonical for resource in app.router.resources()}
    expected = {
        "/p/{profile}/v1/skills/reload",
        "/p/{profile}/v1/connectors/reload",
        "/p/{profile}/v1/profile/reload",
        "/p/{profile}/v1/runtime/reset",
        "/p/{profile}/v1/profile/unload",
        "/p/{profile}/v1/model/switch",
        "/p/{profile}/v1/sessions/{session_id}/model/switch",
        "/p/{profile}/v1/sessions/{session_id}/model",
        "/p/{profile}/v1/sessions/{session_id}/pending",
        "/p/{profile}/v1/sessions/{session_id}/approval/respond",
        "/p/{profile}/v1/sessions/{session_id}/clarify/respond",
        "/p/{profile}/v1/sessions/{session_id}/interrupt",
    }
    assert expected <= registered


@pytest.mark.asyncio
async def test_prefixed_reset_and_unload_return_ok(profile_homes):
    adapter = _make_adapter()
    adapter.gateway_runner = type(
        "Runner",
        (),
        {"invalidate_all_cached_agents": lambda self: 0},
    )()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        reset_resp = await cli.post(
            "/p/coder/v1/runtime/reset",
            headers={"Authorization": "Bearer test-key"},
        )
        unload_resp = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": "Bearer test-key"},
        )
        unload_data = await unload_resp.json()

    assert reset_resp.status == 200
    assert unload_resp.status == 200
    assert unload_data["unloaded"] is True
