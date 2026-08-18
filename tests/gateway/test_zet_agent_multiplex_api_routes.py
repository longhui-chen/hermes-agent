"""Phase 1: zet_agent `/p/<profile>` API routes for local-server mux mode."""

import asyncio
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import ZetAgentAdapter, _ClarifyEntry
from gateway.session_context import clear_turn_vars, set_turn_vars
from tools import approval

TEST_API_KEY = "test-key-0123456789abcdef"


def _make_adapter() -> ZetAgentAdapter:
    return ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": TEST_API_KEY}))


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
    app.router.add_get(
        "/p/{profile}/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}",
        adapter._profile_handler(adapter._handle_interaction_delivery),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}/recovery-fence",
        adapter._profile_handler(adapter._handle_recovery_fence),
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
        "/p/{profile}/v1/sessions/{session_id}/attachment/action",
        adapter._profile_handler(adapter._handle_attachment_action),
    )
    app.router.add_post(
        "/p/{profile}/v1/sessions/{session_id}/interrupt",
        adapter._profile_handler(adapter._handle_session_interrupt),
    )


@pytest.mark.asyncio
async def test_connectors_reload_returns_stable_error_without_internal_details(
    monkeypatch,
):
    import tools.mcp_tool as mcp_tool

    adapter = _make_adapter()
    app = web.Application()
    app.router.add_post("/v1/connectors/reload", adapter._handle_connectors_reload)
    monkeypatch.setattr(
        mcp_tool,
        "reload_single_mcp_server",
        lambda _name: (_ for _ in ()).throw(
            RuntimeError("/private/profile/a/wecom-cli-config")
        ),
    )

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/connectors/reload",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        body = await response.json()

    assert response.status == 500
    assert body["error"]["code"] == "connector_reload_unavailable"
    assert "reference" in body["error"]["message"]
    assert "/private/profile" not in body["error"]["message"]


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        pending_data = await pending.json()
        assert pending.status == 200
        assert pending_data["clarify"]["clarify_id"] == clarify_id

        wrong = await cli.post(
            "/v1/sessions/sid-clarify-id/clarify/respond",
            json={"clarify_id": "not-the-live-card", "response": "wrong"},
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        assert wrong.status == 404
        assert thread.is_alive(), "a mismatched id must not consume FIFO state"

        resolved = await cli.post(
            "/v1/sessions/sid-clarify-id/clarify/respond",
            json={"clarify_id": clarify_id, "response": "B"},
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
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

    headers = {"Authorization": "Bearer test-key-0123456789abcdef"}
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

    headers = {"Authorization": "Bearer test-key-0123456789abcdef"}
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
    for home in (default_home, coder_home):
        (home / ".env").write_text(
            f"API_SERVER_KEY={TEST_API_KEY}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", default_home), ("coder", coder_home)],
    )
    return {"main": default_home, "coder": coder_home}


@pytest.mark.asyncio
async def test_prefixed_main_auth_uses_shared_zet_agent_key_without_profile_api_key(
    tmp_path, monkeypatch
):
    """ZetAgent profile mirrors share the device-internal listener key."""
    root = tmp_path / ".hermes"
    coder_home = root / "profiles" / "coder"
    root.mkdir(parents=True)
    coder_home.mkdir(parents=True)
    (root / ".env").write_text("", encoding="utf-8")
    (coder_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("ZET_AGENT_KEY", TEST_API_KEY)
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", root), ("coder", coder_home)],
    )

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        accepted = await cli.get(
            "/p/main/v1/models",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        rejected = await cli.get(
            "/p/main/v1/models",
            headers={"Authorization": "Bearer wrong-device-key"},
        )

    assert accepted.status == 200
    assert rejected.status == 401


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "coder"


@pytest.mark.asyncio
async def test_prefixed_interaction_delivery_route_is_registered(profile_homes):
    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/p/coder/v1/sessions/session-1/interaction-deliveries/missing",
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        data = await resp.json()

    assert resp.status == 404
    assert data["error"]["code"] == "interaction_delivery_not_found"


@pytest.mark.asyncio
async def test_same_session_id_isolated_across_profile_interaction_queues(profile_homes):
    adapter = _make_adapter()
    setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(
            on_interaction_pending=lambda _sid, **_kwargs: None,
            on_interaction_resolved=lambda _sid: None,
        ),
    )
    seeded = []
    for profile, command in (("main", "main-command"), ("coder", "coder-command")):
        with adapter._profile_api_scope(profile):
            queue_key = adapter._interaction_queue_key("same-session")
            approval_data = {
                "interaction_id": f"approval-{profile}",
                "command": command,
                "description": "test",
            }
            approval_entry = approval.enqueue_gateway_approval(
                queue_key, approval_data
            )
            tokens = set_turn_vars(turn_id=f"turn-{profile}")
            try:
                adapter._make_approval_cb(
                    queue.Queue(), "same-session", queue_key
                )(approval_data)
            finally:
                clear_turn_vars(tokens)

            with approval.reserve_gateway_interaction_generation() as generation:
                clarify_payload = {
                    "type": "hermes.clarify",
                    "interaction_id": f"clarify-{profile}",
                    "interaction_generation": generation,
                    "turn_id": f"turn-{profile}",
                    "question": f"question-{profile}",
                    "choices_offered": [],
                }
                clarify_entry = _ClarifyEntry(
                    f"clarify-{profile}",
                    f"turn-{profile}",
                    clarify_payload,
                    generation,
                )
                with adapter._clarify_state_lock:
                    adapter._clarify_queues[queue_key] = [clarify_entry]
            adapter._pending_clarify[queue_key] = [clarify_payload]
            seeded.append((queue_key, approval_entry, clarify_entry))

    assert seeded[0][0] != seeded[1][0]
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    try:
        async with TestClient(TestServer(app)) as cli:
            main_pending = await cli.get(
                "/p/main/v1/sessions/same-session/pending",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
            )
            coder_pending = await cli.get(
                "/p/coder/v1/sessions/same-session/pending",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
            )
            assert (await main_pending.json())["approval"]["command"] == "main-command"
            assert (await coder_pending.json())["approval"]["command"] == "coder-command"

            main_approval = await cli.post(
                "/p/main/v1/sessions/same-session/approval/respond",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                json={"choice": "deny"},
            )
            assert main_approval.status == 200
            assert seeded[0][1].event.is_set()
            assert not seeded[1][1].event.is_set()

            main_clarify = await cli.post(
                "/p/main/v1/sessions/same-session/clarify/respond",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                json={"response": "main-answer"},
            )
            assert main_clarify.status == 200
            assert seeded[0][2].event.is_set()
            assert not seeded[1][2].event.is_set()
    finally:
        approval._gateway_queues.clear()
        approval._gateway_prepared.clear()


@pytest.mark.asyncio
async def test_same_delivery_ids_have_profile_scoped_durable_receipts(profile_homes):
    class LiveTask:
        def done(self):
            return False

    adapter = _make_adapter()
    setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(
            on_interaction_pending=lambda _sid, **_kwargs: None,
            on_interaction_resolved=lambda _sid: None,
        ),
    )
    entries = {}
    for profile in ("main", "coder"):
        with adapter._profile_api_scope(profile):
            queue_key = adapter._interaction_queue_key("same-session")
            data = {
                "interaction_id": "same-interaction",
                "command": f"command-{profile}",
                "description": "test",
            }
            entry = approval.enqueue_gateway_approval(queue_key, data)
            tokens = set_turn_vars(turn_id=f"turn-{profile}")
            try:
                adapter._make_approval_cb(
                    queue.Queue(), "same-session", queue_key
                )(data)
            finally:
                clear_turn_vars(tokens)
            adapter._active_session_tasks[queue_key] = LiveTask()
            adapter._active_session_turn_ids[queue_key] = f"turn-{profile}"
            entries[profile] = entry

    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    try:
        async with TestClient(TestServer(app)) as cli:
            for profile, choice in (("main", "once"), ("coder", "deny")):
                body = {
                    "choice": choice,
                    "delivery_id": "same-delivery",
                    "interaction_id": "same-interaction",
                }
                prepared = await cli.post(
                    f"/p/{profile}/v1/sessions/same-session/approval/respond",
                    headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                    json={**body, "phase": "prepare"},
                )
                assert prepared.status == 200
                assert entries[profile].event.is_set() is False
                if profile == "main":
                    assert entries["coder"].event.is_set() is False

                finalized = await cli.post(
                    f"/p/{profile}/v1/sessions/same-session/approval/respond",
                    headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                    json={**body, "phase": "finalize"},
                )
                finalized_body = await finalized.json()
                assert finalized.status == 200
                assert finalized_body["turn_id"] == f"turn-{profile}"
                assert finalized_body["fence_id"]
                assert entries[profile].event.is_set() is False
                claim = await cli.post(
                    f"/p/{profile}/v1/sessions/same-session/interaction-deliveries/same-delivery/recovery-fence",
                    headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                    json={
                        "action": "claim",
                        "expected_state_revision": finalized_body["state_revision"],
                    },
                )
                assert claim.status == 200
                ack = await cli.post(
                    f"/p/{profile}/v1/sessions/same-session/interaction-deliveries/same-delivery/recovery-fence",
                    headers={"Authorization": "Bearer test-key-0123456789abcdef"},
                    json={
                        "action": "ack",
                        "fence_id": finalized_body["fence_id"],
                        "expected_state_revision": finalized_body["state_revision"],
                    },
                )
                assert ack.status == 200
                assert entries[profile].event.is_set() is True
                if profile == "main":
                    assert entries["coder"].event.is_set() is False

            main_receipt = await cli.get(
                "/p/main/v1/sessions/same-session/interaction-deliveries/same-delivery",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
            )
            coder_receipt = await cli.get(
                "/p/coder/v1/sessions/same-session/interaction-deliveries/same-delivery",
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
            )
            main_body = await main_receipt.json()
            coder_body = await coder_receipt.json()
            assert main_body["turn_id"] == "turn-main"
            assert coder_body["turn_id"] == "turn-coder"
            assert main_body["payload_digest"] != coder_body["payload_digest"]
    finally:
        approval._gateway_queues.clear()
        approval._gateway_prepared.clear()


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["choices"][0]["message"]["content"] == "scoped"

    assert seen == [(profile_homes["coder"], "hello coder")]


@pytest.mark.asyncio
async def test_prefixed_chat_routes_connector_capability_through_zet_agent_override(
    profile_homes, monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_constants import get_hermes_home

    capability = "c" * 43
    seen = []
    adapter = _make_adapter()

    async def fake_base_run(_self, **kwargs):
        seen.append((
            get_hermes_home(),
            kwargs["connector_route_capability"],
        ))
        return (
            {
                "final_response": "scoped",
                "session_id": kwargs["session_id"],
                "completed": True,
            },
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    class NoopGoals:
        def schedule_after_turn(self, *_args, **_kwargs):
            return None

    async def noop_title(**_kwargs):
        return None

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_base_run)
    monkeypatch.setattr(adapter, "_goals", lambda: NoopGoals())
    monkeypatch.setattr(adapter, "_emit_native_session_title", noop_title)
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/v1/chat/completions",
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "hello coder"}],
                "metadata": {"connector_route_capability": capability},
            },
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        assert resp.status == 200

    assert seen == [(profile_homes["coder"], capability)]


@pytest.mark.asyncio
async def test_prefixed_chat_scope_reaches_agent_executor(profile_homes, monkeypatch):
    """The agent is created in an executor thread, so profile context must cross it."""
    (profile_homes["coder"] / ".env").write_text(
        f"API_SERVER_KEY={TEST_API_KEY}\nZET_AGENT_ID=coder\n",
        encoding="utf-8",
    )
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
        from agent.secret_scope import current_secret_scope
        from gateway.platforms.api_server import _api_request_profile
        from gateway.platforms.zet_agent import (
            _deep_memory_principal,
            _deep_memory_subject,
        )
        from hermes_constants import get_hermes_home

        scope = current_secret_scope()
        seen.append((
            get_hermes_home(),
            None if scope is None else scope.get("ZET_AGENT_ID"),
            _api_request_profile.get(),
            _deep_memory_principal.get(),
            _deep_memory_subject.get(),
        ))
        return FakeAgent()

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/p/coder/v1/chat/completions",
            json={
                "model": "hermes-agent",
                "messages": [{"role": "user", "content": "hello coder"}],
            },
            headers={
                "Authorization": "Bearer test-key-0123456789abcdef",
                "X-Zettlab-Auth-Principal-Id": "iam:issuer:user:user-1",
                "X-Zettlab-User-Id": "user-1",
            },
        )

    assert response.status == 200
    assert seen == [(
        profile_homes["coder"], "coder", "coder",
        "iam:issuer:user:user-1", "user-1",
    )]


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        coder_resp = await cli.post(
            "/p/coder/v1/model/switch",
            json={"model": "coder-model", "provider": "custom"},
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        resp_coder = await cli.get(
            "/p/coder/api/jobs",
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
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
        def fire_due(self, job_id, *, adapters=None, loop=None, fire_at=None):
            from hermes_constants import get_hermes_home
            seen.append((get_hermes_home(), job_id))
            return True

    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: SpyProvider())
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **_kwargs: {"purpose": "cron_fire"}),
    )

    adapter = _make_adapter()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/api/cron/fire",
            json={"job_id": "nightly", "fire_at": "2026-07-21T09:00:00Z"},
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
        async def unload_profile_runtime(self, profile, **kwargs):
            calls.append((profile, kwargs["profile_home"]))
            return {"evicted_sessions": 2, "disconnected_adapters": 1}

    adapter = _make_adapter()
    adapter.gateway_runner = FakeRunner()
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        data = await resp.json()

    assert resp.status == 200
    assert calls == [("coder", profile_homes["coder"])]
    assert data["evicted_sessions"] == 2
    assert data["disconnected_adapters"] == 1


@pytest.mark.asyncio
async def test_prefixed_profile_unload_blocks_active_sessions(profile_homes):
    class FakeRunner:
        async def unload_profile_runtime(self, profile, **_kwargs):
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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        data = await resp.json()

    assert resp.status == 409
    assert data["unloaded"] is False
    assert data["active_sessions"] == 1


@pytest.mark.asyncio
async def test_prefixed_responses_holds_profile_lease_until_agent_finishes(
    profile_homes,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "responses-agent"

        def run_conversation(self, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return {"final_response": "done", "completed": True}

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    app.router.add_post(
        "/p/{profile}/v1/responses",
        adapter._profile_handler(adapter._handle_responses),
    )

    async with TestClient(TestServer(app)) as cli:
        response_task = asyncio.create_task(
            cli.post(
                "/p/coder/v1/responses",
                json={"input": "hello"},
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
        )
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        blocked = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        blocked_data = await blocked.json()
        assert blocked.status == 409
        assert blocked_data["active_api_runs"] >= 1

        release.set()
        response = await response_task
        assert response.status == 200

    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_cancelled_responses_worker_keeps_profile_lease_until_thread_exits(
    profile_homes,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "cancelled-responses-agent"

        def run_conversation(self, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return {"final_response": "done", "completed": True}

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)

    with adapter._profile_api_scope("coder"):
        agent_task = asyncio.create_task(
            adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="cancelled-responses-session",
            )
        )
    assert await asyncio.to_thread(started.wait, 2)

    agent_task.cancel()
    await asyncio.sleep(0)
    assert not agent_task.done()
    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 1

    async with TestClient(TestServer(app)) as cli:
        blocked = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert blocked.status == 409
        assert (await blocked.json())["active_api_runs"] == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await agent_task
    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_prefixed_runs_holds_profile_lease_until_background_task_finishes(
    profile_homes,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "runs-agent"

        def run_conversation(self, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return {"final_response": "done", "completed": True}

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    app.router.add_post(
        "/p/{profile}/v1/runs",
        adapter._profile_handler(adapter._handle_runs),
    )

    async with TestClient(TestServer(app)) as cli:
        started_response = await cli.post(
            "/p/coder/v1/runs",
            json={"input": "hello", "session_id": "public-session"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert started_response.status == 202
        run_id = (await started_response.json())["run_id"]

        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()

        blocked = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        blocked_data = await blocked.json()
        assert blocked.status == 409
        assert blocked_data["active_api_runs"] >= 1

        release.set()
        for _ in range(100):
            if run_id not in adapter._active_run_tasks:
                break
            await asyncio.sleep(0.01)
        assert run_id not in adapter._active_run_tasks

    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_runs_worker_keeps_account_metadata_separate_from_principal(
    profile_homes,
    monkeypatch,
):
    del profile_homes
    started = threading.Event()
    seen = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "runs-account"

        def run_conversation(self, **_kwargs):
            from gateway.session_context import get_session_env

            seen["account"] = get_session_env("HERMES_SESSION_USER_ID", "")
            started.set()
            return {"final_response": "done", "completed": True}

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    app = web.Application()
    app.router.add_post("/v1/runs", adapter._handle_runs)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/v1/runs",
            json={"input": "hello", "session_id": "public-session"},
            headers={
                "Authorization": f"Bearer {TEST_API_KEY}",
                "X-Zettlab-Account-Id": "account-1",
                "X-Zettlab-Auth-Principal-Id": "iam:alice",
            },
        )
        assert response.status == 202
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)

    assert started.is_set()
    assert seen["account"] == "account-1"


@pytest.mark.asyncio
async def test_cancelled_runs_worker_keeps_profile_lease_until_thread_exits(
    profile_homes,
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "cancelled-runs-agent"

        def run_conversation(self, **_kwargs):
            started.set()
            assert release.wait(timeout=5)
            return {"final_response": "done", "completed": True}

    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    unregistered_approvals = []
    original_unregister = approval.unregister_gateway_notify

    def _spy_unregister(session_key):
        unregistered_approvals.append(session_key)
        original_unregister(session_key)

    monkeypatch.setattr(
        approval,
        "unregister_gateway_notify",
        _spy_unregister,
    )
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    app.router.add_post(
        "/p/{profile}/v1/runs",
        adapter._profile_handler(adapter._handle_runs),
    )

    async with TestClient(TestServer(app)) as cli:
        started_response = await cli.post(
            "/p/coder/v1/runs",
            json={"input": "hello", "session_id": "cancelled-public-session"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert started_response.status == 202
        run_id = (await started_response.json())["run_id"]
        assert await asyncio.to_thread(started.wait, 2)

        run_task = adapter._active_run_tasks[run_id]
        run_task.cancel()
        await asyncio.sleep(0)
        assert not run_task.done()
        assert run_id in unregistered_approvals
        assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 1

        blocked = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert blocked.status == 409
        assert (await blocked.json())["active_api_runs"] == 1

        release.set()
        for _ in range(100):
            if run_task.done():
                break
            await asyncio.sleep(0.01)
        assert run_task.done()

    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_cancelled_runs_releases_approval_registered_after_cancel(
    profile_homes,
    monkeypatch,
):
    register_entered = threading.Event()
    allow_register = threading.Event()
    worker_entered = threading.Event()
    release_worker = threading.Event()
    unregister_calls = []
    original_register = approval.register_gateway_notify
    original_unregister = approval.unregister_gateway_notify

    def _delayed_register(session_key, callback):
        register_entered.set()
        assert allow_register.wait(timeout=5)
        original_register(session_key, callback)

    def _spy_unregister(session_key):
        unregister_calls.append(session_key)
        original_unregister(session_key)

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "late-register-agent"

        def run_conversation(self, **_kwargs):
            worker_entered.set()
            assert release_worker.wait(timeout=5)
            return {"final_response": "cancelled", "completed": True}

    monkeypatch.setattr(approval, "register_gateway_notify", _delayed_register)
    monkeypatch.setattr(approval, "unregister_gateway_notify", _spy_unregister)
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: FakeAgent())
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    app.router.add_post(
        "/p/{profile}/v1/runs",
        adapter._profile_handler(adapter._handle_runs),
    )

    async with TestClient(TestServer(app)) as cli:
        started_response = await cli.post(
            "/p/coder/v1/runs",
            json={"input": "hello", "session_id": "late-register-session"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        run_id = (await started_response.json())["run_id"]
        assert await asyncio.to_thread(register_entered.wait, 2)

        run_task = adapter._active_run_tasks[run_id]
        run_task.cancel()
        await asyncio.sleep(0)
        assert unregister_calls == [run_id]
        assert not run_task.done()

        allow_register.set()
        assert await asyncio.to_thread(worker_entered.wait, 2)
        for _ in range(100):
            if len(unregister_calls) >= 2:
                break
            await asyncio.sleep(0.01)
        assert unregister_calls[:2] == [run_id, run_id]
        assert approval.list_gateway_approvals(run_id) == []
        assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 1

        release_worker.set()
        for _ in range(100):
            if run_task.done():
                break
            await asyncio.sleep(0.01)
        assert run_task.done()

    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_profile_unload_barrier_rejects_new_agent_request_during_teardown(
    profile_homes,
    monkeypatch,
):
    unload_entered = asyncio.Event()
    release_unload = asyncio.Event()

    class FakeRunner:
        async def unload_profile_runtime(self, profile, **_kwargs):
            assert profile == "coder"
            unload_entered.set()
            await release_unload.wait()
            return {"evicted_sessions": 0, "disconnected_adapters": 0}

    adapter = _make_adapter()
    adapter.gateway_runner = FakeRunner()
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: pytest.fail("blocked request constructed an agent"),
    )
    app = web.Application()
    _add_prefixed_zet_agent_routes(app, adapter)
    app.router.add_post(
        "/p/{profile}/v1/responses",
        adapter._profile_handler(adapter._handle_responses),
    )

    async with TestClient(TestServer(app)) as cli:
        unload_task = asyncio.create_task(
            cli.post(
                "/p/coder/v1/profile/unload",
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
        )
        await asyncio.wait_for(unload_entered.wait(), timeout=2)
        assert adapter._runtime_import_profile_is_blocked(profile_homes["coder"])

        rejected = await cli.post(
            "/p/coder/v1/responses",
            json={"input": "must not enter"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        rejected_data = await rejected.json()
        assert rejected.status == 409
        assert rejected_data["error"]["code"] == "profile_unloading"

        release_unload.set()
        unloaded = await unload_task
        assert unloaded.status == 200


@pytest.mark.asyncio
async def test_cancelled_idempotent_waiter_cannot_transfer_released_profile_lease(
    profile_homes,
    monkeypatch,
):
    import gateway.platforms.api_server as api_server

    child_waiting = asyncio.Event()
    allow_claim = asyncio.Event()
    child_tasks = []

    async def _shielded_delayed_compute(_key, _fingerprint, compute_coro):
        async def _delayed_compute():
            child_waiting.set()
            await allow_claim.wait()
            return await compute_coro()

        child_task = asyncio.create_task(_delayed_compute())
        child_tasks.append(child_task)
        return await asyncio.shield(child_task)

    class _Request(dict):
        def __init__(self):
            super().__init__()
            self.headers = {
                "Authorization": f"Bearer {TEST_API_KEY}",
                "Idempotency-Key": "cancel-before-claim",
            }
            self.match_info = {"profile": "coder"}
            self.method = "POST"
            self.path_qs = "/p/coder/v1/responses"
            self.remote = "127.0.0.1"
            self.transport = None

        async def json(self):
            return {"input": "must not outlive released lease"}

    adapter = _make_adapter()
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: pytest.fail("stale idempotency task constructed an agent"),
    )
    monkeypatch.setattr(
        api_server._idem_cache,
        "get_or_set",
        _shielded_delayed_compute,
    )
    handler = adapter._profile_handler(adapter._handle_responses)
    request_task = asyncio.create_task(handler(_Request()))
    await asyncio.wait_for(child_waiting.wait(), timeout=2)
    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 1

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0

    active_imports, active_api_runs, barrier_owner = (
        adapter._block_runtime_import_profile(profile_homes["coder"])
    )
    assert (active_imports, active_api_runs) == (0, 0)
    assert barrier_owner is not None

    allow_claim.set()
    try:
        with pytest.raises(RuntimeError, match="profile is unloading"):
            await child_tasks[0]
    finally:
        adapter._unblock_runtime_import_profile(
            profile_homes["coder"], barrier_owner
        )
    assert adapter._active_profile_chat_runs(profile_homes["coder"]) == 0


@pytest.mark.asyncio
async def test_gateway_profile_unload_blocks_pending_sentinel():
    from gateway.run import _AGENT_PENDING_SENTINEL, GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {"agent:coder:api_server:dm:sid": _AGENT_PENDING_SENTINEL}

    result = await runner.unload_profile_runtime("coder")

    assert result["blocked"] is True
    assert result["active_sessions"] == 1


def test_default_profile_unload_fence_covers_all_entry_aliases():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._profile_runtime_unloads = {"default": object()}
    runner._profile_runtime_unload_retry = set()
    for alias in ("", "default", "main"):
        assert runner._profile_runtime_unload_blocked(alias) is True

    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = {"main"}
    for alias in ("", "default", "main"):
        assert runner._profile_runtime_unload_blocked(alias) is True
    assert runner._profile_runtime_unload_blocked("coder") is False


@pytest.mark.asyncio
async def test_gateway_profile_unload_cleans_exact_profile_mcp_and_lsp(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    old_home = tmp_path / "profiles" / "coder-v1"
    seen = []
    monkeypatch.setattr(
        mcp_tool,
        "shutdown_mcp_profile",
        lambda: seen.append(("mcp", get_hermes_home())),
    )
    monkeypatch.setattr(
        lsp,
        "shutdown_service",
        lambda **kwargs: seen.append(("lsp", get_hermes_home(), kwargs)),
    )
    monkeypatch.setattr(
        mcp_startup,
        "clear_mcp_discovery_profile",
        lambda home: seen.append(("discovery", Path(home))),
    )

    result = await runner.unload_profile_runtime(
        "coder",
        profile_home=old_home,
    )

    assert result == {"evicted_sessions": 0, "disconnected_adapters": 0}
    assert seen == [
        ("mcp", old_home),
        ("discovery", old_home),
        ("lsp", old_home, {"raise_on_error": True}),
    ]


@pytest.mark.asyncio
async def test_gateway_profile_unload_cancellation_waits_for_cleanup_worker(
    tmp_path, monkeypatch
):
    """取消 unload 请求不能让不可取消的 MCP cleanup 脱离 barrier。"""
    from gateway.run import GatewayRunner
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0
    entered = threading.Event()
    release = threading.Event()

    def blocked_shutdown():
        entered.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(mcp_tool, "shutdown_mcp_profile", blocked_shutdown)
    task = asyncio.create_task(
        runner.unload_profile_runtime(
            "coder", profile_home=tmp_path / "profiles" / "coder"
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        assert runner._profile_runtime_unloads.get("coder") is not None
        with pytest.raises(RuntimeError, match="rejected during unload"):
            await runner._start_one_profile_adapters(
                "coder", tmp_path / "profiles" / "coder", {}
            )
        task.cancel()
        first_cancel_delivered = asyncio.Event()
        asyncio.get_running_loop().call_soon(first_cancel_delivered.set)
        await first_cancel_delivered.wait()
        assert not task.done()
        task.cancel()
        second_cancel_delivered = asyncio.Event()
        asyncio.get_running_loop().call_soon(second_cancel_delivered.set)
        await second_cancel_delivered.wait()
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runner._profile_runtime_unload_retry == {"coder"}
    finally:
        release.set()
        runner._shutdown_executor()


@pytest.mark.asyncio
async def test_profile_unload_waits_for_prefence_startup_cleanup(
    tmp_path, monkeypatch
):
    """真实 startup 在 fence 前进入 connect，取消后必须先清理再卸载。"""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._profile_adapter_operations = {}
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._retiring_adapter_cleanups = {}
    runner._published_adapter_cleanup_retry = {}
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0
    runner._adapter_disconnect_timeout_secs = lambda: 0
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

    class _Adapter:
        async def disconnect(self):
            cleanup_entered.set()
            await release_cleanup.wait()

    adapter = _Adapter()

    profile_cfg = GatewayConfig(multiplex_profiles=True)
    profile_cfg.platforms = {
        Platform.FEISHU: PlatformConfig(enabled=True, token="profile-token")
    }

    async def blocked_connect(_adapter, _platform):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
    runner._create_adapter = lambda _platform, _config: adapter
    runner._configure_profile_adapter = lambda *_args: None
    runner._adapter_credential_claim = lambda *_args: None
    runner._adapter_listener_claim = lambda *_args: None
    runner._connect_initial_adapter_with_timeout = blocked_connect

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_profile", lambda: None)
    monkeypatch.setattr(lsp, "shutdown_service", lambda **_kwargs: None)
    monkeypatch.setattr(
        mcp_startup, "clear_mcp_discovery_profile", lambda _home: None
    )
    profile_home = tmp_path / "profiles" / "coder"

    startup = asyncio.create_task(
        runner._start_one_profile_adapters("coder", profile_home, {})
    )
    await entered.wait()
    unload = asyncio.create_task(
        runner.unload_profile_runtime("coder", profile_home=profile_home)
    )
    cleanup_wait = asyncio.create_task(cleanup_entered.wait())
    try:
        completed, _pending = await asyncio.wait(
            {cleanup_wait, unload}, return_when=asyncio.FIRST_COMPLETED
        )
        assert cleanup_wait in completed
        assert not unload.done()
        assert runner._partial_adapter_cleanup_retry[
            ("coder", Platform.FEISHU)
        ] is adapter

        release_cleanup.set()
        result = await unload
        assert await startup == 0
        assert result == {"evicted_sessions": 0, "disconnected_adapters": 0}
        assert (
            "coder", Platform.FEISHU
        ) not in runner._partial_adapter_cleanup_retry
    finally:
        release_cleanup.set()
        if not startup.done():
            startup.cancel()
        if not unload.done():
            unload.cancel()
        await asyncio.gather(startup, unload, return_exceptions=True)
        if not cleanup_wait.done():
            cleanup_wait.cancel()
        await asyncio.gather(cleanup_wait, return_exceptions=True)


@pytest.mark.asyncio
async def test_profile_unload_retries_cleanup_ledger_after_cancelling_owner(
    tmp_path, monkeypatch
):
    """取消 cleanup owner 后，ledger 必须重放；失败不能被当成卸载成功。"""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._profile_adapter_operations = {}
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._retiring_adapter_cleanups = {}
    runner._published_adapter_cleanup_retry = {}
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0
    runner._adapter_disconnect_timeout_secs = lambda: 0
    first_cleanup_entered = asyncio.Event()
    first_cleanup_cancelled = asyncio.Event()
    retry_cleanup_entered = asyncio.Event()
    release_retry_cleanup = asyncio.Event()

    class _Adapter:
        def __init__(self):
            self.calls = 0

        async def disconnect(self):
            self.calls += 1
            if self.calls == 1:
                first_cleanup_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cleanup_cancelled.set()
                    raise RuntimeError("cleanup failed after cancellation")
            retry_cleanup_entered.set()
            await release_retry_cleanup.wait()

    adapter = _Adapter()
    profile_cfg = GatewayConfig(multiplex_profiles=True)
    profile_cfg.platforms = {
        Platform.FEISHU: PlatformConfig(enabled=True, token="profile-token")
    }
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
    runner._create_adapter = lambda _platform, _config: adapter
    runner._configure_profile_adapter = lambda *_args: None
    runner._adapter_credential_claim = lambda *_args: None
    runner._adapter_listener_claim = lambda *_args: None
    runner._connect_initial_adapter_with_timeout = AsyncMock(return_value=False)

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_profile", lambda: None)
    monkeypatch.setattr(lsp, "shutdown_service", lambda **_kwargs: None)
    monkeypatch.setattr(
        mcp_startup, "clear_mcp_discovery_profile", lambda _home: None
    )
    profile_home = tmp_path / "profiles" / "coder"
    startup = asyncio.create_task(
        runner._start_one_profile_adapters("coder", profile_home, {})
    )
    await first_cleanup_entered.wait()
    unload = asyncio.create_task(
        runner.unload_profile_runtime("coder", profile_home=profile_home)
    )
    retry_wait = asyncio.create_task(retry_cleanup_entered.wait())
    try:
        await first_cleanup_cancelled.wait()
        completed, _pending = await asyncio.wait(
            {retry_wait, unload}, return_when=asyncio.FIRST_COMPLETED
        )
        assert retry_wait in completed
        assert not unload.done()
        assert runner._partial_adapter_cleanup_retry[
            ("coder", Platform.FEISHU)
        ] is adapter

        release_retry_cleanup.set()
        assert await startup == 0
        assert await unload == {
            "evicted_sessions": 0,
            "disconnected_adapters": 0,
        }
        assert adapter.calls == 2
        assert ("coder", Platform.FEISHU) not in runner._partial_adapter_cleanup_retry
    finally:
        release_retry_cleanup.set()
        for task in (startup, unload, retry_wait):
            if not task.done():
                task.cancel()
        await asyncio.gather(startup, unload, retry_wait, return_exceptions=True)


@pytest.mark.asyncio
async def test_default_profile_unload_replays_cancelled_published_cleanup(
    tmp_path, monkeypatch
):
    """default unload 也必须重放 published ledger，不能因 adapter_map=None 假成功。"""
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._running_agents = {}
    runner._profile_adapters = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._profile_adapter_operations = {}
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._retiring_adapter_cleanups = {}
    runner._published_adapter_cleanup_retry = {}
    runner._published_adapter_cleanup_tasks = {}
    runner._background_tasks = set()
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0
    runner._adapter_disconnect_timeout_secs = lambda: 0
    first_retry_entered = asyncio.Event()
    first_retry_cancelled = asyncio.Event()
    replay_entered = asyncio.Event()
    release_replay = asyncio.Event()

    class _Adapter:
        def __init__(self):
            self.calls = 0

        async def disconnect(self):
            self.calls += 1
            if self.calls == 1:
                first_retry_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_retry_cancelled.set()
                    raise RuntimeError("published cleanup cancelled")
            replay_entered.set()
            await release_replay.wait()

    adapter = _Adapter()
    runner.adapters = {Platform.FEISHU: adapter}

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_profile", lambda: None)
    monkeypatch.setattr(lsp, "shutdown_service", lambda **_kwargs: None)
    monkeypatch.setattr(
        mcp_startup, "clear_mcp_discovery_profile", lambda _home: None
    )
    runner._schedule_published_adapter_cleanup_retry(
        "", Platform.FEISHU, adapter, runner.adapters
    )
    await first_retry_entered.wait()
    unload = asyncio.create_task(
        runner.unload_profile_runtime("", profile_home=tmp_path / "default")
    )
    replay_wait = asyncio.create_task(replay_entered.wait())
    try:
        await first_retry_cancelled.wait()
        completed, _pending = await asyncio.wait(
            {replay_wait, unload}, return_when=asyncio.FIRST_COMPLETED
        )
        assert replay_wait in completed
        assert not unload.done()
        assert runner.adapters[Platform.FEISHU] is adapter

        release_replay.set()
        result = await unload
        assert result == {"evicted_sessions": 0, "disconnected_adapters": 1}
        assert runner.adapters == {}
        assert ("", Platform.FEISHU) not in runner._published_adapter_cleanup_retry
    finally:
        runner._running = False
        release_replay.set()
        for task in (unload, replay_wait):
            if not task.done():
                task.cancel()
        await asyncio.gather(unload, replay_wait, return_exceptions=True)


@pytest.mark.asyncio
async def test_gateway_profile_unload_failure_keeps_exact_home_retry_ownership(
    tmp_path, monkeypatch
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    adapter = SimpleNamespace(disconnect=AsyncMock())
    runner._profile_adapters = {"coder": {Platform.FEISHU: adapter}}
    runner._agent_cache = {"agent:coder:feishu:dm:x": (object(),)}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    evicted = []
    runner._evict_cached_agents_for_profile = lambda profile: evicted.append(profile)

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    old_home = tmp_path / "profiles" / "coder-v1"
    monkeypatch.setattr(
        mcp_tool,
        "shutdown_mcp_profile",
        lambda: (_ for _ in ()).throw(RuntimeError("old child still running")),
    )
    monkeypatch.setattr(lsp, "shutdown_service", lambda **_kwargs: None)
    monkeypatch.setattr(mcp_startup, "clear_mcp_discovery_profile", lambda _home: None)

    with pytest.raises(RuntimeError, match="failed to unload"):
        await runner.unload_profile_runtime("coder", profile_home=old_home)

    assert runner._profile_adapters["coder"][Platform.FEISHU] is adapter
    adapter.disconnect.assert_not_awaited()
    assert evicted == []


@pytest.mark.asyncio
async def test_gateway_profile_unload_adapter_partial_failure_only_retries_owner(
    tmp_path, monkeypatch
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    import agent.lsp as lsp
    import hermes_cli.mcp_startup as mcp_startup
    import tools.mcp_tool as mcp_tool

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    good = SimpleNamespace(disconnect=AsyncMock())
    flaky = SimpleNamespace(
        disconnect=AsyncMock(side_effect=[RuntimeError("old poller alive"), None])
    )
    runner._profile_adapters = {
        "coder": {Platform.FEISHU: good, Platform.SLACK: flaky}
    }
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._evict_cached_agents_for_profile = lambda _profile: 0
    runner._adapter_disconnect_timeout_secs = lambda: 0

    async def _run_inline(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    runner._run_in_executor_with_context = _run_inline
    monkeypatch.setattr(mcp_tool, "shutdown_mcp_profile", lambda: None)
    monkeypatch.setattr(lsp, "shutdown_service", lambda **_kwargs: None)
    monkeypatch.setattr(mcp_startup, "clear_mcp_discovery_profile", lambda _home: None)
    profile_home = tmp_path / "profiles" / "coder"

    with pytest.raises(RuntimeError, match="old poller alive"):
        await runner.unload_profile_runtime("coder", profile_home=profile_home)
    assert Platform.FEISHU not in runner._profile_adapters["coder"]
    assert runner._profile_adapters["coder"][Platform.SLACK] is flaky
    assert runner._profile_runtime_unload_retry == {"coder"}

    result = await runner.unload_profile_runtime("coder", profile_home=profile_home)
    assert result["disconnected_adapters"] == 1
    assert good.disconnect.await_count == 1
    assert flaky.disconnect.await_count == 2
    assert "coder" not in runner._profile_adapters
    assert runner._profile_runtime_unload_retry == set()


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        coder_resp = await cli.get(
            "/p/coder/api/jobs?include_disabled=true",
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
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
        "/p/{profile}/v1/sessions/{session_id}/attachment/action",
        "/p/{profile}/v1/sessions/{session_id}/interrupt",
    }
    assert expected <= registered


def test_zet_base_routes_match_advertised_api_surface():
    """Zet must not hand-copy an older subset of APIServer routes while
    inheriting a capability document that advertises the newer endpoints."""
    adapter = _make_adapter()
    app = web.Application()

    adapter._register_base_http_routes(app.router)

    registered = {resource.canonical for resource in app.router.resources()}
    expected = {
        "/api/model/options",
        "/api/sessions/{session_id}/model",
        "/api/sessions/{session_id}/chat",
        "/v1/skills",
        "/v1/toolsets",
        "/v1/runs/{run_id}/approval",
    }
    assert expected <= registered
    assert {f"/p/{{profile}}{path}" for path in expected} <= registered


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
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        unload_resp = await cli.post(
            "/p/coder/v1/profile/unload",
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        unload_data = await unload_resp.json()

    assert reset_resp.status == 200
    assert unload_resp.status == 200
    assert unload_data["unloaded"] is True


@pytest.mark.asyncio
async def test_deferred_approval_response_requires_matching_id(monkeypatch):
    from tools import approval

    session_id = "sid-public-url"
    approval_session_key = "sid-x-hermes-header"
    approval.clear_session(approval_session_key)
    token = approval.set_current_session_key(approval_session_key)
    monkeypatch.setattr(
        approval, "_approval_profile_scope", lambda: "/profiles/coder"
    )
    monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        approval, "_command_matches_permanent_allowlist", lambda _command: False
    )
    monkeypatch.setattr(
        approval,
        "detect_dangerous_command",
        lambda command: (True, "agentcomputer:file.delete", f"risk:{command}"),
    )
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _command: {"action": "allow", "findings": [], "summary": ""},
        raising=False,
    )
    command = "agentcomputer file.delete notes/a.txt"
    first = approval.check_all_command_guards(command, "local")
    approval_id = first["approval_id"]

    adapter = _make_adapter()
    monkeypatch.setattr(
        adapter,
        "_goals",
        lambda: type("Goals", (), {"on_interaction_resolved": lambda self, sid: None})(),
    )
    app = web.Application()
    app.router.add_post(
        "/v1/sessions/{session_id}/approval/respond",
        adapter._handle_approval_respond,
    )
    headers = {"Authorization": "Bearer test-key-0123456789abcdef"}

    async with TestClient(TestServer(app)) as cli:
        missing = await cli.post(
            f"/v1/sessions/{session_id}/approval/respond",
            json={"choice": "once"},
            headers=headers,
        )
        wrong = await cli.post(
            f"/v1/sessions/{session_id}/approval/respond",
            json={"choice": "once", "approval_id": "A" * 32},
            headers=headers,
        )
        monkeypatch.setattr(
            approval, "_approval_profile_scope", lambda: "/profiles/other"
        )
        wrong_profile = await cli.post(
            f"/v1/sessions/{session_id}/approval/respond",
            json={"choice": "once", "approval_id": approval_id},
            headers=headers,
        )
        monkeypatch.setattr(
            approval, "_approval_profile_scope", lambda: "/profiles/coder"
        )
        matched = await cli.post(
            f"/v1/sessions/{session_id}/approval/respond",
            json={"choice": "once", "approval_id": approval_id},
            headers=headers,
        )
        missing_data = await missing.json()
        wrong_data = await wrong.json()
        wrong_profile_data = await wrong_profile.json()
        matched_data = await matched.json()

    assert missing_data == {"resolved": 0}
    assert wrong_data == {"resolved": 0}
    assert wrong_profile_data == {"resolved": 0}
    assert matched_data == {"resolved": 1}
    exact_retry = approval.check_all_command_guards(command, "local")
    consumed_retry = approval.check_all_command_guards(command, "local")
    different_retry = approval.check_all_command_guards(
        "agentcomputer file.delete notes/b.txt", "local"
    )
    assert exact_retry["approved"] is True
    assert exact_retry["one_shot_approved"] is True
    assert consumed_retry["status"] == "pending_approval"
    assert different_retry["status"] == "pending_approval"
    approval.clear_session(approval_session_key)
    approval.reset_current_session_key(token)


@pytest.mark.asyncio
async def test_legacy_live_approval_uses_x_hermes_session_key(monkeypatch):
    from tools import approval

    session_id = "sid-public-legacy"
    approval_session_key = "sid-header-legacy"
    entry = approval._ApprovalEntry({
        "approval_id": "B" * 32,
        "command": "agentcomputer file.delete notes/a.txt",
        "pattern_key": "agentcomputer:file.delete",
        "pattern_keys": ["agentcomputer:file.delete"],
    })
    with approval._lock:
        approval._gateway_queues[approval_session_key] = [entry]

    adapter = _make_adapter()
    adapter._approval_session_keys = {
        adapter._active_turn_key(session_id): approval_session_key,
    }
    monkeypatch.setattr(
        adapter,
        "_goals",
        lambda: type("Goals", (), {"on_interaction_resolved": lambda self, sid: None})(),
    )
    app = web.Application()
    app.router.add_post(
        "/v1/sessions/{session_id}/approval/respond",
        adapter._handle_approval_respond,
    )
    try:
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                f"/v1/sessions/{session_id}/approval/respond",
                json={"choice": "deny"},
                headers={"Authorization": "Bearer test-key-0123456789abcdef"},
            )
            data = await response.json()

        assert data == {"resolved": 1}
        assert entry.event.is_set()
        assert entry.result == "deny"
    finally:
        approval.cancel_session_approvals(approval_session_key)


@pytest.mark.asyncio
async def test_interrupt_revokes_deferred_approval_before_stale_response():
    from tools import approval

    session_id = "sid-interrupt-stale-approval"
    approval.clear_session(session_id)
    approval_id = approval.submit_pending(
        session_id,
        {
            "command": "agentcomputer file.delete notes/a.txt",
            "pattern_key": "agentcomputer:file.delete",
            "one_shot_pattern_key": "deferred:terminal:exact-a",
            "description": "delete notes/a.txt",
        },
    )
    assert approval_id

    adapter = _make_adapter()
    adapter._interrupt_pending_interactions(session_id, session_id)
    app = web.Application()
    app.router.add_post(
        "/v1/sessions/{session_id}/approval/respond",
        adapter._handle_approval_respond,
    )
    async with TestClient(TestServer(app)) as cli:
        stale = await cli.post(
            f"/v1/sessions/{session_id}/approval/respond",
            json={"choice": "once", "approval_id": approval_id},
            headers={"Authorization": "Bearer test-key-0123456789abcdef"},
        )
        stale_data = await stale.json()

    assert stale_data == {"resolved": 0}
    assert approval._consume_one_shot_approval(
        session_id, "deferred:terminal:exact-a"
    ) is False
    approval.clear_session(session_id)
