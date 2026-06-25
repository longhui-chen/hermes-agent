"""Phase 1: zet_agent `/p/<profile>` API routes for local-server mux mode."""

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
