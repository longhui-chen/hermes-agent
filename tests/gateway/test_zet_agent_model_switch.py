import os
import types
from collections import OrderedDict

import pytest
import yaml
import queue

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeRequest:
    def __init__(self, body, match_info=None):
        self._body = body
        self.headers = {"Authorization": "Bearer test-key"}
        self.match_info = match_info or {}

    async def json(self):
        return self._body


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


@pytest.mark.asyncio
async def test_model_switch_writes_api_mode_and_context_length(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "default": "old",
                    "api_mode": "anthropic_messages",
                    "context_length": 128000,
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    resp = await adapter._handle_model_switch(
        _FakeRequest(
            {
                "model": "glm-5",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                "api_key": "local-ai-proxy",
                "api_mode": "openai_chat",
                "context_length": 1000000,
            }
        )
    )
    assert resp.status == 200

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"
    assert cfg["model"]["api_mode"] == "openai_chat"
    assert cfg["model"]["context_length"] == 1000000


@pytest.mark.asyncio
async def test_model_switch_clears_stale_api_mode_and_context_length(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "default": "old",
                    "api_mode": "anthropic_messages",
                    "context_length": 128000,
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    resp = await adapter._handle_model_switch(
        _FakeRequest(
            {
                "model": "glm-5",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                "api_key": "local-ai-proxy",
                "api_mode": "",
                "context_length": 0,
            }
        )
    )
    assert resp.status == 200

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"
    assert "api_mode" not in cfg["model"]
    assert "context_length" not in cfg["model"]


@pytest.mark.asyncio
async def test_session_model_switch_persists_override_no_note(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    evicted = []
    gw = types.SimpleNamespace(
        _session_model_overrides={},
        _evict_cached_agent=lambda sid: evicted.append(sid),
    )
    adapter.gateway_runner = gw

    session_id = "zettlab:user1:agent-1:42"
    resp = await adapter._handle_session_model_switch(
        _FakeRequest(
            {
                "model": "deepseek-v4",
                "provider": "custom",
                "supports_vision": False,
                "auxiliary": {"vision": {}},
            },
            match_info={"session_id": session_id},
        )
    )

    assert resp.status == 200
    # Override persisted + cached agent evicted. The identity note is no longer
    # pushed here — it's injected at the session's next turn by _run_agent's
    # open-time effective-model compare.
    state_key = f"agent:main:zet_agent:dm:{session_id}"
    assert gw._session_model_overrides[state_key]["model"] == "deepseek-v4"
    assert gw._session_model_overrides[state_key]["supports_vision"] is False
    assert gw._session_model_overrides[state_key]["auxiliary"] == {"vision": {}}
    assert evicted == [adapter._interaction_queue_key(session_id)]
    assert not hasattr(gw, "_pending_model_notes")


@pytest.mark.asyncio
async def test_agent_model_switch_writes_config_only(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"model": {"default": "old-model"}}), encoding="utf-8"
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    # Agent-level switch only writes config.yaml now; no broadcast, so it
    # doesn't need gateway_runner at all.
    adapter.gateway_runner = None

    resp = await adapter._handle_model_switch(
        _FakeRequest({"model": "glm-5", "provider": "custom"})
    )

    assert resp.status == 200
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"


def test_status_callback_forwards_context_compaction_to_tool_progress_lane():
    stream_q = queue.Queue()
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    cb = adapter._make_status_cb(stream_q)

    cb("context.compaction", {
        "state": "succeeded",
        "message": "上下文压缩成功",
        "old_session_id": "old",
        "new_session_id": "new",
    })

    tag, payload = stream_q.get_nowait()
    assert tag == "__tool_progress__"
    assert payload == {
        "type": "context.compaction",
        "state": "succeeded",
        "message": "上下文压缩成功",
        "old_session_id": "old",
        "new_session_id": "new",
    }


def test_status_callback_ignores_unstructured_status():
    stream_q = queue.Queue()
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    cb = adapter._make_status_cb(stream_q)

    cb("lifecycle", "Compacting context")

    assert stream_q.empty()


def test_status_callback_preserves_existing_callback():
    stream_q = queue.Queue()
    seen = []
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    cb = adapter._make_status_cb(
        stream_q,
        lambda kind, payload=None: seen.append((kind, payload)),
    )

    cb("lifecycle", "Compacting context")
    cb("context.compaction", {"state": "started"})

    assert seen == [("lifecycle", "Compacting context")]
    assert stream_q.get_nowait() == ("__tool_progress__", {
        "type": "context.compaction",
        "state": "started",
    })
    assert getattr(cb, "_hermes_accepts_structured_status") is True


def _seen_adapter(monkeypatch, *, config_model, seen, override=None):
    """Build a ZetAgentAdapter wired for open-time model-compare tests:
    a config default model, a pre-seeded seen-map, an optional session
    override, and a no-op persistence so tests don't touch disk."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: config_model)
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    scoped_overrides = {
        adapter._session_model_state_key(session_id): value
        for session_id, value in dict(override or {}).items()
    }
    adapter.gateway_runner = types.SimpleNamespace(
        _session_model_overrides=scoped_overrides
    )
    adapter._seen_models = OrderedDict(seen)
    adapter._seen_loaded = True
    monkeypatch.setattr(adapter, "_save_seen_models", lambda: None)
    return adapter


async def _capture_run_agent(monkeypatch, adapter, **kwargs):
    from gateway.platforms.api_server import APIServerAdapter

    captured = {}

    async def fake_super(self, **kw):
        captured.update(kw)
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(conversation_history=[], **kwargs)
    return captured


@pytest.mark.asyncio
async def test_run_agent_injects_note_on_effective_model_change(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={"sess-1": "deepseek-v4"})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    assert captured["user_message"].startswith("[Note: the model has changed and is now glm-5.1")
    assert captured["user_message"].endswith("hi")
    # last-seen updated to the new effective model.
    assert adapter._seen_models["sess-1"] == "glm-5.1"


@pytest.mark.asyncio
async def test_run_agent_no_note_when_model_unchanged(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={"sess-1": "glm-5.1"})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    assert captured["user_message"] == "hi"


@pytest.mark.asyncio
async def test_run_agent_forwards_structured_plan_ack(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    plan_ack = {
        "turn_id": "plan-turn-1",
        "status": "cancelled",
        "revision_requested": False,
    }

    captured = await _capture_run_agent(
        monkeypatch,
        adapter,
        user_message="cancel",
        session_id="plan-session",
        plan_ack=plan_ack,
    )

    assert captured["plan_ack"] == plan_ack


@pytest.mark.asyncio
async def test_run_agent_legacy_unbound_plan_ack_flow_has_no_execution_capability(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import business_execution_token, current_turn_identity
    from tools.environments.local import build_video_edit_runtime_env

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    captured = {}
    boundary_checks = []
    monkeypatch.setattr(
        "gateway.platforms.zet_agent.gateway_sensitive_process_boundary_ready",
        lambda: boundary_checks.append(True) or True,
    )

    async def fake_super(self, **kwargs):
        with pytest.raises(
            PermissionError,
            match="trusted video-edit execution receipt unavailable",
        ):
            build_video_edit_runtime_env({})
        captured["runtime_rejected"] = True
        captured["turn_identity"] = current_turn_identity()
        captured["scoped_token"] = business_execution_token()
        captured["forwarded_token"] = kwargs["business_execution_token"]
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(
        user_message="确认执行",
        conversation_history=[],
        session_id="legacy-plan-session",
        turn_id="",
        business_execution_token="a" * 64,
        plan_ack={
            "status": "confirmed",
            "revision_requested": False,
        },
    )

    assert captured == {
        "runtime_rejected": True,
        "turn_identity": None,
        "scoped_token": "",
        "forwarded_token": "",
    }
    assert boundary_checks == [True]


@pytest.mark.asyncio
async def test_run_agent_direct_unbound_flow_preserves_business_capability(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import business_execution_token
    from tools.environments.local import build_video_edit_runtime_env

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    captured = {}
    monkeypatch.setattr(
        "gateway.platforms.zet_agent.gateway_sensitive_process_boundary_ready",
        lambda: True,
    )

    async def fake_super(self, **kwargs):
        with pytest.raises(
            PermissionError,
            match="trusted video-edit execution receipt unavailable",
        ):
            build_video_edit_runtime_env({})
        captured["video_runtime_rejected"] = True
        captured["scoped_token"] = business_execution_token()
        captured["forwarded_token"] = kwargs["business_execution_token"]
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(
        user_message="直接执行",
        conversation_history=[],
        session_id="direct-session",
        turn_id="",
        business_execution_token="a" * 64,
        plan_ack={},
    )

    assert captured == {
        "video_runtime_rejected": True,
        "scoped_token": "a" * 64,
        "forwarded_token": "a" * 64,
    }


@pytest.mark.asyncio
async def test_run_agent_cancelled_plan_ack_drops_business_capability_unit(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import business_execution_token

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    captured = {}
    monkeypatch.setattr(
        "gateway.platforms.zet_agent.gateway_sensitive_process_boundary_ready",
        lambda: True,
    )

    async def fake_super(self, **kwargs):
        captured["scoped_token"] = business_execution_token()
        captured["forwarded_token"] = kwargs["business_execution_token"]
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(
        user_message="取消计划",
        conversation_history=[],
        session_id="plan-session",
        turn_id="confirmation-turn-2",
        business_execution_token="a" * 64,
        plan_ack={
            "turn_id": "plan-turn-1",
            "status": "cancelled",
            "revision_requested": False,
        },
    )

    assert captured == {"scoped_token": "", "forwarded_token": ""}


@pytest.mark.asyncio
async def test_run_agent_binds_structured_plan_receipt_only_for_current_turn(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from tools.environments.local import LocalEnvironment

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    monkeypatch.setattr(
        "gateway.platforms.zet_agent.gateway_sensitive_process_boundary_ready",
        lambda: True,
    )
    captured = []
    scoped_tokens = []
    env = LocalEnvironment()

    async def fake_super(self, **kw):
        from gateway.session_context import business_execution_token

        result = env.execute(
            "printf '%s|%s|%s|%s|%s' \"$HERMES_TURN_ID\" "
            "\"$HERMES_PLAN_ACK_STATUS\" \"$HERMES_PLAN_ACK_TURN_ID\" "
            "\"$HERMES_PLAN_ACK_REVISION_REQUESTED\" "
            "\"$ZETTLAB_BUSINESS_EXECUTION_TOKEN\""
        )
        captured.append(result["output"].split("|"))
        scoped_tokens.append(business_execution_token())
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(
        user_message="plan",
        conversation_history=[],
        session_id="plan-session",
        turn_id="plan-turn-1",
    )
    await adapter._run_agent(
        user_message="确认执行",
        conversation_history=[],
        session_id="plan-session",
        turn_id="confirmation-turn-2",
        business_execution_token="a" * 64,
        plan_ack={
            "turn_id": "plan-turn-1",
            "status": "cancelled",
            "revision_requested": True,
        },
    )

    assert captured == [
        ["plan-turn-1", "", "", "", ""],
        ["confirmation-turn-2", "cancelled", "plan-turn-1", "1", ""],
    ]
    assert scoped_tokens == ["", ""]
    assert os.environ.get("HERMES_TURN_ID") is None
    assert os.environ.get("HERMES_PLAN_ACK_STATUS") is None
    assert os.environ.get("HERMES_PLAN_ACK_TURN_ID") is None
    assert os.environ.get("HERMES_PLAN_ACK_REVISION_REQUESTED") is None
    assert os.environ.get("ZETTLAB_BUSINESS_EXECUTION_TOKEN") is None


@pytest.mark.asyncio
async def test_run_agent_requires_process_boundary_before_binding_business_token(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import business_execution_token

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    events = []

    def boundary_ready():
        events.append(("boundary", business_execution_token()))
        return True

    async def fake_super(self, **_kwargs):
        events.append(("model", business_execution_token()))
        return ({}, {})

    monkeypatch.setattr(
        zet_agent,
        "gateway_sensitive_process_boundary_ready",
        boundary_ready,
        raising=False,
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)

    await adapter._run_agent(
        user_message="确认执行",
        conversation_history=[],
        session_id="plan-session",
        turn_id="confirmation-turn-2",
        business_execution_token="a" * 64,
    )

    assert events == [
        ("boundary", ""),
        ("model", "a" * 64),
    ]
    assert business_execution_token() == ""


@pytest.mark.asyncio
async def test_run_agent_boundary_failure_keeps_business_token_out_of_context(
    monkeypatch,
):
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_context import business_execution_token

    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    model_started = []

    async def fake_super(self, **_kwargs):
        model_started.append(True)
        return ({}, {})

    monkeypatch.setattr(
        zet_agent,
        "gateway_sensitive_process_boundary_ready",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)

    with pytest.raises(
        PermissionError,
        match="gateway process memory boundary is unavailable",
    ):
        await adapter._run_agent(
            user_message="确认执行",
            conversation_history=[],
            session_id="plan-session",
            turn_id="confirmation-turn-2",
            business_execution_token="a" * 64,
        )

    assert business_execution_token() == ""
    assert model_started == []


@pytest.mark.asyncio
async def test_run_agent_new_session_records_baseline_no_note(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="new-sess")
    # New session: no note (its system prompt is already built with the
    # current model), but baseline recorded so a later switch is detected.
    assert captured["user_message"] == "hi"
    assert adapter._seen_models["new-sess"] == "glm-5.1"


@pytest.mark.asyncio
async def test_run_agent_effective_model_prefers_override(monkeypatch):
    adapter = _seen_adapter(
        monkeypatch,
        config_model="config-default",
        seen={"sess-1": "config-default"},
        override={"sess-1": {"model": "override-model"}},
    )
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    # Effective model = override (not config default) → note announces it.
    assert captured["user_message"].startswith("[Note: the model has changed and is now override-model")


@pytest.mark.asyncio
async def test_run_agent_evicts_oldest_seen_over_cap(monkeypatch):
    # Bound the map so it can't grow without limit (Hard Rule 第 1 条).
    monkeypatch.setattr(zet_agent, "_SEEN_MODELS_CAP", 2)
    adapter = _seen_adapter(monkeypatch, config_model="m1", seen={})
    for sid in ("s1", "s2", "s3"):
        await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id=sid)
    # cap=2 → oldest (s1) evicted LRU-style.
    assert len(adapter._seen_models) == 2
    assert set(adapter._seen_models) == {"s2", "s3"}


@pytest.mark.asyncio
async def test_run_agent_active_session_not_evicted(monkeypatch):
    # An active session that keeps the SAME model must refresh its LRU position
    # each open, so it isn't wrongly evicted as "oldest" (which would drop its
    # baseline and miss the next switch's note).
    monkeypatch.setattr(zet_agent, "_SEEN_MODELS_CAP", 2)
    adapter = _seen_adapter(monkeypatch, config_model="m1", seen={})
    # s_active opens, then s_other, then s_active again (same model m1 → elif branch).
    for sid in ("s_active", "s_other", "s_active"):
        await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id=sid)
    # New session pushes over cap=2 → the truly-oldest (s_other) is evicted,
    # not s_active (which was touched by its second open).
    await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="s_new")
    assert "s_active" in adapter._seen_models
    assert "s_other" not in adapter._seen_models
    assert set(adapter._seen_models) == {"s_active", "s_new"}


# ---------------------------------------------------------------------------
# DELETE /v1/sessions/{sid}/model — inverse of the switch endpoint above.
# Local-server forwards DELETE here right after deleting the on-disk override
# from session_model_overrides.json. The handler's contract: drop the live
# in-memory override + evict the cached agent so the next turn rebuilds on
# config.yaml's default, preserving conversation history.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_model_clear_pops_override_and_evicts(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    evicted = []
    session_id = "zettlab:user1:agent-1:42"
    state_key = f"agent:main:zet_agent:dm:{session_id}"
    gw = types.SimpleNamespace(
        _session_model_overrides={
            state_key: {"model": "deepseek-v4", "provider": "custom"},
            "other-session": {"model": "glm-5", "provider": "custom"},
        },
        _evict_cached_agent=lambda sid: evicted.append(sid),
    )
    adapter.gateway_runner = gw

    resp = await adapter._handle_session_model_clear(
        _FakeRequest(None, match_info={"session_id": session_id})
    )

    assert resp.status == 200
    assert resp.payload["ok"] is True
    assert resp.payload["cleared"] is True
    assert resp.payload["session_id"] == session_id
    # Target session's override is gone; the unrelated entry survives.
    assert state_key not in gw._session_model_overrides
    assert "other-session" in gw._session_model_overrides
    # Cached agent for this session was evicted so the next turn rebuilds.
    assert evicted == [adapter._interaction_queue_key(session_id)]


@pytest.mark.asyncio
async def test_session_model_switch_isolates_same_id_by_profile(monkeypatch):
    from gateway.platforms.api_server import _api_request_profile

    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None
    adapter.gateway_runner = types.SimpleNamespace(
        _session_model_overrides={},
        _evict_cached_agent=lambda _key: None,
    )
    session_id = "same-public-session"

    main_token = _api_request_profile.set("main")
    try:
        await adapter._handle_session_model_switch(
            _FakeRequest({"model": "model-main"}, {"session_id": session_id})
        )
    finally:
        _api_request_profile.reset(main_token)

    coder_token = _api_request_profile.set("coder")
    try:
        await adapter._handle_session_model_switch(
            _FakeRequest({"model": "model-coder"}, {"session_id": session_id})
        )
    finally:
        _api_request_profile.reset(coder_token)

    assert adapter.gateway_runner._session_model_overrides == {
        f"agent:main:zet_agent:dm:{session_id}": {"model": "model-main"},
        f"agent:coder:zet_agent:dm:{session_id}": {"model": "model-coder"},
    }


def test_seen_models_cache_is_profile_local(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from gateway.session_seen_models import save_seen_models

    main_home = tmp_path / "main"
    coder_home = tmp_path / "coder"
    save_seen_models({"main-session": "model-main"}, main_home / "session_seen_models.json")
    save_seen_models({"coder-session": "model-coder"}, coder_home / "session_seen_models.json")
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

    main_token = set_hermes_home_override(str(main_home))
    try:
        main_seen = adapter._ensure_seen_models()
    finally:
        reset_hermes_home_override(main_token)

    coder_token = set_hermes_home_override(str(coder_home))
    try:
        coder_seen = adapter._ensure_seen_models()
        coder_seen["new-coder-session"] = "model-coder-2"
        adapter._save_seen_models()
    finally:
        reset_hermes_home_override(coder_token)

    assert main_seen == {"main-session": "model-main"}
    assert coder_seen == {
        "coder-session": "model-coder",
        "new-coder-session": "model-coder-2",
    }
    assert "main-session" not in (
        coder_home / "session_seen_models.json"
    ).read_text(encoding="utf-8")


def test_last_resolved_model_cache_is_bounded(tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._last_resolved_model_cap = 2
    profile_home = tmp_path / "profile"
    token = set_hermes_home_override(str(profile_home))
    try:
        adapter._remember_last_resolved_model("s1", "m1")
        adapter._remember_last_resolved_model("s2", "m2")
        adapter._remember_last_resolved_model("s3", "m3")
    finally:
        reset_hermes_home_override(token)

    prefix = str(profile_home.resolve())
    assert set(adapter._last_resolved_model) == {
        (prefix, "s2"),
        (prefix, "s3"),
        (prefix, None),
    }


def test_last_resolved_model_fallback_is_profile_scoped_and_unloadable(tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    main_home = tmp_path / "main"
    coder_home = tmp_path / "coder"

    main_token = set_hermes_home_override(str(main_home))
    try:
        adapter._remember_last_resolved_model("", "main-model")
        assert adapter._last_resolved_model_for("") == "main-model"
    finally:
        reset_hermes_home_override(main_token)

    coder_token = set_hermes_home_override(str(coder_home))
    try:
        assert adapter._last_resolved_model_for("") is None
        adapter._remember_last_resolved_model("", "coder-model")
        assert adapter._last_resolved_model_for("") == "coder-model"
    finally:
        reset_hermes_home_override(coder_token)

    adapter._drop_profile_local_model_caches(str(coder_home))
    assert (str(coder_home.resolve()), None) not in adapter._last_resolved_model
    assert (str(main_home.resolve()), None) in adapter._last_resolved_model


def test_response_format_precheck_uses_profile_scoped_model_override(monkeypatch):
    import gateway.run as gateway_run
    from gateway.platforms.api_server import _api_request_profile

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    profile_token = _api_request_profile.set("coder")
    try:
        adapter.gateway_runner = types.SimpleNamespace(
            _session_model_overrides={
                adapter._session_model_state_key("public-session"): {
                    "provider": "anthropic",
                    "api_mode": "anthropic_messages",
                }
            }
        )
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai",
                "api_mode": "responses",
                "base_url": "",
            },
        )
        error = adapter._response_format_transport_error(
            {"response_format": {"type": "json_object"}},
            gateway_session_key="public-session",
        )
    finally:
        _api_request_profile.reset(profile_token)

    assert error is not None and "Anthropic" in error


@pytest.mark.asyncio
async def test_session_model_clear_is_idempotent_when_no_override(monkeypatch):
    """Clearing a session that has no override returns cleared=False without
    invoking the eviction hook — there is no cached agent built on a stale
    override to throw away, so calling evict would be a wasted rebuild."""
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    evicted = []
    gw = types.SimpleNamespace(
        _session_model_overrides={},  # No overrides at all.
        _evict_cached_agent=lambda sid: evicted.append(sid),
    )
    adapter.gateway_runner = gw

    resp = await adapter._handle_session_model_clear(
        _FakeRequest(None, match_info={"session_id": "ghost-session"})
    )

    assert resp.status == 200
    assert resp.payload["ok"] is True
    assert resp.payload["cleared"] is False
    assert evicted == []  # Eviction skipped on no-op.


@pytest.mark.asyncio
async def test_session_model_clear_without_gateway_runner(monkeypatch):
    """If gateway_runner isn't wired (pathological / startup race), the handler
    must still return 200 cleared=False rather than raise — local-server's
    forward path is best-effort and any 5xx would mask the persisted delete."""
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None
    adapter.gateway_runner = None

    resp = await adapter._handle_session_model_clear(
        _FakeRequest(None, match_info={"session_id": "any-session"})
    )

    assert resp.status == 200
    assert resp.payload["ok"] is True
    assert resp.payload["cleared"] is False
