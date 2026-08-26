"""Zet adapter integration contracts for ordinary runtime-shell reuse."""

import asyncio
import os
from pathlib import Path

import pytest

from agent.prestream_timing import PRESTREAM_TIMING_CONTEXT
from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import (
    ZetAgentAdapter,
    _api_request_profile,
    _deep_memory_principal,
    _deep_memory_subject,
    _zettlab_request_account_id,
    _zet_runtime_shell_cache_allowed,
)
from gateway.session_context import (
    pop_zettlab_auth_principal,
    push_zettlab_auth_principal,
)


class _SessionDB:
    def __init__(self) -> None:
        self.message_counts = {}

    def get_session(self, session_id):
        if session_id not in self.message_counts:
            return None
        return {"message_count": self.message_counts[session_id]}


class _FakeAgent:
    constructed = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self._session_db = kwargs.get("session_db")
        self._gateway_session_key = kwargs.get("gateway_session_key")
        self._profile_name = kwargs.get("profile_name")
        self._interrupt_requested = False
        self._persist_disabled = False
        self._last_flushed_db_idx = 0
        self._end_session_on_close = True
        self.end_session_calls = 0
        self.released = 0
        self.closed = 0
        self.__class__.constructed.append(self)

    def release_clients(self):
        self.released += 1

    def close(self):
        self.closed += 1
        if self._end_session_on_close:
            self.end_session_calls += 1

    def run_conversation(self, **_kwargs):
        current = self._session_db.message_counts.get(self.session_id, 0)
        self._session_db.message_counts[self.session_id] = current + 2
        self._current_turn_id = f"turn-{current // 2 + 1}"
        self.session_prompt_tokens = 10
        self.session_completion_tokens = 4
        self.session_total_tokens = 14
        return {
            "final_response": "ok",
            "completed": True,
            "failed": False,
        }

    def _drain_pending_steer(self, **_kwargs):
        return None


@pytest.fixture
def runtime_adapter(monkeypatch, tmp_path):
    profile_home = tmp_path / "profiles" / "main"
    profile_home.mkdir(parents=True)
    db = _SessionDB()
    runtime = {
        "provider": "test-provider",
        "model": "test/model",
        "api_key": "credential-a",
        "base_url": "https://example.invalid/v1",
    }
    _FakeAgent.constructed = []

    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs", lambda: dict(runtime)
    )
    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model", lambda: runtime["model"]
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        lambda: {"enabled": False},
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", lambda: None
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda *_: set()
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: Path(profile_home)
    )

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: db)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)
    retired = []
    monkeypatch.setattr(
        adapter,
        "_schedule_runtime_shell_retirement",
        lambda agents: retired.extend(tuple(agents or ())),
    )
    return adapter, db, runtime, retired, profile_home


def _create(
    adapter,
    *,
    session_id="session-1",
    gateway_session_key="/profiles/main|session-1",
    request_overrides=None,
    requested_model=None,
    model_options=None,
    confirmed_runtime_lock=False,
):
    return adapter._create_agent(
        session_id=session_id,
        gateway_session_key=gateway_session_key,
        request_overrides=request_overrides,
        requested_model=requested_model,
        model_options=model_options,
        confirmed_runtime_lock=confirmed_runtime_lock,
    )


def _interactive_create(adapter, **kwargs):
    token = _zet_runtime_shell_cache_allowed.set(True)
    try:
        return _create(adapter, **kwargs)
    finally:
        _zet_runtime_shell_cache_allowed.reset(token)


def test_second_interactive_turn_reuses_shell_and_rebinds_request_state(
    runtime_adapter,
):
    adapter, db, _runtime, _retired, _home = runtime_adapter
    old_stream = object()
    first = _interactive_create(adapter)
    first.stream_delta_callback = old_stream
    first.session_total_tokens = 99
    first._current_user_message = "must not survive"
    first._db_flush_scan_prefix = [{"role": "user", "content": "old"}]
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    assert first._db_flush_scan_prefix is None

    new_stream = object()
    token = _zet_runtime_shell_cache_allowed.set(True)
    try:
        second = adapter._create_agent(
            session_id="session-1",
            gateway_session_key="/profiles/main|session-1",
            stream_delta_callback=new_stream,
        )
    finally:
        _zet_runtime_shell_cache_allowed.reset(token)

    assert second is first
    assert len(_FakeAgent.constructed) == 1
    assert second.stream_delta_callback is new_stream
    assert second.session_total_tokens == 0
    assert second._current_user_message == ""
    assert second._zet_runtime_shell_force_tool_refresh is True


def test_session_prewarm_populates_only_the_exact_identity_shell_without_running_llm(
    runtime_adapter,
    monkeypatch,
):
    adapter, db, _runtime, retired, _home = runtime_adapter
    real_init = _FakeAgent.__init__

    def init_with_real_session_mirror(self, **kwargs):
        from gateway.session_context import set_current_session_id

        set_current_session_id(kwargs["session_id"])
        real_init(self, **kwargs)

    monkeypatch.setattr(_FakeAgent, "__init__", init_with_real_session_mirror)
    monkeypatch.setenv("HERMES_SESSION_ID", "foreground-session")
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._prewarm_runtime_shell_sync("session-1") is True
        assert db.message_counts == {}
        assert os.environ["HERMES_SESSION_ID"] == "foreground-session"
        warmed = _FakeAgent.constructed[0]

        same_identity = _interactive_create(
            adapter,
            session_id="session-1",
            gateway_session_key="session-1",
        )
        assert same_identity is warmed
        adapter._finish_runtime_shell_turn(same_identity, reusable=True)
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert len(_FakeAgent.constructed) == 1
    assert retired == []
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 1,
        "idle": 1,
        "leased": 0,
    }


def test_session_prewarm_never_reuses_across_account_or_principal(
    runtime_adapter,
):
    adapter, _db, _runtime, retired, _home = runtime_adapter
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._prewarm_runtime_shell_sync("session-1") is True
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    account_token = _zettlab_request_account_id.set("account-b")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-b")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-b")
    deep_subject_token = _deep_memory_subject.set("account-b")
    try:
        other_identity = _interactive_create(
            adapter,
            session_id="session-1",
            gateway_session_key="session-1",
        )
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert len(_FakeAgent.constructed) == 2
    assert other_identity is _FakeAgent.constructed[1]
    assert retired == [_FakeAgent.constructed[0]]


def test_session_prewarm_releases_a_runtime_that_cannot_enter_the_cache(
    runtime_adapter,
):
    adapter, _db, runtime, _retired, _home = runtime_adapter
    runtime["api_mode"] = "codex_app_server"
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._prewarm_runtime_shell_sync("session-1") is False
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert len(_FakeAgent.constructed) == 1
    assert _FakeAgent.constructed[0].released == 1
    assert adapter._runtime_shell_cache.counts()["entries"] == 0


@pytest.mark.asyncio
async def test_session_prewarm_scheduler_deduplicates_caps_and_releases_profile_barriers(
    runtime_adapter,
    monkeypatch,
):
    adapter, _db, _runtime, _retired, profile_home = runtime_adapter
    releases = {
        "session-1": asyncio.Event(),
        "session-2": asyncio.Event(),
    }
    started = []

    async def fake_prewarm(session_id: str) -> bool:
        started.append(session_id)
        await releases[session_id].wait()
        return True

    monkeypatch.setattr(adapter, "_prewarm_runtime_shell", fake_prewarm)
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._schedule_runtime_shell_prewarm("session-1") is True
        assert adapter._schedule_runtime_shell_prewarm("session-1") is False
        assert adapter._schedule_runtime_shell_prewarm("session-2") is True
        assert adapter._schedule_runtime_shell_prewarm("session-3") is False
        await asyncio.sleep(0)
        assert sorted(started) == ["session-1", "session-2"]
        assert adapter._active_profile_chat_runs(profile_home) == 2
        releases["session-1"].set()
        releases["session-2"].set()
        await asyncio.gather(
            *tuple(adapter._runtime_shell_prewarm_tasks.values())
        )
        await asyncio.sleep(0)
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert adapter._runtime_shell_prewarm_tasks == {}
    assert adapter._active_profile_chat_runs(profile_home) == 0


@pytest.mark.asyncio
async def test_session_prewarm_scheduler_releases_profile_barrier_after_failure(
    runtime_adapter,
    monkeypatch,
):
    adapter, _db, _runtime, _retired, profile_home = runtime_adapter

    async def failing_prewarm(_session_id: str) -> bool:
        raise RuntimeError("private prewarm failure")

    monkeypatch.setattr(adapter, "_prewarm_runtime_shell", failing_prewarm)
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._schedule_runtime_shell_prewarm("session-1") is True
        tasks = tuple(adapter._runtime_shell_prewarm_tasks.values())
        assert await asyncio.gather(*tasks) == [False]
        await asyncio.sleep(0)
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert adapter._runtime_shell_prewarm_tasks == {}
    assert adapter._active_profile_chat_runs(profile_home) == 0


def test_runtime_shell_timing_distinguishes_created_from_cache_hit(
    runtime_adapter,
):
    class _Timing:
        def __init__(self):
            self.started = 0
            self.outcomes = []

        def agent_shell_started(self):
            self.started += 1

        def agent_shell_finished(self, outcome):
            self.outcomes.append(outcome)

    adapter, db, _runtime, _retired, _home = runtime_adapter
    timing = _Timing()
    token = PRESTREAM_TIMING_CONTEXT.set(timing)
    try:
        first = _interactive_create(adapter)
        db.message_counts["session-1"] = 2
        adapter._finish_runtime_shell_turn(first, reusable=True)
        second = _interactive_create(adapter)
    finally:
        PRESTREAM_TIMING_CONTEXT.reset(token)

    assert second is first
    assert timing.started == 2
    assert timing.outcomes == ["created", "runtime_cache_hit"]


def test_reuse_refreshes_reasoning_service_tier_and_request_overrides(
    runtime_adapter,
):
    adapter, db, _runtime, _retired, _home = runtime_adapter
    first = _interactive_create(
        adapter,
        model_options={"reasoning_effort": "high", "service_tier": "priority"},
        request_overrides={"speed": "fast"},
    )
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    second = _interactive_create(
        adapter,
        model_options={"reasoning_effort": "low", "service_tier": "flex"},
        request_overrides={"response_format": {"type": "json_object"}},
    )

    assert second is first
    assert second.reasoning_config == {"enabled": True, "effort": "low"}
    assert second.service_tier == "flex"
    assert second.request_overrides == {
        "response_format": {"type": "json_object"}
    }


@pytest.mark.asyncio
async def test_public_run_agent_path_reuses_and_releases_shell_each_turn(
    runtime_adapter, monkeypatch
):
    adapter, db, _runtime, _retired, _home = runtime_adapter
    monkeypatch.setattr(adapter, "_effective_model", lambda *_args: "")
    monkeypatch.setattr(
        "tools.zettlab_snapshot_guard.finish_turn", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.collect_answer_attribution_citations",
        lambda *_args, **_kwargs: None,
    )

    first_result, first_usage = await adapter._run_agent(
        user_message="[ZETTLAB:test] first",
        conversation_history=[],
        session_id="session-1",
        gateway_session_key="/profiles/main|session-1",
        turn_id="turn-1",
    )
    second_result, second_usage = await adapter._run_agent(
        user_message="[ZETTLAB:test] second",
        conversation_history=[],
        session_id="session-1",
        gateway_session_key="/profiles/main|session-1",
        turn_id="turn-2",
    )

    assert first_result["final_response"] == "ok"
    assert second_result["final_response"] == "ok"
    assert first_usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert second_usage == first_usage
    assert len(_FakeAgent.constructed) == 1
    assert db.message_counts["session-1"] == 4
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 1,
        "idle": 1,
        "leased": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_state",
    [
        {"failed": True, "completed": False},
        {"interrupted": True, "completed": False},
    ],
)
async def test_failed_or_interrupted_turn_is_not_reused(
    runtime_adapter, monkeypatch, terminal_state
):
    adapter, _db, _runtime, retired, _home = runtime_adapter
    monkeypatch.setattr(adapter, "_effective_model", lambda *_args: "")
    monkeypatch.setattr(
        "tools.zettlab_snapshot_guard.finish_turn", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        _FakeAgent,
        "run_conversation",
        lambda self, **_kwargs: {
            "final_response": "not reusable",
            **terminal_state,
        },
    )

    await adapter._run_agent(
        user_message="[ZETTLAB:test] terminal state",
        conversation_history=[],
        session_id="session-1",
        gateway_session_key="/profiles/main|session-1",
        turn_id="turn-1",
    )

    assert retired == [_FakeAgent.constructed[0]]
    assert adapter._runtime_shell_cache.counts()["entries"] == 0


@pytest.mark.parametrize(
    "create_kwargs",
    [
        {"request_overrides": {"tool_choice": "none"}},
        {"request_overrides": {"_zet_execution_policy": "silent_automation"}},
        {"confirmed_runtime_lock": True},
        {"gateway_session_key": None},
    ],
)
def test_special_or_unstable_requests_bypass_runtime_cache(
    runtime_adapter, create_kwargs
):
    adapter, _db, _runtime, _retired, _home = runtime_adapter

    first = _interactive_create(adapter, **create_kwargs)
    second = _interactive_create(adapter, **create_kwargs)

    assert first is not second
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 0,
        "idle": 0,
        "leased": 0,
    }


def test_command_style_runtime_bypasses_runtime_cache(runtime_adapter):
    adapter, _db, runtime, _retired, _home = runtime_adapter
    runtime["api_mode"] = "codex_app_server"

    first = _interactive_create(adapter)
    second = _interactive_create(adapter)

    assert first is not second
    assert adapter._runtime_shell_cache.counts()["entries"] == 0


def test_direct_create_agent_path_does_not_cache_async_runs(runtime_adapter):
    adapter, _db, _runtime, _retired, _home = runtime_adapter

    first = _create(adapter)
    second = _create(adapter)

    assert first is not second
    assert adapter._runtime_shell_cache.counts()["entries"] == 0


def test_unexpected_overlapping_session_gets_temporary_agent(runtime_adapter):
    adapter, db, _runtime, retired, _home = runtime_adapter
    owner = _interactive_create(adapter)

    temporary = _interactive_create(adapter)

    assert temporary is not owner
    assert temporary._zet_runtime_shell_ephemeral is True
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 1,
        "idle": 0,
        "leased": 1,
    }
    adapter._finish_runtime_shell_turn(temporary, reusable=True)
    assert retired == [temporary]
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(owner, reusable=True)
    assert adapter._runtime_shell_cache.counts()["idle"] == 1


@pytest.mark.parametrize("change", ["model", "credential"])
def test_model_or_credential_switch_rebuilds_shell(runtime_adapter, change):
    adapter, db, runtime, retired, _home = runtime_adapter
    first = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    kwargs = {}
    if change == "model":
        kwargs["requested_model"] = "test/model-v2"
    else:
        runtime["api_key"] = "credential-b"
    second = _interactive_create(adapter, **kwargs)

    assert second is not first
    assert retired == [first]
    assert len(_FakeAgent.constructed) == 2


def test_external_session_db_change_rebuilds_shell(runtime_adapter):
    adapter, db, _runtime, retired, _home = runtime_adapter
    first = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    db.message_counts["session-1"] = 99
    second = _interactive_create(adapter)

    assert second is not first
    assert retired == [first]


def test_registry_generation_change_does_not_destroy_reusable_shell(
    runtime_adapter, monkeypatch
):
    adapter, db, _runtime, retired, _home = runtime_adapter
    generation = [(1, 1)]
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._extract_cache_busting_config",
        lambda _cfg: {
            "tools.registry_generation": generation[0],
            "compression.enabled": True,
        },
    )
    first = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    generation[0] = (2, 4)
    second = _interactive_create(adapter)

    assert second is first
    assert retired == []


def test_profile_identity_prevents_cross_profile_reuse(runtime_adapter, monkeypatch):
    adapter, db, _runtime, _retired, _home = runtime_adapter
    first = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    coder_home = Path(adapter._profile_home_key()).parent / "coder"
    coder_home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: coder_home)
    profile_token = _api_request_profile.set("coder")
    try:
        second = _interactive_create(
            adapter,
            gateway_session_key="/profiles/coder|session-1",
        )
    finally:
        _api_request_profile.reset(profile_token)

    assert second is not first
    assert adapter._runtime_shell_cache.counts()["entries"] == 2


def test_profile_cleanup_hard_closes_only_target_shell(runtime_adapter, monkeypatch):
    adapter, db, _runtime, _retired, main_home = runtime_adapter
    main = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(main, reusable=True)

    coder_home = main_home.parent / "coder"
    coder_home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: coder_home)
    profile_token = _api_request_profile.set("coder")
    try:
        coder = _interactive_create(
            adapter,
            session_id="coder-session",
            gateway_session_key="/profiles/coder|coder-session",
        )
        db.message_counts["coder-session"] = 2
        adapter._finish_runtime_shell_turn(coder, reusable=True)
    finally:
        _api_request_profile.reset(profile_token)

    assert adapter._close_runtime_shells_for_profile(main_home) == 1
    assert main.closed == 1
    assert main.end_session_calls == 0
    assert main._end_session_on_close is False
    assert coder.closed == 0
    assert adapter._runtime_shell_cache.counts()["entries"] == 1


def test_shutdown_cleanup_closes_shell_without_ending_session(runtime_adapter):
    adapter, db, _runtime, _retired, _home = runtime_adapter
    agent = _interactive_create(adapter)
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(agent, reusable=True)

    assert adapter._stop_runtime_shell_cache() == 1
    assert agent.closed == 1
    assert agent.end_session_calls == 0
    assert agent._end_session_on_close is False
