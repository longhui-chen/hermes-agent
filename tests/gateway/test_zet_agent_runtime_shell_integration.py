"""Zet adapter integration contracts for ordinary runtime-shell reuse."""

import asyncio
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

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
    set_zettlab_connector_route_capability,
    zettlab_connector_route_capability,
)
from tools import terminal_tool as terminal_tool_module


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
        self.api_mode = kwargs.get("api_mode", "chat_completions")
        self._session_db = kwargs.get("session_db")
        self._gateway_session_key = kwargs.get("gateway_session_key")
        self._profile_name = kwargs.get("profile_name")
        self._interrupt_requested = False
        self._persist_disabled = False
        self._last_flushed_db_idx = 0
        self._end_session_on_close = True
        self._cached_system_prompt = None
        self._cached_system_prompt_static = None
        self.prompt_invalidations = 0
        self.end_session_calls = 0
        self.released = 0
        self.closed = 0
        self.request_clients_created = 0
        self.request_client_close_reasons = []
        self.__class__.constructed.append(self)

    def release_clients(self):
        self.released += 1

    def close(self):
        self.closed += 1
        if self._end_session_on_close:
            self.end_session_calls += 1

    def _invalidate_system_prompt(self):
        self.prompt_invalidations += 1
        self._cached_system_prompt = None
        self._cached_system_prompt_static = None

    def _create_request_openai_client(self, *, reason, api_kwargs=None):
        self.request_clients_created += 1
        return object()

    def _close_request_openai_client(self, _client, *, reason):
        self.request_client_close_reasons.append(reason)

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
    ephemeral_system_prompt=None,
    request_overrides=None,
    requested_model=None,
    model_options=None,
    confirmed_runtime_lock=False,
):
    return adapter._create_agent(
        session_id=session_id,
        gateway_session_key=gateway_session_key,
        ephemeral_system_prompt=ephemeral_system_prompt,
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


def _http_app(adapter: ZetAgentAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


def _write_connector_runtime(tmp_path: Path) -> None:
    script = (
        tmp_path
        / "presets"
        / "skills"
        / "github"
        / "scripts"
        / "connector_runtime.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        "import json, os\n"
        "print(json.dumps({'route': os.environ.get('HERMES_SESSION_KEY', '')}))\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_zet_http_connector_turn_recovers_lost_tool_context_without_cross_turn_leak(
    runtime_adapter,
    monkeypatch,
    tmp_path,
):
    """Exercise HTTP -> ZetAgent -> real dispatch -> trusted runner.

    The test intentionally clears the ContextVar immediately before the real
    tool-dispatch middleware runs.  A regression in either ZetAgent's private
    per-turn binding or tool_executor's short-lived recovery therefore makes
    the connector runner miss ``HERMES_SESSION_KEY``.  A second request proves
    the cached runtime shell is rebound rather than borrowing the first turn.
    """
    from agent import relay_tools
    from agent.tool_executor import _run_agent_tool_execution_middleware
    from hermes_cli import middleware as hermes_middleware

    adapter, _db, _runtime, _retired, _home = runtime_adapter
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
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
    # Keep the test at the production dispatch boundary while making plugin
    # policy/relay dependencies deterministic; the runner itself is real.
    monkeypatch.setattr(
        relay_tools,
        "execute",
        lambda _name, args, dispatch, **_kwargs: (dispatch(args), args),
    )
    monkeypatch.setattr(
        hermes_middleware,
        "apply_tool_request_middleware",
        lambda _name, args, **_kwargs: SimpleNamespace(payload=args, trace=[]),
    )
    monkeypatch.setattr(
        hermes_middleware,
        "run_tool_execution_middleware",
        lambda _name, args, dispatch, **_kwargs: dispatch(args),
    )

    def _connector_turn(self, **_kwargs):
        self._tool_guardrails = SimpleNamespace(
            before_call=lambda *_args, **_kwargs: SimpleNamespace(
                allows_execution=True
            )
        )
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        # Model the observed defect: a real runtime boundary reaches dispatch
        # after the request ContextVar was dropped.
        set_zettlab_connector_route_capability("")
        outcome = _run_agent_tool_execution_middleware(
            self,
            function_name="terminal",
            function_args={},
            effective_task_id="connector-route-flow",
            tool_call_id="connector-route-call",
            execute=lambda _args: terminal_tool_module.terminal_tool(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/github/scripts/'
                'connector_runtime.py" list-tools',
                task_id="connector-route-flow",
            ),
        )
        self.session_prompt_tokens = 1
        self.session_completion_tokens = 1
        self.session_total_tokens = 2
        return {"final_response": outcome.result, "completed": True, "failed": False}

    monkeypatch.setattr(_FakeAgent, "run_conversation", _connector_turn)

    async with TestClient(TestServer(_http_app(adapter))) as client:
        async def _post(capability: str) -> dict:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer test-key"},
                json={
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "list repos"}],
                    "metadata": {"connector_route_capability": capability},
                },
            )
            assert response.status == 200
            body = await response.json()
            result = json.loads(body["choices"][0]["message"]["content"])
            assert result["connector_runtime_direct"] is True
            assert result["exit_code"] == 0
            return json.loads(result["output"])

        route_a = "A" * 43
        route_b = "B" * 43
        assert await _post(route_a) == {"route": route_a}
        assert await _post(route_b) == {"route": route_b}

    assert zettlab_connector_route_capability() == ""


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
            ephemeral_system_prompt="LS system prompt",
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
    assert warmed.request_clients_created == 1
    assert warmed.request_client_close_reasons == ["request_complete"]
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 1,
        "idle": 1,
        "leased": 0,
    }


def test_session_prewarm_uses_effective_context_length_for_config_signature(
    runtime_adapter,
    monkeypatch,
):
    adapter, _db, runtime, retired, _home = runtime_adapter
    runtime["config_context_length"] = 200_000
    configs = iter((
        {},
        {"model": {"context_length": 200_000}},
        {"model": {"context_length": 200_000}},
    ))
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: next(configs),
    )
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._prewarm_runtime_shell_sync("session-1") is True
        warmed = _FakeAgent.constructed[0]
        same_identity = _interactive_create(
            adapter,
            session_id="session-1",
            gateway_session_key="session-1",
        )
        assert same_identity is warmed
        adapter._finish_runtime_shell_turn(same_identity, reusable=True)

        runtime["config_context_length"] = 256_000
        changed_runtime = _interactive_create(
            adapter,
            session_id="session-1",
            gateway_session_key="session-1",
        )
        assert changed_runtime is not warmed
        adapter._finish_runtime_shell_turn(changed_runtime, reusable=True)
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert len(_FakeAgent.constructed) == 2
    assert retired == [warmed]


def test_request_client_prewarm_skips_non_openai_and_moa_runtimes(
    runtime_adapter,
):
    adapter, _db, _runtime, _retired, _home = runtime_adapter

    anthropic = _FakeAgent(api_mode="anthropic_messages", provider="anthropic")
    moa = _FakeAgent(api_mode="chat_completions", provider="moa")

    assert adapter._prewarm_runtime_shell_request_client(anthropic) is False
    assert adapter._prewarm_runtime_shell_request_client(moa) is False
    assert anthropic.request_clients_created == 0
    assert moa.request_clients_created == 0


def test_request_client_prewarm_failure_retires_the_partial_client(
    runtime_adapter,
):
    adapter, _db, _runtime, _retired, _home = runtime_adapter
    agent = _FakeAgent(api_mode="chat_completions", provider="custom")
    release_calls = []

    def fail_first_release(_client, *, reason):
        release_calls.append(reason)
        if reason == "request_complete":
            raise RuntimeError("cannot publish warm client")

    agent._close_request_openai_client = fail_first_release

    assert adapter._prewarm_runtime_shell_request_client(agent) is False
    assert release_calls == ["request_complete", "request_error_cleanup"]


def test_runtime_shell_prompt_rebind_invalidates_built_prompt_without_reconstruction(
    runtime_adapter,
):
    adapter, db, _runtime, retired, _home = runtime_adapter
    first = _interactive_create(
        adapter,
        ephemeral_system_prompt="prompt-a",
    )
    first._cached_system_prompt = "built prompt a"
    first._cached_system_prompt_static = "static prompt a"
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    second = _interactive_create(
        adapter,
        ephemeral_system_prompt="prompt-b",
    )

    assert second is first
    assert len(_FakeAgent.constructed) == 1
    assert second.ephemeral_system_prompt.endswith("prompt-b")
    assert "prompt-a" not in second.ephemeral_system_prompt
    assert second._cached_system_prompt is None
    assert second._cached_system_prompt_static is None
    assert second.prompt_invalidations == 1
    assert retired == []


def test_runtime_shell_prompt_rebind_failure_retires_and_rebuilds(
    runtime_adapter,
    monkeypatch,
):
    adapter, db, _runtime, retired, _home = runtime_adapter
    first = _interactive_create(
        adapter,
        ephemeral_system_prompt="prompt-a",
    )
    first._cached_system_prompt = "built prompt a"
    db.message_counts["session-1"] = 2
    adapter._finish_runtime_shell_turn(first, reusable=True)

    def fail_invalidation():
        raise RuntimeError("cannot invalidate")

    monkeypatch.setattr(first, "_invalidate_system_prompt", fail_invalidation)
    second = _interactive_create(
        adapter,
        ephemeral_system_prompt="prompt-b",
    )

    assert second is not first
    assert len(_FakeAgent.constructed) == 2
    assert retired == [first]


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


@pytest.mark.asyncio
async def test_public_run_agent_hits_exact_prewarm_with_real_prompt_and_context_handoff(
    runtime_adapter,
    monkeypatch,
):
    adapter, db, _runtime, retired, profile_home = runtime_adapter
    monkeypatch.setattr(adapter, "_effective_model", lambda *_args: "")
    monkeypatch.setattr(
        "tools.zettlab_snapshot_guard.finish_turn", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "agent.agent_runtime_helpers.collect_answer_attribution_citations",
        lambda *_args, **_kwargs: None,
    )
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert await adapter._prewarm_runtime_shell(
            "session-1",
            str(profile_home),
        ) is True
        result, _usage = await adapter._run_agent(
            user_message="[ZETTLAB:test] first",
            conversation_history=[],
            ephemeral_system_prompt="LS system prompt",
            session_id="session-1",
            gateway_session_key="session-1",
            turn_id="turn-1",
        )
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert result["final_response"] == "ok"
    assert len(_FakeAgent.constructed) == 1
    assert db.message_counts["session-1"] == 2
    assert retired == []
    assert adapter._runtime_shell_cache.counts() == {
        "entries": 1,
        "idle": 1,
        "leased": 0,
    }


@pytest.mark.asyncio
async def test_async_session_prewarm_passes_authoritative_profile_home_to_worker(
    runtime_adapter,
    monkeypatch,
):
    adapter, _db, _runtime, _retired, profile_home = runtime_adapter
    calls = []

    def fake_sync(session_id: str, profile_home_key: str) -> bool:
        calls.append((session_id, profile_home_key))
        return True

    monkeypatch.setattr(adapter, "_prewarm_runtime_shell_sync", fake_sync)

    assert await adapter._prewarm_runtime_shell(
        "session-1",
        str(profile_home),
    ) is True
    assert calls == [("session-1", str(profile_home))]


def test_sync_session_prewarm_reenters_authoritative_profile_runtime_scope(
    runtime_adapter,
    monkeypatch,
):
    adapter, _db, _runtime, _retired, profile_home = runtime_adapter
    entered = []

    @contextmanager
    def fake_profile_scope(home):
        entered.append(Path(home))
        yield

    monkeypatch.setattr("gateway.run._profile_runtime_scope", fake_profile_scope)
    account_token = _zettlab_request_account_id.set("account-a")
    principal_token = push_zettlab_auth_principal("iam:cn:user:account-a")
    deep_principal_token = _deep_memory_principal.set("iam:cn:user:account-a")
    deep_subject_token = _deep_memory_subject.set("account-a")
    try:
        assert adapter._prewarm_runtime_shell_sync(
            "session-1",
            str(profile_home),
        ) is True
    finally:
        _deep_memory_subject.reset(deep_subject_token)
        _deep_memory_principal.reset(deep_principal_token)
        pop_zettlab_auth_principal(principal_token)
        _zettlab_request_account_id.reset(account_token)

    assert entered == [profile_home]


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

    async def fake_prewarm(session_id: str, _profile_home_key: str) -> bool:
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

    async def failing_prewarm(
        _session_id: str,
        _profile_home_key: str,
    ) -> bool:
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


def test_runtime_shell_lookup_log_contains_only_bounded_diagnostics(
    runtime_adapter,
    caplog,
):
    adapter, _db, _runtime, _retired, _home = runtime_adapter

    with caplog.at_level("INFO", logger="gateway.platforms.zet_agent"):
        _interactive_create(adapter)

    message = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(
            "zet_agent runtime shell cache lookup:"
        )
    )
    assert "result=reserved acquire=miss" in message
    assert "message_count=0" in message
    for field in (
        "signature",
        "model_fp",
        "runtime_fp",
        "toolsets_fp",
        "config_fp",
        "fallback_fp",
    ):
        value = message.split(f"{field}=", 1)[1].split(" ", 1)[0]
        assert len(value) == 12
        assert all(char in "0123456789abcdef" for char in value)
    cache_keys = message.split("cache_keys=", 1)[1].split(" ", 1)[0]
    assert "model.context_length:" in cache_keys
    assert "compression.enabled:" in cache_keys
    assert "memory.deep_memory_mode:" in cache_keys
    assert "tools.registry_generation" not in cache_keys
    assert "credential-a" not in message
    assert "https://example.invalid" not in message


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
