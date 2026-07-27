"""Async-delegation delivery to local-server (delegation-app-foundation).

Covers the zet_agent side of ``delegate_task(background=true)``:

- ``supports_async_delivery`` is gated on ``ZET_DELEGATION_ADVANCE_URL`` so
  rollout order is safe (new hermes + old local-server keeps today's
  synchronous fallback);
- ``handle_message`` diverts internal async-delegation completions to the
  loopback POST instead of forging an internal turn whose output the
  api_server family cannot deliver;
- delivery failures raise — the watcher's retry signal (durable claim is
  released, event redelivered later);
- session interrupt also cancels the session's background delegations
  (``agent.interrupt()`` cannot reach them: background dispatch detaches
  children from ``_active_children``).
"""

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter, _delegation_advance_url

_ADVANCE_ENV = "ZET_DELEGATION_ADVANCE_URL"
_ADVANCE_URL = "http://127.0.0.1:9091/api/v1/internal/delegation/advance"


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


class _FakeRequest:
    def __init__(self, body, match_info=None, auth="Bearer test-key"):
        self._body = body
        self.headers = {"Authorization": auth} if auth else {}
        self.match_info = match_info or {}
        self.method = "POST"
        self.path_qs = "/v1/sessions/test/interrupt"
        self.remote = "127.0.0.1"
        self.transport = None
        self.can_read_body = body is not None

    def get(self, key, default=None):
        # aiohttp Request is a MutableMapping; profile routes stamp
        # hermes_profile_home on it. Legacy /v1 routes have no stamp.
        return default

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeTask:
    def __init__(self, done=False):
        self._done = done
        self.cancelled = False

    def done(self):
        return self._done

    def cancel(self):
        self.cancelled = True


class _Event:
    def __init__(self, internal=True, metadata=None, text="synth"):
        self.internal = internal
        self.metadata = metadata
        self.text = text


def _adapter(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    return ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))


def _delegation_process_event():
    return {
        "type": "async_delegation",
        "delegation_id": "deleg_abc12345",
        "session_key": "agent:main:zet_agent:dm:zettlab:u1:agentA:1",
        "parent_session_id": "zettlab:u1:agentA:1",
        "status": "completed",
        "goal": "2 parallel subagents: research A; research B",
        "goals": ["research A", "research B"],
        "is_batch": True,
        "results": [
            {"task_index": 0, "status": "completed", "summary": "found A"},
            {"task_index": 1, "status": "completed", "summary": "found B"},
        ],
        "dispatched_at": 100.0,
        "completed_at": 200.0,
        "total_duration_seconds": 100.0,
    }


def _delegation_event():
    return _Event(
        metadata={"process_event": _delegation_process_event()},
        text="[ASYNC DELEGATION BATCH COMPLETE — deleg_abc12345] ...",
    )


# ---------------------------------------------------------------------------
# supports_async_delivery gating
# ---------------------------------------------------------------------------


def test_async_delivery_off_without_env(monkeypatch):
    monkeypatch.delenv(_ADVANCE_ENV, raising=False)
    adapter = _adapter(monkeypatch)
    assert adapter.supports_async_delivery is False


def test_async_delivery_on_with_env(monkeypatch):
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)
    assert adapter.supports_async_delivery is True


def test_advance_url_resolution_prefers_env(monkeypatch):
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    assert _delegation_advance_url() == _ADVANCE_URL
    monkeypatch.delenv(_ADVANCE_ENV, raising=False)
    assert _delegation_advance_url() == ""


# ---------------------------------------------------------------------------
# handle_message divert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_message_diverts_delegation_completion(monkeypatch):
    adapter = _adapter(monkeypatch)
    delivered = []
    super_called = []

    async def _fake_deliver(evt, synth_text):
        delivered.append((evt, synth_text))

    async def _fake_super(self, event):
        super_called.append(event)

    monkeypatch.setattr(adapter, "_deliver_delegation_completion", _fake_deliver)
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter.handle_message", _fake_super
    )

    event = _delegation_event()
    await adapter.handle_message(event)

    assert len(delivered) == 1
    assert delivered[0][0]["delegation_id"] == "deleg_abc12345"
    assert delivered[0][1] == event.text
    assert super_called == []


@pytest.mark.asyncio
async def test_handle_message_passes_through_non_delegation(monkeypatch):
    adapter = _adapter(monkeypatch)
    delivered = []
    super_called = []

    async def _fake_deliver(evt, synth_text):
        delivered.append(evt)

    async def _fake_super(self, event):
        super_called.append(event)

    monkeypatch.setattr(adapter, "_deliver_delegation_completion", _fake_deliver)
    monkeypatch.setattr(
        "gateway.platforms.api_server.APIServerAdapter.handle_message", _fake_super
    )

    # Internal but not an async-delegation completion (e.g. a watch-pattern
    # process notification) — must keep upstream behavior.
    await adapter.handle_message(
        _Event(metadata={"process_event": {"type": "completion"}})
    )
    # Non-internal message carrying a forged marker — must NOT divert.
    await adapter.handle_message(
        _Event(internal=False, metadata={"process_event": _delegation_process_event()})
    )
    # Internal with no metadata at all.
    await adapter.handle_message(_Event(metadata=None))

    assert delivered == []
    assert len(super_called) == 3


# ---------------------------------------------------------------------------
# _deliver_delegation_completion
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.asyncio
async def test_deliver_posts_structured_payload(monkeypatch):
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)
    seen = []

    def _fake_urlopen(req, timeout=None):
        seen.append((req, timeout))
        return _FakeHTTPResponse(200)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    await adapter._deliver_delegation_completion(
        _delegation_process_event(), "[ASYNC DELEGATION BATCH COMPLETE — deleg_abc12345]"
    )

    assert len(seen) == 1
    req, timeout = seen[0]
    assert req.full_url == _ADVANCE_URL
    assert timeout == 10
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["schema"] == 1
    assert payload["kind"] == "delegation"
    assert payload["delegation_id"] == "deleg_abc12345"
    assert payload["session_key"] == "agent:main:zet_agent:dm:zettlab:u1:agentA:1"
    assert payload["session_id"] == "zettlab:u1:agentA:1"
    assert payload["is_batch"] is True
    assert len(payload["results"]) == 2
    assert payload["duration_seconds"] == 100.0
    assert payload["synth_text"].startswith("[ASYNC DELEGATION BATCH COMPLETE")


@pytest.mark.asyncio
async def test_deliver_raises_on_http_error_status(monkeypatch):
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: _FakeHTTPResponse(503)
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await adapter._deliver_delegation_completion(_delegation_process_event(), "x")


@pytest.mark.asyncio
async def test_deliver_dead_letters_on_permanent_4xx(monkeypatch):
    """404/400-class rejections can never succeed on retry: the delivery
    returns normally (→ watcher acks the durable row) instead of feeding the
    2s retry loop forever — across restarts too."""
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)

    def _gone(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "session deleted", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", _gone)
    # Must NOT raise: permanent rejection is dead-lettered (logged + acked).
    await adapter._deliver_delegation_completion(_delegation_process_event(), "x")


@pytest.mark.asyncio
async def test_deliver_retries_transient_http_errors(monkeypatch):
    """5xx / 408 / 429 stay retryable — local-server may just be restarting."""
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)
    for code in (503, 429, 408):
        def _busy(req, timeout=None, _code=code):
            raise urllib.error.HTTPError(req.full_url, _code, "busy", None, None)

        monkeypatch.setattr(urllib.request, "urlopen", _busy)
        with pytest.raises(RuntimeError, match=f"HTTP {code}"):
            await adapter._deliver_delegation_completion(_delegation_process_event(), "x")


@pytest.mark.asyncio
async def test_deliver_raises_on_network_error(monkeypatch):
    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)

    def _boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(RuntimeError, match="delivery failed"):
        await adapter._deliver_delegation_completion(_delegation_process_event(), "x")


@pytest.mark.asyncio
async def test_deliver_raises_without_url(monkeypatch):
    monkeypatch.delenv(_ADVANCE_ENV, raising=False)
    adapter = _adapter(monkeypatch)
    with pytest.raises(RuntimeError, match="ZET_DELEGATION_ADVANCE_URL unset"):
        await adapter._deliver_delegation_completion(_delegation_process_event(), "x")


# ---------------------------------------------------------------------------
# session interrupt → background delegation cancel
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, session_id="s1-rotated"):
        self.session_id = session_id
        self.interrupted = []

    def interrupt(self, reason):
        self.interrupted.append(reason)


@pytest.mark.asyncio
async def test_session_interrupt_cancels_async_delegations(monkeypatch):
    import tools.async_delegation as async_delegation

    adapter = _adapter(monkeypatch)
    agent = _FakeAgent(session_id="s1-rotated")
    adapter._active_session_agents["s1"] = [agent]
    adapter._active_session_tasks["s1"] = _FakeTask()

    calls = []

    def _fake_interrupt_for_session(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        async_delegation, "interrupt_for_session", _fake_interrupt_for_session
    )

    resp = await adapter._handle_session_interrupt(
        _FakeRequest(None, match_info={"session_id": "s1"})
    )

    assert resp.status == 200
    assert resp.payload["status"] == "stopping"
    assert agent.interrupted  # foreground turn interrupted as before
    # Background delegations interrupted under BOTH the URL session id and
    # the (possibly compaction-rotated) live agent session id.
    seen_sids = {c.get("parent_session_id") for c in calls}
    assert seen_sids == {"s1", "s1-rotated"}
    for c in calls:
        assert c.get("reason") == "user_cancel"
        # User-stop kills suppress the completion turn: the interrupted turn
        # itself carries the outcome card (local-server terminal hook), so the
        # killed batch must not re-enter the chat afterwards.
        assert c.get("suppress_completion") is True


@pytest.mark.asyncio
async def test_session_interrupt_survives_delegation_cancel_failure(monkeypatch):
    import tools.async_delegation as async_delegation

    adapter = _adapter(monkeypatch)
    agent = _FakeAgent()
    adapter._active_session_agents["s1"] = [agent]
    adapter._active_session_tasks["s1"] = _FakeTask()

    def _boom(**kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(async_delegation, "interrupt_for_session", _boom)

    resp = await adapter._handle_session_interrupt(
        _FakeRequest(None, match_info={"session_id": "s1"})
    )

    # Delegation-cancel failure must never break the primary interrupt path.
    assert resp.status == 200
    assert resp.payload["status"] == "stopping"
    assert agent.interrupted


# ---------------------------------------------------------------------------
# Synthetic-event source resolution (run.py adapter hook + zet_agent resolver)
#
# Real zet_agent turns bind the local-server session id verbatim as the
# gateway session_key (no "agent:main:<platform>:…" shape), so run.py's
# generic parse fails and — before the resolver hook — async delegation
# completions were dropped as "Synthetic event source unresolvable".
# ---------------------------------------------------------------------------


class _FakeSessionDB:
    def __init__(self, known):
        self._known = set(known)

    def get_session(self, session_id):
        return {"session_id": session_id} if session_id in self._known else None


def test_resolver_claims_own_session(monkeypatch):
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        adapter, "_ensure_session_db", lambda: _FakeSessionDB({"zettlab:u1:agentA:1"})
    )
    src = adapter.resolve_process_event_source("zettlab:u1:agentA:1")
    assert src is not None
    assert src.chat_id == "zettlab:u1:agentA:1"
    assert src.platform.value == "zet_agent"
    assert src.chat_type == "dm"


def test_resolver_fail_closed(monkeypatch):
    adapter = _adapter(monkeypatch)
    # 不认识的会话 / 空 key 不认领；DB 探测异常改抛 transient（见下组测试）
    # —— 折进 None 会把可重试的完成事件当「无路由」丢弃。
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: _FakeSessionDB(set()))
    assert adapter.resolve_process_event_source("zettlab:u1:agentA:1") is None
    assert adapter.resolve_process_event_source("") is None


def _fake_runner(adapter):
    from types import SimpleNamespace

    from gateway.config import Platform

    return SimpleNamespace(
        session_store=SimpleNamespace(_ensure_loaded=lambda: None, _entries={}),
        _get_cached_session_source=lambda key: None,
        adapters={Platform.ZET_AGENT: adapter},
    )


def test_build_process_event_source_falls_back_to_adapter_resolver(monkeypatch):
    """Flow test: raw zet_agent session_key → adapter resolver claims it."""
    from gateway.run import GatewayRunner

    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        adapter, "_ensure_session_db", lambda: _FakeSessionDB({"zettlab:u1:agentA:1"})
    )
    runner = _fake_runner(adapter)
    evt = {"type": "async_delegation", "session_key": "zettlab:u1:agentA:1"}
    src = GatewayRunner._build_process_event_source(runner, evt)
    assert src is not None
    assert src.platform.value == "zet_agent"
    assert src.chat_id == "zettlab:u1:agentA:1"


def test_build_process_event_source_still_unresolvable_for_foreign_keys(monkeypatch):
    """Foreign / unknown keys keep the fail-closed None（不误认领）。"""
    from gateway.run import GatewayRunner

    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: _FakeSessionDB(set()))
    runner = _fake_runner(adapter)
    evt = {"type": "async_delegation", "session_key": "zettlab:u1:agentA:1"}
    assert GatewayRunner._build_process_event_source(runner, evt) is None


# ---------------------------------------------------------------------------
# Transient vs definitive route resolution (completion delivery retry)
# ---------------------------------------------------------------------------


def test_resolver_returns_none_for_unknown_session(monkeypatch):
    """Definitive "not ours" stays None — foreign keys remain unroutable."""
    adapter = _adapter(monkeypatch)

    class _DB:
        @staticmethod
        def get_session(key):
            return None

    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: _DB())
    assert adapter.resolve_process_event_source("zettlab:u1:agentA:1") is None


def test_resolver_raises_transient_on_probe_error(monkeypatch):
    """SQLite busy / unreadable DB must NOT fold into "no route": the event
    would leave the in-memory queue while its durable row is never rescanned
    in this process — the user waits for a gateway restart. Raising the
    transient marker lets delivery loops requeue and retry."""
    import sqlite3

    from gateway.run import TransientRouteResolutionError

    adapter = _adapter(monkeypatch)

    def _boom():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(adapter, "_ensure_session_db", _boom)
    with pytest.raises(TransientRouteResolutionError):
        adapter.resolve_process_event_source("zettlab:u1:agentA:1")


def test_build_process_event_source_propagates_transient(monkeypatch):
    """GatewayRunner._build_process_event_source re-raises the transient
    marker instead of swallowing it into the generic resolver except."""
    from gateway.run import GatewayRunner, TransientRouteResolutionError

    runner = object.__new__(GatewayRunner)

    class _TransientResolver:
        @staticmethod
        def resolve_process_event_source(session_key):
            raise TransientRouteResolutionError("probe failed")

    runner.adapters = {object(): _TransientResolver()}
    evt = {"type": "async_delegation", "session_key": "zettlab:u1:agentA:1"}
    with pytest.raises(TransientRouteResolutionError):
        runner._build_process_event_source(evt)


# ---------------------------------------------------------------------------
# Turn rebind carries the async-delivery capability (env-gated)
# ---------------------------------------------------------------------------


def _with_api_server_binding():
    """Simulate _bind_api_server_session's False that the rebind overwrites."""
    from gateway.session_context import set_session_vars

    return set_session_vars(
        platform="api_server",
        chat_id="s1",
        session_key="s1",
        session_id="s1",
        async_delivery=False,
    )


def test_turn_rebind_enables_async_delivery_with_env(monkeypatch):
    from gateway.session_context import async_delivery_supported, clear_session_vars

    monkeypatch.setenv(_ADVANCE_ENV, _ADVANCE_URL)
    adapter = _adapter(monkeypatch)
    tokens = _with_api_server_binding()
    try:
        adapter._bind_turn_session_context("zettlab:u1:agentA:1")
        assert async_delivery_supported() is True
    finally:
        clear_session_vars(tokens)


def test_turn_rebind_keeps_async_delivery_off_without_env(monkeypatch):
    """Without ZET_DELEGATION_ADVANCE_URL the rebind must NOT flip the
    contextvar back to default-True: delegate_task would promise background
    delivery nobody can fulfil (#10760-style silent no-op)."""
    from gateway.session_context import async_delivery_supported, clear_session_vars

    monkeypatch.delenv(_ADVANCE_ENV, raising=False)
    adapter = _adapter(monkeypatch)
    tokens = _with_api_server_binding()
    try:
        adapter._bind_turn_session_context("zettlab:u1:agentA:1")
        assert async_delivery_supported() is False
    finally:
        clear_session_vars(tokens)
