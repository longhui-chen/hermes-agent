"""Tests for ZetAgentAdapter prompt-class reload endpoints (ZET-1139).

Covers ``POST /v1/profile/reload`` (new in ZET-1139) and the ZET-1139
additions to ``POST /v1/skills/reload`` (in-process invalidate + DB clear).
"""
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeRequest:
    def __init__(self, auth="Bearer test-key"):
        self.headers = {"Authorization": auth} if auth else {}
        # Upstream's api_server _request_audit_context reads these directly
        # when logging auth-rejection audit context.
        self.method = "POST"
        self.path_qs = "/v1/profile/reload"


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


def _make_adapter(monkeypatch, *, gateway_runner=None):
    # _check_auth lives on the api_server base class, so patch its `web`
    # too — otherwise the 401 path raises AttributeError under our stub.
    import gateway.platforms.api_server as api_server
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    monkeypatch.setattr(api_server, "web", _FakeWeb)
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    if gateway_runner is not None:
        adapter.gateway_runner = gateway_runner
    return adapter


def _make_runner(*, invalidate_returns=3, db_clears=7,
                 invalidate_raises=False, db_raises=False,
                 no_db=False):
    runner = MagicMock()
    if no_db:
        runner._session_db = None
    else:
        runner._session_db = MagicMock()
        if db_raises:
            runner._session_db.clear_all_system_prompts.side_effect = (
                RuntimeError("db boom")
            )
        else:
            runner._session_db.clear_all_system_prompts.return_value = db_clears
    if invalidate_raises:
        runner.invalidate_all_cached_agents.side_effect = RuntimeError(
            "evict boom"
        )
    else:
        runner.invalidate_all_cached_agents.return_value = invalidate_returns
    return runner


class _StubLock:
    """Threading-lock stub usable as a context manager — the real
    invalidate_all_cached_agents uses ``with self._agent_cache_lock:``."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _make_runner_with_real_cache(*, agents):
    """Build a GatewayRunner-style stub with a populated _agent_cache that
    mirrors the production tuple shape ``(agent, signature)``. Used to
    exercise the real invalidate_all_cached_agents loop (the simpler
    MagicMock runner above stops at "did it get called", not "does it
    correctly walk the cache value shape")."""
    from collections import OrderedDict
    runner = MagicMock()
    runner._agent_cache = OrderedDict(
        (f"sess-{i}", (a, "sig-stub")) for i, a in enumerate(agents)
    )
    runner._agent_cache_lock = _StubLock()
    runner._session_db = MagicMock()
    runner._session_db.clear_all_system_prompts.return_value = 0
    return runner


# =========================================================================
# /v1/profile/reload
# =========================================================================


@pytest.mark.asyncio
async def test_profile_reload_happy_path(monkeypatch):
    runner = _make_runner(invalidate_returns=4, db_clears=12)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload == {
        "reloaded": True,
        "invalidated_sessions": 4,
        "db_rows_cleared": 12,
    }
    runner._session_db.clear_all_system_prompts.assert_called_once_with()
    runner.invalidate_all_cached_agents.assert_called_once_with()


@pytest.mark.asyncio
async def test_profile_reload_clears_db_before_invalidate(monkeypatch):
    """DB clear must run before invalidate — otherwise the invalidated
    in-process prompt would just be reloaded from the still-stale DB
    on the next turn."""
    runner = _make_runner()
    call_order = []
    runner._session_db.clear_all_system_prompts.side_effect = (
        lambda: call_order.append("db") or 0
    )
    runner.invalidate_all_cached_agents.side_effect = (
        lambda: call_order.append("invalidate") or 0
    )

    adapter = _make_adapter(monkeypatch, gateway_runner=runner)
    await adapter._handle_profile_reload(_FakeRequest())

    assert call_order == ["db", "invalidate"]


@pytest.mark.asyncio
async def test_profile_reload_no_gateway_runner_returns_500(monkeypatch):
    adapter = _make_adapter(monkeypatch, gateway_runner=None)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 500


@pytest.mark.asyncio
async def test_profile_reload_db_failure_returns_500(monkeypatch):
    """A DB clear exception is CRITICAL — without DB clear the continuing
    session would replay the stale stored_prompt from SQLite, exactly the
    ZET-1139 bug this PR fixes. Returning 200 here would silently regress
    the fix and also short-circuit local-server's registry.Stop fallback
    (reload.Forward only checks HTTP status). So we return 500 instead.
    Reviewed-by: iwgyyyy on PR #77."""
    runner = _make_runner(invalidate_returns=2, db_raises=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 500
    # invalidate must NOT be called when DB clear failed — we want to
    # surface the failure cleanly, not partially apply.
    runner.invalidate_all_cached_agents.assert_not_called()


@pytest.mark.asyncio
async def test_profile_reload_invalidate_failure_is_failsoft(monkeypatch):
    """An invalidate exception is logged and swallowed; DB-clear result
    still surfaces."""
    runner = _make_runner(db_clears=5, invalidate_raises=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload["db_rows_cleared"] == 5
    assert resp.payload["invalidated_sessions"] == 0


@pytest.mark.asyncio
async def test_profile_reload_missing_session_db_returns_500(monkeypatch):
    """No SessionDB on the runner is treated as a critical configuration
    error (production gateway always has one) — same reasoning as the DB
    clear failure path: returning 200 would let local-server skip its
    registry.Stop fallback while the new SOUL.md is invisible to all
    existing sessions."""
    runner = _make_runner(invalidate_returns=1, no_db=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 500
    runner.invalidate_all_cached_agents.assert_not_called()


@pytest.mark.asyncio
async def test_profile_reload_rejects_missing_bearer(monkeypatch):
    runner = _make_runner()
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest(auth=None))

    assert resp.status == 401
    runner._session_db.clear_all_system_prompts.assert_not_called()
    runner.invalidate_all_cached_agents.assert_not_called()


# =========================================================================
# /v1/skills/reload — ZET-1139 additions
# =========================================================================


@pytest.mark.asyncio
async def test_skills_reload_also_invalidates_current_sessions(monkeypatch):
    """Post-ZET-1139, /v1/skills/reload also drives the in-process
    invalidate + DB clear hot-reload pair so skill changes land on the
    next turn of every session, not just new ones."""
    runner = _make_runner(invalidate_returns=2, db_clears=9)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    # Stub out the skill-cache module functions so the test stays
    # self-contained — the focus here is the ZET-1139 additions, not
    # the pre-existing skill-cache mechanics.
    import agent.prompt_builder as pb
    import agent.skill_commands as sc
    monkeypatch.setattr(pb, "clear_skills_system_prompt_cache",
                        lambda clear_snapshot=False: None)
    monkeypatch.setattr(sc, "scan_skill_commands", lambda: ["a", "b", "c"])

    resp = await adapter._handle_skills_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload == {
        "cleared": True,
        "skills_total": 3,
        "invalidated_sessions": 2,
        "db_rows_cleared": 9,
    }
    runner._session_db.clear_all_system_prompts.assert_called_once_with()
    runner.invalidate_all_cached_agents.assert_called_once_with()


# =========================================================================
# Real invalidate_all_cached_agents loop (against the production tuple
# value shape) — guards against the regression where the loop walked the
# raw tuple instead of unwrapping the agent and silently invalidated zero.
# =========================================================================


def test_invalidate_all_cached_agents_unwraps_tuple_values():
    """The real cache stores (agent, signature) tuples (gateway/run.py
    L15382). invalidate_all_cached_agents must unwrap to find the agent
    or it'll silently no-op even with a full cache — exactly the
    sim-01 production regression caught during e2e."""
    from gateway.run import GatewayRunner

    a1 = MagicMock()
    a2 = MagicMock()
    a3 = MagicMock()
    runner = _make_runner_with_real_cache(agents=[a1, a2, a3])

    count = GatewayRunner.invalidate_all_cached_agents(runner)

    assert count == 3, "expected 3 agents invalidated through the tuple values"
    a1._invalidate_system_prompt.assert_called_once_with()
    a2._invalidate_system_prompt.assert_called_once_with()
    a3._invalidate_system_prompt.assert_called_once_with()


def test_invalidate_all_cached_agents_tolerates_bare_agent_value():
    """Defensive: if a future refactor switches cache values from
    (agent, sig) tuple to bare agent, the loop still works."""
    from gateway.run import GatewayRunner
    from collections import OrderedDict

    a1 = MagicMock()
    runner = MagicMock()
    runner._agent_cache = OrderedDict([("sess-0", a1)])  # bare agent, no tuple
    runner._agent_cache_lock = _StubLock()

    count = GatewayRunner.invalidate_all_cached_agents(runner)

    assert count == 1
    a1._invalidate_system_prompt.assert_called_once_with()


def test_invalidate_all_cached_agents_skips_broken_agents():
    """One agent without _invalidate_system_prompt doesn't block others."""
    from gateway.run import GatewayRunner

    good = MagicMock()
    bad = object()  # plain object — no _invalidate_system_prompt
    raises = MagicMock()
    raises._invalidate_system_prompt.side_effect = RuntimeError("agent boom")

    runner = _make_runner_with_real_cache(agents=[good, bad, raises])
    count = GatewayRunner.invalidate_all_cached_agents(runner)

    assert count == 1  # only `good` counted
    good._invalidate_system_prompt.assert_called_once_with()


@pytest.mark.asyncio
async def test_skills_reload_invalidate_failure_does_not_fail_response(monkeypatch):
    """Skills reload's primary contract (clear skill cache + rescan + DB
    clear) keeps working even if the new invalidate-existing-sessions
    step blows up — the DB has already been cleared so the next turn
    rebuilds fresh anyway."""
    runner = _make_runner(invalidate_raises=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    import agent.prompt_builder as pb
    import agent.skill_commands as sc
    monkeypatch.setattr(pb, "clear_skills_system_prompt_cache",
                        lambda clear_snapshot=False: None)
    monkeypatch.setattr(sc, "scan_skill_commands", lambda: [])

    resp = await adapter._handle_skills_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload["cleared"] is True
    assert resp.payload["invalidated_sessions"] == 0


@pytest.mark.asyncio
async def test_skills_reload_db_failure_returns_500(monkeypatch):
    """DB clear failure on skills/reload is also critical — without it
    continuing sessions won't see new skills. Same semantics as
    profile/reload's DB failure path."""
    runner = _make_runner(db_raises=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    import agent.prompt_builder as pb
    import agent.skill_commands as sc
    monkeypatch.setattr(pb, "clear_skills_system_prompt_cache",
                        lambda clear_snapshot=False: None)
    monkeypatch.setattr(sc, "scan_skill_commands", lambda: ["a"])

    resp = await adapter._handle_skills_reload(_FakeRequest())

    assert resp.status == 500
    runner.invalidate_all_cached_agents.assert_not_called()


@pytest.mark.asyncio
async def test_skills_reload_missing_session_db_returns_500(monkeypatch):
    """No SessionDB → 500 (same reasoning as profile_reload)."""
    runner = _make_runner(no_db=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    import agent.prompt_builder as pb
    import agent.skill_commands as sc
    monkeypatch.setattr(pb, "clear_skills_system_prompt_cache",
                        lambda clear_snapshot=False: None)
    monkeypatch.setattr(sc, "scan_skill_commands", lambda: ["a"])

    resp = await adapter._handle_skills_reload(_FakeRequest())

    assert resp.status == 500
    runner.invalidate_all_cached_agents.assert_not_called()
