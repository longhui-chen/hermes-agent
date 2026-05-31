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
async def test_profile_reload_db_failure_is_failsoft(monkeypatch):
    """A DB clear exception is logged and swallowed — the endpoint still
    returns 200 so the in-process invalidate can still run."""
    runner = _make_runner(invalidate_returns=2, db_raises=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload["db_rows_cleared"] == 0
    assert resp.payload["invalidated_sessions"] == 2
    runner.invalidate_all_cached_agents.assert_called_once_with()


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
async def test_profile_reload_missing_session_db_skips_db_step(monkeypatch):
    """If the gateway has no SessionDB (test/mock paths), DB clear is
    skipped and only invalidate runs."""
    runner = _make_runner(invalidate_returns=1, no_db=True)
    adapter = _make_adapter(monkeypatch, gateway_runner=runner)

    resp = await adapter._handle_profile_reload(_FakeRequest())

    assert resp.status == 200
    assert resp.payload["db_rows_cleared"] == 0
    assert resp.payload["invalidated_sessions"] == 1


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


@pytest.mark.asyncio
async def test_skills_reload_invalidate_failure_does_not_fail_response(monkeypatch):
    """Skills reload's primary contract (clear skill cache + rescan) keeps
    working even if the new invalidate-existing-sessions step blows up."""
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
