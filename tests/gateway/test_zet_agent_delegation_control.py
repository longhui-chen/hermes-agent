"""Delegation control-plane routes on zet_agent (status / cancel / interrupt).

Thin wrappers over the same registries the TUI /agents overlay reads —
these tests pin auth, wiring, and found/not-found semantics.
"""

import pytest

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


class _FakeRequest:
    def __init__(self, match_info=None, auth="Bearer test-key", scope=None):
        self.headers = {"Authorization": auth} if auth else {}
        self.match_info = match_info or {}
        self.method = "GET"
        self.path_qs = "/v1/delegations/status"
        self.remote = "127.0.0.1"
        self.transport = None
        self.can_read_body = False
        # aiohttp Request is a MutableMapping; _profile_handler stamps
        # hermes_profile_home on it and the control plane reads it via .get.
        self._scope = scope or {}

    def get(self, key, default=None):
        return self._scope.get(key, default)


def _adapter(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    return ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))


@pytest.mark.asyncio
async def test_status_returns_both_registries(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.delegate_tool as delegate_tool

    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        delegate_tool, "list_active_subagents",
        lambda profile_home="": [{"subagent_id": "sa_1", "goal": "g"}],
    )
    monkeypatch.setattr(
        async_delegation, "list_async_delegations",
        lambda profile_home="": [{"delegation_id": "deleg_x", "status": "running"}],
    )
    resp = await adapter._handle_delegations_status(_FakeRequest())
    assert resp.status == 200
    assert resp.payload["active"][0]["subagent_id"] == "sa_1"
    assert resp.payload["async"][0]["delegation_id"] == "deleg_x"


@pytest.mark.asyncio
async def test_status_scopes_to_route_profile(monkeypatch):
    """/p/{profile} routes pass their profile home down to both registries —
    one agent's goals/session keys must never leak into another's route."""
    import tools.async_delegation as async_delegation
    import tools.delegate_tool as delegate_tool

    adapter = _adapter(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        delegate_tool, "list_active_subagents",
        lambda profile_home="": seen.setdefault("active", profile_home) and [] or [],
    )
    monkeypatch.setattr(
        async_delegation, "list_async_delegations",
        lambda profile_home="": seen.setdefault("async", profile_home) and [] or [],
    )
    resp = await adapter._handle_delegations_status(
        _FakeRequest(scope={"hermes_profile_home": "/hh/profiles/agent-a"})
    )
    assert resp.status == 200
    assert seen == {
        "active": "/hh/profiles/agent-a",
        "async": "/hh/profiles/agent-a",
    }


@pytest.mark.asyncio
async def test_cancel_found_and_not_found(monkeypatch):
    import tools.async_delegation as async_delegation

    adapter = _adapter(monkeypatch)
    calls = []
    monkeypatch.setattr(
        async_delegation, "interrupt_delegation",
        lambda deleg_id, reason="user_cancel", profile_home="":
            calls.append(deleg_id) or deleg_id == "deleg_hit",
    )
    resp = await adapter._handle_delegation_cancel(
        _FakeRequest(match_info={"delegation_id": "deleg_hit"})
    )
    assert resp.status == 200 and resp.payload["interrupted"] is True
    resp = await adapter._handle_delegation_cancel(
        _FakeRequest(match_info={"delegation_id": "deleg_miss"})
    )
    assert resp.status == 404 and resp.payload["interrupted"] is False
    assert calls == ["deleg_hit", "deleg_miss"]


@pytest.mark.asyncio
async def test_subagent_interrupt_found_and_not_found(monkeypatch):
    import tools.delegate_tool as delegate_tool

    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(
        delegate_tool, "interrupt_subagent",
        lambda sid, profile_home="": sid == "sa_hit",
    )
    resp = await adapter._handle_subagent_interrupt(
        _FakeRequest(match_info={"subagent_id": "sa_hit"})
    )
    assert resp.status == 200 and resp.payload["interrupted"] is True
    resp = await adapter._handle_subagent_interrupt(
        _FakeRequest(match_info={"subagent_id": "sa_miss"})
    )
    assert resp.status == 404


@pytest.mark.asyncio
async def test_routes_reject_bad_auth(monkeypatch):
    adapter = _adapter(monkeypatch)
    for handler, mi in (
        (adapter._handle_delegations_status, {}),
        (adapter._handle_delegation_cancel, {"delegation_id": "deleg_x"}),
        (adapter._handle_subagent_interrupt, {"subagent_id": "sa_x"}),
    ):
        resp = await handler(_FakeRequest(match_info=mi, auth="Bearer wrong"))
        assert resp.status in (401, 403)


def test_interrupt_delegation_helper_contract():
    import tools.async_delegation as async_delegation

    # 直接注入一条 running 记录（unit 层面验证锁/状态门/回调调用）。
    called = []
    with async_delegation._records_lock:
        async_delegation._records["deleg_unit"] = {
            "delegation_id": "deleg_unit",
            "status": "running",
            "interrupt_fn": lambda: called.append(True),
        }
    try:
        assert async_delegation.interrupt_delegation("deleg_unit") is True
        assert called == [True]
        # 非 running / 未知 id / 空 id 都拒绝
        with async_delegation._records_lock:
            async_delegation._records["deleg_unit"]["status"] = "completed"
        assert async_delegation.interrupt_delegation("deleg_unit") is False
        assert async_delegation.interrupt_delegation("deleg_nope") is False
        assert async_delegation.interrupt_delegation("") is False
    finally:
        with async_delegation._records_lock:
            async_delegation._records.pop("deleg_unit", None)
