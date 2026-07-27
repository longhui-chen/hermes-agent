"""Tests for the Chronos cron-fire webhook (POST /api/cron/fire) — Phase 4E.2.

The webhook authenticates a NAS-minted JWT via the pluggable fire-verifier
(NOT API_SERVER_KEY), then runs the job via the resolved provider's fire_due in
the background, returning 202. These tests monkeypatch the verifier and
resolve_cron_scheduler — the verifier itself is tested with real crypto in
test_chronos_verify.py.
"""

import asyncio
import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/cron/fire", adapter._handle_cron_fire)
    return app


def _create_cron_control_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    adapter._register_unprefixed_cron_control_routes(app.router)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class _SpyProvider:
    """Records fire_due calls; stands in for the resolved provider."""

    def __init__(self):
        self.fired = []

    def fire_due(self, job_id, *, adapters=None, loop=None, fire_at=None):
        self.fired.append((job_id, fire_at))
        return True


@pytest.mark.asyncio
async def test_unprefixed_calendar_reconcile_accepts_only_planner_job_ids(adapter, monkeypatch):
    observed = []

    class Provider:
        def reconcile_calendar_job(self, job_id, action, revision):
            observed.append((job_id, action, revision))
            return {"status": "not_required", "provider": "builtin"}

    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: Provider())
    app = _create_cron_control_app(adapter)
    # Mirrors local-server's production contract: cal-alert- + the first
    # 32 lowercase hex characters of a canonical SHA-256 delivery key.
    valid_id = "cal-alert-" + hashlib.sha256(b"planner-delivery").hexdigest()[:32]
    headers = {"Authorization": "Bearer sk-secret"}
    async with TestClient(TestServer(app)) as cli:
        accepted = await cli.post(
            f"/internal/v1/cron/jobs/{valid_id}/reconcile",
            headers=headers,
            json={"expected_action": "upsert", "projection_revision": 3},
        )
        accepted_status = accepted.status
        accepted_body = await accepted.json()
        rejected = await cli.post(
            "/internal/v1/cron/jobs/..%2Fetc%2Fpasswd/reconcile",
            headers=headers,
            json={"expected_action": "upsert", "projection_revision": 3},
        )
        rejected_status = rejected.status
        noncanonical_statuses = []
        for invalid_id in (
            "cal-alert-recovery",
            "cal-alert-" + "A" * 32,
            "cal-alert-" + "a" * 31 + "_",
        ):
            response = await cli.post(
                f"/internal/v1/cron/jobs/{invalid_id}/reconcile",
                headers=headers,
                json={"expected_action": "upsert", "projection_revision": 3},
            )
            noncanonical_statuses.append(response.status)

    assert accepted_status == 200
    assert accepted_body == {"status": "not_required", "provider": "builtin"}
    assert rejected_status in {400, 404}
    assert noncanonical_statuses == [400, 400, 400]
    assert observed == [(valid_id, "upsert", 3)]


RECOVERY_RECONCILE_2XX_CONTRACT = [
    (
        {"status": "armed", "provider": "chronos", "observed_fire_at": "2026-07-22T12:34:56Z"},
        {"status": "armed", "provider": "chronos", "observed_fire_at": "2026-07-22T12:34:56Z"},
    ),
    (
        {"status": "not_required", "provider": "builtin", "observed_fire_at": "2026-07-22T12:34:56Z"},
        {"status": "not_required", "provider": "builtin", "observed_fire_at": "2026-07-22T12:34:56Z"},
    ),
    (
        {"status": "cancelled", "provider": "chronos", "observed_fire_at": "forged"},
        {"status": "cancelled", "provider": "chronos"},
    ),
    (
        {"status": "superseded", "provider": "chronos", "observed_fire_at": "forged"},
        {"status": "superseded", "provider": "chronos"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_result", "expected_body"), RECOVERY_RECONCILE_2XX_CONTRACT)
async def test_calendar_recovery_reconcile_2xx_wire_contract(
    adapter, monkeypatch, provider_result, expected_body,
):
    class Provider:
        def reconcile_calendar_recovery_arm(self, _body):
            return provider_result

    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: Provider())
    app = _create_cron_control_app(adapter)
    dedupe = "d" * 64
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/internal/v1/cron/calendar-recovery-arms/{dedupe}/reconcile",
            headers={"Authorization": "Bearer sk-secret"},
            json={"dedupe_key": dedupe},
        )
        status = response.status
        body = await response.json()

    assert status == 200
    assert body == expected_body


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_result", [
    {"status": "armed", "provider": "chronos"},
    {"status": "unknown", "provider": "chronos"},
])
async def test_calendar_recovery_reconcile_rejects_invalid_provider_result(
    adapter, monkeypatch, provider_result,
):
    class Provider:
        def reconcile_calendar_recovery_arm(self, _body):
            return provider_result

    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: Provider())
    app = _create_cron_control_app(adapter)
    dedupe = "d" * 64
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/internal/v1/cron/calendar-recovery-arms/{dedupe}/reconcile",
            headers={"Authorization": "Bearer sk-secret"},
            json={"dedupe_key": dedupe},
        )

    assert response.status == 503


@pytest.mark.asyncio
async def test_valid_token_accepts_and_fires(adapter, monkeypatch):
    """Valid NAS-JWT + {job_id} → 202 and fire_due invoked with that id."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    # verifier returns claims (valid token)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "aud": "agent:x"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={"job_id": "abc123", "fire_at": "2026-07-21T09:00:00Z"})
        assert resp.status == 202
        data = await resp.json()
        assert data["job_id"] == "abc123"

    # fire runs in a background thread/task — give it a beat to land.
    for _ in range(50):
        if spy.fired:
            break
        await asyncio.sleep(0.01)
    assert spy.fired == [("abc123", "2026-07-21T09:00:00+00:00")]


@pytest.mark.asyncio
async def test_ordinary_fire_requires_protocol_fire_at(adapter, monkeypatch):
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "ordinary-no-fire-at"},
        )
        assert resp.status == 400
    assert spy.fired == []


@pytest.mark.asyncio
async def test_invalid_token_401_and_no_fire(adapter, monkeypatch):
    """Bad/forged token → 401, fire_due NOT invoked."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: None),  # verification fails
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer forged"},
                              json={"job_id": "abc123"})
        assert resp.status == 401

    await asyncio.sleep(0.05)
    assert spy.fired == []


@pytest.mark.asyncio
async def test_missing_token_401(adapter, monkeypatch):
    """No Authorization header → verifier gets empty token → 401."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    # Real verifier: empty token returns None.
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire", json={"job_id": "abc123"})
        assert resp.status == 401
    assert spy.fired == []


@pytest.mark.asyncio
async def test_valid_token_refuses_during_gateway_drain(adapter, monkeypatch):
    spy = _SpyProvider()
    runner = SimpleNamespace(_draining=False, _external_drain_active=True)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/cron/fire",
                headers={"Authorization": "Bearer good"},
                json={"job_id": "abc123"},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "gateway_draining"
    assert spy.fired == []


@pytest.mark.asyncio
async def test_valid_fire_reservation_blocks_drain_before_body_and_task(adapter, monkeypatch):
    runner = SimpleNamespace(_draining=False, _external_drain_active=False)
    body_started = asyncio.Event()
    release_body = asyncio.Event()
    fired = threading.Event()
    release_fire = threading.Event()

    class BlockingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None, fire_at=None):
            fired.set()
            release_fire.wait(timeout=2)
            return True

    original_json = web.Request.json

    async def delayed_json(request):
        body_started.set()
        await release_body.wait()
        return await original_json(request)

    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", BlockingProvider)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    app = _create_app(adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner), patch.object(
        web.Request, "json", delayed_json
    ):
        async with TestClient(TestServer(app)) as cli:
            request_task = asyncio.create_task(
                cli.post(
                    "/api/cron/fire",
                    headers={"Authorization": "Bearer good"},
                    json={"job_id": "abc123", "fire_at": "2026-07-21T09:00:00Z"},
                )
            )
            await body_started.wait()
            assert adapter.active_agent_work_count() == 1

            release_body.set()
            response = await request_task
            assert response.status == 202
            await asyncio.to_thread(fired.wait, 2)
            assert adapter.active_agent_work_count() == 1
            release_fire.set()
            for _ in range(50):
                if adapter.active_agent_work_count() == 0:
                    break
                await asyncio.sleep(0.01)

    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_missing_job_id_400(adapter, monkeypatch):
    """Valid token but no job_id → 400, no fire."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={})
        assert resp.status == 400
    assert spy.fired == []


@pytest.mark.asyncio
async def test_fire_does_not_require_api_server_key(adapter, monkeypatch):
    """The fire endpoint must NOT gate on API_SERVER_KEY — auth is the NAS-JWT.
    A request with NO API key header but a valid fire token still succeeds."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Bearer is the FIRE token, not the API_SERVER_KEY "sk-secret".
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer nas-jwt"},
                              json={"job_id": "j9", "fire_at": "2026-07-21T09:00:00Z"})
        assert resp.status == 202
    for _ in range(50):
        if spy.fired:
            break
        await asyncio.sleep(0.01)
    assert spy.fired == [("j9", "2026-07-21T09:00:00+00:00")]


@pytest.mark.asyncio
async def test_calendar_fire_persists_attempt_before_202_and_runs_in_background(adapter, monkeypatch):
    """202 means the execution attempt is durable, not that delivery finished."""
    from tests.cron.test_calendar_delivery_v2 import managed_job
    monkeypatch.setattr("plugins.cron_providers.chronos.verify.get_fire_verifier", lambda: (lambda **kw: {"purpose": "cron_fire", "jti": "jwt-fire-7"}))
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda _job_id: managed_job())
    began = []
    executed = []
    monkeypatch.setattr("cron.calendar_delivery.begin_external_calendar_fire", lambda job, **kw: began.append((job, kw)) or {
        "state": "claimed", "attempt_sequence": 7, "dedupe_key": "b" * 64,
        "delivery_generation": 1, "fence_token": 2, "_worker_id": "worker-a",
    })
    monkeypatch.setattr("cron.calendar_delivery.run_external_calendar_delivery", lambda job, begin: executed.append((job, begin)) or {"terminal": False})
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire", headers={"Authorization": "Bearer nas-jwt"}, json={"job_id": "cal-alert-abc"})
        assert resp.status == 202
        assert (await resp.json())["attempt_sequence"] == 7
        assert len(began) == 1
        for _ in range(50):
            if executed:
                break
            await asyncio.sleep(0.01)
        assert len(executed) == 1
    assert began[0][1]["provider_fire_id"] == "jwt-fire-7"


@pytest.mark.asyncio
async def test_calendar_fire_preflight_failure_remains_non_2xx(adapter, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    monkeypatch.setattr("plugins.cron_providers.chronos.verify.get_fire_verifier", lambda: (lambda **kw: {"purpose": "cron_fire"}))
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda _job_id: managed_job())
    monkeypatch.setattr("cron.calendar_delivery.begin_external_calendar_fire", lambda *_args, **_kw: (_ for _ in ()).throw(RuntimeError("offline")))
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire", headers={"Authorization": "Bearer nas-jwt"}, json={"job_id": "cal-alert-abc"})
        assert resp.status == 503
        assert resp.headers["Retry-After"] == "5"


@pytest.mark.asyncio
async def test_paused_calendar_fire_is_consumed_without_delivery(adapter, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job

    class Provider:
        name = "chronos"

        @staticmethod
        def calendar_capabilities():
            return {"provider": "chronos", "contract_version": 1}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(
        "cron.jobs.get_job_raw",
        lambda _job_id: managed_job(enabled=False, state="paused"),
    )
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: Provider())
    executed = []
    monkeypatch.setattr(
        "cron.calendar_delivery.run_external_calendar_delivery",
        lambda *args: executed.append(args),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer legacy-nas-jwt"},
            json={"job_id": "cal-alert-abc"},
        )
        assert resp.status == 202
        assert (await resp.json())["state"] == "cancelled"
        await asyncio.sleep(0)

    assert executed == []


@pytest.mark.asyncio
async def test_duplicate_calendar_webhook_does_not_start_second_executor(adapter, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    monkeypatch.setattr("plugins.cron_providers.chronos.verify.get_fire_verifier", lambda: (lambda **kw: {"purpose": "cron_fire"}))
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda _job_id: managed_job())
    monkeypatch.setattr("cron.calendar_delivery.begin_external_calendar_fire", lambda *_args, **_kw: {
        "state": "claimed", "attempt_sequence": 7, "dedupe_key": "b" * 64,
        "attempt_replayed": True, "delivery_generation": 1, "fence_token": 2,
    })
    executed = []
    monkeypatch.setattr("cron.calendar_delivery.run_external_calendar_delivery", lambda *args: executed.append(args))
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire", headers={"Authorization": "Bearer nas-jwt"}, json={"job_id": "cal-alert-abc", "fire_id": "nas-1"})
        assert resp.status == 202
        await asyncio.sleep(0)
        assert executed == []
