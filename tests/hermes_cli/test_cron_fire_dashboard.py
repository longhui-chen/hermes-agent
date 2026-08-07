"""Tests for the Chronos cron-fire webhook ON THE DASHBOARD APP (web_server).

Regression guard for the relocation bug: the fire webhook MUST live on the
dashboard FastAPI app (`hermes_cli.web_server.app`) — the agent's public HTTP
surface on hosted deployments — not only on the aiohttp APIServerAdapter (which
hosted agents don't expose). It must:
  - be a registered route on the dashboard app,
  - be in PUBLIC_API_PATHS so the dashboard cookie gate doesn't 401 it before
    the JWT verifier runs,
  - reject a bad/missing NAS-JWT with 401 (the JWT is the real gate),
  - 400 on missing job_id,
  - on a valid token, resolve the job's profile and run fire_due in the
    background, returning 202.
"""

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


def _client(auth_required: bool):
    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = auth_required
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host):
    if prev_auth is None:
        if hasattr(web_server.app.state, "auth_required"):
            delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_host




def test_fire_path_is_public():
    """Must bypass the dashboard cookie gate so the NAS bearer-JWT callback
    reaches the verifier (the JWT is the real auth)."""
    assert "/api/cron/fire" in PUBLIC_API_PATHS


def test_bad_token_401(monkeypatch):
    """Invalid NAS-JWT -> 401, even with the dashboard auth gate ENGAGED
    (proves the route is reachable past the cookie gate and the verifier is the
    gate). fire_due must NOT run."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: None),  # verification fails
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j: fired.append((p, j)))

    client, pa, ph = _client(auth_required=True)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer forged"},
                           json={"job_id": "abc"})
        assert resp.status_code == 401
        assert fired == []
    finally:
        _restore(pa, ph)
        client.close()


def test_missing_job_id_400(monkeypatch):
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={})
        assert resp.status_code == 400
    finally:
        _restore(pa, ph)
        client.close()


def test_unknown_job_200_gone(monkeypatch):
    """Valid token but the job isn't found in any profile -> 200 'gone'
    (NAS shouldn't retry a fire for a cancelled/completed job)."""
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: None)
    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "ghost"})
        assert resp.status_code == 200
        assert resp.json().get("status") == "gone"
    finally:
        _restore(pa, ph)
        client.close()


def test_valid_token_accepts_and_fires(monkeypatch):
    """Valid token + known job -> 202 and fire_due invoked for the resolved
    profile."""
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "aud": "agent:x"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile",
                        lambda p, j, fire_at=None: fired.append((p, j, fire_at)) or True)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post("/api/cron/fire",
                           headers={"Authorization": "Bearer good"},
                           json={"job_id": "j1", "fire_at": "2026-07-21T09:00:00Z"})
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "j1"
    finally:
        _restore(pa, ph)
        client.close()
    # background task ran the fire for the resolved profile
    assert fired == [("default", "j1", "2026-07-21T09:00:00+00:00")]


def test_ordinary_fire_requires_fire_at_and_does_not_execute(monkeypatch):
    fired = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(
        web_server,
        "_preflight_calendar_fire_for_profile",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        web_server,
        "_fire_cron_job_for_profile",
        lambda *args: fired.append(args) or True,
    )

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "ordinary-no-fire-at"},
        )
        assert resp.status_code == 400
    finally:
        _restore(pa, ph)
        client.close()
    assert fired == []


def test_calendar_fire_uses_jwt_jti_as_redelivery_identity(monkeypatch):
    """A provider retry carrying the same signed JWT must reach Planner with
    the same fire identity even when the JSON body has only job_id."""
    observed = []
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "jti": "jwt-fire-dashboard"}),
    )
    monkeypatch.setattr(web_server, "_find_cron_job_profile", lambda jid: "default")
    monkeypatch.setattr(
        web_server,
        "_preflight_calendar_fire_for_profile",
        lambda profile, job_id, fire_id: observed.append((profile, job_id, fire_id)) or None,
    )
    monkeypatch.setattr(web_server, "_fire_cron_job_for_profile", lambda *_args: True)

    client, pa, ph = _client(auth_required=False)
    try:
        resp = client.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "cal-alert-1", "fire_at": "2026-07-21T09:00:00Z"},
        )
        assert resp.status_code == 202
    finally:
        _restore(pa, ph)
        client.close()
    assert observed == [("default", "cal-alert-1", "jwt-fire-dashboard")]


def test_dashboard_calendar_preflight_flow_uses_raw_null_contract(tmp_path, monkeypatch):
    from cron.jobs import save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job

    job = managed_job(
        enabled=True,
        state="scheduled",
        next_run_at="2026-07-21T09:00:00Z",
    )
    with web_server._cron_profile_scope(tmp_path):
        save_jobs([job])

    class Provider:
        name = "chronos"

        @staticmethod
        def calendar_capabilities():
            return {"provider": "chronos", "contract_version": 1}

    began = []
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: Provider(),
    )
    monkeypatch.setattr(
        "cron.calendar_delivery.begin_external_calendar_fire",
        lambda raw_job, **kwargs: began.append((raw_job, kwargs)) or {"state": "claimed"},
    )
    monkeypatch.setattr(web_server, "_cron_profile_home", lambda _profile: ("default", tmp_path))

    result = web_server._preflight_calendar_fire_for_profile(
        "default", job["id"], "provider-fire-1",
    )

    assert result == (job, {"state": "claimed"})
    assert began[0][0]["prompt"] is None
    assert began[0][1]["provider_fire_id"] == "provider-fire-1"
