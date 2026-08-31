"""ADIC v1: app_slug/import_operation must survive the real HTTP → jobs.json
→ run_one_job round trip, not just each side's own unit tests.

The P0 this file exists to catch: local-server's dedicated-maintainer bridge
POSTs app_slug/import_operation to /api/jobs
(internal/agent/service/app_dedicated_create.go), but
gateway/platforms/api_server.py's `_handle_create_job` used to forward only a
fixed whitelist of named fields — app_slug fell on the floor before it ever
reached `cron.jobs.create_job`, so `cron/scheduler.py:run_one_job`'s
`job.get("app_slug")` was always falsy and the whole verdict override in
tests/cron/test_import_contract_verdict.py never fired in production. BOTH
sides' unit tests were green: local-server asserted its outgoing payload
shape, hermes's verdict tests constructed job dicts by hand instead of
driving them through the HTTP handler. Only a test that actually walks
HTTP → jobs.json → run_one_job catches "each side green, wired together
dead" — see feedback_scope_by_problem_boundary / the review-loop discipline
this workstream follows.
"""
import cron.scheduler as s
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from cron.jobs import get_job_raw, update_job
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from tests.cron.test_import_contract_verdict import _mark_call, _patch_pipeline


def _make_jobs_api_app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/jobs", adapter._handle_create_job)
    app.router.add_get("/api/jobs/{job_id}", adapter._handle_get_job)
    app.router.add_patch("/api/jobs/{job_id}", adapter._handle_update_job)
    return app


_CREATE_BODY = {
    "name": "hangzhou weather sync",
    "schedule": "*/10 * * * *",
    "prompt": "fetch and import today's forecast",
    "app_slug": "hangzhou-weather-live",
    "import_operation": "data.import",
}


@pytest.mark.asyncio
async def test_app_slug_reaches_jobs_json_through_the_real_http_handler():
    """Not a mock of _cron_create: a real POST through _handle_create_job,
    into a real (per-test tempdir) jobs.json via cron.jobs.create_job."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json=_CREATE_BODY)
        assert resp.status == 200
        created = (await resp.json())["job"]

    on_disk = get_job_raw(created["id"])
    assert on_disk is not None
    assert on_disk["app_slug"] == "hangzhou-weather-live"
    assert on_disk["import_operation"] == "data.import"


@pytest.mark.asyncio
async def test_app_slug_persisted_via_http_actually_drives_run_one_job_verdict(monkeypatch):
    """The full chain: POST /api/jobs -> jobs.json -> run_one_job reads
    job.get("app_slug") and applies the ADIC verdict override. This is the
    test that would have caught the P0: with the whitelist bug, on_disk here
    has no app_slug key, job.get("app_slug") is falsy, and the assertions
    below fail because the old 'agent replied cleanly' verdict (ok=True)
    would have won instead."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json=_CREATE_BODY)
        created = (await resp.json())["job"]

    on_disk = get_job_raw(created["id"])
    # Agent replied cleanly and said nothing about failing — the exact shape
    # of the Hangzhou 17:57 incident — but never attempted an import this run.
    calls = _patch_pipeline(monkeypatch, success=True, final="今天天气不错，已完成同步",
                             import_attempts=[])

    s.run_one_job(on_disk)

    _, jid, ok, err = _mark_call(calls)
    assert jid == created["id"]
    assert ok is False
    assert err == "no import attempted in this run"


@pytest.mark.asyncio
async def test_empty_app_slug_is_rejected_not_silently_dropped():
    """A garbage app_slug from a misbehaving writer must fail loudly (400),
    not silently create a job that can never be judged by its import
    outcomes — that would recreate the exact silent-failure bug this
    workstream exists to close. This layer's bound is safety-only (see the
    module docstring), so the case worth testing here is genuinely unsafe —
    not merely "doesn't look like a slug"."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={**_CREATE_BODY, "app_slug": ""})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_overlong_app_slug_is_rejected():
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={**_CREATE_BODY, "app_slug": "x" * 129})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_app_slug_with_embedded_newline_is_rejected():
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={
            **_CREATE_BODY, "app_slug": "weather\ninjected-line",
        })
        assert resp.status == 400


@pytest.mark.asyncio
async def test_app_slug_with_path_separator_is_rejected():
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={
            **_CREATE_BODY, "app_slug": "weather/../secrets",
        })
        assert resp.status == 400


@pytest.mark.asyncio
async def test_app_slug_outside_local_servers_own_slug_format_is_still_accepted():
    """The whole point of the safety-only bound: a value that would fail
    local-server's own apphost slug pattern (^[a-z][a-z0-9-]{2,31}$) —
    uppercase, spaces, underscores, or just longer than 32 chars — must NOT
    400 here. This layer stores the value and runs truthy checks on it; it
    does not get to veto a slug local-server considers (or later widens to
    consider) valid. Regression pin for the exact mistake this field's
    validation made on its first pass: copying local-server's business
    format instead of only bounding what this layer needs as custodian."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={
            **_CREATE_BODY, "app_slug": "Hangzhou_Weather LIVE v2 (beta)",
        })
        assert resp.status == 200
        created = (await resp.json())["job"]

    on_disk = get_job_raw(created["id"])
    assert on_disk["app_slug"] == "Hangzhou_Weather LIVE v2 (beta)"


@pytest.mark.asyncio
async def test_unknown_body_fields_still_dropped_not_blanket_passthrough():
    """The fix widens the whitelist by exactly app_slug/import_operation — it
    must not regress into forwarding arbitrary body keys to create_job."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/jobs", json={
            **_CREATE_BODY, "evil_field": "should never reach jobs.json",
        })
        assert resp.status == 200
        created = (await resp.json())["job"]

    on_disk = get_job_raw(created["id"])
    assert "evil_field" not in on_disk
    assert on_disk["app_slug"] == "hangzhou-weather-live"  # the real field still worked


@pytest.mark.asyncio
async def test_patch_cannot_change_app_slug_and_does_not_drop_it():
    """PATCH /api/jobs/{id} must not be a second way to set app_slug (it is
    server-stamped once, at provision time only), and an unrelated field
    update (e.g. renaming the job) must not lose the value already on disk —
    update_job merges {**job, **updates}, it does not rebuild the record."""
    app = _make_jobs_api_app()
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/jobs", json=_CREATE_BODY)
        job_id = (await create_resp.json())["job"]["id"]

        # app_slug in the PATCH body is silently ignored (not in
        # _UPDATE_ALLOWED_FIELDS) — same behavior as any other unknown field.
        patch_resp = await cli.patch(
            f"/api/jobs/{job_id}",
            json={"name": "renamed sync", "app_slug": "a-different-app"},
        )
        assert patch_resp.status == 200
        patched = (await patch_resp.json())["job"]
        assert patched["name"] == "renamed sync"

    on_disk = get_job_raw(job_id)
    assert on_disk["name"] == "renamed sync"
    # Unchanged from creation — PATCH neither adopted the attempted override
    # nor dropped the original value as a side effect of updating "name".
    assert on_disk["app_slug"] == "hangzhou-weather-live"


def test_update_job_hard_rejects_app_slug_even_bypassing_the_http_whitelist():
    """Defense in depth below the HTTP layer: cron.jobs._IMMUTABLE_JOB_FIELDS
    must reject a direct update_job() call carrying app_slug/import_operation
    regardless of caller — the HTTP handler's allowlist omission is not the
    only thing standing between a future bug and a silently overwritten
    app_slug."""
    from cron.jobs import create_job

    job = create_job(
        prompt="p", schedule="*/10 * * * *", name="j",
        app_slug="app-a", import_operation="data.import",
    )
    with pytest.raises(ValueError, match="app_slug"):
        update_job(job["id"], {"app_slug": "app-b"})
    with pytest.raises(ValueError, match="import_operation"):
        update_job(job["id"], {"import_operation": "data.export"})

    # Neither rejected attempt mutated the stored record.
    on_disk = get_job_raw(job["id"])
    assert on_disk["app_slug"] == "app-a"
    assert on_disk["import_operation"] == "data.import"
