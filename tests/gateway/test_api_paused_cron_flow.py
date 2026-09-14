"""Real HTTP/store boundary for prepare-before-trigger callers."""

from datetime import datetime, timedelta, timezone

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from cron import jobs
from gateway.config import PlatformConfig
from gateway.platforms import api_server


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True, "omitted", None, "false", 0])
async def test_http_initial_state_is_atomic(tmp_path, monkeypatch, enabled):
    now = [datetime(2030, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    monkeypatch.setattr(api_server, "_CRON_AVAILABLE", True)
    monkeypatch.setattr(api_server, "_notify_cron_provider_jobs_changed", lambda: None)
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "fixture-key"}))
    app = web.Application()
    app.router.add_post("/api/jobs", adapter._handle_create_job)
    app.router.add_post("/api/jobs/{job_id}/run", adapter._handle_run_job)
    body = {"name": "finite observation", "schedule": "1m", "prompt": "fixture"}
    if enabled != "omitted":
        body["enabled"] = enabled
    with jobs.use_cron_store(tmp_path):
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/api/jobs", json=body)
            assert response.status == 401
            assert jobs.load_jobs() == []
            headers = {"Authorization": "Bearer fixture-key"}
            response = await client.post("/api/jobs", json=body, headers=headers)
            if enabled not in (False, True, "omitted") or type(enabled) is int:
                assert response.status == 400
                assert jobs.load_jobs() == []
                return
            assert response.status == 200
            job = (await response.json())["job"]
            assert job["enabled"] is (enabled is not False)
            assert jobs.get_job(job["id"])["enabled"] is (enabled is not False)
            if enabled is False:
                now[0] += timedelta(minutes=2)
                assert jobs.get_due_jobs() == []
                assert jobs.claim_job_for_fire(job["id"]) is False
                response = await client.post(f"/api/jobs/{job['id']}/run", headers=headers)
                assert response.status == 200
                assert [item["id"] for item in jobs.get_due_jobs()] == [job["id"]]
