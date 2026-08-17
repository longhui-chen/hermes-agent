"""POST /api/jobs/{job_id}/defer — the governor's postpone endpoint.

The endpoint must pass seconds/until/reason through to jobs.defer_job,
reject a body with neither time argument, 404 an unknown job, and surface
ValueError as 400 — mirroring the pause/resume contract.
"""

from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import APIServerAdapter, cors_middleware

from tests.gateway.test_api_server_jobs import SAMPLE_JOB, VALID_JOB_ID, _make_adapter

_MOD = "gateway.platforms.api_server"


def _create_app(adapter: APIServerAdapter):
    from aiohttp import web

    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/jobs/{job_id}/defer", adapter._handle_defer_job)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.mark.asyncio
async def test_defer_passes_arguments_through(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch(_MOD + "._cron_defer") as defer_mock:
            defer_mock.return_value = {**SAMPLE_JOB, "defer_count": 1}
            resp = await cli.post(
                f"/api/jobs/{VALID_JOB_ID}/defer",
                json={"seconds": 300, "reason": "governor:memory_pressure"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["job"]["defer_count"] == 1
            defer_mock.assert_called_once_with(
                VALID_JOB_ID, seconds=300.0, until=None, reason="governor:memory_pressure"
            )


@pytest.mark.asyncio
async def test_defer_requires_a_time_argument(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/defer", json={"reason": "x"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_defer_unknown_job_is_404(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch(_MOD + "._cron_defer", return_value=None):
            resp = await cli.post(f"/api/jobs/{VALID_JOB_ID}/defer", json={"seconds": 60})
            assert resp.status == 404


@pytest.mark.asyncio
async def test_defer_validation_error_is_400(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch(_MOD + "._cron_defer", side_effect=ValueError("until must be ISO-8601")):
            resp = await cli.post(
                f"/api/jobs/{VALID_JOB_ID}/defer", json={"until": "not-a-date"}
            )
            assert resp.status == 400
