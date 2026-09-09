"""Bounded camera input and existing auxiliary vision bridge, without networking."""

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import zet_agent_camera_vision as bridge


def payload():
    return {
        "image_data_uri": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8\xfftest").decode(),
        "frame_times": ["2026-09-09T00:00:00Z", "2026-09-09T00:00:02Z", "2026-09-09T00:00:04Z"],
        "subject_kind": "person", "subject_ref": "", "predicate": "appears",
        "zone_id": "", "min_duration_seconds": 0, "evidence_ref": "observation:abc:1",
    }


def verdict(**overrides):
    result = {"unknown": False, "matched": True, "duration_seconds": 4,
              "frames": [{"state": "absent"}, {"state": "present", "track_id": "a"},
                         {"state": "present", "track_id": "a"}]}
    result.update(overrides)
    return json.dumps({"success": True, "analysis": json.dumps(result)})


@pytest.mark.parametrize("field,value", [
    ("subject_kind", []), ("predicate", "execute"), ("min_duration_seconds", True),
    ("image_data_uri", "https://example.com/image.jpg"),
    ("image_data_uri", "data:image/jpeg;base64,invalid"),
    ("evidence_ref", "../../secret"), ("frame_times", ["2026-09-09T00:00:00Z"] * 3),
])
def test_invalid_input_rejected(field, value):
    request = payload()
    request[field] = value
    with pytest.raises(ValueError):
        bridge.validate_payload(request)


@pytest.mark.parametrize("changes", [
    {"matched": "true"}, {"frames": []}, {"duration_seconds": True},
    {"subject_ref": "model-selected-person"},
])
def test_untrusted_verdict_rejected(changes):
    with pytest.raises(bridge.VisionUnavailable):
        bridge.parse_verdict(verdict(**changes), payload())


def test_unknown_never_becomes_positive():
    result = bridge.parse_verdict(verdict(unknown=True), payload())
    assert result["matched"] is False
    assert result["evidence_ref"] == payload()["evidence_ref"]


@pytest.mark.asyncio
async def test_uses_existing_vision_tool(monkeypatch):
    from tools import vision_tools

    provider = AsyncMock(return_value=verdict())
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", provider)
    result = await bridge.analyze_batch(payload())
    assert result["matched"] is True
    assert provider.await_args.args[0] == payload()["image_data_uri"]
    assert "never verified personal identity" in provider.await_args.args[1]


@pytest.mark.asyncio
async def test_cancellation_reaches_existing_tool(monkeypatch):
    from tools import vision_tools

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def provider(*args):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(vision_tools, "vision_analyze_tool", provider)
    task = asyncio.create_task(bridge.analyze_batch(payload()))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_http_requires_profile_and_auth_and_sanitizes_failure(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "camera-test-key"}))

    async def handler(request):
        if request.headers.get("X-Test-Profile"):
            request["hermes_profile_home"] = "/isolated/test/profile"
        return await bridge.handle_camera_vision(adapter, request)

    app = web.Application()
    app.router.add_post("/vision", handler)
    mock = AsyncMock(side_effect=bridge.VisionUnavailable("private provider secret"))
    monkeypatch.setattr(bridge, "analyze_batch", mock)
    async with TestClient(TestServer(app)) as client:
        assert (await client.post("/vision", json=payload())).status == 401
        assert (await client.post("/vision", json=payload(), headers={"X-Test-Profile": "1"})).status == 401
        headers = {"X-Test-Profile": "1", "Authorization": "Bearer camera-test-key"}
        response = await client.post("/vision", json=payload(), headers=headers)
        assert response.status == 502
        assert await response.json() == {"code": "camera_vision_unavailable"}
        mock.return_value = bridge.parse_verdict(verdict(), payload())
        mock.side_effect = None
        response = await client.post("/vision", json=payload(), headers=headers)
        assert response.status == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert (await response.json())["data"]["matched"] is True


@pytest.mark.asyncio
async def test_real_profile_route_scopes_vision_and_has_no_unprefixed_route(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zet_agent import ZetAgentAdapter
    from hermes_cli.config import get_hermes_home

    root = tmp_path / ".hermes"
    coder = root / "profiles" / "coder"
    coder.mkdir(parents=True)
    for home in (root, coder):
        (home / ".env").write_text("API_SERVER_KEY=camera-profile-key\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: [("default", root), ("coder", coder)])
    observed = []

    async def analyze(request):
        observed.append(Path(get_hermes_home()))
        return bridge.parse_verdict(verdict(), request)

    monkeypatch.setattr(bridge, "analyze_batch", analyze)
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "camera-profile-key"}))
    app = web.Application()
    adapter._register_camera_vision_route(app.router)
    async with TestClient(TestServer(app)) as client:
        headers = {"Authorization": "Bearer camera-profile-key"}
        for profile in ("main", "coder"):
            response = await client.post(f"/p/{profile}/internal/v1/camera/vision", json=payload(), headers=headers)
            assert response.status == 200, await response.text()
        response = await client.post("/internal/v1/camera/vision", json=payload(), headers=headers)
        assert response.status == 404
        response = await client.post("/p/missing/internal/v1/camera/vision", json=payload(), headers=headers)
        assert response.status == 404
    assert observed == [root, coder]


@pytest.mark.asyncio
async def test_http_disconnect_cancels_pending_analysis(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.zet_agent import ZetAgentAdapter

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "camera-test-key"}))
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def analyze(request):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    async def handler(request):
        request["hermes_profile_home"] = "/isolated/test/profile"
        return await bridge.handle_camera_vision(adapter, request)

    monkeypatch.setattr(bridge, "analyze_batch", analyze)
    app = web.Application()
    app.router.add_post("/vision", handler)
    async with TestClient(TestServer(app)) as client:
        task = asyncio.create_task(client.post("/vision", json=payload(), headers={"Authorization": "Bearer camera-test-key"}))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), 2)
