"""Flow coverage for preserving model-visibility policy across API forks."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_session_fork_flow_keeps_calendar_notification_hidden_from_model(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    app = web.Application()
    app.router.add_post(
        "/api/sessions/{session_id}/fork", adapter._handle_fork_session,
    )
    try:
        source_id = db.create_session(
            "zettlab:oh_test:main:calendar-reminders", "api_server",
        )
        delivery_key = "a" * 64
        message_id = db.stage_calendar_notification(
            source_id, "display-only calendar text", delivery_key, 1,
        )
        assert db.activate_calendar_notification(delivery_key, 1, message_id) is True

        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                f"/api/sessions/{source_id}/fork",
                json={"id": "calendar-fork", "title": "Calendar copy"},
            )
            assert response.status == 201

        displayed = db.get_messages("calendar-fork")
        assert [message["content"] for message in displayed] == [
            "display-only calendar text",
        ]
        assert displayed[0]["llm_visible"] == 0
        assert db.get_messages_for_model("calendar-fork") == []
    finally:
        db.close()
