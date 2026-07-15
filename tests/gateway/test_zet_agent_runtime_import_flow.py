import hashlib

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import ZetAgentAdapter
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_completed_transcript_http_flow_requires_auth_and_commits_atomically(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._session_db = db
    adapter._ensure_session_db = lambda: db
    app = web.Application()
    app.router.add_post("/api/sessions/import", adapter._handle_session_import)
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    payload = {
        "import_id": "flow-import", "operation": "stage", "source": "marvis",
        "source_session_id": "marvis-1", "target_session_id": "hermes-1",
        "title": "Marvis chat", "expected_message_count": 1, "chunk_index": 0,
        "payload_sha256": hashlib.sha256(b"marvis-export").hexdigest(),
        "messages": [{"source_id": "m1", "role": "user", "content": "你好",
                      "created_at": 1_700_000_000}],
    }
    try:
        async with TestClient(TestServer(app)) as cli:
            assert (await cli.post("/api/sessions/import", json=payload)).status == 401
            headers = {"Authorization": "Bearer test-key"}
            staged = await cli.post("/api/sessions/import", json=payload, headers=headers)
            assert staged.status == 200
            assert db.get_session("hermes-1") is None
            committed = await cli.post(
                "/api/sessions/import",
                json={"import_id": "flow-import", "operation": "commit"},
                headers=headers,
            )
            assert committed.status == 200
            assert (await committed.json())["message_count"] == 1
            caps = await cli.get("/v1/capabilities", headers=headers)
            assert (await caps.json())["features"]["completed_transcript_import"] is True
            memory = await cli.post(
                "/api/memory/import",
                json={"import_id": "memory-flow", "mode": "replace", "target": "memory",
                      "payload_sha256": hashlib.sha256(b"memory").hexdigest(),
                      "entries": ["用户喜欢简洁回答"]},
                headers=headers,
            )
            assert memory.status == 200
            assert (await memory.json())["effective_from"] == "next_session"
        assert db.get_messages_as_conversation("hermes-1")[0]["content"] == "你好"
    finally:
        db.close()
