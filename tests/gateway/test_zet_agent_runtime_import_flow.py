import asyncio
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.zet_agent import (
    ZetAgentAdapter,
    _to_thread_with_completion_barrier,
)
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
            unsafe = await cli.post(
                "/api/sessions/import",
                json={**payload, "import_id": "glob-flow", "target_session_id": "*"},
                headers=headers,
            )
            assert unsafe.status == 400
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


@pytest.mark.asyncio
async def test_memory_import_http_flow_returns_recoverable_backup(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    memory_path = home / "memories" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text("existing curated fact", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    app = web.Application()
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/memory/import",
            json={
                "import_id": "backup-flow", "mode": "replace", "target": "memory",
                "payload_sha256": hashlib.sha256(b"backup-flow").hexdigest(),
                "entries": ["replacement fact"],
            },
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status == 200
        result = await response.json()
        assert Path(result["backup_path"]).read_text(encoding="utf-8") == "existing curated fact"


@pytest.mark.asyncio
async def test_memory_import_http_flow_rejects_canonical_symlink(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    memory_path = home / "memories" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    outside = tmp_path / "outside-memory.md"
    outside.write_text("external content must survive", encoding="utf-8")
    memory_path.symlink_to(outside)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    app = web.Application()
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/memory/import",
            json={
                "import_id": "symlink-flow",
                "mode": "replace",
                "target": "memory",
                "payload_sha256": hashlib.sha256(b"symlink-flow").hexdigest(),
                "entries": ["replacement fact"],
            },
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status == 409
    assert memory_path.is_symlink()
    assert outside.read_text(encoding="utf-8") == "external content must survive"
    assert not (home / "memories" / ".imports").exists()


@pytest.mark.asyncio
async def test_memory_import_http_flow_returns_conflict_for_live_cas_edit(
    tmp_path, monkeypatch
):
    import tools.memory_tool as memory_tool

    home = tmp_path / ".hermes"
    memory_path = home / "memories" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text("existing curated fact", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = memory_tool.MemoryStore(memory_char_limit=100, user_char_limit=100)
    original_write_receipt = store._write_import_receipt

    def edit_after_prepare(path, receipt):
        original_write_receipt(path, receipt)
        if receipt["state"] == "prepared":
            memory_path.write_text("external edit wins", encoding="utf-8")

    monkeypatch.setattr(store, "_write_import_receipt", edit_after_prepare)
    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda: store)
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    app = web.Application()
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/memory/import",
            json={
                "import_id": "cas-http-flow", "mode": "replace", "target": "memory",
                "payload_sha256": hashlib.sha256(b"cas-http-flow").hexdigest(),
                "entries": ["replacement fact"],
            },
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status == 409
        assert (await response.json())["error"]["code"] == "memory_import_conflict"
    assert memory_path.read_text(encoding="utf-8") == "external edit wins"


@pytest.mark.asyncio
async def test_runtime_import_flow_rejects_credentials_at_final_consumer(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(tmp_path / "state.db")
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._session_db = db
    adapter._ensure_session_db = lambda: db
    app = web.Application()
    app.router.add_post("/api/sessions/import", adapter._handle_session_import)
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)
    headers = {"Authorization": "Bearer test-key"}
    payload = {
        "import_id": "credential-flow", "operation": "stage", "source": "marvis",
        "source_session_id": "source", "target_session_id": "target",
        "title": None, "expected_message_count": 1, "chunk_index": 0,
        "payload_sha256": hashlib.sha256(b"credential").hexdigest(),
        "messages": [{"role": "user",
                      "content": '{"Authorization": "Bearer abcdefghexamplehijklmnop"}',
                      "created_at": 1}],
    }
    try:
        async with TestClient(TestServer(app)) as cli:
            transcript = await cli.post(
                "/api/sessions/import", json=payload, headers=headers
            )
            assert transcript.status == 400
            assert (await transcript.json())["error"]["code"] == "invalid_runtime_import"
            memory = await cli.post(
                "/api/memory/import",
                json={"import_id": "credential-memory", "mode": "replace",
                      "target": "memory",
                      "payload_sha256": hashlib.sha256(b"memory").hexdigest(),
                      "entries": ["{'Authorization': 'Bearer abcdefghexamplehijklmnop'}"]},
                headers=headers,
            )
            assert memory.status == 400
            assert (await memory.json())["error"]["code"] == "invalid_memory_import"
            aws_query = await cli.post(
                "/api/sessions/import",
                json={
                    **payload,
                    "import_id": "aws-query-credential",
                    "messages": [{
                        "role": "user",
                        "content": "https://bucket.s3.amazonaws.com/item?X-Amz-Signature=" + "a" * 64,
                        "created_at": 1,
                    }],
                },
                headers=headers,
            )
            assert aws_query.status == 400
            slack_app = await cli.post(
                "/api/memory/import",
                json={
                    "import_id": "slack-app-credential", "mode": "replace",
                    "target": "memory",
                    "payload_sha256": hashlib.sha256(b"slack-app").hexdigest(),
                    "entries": ["xapp-1-123456789012-abcdefghijklmnopqrstuvwxyzABCD"],
                },
                headers=headers,
            )
            assert slack_app.status == 400
            pgp_transcript = await cli.post(
                "/api/sessions/import",
                json={
                    **payload,
                    "import_id": "pgp-transcript-credential",
                    "messages": [{
                        "role": "user",
                        "content": "-----BEGIN PGP PRIVATE KEY BLOCK-----",
                        "created_at": 1,
                    }],
                },
                headers=headers,
            )
            assert pgp_transcript.status == 400
            pgp_memory = await cli.post(
                "/api/memory/import",
                json={
                    "import_id": "pgp-memory-credential",
                    "mode": "replace",
                    "target": "memory",
                    "payload_sha256": hashlib.sha256(b"pgp-memory").hexdigest(),
                    "entries": ["-----BEGIN PGP PRIVATE KEY BLOCK-----"],
                },
                headers=headers,
            )
            assert pgp_memory.status == 400
            credential_id = "sk-1234567890abcdefghij"
            metadata = await cli.post(
                "/api/sessions/import",
                json={
                    **payload,
                    "import_id": credential_id,
                    "messages": [
                        {"role": "user", "content": "safe", "created_at": 1}
                    ],
                },
                headers=headers,
            )
            assert metadata.status == 400
            assert (await metadata.json())["error"]["code"] == "invalid_runtime_import"
            memory_metadata = await cli.post(
                "/api/memory/import",
                json={
                    "import_id": credential_id,
                    "mode": "replace",
                    "target": "memory",
                    "payload_sha256": hashlib.sha256(b"memory-id").hexdigest(),
                    "entries": ["safe fact"],
                },
                headers=headers,
            )
            assert memory_metadata.status == 400
            assert (await memory_metadata.json())["error"]["code"] == "invalid_memory_import"
        assert db.get_session("target") is None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_chunks"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_message_ids"
        ).fetchone()[0] == 0
        assert not (home / "memories" / "MEMORY.md").exists()
        assert not (home / "memories" / "USER.md").exists()
        assert not (home / "memories" / ".imports").exists()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_memory_import_flow_blocks_unload_until_worker_finishes(monkeypatch):
    import tools.memory_tool as memory_tool

    started = threading.Event()
    release = threading.Event()

    class _Store:
        def import_replace(self, **_kwargs):
            started.set()
            assert release.wait(2)
            return {"status": "completed"}

    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda: _Store())
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    app = web.Application()
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)
    app.router.add_post("/v1/profile/unload", adapter._handle_profile_unload)
    headers = {"Authorization": "Bearer test-key"}
    body = {"import_id": "blocking", "mode": "replace", "target": "memory",
            "payload_sha256": hashlib.sha256(b"blocking").hexdigest(),
            "entries": ["safe fact"]}

    async with TestClient(TestServer(app)) as cli:
        import_task = asyncio.create_task(
            cli.post("/api/memory/import", json=body, headers=headers)
        )
        assert await asyncio.to_thread(started.wait, 1)
        blocked = await cli.post("/v1/profile/unload", headers=headers)
        assert blocked.status == 409
        assert (await blocked.json())["active_imports"] == 1
        release.set()
        assert (await import_task).status == 200

        unloaded = await cli.post("/v1/profile/unload", headers=headers)
        assert unloaded.status == 200
        after_unload = await cli.post(
            "/api/memory/import", json=body, headers=headers
        )
        assert after_unload.status == 409


@pytest.mark.asyncio
async def test_session_import_flow_blocks_unload_until_stage_finishes():
    started = threading.Event()
    release = threading.Event()

    class _DB:
        def stage_completed_transcript_import(self, **_kwargs):
            started.set()
            assert release.wait(2)
            return {"status": "staged"}

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._ensure_session_db = lambda: _DB()
    app = web.Application()
    app.router.add_post("/api/sessions/import", adapter._handle_session_import)
    app.router.add_post("/v1/profile/unload", adapter._handle_profile_unload)
    headers = {"Authorization": "Bearer test-key"}
    body = {"import_id": "blocking-session", "operation": "stage",
            "source": "workbuddy", "source_session_id": "source",
            "target_session_id": "target", "title": None,
            "payload_sha256": hashlib.sha256(b"blocking").hexdigest(),
            "expected_message_count": 1, "chunk_index": 0,
            "messages": [{"role": "user", "content": "safe", "created_at": 1}]}

    async with TestClient(TestServer(app)) as cli:
        import_task = asyncio.create_task(
            cli.post("/api/sessions/import", json=body, headers=headers)
        )
        assert await asyncio.to_thread(started.wait, 1)
        blocked = await cli.post("/v1/profile/unload", headers=headers)
        assert blocked.status == 409
        release.set()
        assert (await import_task).status == 200


class _DirectImportRequest(dict):
    def __init__(self, body):
        super().__init__()
        self._body = body
        self.headers = {"Authorization": "Bearer test-key"}
        self.method = "POST"
        self.path_qs = "/api/import"
        self.remote = "127.0.0.1"
        self.transport = None

    async def json(self):
        return self._body


@pytest.mark.asyncio
async def test_cancelled_memory_import_holds_unload_barrier_until_worker_exits(
    monkeypatch,
):
    import tools.memory_tool as memory_tool

    started = threading.Event()
    release = threading.Event()

    class _Store:
        def import_replace(self, **_kwargs):
            started.set()
            assert release.wait(2)
            return {"status": "completed"}

    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda: _Store())
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "cancel-memory", "mode": "replace", "target": "memory",
        "payload_sha256": hashlib.sha256(b"cancel-memory").hexdigest(),
        "entries": ["safe fact"],
    })
    task = asyncio.create_task(adapter._handle_memory_import(request))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    blocked = await adapter._handle_profile_unload(_DirectImportRequest(None))
    assert blocked.status == 409
    assert json.loads(blocked.text)["active_imports"] == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await adapter._handle_profile_unload(_DirectImportRequest(None))).status == 200


@pytest.mark.asyncio
async def test_cancelled_session_import_holds_unload_barrier_until_worker_exits():
    started = threading.Event()
    release = threading.Event()

    class _DB:
        def stage_completed_transcript_import(self, **_kwargs):
            started.set()
            assert release.wait(2)
            return {"status": "staged"}

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._ensure_session_db = lambda: _DB()
    request = _DirectImportRequest({
        "import_id": "cancel-session", "operation": "stage",
        "source": "workbuddy", "source_session_id": "source",
        "target_session_id": "target", "title": None,
        "payload_sha256": hashlib.sha256(b"cancel-session").hexdigest(),
        "expected_message_count": 1, "chunk_index": 0,
        "messages": [{"role": "user", "content": "safe", "created_at": 1}],
    })
    task = asyncio.create_task(adapter._handle_session_import(request))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    blocked = await adapter._handle_profile_unload(_DirectImportRequest(None))
    assert blocked.status == 409
    assert json.loads(blocked.text)["active_imports"] == 1
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await adapter._handle_profile_unload(_DirectImportRequest(None))).status == 200


@pytest.mark.asyncio
async def test_cancelled_import_keeps_cancellation_when_worker_fails():
    started = threading.Event()
    release = threading.Event()

    def fail_after_release():
        started.set()
        assert release.wait(2)
        raise RuntimeError("worker failed")

    task = asyncio.create_task(
        _to_thread_with_completion_barrier(fail_after_release)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_runtime_import_barrier_allows_only_a_recreated_profile_generation(tmp_path):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))

    active, owner = adapter._block_runtime_import_profile(profile_home)
    assert active == 0
    adapter._complete_runtime_import_profile_unload(profile_home, owner)
    assert adapter._begin_runtime_import_operation(profile_home) is None
    profile_home.rmdir()
    assert adapter._begin_runtime_import_operation(profile_home) is None
    profile_home.mkdir()
    key = adapter._begin_runtime_import_operation(profile_home)
    assert key == adapter._profile_home_key(profile_home)
    adapter._end_runtime_import_operation(key)


@pytest.mark.asyncio
async def test_unload_then_sweep_does_not_recreate_deleted_profile(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "coder"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    db = SessionDB(profile_home / "state.db")
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    key = adapter._profile_home_key(profile_home)
    adapter._session_db = db
    adapter._session_dbs = {key: db}
    monkeypatch.setattr(adapter, "_multiplex_profile_homes", lambda: {})

    class _Request(dict):
        def __init__(self):
            super().__init__(
                hermes_profile="coder", hermes_profile_home=str(profile_home)
            )
            self.headers = {"Authorization": "Bearer test-key"}
            self.method = "POST"
            self.path_qs = "/p/coder/v1/profile/unload"
            self.remote = "127.0.0.1"
            self.transport = None

    response = await adapter._handle_profile_unload(_Request())
    assert response.status == 200
    assert adapter._session_db is None
    shutil.rmtree(profile_home)

    assert await adapter._cleanup_stale_runtime_imports_once() == 0
    assert not profile_home.exists()


@pytest.mark.asyncio
async def test_uncached_profile_cleanup_db_open_blocks_profile_unload(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "coder"
    db = SessionDB(profile_home / "state.db")
    db.close()

    started = threading.Event()
    release = threading.Event()
    real_session_db = SessionDB

    def slow_open(path):
        started.set()
        assert release.wait(2)
        return real_session_db(path)

    monkeypatch.setattr("hermes_state.SessionDB", slow_open)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._session_db = None
    adapter._session_dbs = {}
    adapter._ensure_session_db = lambda: None
    monkeypatch.setattr(
        adapter, "_multiplex_profile_homes", lambda: {"coder": profile_home}
    )

    class _Request(dict):
        def __init__(self):
            super().__init__(
                hermes_profile="coder", hermes_profile_home=str(profile_home)
            )
            self.headers = {"Authorization": "Bearer test-key"}
            self.method = "POST"
            self.path_qs = "/p/coder/v1/profile/unload"
            self.remote = "127.0.0.1"
            self.transport = None

    cleanup_task = asyncio.create_task(adapter._cleanup_stale_runtime_imports_once())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        response = await adapter._handle_profile_unload(_Request())
        assert response.status == 409
        assert json.loads(response.body)["active_imports"] == 1
    finally:
        release.set()
        await cleanup_task


@pytest.mark.asyncio
async def test_failed_reload_and_unload_do_not_release_a_successful_unload_barrier(
    tmp_path,
):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    class _Request(dict):
        def __init__(self, *, body=None, path="/p/coder/v1/profile/unload"):
            super().__init__(
                hermes_profile="coder", hermes_profile_home=str(profile_home)
            )
            self._body = body
            self.headers = {"Authorization": "Bearer test-key"}
            self.method = "POST"
            self.path_qs = path
            self.remote = "127.0.0.1"
            self.transport = None

        async def json(self):
            return self._body

    first = await adapter._handle_profile_unload(_Request())
    assert first.status == 200

    class _FailingRunner:
        async def unload_profile_runtime(self, _profile):
            raise RuntimeError("unload failed")

        def invalidate_cached_agents_for_profile(self, _profile):
            return 0

    class _FailingDB:
        def clear_all_system_prompts(self):
            raise RuntimeError("reload failed")

    adapter.gateway_runner = _FailingRunner()
    adapter._ensure_session_db = lambda: _FailingDB()
    failed_reload = await adapter._handle_profile_reload(
        _Request(path="/p/coder/v1/profile/reload")
    )
    assert failed_reload.status == 500
    failed = await adapter._handle_profile_unload(_Request())
    assert failed.status == 500

    blocked = await adapter._handle_memory_import(
        _Request(
            path="/p/coder/api/memory/import",
            body={
                "import_id": "after-failed-unload",
                "mode": "replace",
                "target": "memory",
                "payload_sha256": hashlib.sha256(b"after-failed-unload").hexdigest(),
                "entries": ["safe fact"],
            },
        )
    )
    assert blocked.status == 409


@pytest.mark.asyncio
async def test_reload_cannot_release_a_concurrent_successful_unload_barrier(tmp_path):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    class _Request(dict):
        def __init__(self, *, body=None, path="/p/coder/v1/profile/unload"):
            super().__init__(
                hermes_profile="coder", hermes_profile_home=str(profile_home)
            )
            self._body = body
            self.headers = {"Authorization": "Bearer test-key"}
            self.method = "POST"
            self.path_qs = path
            self.remote = "127.0.0.1"
            self.transport = None

        async def json(self):
            return self._body

    class _Runner:
        async def unload_profile_runtime(self, _profile):
            return {"evicted_sessions": 0, "disconnected_adapters": 0}

        def invalidate_cached_agents_for_profile(self, _profile):
            return 0

    adapter.gateway_runner = _Runner()
    assert (await adapter._handle_profile_unload(_Request())).status == 200

    reload_started = threading.Event()
    allow_reload = threading.Event()

    class _BlockingDB:
        def clear_all_system_prompts(self):
            reload_started.set()
            assert allow_reload.wait(2)
            return 0

    adapter._ensure_session_db = lambda: _BlockingDB()
    concurrent_unload = []

    def unload_while_reload_is_in_progress():
        assert reload_started.wait(2)
        concurrent_unload.append(
            asyncio.run(adapter._handle_profile_unload(_Request())).status
        )
        allow_reload.set()

    thread = threading.Thread(target=unload_while_reload_is_in_progress)
    thread.start()
    reloaded = await adapter._handle_profile_reload(
        _Request(path="/p/coder/v1/profile/reload")
    )
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert reloaded.status == 200
    assert concurrent_unload == [200]

    blocked = await adapter._handle_memory_import(
        _Request(
            path="/p/coder/api/memory/import",
            body={
                "import_id": "after-concurrent-unload",
                "mode": "replace",
                "target": "memory",
                "payload_sha256": hashlib.sha256(b"after-concurrent-unload").hexdigest(),
                "entries": ["safe fact"],
            },
        )
    )
    assert blocked.status == 409


@pytest.mark.asyncio
async def test_gateway_runtime_import_cleanup_converges_without_new_import(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.stage_completed_transcript_import(
        import_id="abandoned", source="marvis", source_session_id="source",
        target_session_id="target", title=None,
        payload_sha256=hashlib.sha256(b"abandoned").hexdigest(),
        expected_message_count=1, chunk_index=0,
        messages=[{"source_id": "m1", "role": "user", "content": "private",
                   "created_at": 0}],
    )
    db._conn.execute(
        "UPDATE runtime_imports SET updated_at = ? WHERE import_id = ?",
        (time.time() - 25 * 60 * 60, "abandoned"),
    )
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter._session_db = db
    adapter._ensure_session_db = lambda: db
    try:
        assert await adapter._cleanup_stale_runtime_imports_once() == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = 'abandoned'"
        ).fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_gateway_runtime_import_cleanup_sweeps_every_cached_profile(tmp_path):
    first = SessionDB(tmp_path / "first" / "state.db")
    second = SessionDB(tmp_path / "second" / "state.db")
    for index, db in enumerate((first, second), start=1):
        db.stage_completed_transcript_import(
            import_id=f"abandoned-{index}", source="marvis",
            source_session_id=f"source-{index}",
            target_session_id=f"target-{index}", title=None,
            payload_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            expected_message_count=1, chunk_index=0,
            messages=[{
                "source_id": f"m-{index}", "role": "user",
                "content": "private", "created_at": 0,
            }],
        )
        db._conn.execute(
            "UPDATE runtime_imports SET updated_at = ? WHERE import_id = ?",
            (time.time() - 25 * 60 * 60, f"abandoned-{index}"),
        )

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._session_db = first
    adapter._session_dbs = {"first": first, "second": second}
    adapter._ensure_session_db = lambda: first
    try:
        assert await adapter._cleanup_stale_runtime_imports_once() == 2
        for index, db in enumerate((first, second), start=1):
            assert db._conn.execute(
                "SELECT COUNT(*) FROM runtime_imports WHERE import_id = ?",
                (f"abandoned-{index}",),
            ).fetchone()[0] == 0
    finally:
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_gateway_runtime_import_cleanup_discovers_uncached_served_profile(
    tmp_path, monkeypatch
):
    profile_home = tmp_path / "profiles" / "coder"
    db = SessionDB(profile_home / "state.db")
    db.stage_completed_transcript_import(
        import_id="uncached", source="marvis", source_session_id="source",
        target_session_id="target", title=None,
        payload_sha256=hashlib.sha256(b"uncached").hexdigest(),
        expected_message_count=1, chunk_index=0,
        messages=[{"role": "user", "content": "private", "created_at": 0}],
    )
    db._conn.execute(
        "UPDATE runtime_imports SET updated_at = ?",
        (time.time() - 25 * 60 * 60,),
    )
    db.close()

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._session_db = None
    adapter._session_dbs = {}
    adapter._ensure_session_db = lambda: None
    monkeypatch.setattr(
        adapter, "_multiplex_profile_homes", lambda: {"coder": profile_home}
    )

    await adapter._cleanup_stale_runtime_imports_once()
    reopened = SessionDB(profile_home / "state.db")
    try:
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = 'uncached'"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_uncached_profile_open_failure_does_not_block_other_profile_cleanup(
    tmp_path, monkeypatch
):
    broken_home = tmp_path / "profiles" / "broken"
    broken_home.mkdir(parents=True)
    (broken_home / "state.db").write_bytes(b"not sqlite")
    healthy_home = tmp_path / "profiles" / "healthy"
    db = SessionDB(healthy_home / "state.db")
    db.stage_completed_transcript_import(
        import_id="healthy-stale", source="marvis", source_session_id="source",
        target_session_id="target", title=None,
        payload_sha256=hashlib.sha256(b"healthy").hexdigest(),
        expected_message_count=1, chunk_index=0,
        messages=[{"role": "user", "content": "private", "created_at": 0}],
    )
    db._conn.execute(
        "UPDATE runtime_imports SET updated_at = ?",
        (time.time() - 25 * 60 * 60,),
    )
    db.close()

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._session_db = None
    adapter._session_dbs = {}
    adapter._ensure_session_db = lambda: None
    monkeypatch.setattr(
        adapter, "_multiplex_profile_homes",
        lambda: {"broken": broken_home, "healthy": healthy_home},
    )

    await adapter._cleanup_stale_runtime_imports_once()
    reopened = SessionDB(healthy_home / "state.db")
    try:
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = 'healthy-stale'"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_gateway_runtime_import_cleanup_continues_after_profile_failure():
    class _DB:
        def __init__(self, result=0, error=None):
            self.result = result
            self.error = error
            self.calls = 0

        def cleanup_stale_runtime_imports(self):
            self.calls += 1
            if self.error is not None:
                raise self.error
            return self.result

    failed = _DB(error=RuntimeError("closed"))
    healthy = _DB(result=3)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._session_db = failed
    adapter._session_dbs = {"failed": failed, "healthy": healthy}
    adapter._ensure_session_db = lambda: failed

    assert await adapter._cleanup_stale_runtime_imports_once() == 3
    assert failed.calls == 1
    assert healthy.calls == 1
