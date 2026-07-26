import asyncio
import hashlib
import json
import os
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
async def test_unsupported_platform_disables_v25_memory_import(monkeypatch):
    import tools.memory_tool as memory_tool

    monkeypatch.setattr(
        memory_tool, "portable_memory_import_supported", lambda: False
    )
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)
    headers = {"Authorization": "Bearer test-key"}

    async with TestClient(TestServer(app)) as cli:
        capabilities = await cli.get("/v1/capabilities", headers=headers)
        payload = await capabilities.json()
        assert payload["features"]["curated_memory_import"] is False
        assert payload["endpoints"]["curated_memory_import"]["enabled"] is False

        unauthenticated = await cli.post("/api/memory/import", json={})
        assert unauthenticated.status == 401
        unsupported = await cli.post(
            "/api/memory/import", json={}, headers=headers
        )
        assert unsupported.status == 501
        assert (await unsupported.json())["error"]["code"] == (
            "memory_import_unsupported"
        )


@pytest.mark.asyncio
async def test_memory_import_durability_race_returns_unsupported(monkeypatch):
    import tools.memory_tool as memory_tool

    class _Store:
        def import_replace(self, **_kwargs):
            raise memory_tool.MemoryImportUnsupported(
                "profile filesystem stopped supporting directory fsync"
            )

    monkeypatch.setattr(
        memory_tool, "portable_memory_import_supported", lambda: True
    )
    monkeypatch.setattr(
        memory_tool, "load_on_disk_store", lambda **_kwargs: _Store()
    )
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "durability-race",
        "mode": "replace",
        "target": "memory",
        "payload_sha256": hashlib.sha256(b"durability-race").hexdigest(),
        "entries": ["safe fact"],
    })

    response = await adapter._handle_memory_import(request)

    assert response.status == 501
    assert json.loads(response.text)["error"]["code"] == (
        "memory_import_unsupported"
    )


@pytest.mark.asyncio
async def test_capabilities_get_never_runs_mutating_hardlink_probe(
    tmp_path, monkeypatch
):
    import tools.memory_tool as memory_tool

    home = tmp_path / ".hermes"
    (home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    def unexpected_link(*_args, **_kwargs):
        raise AssertionError("capabilities GET attempted a hardlink probe")

    monkeypatch.setattr(memory_tool.os, "link", unexpected_link)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer test-key"},
        )

    assert response.status == 200
    assert not list(
        (home / "memories").glob(
            f"{memory_tool._IMPORT_LINK_PROBE_PREFIX}*"
        )
    )
    assert not list((home / "memories").glob("*.lock"))


@pytest.mark.asyncio
async def test_memory_import_hardlink_probe_failure_returns_501_before_state(
    tmp_path, monkeypatch
):
    import tools.memory_tool as memory_tool

    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    canonical = memories / "MEMORY.md"
    canonical.write_text("old fact", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    original_link = memory_tool.os.link

    def reject_probe(source, target, *args, **kwargs):
        if str(source).startswith(memory_tool._IMPORT_LINK_PROBE_PREFIX):
            raise OSError(memory_tool.errno.ENOTSUP, "hardlinks unavailable")
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(memory_tool.os, "link", reject_probe)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    response = await adapter._handle_memory_import(_DirectImportRequest({
        "import_id": "hardlink-unsupported",
        "mode": "replace",
        "target": "memory",
        "payload_sha256": hashlib.sha256(b"hardlink-unsupported").hexdigest(),
        "entries": ["safe fact"],
    }))

    assert response.status == 501
    assert json.loads(response.text)["error"]["code"] == "memory_import_unsupported"
    assert canonical.read_text(encoding="utf-8") == "old fact"
    assert not (memories / ".imports").exists()
    assert not list(memories.glob("*.displaced"))
    assert not list(memories.glob(f"{memory_tool._IMPORT_LINK_PROBE_PREFIX}*"))


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
    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda **_kwargs: store)
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

    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda **_kwargs: _Store())
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
@pytest.mark.parametrize("endpoint", ["capabilities", "memory-import"])
async def test_durability_probe_runs_off_loop_and_holds_unload_barrier(
    monkeypatch, endpoint
):
    import tools.memory_tool as memory_tool

    event_loop_thread = threading.get_ident()
    probe_threads = []
    started = threading.Event()
    release = threading.Event()

    def slow_probe():
        probe_threads.append(threading.get_ident())
        started.set()
        assert release.wait(2)
        return True

    class _Store:
        def import_replace(self, **_kwargs):
            return {"status": "completed"}

    monkeypatch.setattr(memory_tool, "portable_memory_import_supported", slow_probe)
    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda **_kwargs: _Store())
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/api/memory/import", adapter._handle_memory_import)
    app.router.add_post("/v1/profile/unload", adapter._handle_profile_unload)
    headers = {"Authorization": "Bearer test-key"}

    async with TestClient(TestServer(app)) as cli:
        if endpoint == "capabilities":
            probe_task = asyncio.create_task(
                cli.get("/v1/capabilities", headers=headers)
            )
        else:
            probe_task = asyncio.create_task(
                cli.post(
                    "/api/memory/import",
                    headers=headers,
                    json={
                        "import_id": "slow-probe",
                        "mode": "replace",
                        "target": "memory",
                        "payload_sha256": hashlib.sha256(b"slow-probe").hexdigest(),
                        "entries": ["safe fact"],
                    },
                )
            )
        assert await asyncio.to_thread(started.wait, 1)
        blocked = await cli.post("/v1/profile/unload", headers=headers)
        assert blocked.status == 409
        assert (await blocked.json())["active_imports"] == 1
        release.set()
        assert (await probe_task).status == 200

    assert probe_threads == [probe_threads[0]]
    assert probe_threads[0] != event_loop_thread


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
async def test_uncached_multiplex_session_imports_use_each_profile_database(
    tmp_path, monkeypatch
):
    import hermes_state
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"
    first_home = tmp_path / "profiles" / "first"
    second_home = tmp_path / "profiles" / "second"
    first_home.mkdir(parents=True)
    second_home.mkdir(parents=True)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", default_home / "state.db")
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    async def import_one(profile_home, suffix):
        token = set_hermes_home_override(profile_home)
        try:
            stage = _DirectImportRequest({
                "import_id": f"import-{suffix}",
                "operation": "stage",
                "source": "workbuddy",
                "source_session_id": f"source-{suffix}",
                "target_session_id": f"target-{suffix}",
                "title": None,
                "payload_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
                "expected_message_count": 1,
                "chunk_index": 0,
                "messages": [{
                    "source_id": f"message-{suffix}",
                    "role": "user",
                    "content": f"profile {suffix}",
                    "created_at": 1,
                }],
            })
            stage["hermes_profile_home"] = str(profile_home)
            assert (await adapter._handle_session_import(stage)).status == 200
            commit = _DirectImportRequest({
                "import_id": f"import-{suffix}",
                "operation": "commit",
            })
            commit["hermes_profile_home"] = str(profile_home)
            assert (await adapter._handle_session_import(commit)).status == 200
        finally:
            reset_hermes_home_override(token)

    try:
        await import_one(first_home, "first")
        await import_one(second_home, "second")
        first = SessionDB(first_home / "state.db")
        second = SessionDB(second_home / "state.db")
        try:
            assert first.get_session("target-first") is not None
            assert first.get_session("target-second") is None
            assert second.get_session("target-second") is not None
            assert second.get_session("target-first") is None
            assert not (default_home / "state.db").exists()
        finally:
            first.close()
            second.close()
    finally:
        for db in set(adapter._session_dbs.values()):
            db.close()


@pytest.mark.asyncio
async def test_memory_import_constructs_store_inside_worker_thread(monkeypatch):
    import tools.memory_tool as memory_tool

    event_loop_thread = threading.get_ident()
    constructor_threads = []

    class _Store:
        def import_replace(self, **_kwargs):
            return {"status": "completed"}

    def load_store(**_kwargs):
        constructor_threads.append(threading.get_ident())
        return _Store()

    monkeypatch.setattr(memory_tool, "load_on_disk_store", load_store)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    response = await adapter._handle_memory_import(_DirectImportRequest({
        "import_id": "worker-construction",
        "mode": "replace",
        "target": "memory",
        "payload_sha256": hashlib.sha256(b"worker-construction").hexdigest(),
        "entries": ["safe fact"],
    }))

    assert response.status == 200
    assert constructor_threads
    assert constructor_threads[0] != event_loop_thread


@pytest.mark.parametrize("unsafe_kind", ["fifo", "oversize"])
@pytest.mark.asyncio
async def test_memory_import_constructor_rejects_unsafe_live_file_without_blocking_loop(
    tmp_path, monkeypatch, unsafe_kind
):
    home = tmp_path / ".hermes"
    memory_path = home / "memories" / "MEMORY.md"
    memory_path.parent.mkdir(parents=True)
    if unsafe_kind == "fifo":
        os.mkfifo(memory_path)
    else:
        import tools.memory_tool as memory_tool

        memory_path.write_bytes(
            b"x" * (memory_tool.MAX_CURATED_MEMORY_FILE_BYTES + 1)
        )
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": f"constructor-{unsafe_kind}",
        "mode": "replace",
        "target": "memory",
        "payload_sha256": hashlib.sha256(unsafe_kind.encode()).hexdigest(),
        "entries": ["safe fact"],
    })
    heartbeat = asyncio.create_task(asyncio.sleep(0))
    response = await asyncio.wait_for(adapter._handle_memory_import(request), 1)
    await heartbeat

    assert response.status == 409
    assert json.loads(response.text)["error"]["code"] == "memory_import_conflict"


@pytest.mark.asyncio
async def test_cancelled_memory_import_holds_barrier_during_store_construction(
    monkeypatch,
):
    import tools.memory_tool as memory_tool

    started = threading.Event()
    release = threading.Event()

    class _Store:
        def import_replace(self, **_kwargs):
            return {"status": "completed"}

    def load_store(**_kwargs):
        started.set()
        assert release.wait(2)
        return _Store()

    monkeypatch.setattr(memory_tool, "load_on_disk_store", load_store)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "cancel-construction",
        "mode": "replace",
        "target": "memory",
        "payload_sha256": hashlib.sha256(b"cancel-construction").hexdigest(),
        "entries": ["safe fact"],
    })
    task = asyncio.create_task(adapter._handle_memory_import(request))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    blocked = await adapter._handle_profile_unload(_DirectImportRequest(None))
    assert blocked.status == 409
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await adapter._handle_profile_unload(_DirectImportRequest(None))).status == 200


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

    monkeypatch.setattr(memory_tool, "load_on_disk_store", lambda **_kwargs: _Store())
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

    def slow_open(path, **kwargs):
        started.set()
        assert release.wait(2)
        return real_session_db(path, **kwargs)

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


@pytest.mark.asyncio
async def test_session_import_rejects_profile_state_db_symlink(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    victim_home = tmp_path / "profiles" / "victim"
    victim = SessionDB(victim_home / "state.db")
    profile_home = tmp_path / "profiles" / "attacker"
    profile_home.mkdir(parents=True)
    (profile_home / "state.db").symlink_to(victim_home / "state.db")
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "symlink-import", "operation": "stage",
        "source": "workbuddy", "source_session_id": "source",
        "target_session_id": "target", "title": None,
        "payload_sha256": hashlib.sha256(b"symlink-import").hexdigest(),
        "expected_message_count": 1, "chunk_index": 0,
        "messages": [{"role": "user", "content": "private", "created_at": 1}],
    })
    request["hermes_profile_home"] = str(profile_home)
    token = set_hermes_home_override(str(profile_home))
    try:
        response = await adapter._handle_session_import(request)
        assert response.status == 500
        assert json.loads(response.text)["error"]["code"] == "runtime_import_failed"
        assert (profile_home / "state.db").is_symlink()
        assert victim._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
    finally:
        reset_hermes_home_override(token)
        victim.close()


@pytest.mark.asyncio
async def test_session_import_rejects_profile_home_symlink(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    victim_home = tmp_path / "profiles" / "victim"
    victim = SessionDB(victim_home / "state.db")
    linked_home = tmp_path / "profiles" / "linked"
    linked_home.symlink_to(victim_home, target_is_directory=True)
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "linked-home", "operation": "stage",
        "source": "workbuddy", "source_session_id": "source",
        "target_session_id": "target", "title": None,
        "payload_sha256": hashlib.sha256(b"linked-home").hexdigest(),
        "expected_message_count": 1, "chunk_index": 0,
        "messages": [{"role": "user", "content": "private", "created_at": 1}],
    })
    request["hermes_profile_home"] = str(linked_home)
    token = set_hermes_home_override(str(linked_home))
    try:
        response = await adapter._handle_session_import(request)
        assert response.status == 500
        assert victim._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
        assert adapter._session_dbs == {}
    finally:
        reset_hermes_home_override(token)
        victim.close()


@pytest.mark.asyncio
async def test_session_import_rejects_hardlinked_profile_state_db(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    victim_home = tmp_path / "profiles" / "victim"
    victim = SessionDB(victim_home / "state.db")
    victim.close()
    profile_home = tmp_path / "profiles" / "attacker"
    profile_home.mkdir(parents=True)
    os.link(victim_home / "state.db", profile_home / "state.db")
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    request = _DirectImportRequest({
        "import_id": "hardlink-import", "operation": "stage",
        "source": "workbuddy", "source_session_id": "source",
        "target_session_id": "target", "title": None,
        "payload_sha256": hashlib.sha256(b"hardlink-import").hexdigest(),
        "expected_message_count": 1, "chunk_index": 0,
        "messages": [{"role": "user", "content": "private", "created_at": 1}],
    })
    request["hermes_profile_home"] = str(profile_home)
    token = set_hermes_home_override(str(profile_home))
    try:
        response = await adapter._handle_session_import(request)
        assert response.status == 500
        assert adapter._session_dbs == {}
        reopened = SessionDB(victim_home / "state.db", read_only=True)
        try:
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM runtime_imports"
            ).fetchone()[0] == 0
        finally:
            reopened.close()
    finally:
        reset_hermes_home_override(token)


def test_session_db_open_rejects_leaf_replacement_before_initialization(
    tmp_path, monkeypatch
):
    import gateway.platforms.api_server as api_server

    victim_path = tmp_path / "victim.db"
    victim = api_server.sqlite3.connect(victim_path)
    victim.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    victim.execute("INSERT INTO marker VALUES ('unchanged')")
    victim.commit()
    victim.close()
    victim_before = victim_path.read_bytes()
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    real_connect = api_server.sqlite3.connect
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    def replace_after_connect(path, *args, **kwargs):
        connection = real_connect(path, *args, **kwargs)
        state_path = profile_home / "state.db"
        moved_path = profile_home / "state.db.original"
        state_path.rename(moved_path)
        state_path.symlink_to(victim_path)
        return connection

    monkeypatch.setattr(api_server.sqlite3, "connect", replace_after_connect)

    assert adapter._ensure_session_db(profile_home) is None
    assert adapter._session_dbs == {}
    assert victim_path.read_bytes() == victim_before
    check = real_connect(victim_path)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "unchanged"
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'runtime_imports'"
        ).fetchone()[0] == 0
    finally:
        check.close()


def test_session_db_open_rejects_hardlink_added_during_connection(
    tmp_path, monkeypatch
):
    import gateway.platforms.api_server as api_server

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    linked_path = tmp_path / "other-profile-state.db"
    real_connect = api_server.sqlite3.connect
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    def add_hardlink_after_connect(path, *args, **kwargs):
        connection = real_connect(path, *args, **kwargs)
        os.link(profile_home / "state.db", linked_path)
        return connection

    monkeypatch.setattr(api_server.sqlite3, "connect", add_hardlink_after_connect)

    assert adapter._ensure_session_db(profile_home) is None
    assert adapter._session_dbs == {}
    assert linked_path.stat().st_nlink == 2


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason="Linux procfd anchoring is the production security boundary",
)
def test_linux_procfd_aba_never_touches_victim(
    tmp_path, monkeypatch
):
    import gateway.platforms.api_server as api_server

    victim_path = tmp_path / "victim.db"
    victim = api_server.sqlite3.connect(victim_path)
    victim.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    victim.execute("INSERT INTO marker VALUES ('unchanged')")
    victim.commit()
    victim.close()
    victim_before = victim_path.read_bytes()

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    real_connect = api_server.sqlite3.connect
    observed_paths = []
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    def aba_during_connect(path, *args, **kwargs):
        path = os.fspath(path)
        observed_paths.append(path)
        assert path.startswith("/proc/self/fd/")
        state_path = profile_home / "state.db"
        moved_path = profile_home / "state.db.original"
        state_path.rename(moved_path)
        state_path.symlink_to(victim_path)
        try:
            # Connecting through procfd must still bind SQLite to the inode
            # opened before the canonical pathname was redirected.
            return real_connect(path, *args, **kwargs)
        finally:
            state_path.unlink()
            moved_path.rename(state_path)

    monkeypatch.setattr(api_server.sqlite3, "connect", aba_during_connect)

    # Depending on SQLite's resolved WAL sidecar path, the original database
    # may continue or fail closed after the double rename. Either result is
    # acceptable; publishing a connection to the victim is not.
    db = adapter._ensure_session_db(profile_home)
    if db is not None:
        db.close()
    assert observed_paths and observed_paths[0].startswith("/proc/self/fd/")
    assert victim_path.read_bytes() == victim_before
    check = real_connect(victim_path)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "unchanged"
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'runtime_imports'"
        ).fetchone()[0] == 0
    finally:
        check.close()


@pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(),
    reason="Linux procfd anchoring is the production security boundary",
)
def test_linux_unlinked_procfd_fails_closed_without_touching_replacement(
    tmp_path, monkeypatch
):
    import gateway.platforms.api_server as api_server

    victim_path = tmp_path / "victim.db"
    victim = api_server.sqlite3.connect(victim_path)
    victim.execute("CREATE TABLE marker (value TEXT NOT NULL)")
    victim.execute("INSERT INTO marker VALUES ('unchanged')")
    victim.commit()
    victim.close()
    victim_before = victim_path.read_bytes()

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    real_connect = api_server.sqlite3.connect
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    def unlink_and_replace(path, *args, **kwargs):
        path = os.fspath(path)
        assert path.startswith("/proc/self/fd/")
        state_path = profile_home / "state.db"
        state_path.unlink()
        state_path.symlink_to(victim_path)
        # The deleted procfd target cannot be reopened by SQLite. This must
        # fail instead of retrying the now-victim-controlled canonical path.
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(api_server.sqlite3, "connect", unlink_and_replace)

    assert adapter._ensure_session_db(profile_home) is None
    assert adapter._session_dbs == {}
    assert victim_path.read_bytes() == victim_before
    check = real_connect(victim_path)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "unchanged"
        assert check.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'runtime_imports'"
        ).fetchone()[0] == 0
    finally:
        check.close()


def test_cached_session_db_is_evicted_after_profile_directory_replacement(
    tmp_path,
):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    old_home = tmp_path / "profiles" / "coder-old"
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )

    first = adapter._ensure_session_db(profile_home)
    assert first is not None
    profile_home.rename(old_home)
    profile_home.mkdir()

    second = adapter._ensure_session_db(profile_home)
    assert second is not None
    assert second is not first
    assert first._conn is None
    second.stage_completed_transcript_import(
        import_id="new-generation",
        source="workbuddy",
        source_session_id="source",
        target_session_id="target",
        title=None,
        payload_sha256=hashlib.sha256(b"new-generation").hexdigest(),
        expected_message_count=1,
        chunk_index=0,
        messages=[{"role": "user", "content": "new", "created_at": 1}],
    )
    second.close()

    old = SessionDB(old_home / "state.db", read_only=True)
    new = SessionDB(profile_home / "state.db", read_only=True)
    try:
        assert old._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = ?",
            ("new-generation",),
        ).fetchone()[0] == 0
        assert new._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = ?",
            ("new-generation",),
        ).fetchone()[0] == 1
    finally:
        old.close()
        new.close()


@pytest.mark.asyncio
async def test_uncached_cleanup_rejects_profile_state_db_symlink(
    tmp_path, monkeypatch
):
    victim_home = tmp_path / "profiles" / "victim"
    victim = SessionDB(victim_home / "state.db")
    victim.stage_completed_transcript_import(
        import_id="victim-staging", source="marvis", source_session_id="source",
        target_session_id="target", title=None,
        payload_sha256=hashlib.sha256(b"victim-staging").hexdigest(),
        expected_message_count=1, chunk_index=0,
        messages=[{"role": "user", "content": "private", "created_at": 0}],
    )
    victim._conn.execute(
        "UPDATE runtime_imports SET updated_at = ?",
        (time.time() - 25 * 60 * 60,),
    )
    victim.close()

    profile_home = tmp_path / "profiles" / "attacker"
    profile_home.mkdir(parents=True)
    (profile_home / "state.db").symlink_to(victim_home / "state.db")
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    adapter._ensure_session_db = lambda: None
    monkeypatch.setattr(
        adapter, "_multiplex_profile_homes", lambda: {"attacker": profile_home}
    )

    assert await adapter._cleanup_stale_runtime_imports_once() == 0
    reopened = SessionDB(victim_home / "state.db", read_only=True)
    try:
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = ?",
            ("victim-staging",),
        ).fetchone()[0] == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_first_session_db_open_is_off_loop_and_published_once(
    tmp_path, monkeypatch
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    event_loop_thread = threading.get_ident()
    open_threads = []
    open_calls = 0
    stage_calls = []
    started = threading.Event()
    release = threading.Event()

    class _DB:
        def stage_completed_transcript_import(self, **kwargs):
            stage_calls.append(kwargs["import_id"])
            return {"status": "staged"}

    db = _DB()

    def slow_open(_profile_home, *, create=True):
        nonlocal open_calls
        assert create is True
        open_calls += 1
        open_threads.append(threading.get_ident())
        started.set()
        assert release.wait(10)
        return db

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    monkeypatch.setattr(adapter, "_open_profile_session_db", slow_open)
    monkeypatch.setattr(
        adapter, "_profile_session_db_is_current", lambda *_args: True
    )

    def request(import_id):
        value = _DirectImportRequest({
            "import_id": import_id, "operation": "stage", "source": "workbuddy",
            "source_session_id": import_id, "target_session_id": import_id,
            "title": None, "payload_sha256": hashlib.sha256(import_id.encode()).hexdigest(),
            "expected_message_count": 1, "chunk_index": 0,
            "messages": [{"role": "user", "content": "safe", "created_at": 1}],
        })
        value["hermes_profile_home"] = str(profile_home)
        return value

    token = set_hermes_home_override(str(profile_home))
    try:
        first = asyncio.create_task(adapter._handle_session_import(request("first")))
        second = asyncio.create_task(adapter._handle_session_import(request("second")))
        assert await asyncio.to_thread(started.wait, 2)
        assert not first.done() and not second.done()
        release.set()
        responses = await asyncio.gather(first, second)
    finally:
        reset_hermes_home_override(token)
        release.set()

    assert [response.status for response in responses] == [200, 200]
    assert open_calls == 1
    assert open_threads == [open_threads[0]]
    assert open_threads[0] != event_loop_thread
    assert sorted(stage_calls) == ["first", "second"]
    assert list(adapter._session_dbs.values()) == [db]


@pytest.mark.asyncio
async def test_cancelled_unload_holds_barrier_until_staging_cleanup_exits(tmp_path):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    started = threading.Event()
    release = threading.Event()
    close_started = threading.Event()
    close_release = threading.Event()
    events = []

    class _DB:
        closed = False

        def discard_runtime_import_staging(self):
            events.append("discard-start")
            started.set()
            assert release.wait(10)
            events.append("discard-end")

        def close(self):
            events.append("close-start")
            close_started.set()
            assert close_release.wait(10)
            self.closed = True
            events.append("close-end")

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

    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    db = _DB()
    adapter._session_dbs[adapter._profile_home_key(profile_home)] = db
    task = asyncio.create_task(adapter._handle_profile_unload(_Request()))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert adapter._begin_runtime_import_operation(profile_home) is None

    release.set()
    assert await asyncio.to_thread(close_started.wait, 2)
    assert not task.done()
    assert adapter._begin_runtime_import_operation(profile_home) is None
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert db.closed is True
    assert events == ["discard-start", "discard-end", "close-start", "close-end"]
    key = adapter._begin_runtime_import_operation(profile_home)
    assert key == adapter._profile_home_key(profile_home)
    adapter._end_runtime_import_operation(key)


@pytest.mark.parametrize("sidecar_suffix", ["-wal", "-shm", "-journal"])
def test_open_profile_session_db_rejects_symlinked_sidecar(tmp_path, sidecar_suffix):
    profile_home = tmp_path / "victim"
    profile_home.mkdir()
    foreign = tmp_path / "other-profile-file"
    foreign.write_bytes(b"")
    (profile_home / f"state.db{sidecar_suffix}").symlink_to(foreign)

    with pytest.raises(RuntimeError, match="private regular file"):
        ZetAgentAdapter._open_profile_session_db(profile_home)


def test_open_profile_session_db_rejects_hardlinked_sidecar(tmp_path):
    profile_home = tmp_path / "victim"
    profile_home.mkdir()
    foreign = tmp_path / "other-profile-journal"
    foreign.write_bytes(b"")
    os.link(foreign, profile_home / "state.db-journal")

    with pytest.raises(RuntimeError, match="private regular file"):
        ZetAgentAdapter._open_profile_session_db(profile_home)


def test_open_profile_session_db_accepts_stale_regular_sidecar(tmp_path):
    profile_home = tmp_path / "victim"
    profile_home.mkdir()
    (profile_home / "state.db-wal").write_bytes(b"")

    db = ZetAgentAdapter._open_profile_session_db(profile_home)
    try:
        assert db is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_ensure_session_db_async_runs_off_event_loop():
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    seen = {}
    sentinel = object()

    def _record():
        seen["thread"] = threading.get_ident()
        return sentinel

    adapter._ensure_session_db = _record
    result = await adapter._ensure_session_db_async()
    assert result is sentinel
    assert seen["thread"] != threading.get_ident()
