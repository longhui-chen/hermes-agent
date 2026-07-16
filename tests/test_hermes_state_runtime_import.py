import hashlib
import json
import sqlite3
import time

import pytest

import hermes_state

from hermes_state import (
    RuntimeImportConflict,
    RuntimeImportIncomplete,
    SessionDB,
)


def _stage(db, *, import_id="imp-1", chunk_index=0, messages=None, expected=2,
           target_session_id="imported-session"):
    return db.stage_completed_transcript_import(
        import_id=import_id,
        source="workbuddy",
        source_session_id="source-session",
        target_session_id=target_session_id,
        title="Imported chat",
        payload_sha256=hashlib.sha256(b"source-payload").hexdigest(),
        expected_message_count=expected,
        chunk_index=chunk_index,
        messages=messages or [{
            "source_id": f"m-{chunk_index}",
            "role": "user" if chunk_index == 0 else "assistant",
            "content": "hello" if chunk_index == 0 else "hi",
            "created_at": 1_700_000_000 + chunk_index,
        }],
    )


def test_completed_transcript_import_is_chunk_and_commit_idempotent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        first = _stage(db)
        assert first == {
            "import_id": "imp-1", "status": "staged", "next_chunk_index": 1,
            "staged_message_count": 1, "replayed": False,
        }
        assert _stage(db)["replayed"] is True
        _stage(db, chunk_index=1)

        committed = db.commit_completed_transcript_import("imp-1")
        assert committed["status"] == "completed"
        assert committed["message_count"] == 2
        assert committed["replayed"] is False
        assert db.commit_completed_transcript_import("imp-1")["replayed"] is True

        messages = db.get_messages_as_conversation("imported-session")
        assert [(m["role"], m["content"]) for m in messages] == [
            ("user", "hello"), ("assistant", "hi")
        ]
        session = db.get_session("imported-session")
        assert session["source"] == "import:workbuddy"
        assert session["message_count"] == 2
        runtime_meta = json.loads(session["model_config"])["_runtime_import"]
        assert "source_session_id" not in runtime_meta
        assert runtime_meta["source_session_id_sha256"] == hashlib.sha256(
            b"source-session"
        ).hexdigest()
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE platform_message_id IS NOT NULL"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_chunks WHERE import_id = 'imp-1'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_message_ids WHERE import_id = 'imp-1'"
        ).fetchone()[0] == 0

        assert db.delete_session("imported-session") is True
        assert db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = 'imported-session'"
        ).fetchone()[0] == 0
        # Only the non-sensitive idempotency receipt metadata remains.
        receipt = db._conn.execute(
            "SELECT status, normalized_sha256, source_session_id, title "
            "FROM runtime_imports WHERE import_id = 'imp-1'"
        ).fetchone()
        assert receipt["status"] == "completed"
        assert receipt["normalized_sha256"]
        assert receipt["source_session_id"] != "source-session"
        assert len(receipt["source_session_id"]) == 64
        assert receipt["title"] is None

        # Completed replay is valid only while the imported target still
        # exists; deletion is a tombstone conflict, never a false success.
        with pytest.raises(RuntimeImportConflict, match="missing or replaced"):
            db.commit_completed_transcript_import("imp-1")
        with pytest.raises(RuntimeImportConflict, match="missing or replaced"):
            _stage(db)
        with pytest.raises(RuntimeImportConflict):
            db.stage_completed_transcript_import(
                import_id="imp-1", source="workbuddy",
                source_session_id="different-source-session",
                target_session_id="imported-session", title="Imported chat",
                payload_sha256=hashlib.sha256(b"source-payload").hexdigest(),
                expected_message_count=2, chunk_index=0,
                messages=[{
                    "source_id": "m-0", "role": "user", "content": "hello",
                    "created_at": 1_700_000_000,
                }],
            )
    finally:
        db.close()


def test_incomplete_or_conflicting_import_never_changes_target(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db)
        with pytest.raises(RuntimeImportIncomplete):
            db.commit_completed_transcript_import("imp-1")
        assert db.get_session("imported-session") is None

        changed = [{
            "source_id": "changed", "role": "user", "content": "different",
            "created_at": 1_700_000_000,
        }]
        with pytest.raises(RuntimeImportConflict):
            _stage(db, messages=changed)
        assert db.get_session("imported-session") is None
    finally:
        db.close()


def test_runtime_import_rejects_existing_target_before_staging(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("imported-session", "api")
        with pytest.raises(RuntimeImportConflict, match="target_session_id already exists"):
            _stage(db)
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = 'imp-1'"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_chunks WHERE import_id = 'imp-1'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_runtime_import_rejects_non_completed_runtime_messages(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="system/tool/in-flight"):
            _stage(db, messages=[{
                "source_id": "tool-1", "role": "tool", "content": "secret output",
                "created_at": 1_700_000_000,
            }])
        assert db.get_session("imported-session") is None
    finally:
        db.close()


@pytest.mark.parametrize("message", [
    {"source_id": "m", "role": "user", "content": "Authorization: Bearer abcdefghijklmnop", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "api_key = sk-abcdefghijklmnop", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "-----BEGIN RSA PRIVATE KEY-----", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "ghp_abcdefghijklmnopqrst", "created_at": 1},
    {"source_id": "Authorization: Bearer abcdefghijklmnop", "role": "user",
     "content": "safe", "created_at": 1},
])
def test_runtime_import_rejects_every_credential_free_policy_category(tmp_path, message):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="forbidden"):
            _stage(db, expected=1, messages=[message])
        assert db.get_session("imported-session") is None
    finally:
        db.close()


def test_runtime_import_ignores_deep_unknown_fields_without_recursion(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    unknown = {}
    cursor = unknown
    for _ in range(2000):
        child = {}
        cursor["unknown"] = child
        cursor = child
    message = {"source_id": "m", "role": "user", "content": "safe",
               "created_at": 1, "extension": unknown}
    try:
        assert _stage(db, expected=1, messages=[message])["status"] == "staged"
    finally:
        db.close()


@pytest.mark.parametrize("content", [
    "token: <redacted>",
    "api_key=${OPENAI_API_KEY}",
    "password=changeme",
    "sk-example",
    "Authorization: Bearer redacted",
    '{"credentials": {}}',
])
def test_runtime_import_allows_non_secret_examples_and_placeholders(tmp_path, content):
    db = SessionDB(tmp_path / "state.db")
    try:
        result = _stage(
            db, expected=1,
            messages=[{"source_id": "m", "role": "user", "content": content,
                       "created_at": 1}],
        )
        assert result["status"] == "staged"
    finally:
        db.close()


def test_runtime_import_rejects_long_title_before_staging(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="Title too long"):
            db.stage_completed_transcript_import(
                import_id="long-title", source="workbuddy", source_session_id="source",
                target_session_id="target", title="x" * 101,
                payload_sha256=hashlib.sha256(b"payload").hexdigest(),
                expected_message_count=1, chunk_index=0,
                messages=[{"role": "user", "content": "safe", "created_at": 1}],
            )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_runtime_import_rejects_credential_in_title_before_staging(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="forbidden"):
            db.stage_completed_transcript_import(
                import_id="credential-title", source="workbuddy",
                source_session_id="source", target_session_id="target",
                title="Authorization: Bearer abcdefghijklmnop",
                payload_sha256=hashlib.sha256(b"payload").hexdigest(),
                expected_message_count=1, chunk_index=0,
                messages=[{"role": "user", "content": "safe", "created_at": 1}],
            )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_runtime_import_reserves_target_across_database_connections(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)
    try:
        _stage(first, import_id="first", expected=1)
        with pytest.raises(RuntimeImportConflict, match="reserved"):
            _stage(second, import_id="second", expected=1)
    finally:
        first.close()
        second.close()


def test_runtime_import_enforces_profile_aggregate_staging_quota(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db, import_id="first", expected=1, target_session_id="target-1")
        staged_bytes = db._conn.execute(
            "SELECT staged_bytes FROM runtime_imports WHERE import_id = 'first'"
        ).fetchone()[0]
        monkeypatch.setattr(
            hermes_state, "RUNTIME_IMPORT_MAX_TOTAL_STAGED_BYTES", staged_bytes
        )
        with pytest.raises(ValueError, match="profile staged transcripts"):
            _stage(db, import_id="second", expected=1, target_session_id="target-2")
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE staged_bytes > 0"
        ).fetchone()[0] == 1
    finally:
        db.close()


def test_runtime_import_accepts_unknown_source_timestamp_sentinel(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(
            db,
            expected=1,
            messages=[{
                "source_id": "unknown-time",
                "role": "user",
                "content": "timestamp was not exported",
                "created_at": 0,
            }],
        )
        assert db.commit_completed_transcript_import("imp-1")["status"] == "completed"
        timestamp = db._conn.execute(
            "SELECT timestamp FROM messages WHERE session_id = ?",
            ("imported-session",),
        ).fetchone()[0]
        assert timestamp == 0
    finally:
        db.close()


def test_runtime_import_rejects_duplicate_source_id_across_chunks(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db)
        duplicate = [{
            "source_id": "m-0", "role": "assistant", "content": "duplicate",
            "created_at": 1_700_000_001,
        }]
        with pytest.raises(RuntimeImportConflict, match="across chunks"):
            _stage(db, chunk_index=1, messages=duplicate)
        assert db.get_session("imported-session") is None
    finally:
        db.close()


@pytest.mark.parametrize("field", ["import_id", "source", "source_session_id", "target_session_id"])
@pytest.mark.parametrize("unsafe", ["../auth", "nested/session", "win\\session", ".."])
def test_runtime_import_rejects_path_unsafe_ids(tmp_path, field, unsafe):
    db = SessionDB(tmp_path / "state.db")
    try:
        values = {
            "import_id": "safe-import", "source": "workbuddy",
            "source_session_id": "source-session", "target_session_id": "target-session",
        }
        values[field] = unsafe
        with pytest.raises(ValueError, match="path-safe"):
            db.stage_completed_transcript_import(
                **values,
                title=None, payload_sha256=hashlib.sha256(b"x").hexdigest(),
                expected_message_count=1, chunk_index=0,
                messages=[{"role": "user", "content": "hi", "created_at": 1}],
            )
    finally:
        db.close()


def test_delete_legacy_unsafe_session_id_cannot_traverse_sessions_dir(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    outside = tmp_path / "auth.json"
    outside.write_text("must survive", encoding="utf-8")
    try:
        db.create_session("../auth", "legacy")
        assert db.delete_session("../auth", sessions_dir=sessions_dir) is True
        assert outside.read_text(encoding="utf-8") == "must survive"
    finally:
        db.close()


def test_runtime_import_chunk_iterator_never_calls_fetchall():
    rows = iter([{"messages_json": "[]"}, None])

    class Cursor:
        def fetchone(self):
            return next(rows)

        def fetchall(self):
            raise AssertionError("streaming iterator must not call fetchall")

    class Conn:
        def execute(self, *_args):
            return Cursor()

    assert list(SessionDB._iter_runtime_import_chunks(Conn(), "import")) == [
        {"messages_json": "[]"}
    ]


def test_expired_runtime_import_staging_is_cleaned_on_db_open(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    try:
        _stage(db, import_id="abandoned", expected=1)
        db._conn.execute(
            "UPDATE runtime_imports SET updated_at = ? WHERE import_id = ?",
            (time.time() - 25 * 60 * 60, "abandoned"),
        )
    finally:
        db.close()

    reopened = SessionDB(path)
    try:
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE import_id = 'abandoned'"
        ).fetchone()[0] == 0
        assert reopened._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_chunks WHERE import_id = 'abandoned'"
        ).fetchone()[0] == 0
    finally:
        reopened.close()


def test_db_open_tolerates_runtime_import_cleanup_failure(tmp_path, monkeypatch):
    def _locked(_self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(SessionDB, "cleanup_stale_runtime_imports", _locked)
    db = SessionDB(tmp_path / "state.db")
    try:
        assert db.get_session("missing") is None
    finally:
        db.close()


def test_profile_unload_cleanup_discards_all_unpublished_staging(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db, import_id="fresh", expected=1)
        assert db.discard_runtime_import_staging() == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE status = 'staging'"
        ).fetchone()[0] == 0
    finally:
        db.close()


def test_runtime_import_cleanup_is_bounded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for index in range(3):
            _stage(
                db, import_id=f"abandoned-{index}", expected=1,
                target_session_id=f"target-{index}",
            )
        db._conn.execute(
            "UPDATE runtime_imports SET updated_at = ?",
            (time.time() - 25 * 60 * 60,),
        )
        assert db.cleanup_stale_runtime_imports(limit=2) == 2
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE status = 'staging'"
        ).fetchone()[0] == 1
        assert db.cleanup_stale_runtime_imports(limit=2) == 1
    finally:
        db.close()


def test_runtime_import_cleanup_is_also_byte_bounded(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for index in range(3):
            _stage(
                db, import_id=f"abandoned-{index}", expected=1,
                target_session_id=f"target-{index}",
            )
        db._conn.execute(
            "UPDATE runtime_imports SET updated_at = ?",
            (time.time() - 25 * 60 * 60,),
        )
        one_import_bytes = db._conn.execute(
            "SELECT staged_bytes FROM runtime_imports LIMIT 1"
        ).fetchone()[0]
        assert db.cleanup_stale_runtime_imports(
            limit=8, max_bytes=one_import_bytes
        ) == 1
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports WHERE status = 'staging'"
        ).fetchone()[0] == 2
    finally:
        db.close()
