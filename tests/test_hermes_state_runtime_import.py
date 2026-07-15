import hashlib

import pytest

from hermes_state import (
    RuntimeImportConflict,
    RuntimeImportIncomplete,
    SessionDB,
)


def _stage(db, *, import_id="imp-1", chunk_index=0, messages=None, expected=2):
    return db.stage_completed_transcript_import(
        import_id=import_id,
        source="workbuddy",
        source_session_id="source-session",
        target_session_id="imported-session",
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

        # A lost stage response can still replay after commit without retaining
        # the raw source session identifier, while a different binding fails.
        assert _stage(db)["status"] == "completed"
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
