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
