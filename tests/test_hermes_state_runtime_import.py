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
           target_session_id="imported-session", owner_principal=""):
    return db.stage_completed_transcript_import(
        import_id=import_id,
        source="workbuddy",
        source_session_id="source-session",
        target_session_id=target_session_id,
        owner_principal=owner_principal,
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
        assert session["ended_at"] is not None
        assert session["end_reason"] == "import_completed"
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


def test_owned_import_binds_target_and_completed_receipt_to_owner(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db, expected=1, owner_principal="iam:alice")
        db.commit_completed_transcript_import("imp-1")
        target = db._conn.execute(
            "SELECT user_id, model_config FROM sessions WHERE id = 'imported-session'"
        ).fetchone()
        assert target["user_id"] == "iam:alice"
        assert json.loads(target["model_config"])["_runtime_import"]["owner_principal"] == "iam:alice"
        with pytest.raises(RuntimeImportConflict, match="different metadata"):
            _stage(db, expected=1, owner_principal="iam:bob")
    finally:
        db.close()


def test_runtime_import_owner_column_reconciles_on_an_existing_database(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.close()
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE runtime_imports RENAME TO runtime_imports_old")
    conn.execute(
        "CREATE TABLE runtime_imports (import_id TEXT PRIMARY KEY, source TEXT NOT NULL, "
        "source_session_id TEXT NOT NULL, target_session_id TEXT NOT NULL, title TEXT, "
        "payload_sha256 TEXT NOT NULL, expected_message_count INTEGER NOT NULL, "
        "source_message_ids_json TEXT, source_total_rows INTEGER, next_chunk_index INTEGER NOT NULL DEFAULT 0, "
        "staged_message_count INTEGER NOT NULL DEFAULT 0, staged_bytes INTEGER NOT NULL DEFAULT 0, "
        "status TEXT NOT NULL DEFAULT 'staging', normalized_sha256 TEXT, created_at REAL NOT NULL, "
        "updated_at REAL NOT NULL, completed_at REAL)"
    )
    conn.execute("DROP TABLE runtime_imports_old")
    conn.commit()
    conn.close()
    upgraded = SessionDB(path)
    try:
        columns = {row["name"] for row in upgraded._conn.execute("PRAGMA table_info(runtime_imports)")}
        assert "owner_principal" in columns
    finally:
        upgraded.close()


def test_completed_transcript_can_reopen_and_remains_retention_eligible(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(db)
        _stage(db, chunk_index=1)
        db.commit_completed_transcript_import("imp-1")
        imported = db.get_session("imported-session")
        assert imported["ended_at"] is not None
        assert imported["end_reason"] == "import_completed"

        db.reopen_session("imported-session")
        reopened = db.get_session("imported-session")
        assert reopened["ended_at"] is None
        assert reopened["end_reason"] is None
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, "imported-session"),
        )
        assert db.prune_sessions(older_than_days=90) == 0

        db.end_session("imported-session", "resumed_after_import")
        assert db.prune_sessions(older_than_days=90) == 1
        assert db.get_session("imported-session") is None
    finally:
        db.close()


def test_normalized_sha_is_independent_of_chunks_and_source_ids(tmp_path):
    messages = [
        {"source_id": "one-a", "role": "user", "content": "hello", "created_at": 1},
        {"source_id": "one-b", "role": "assistant", "content": "hi", "created_at": 2},
    ]
    canonical = [
        {"role": "user", "content": "hello", "timestamp": 1.0},
        {"role": "assistant", "content": "hi", "timestamp": 2.0},
    ]
    expected_digest = hashlib.sha256(json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    db = SessionDB(tmp_path / "state.db")
    try:
        db.stage_completed_transcript_import(
            import_id="one-chunk",
            source="workbuddy",
            source_session_id="source-one",
            target_session_id="target-one",
            title=None,
            payload_sha256=hashlib.sha256(b"one-chunk").hexdigest(),
            expected_message_count=2,
            chunk_index=0,
            messages=messages,
        )
        one_chunk = db.commit_completed_transcript_import("one-chunk")

        for chunk_index, message in enumerate(messages):
            equivalent = dict(message)
            equivalent["source_id"] = f"different-{chunk_index}"
            db.stage_completed_transcript_import(
                import_id="two-chunks",
                source="workbuddy",
                source_session_id="source-two",
                target_session_id="target-two",
                title=None,
                payload_sha256=hashlib.sha256(b"two-chunks").hexdigest(),
                expected_message_count=2,
                chunk_index=chunk_index,
                messages=[equivalent],
            )
        two_chunks = db.commit_completed_transcript_import("two-chunks")

        assert one_chunk["normalized_sha256"] == expected_digest
        assert two_chunks["normalized_sha256"] == expected_digest
        assert db.get_messages_as_conversation("target-one") == (
            db.get_messages_as_conversation("target-two")
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
    {"source_id": "m-basic", "role": "user",
     "content": "Authorization: Basic dXNlcjpwYXNzd29yZA==", "created_at": 1},
    {"source_id": "m-json", "role": "user",
     "content": '{"Authorization": "Bearer abcdefghijklmnop"}', "created_at": 1},
    {"source_id": "m-js", "role": "user",
     "content": "{'Authorization': 'Bearer abcdefghijklmnop'}", "created_at": 1},
    {"source_id": "m-escaped", "role": "user",
     "content": r'{\"Authorization\": \"Bearer abcdefghijklmnop\"}', "created_at": 1},
    {"source_id": "m-equals", "role": "user",
     "content": "Authorization=Bearer abcdefghijklmnop", "created_at": 1},
    {"source_id": "m-spaced-key", "role": "user",
     "content": "api key = abcdefghijklmnop", "created_at": 1},
    {"source_id": "m-decoded-unicode-space", "role": "user",
     "content": "Authorization:\u0020Bearer\u0020abcdefghijklmnop", "created_at": 1},
    {"source_id": "m-literal-json-escapes", "role": "user",
     "content": r'{\"Authorization\":\u0020\"Bearer\u0020abcdefghijklmnop\"}', "created_at": 1},
    {"source_id": "m", "role": "user",
     "content": r"Authorization\u005cu003a\u005cu0020Bearer\u005cu0020abcdefghijklmnop", "created_at": 1},
    {"source_id": "m-placeholder-substring", "role": "user",
     "content": "Authorization: Bearer abcdefghexamplehijklmnop", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "api_key = sk-abcdefghijklmnop", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "-----BEGIN RSA PRIVATE KEY-----", "created_at": 1},
    {"source_id": "m-pgp", "role": "user", "content": "-----BEGIN PGP PRIVATE KEY BLOCK-----", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "ghp_abcdefghijklmnopqrst", "created_at": 1},
    {"source_id": "m", "role": "user", "content":
     "xoxb-123456789012-123456789012-abcdefghijklmnopqrstuvwxyzABCD", "created_at": 1},
    {"source_id": "m", "role": "user", "content": "AIza" + "A" * 35, "created_at": 1},
    {"source_id": "m", "role": "user", "content":
     "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlX3ZhbHVl", "created_at": 1},
    {"source_id": "m", "role": "user", "content":
     "https://api.example.test/v1/items?access_token=abcdefghijklmnop", "created_at": 1},
    {"source_id": "Authorization: Bearer abcdefghijklmnop", "role": "user",
     "content": "safe", "created_at": 1},
])
def test_runtime_import_rejects_every_credential_free_policy_category(tmp_path, message):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(ValueError, match="forbidden"):
            _stage(db, expected=1, messages=[message])
        assert db.get_session("imported-session") is None
        assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        assert db._conn.execute("SELECT COUNT(*) FROM runtime_imports").fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_chunks"
        ).fetchone()[0] == 0
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_import_message_ids"
        ).fetchone()[0] == 0
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
    "Authorization=Bearer redacted",
    "api key = ${OPENAI_API_KEY}",
    r'{\"Authorization\":\u0020\"Bearer\u0020${ACCESS_TOKEN}\"}',
    '{"Authorization": "Bearer redacted"}',
    "{'Authorization': 'Bearer ${ACCESS_TOKEN}'}",
    '{"credentials": {}}',
    "https://api.example.test/v1/items?access_token=${ACCESS_TOKEN}",
    "https://api.example.test/v1/items?api_key=redacted",
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


@pytest.mark.parametrize(
    ("field", "credential"),
    [
        (field, credential)
        for field in ("import_id", "source", "source_session_id", "target_session_id")
        for credential in (
            "sk-1234567890abcdefghij",
            "github_pat_abcdefghijklmnopqrstuvwxyz1234",
        )
    ],
)
def test_runtime_import_rejects_credential_metadata_before_staging(
    tmp_path, field, credential
):
    db = SessionDB(tmp_path / "state.db")
    try:
        values = {
            "import_id": "safe-import",
            "source": "workbuddy",
            "source_session_id": "source-session",
            "target_session_id": "target-session",
        }
        values[field] = credential
        with pytest.raises(ValueError, match=f"{field} contains forbidden"):
            db.stage_completed_transcript_import(
                **values,
                title=None,
                payload_sha256=hashlib.sha256(b"payload").hexdigest(),
                expected_message_count=1,
                chunk_index=0,
                messages=[{"role": "user", "content": "safe", "created_at": 1}],
            )
        assert db._conn.execute(
            "SELECT COUNT(*) FROM runtime_imports"
        ).fetchone()[0] == 0
        assert db.get_session(values["target_session_id"]) is None
    finally:
        db.close()


def test_runtime_import_allows_placeholder_import_id(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        result = _stage(db, import_id="sk-example", expected=1)
        assert result["status"] == "staged"
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


def test_unknown_import_timestamp_falls_back_for_recency_but_stays_raw(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        now = time.time()
        for index in range(50):
            session_id = f"normal-{index:02d}"
            db.create_session(session_id, "cli")
            db.append_message(session_id, "user", f"normal {index}")
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (now - 100 + index, session_id),
            )

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
        db.commit_completed_transcript_import("imp-1")

        page = db.list_sessions_rich(limit=50, order_by_last_active=True)
        imported = db._get_session_rich_row("imported-session")
        session = db.get_session("imported-session")
        raw_timestamp = db._conn.execute(
            "SELECT timestamp FROM messages WHERE session_id = ?",
            ("imported-session",),
        ).fetchone()[0]

        assert page[0]["id"] == "imported-session"
        assert imported["last_active"] == session["started_at"]
        assert db.search_sessions(limit=1)[0]["id"] == "imported-session"
        assert raw_timestamp == 0
    finally:
        db.close()


def test_import_preview_uses_canonical_message_order_when_timestamp_is_unknown(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        _stage(
            db,
            expected=2,
            messages=[
                {
                    "source_id": "first",
                    "role": "user",
                    "content": "first in the exported transcript",
                    "created_at": 100,
                },
                {
                    "source_id": "second",
                    "role": "user",
                    "content": "later message with unknown time",
                    "created_at": 0,
                },
            ],
        )
        db.commit_completed_transcript_import("imp-1")

        listed = db.list_sessions_rich(limit=1, order_by_last_active=True)[0]
        assert listed["preview"] == "first in the exported transcript"
    finally:
        db.close()


def test_unknown_import_timestamp_keeps_compression_tip_and_chain_recent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("root", "cli")
        db.end_session("root", end_reason="compression")
        db.create_session("stale-child", "cli", parent_session_id="root")
        db.append_message("stale-child", "user", "stale sibling")
        db._conn.execute(
            "UPDATE sessions SET started_at = 1, ended_at = 2 WHERE id = 'stale-child'"
        )
        db._conn.execute(
            "UPDATE messages SET timestamp = 1 WHERE session_id = 'stale-child'"
        )

        _stage(
            db,
            expected=1,
            messages=[{
                "source_id": "unknown-time",
                "role": "user",
                "content": "recent imported continuation",
                "created_at": 0,
            }],
        )
        db.commit_completed_transcript_import("imp-1")
        db._conn.execute(
            "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
            ("root", "imported-session"),
        )

        assert db.get_compression_tip("root") == "imported-session"
        listed = db.list_sessions_rich(limit=1, order_by_last_active=True)
        assert listed[0]["id"] == "imported-session"
        assert listed[0]["preview"] == "recent imported continuation"
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
@pytest.mark.parametrize(
    "unsafe",
    ["../auth", "nested/session", "win\\session", "..", "*", "?", "[session]"],
)
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


def test_delete_legacy_glob_session_id_cannot_remove_other_request_dumps(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    other_dump = sessions_dir / "request_dump_other_001.json"
    other_dump.write_text("must survive", encoding="utf-8")
    try:
        db.create_session("*", "legacy")
        assert db.delete_session("*", sessions_dir=sessions_dir) is True
        assert other_dump.read_text(encoding="utf-8") == "must survive"
    finally:
        db.close()


def test_delete_session_with_long_legit_id_removes_transcript_files(tmp_path):
    """The API server accepts session ids up to 256 chars
    (HermesAPIServer._MAX_SESSION_HEADER_LEN in gateway/platforms/api_server.py),
    well past the 128-char default that guards the runtime-import identifiers.
    Cleanup must not silently skip real, glob-free ids in that 129..256
    range — a prior fix reused the 128-char validator here and regressed
    long-id cleanup: the DB row deleted but transcript files (.json /
    .jsonl / request_dump_*) leaked on disk forever."""
    db = SessionDB(tmp_path / "state.db")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    long_id = "s" * 200
    try:
        db.create_session(long_id, "cli")
        (sessions_dir / f"{long_id}.json").write_text("{}", encoding="utf-8")
        (sessions_dir / f"{long_id}.jsonl").write_text("", encoding="utf-8")
        dump = sessions_dir / f"request_dump_{long_id}_001.json"
        dump.write_text("{}", encoding="utf-8")

        assert db.delete_session(long_id, sessions_dir=sessions_dir) is True

        assert not (sessions_dir / f"{long_id}.json").exists()
        assert not (sessions_dir / f"{long_id}.jsonl").exists()
        assert not dump.exists()
    finally:
        db.close()


def test_remove_session_files_rejects_id_past_api_server_ceiling(tmp_path):
    """An id longer than the API server ever admits (>256) falls outside
    every real entry point. Cleanup must refuse it rather than trust an
    unbounded-length string into a filesystem path/glob, and must not
    raise — a filesystem hiccup or a malformed legacy row should never
    block the DB-side delete. The validator rejects before touching the
    filesystem, so no 300+ char path component is ever created."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    too_long_id = "s" * 300

    hermes_state.SessionDB._remove_session_files(sessions_dir, too_long_id)


@pytest.mark.parametrize(
    "unsafe", ["../auth", "nested/session", "win\\session", "..", "*", "?", "[session]"]
)
def test_remove_session_files_rejects_unsafe_id_regardless_of_length(tmp_path, unsafe):
    """Glob metacharacters and path traversal must stay rejected for
    session-file cleanup even after widening the accepted length range —
    the length ceiling moved, the character-class checks did not."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    other_dump = sessions_dir / "request_dump_sentinel_001.json"
    other_dump.write_text("must survive", encoding="utf-8")

    hermes_state.SessionDB._remove_session_files(sessions_dir, unsafe)

    assert other_dump.read_text(encoding="utf-8") == "must survive"


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
