"""Tests for SessionDB.delete_sessions_for_agent (TB-20260814-012 延后项 2 + Codex P1)."""

import sqlite3

from hermes_state import SessionDB


def _seed(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    for sid, parent, archived in rows:
        conn.execute(
            "INSERT INTO sessions (id, source, user_id, started_at, archived, parent_session_id)"
            " VALUES (?, 'zettlab', 'alice', 100.0, ?, ?)",
            (sid, archived, parent),
        )
    conn.commit()
    conn.close()


def test_delete_sessions_for_agent_deletes_roots_descendants_and_archived(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    _seed(db_path, [
        ("zettlab:alice:agent-a:root", None, 0),
        ("zettlab:alice:agent-a:archived", None, 1),
        # 压缩 continuation（timestamp-hex，继承根所有权）及其下一级。
        ("20260518_120000_abc123", "zettlab:alice:agent-a:root", 0),
        ("20260518_130000_def456", "20260518_120000_abc123", 0),
        # fork 到别的 agent 的子会话——不得误删。
        ("zettlab:alice:agent-b:forked", "zettlab:alice:agent-a:root", 0),
        # 别的 agent 的独立会话——不得删。
        ("zettlab:alice:agent-b:other", None, 0),
    ])

    db = SessionDB(db_path=db_path)
    try:
        deleted = db.delete_sessions_for_agent("agent-a")
    finally:
        db.close()

    assert deleted == 4  # root + archived + 2 级 continuation

    conn = sqlite3.connect(db_path)
    remaining = {r[0] for r in conn.execute("SELECT id FROM sessions")}
    conn.close()
    assert remaining == {"zettlab:alice:agent-b:forked", "zettlab:alice:agent-b:other"}


def test_delete_sessions_for_agent_no_roots_returns_zero(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    _seed(db_path, [("zettlab:alice:agent-b:other", None, 0)])

    db = SessionDB(db_path=db_path)
    try:
        deleted = db.delete_sessions_for_agent("agent-a")
    finally:
        db.close()
    assert deleted == 0
