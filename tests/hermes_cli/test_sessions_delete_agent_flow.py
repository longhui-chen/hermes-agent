"""Flow test: delete-agent must enumerate archived + child sessions (TB-20260814-012 延后项 2)."""

import sqlite3

from hermes_state import SessionDB

from hermes_cli.sessions_cmd import _filter_agent_session_ids


def test_delete_agent_enumerates_archived_and_children(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO sessions (id, source, user_id, started_at, archived) VALUES (?, ?, ?, ?, ?)",
        ("zettlab:alice:agent-a:normal", "zettlab", "alice", 100.0, 0),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, user_id, started_at, archived) VALUES (?, ?, ?, ?, ?)",
        ("zettlab:alice:agent-a:archived", "zettlab", "alice", 200.0, 1),
    )
    conn.execute(
        "INSERT INTO sessions (id, source, user_id, started_at, archived, parent_session_id) VALUES (?, ?, ?, ?, ?, ?)",
        ("zettlab:alice:agent-a:child", "zettlab", "alice", 300.0, 0, "zettlab:alice:agent-a:normal"),
    )
    conn.commit()
    conn.close()

    db = SessionDB(db_path=db_path)
    try:
        rows = db.list_sessions_rich(
            sources=["zettlab"], limit=1000, offset=0,
            compact_rows=True, project_compression_tips=False,
            include_archived=True, include_children=True,
        )
        ids = _filter_agent_session_ids(rows, "agent-a")
    finally:
        db.close()

    assert "zettlab:alice:agent-a:normal" in ids
    assert "zettlab:alice:agent-a:archived" in ids
    assert "zettlab:alice:agent-a:child" in ids
