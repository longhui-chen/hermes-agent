"""Unit tests for the sessions delete-agent filter (TB-20260814-012 延后项 2)."""

from hermes_cli.sessions_cmd import _filter_agent_session_ids


def test_filter_agent_session_ids_scopes_to_agent():
    rows = [
        {"id": "zettlab:alice:agent-a:one"},
        {"id": "zettlab:alice:agent-a:two"},
        {"id": "zettlab:alice:agent-b:one"},
        # lookalike: tail contains ":agent-a:" but the agent segment is agent-b
        {"id": "zettlab:alice:agent-b:tail:agent-a:lookalike"},
        {"id": "20260430_cli_xyz"},  # non-zettlab prefix
        {"id": "cron_job-aaa_20260508_073000"},  # cron shape
    ]
    got = _filter_agent_session_ids(rows, "agent-a")
    assert got == ["zettlab:alice:agent-a:one", "zettlab:alice:agent-a:two"]


def test_filter_agent_session_ids_cross_user_allowed():
    rows = [{"id": "zettlab:bob:agent-a:abc"}]
    assert _filter_agent_session_ids(rows, "agent-a") == ["zettlab:bob:agent-a:abc"]


def test_filter_agent_session_ids_empty_agent_returns_nothing():
    rows = [{"id": "zettlab:alice:agent-a:one"}]
    assert _filter_agent_session_ids(rows, "") == []

import sqlite3

from hermes_cli.sessions_cmd import _collect_descendant_session_ids


def test_collect_descendant_session_ids_walks_compression_chain():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT)")
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("zettlab:alice:agent-a:root", None))
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("20260518_120000_abc123", "zettlab:alice:agent-a:root"))
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("20260518_130000_def456", "20260518_120000_abc123"))
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("zettlab:alice:agent-b:other", None))

    got = _collect_descendant_session_ids(conn, ["zettlab:alice:agent-a:root"], "agent-a")
    assert set(got) == {
        "zettlab:alice:agent-a:root",
        "20260518_120000_abc123",
        "20260518_130000_def456",
    }


def test_collect_descendant_skips_other_agent_zettlab_child():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT)")
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("zettlab:alice:agent-a:root", None))
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("20260518_120000_abc123", "zettlab:alice:agent-a:root"))
    # fork 到别的 agent 的 zettlab 子会话——不能误删。
    conn.execute("INSERT INTO sessions VALUES (?, ?)", ("zettlab:alice:agent-b:forked", "zettlab:alice:agent-a:root"))

    got = _collect_descendant_session_ids(conn, ["zettlab:alice:agent-a:root"], "agent-a")
    assert set(got) == {"zettlab:alice:agent-a:root", "20260518_120000_abc123"}

