"""``hermes sessions import-transcript`` — forking a session across profiles.

The app-dedicated maintainer agent is created for one generated app and is
reachable only from that app's sidebar. It should open on the conversation the
app was born from, not on a blank session — but the runtime-import contract
accepts only user/assistant text, so most of a generation transcript cannot
cross over. These tests pin both halves: what survives, and that a retry does
not fork twice.
"""

import json
import sqlite3
import types

import pytest

from hermes_cli.sessions_cmd import cmd_sessions, importable_transcript_messages


def row(role, content, timestamp=1_700_000_000.0, *, active=1, llm_visible=1):
    return {
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "active": active,
        "llm_visible": llm_visible,
    }


class TestImportableTranscriptMessages:
    def test_keeps_user_and_assistant_text_in_source_order(self):
        kept = importable_transcript_messages([
            row("user", "build me a price board"),
            row("assistant", "done — it refreshes every 3 minutes"),
        ])
        assert [m["role"] for m in kept] == ["user", "assistant"]
        assert kept[0]["content"] == "build me a price board"

    # The transcript's bulk is tool traffic. Dropping it is the contract, not a
    # bug: the target profile has its own toolset and a replayed history of
    # calls it cannot make would teach it to invoke missing tools.
    def test_drops_tool_and_system_rows(self):
        kept = importable_transcript_messages([
            row("user", "go"),
            row("tool", '{"stdout": "ok"}'),
            row("system", "you are a helpful assistant"),
        ])
        assert [m["role"] for m in kept] == ["user"]

    # An assistant row that only carries tool_calls has NULL content. On the
    # first real app this ran against, 29 of 30 assistant rows looked like this
    # — forwarding them would fail the import on "content must be non-empty".
    def test_drops_assistant_rows_whose_content_is_empty(self):
        kept = importable_transcript_messages([
            row("assistant", None),
            row("assistant", "   "),
            row("assistant", "here is the result"),
        ])
        assert len(kept) == 1
        assert kept[0]["content"] == "here is the result"

    def test_uses_a_stable_sentinel_for_a_missing_timestamp(self):
        kept = importable_transcript_messages([row("user", "hi", timestamp=None)])
        assert kept[0]["created_at"] == 0.0

    def test_drops_structured_content_after_decoding(self):
        kept = importable_transcript_messages([
            row("user", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]),
            row("assistant", "text survives"),
        ])
        assert [message["content"] for message in kept] == ["text survives"]


def _seed_source_profile(db_path, session_id, rows):
    """Write a minimal source transcript — only the columns the fork reads."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, "
        "model_history_cutoff_message_id INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, "
        "active INTEGER NOT NULL DEFAULT 1, llm_visible INTEGER NOT NULL DEFAULT 1)"
    )
    conn.execute("INSERT INTO sessions (id, title) VALUES (?, ?)",
                 (session_id, "app build"))
    for r in rows:
        conn.execute(
            "INSERT INTO messages "
            "(session_id, role, content, timestamp, active, llm_visible) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                r["role"],
                r["content"],
                r["timestamp"],
                r.get("active", 1),
                r.get("llm_visible", 1),
            ),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def forked(tmp_path, monkeypatch, capsys):
    """Run the fork between two sibling profiles under one profiles/ root."""
    from hermes_cli import main as cli_main

    profiles = tmp_path / "profiles"
    (profiles / "source").mkdir(parents=True)
    (profiles / "target").mkdir(parents=True)
    monkeypatch.setattr(cli_main, "get_hermes_home", lambda: str(profiles / "target"))
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: tmp_path if name == "default" else profiles / name,
    )

    source_session = "zettlab:u:source:gen1"
    _seed_source_profile(profiles / "source" / "state.db", source_session, [
        row("user", "build me a price board"),
        row("assistant", None),
        row("tool", '{"stdout": "compiled"}'),
        row("assistant", "done — it refreshes every 3 minutes"),
    ])

    staged = []
    committed = {"count": 0}

    class FakeDB:
        def stage_completed_transcript_import(self, **kwargs):
            staged.append(kwargs)

        def commit_completed_transcript_import(self, import_id):
            committed["count"] += 1
            return {"replayed": committed["count"] > 1}

    def run(target_session="zettlab:u:target:fork1", source_profile="source"):
        args = types.SimpleNamespace(
            sessions_action="import-transcript",
            source_profile=source_profile,
            source_session=source_session,
            target_session=target_session,
            title=None,
            json=True,
        )
        cmd_sessions_with_db(args, FakeDB())
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    return types.SimpleNamespace(run=run, staged=staged, source_session=source_session)


def cmd_sessions_with_db(args, db):
    """Invoke the dispatcher with an injected store.

    Source-profile reads still use the real read-only ``SessionDB``; only the
    active target store is replaced.
    """
    from hermes_cli import sessions_cmd

    original = sessions_cmd._open_session_db
    sessions_cmd._open_session_db = lambda: db
    try:
        return cmd_sessions(args)
    finally:
        sessions_cmd._open_session_db = original


class TestForkAcrossProfiles:
    def test_reports_what_crossed_over_and_what_did_not(self, forked):
        result = forked.run()
        assert result["ok"] is True
        # 4 source rows in, 2 out: the NULL-content assistant and the tool row
        # are contract-rejected, and saying so beats implying a faithful copy.
        assert result["imported"] == 2
        assert result["skipped"] == 2

    def test_stages_only_contract_shaped_messages(self, forked):
        forked.run()
        messages = forked.staged[0]["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert all(m["content"].strip() for m in messages)

    # Same source + target must reuse one import_id, or a retried creation
    # step forks the conversation twice into the maintainer's session list.
    def test_retry_replays_instead_of_forking_twice(self, forked):
        first = forked.run()
        second = forked.run()
        assert first["replayed"] is False
        assert second["replayed"] is True
        assert forked.staged[0]["import_id"] == forked.staged[1]["import_id"]

    def test_a_different_target_gets_a_different_import_id(self, forked):
        forked.run(target_session="zettlab:u:target:fork1")
        forked.run(target_session="zettlab:u:target:fork2")
        assert forked.staged[0]["import_id"] != forked.staged[-1]["import_id"]

    def test_filters_rewound_hidden_and_pre_cutoff_rows(self, tmp_path, monkeypatch, capsys):
        from hermes_cli import main as cli_main

        profiles = tmp_path / "profiles"
        (profiles / "source").mkdir(parents=True)
        (profiles / "target").mkdir(parents=True)
        monkeypatch.setattr(cli_main, "get_hermes_home", lambda: str(profiles / "target"))
        monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: profiles / name)
        session_id = "zettlab:u:source:filtered"
        source_db = profiles / "source" / "state.db"
        _seed_source_profile(source_db, session_id, [
            row("user", "compressed old text"),
            row("assistant", "rewound text", active=0),
            row("user", "hidden control text", llm_visible=0),
            row("assistant", "current visible text"),
        ])
        conn = sqlite3.connect(source_db)
        conn.execute(
            "UPDATE sessions SET model_history_cutoff_message_id = 1 WHERE id = ?",
            (session_id,),
        )
        conn.commit()
        conn.close()

        staged = []

        class FakeDB:
            def stage_completed_transcript_import(self, **kwargs):
                staged.extend(kwargs["messages"])

            def commit_completed_transcript_import(self, _import_id):
                return {"replayed": False}

        args = types.SimpleNamespace(
            sessions_action="import-transcript",
            source_profile="source",
            source_session=session_id,
            target_session="zettlab:u:target:filtered",
            title=None,
            json=True,
        )
        assert cmd_sessions_with_db(args, FakeDB()) == 0
        assert [message["content"] for message in staged] == ["current visible text"]
        assert json.loads(capsys.readouterr().out)["skipped"] == 3

    def test_resolves_the_builtin_default_profile_at_the_root(self, tmp_path, monkeypatch, capsys):
        from hermes_cli import main as cli_main

        target = tmp_path / "profiles" / "target"
        target.mkdir(parents=True)
        monkeypatch.setattr(cli_main, "get_hermes_home", lambda: str(target))
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: tmp_path if name == "default" else tmp_path / "profiles" / name,
        )
        session_id = "zettlab:u:default:gen"
        _seed_source_profile(tmp_path / "state.db", session_id, [row("user", "from default")])
        staged = []

        class FakeDB:
            def stage_completed_transcript_import(self, **kwargs):
                staged.extend(kwargs["messages"])

            def commit_completed_transcript_import(self, _import_id):
                return {"replayed": False}

        args = types.SimpleNamespace(
            sessions_action="import-transcript",
            source_profile="default",
            source_session=session_id,
            target_session="zettlab:u:target:default",
            title=None,
            json=True,
        )
        assert cmd_sessions_with_db(args, FakeDB()) == 0
        assert staged[0]["content"] == "from default"
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_rejects_a_source_profile_path_traversal(self, forked):
        result = forked.run(source_profile="../source")
        assert result["ok"] is False
        assert "Invalid profile name" in result["error"]
