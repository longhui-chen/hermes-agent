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
