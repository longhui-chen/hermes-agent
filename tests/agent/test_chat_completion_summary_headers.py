from types import SimpleNamespace


def test_summary_kwargs_stamp_zettlab_headers_for_session():
    from agent.chat_completion_helpers import _apply_zettlab_summary_headers

    kwargs = {}
    agent = SimpleNamespace(session_id="zettlab:user-1:agent-a:conversation-1")

    _apply_zettlab_summary_headers(kwargs, agent)

    assert kwargs["extra_headers"]["X-Task-Id"] == agent.session_id
    assert kwargs["extra_headers"]["X-Zettlab-Conversation-ID"] == agent.session_id
    assert kwargs["extra_headers"]["X-Scene-Type"] == "agent"


def test_summary_kwargs_skip_non_zettlab_session():
    from agent.chat_completion_helpers import _apply_zettlab_summary_headers

    kwargs = {}
    agent = SimpleNamespace(session_id="local-session")

    _apply_zettlab_summary_headers(kwargs, agent)

    assert "extra_headers" not in kwargs
