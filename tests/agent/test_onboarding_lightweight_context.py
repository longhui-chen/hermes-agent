from types import SimpleNamespace

from agent.conversation_loop import _compact_lightweight_api_messages


def test_onboarding_lightweight_context_keeps_only_system_and_current_user():
    messages = [
        {"role": "system", "content": "compact policy"},
        {"role": "user", "content": "old answer"},
        {"role": "assistant", "content": "old guide"},
        {"role": "user", "content": "current snapshot"},
    ]

    result = _compact_lightweight_api_messages(
        SimpleNamespace(_onboarding_lightweight=True), messages
    )

    assert result == [messages[0], messages[-1]]


def test_normal_agent_context_is_unchanged():
    messages = [
        {"role": "system", "content": "normal policy"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "next"},
    ]

    assert _compact_lightweight_api_messages(SimpleNamespace(), messages) is messages
