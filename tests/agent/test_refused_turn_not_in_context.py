"""A content-policy refusal must not survive into the model's context.

The transcript hermes persists is what the model reads next turn. Leaving a
refused exchange in it has two concrete costs: the refused text is re-submitted
on every later turn (and re-scanned by the moderation gateway, which bills per
call), and the model treats its own refusal as precedent and starts hedging on
neighbouring topics.

This is deliberately NOT the same as hiding it from the user —
zettlab-local-server keeps its own transcript of the turn, error code included,
so the App still shows which messages were blocked. The two histories diverge
by design.
"""

from agent.conversation_loop import _transcript_without_refused_turn


def _turn(idx_user_content):
    """A short session ending in the turn that was refused."""
    return [
        {"role": "user", "content": "早上好"},
        {"role": "assistant", "content": "早上好，有什么可以帮你？"},
        {"role": "user", "content": idx_user_content},
    ]


def test_refused_user_turn_is_dropped_and_prior_history_kept():
    messages = _turn("介绍一下敏感人物")
    kept = _transcript_without_refused_turn(messages, messages[2], 2)

    assert kept == messages[:2]
    assert all("敏感人物" not in str(m.get("content", "")) for m in kept)


def test_scaffolding_appended_after_the_user_message_is_dropped_too():
    # A turn can append assistant/tool scaffolding before it fails; none of it
    # belongs to a turn the model never legitimately completed.
    messages = _turn("介绍一下敏感人物") + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "..."},
    ]
    kept = _transcript_without_refused_turn(messages, messages[2], 2)

    assert kept == messages[:2]


def test_stale_index_is_reanchored_rather_than_trusted():
    # Compaction rebuilds `messages` mid-turn, which invalidates the index
    # captured at turn start. Trusting it blindly would slice at the wrong
    # place — either keeping the refused text or eating unrelated history.
    messages = _turn("介绍一下敏感人物")
    kept = _transcript_without_refused_turn(messages, messages[2], 99)

    assert kept == messages[:2]


def test_unlocatable_turn_drops_from_the_last_user_message():
    # Conservative fallback: over-trimming one turn is recoverable, keeping
    # refused content in the model's context is not.
    messages = _turn("介绍一下敏感人物")
    kept = _transcript_without_refused_turn(messages, {"role": "user", "content": "对不上"}, -1)

    assert kept == messages[:2]


def test_session_without_any_user_message_is_left_alone():
    messages = [{"role": "system", "content": "You are..."}]
    kept = _transcript_without_refused_turn(messages, {"role": "user", "content": "x"}, -1)

    assert kept == messages


def test_original_list_is_not_mutated():
    # The caller still holds the live list; trimming must produce a new one so
    # in-flight bookkeeping elsewhere doesn't observe a truncated transcript.
    messages = _turn("介绍一下敏感人物")
    before = list(messages)
    _transcript_without_refused_turn(messages, messages[2], 2)

    assert messages == before
