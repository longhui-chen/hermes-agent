"""End-to-end wire contract for a content-moderation block.

Exercises the real chain a blocked turn takes on its way out of hermes:

    classify_api_error(gateway 400)
      → normalized_provider_error_code
      → _content_policy_blocked_result(provider_error=...)
      → _chat_finish_reason_from_result
      → _chat_stream_error_payload            ← what local-server parses

zettlab-local-server reads ``payload["code"]`` off the ``event: hermes.error``
frame to pick its wire error code. Before this chain carried provider_error the
payload fell back to the generic ``agent_error``, which is not in the App's
error catalog — the App dropped it silently and the user saw a turn that ended
with no content and no explanation.
"""

from agent.conversation_loop import _content_policy_blocked_result
from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    normalized_provider_error_code,
)
from gateway.platforms.api_server import (
    _chat_finish_reason_from_result,
    _chat_stream_error_payload,
)


class _GatewayModerationError(Exception):
    """The HTTP 400 the Zettlab CN moderation gateway returns on a block."""

    def __init__(self):
        self.body = {
            "error": {
                "code": "moderation_input_blocked",
                "type": "content_policy_violation",
                "message": "内容不合规",
                "request_id": "req-abc",
            }
        }
        self.status_code = 400
        super().__init__("Error code: 400 - " + str(self.body))


def _blocked_result():
    err = _GatewayModerationError()
    classified = classify_api_error(err, provider="zettlab", model="glm-5")
    assert classified.reason == FailoverReason.content_policy_blocked
    return _content_policy_blocked_result(
        [],
        1,
        final_response="⚠️  blocked",
        error_detail="内容不合规",
        provider_error={
            "code": normalized_provider_error_code(classified),
            "reason": classified.reason.value,
            "provider": "zettlab",
            "model": "glm-5",
            "status_code": classified.status_code,
            "retryable": False,
            "recoverable": False,
        },
    )


def test_blocked_turn_emits_content_blocked_code():
    result = _blocked_result()
    finish_reason = _chat_finish_reason_from_result(result)
    payload = _chat_stream_error_payload(result, finish_reason)

    assert payload is not None, "a blocked turn must emit an error frame"
    # The single field local-server keys off. "agent_error" here means the App
    # shows nothing at all.
    assert payload["code"] == "content_blocked"
    assert payload["recoverable"] is False
    assert payload["reason"] == "content_policy_blocked"


def test_blocked_turn_finishes_as_error_not_stop():
    # A block must not look like a normal completion: local-server's translator
    # maps finish_reason "stop" to turn.end{stop} and the App would render an
    # empty successful reply.
    assert _chat_finish_reason_from_result(_blocked_result()) == "error"


def test_blocked_turn_is_terminal():
    result = _blocked_result()
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["error"].startswith("content_policy_blocked:")


def test_result_without_provider_error_still_degrades_safely():
    # Older callers (and the defensive path) may omit provider_error. The frame
    # must still be emitted — just with the generic code — rather than raising.
    result = _content_policy_blocked_result(
        [], 1, final_response="⚠️  blocked", error_detail="内容不合规"
    )
    assert "provider_error" not in result
    payload = _chat_stream_error_payload(result, _chat_finish_reason_from_result(result))
    assert payload is not None
    assert payload["code"] == "agent_error"


def test_blocked_turn_flow_strips_refused_text_from_the_persisted_transcript():
    """Full outbound shape of a blocked turn: wire code out, refused text gone.

    Chains the two halves that ship together — the client must learn the turn
    was refused (``code``), and the model must not read the refused text on the
    next turn (``messages``). They are asserted together because fixing one
    without the other is a silent half-measure: a correct error code on a
    transcript that still carries the refused prompt keeps re-submitting it to
    the moderation gateway on every subsequent turn.
    """
    from agent.conversation_loop import _transcript_without_refused_turn

    live_messages = [
        {"role": "user", "content": "早上好"},
        {"role": "assistant", "content": "早上好"},
        {"role": "user", "content": "介绍一下敏感人物"},
    ]
    kept = _transcript_without_refused_turn(live_messages, live_messages[2], 2)

    err = _GatewayModerationError()
    classified = classify_api_error(err, provider="zettlab", model="glm-5")
    result = _content_policy_blocked_result(
        kept,
        1,
        final_response="⚠️  blocked",
        error_detail="内容不合规",
        provider_error={
            "code": normalized_provider_error_code(classified),
            "reason": classified.reason.value,
            "retryable": False,
            "recoverable": False,
        },
    )

    payload = _chat_stream_error_payload(result, _chat_finish_reason_from_result(result))
    assert payload["code"] == "content_blocked"
    # zet_agent writes result["messages"] to the session DB itself, so this is
    # the list the model actually re-reads next turn.
    assert result["messages"] == live_messages[:2]
    assert all("敏感人物" not in str(m.get("content", "")) for m in result["messages"])
