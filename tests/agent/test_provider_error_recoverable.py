"""``recoverable`` on the outbound provider-error payload.

Clients key their recovery affordance off this one boolean, so the value the
payload carries decides what the user is offered. zettlab-app's
``classifyErrorChannel`` reads the wire flag first and only falls back to its
own catalog when the flag is absent — so a wrong ``true`` here silently
overrides the client's own "this is terminal" classification.

The regression these tests pin was found on a real device (2026-08-03): a
content-moderation block arrived with ``recoverable: true`` because
``should_fallback`` was still set, and the App rendered a plain "reply failed /
resend" row instead of the compliance notice. Resending unchanged is the one
action that cannot possibly work for a content block.
"""

from types import SimpleNamespace

from agent.error_classifier import ClassifiedError, FailoverReason
from run_agent import AIAgent


def _payload(reason, *, retryable=False, should_fallback=False, should_compress=False):
    classified = ClassifiedError(
        reason=reason,
        retryable=retryable,
        should_fallback=should_fallback,
        should_compress=should_compress,
        status_code=400,
        message="内容不合规",
        provider="zettlab",
        model="glm-5",
    )
    agent = SimpleNamespace(provider="zettlab", model="glm-5")
    return AIAgent._provider_error_payload(agent, classified, Exception("boom"))


def test_content_policy_block_is_never_recoverable_even_with_fallback_configured():
    # should_fallback=True is the realistic case: it stays true whenever the
    # profile has a second model configured, and it is what used to leak through.
    payload = _payload(FailoverReason.content_policy_blocked, should_fallback=True)
    assert payload["code"] == "content_blocked"
    assert payload["recoverable"] is False
    assert payload["retryable"] is False


def test_content_policy_block_not_recoverable_without_fallback_either():
    payload = _payload(FailoverReason.content_policy_blocked)
    assert payload["recoverable"] is False


def test_other_reasons_keep_deriving_recoverable_from_the_retry_signals():
    # The override is scoped to content policy only — a transient overload must
    # still tell the client it is worth retrying, or we would turn every
    # recoverable failure into a dead end.
    assert _payload(FailoverReason.overloaded, retryable=True)["recoverable"] is True
    assert _payload(FailoverReason.context_overflow, should_compress=True)["recoverable"] is True
    assert _payload(FailoverReason.rate_limit, should_fallback=True)["recoverable"] is True


def test_terminal_non_policy_reason_stays_non_recoverable():
    assert _payload(FailoverReason.auth_permanent)["recoverable"] is False
