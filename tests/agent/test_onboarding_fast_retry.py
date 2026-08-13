from types import SimpleNamespace

from agent.conversation_loop import _onboarding_fast_retry_delay


def test_onboarding_502_uses_quick_retry():
    agent = SimpleNamespace(_onboarding_fast_retry=True)
    assert _onboarding_fast_retry_delay(agent, 502, False) == 0.25


def test_normal_agent_and_rate_limit_keep_default_retry_policy():
    normal = SimpleNamespace(_onboarding_fast_retry=False)
    onboarding = SimpleNamespace(_onboarding_fast_retry=True)
    assert _onboarding_fast_retry_delay(normal, 502, False) is None
    assert _onboarding_fast_retry_delay(onboarding, 500, False) is None
    assert _onboarding_fast_retry_delay(onboarding, 502, True) is None
