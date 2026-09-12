from gateway.platforms import zet_agent_metrics as metrics


def test_interaction_metrics_use_bounded_labels_and_snapshot():
    metrics.reset_for_tests()
    metrics.interaction_opened()
    metrics.interaction_answered()
    metrics.interaction_terminal("expired")
    metrics.interaction_terminal("runtime_lost")
    metrics.clarify_rejected("caller_inactive")
    metrics.clarify_rejected("unexpected-secret")
    assert metrics.snapshot() == {
        "interaction_opened": 1,
        "interaction_answered": 1,
        "interaction_terminal{source=hermes,state=expired}": 1,
        "interaction_terminal{source=hermes,state=runtime_lost}": 1,
        "clarify_rejected{reason=caller_inactive}": 1,
        "clarify_rejected{reason=other}": 1,
    }
