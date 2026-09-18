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
        **metrics._ITEM_FRAME_DEFAULTS,
        "interaction_opened": 1,
        "interaction_answered": 1,
        "interaction_terminal{source=hermes,state=expired}": 1,
        "interaction_terminal{source=hermes,state=runtime_lost}": 1,
        "clarify_rejected{reason=caller_inactive}": 1,
        "clarify_rejected{reason=other}": 1,
    }


def test_item_metrics_are_fixed_process_aggregate_and_present_at_zero():
    metrics.reset_for_tests()
    assert metrics.snapshot()["item_frame_stranded"] == 0
    metrics.item_frame_count("item_frame_stranded", 3)
    for index in range(1000):
        metrics.item_frame_count(f"item_frame_unregistered:secret-{index}")
        metrics.item_frame_dropped_backlog(f"secret-{index}")
    metrics.item_frame_count("item_frame_stranded", -1)
    assert metrics.snapshot() == {
        **metrics._ITEM_FRAME_DEFAULTS,
        "item_frame_stranded": 3,
        "item_frame_dropped_backlog{kind=other}": 1000,
    }


def test_item_metrics_count_concurrent_request_threads():
    from concurrent.futures import ThreadPoolExecutor

    metrics.reset_for_tests()

    def count_request(_):
        for _ in range(100):
            metrics.item_frame_count("item_frame_stranded", 2)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(count_request, range(8)))
    assert metrics.snapshot()["item_frame_stranded"] == 1600

def test_t6_item_counters_each_increment_once_through_snapshot():
    metrics.reset_for_tests()
    metrics.item_frame_count("item_frame_dropped_backlog", 0)  # fixed-bucket name is rejected
    metrics.item_frame_count("item_frame_stranded", 1)
    metrics.item_frame_count("item_frame_unclassified", 1)
    metrics.item_frame_dropped_backlog("attachment")
    snapshot = metrics.snapshot()
    assert snapshot["item_frame_stranded"] == 1
    assert snapshot["item_frame_unclassified"] == 1
    assert snapshot["item_frame_dropped_backlog{kind=attachment}"] == 1
