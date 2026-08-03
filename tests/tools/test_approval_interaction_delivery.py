import threading

from tools import approval


def setup_function():
    approval._gateway_queues.clear()
    approval._gateway_prepared.clear()


def teardown_function():
    approval._gateway_queues.clear()
    approval._gateway_prepared.clear()


def _entry(interaction_id: str):
    return approval._ApprovalEntry({
        "interaction_id": interaction_id,
        "command": interaction_id,
    })


def test_prepare_and_finalize_are_fifo_exact_and_do_not_wake_early():
    first = _entry("interaction-a")
    second = _entry("interaction-b")
    approval._gateway_queues["session-1"] = [first, second]

    status, _ = approval.prepare_gateway_approval(
        "session-1", "interaction-b", "delivery-b"
    )
    assert status == "interaction_conflict"
    assert not first.event.is_set()
    assert not second.event.is_set()

    status, data = approval.prepare_gateway_approval(
        "session-1", "interaction-a", "delivery-a"
    )
    assert status == "prepared"
    assert data is not None
    assert data["interaction_id"] == "interaction-a"
    assert not first.event.is_set()

    # A different delivery cannot steal an interaction once prepare owns it.
    status, _ = approval.prepare_gateway_approval(
        "session-1", "interaction-a", "delivery-other"
    )
    assert status == "delivery_conflict"
    assert approval.resolve_gateway_approval("session-1", "deny") == 0

    status, data = approval.finalize_gateway_approval(
        "session-1", "interaction-a", "delivery-a", "once"
    )
    assert status == "resolved"
    assert data is not None
    assert data["interaction_id"] == "interaction-a"
    assert first.result == "once"
    assert first.event.is_set()
    assert not second.event.is_set()
    assert approval._gateway_queues["session-1"] == [second]


def test_gateway_wait_assigns_stable_interaction_id_before_notify(monkeypatch):
    notified = []
    result = {}
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 2},
    )

    def run():
        result["value"] = approval._await_gateway_decision(
            "session-1",
            lambda data: notified.append(dict(data)),
            {"command": "rm demo", "description": "test"},
        )

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if notified:
            break
        thread.join(0.01)

    assert notified[0]["interaction_id"]
    assert notified[0]["interaction_generation"] > 0
    interaction_id = notified[0]["interaction_id"]
    status, _ = approval.prepare_gateway_approval(
        "session-1", interaction_id, "delivery-1"
    )
    assert status == "prepared"
    status, _ = approval.finalize_gateway_approval(
        "session-1", interaction_id, "delivery-1", "deny"
    )
    assert status == "resolved"
    thread.join(2)
    assert result["value"]["choice"] == "deny"


def test_deferred_finalize_waiter_has_bounded_fail_safe(monkeypatch):
    notified = []
    result = {}
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 3},
    )

    def run():
        result["value"] = approval._await_gateway_decision(
            "session-1",
            lambda data: notified.append(data),
            {"command": "rm demo", "description": "test"},
        )

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if notified:
            break
        thread.join(0.01)

    interaction_id = notified[0]["interaction_id"]
    assert approval.prepare_gateway_approval(
        "session-1", interaction_id, "delivery-1"
    )[0] == "prepared"
    status, data, wake_event = approval.finalize_gateway_approval_deferred(
        "session-1",
        interaction_id,
        "delivery-1",
        "deny",
        wake_after_seconds=0.01,
    )
    assert status == "resolved"
    assert data is not None
    assert wake_event is not None and not wake_event.is_set()
    assert thread.is_alive()

    thread.join(1.5)
    assert not thread.is_alive()
    assert wake_event.is_set()
    assert result["value"]["choice"] == "deny"


def test_cancel_gateway_approvals_clears_prepare_and_wakes_all():
    first = _entry("interaction-a")
    second = _entry("interaction-b")
    approval._gateway_queues["session-1"] = [first, second]
    assert approval.prepare_gateway_approval(
        "session-1", "interaction-a", "delivery-a"
    )[0] == "prepared"

    assert approval.cancel_gateway_approvals("session-1") == 2

    assert "session-1" not in approval._gateway_queues
    assert approval._gateway_prepared == {}
    assert first.result == second.result == "deny"
    assert first.event.is_set() and second.event.is_set()
