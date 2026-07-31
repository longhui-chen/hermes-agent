import tools.approval as approval


def _reset_state():
    with approval._lock:
        approval._pending.clear()
        approval._one_shot_approved.clear()


def test_deferred_payload_is_bounded_and_fingerprinted(monkeypatch):
    _reset_state()
    monkeypatch.setattr(approval, "_ensure_deferred_sweeper", lambda: None)
    approval_id = approval.submit_pending(
        "bounded-session",
        {
            "command": "x" * 1_000_000,
            "description": "d" * 50_000,
            "pattern_key": "danger",
            "pattern_keys": ["danger"],
        },
    )
    assert approval_id
    with approval._lock:
        item = approval._pending["bounded-session"][0]
        assert len(item["command"]) < 5000
        assert len(item["description"]) < 2000
        assert len(item["payload_fingerprint"]) == 64
        assert item["_estimated_bytes"] < 10_000
    _reset_state()


def test_global_lru_ceiling_evicts_oldest_sessions(monkeypatch):
    _reset_state()
    clock = [1000.0]
    monkeypatch.setattr(approval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 600.0)
    monkeypatch.setattr(approval, "_ensure_deferred_sweeper", lambda: None)
    monkeypatch.setattr(approval, "_MAX_DEFERRED_APPROVALS_GLOBAL", 8)
    monkeypatch.setattr(approval, "_MAX_DEFERRED_APPROVAL_BYTES_GLOBAL", 64_000)

    for index in range(20):
        assert approval.submit_pending(
            f"session-{index}",
            {"command": f"command-{index}", "pattern_key": "danger"},
        )
        clock[0] += 1

    with approval._lock:
        count, total_bytes = approval._deferred_usage_locked()
        assert count == 8
        assert total_bytes <= 64_000
        assert "session-0" not in approval._pending
        assert "session-19" in approval._pending
    _reset_state()


def test_expired_sessions_are_actively_swept(monkeypatch):
    _reset_state()
    clock = [2000.0]
    monkeypatch.setattr(approval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(approval, "_get_approval_timeout", lambda: 1.0)
    monkeypatch.setattr(approval, "_ensure_deferred_sweeper", lambda: None)
    for index in range(40):
        assert approval.submit_pending(
            f"expired-{index}",
            {"command": "command", "pattern_key": "danger"},
        )
    clock[0] += 2.0
    approval._sweep_deferred_approvals()
    with approval._lock:
        assert approval._pending == {}
    _reset_state()
