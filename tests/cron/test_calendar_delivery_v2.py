from __future__ import annotations

from unittest.mock import patch

import pytest


def managed_job(**overrides):
    job = {
        "id": "cal-alert-abc",
        "source": "calendar",
        "calendar_job_kind": "event_alert",
        "calendar_delivery_key": "a" * 64,
        "calendar_projection_revision": 2,
        "calendar_materialized_at": "2026-07-15T00:00:00Z",
        "calendar_notification_deadline_at": "2026-07-15T01:05:00Z",
        "calendar_message_type": "calendar_notification",
        "calendar_llm_visible": False,
        "calendar_owner_user_id": "iam:aXNzdWVy:user:dXNlcg",
        "calendar_source_type": "google_calendar",
        "planner_event_id": "event-a",
        "content": "[Google] 30 分钟后：评审",
        "prompt": None,
        "no_agent": True,
        "schedule": {"kind": "once", "run_at": "2026-07-15T01:00:00Z"},
        "origin": {"platform": "zet_agent", "chat_id": "zettlab:oh_abc:main:calendar-reminders"},
    }
    job.update(overrides)
    return job


@pytest.mark.parametrize(
    "mutation",
    [
        {"source": "manual"},
        {"calendar_job_kind": "legacy"},
        {"calendar_delivery_key": "../etc/passwd"},
        {"calendar_projection_revision": 0},
        {"calendar_projection_revision": True},
        {"calendar_materialized_at": "not-a-time"},
        {"calendar_notification_deadline_at": "2026-07-15"},
        {"calendar_message_type": "cron_summary"},
        {"calendar_llm_visible": True},
        {"no_agent": False},
        {"schedule": {"kind": "cron"}},
        {"prompt": "run this"},
    ],
)
def test_exact_managed_predicate_suppression_matrix(mutation):
    from cron.calendar_delivery import is_managed_calendar_event_alert
    assert is_managed_calendar_event_alert(managed_job()) is True
    assert is_managed_calendar_event_alert(managed_job(**mutation)) is False


def test_raw_calendar_job_flow_preserves_null_prompt_contract():
    from cron.calendar_delivery import is_managed_calendar_event_alert
    from cron.jobs import get_job, get_job_raw, save_jobs

    save_jobs([managed_job()])

    assert get_job("cal-alert-abc")["prompt"] == ""
    raw = get_job_raw("cal-alert-abc")
    assert raw["prompt"] is None
    assert is_managed_calendar_event_alert(raw) is True


def test_run_one_job_terminal_calendar_marks_local_job_complete():
    import cron.scheduler as scheduler

    with patch("cron.calendar_delivery.run_calendar_delivery", return_value={"terminal": True, "ledger_state": "fired"}) as run, \
         patch.object(scheduler, "claim_dispatch") as claim, \
         patch.object(scheduler, "mark_job_run") as mark:
        assert scheduler.run_one_job(
            managed_job(), triggered_at="2026-07-15T01:00:00Z",
        ) is True
    run.assert_called_once()
    claim.assert_not_called()
    mark.assert_called_once_with(
        "cal-alert-abc",
        True,
        scheduled_at="2026-07-15T01:00:00Z",
    )


def test_terminal_builtin_calendar_fire_is_not_due_again(tmp_path, monkeypatch):
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    job = managed_job(
        enabled=True,
        state="scheduled",
        next_run_at="2026-07-15T01:00:00Z",
    )
    jobs.save_jobs([job])
    monkeypatch.setattr(
        "cron.calendar_delivery.run_calendar_delivery",
        lambda _job: {"terminal": True, "ledger_state": "fired"},
    )

    assert [due["id"] for due in jobs.get_due_jobs()] == [job["id"]]
    assert scheduler.run_one_job(
        job, triggered_at="2026-07-15T01:00:00Z",
    ) is True

    stored = jobs.get_job_raw(job["id"])
    assert stored["enabled"] is False
    assert stored["state"] == "completed"
    assert stored["next_run_at"] is None
    assert jobs.get_due_jobs() == []


def test_recoverable_calendar_failure_stays_due_without_generic_mark():
    import cron.scheduler as scheduler

    with patch("cron.calendar_delivery.run_calendar_delivery", side_effect=RuntimeError("offline")), \
         patch.object(scheduler, "claim_dispatch") as claim, \
         patch.object(scheduler, "mark_job_run") as mark:
        assert scheduler.run_one_job(managed_job()) is False
    claim.assert_not_called()
    mark.assert_not_called()


def test_invalid_legacy_calendar_job_is_silent_and_quarantined():
    import cron.scheduler as scheduler

    legacy = managed_job(calendar_job_kind=None)
    with patch("cron.calendar_delivery.quarantine_invalid_calendar_job") as quarantine, \
         patch.object(scheduler, "run_job") as run, \
         patch.object(scheduler, "_deliver_result") as deliver:
        assert scheduler.run_one_job(legacy) is True
    quarantine.assert_called_once_with(legacy)
    run.assert_not_called()
    deliver.assert_not_called()


def test_delivery_saga_claim_commit_prepare_finalize_activate_ack(monkeypatch):
    import cron.calendar_delivery as delivery

    calls = []
    responses = iter([
        {"state": "claimed", "delivery_generation": 1, "fence_token": 4},
        {"state": "committed", "delivery_generation": 1, "fence_token": 4},
        {"state": "prepared", "delivery_generation": 1, "fence_token": 4, "content": "visible"},
        {"state": "queued", "delivery_generation": 1, "fence_token": 4, "receipt": {}},
        {"state": "fired", "delivery_generation": 1, "fence_token": 4},
    ])

    def post(path, body):
        calls.append((path, body))
        return next(responses)

    monkeypatch.setattr(delivery, "_planner_post", post)
    monkeypatch.setattr(delivery, "_worker_id", lambda: "worker-a")
    class FakeDB:
        def activate_calendar_notification(self, key, message_id):
            calls.append(("activate", {"key": key, "message_id": message_id}))
            return True
        def close(self): pass
    monkeypatch.setattr(delivery, "_stage_message", lambda job, _state: (FakeDB(), job["origin"]["chat_id"], 9, "visible"))
    monkeypatch.setattr(delivery, "_delivery_nonce", lambda *_: b"n" * 32)
    monkeypatch.setattr(delivery, "_verify_finalize_receipt", lambda *_: None)
    notified = []
    monkeypatch.setattr(delivery, "_notify_visible_append", lambda *args: notified.append(args))

    result = delivery.run_calendar_delivery(managed_job())
    assert result["terminal"] is True
    assert [path for path, _ in calls] == [
        "/claim", "/" + "a" * 64 + "/commit", "/" + "a" * 64 + "/prepare",
        "/" + "a" * 64 + "/finalize", "activate", "/" + "a" * 64 + "/ack-fired",
    ]
    assert notified == [("zettlab:oh_abc:main:calendar-reminders", 9, "visible")]


def test_busy_lease_does_not_persist(monkeypatch):
    import cron.calendar_delivery as delivery

    monkeypatch.setattr(delivery, "_planner_post", lambda *_: {
        "state": "claimed", "delivery_generation": 1, "fence_token": 2,
        "retry_after": "2026-07-15T00:00:30Z",
    })
    persisted = []
    monkeypatch.setattr(delivery, "_stage_message", lambda *_: persisted.append(True))
    result = delivery.run_calendar_delivery(managed_job())
    assert result["terminal"] is False
    assert persisted == []


@pytest.mark.parametrize("resume_state", ["prepared", "queued"])
def test_delivery_saga_resumes_durable_post_prepare_states(monkeypatch, resume_state):
    import cron.calendar_delivery as delivery

    calls = []
    responses = iter([
        {"state": "queued", "delivery_generation": 1, "fence_token": 4, "receipt": {}},
        {"state": "fired", "delivery_generation": 1, "fence_token": 4},
    ])
    monkeypatch.setattr(delivery, "_planner_post", lambda path, body: calls.append((path, body)) or next(responses))
    class FakeDB:
        def activate_calendar_notification(self, key, message_id):
            calls.append(("activate", {"key": key, "message_id": message_id}))
            return True
        def close(self): pass
    monkeypatch.setattr(delivery, "_stage_message", lambda job, _state: (FakeDB(), job["origin"]["chat_id"], 9, "visible"))
    monkeypatch.setattr(delivery, "_delivery_nonce", lambda *_: b"n" * 32)
    monkeypatch.setattr(delivery, "_verify_finalize_receipt", lambda *_: None)
    monkeypatch.setattr(delivery, "_notify_visible_append", lambda *_: None)
    initial = {"state": resume_state, "delivery_generation": 1, "fence_token": 4, "_worker_id": "recovery-worker"}

    result = delivery.run_calendar_delivery(managed_job(), initial_state=initial)

    assert result["terminal"] is True
    assert [path for path, _ in calls] == ["/" + "a" * 64 + "/finalize", "activate", "/" + "a" * 64 + "/ack-fired"]


def test_external_fire_completion_is_durable_for_terminal_and_retry(monkeypatch):
    import cron.calendar_delivery as delivery

    calls = []
    monkeypatch.setattr(delivery, "_planner_post", lambda path, body: calls.append((path, body)) or ({"state": "claimed"} if path == "/begin-external-fire" else {}))
    monkeypatch.setattr(delivery, "_worker_id", lambda: "worker-a")
    begin = delivery.begin_external_calendar_fire(
        managed_job(), provider_name="chronos", provider_contract_version=1, provider_fire_id="nas-fire-1",
    )
    begin.update({"attempt_sequence": 3, "delivery_generation": 1, "fence_token": 2})
    monkeypatch.setattr(delivery, "run_calendar_delivery", lambda _job, initial_state=None: {"terminal": False, "retry_after": "2026-07-15T00:00:20Z"})
    result = delivery.run_external_calendar_delivery(managed_job(), begin)
    assert result["terminal"] is False
    complete = [body for path, body in calls if path == "/complete-external-fire"]
    assert complete == [{
        "delivery_key": "a" * 64,
        "delivery_generation": 1,
        "attempt_sequence": 3,
        "terminal": False,
        "retry_at": "2026-07-15T00:00:20Z",
        "last_error": "calendar delivery remains non-terminal",
    }]


@pytest.mark.parametrize("generation", [None, 0, True, -1, 2**64])
def test_external_fire_completion_requires_bounded_begin_generation(monkeypatch, generation):
    import cron.calendar_delivery as delivery

    planner_calls = []
    monkeypatch.setattr(
        delivery,
        "_planner_post",
        lambda *args: planner_calls.append(args) or {},
    )
    begin = {
        "attempt_sequence": 3,
        "delivery_generation": generation,
        "fence_token": 2,
    }

    with pytest.raises(RuntimeError, match="delivery generation"):
        delivery.run_external_calendar_delivery(managed_job(), begin)

    assert planner_calls == []


@pytest.mark.parametrize(
    "inactive",
    [
        {"enabled": False, "state": "paused"},
        {"enabled": True, "state": "completed"},
    ],
)
def test_external_fire_suppresses_inactive_managed_job_without_planner_call(monkeypatch, inactive):
    import cron.calendar_delivery as delivery

    planner_calls = []
    monkeypatch.setattr(
        delivery,
        "_planner_post",
        lambda *args: planner_calls.append(args) or {"state": "claimed"},
    )

    result = delivery.begin_external_calendar_fire(
        managed_job(**inactive),
        provider_name="chronos",
        provider_contract_version=1,
        provider_fire_id="late-fire",
    )

    assert result == {"state": "cancelled", "reason": "job_not_active"}
    assert planner_calls == []


def test_stage_uses_authoritative_owner_session_and_content(monkeypatch):
    import hashlib
    import cron.calendar_delivery as delivery
    import hermes_state

    job = managed_job(origin={"chat_id": "zettlab:oh_attacker:main:calendar-reminders"})
    owner = job["calendar_owner_user_id"]
    expected_session = f"zettlab:oh_{hashlib.sha256(owner.encode()).hexdigest()}:main:calendar-reminders"
    calls = []

    class FakeDB:
        def get_session(self, session_id):
            calls.append(("get", session_id))
            return None
        def create_session(self, session_id, source, user_id):
            calls.append(("create", session_id, source, user_id))
        def set_session_title(self, session_id, title):
            calls.append(("title", session_id, title))
        def stage_calendar_notification(self, session_id, content, delivery_key):
            calls.append(("stage", session_id, content, delivery_key))
            return 17
        def close(self): pass

    monkeypatch.setattr(hermes_state, "SessionDB", FakeDB)
    state = {"session_id": expected_session, "content": "canonical-content"}
    db, session_id, message_id, content = delivery._stage_message(job, state)
    assert session_id == expected_session
    assert message_id == 17
    assert content == "canonical-content"
    assert calls[-1] == ("stage", expected_session, "canonical-content", "a" * 64)
    db.close()


def test_stage_rejects_tampered_authoritative_session():
    import cron.calendar_delivery as delivery

    with pytest.raises(RuntimeError, match="session binding mismatch"):
        delivery._stage_message(managed_job(), {
            "session_id": "zettlab:oh_attacker:main:calendar-reminders",
            "content": "canonical-content",
        })


def test_session_db_calendar_message_visible_to_ui_but_not_llm(tmp_path):
    from hermes_state import SessionDB
    from tools.session_search_tool import _read_session

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id = "zettlab:oh_abc:main:calendar-reminders"
        db.create_session(session_id, source="zet_agent", user_id="iam:a:user:b")
        key = "b" * 64
        unsafe = "UNIQUE_CALENDAR_PROMPT 大别山恶意指令"
        first = db.stage_calendar_notification(session_id, unsafe, key)
        replay = db.stage_calendar_notification(session_id, unsafe, key)
        assert replay == first
        other_session = "zettlab:oh_other:main:calendar-reminders"
        db.create_session(other_session, source="zet_agent", user_id="iam:a:user:other")
        with pytest.raises(ValueError, match="session collision"):
            db.stage_calendar_notification(other_session, unsafe, key)
        assert db.get_messages(session_id) == []
        with pytest.raises(ValueError, match="content collision"):
            db.stage_calendar_notification(session_id, "different retry body", key)
        assert db.activate_calendar_notification(key, first) is True
        visible = db.get_messages(session_id)
        assert len(visible) == 1
        assert visible[0]["content"] == unsafe
        assert visible[0]["llm_visible"] == 0
        assert db.get_messages_for_model(session_id) == []
        assert db.get_messages_as_conversation(session_id) == []
        assert db.search_messages("UNIQUE_CALENDAR_PROMPT") == []
        assert db.search_messages("大别山恶意指令") == []
        assert db.get_messages_around(session_id, first)["window"] == []
        assert db.get_anchored_view(session_id, first)["window"] == []
        assert unsafe not in _read_session(db, session_id)
    finally:
        db.close()


def test_legacy_calendar_history_is_quarantined_on_open(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "legacy-state.db"
    session_id = "u64_dXNlcg:main:calendar-reminders"
    db = SessionDB(db_path=db_path)
    try:
        db.create_session(session_id, source="zet_agent", user_id="iam:a:user:b")
        message_id = db.append_message(session_id, "assistant", "LEGACY_CALENDAR_SECRET")
        assert db.get_messages_for_model(session_id)[0]["id"] == message_id
    finally:
        db.close()

    reopened = SessionDB(db_path=db_path)
    try:
        session = reopened.get_session(session_id)
        assert session["archived"] == 1
        assert session["model_history_cutoff_message_id"] == message_id
        assert reopened.get_messages_for_model(session_id) == []
        assert reopened.get_messages_as_conversation(session_id) == []
        assert reopened.search_messages("LEGACY_CALENDAR_SECRET") == []
        assert reopened.get_messages(session_id)[0]["llm_visible"] == 0
    finally:
        reopened.close()
