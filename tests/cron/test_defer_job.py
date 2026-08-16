"""defer_job: the governor's postponement primitive.

A deferral keeps the job scheduled (unlike pause), never pulls a future
slot earlier, and records deferred_at / defer_reason / defer_count for
observability. The scheduler's at-most-once advance means the deferred slot
simply folds away — a lapsed deferral is one catch-up run, not a backlog.
"""

import pytest

from cron import jobs


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib
    import hermes_constants
    import cron.jobs as jobs_mod
    import cron.scheduler  # noqa: F401 - ensures sibling module init

    importlib.reload(hermes_constants)
    importlib.reload(jobs_mod)
    return home


@pytest.fixture
def interval_job(hermes_env):
    job = jobs.create_job(
        name="refresh",
        schedule="every 30m",
        prompt="refresh the app data",
        deliver="local",
    )
    assert job is not None
    return job


def test_defer_pushes_next_run_and_stays_scheduled(interval_job):
    # The real deferral scenario: the slot is due (the scheduler's at-most-once
    # advance already consumed it), and the gate pushes the next one out.
    from cron.jobs import _hermes_now

    due = _hermes_now().isoformat()
    jobs.update_job(interval_job["id"], {"next_run_at": due})
    deferred = jobs.defer_job(interval_job["id"], seconds=300, reason="governor:memory_pressure")
    assert deferred is not None
    assert deferred["enabled"] is True, "a deferred job is not paused"
    assert deferred["state"] == "scheduled"
    assert deferred["defer_reason"] == "governor:memory_pressure"
    assert deferred["defer_count"] == 1
    assert deferred["deferred_at"] is not None
    assert deferred["next_run_at"] > due


def test_defer_never_pulls_a_future_slot_earlier(interval_job):
    far_future = "2099-01-01T00:00:00+00:00"
    jobs.update_job(interval_job["id"], {"next_run_at": far_future})
    deferred = jobs.defer_job(interval_job["id"], seconds=60, reason="x")
    assert deferred["next_run_at"] == far_future
    # The deferral is still recorded even when the schedule was not moved.
    assert deferred["defer_count"] == 1


def test_defer_counts_consecutive_deferrals(interval_job):
    jobs.defer_job(interval_job["id"], seconds=60, reason="a")
    deferred = jobs.defer_job(interval_job["id"], seconds=300, reason="b")
    assert deferred["defer_count"] == 2
    assert deferred["defer_reason"] == "b"


def test_defer_requires_exactly_one_time_argument(interval_job):
    with pytest.raises(ValueError):
        jobs.defer_job(interval_job["id"], reason="x")
    with pytest.raises(ValueError):
        jobs.defer_job(interval_job["id"], seconds=60, until="2099-01-01T00:00:00+00:00")


def test_defer_until_accepts_iso_timestamp(interval_job):
    deferred = jobs.defer_job(
        interval_job["id"],
        until="2098-06-01T12:00:00+00:00",
        reason="planned window",
    )
    assert deferred is not None
    assert deferred["next_run_at"].startswith("2098-06-01T12:00:00")


def test_defer_unknown_job_returns_none(hermes_env):
    assert jobs.defer_job("no-such-job", seconds=60) is None


def test_mark_job_run_honors_deferred_until_watermark(interval_job):
    # A user defers a job that is already claimed and running. When the run
    # completes, mark_job_run must honor the defer watermark instead of
    # recomputing the natural next slot and silently dropping the deferral.
    from cron.jobs import _hermes_now
    now = _hermes_now()
    jobs.update_job(
        interval_job["id"],
        {
            "next_run_at": now.isoformat(),
            "fire_claim": {"at": now.isoformat(), "fire_at": now.isoformat()},
            "in_flight_occurrence": {"scheduled_at": now.isoformat()},
        },
    )
    deferred = jobs.defer_job(interval_job["id"], seconds=3600, reason="user")
    assert deferred["deferred_until"] is not None

    jobs.mark_job_run(interval_job["id"], success=True)
    updated = jobs.resolve_job_ref(interval_job["id"])
    # 3600s > the natural 30m interval, so the deferral is the later value.
    assert updated["next_run_at"] == deferred["deferred_until"]


def test_defer_clear_claim_terminates_occurrence_when_next_run_does_not_move(interval_job):
    # Regression: the governor's pre-execution gate consumes the occurrence
    # without running it. When the natural next slot is later than the retry
    # point, next_run_at stays unchanged, so update_job's trigger-identity check
    # wouldn't clear the claim — and a still-fresh claim would reject the next
    # callback within the claim TTL, silently stopping the job.
    far_future = "2099-01-01T00:00:00+00:00"
    # Set the schedule first (this clears any claim via the identity check),
    # then stamp a claim WITHOUT moving the schedule.
    jobs.update_job(interval_job["id"], {"next_run_at": far_future})
    jobs.update_job(
        interval_job["id"],
        {
            "fire_claim": {"at": far_future, "fire_at": far_future},
            "in_flight_occurrence": {"scheduled_at": far_future},
        },
    )
    deferred = jobs.defer_job(interval_job["id"], seconds=60, reason="governor", clear_claim=True)
    assert deferred["next_run_at"] == far_future  # schedule did not move
    assert deferred.get("fire_claim") is None
    assert deferred.get("in_flight_occurrence") is None


def test_defer_preserves_claim_when_not_clearing(interval_job):
    # The generic (user/API) defer must NOT clear a still-firing occurrence's
    # claim, even when it moves next_run_at forward — that forward move would
    # otherwise trip update_job's trigger-identity auto-clear and admit a
    # duplicate concurrent run.
    from cron.jobs import _hermes_now
    near = _hermes_now().isoformat()
    jobs.update_job(interval_job["id"], {"next_run_at": near})
    jobs.update_job(
        interval_job["id"],
        {
            "fire_claim": {"at": near, "fire_at": near},
            "in_flight_occurrence": {"scheduled_at": near},
        },
    )
    # 3600s pushes next_run_at well past `near`, so trigger_identity_changed is
    # True inside update_job; the claim must still survive.
    deferred = jobs.defer_job(interval_job["id"], seconds=3600, reason="user")
    assert deferred["next_run_at"] != near
    assert deferred.get("fire_claim") is not None
    assert deferred.get("in_flight_occurrence") is not None
