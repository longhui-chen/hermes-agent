"""Behavior contract for the read-only cron calendar occurrence projection."""

from datetime import datetime, timedelta, timezone
import os

import cron.jobs as jobs


UTC = timezone.utc


def _job(**overrides):
    base = {
        "id": "jobdaily",
        "name": "daily task",
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "daily"},
        "timezone": "Asia/Shanghai",
        "repeat": {"times": None, "completed": 2},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": "2026-07-21T01:00:00+00:00",
    }
    base.update(overrides)
    return base


def _seed_output(tmp_path, monkeypatch, job_id, filename, instant, body):
    output_root = tmp_path / "cron" / "output"
    monkeypatch.setattr(jobs, "_output_dir", lambda: output_root)
    output = output_root / job_id
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    path.write_text(body, encoding="utf-8")
    os.utime(path, (instant.timestamp(), instant.timestamp()))


def test_behavior_matrix_daily_keeps_real_completed_run_and_expands_future(tmp_path, monkeypatch):
    completed_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        completed_at,
        "# Cron Job: daily task\n\n## Response\n\ndone\n",
    )

    result = jobs.list_job_occurrences(
        [_job()],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 24, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )

    assert [(item["scheduled_at"], item["status"]) for item in result] == [
        ("2026-07-20T01:00:00+00:00", "completed"),
        ("2026-07-21T01:00:00+00:00", "scheduled"),
        ("2026-07-22T01:00:00+00:00", "scheduled"),
        ("2026-07-23T01:00:00+00:00", "scheduled"),
    ]


def test_behavior_matrix_weekly_and_finite_repeat_stop_at_remaining_count(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    weekly = _job(
        id="jobweekly",
        schedule={"kind": "cron", "expr": "30 8 * * 1", "display": "weekly"},
        timezone="UTC",
        repeat={"times": 4, "completed": 2},
        next_run_at="2026-07-27T08:30:00+00:00",
    )
    result = jobs.list_job_occurrences(
        [weekly],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 8, 31, tzinfo=UTC),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )
    assert [item["scheduled_at"] for item in result] == [
        "2026-07-27T08:30:00+00:00",
        "2026-08-03T08:30:00+00:00",
    ]


def test_suppression_matrix_paused_task_keeps_history_but_has_no_future(tmp_path, monkeypatch):
    completed_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        completed_at,
        "# Cron Job: daily task\n\n## Response\n\ndone\n",
    )
    paused = _job(enabled=False, state="paused")
    result = jobs.list_job_occurrences(
        [paused],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 24, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )
    assert len(result) == 1
    assert result[0]["status"] == "completed"


def test_status_matrix_failed_run_is_not_marked_completed(tmp_path, monkeypatch):
    failed_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        failed_at,
        "# Cron Job: daily task (FAILED)\n\n## Error\n\nboom\n",
    )
    result = jobs.list_job_occurrences(
        [_job(enabled=False, state="paused")],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )
    assert result[0]["status"] == "failed"


def test_status_matrix_response_heading_cannot_forge_delivery_failure(tmp_path, monkeypatch):
    completed_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        completed_at,
        "# Cron Job: daily task\n\n## Response\n\nuser text\n## Delivery Error\nnot metadata\n",
    )
    result = jobs.list_job_occurrences(
        [_job(enabled=False, state="paused")],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )
    assert result[0]["status"] == "completed"


def test_security_matrix_symlinked_output_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    outside = tmp_path / "outside.md"
    outside.write_text("# Cron Job: leaked\n", encoding="utf-8")
    output = tmp_path / "cron" / "output" / "jobdaily"
    output.mkdir(parents=True)
    (output / "2026-07-20.md").symlink_to(outside)

    result = jobs.list_job_occurrences(
        [_job(enabled=False, state="paused")],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert result == []


def test_stale_next_run_projects_one_catch_up_then_reanchors_interval(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    interval = _job(
        id="jobinterval",
        schedule={"kind": "interval", "minutes": 60, "display": "hourly"},
        next_run_at="2026-07-21T08:00:00+00:00",
    )
    result = jobs.list_job_occurrences(
        [interval],
        datetime(2026, 7, 21, 7, tzinfo=UTC),
        datetime(2026, 7, 21, 13, tzinfo=UTC),
        now=datetime(2026, 7, 21, 10, 30, tzinfo=UTC),
    )
    assert [item["scheduled_at"] for item in result] == [
        "2026-07-21T10:30:00+00:00",
        "2026-07-21T11:30:00+00:00",
        "2026-07-21T12:30:00+00:00",
    ]
    assert result[0]["original_scheduled_at"] == "2026-07-21T08:00:00+00:00"


def test_structured_journal_keeps_status_after_verbose_output_pruning(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    job = _job(enabled=False, state="completed")
    run_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    jobs._append_job_occurrence(job, run_at, success=True, delivery_error="offline")
    result = jobs.list_job_occurrences(
        [job],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )
    assert [(item["scheduled_at"], item["status"]) for item in result] == [
        ("2026-07-20T01:00:00+00:00", "delivery_failed"),
    ]


def test_completed_run_keeps_scheduled_identity_and_records_actual_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    job = _job(enabled=False, state="completed")
    scheduled_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    actual_at = datetime(2026, 7, 20, 1, 10, tzinfo=UTC)
    jobs._append_job_occurrence(
        job,
        actual_at,
        scheduled_at=scheduled_at,
        success=True,
        delivery_error=None,
    )

    result = jobs.list_job_occurrences(
        [job],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=actual_at,
    )

    assert result == [{
        "id": "jobdaily:scheduled:2026-07-20T01:00:00+00:00",
        "job_id": "jobdaily",
        "scheduled_at": "2026-07-20T01:00:00+00:00",
        "actual_run_at": "2026-07-20T01:10:00+00:00",
        "status": "completed",
    }]


def test_structured_actual_time_dedupes_matching_verbose_output(tmp_path, monkeypatch):
    scheduled_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    actual_at = datetime(2026, 7, 20, 1, 10, tzinfo=UTC)
    output_at = datetime(2026, 7, 20, 1, 5, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        output_at,
        "# Cron Job: daily task\n\n## Response\n\ndone\n",
    )
    job = _job(enabled=False, state="completed")
    jobs._append_job_occurrence(
        job,
        actual_at,
        scheduled_at=scheduled_at,
        output_filename="2026-07-20_09-00-00.md",
        success=True,
        delivery_error=None,
    )

    result = jobs.list_job_occurrences(
        [job],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=actual_at,
    )

    assert len(result) == 1
    assert result[0]["scheduled_at"] == "2026-07-20T01:00:00+00:00"
    assert result[0]["actual_run_at"] == "2026-07-20T01:10:00+00:00"


def test_cross_day_journal_index_suppresses_markdown_phantom(tmp_path, monkeypatch):
    scheduled_at = datetime(2026, 7, 20, 23, 59, tzinfo=UTC)
    actual_at = datetime(2026, 7, 21, 0, 5, tzinfo=UTC)
    filename = "2026-07-21_00-05-00.md"
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        filename,
        actual_at,
        "# Cron Job: daily task\n\n## Response\n\ndone\n",
    )
    job = _job(enabled=False, state="completed")
    jobs._append_job_occurrence(
        job,
        actual_at,
        scheduled_at=scheduled_at,
        output_filename=filename,
        success=True,
        delivery_error=None,
    )

    next_day = jobs.list_job_occurrences(
        [job],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=actual_at,
    )

    assert next_day == []


def test_stale_catchup_freezes_same_identity_for_preview_and_completion(tmp_path, monkeypatch):
    now = datetime(2026, 7, 21, 10, 30, tzinfo=UTC)
    stale = _job(
        timezone="UTC",
        next_run_at="2026-07-21T08:00:00+00:00",
        schedule={"kind": "cron", "expr": "0 8 * * *", "display": "daily"},
    )
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(jobs, "load_jobs", lambda: [dict(stale)])
    monkeypatch.setattr(jobs, "save_jobs", lambda _jobs: None)
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")

    due = jobs._get_due_jobs_locked()
    assert due[0]["_occurrence_triggered_at"] == now.isoformat()
    preview = jobs.list_job_occurrences(
        due,
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=now,
    )
    jobs._append_job_occurrence(
        stale,
        now + timedelta(minutes=10),
        scheduled_at=datetime.fromisoformat(due[0]["_occurrence_triggered_at"]),
        success=True,
        delivery_error=None,
    )
    completed = jobs.list_job_occurrences(
        [{**stale, "enabled": False, "state": "completed"}],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=now + timedelta(minutes=10),
    )

    assert preview[0]["id"] == completed[0]["id"]
    assert preview[0]["scheduled_at"] == completed[0]["scheduled_at"] == now.isoformat()


def test_behavior_matrix_real_store_keeps_recurring_occurrence_visible_while_running(tmp_path, monkeypatch):
    """The API reloads jobs.json after the ticker advances to tomorrow."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime(2026, 7, 21, 9, 0, 5, tzinfo=UTC)
    job = _job(
        timezone="UTC",
        next_run_at="2026-07-21T09:00:00+00:00",
        schedule={"kind": "cron", "expr": "0 9 * * *", "display": "daily"},
    )
    jobs.save_jobs([job])
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    due = jobs.get_due_jobs()
    assert due[0]["_occurrence_triggered_at"] == "2026-07-21T09:00:00+00:00"
    assert jobs.advance_next_run(job["id"]) is True

    persisted = jobs.load_jobs()[0]
    assert persisted["next_run_at"] != "2026-07-21T09:00:00+00:00"
    preview = jobs.list_job_occurrences(
        [persisted],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=now,
    )
    assert [(item["scheduled_at"], item["status"]) for item in preview] == [
        ("2026-07-21T09:00:00+00:00", "scheduled"),
    ]

    finished_at = now + timedelta(minutes=1)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: finished_at)
    jobs.mark_job_run(
        job["id"],
        success=True,
        scheduled_at=due[0]["_occurrence_triggered_at"],
    )
    terminal = jobs.load_jobs()[0]
    assert terminal.get("in_flight_occurrence") is None
    completed = jobs.list_job_occurrences(
        [terminal],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=finished_at,
    )
    assert completed[0]["id"] == preview[0]["id"]
    assert completed[0]["status"] == "completed"


def test_behavior_matrix_real_store_keeps_stale_catchup_visible_after_fast_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = datetime(2026, 7, 21, 10, 30, tzinfo=UTC)
    job = _job(
        timezone="UTC",
        next_run_at="2026-07-21T08:00:00+00:00",
        schedule={"kind": "cron", "expr": "0 8 * * *", "display": "daily"},
    )
    jobs.save_jobs([job])
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    due = jobs.get_due_jobs()
    persisted = jobs.load_jobs()[0]
    preview = jobs.list_job_occurrences(
        [persisted],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=now,
    )

    assert due[0]["_occurrence_triggered_at"] == now.isoformat()
    assert preview[0]["scheduled_at"] == now.isoformat()
    assert preview[0]["original_scheduled_at"] == "2026-07-21T08:00:00+00:00"


def test_behavior_matrix_in_flight_lease_is_bounded_and_configurable(monkeypatch):
    now = datetime(2026, 7, 21, 10, 0, tzinfo=UTC)
    job = _job()
    job["in_flight_occurrence"] = {
        "scheduled_at": "2026-07-21T09:00:00+00:00",
        "claimed_at": "2026-07-21T09:40:00+00:00",
    }

    # Default lease is 15 minutes, so a crashed execution cannot leave APP
    # polling an in-flight calendar node for the old 24-hour window.
    assert jobs._in_flight_occurrence(job, now) is None

    # Operator overrides are accepted but clamped to one hour.
    monkeypatch.setenv("HERMES_CRON_OCCURRENCE_LEASE_SECONDS", "999999")
    assert jobs._in_flight_occurrence(job, now) is not None
    job["in_flight_occurrence"]["claimed_at"] = "2026-07-21T08:59:59+00:00"
    assert jobs._in_flight_occurrence(job, now) is None


def test_upgrade_matrix_first_journal_entry_does_not_hide_legacy_history(tmp_path, monkeypatch):
    legacy_at = datetime(2026, 7, 19, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-19_09-00-00.md",
        legacy_at,
        "# Cron Job: daily task\n\n## Response\n\nlegacy done\n",
    )
    job = _job(enabled=False, state="completed")
    jobs._append_job_occurrence(
        job,
        datetime(2026, 7, 20, 1, 0, tzinfo=UTC),
        success=True,
        delivery_error=None,
    )

    result = jobs.list_job_occurrences(
        [job],
        datetime(2026, 7, 19, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )

    assert [(item["scheduled_at"], item["status"]) for item in result] == [
        ("2026-07-19T01:00:00+00:00", "completed"),
        ("2026-07-20T01:00:00+00:00", "completed"),
    ]


def test_corrupt_journal_falls_back_to_legacy_and_reports_truncation(tmp_path, monkeypatch):
    legacy_at = datetime(2026, 7, 19, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-19_09-00-00.md",
        legacy_at,
        "# Cron Job: daily task\n\n## Response\n\nlegacy done\n",
    )
    (tmp_path / "cron" / "output" / "jobdaily" / ".occurrences.json").write_text(
        "not-json",
        encoding="utf-8",
    )

    result = jobs.job_occurrence_projection(
        [_job(enabled=False, state="completed")],
        datetime(2026, 7, 19, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )

    assert [(item["scheduled_at"], item["status"]) for item in result["occurrences"]] == [
        ("2026-07-19T01:00:00+00:00", "completed"),
    ]
    assert result["history_truncated"] is True


def test_unlimited_verbose_retention_does_not_report_false_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_cron_output_keep", lambda: 0)
    run_at = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
    _seed_output(
        tmp_path,
        monkeypatch,
        "jobdaily",
        "2026-07-20_09-00-00.md",
        run_at,
        "# Cron Job: daily task\n\n## Response\n\ndone\n",
    )

    result = jobs.job_occurrence_projection(
        [_job(enabled=False, state="completed")],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
    )

    assert result["history_truncated"] is False
    assert len(result["occurrences"]) == 1


def test_query_limit_and_scan_cap_report_incomplete_history(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_cron_output_keep", lambda: 0)
    for index in range(3):
        run_at = datetime(2026, 7, 20, 1, index, tzinfo=UTC)
        _seed_output(
            tmp_path,
            monkeypatch,
            "jobdaily",
            f"2026-07-20_09-0{index}-00.md",
            run_at,
            "# Cron Job: daily task\n\n## Response\n\ndone\n",
        )
    job = _job(enabled=False, state="completed")

    limited = jobs.job_occurrence_projection(
        [job],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
        limit=2,
    )
    assert len(limited["occurrences"]) == 2
    assert limited["history_truncated"] is True

    monkeypatch.setattr(jobs, "_MAX_OCCURRENCE_HISTORY_FILES", 1)
    scan_capped = jobs.job_occurrence_projection(
        [job],
        datetime(2026, 7, 20, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
        now=datetime(2026, 7, 20, 12, tzinfo=UTC),
        limit=10,
    )
    assert scan_capped["history_truncated"] is True


def test_dst_fallback_preview_matches_scheduler_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    base = datetime.fromisoformat("2026-11-01T01:15:00-05:00")
    schedule = {"kind": "cron", "expr": "30 1 * * *", "display": "daily"}
    monkeypatch.setattr(jobs, "_hermes_now", lambda: base)

    scheduler_next = jobs.compute_next_run(schedule, tz_name="America/New_York")
    preview_next = jobs._next_preview_instant(
        _job(schedule=schedule, timezone="America/New_York"),
        base.astimezone(UTC),
    )

    assert scheduler_next is not None
    assert preview_next == datetime.fromisoformat(scheduler_next).astimezone(UTC)


def test_corrupt_interval_fails_closed_without_breaking_other_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    corrupt = _job(id="bad", schedule={"kind": "interval", "minutes": float("inf")})
    healthy = _job(id="good", next_run_at="2026-07-21T01:00:00+00:00")
    result = jobs.list_job_occurrences(
        [corrupt, healthy],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    assert result
    assert all(item["job_id"] == "good" for item in result)


def test_limit_matrix_high_frequency_job_cannot_starve_other_task(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    noisy = _job(
        id="noisy",
        schedule={"kind": "interval", "minutes": 1},
        next_run_at="2026-07-21T00:00:00+00:00",
    )
    ordinary = _job(id="ordinary", next_run_at="2026-07-21T01:00:00+00:00")
    result = jobs.list_job_occurrences(
        [noisy, ordinary],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=datetime(2026, 7, 20, tzinfo=UTC),
        limit=10,
    )
    assert {item["job_id"] for item in result} == {"noisy", "ordinary"}


def test_claimed_oneshot_stays_visible_until_terminal_run_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "_output_dir", lambda: tmp_path / "cron" / "output")
    claimed = _job(
        id="claimed",
        schedule={"kind": "once", "run_at": "2026-07-21T06:02:00+00:00"},
        repeat={"times": 1, "completed": 1},
        next_run_at="2026-07-21T06:02:00+00:00",
        last_run_at=None,
    )
    result = jobs.list_job_occurrences(
        [claimed],
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        now=datetime(2026, 7, 21, 6, 3, tzinfo=UTC),
    )
    assert len(result) == 1
    assert result[0]["status"] == "scheduled"
    assert result[0]["scheduled_at"] == "2026-07-21T06:03:00+00:00"
