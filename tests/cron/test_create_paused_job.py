"""Atomic paused creation reuses the original store and scheduler transitions."""

from datetime import datetime, timedelta, timezone
import random

import pytest

from cron import jobs


@pytest.fixture
def store(tmp_path, monkeypatch):
    now = [datetime(2030, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now[0])
    with jobs.use_cron_store(tmp_path):
        yield now


@pytest.mark.parametrize("enabled", [False, True])
def test_creation_persists_requested_initial_state(store, enabled):
    job = jobs.create_job("observe", "1m", enabled=enabled)
    persisted = jobs.get_job(job["id"])
    assert persisted["enabled"] is enabled
    assert persisted["state"] == ("scheduled" if enabled else "paused")
    assert bool(persisted["paused_at"]) is (not enabled)
    assert persisted["revision"] == 0


@pytest.mark.parametrize("enabled", [None, 0, 1, "false", [], {}])
def test_invalid_enabled_never_persists(store, enabled):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        jobs.create_job("observe", "1m", enabled=enabled)
    assert jobs.load_jobs() == []


def test_legacy_creation_remains_scheduled(store):
    job = jobs.create_job("ordinary", "1m")
    assert job["enabled"] is True
    assert job["state"] == "scheduled"
    assert job["paused_at"] is None


def test_paused_once_survives_due_scan_then_explicit_trigger_flow(store):
    job = jobs.create_job("observe", "1m", enabled=False)
    store[0] += timedelta(minutes=2)
    assert jobs.get_due_jobs() == []
    assert jobs.claim_job_for_fire(job["id"]) is False
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 0
    assert jobs.get_job(job["id"])["state"] == "paused"
    jobs.trigger_job(job["id"])
    due = jobs.get_due_jobs()
    assert [item["id"] for item in due] == [job["id"]]
    jobs.mark_job_run(job["id"], success=True)
    assert jobs.get_due_jobs() == []
    assert jobs.get_job(job["id"])["state"] == "completed"
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1


def test_randomized_paused_creation_never_becomes_due_before_trigger(store):
    rng = random.Random(432)
    pending = set()
    for _ in range(100):
        action = rng.choice(("create", "scan", "trigger", "remove"))
        if action == "create" or not pending:
            pending.add(jobs.create_job("observe", "1m", enabled=False)["id"])
        elif action == "remove":
            target = rng.choice(sorted(pending))
            jobs.remove_job(target)
            pending.remove(target)
        elif action == "trigger":
            target = rng.choice(sorted(pending))
            jobs.trigger_job(target)
            assert [item["id"] for item in jobs.get_due_jobs()] == [target]
            jobs.mark_job_run(target, success=True)
            pending.remove(target)
        store[0] += timedelta(seconds=rng.randint(1, 120))
        assert jobs.get_due_jobs() == []
        for target in pending:
            saved = jobs.get_job(target)
            assert saved["state"] == "paused"
            assert saved["enabled"] is False
            assert saved["repeat"]["completed"] == 0


def test_first_publication_is_already_paused(store, monkeypatch):
    original = jobs.save_jobs
    publications = []

    def save(records):
        publications.append([(item["enabled"], item["state"]) for item in records])
        return original(records)

    monkeypatch.setattr(jobs, "save_jobs", save)
    jobs.create_job("observe", "1m", enabled=False)
    assert publications == [[(False, "paused")]]
