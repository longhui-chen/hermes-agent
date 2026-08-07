"""Tests for the store-level CAS fire claim (Phase 4C).

`claim_job_for_fire` gives multi-machine at-most-once semantics when an external
scheduler (Chronos) fires a job: across N gateway replicas, exactly ONE wins the
claim for a given fire. Single-machine deployments always win (unaffected).

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
import pytest
from datetime import datetime, timezone


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # cron.jobs caches no home at import; get_hermes_home() reads the env live.
    yield tmp_path


def test_claim_succeeds_once_then_blocks(temp_home):
    """First claim for a fire wins; a second claim for the same fire loses, and
    next_run_at is advanced (a re-delivery for the old time can't re-fire)."""
    from cron.jobs import create_job, claim_job_for_fire, get_job

    job = create_job(prompt="x", schedule="every 5m", name="t")
    jid = job["id"]
    before = get_job(jid)["next_run_at"]

    assert claim_job_for_fire(jid) is True
    assert claim_job_for_fire(jid) is False
    assert get_job(jid)["next_run_at"] != before


def test_stale_claim_is_reclaimable(temp_home, monkeypatch):
    """A claim older than the TTL is overwritten — the fire isn't stuck forever
    if the winning machine crashed before mark_job_run cleared the claim."""
    from cron.jobs import create_job, claim_job_for_fire

    job = create_job(prompt="x", schedule="every 5m", name="s")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    # With a 0s TTL, the existing claim is always considered stale.
    assert claim_job_for_fire(jid, claim_ttl_seconds=0) is True


def test_stale_external_retry_reuses_occurrence_without_advancing_again(temp_home, monkeypatch):
    """Reclaiming one webhook occurrence must not consume the next schedule."""
    import cron.jobs as jobs

    fired = datetime(2026, 7, 21, 9, 0, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: fired)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    jobs.update_job(job["id"], {"next_run_at": "2026-07-21T09:00:00+00:00"})

    assert jobs.claim_job_for_fire(job["id"]) is True
    after_first = jobs.get_job(job["id"])
    first_next_run = after_first["next_run_at"]
    first_occurrence = dict(after_first["in_flight_occurrence"])

    assert jobs.claim_job_for_fire(job["id"], claim_ttl_seconds=0) is True
    after_retry = jobs.get_job(job["id"])
    assert after_retry["next_run_at"] == first_next_run
    assert after_retry["in_flight_occurrence"]["scheduled_at"] == first_occurrence["scheduled_at"]


def test_explicit_run_now_claim_uses_manual_trigger_instead_of_future_schedule(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 16, 30, tzinfo=timezone.utc)
    future = "2026-07-22T09:00:00+00:00"
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    jobs.update_job(job["id"], {"next_run_at": future})

    assert jobs.claim_job_for_fire(job["id"], triggered_at=now.isoformat()) is True
    claimed = jobs.get_job(job["id"])
    assert claimed["in_flight_occurrence"]["scheduled_at"] == now.isoformat()
    assert claimed["fire_claim"]["scheduled_at"] == now.isoformat()


def test_completed_external_fire_at_is_idempotent_across_later_completion(temp_home, monkeypatch):
    """A lost HTTP response must not make a completed webhook run tomorrow now."""
    import cron.jobs as jobs

    current = [datetime(2026, 7, 21, 9, 0, 5, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current[0])
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    jobs.update_job(job["id"], {"next_run_at": "2026-07-21T09:00:00+00:00"})

    fire_a = "2026-07-21T09:00:00+00:00"
    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_a) is True
    claimed = jobs.get_job(job["id"])
    occurrence = claimed["in_flight_occurrence"]["scheduled_at"]
    jobs.mark_job_run(job["id"], success=True, scheduled_at=occurrence)
    current[0] = datetime(2026, 7, 22, 9, 0, 5, tzinfo=timezone.utc)
    fire_b = "2026-07-22T09:00:00+00:00"
    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_b) is True
    claimed_b = jobs.get_job(job["id"])
    jobs.mark_job_run(
        job["id"],
        success=True,
        scheduled_at=claimed_b["in_flight_occurrence"]["scheduled_at"],
    )
    next_run = jobs.get_job(job["id"])["next_run_at"]

    # A delayed retry of A remains suppressed even after B became the latest.
    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_a) is False
    completed = jobs.get_job(job["id"])
    assert completed["next_run_at"] == next_run
    assert completed["last_completed_external_fire_at"] == fire_b


def test_external_fire_flow_rejects_arm_from_abandoned_schedule(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 9, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    jobs.update_job(job["id"], {"next_run_at": "2026-07-22T09:00:00+00:00"})

    assert jobs.claim_job_for_fire(
        job["id"], fire_at="2026-07-21T09:00:00+00:00",
    ) is False
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_external_fire_retry_matches_valid_in_flight_occurrence(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 9, 0, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    fire_at = "2026-07-21T09:00:00+00:00"
    jobs.update_job(job["id"], {"next_run_at": fire_at})

    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is True
    advanced = jobs.get_job(job["id"])["next_run_at"]
    assert advanced != fire_at
    assert jobs.claim_job_for_fire(
        job["id"], fire_at=fire_at, claim_ttl_seconds=0,
    ) is True
    assert jobs.get_job(job["id"])["next_run_at"] == advanced


def test_external_fire_retry_survives_expired_occurrence_lease(temp_home, monkeypatch):
    """A crashed claimant remains recoverable after the display lease expires."""
    import cron.jobs as jobs

    current = [datetime(2026, 7, 21, 9, 0, 5, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current[0])
    job = jobs.create_job(prompt="x", schedule="every 5m", name="frequent", timezone="UTC")
    fire_at = "2026-07-21T09:00:00+00:00"
    jobs.update_job(job["id"], {"next_run_at": fire_at})

    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is True
    advanced = jobs.get_job(job["id"])["next_run_at"]
    current[0] = datetime(2026, 7, 21, 9, 20, 5, tzinfo=timezone.utc)

    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is True
    assert jobs.get_job(job["id"])["next_run_at"] == advanced


def test_reschedule_invalidates_crashed_external_fire_identity(temp_home, monkeypatch):
    """A stale claim is recoverable only until the user replaces its schedule."""
    import cron.jobs as jobs

    current = [datetime(2026, 7, 21, 9, 0, 5, tzinfo=timezone.utc)]
    monkeypatch.setattr(jobs, "_hermes_now", lambda: current[0])
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    fire_at = "2026-07-21T09:00:00+00:00"
    jobs.update_job(job["id"], {"next_run_at": fire_at})
    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is True

    jobs.update_job(job["id"], {"schedule": "0 10 * * *"})
    current[0] = datetime(2026, 7, 21, 9, 20, 5, tzinfo=timezone.utc)

    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is False
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_external_fire_rejects_future_occurrence(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")

    with pytest.raises(ValueError, match="outside the accepted execution window"):
        jobs.claim_job_for_fire(job["id"], fire_at="2026-07-22T09:00:00+00:00")
    assert jobs.get_job(job["id"]).get("fire_claim") is None


def test_external_fire_clock_skew_advances_from_protocol_fire_at(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 8, 59, tzinfo=timezone.utc)
    fire_at = "2026-07-21T09:00:00+00:00"
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="0 9 * * *", name="daily", timezone="UTC")
    jobs.update_job(job["id"], {"next_run_at": fire_at})

    assert jobs.claim_job_for_fire(job["id"], fire_at=fire_at) is True
    claimed = jobs.get_job(job["id"])
    assert claimed["next_run_at"] == "2026-07-22T09:00:00+00:00"
    jobs.mark_job_run(
        job["id"],
        success=True,
        scheduled_at=claimed["in_flight_occurrence"]["scheduled_at"],
    )

    assert jobs.get_job(job["id"])["next_run_at"] == "2026-07-22T09:00:00+00:00"


def test_external_fire_watermark_rejects_arbitrarily_old_completed_fire(temp_home, monkeypatch):
    import cron.jobs as jobs

    now = datetime(2026, 7, 21, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    job = jobs.create_job(prompt="x", schedule="every 1m", name="minute")
    # Represents far more than the removed 256-entry history: only the
    # monotonic completion watermark is required to reject every older retry.
    jobs.update_job(job["id"], {"last_completed_external_fire_at": now.isoformat()})

    assert jobs.claim_job_for_fire(
        job["id"], fire_at="2026-07-21T04:00:00+00:00"
    ) is False


def test_mark_job_run_clears_claim(temp_home):
    """After a recurring job completes, its claim is cleared so the next fire
    can be claimed again."""
    from cron.jobs import create_job, claim_job_for_fire, mark_job_run, get_job

    job = create_job(prompt="x", schedule="every 5m", name="c")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    assert get_job(jid).get("fire_claim") is not None

    mark_job_run(jid, success=True)
    assert get_job(jid).get("fire_claim") is None
    # …and the re-armed recurring job is claimable again.
    assert claim_job_for_fire(jid) is True
