"""Unit tests for the Chronos NAS-mediated cron provider (Phase 4D).

All NAS calls are mocked — ZERO live network. These prove:
  - is_available is config-only (no network), false without config.
  - one-shot arming sends the right provision payload (incl. sub-minute fires —
    the agent owns the time, so there's no 1-minute floor).
  - reconcile arms missing, cancels orphaned, skips paused.
  - fire_due re-arms the next one-shot after a successful run, and repeat-N
    (job gone) stops re-arming.
"""

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def chronos(monkeypatch):
    """A ChronosCronScheduler with a fake NAS client capturing calls."""
    from plugins.cron_providers.chronos import ChronosCronScheduler

    class FakeClient:
        def __init__(self):
            self.provisions = []
            self.cancels = []
            self._armed = []

        def provision(self, *, job_id, fire_at, agent_callback_url, dedup_key):
            self.provisions.append({
                "job_id": job_id, "fire_at": fire_at,
                "agent_callback_url": agent_callback_url, "dedup_key": dedup_key,
            })
            return {"schedule_id": f"sched-{job_id}"}

        def cancel(self, *, job_id):
            self.cancels.append(job_id)
            return {}

        def list_armed(self):
            return list(self._armed)

    prov = ChronosCronScheduler()
    fake = FakeClient()
    prov._client = fake
    # callback_url is read via _cfg; patch the module helper to avoid config.
    monkeypatch.setattr("plugins.cron_providers.chronos._cfg",
                        lambda *k, default="": "https://agent.example/" if k[-1] == "callback_url" else "https://portal.test")
    return prov, fake


# -- is_available -------------------------------------------------------------

def test_is_available_false_without_config(temp_home, monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    monkeypatch.setattr("plugins.cron_providers.chronos._cfg", lambda *k, default="": "")
    assert ChronosCronScheduler().is_available() is False


def test_is_available_true_with_config_and_token(temp_home, monkeypatch):
    import plugins.cron_providers.chronos as mod
    from plugins.cron_providers.chronos import ChronosCronScheduler

    monkeypatch.setattr(mod, "_cfg", lambda *k, default="": "https://x" )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state",
                        lambda pid: {"access_token": "tok"})
    assert ChronosCronScheduler().is_available() is True


def test_is_available_makes_no_network(temp_home, monkeypatch):
    """is_available must not construct the NAS client / hit network."""
    import plugins.cron_providers.chronos as mod
    from plugins.cron_providers.chronos import ChronosCronScheduler

    monkeypatch.setattr(mod, "_cfg", lambda *k, default="": "https://x")
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state",
                        lambda pid: {"access_token": "tok"})
    p = ChronosCronScheduler()

    def explode():
        raise AssertionError("is_available must not build the NAS client")

    monkeypatch.setattr(p, "_get_client", explode)
    assert p.is_available() is True  # did not call _get_client


# -- arming -------------------------------------------------------------------

def test_arm_one_shot_sends_provision(chronos):
    prov, fake = chronos
    prov._arm_one_shot({"id": "j1", "next_run_at": "2026-06-18T12:00:00+00:00"})

    assert len(fake.provisions) == 1
    p = fake.provisions[0]
    assert p["job_id"] == "j1"
    assert p["fire_at"] == "2026-06-18T12:00:00+00:00"
    assert p["dedup_key"] == "j1:2026-06-18T12:00:00+00:00"
    assert p["agent_callback_url"] == "https://agent.example/"


def test_arm_one_shot_preserves_sub_minute_fire(chronos):
    """Sub-minute fire times survive — the agent owns the time, so there's no
    1-minute scheduler floor."""
    prov, fake = chronos
    prov._arm_one_shot({"id": "j2", "next_run_at": "2026-06-18T12:00:30+00:00"})
    assert fake.provisions[0]["fire_at"] == "2026-06-18T12:00:30+00:00"


def test_arm_one_shot_noop_without_next_run(chronos):
    prov, fake = chronos
    prov._arm_one_shot({"id": "j3", "next_run_at": None})
    assert fake.provisions == []


# -- reconcile ----------------------------------------------------------------

def test_reconcile_arms_all_enabled(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    jobs = [
        {"id": "a", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "scheduled"},
        {"id": "b", "enabled": True, "next_run_at": "2026-06-18T12:05:00+00:00", "state": "scheduled"},
    ]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next(j for j in jobs if j["id"] == jid))

    prov.reconcile()
    assert {p["job_id"] for p in fake.provisions} == {"a", "b"}
    assert fake.cancels == []


def test_reconcile_cancels_orphan_arms_desired(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    # NAS already has a stale arm for deleted job "gone".
    prov._armed = {"gone": "2026-06-18T11:00:00+00:00"}
    jobs = [{"id": "a", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "scheduled"}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next((j for j in jobs if j["id"] == jid), None))

    prov.reconcile()
    assert [p["job_id"] for p in fake.provisions] == ["a"]
    assert fake.cancels == ["gone"]


def test_reconcile_skips_paused(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    jobs = [{"id": "p", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "paused"}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: next((j for j in jobs if j["id"] == jid), None))

    prov.reconcile()
    assert fake.provisions == []


def test_reconcile_skips_already_armed_same_time(temp_home, chronos, monkeypatch):
    prov, fake = chronos
    prov._armed = {"a": "2026-06-18T12:00:00+00:00"}
    jobs = [{"id": "a", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "scheduled"}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: jobs[0])

    prov.reconcile()
    assert fake.provisions == []  # already armed at the same time → no re-arm


def test_calendar_reconcile_requires_remote_observation(chronos, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda _jid: job)
    fake._armed = [{"job_id": job["id"], "fire_at": job["next_run_at"]}]
    result = prov.reconcile_calendar_job(job["id"], "upsert", 2)
    assert result["status"] == "armed"
    assert result["observed_fire_at"] == job["next_run_at"]
    fake._armed = [{"job_id": job["id"], "fire_at": "2026-07-15T02:00:00Z"}]
    with pytest.raises(RuntimeError, match="durably observed"):
        prov.reconcile_calendar_job(job["id"], "upsert", 2)


def test_calendar_cancel_requires_remote_absence(chronos):
    prov, fake = chronos
    fake._armed = []
    assert prov.reconcile_calendar_job("cal-alert", "delete", 2)["status"] == "cancelled"
    assert fake.cancels == ["cal-alert"]


def test_stale_calendar_delete_flow_preserves_newer_projection(chronos, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    monkeypatch.setattr(
        "cron.jobs.get_job_raw",
        lambda _jid: managed_job(calendar_projection_revision=3),
    )

    result = prov.reconcile_calendar_job("cal-alert-abc", "delete", 2)

    assert result["status"] == "superseded"
    assert fake.cancels == []


def test_generic_reconcile_flow_preserves_managed_calendar_recovery_arm(
    temp_home, chronos, monkeypatch,
):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(
        enabled=True,
        state="scheduled",
        next_run_at="2026-07-15T01:00:00Z",
    )
    retry_at = "2026-07-15T01:02:03Z"
    fake._armed = [{"job_id": job["id"], "fire_at": retry_at}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: [job])

    prov.reconcile()

    assert fake.provisions == []
    assert fake.cancels == []
    assert fake._armed == [{"job_id": job["id"], "fire_at": retry_at}]


def test_generic_reconcile_cancels_paused_managed_calendar_arm(
    temp_home, chronos, monkeypatch,
):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(
        enabled=True,
        state="paused",
        next_run_at="2026-07-15T01:00:00Z",
    )
    fake._armed = [{"job_id": job["id"], "fire_at": job["next_run_at"]}]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: [job])

    prov.reconcile()

    assert fake.cancels == [job["id"]]


def test_calendar_recovery_arm_uses_attempt_dedupe_and_observed_time(chronos, monkeypatch):
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    monkeypatch.setattr("cron.jobs.get_job_raw", lambda _jid: job)
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    intent = {
        "job_id": job["id"], "projection_revision": 2,
        "attempt_sequence": 4, "dedupe_key": "d" * 64,
        "retry_at": "2026-07-15T01:02:03Z", "deadline_at": "2026-07-15T01:05:00Z",
    }
    result = prov.reconcile_calendar_recovery_arm(intent)
    assert result["observed_fire_at"] == intent["retry_at"]
    assert fake.provisions[-1]["dedup_key"] == "d" * 64
    assert fake.provisions[-1]["fire_at"] == intent["retry_at"]


# -- fire_due re-arm ----------------------------------------------------------

def test_fire_due_rearms_next_oneshot(chronos, monkeypatch):
    prov, fake = chronos
    # super().fire_due runs the job; stub the ABC default to "ran".
    monkeypatch.setattr("cron.scheduler_provider.CronScheduler.fire_due",
                        lambda self, jid, **kw: True)
    monkeypatch.setattr("cron.jobs.get_job",
                        lambda jid: {"id": jid, "enabled": True, "next_run_at": "2026-06-18T12:05:00+00:00"})

    assert prov.fire_due("j1") is True
    assert [p["job_id"] for p in fake.provisions] == ["j1"]
    assert fake.provisions[0]["fire_at"] == "2026-06-18T12:05:00+00:00"


def test_fire_due_no_rearm_when_job_gone(chronos, monkeypatch):
    """repeat-N exhausted / one-shot completed → mark_job_run deleted the job →
    get_job None → no re-arm (the schedule stops cleanly)."""
    prov, fake = chronos
    monkeypatch.setattr("cron.scheduler_provider.CronScheduler.fire_due",
                        lambda self, jid, **kw: True)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: None)

    assert prov.fire_due("j1") is True
    assert fake.provisions == []


def test_fire_due_no_rearm_when_claim_lost(chronos, monkeypatch):
    """If the run didn't happen (claim lost), don't re-arm."""
    prov, fake = chronos
    monkeypatch.setattr("cron.scheduler_provider.CronScheduler.fire_due",
                        lambda self, jid, **kw: False)

    assert prov.fire_due("j1") is False
    assert fake.provisions == []
