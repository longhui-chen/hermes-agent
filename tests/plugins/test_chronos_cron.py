"""Unit tests for the Chronos NAS-mediated cron provider (Phase 4D).

All NAS calls are mocked — ZERO live network. These prove:
  - is_available is config-only (no network), false without config.
  - one-shot arming sends the right provision payload (incl. sub-minute fires —
    the agent owns the time, so there's no 1-minute floor).
  - reconcile arms missing, cancels orphaned, skips paused.
  - fire_due re-arms the next one-shot after a successful run, and repeat-N
    (job gone) stops re-arming.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading
import time

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


def test_calendar_reconcile_requires_remote_observation(temp_home, chronos):
    from cron.jobs import save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    fake._armed = [{"job_id": job["id"], "fire_at": job["next_run_at"]}]
    result = prov.reconcile_calendar_job(job["id"], "upsert", 2)
    assert result["status"] == "armed"
    assert result["observed_fire_at"] == job["next_run_at"]
    fake._armed = [{"job_id": job["id"], "fire_at": "2026-07-15T02:00:00Z"}]
    with pytest.raises(RuntimeError, match="durably observed"):
        prov.reconcile_calendar_job(job["id"], "upsert", 2)


def test_delayed_same_revision_upsert_preserves_authoritative_recovery_arm(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    retry_at = "2026-07-15T01:02:03Z"
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    prov.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 4, "dedupe_key": "d" * 64,
        "retry_at": retry_at, "deadline_at": "2026-07-15T01:05:00Z",
    })
    fake.provisions.clear()

    restarted = type(prov)()
    restarted._client = fake
    result = restarted.reconcile_calendar_job(job["id"], "upsert", 2)

    assert result == {
        "status": "recovery_preserved",
        "provider": "chronos",
        "observed_fire_at": retry_at,
    }
    assert fake.provisions == []
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == {
        "projection_revision": 2,
        "delivery_generation": 1,
        "retry_at": retry_at,
        "attempt_sequence": 4,
    }


def test_restart_rearms_durable_recovery_after_crash_before_nas_provision(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    retry_at = "2026-07-15T01:02:03.123400Z"
    save_jobs([job])
    original_provision = fake.provision

    def crash_before_nas(**_kwargs):
        raise RuntimeError("simulated crash before NAS provision")

    fake.provision = crash_before_nas
    with pytest.raises(RuntimeError, match="simulated crash"):
        prov.reconcile_calendar_recovery_arm({
            "job_id": job["id"], "projection_revision": 2,
            "delivery_generation": 1, "attempt_sequence": 4, "dedupe_key": "d" * 64,
            "retry_at": retry_at, "deadline_at": "2026-07-15T01:05:00Z",
        })
    marker = {
        "projection_revision": 2,
        "delivery_generation": 1,
        "retry_at": retry_at,
        "attempt_sequence": 4,
    }
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == marker

    restarted = type(prov)()
    restarted._client = fake
    with pytest.raises(RuntimeError, match="simulated crash"):
        restarted.reconcile_calendar_job(job["id"], "upsert", 2)
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == marker

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    result = restarted.reconcile_calendar_job(job["id"], "upsert", 2)

    assert result["status"] == "recovery_preserved"
    assert fake.provisions[-1]["fire_at"] == retry_at
    assert fake.provisions[-1]["fire_at"] != job["next_run_at"]
    assert fake.provisions[-1]["dedup_key"] == hashlib.sha256(
        f"calendar-recovery\x00{job['id']}\x002\x001\x004\x00{retry_at}".encode()
    ).hexdigest()
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == marker


def test_new_revision_upsert_supersedes_older_recovery_arm(temp_home, chronos):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    prov.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 4, "dedupe_key": "d" * 64,
        "retry_at": "2026-07-15T01:02:03Z", "deadline_at": "2026-07-15T01:05:00Z",
    })
    current = get_job_raw(job["id"])
    current["calendar_projection_revision"] = 3
    current["next_run_at"] = "2026-07-15T02:00:00Z"
    save_jobs([current])
    fake.provisions.clear()

    result = prov.reconcile_calendar_job(current["id"], "upsert", 3)

    assert result["status"] == "armed"
    assert fake.provisions[-1]["fire_at"] == current["next_run_at"]
    assert "chronos" not in get_job_raw(job["id"]).get("provider_state", {})


def test_older_upsert_revision_acks_superseded_and_future_revision_fails(
    temp_home, chronos,
):
    from cron.jobs import save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(calendar_projection_revision=3)
    save_jobs([job])

    assert prov.reconcile_calendar_job(job["id"], "upsert", 2) == {
        "status": "superseded",
        "provider": "chronos",
    }
    assert fake.provisions == []
    with pytest.raises(RuntimeError, match="projection revision mismatch"):
        prov.reconcile_calendar_job(job["id"], "upsert", 4)


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

    result = prov.reconcile_calendar_job("cal-alert-" + "a" * 32, "delete", 2)

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


def test_calendar_recovery_arm_uses_attempt_dedupe_and_observed_time(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job
    prov, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    intent = {
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 4, "dedupe_key": "d" * 64,
        "retry_at": "2026-07-15T01:02:03Z", "deadline_at": "2026-07-15T01:05:00Z",
    }
    result = prov.reconcile_calendar_recovery_arm(intent)
    assert result["observed_fire_at"] == intent["retry_at"]
    assert fake.provisions[-1]["dedup_key"] == "d" * 64
    assert fake.provisions[-1]["fire_at"] == intent["retry_at"]
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == {
        "projection_revision": 2,
        "delivery_generation": intent["delivery_generation"],
        "retry_at": intent["retry_at"],
        "attempt_sequence": intent["attempt_sequence"],
    }


def test_restart_delayed_older_recovery_sequence_cannot_overwrite_newer_arm(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job

    provider, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    provider.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 2, "dedupe_key": "2" * 64,
        "retry_at": "2026-07-15T01:03:00Z",
        "deadline_at": "2026-07-15T01:05:00Z",
    })
    expected_marker = {
        "projection_revision": 2,
        "delivery_generation": 1,
        "retry_at": "2026-07-15T01:03:00Z",
        "attempt_sequence": 2,
    }
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == expected_marker
    provision_count = len(fake.provisions)

    restarted = type(provider)()
    restarted._client = fake
    result = restarted.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 1, "dedupe_key": "1" * 64,
        "retry_at": "2026-07-15T01:02:00Z",
        "deadline_at": "2026-07-15T01:05:00Z",
    })

    assert result == {
        "status": "superseded",
        "provider": "chronos",
        "observed_fire_at": expected_marker["retry_at"],
    }
    assert len(fake.provisions) == provision_count
    assert fake._armed == [{"job_id": job["id"], "fire_at": expected_marker["retry_at"]}]
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == expected_marker


def test_older_generation_high_sequence_cannot_overwrite_newer_generation(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job

    provider, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    provider.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 2, "attempt_sequence": 1, "dedupe_key": "2" * 64,
        "retry_at": "2026-07-15T01:03:00Z", "deadline_at": "2026-07-15T01:05:00Z",
    })
    marker = get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"]
    provision_count = len(fake.provisions)

    result = provider.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 99, "dedupe_key": "1" * 64,
        "retry_at": "2026-07-15T01:02:00Z", "deadline_at": "2026-07-15T01:05:00Z",
    })

    assert result["status"] == "superseded"
    assert len(fake.provisions) == provision_count
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == marker


@pytest.mark.parametrize(
    ("incoming_sequence", "expected_status", "expected_sequence", "expected_calls"),
    [(4, "superseded", 5, 0), (5, "armed", 5, 0), (6, "armed", 6, 1)],
)
def test_same_generation_attempt_sequence_ordering_matrix(
    temp_home, chronos, incoming_sequence, expected_status, expected_sequence, expected_calls,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job

    provider, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    provider.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 7, "attempt_sequence": 5, "dedupe_key": "5" * 64,
        "retry_at": "2026-07-15T01:02:00Z", "deadline_at": "2026-07-15T01:05:00Z",
    })
    fake.provisions.clear()
    result = provider.reconcile_calendar_recovery_arm({
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 7, "attempt_sequence": incoming_sequence,
        "dedupe_key": str(incoming_sequence) * 64,
        "retry_at": "2026-07-15T01:03:00Z", "deadline_at": "2026-07-15T01:05:00Z",
    })

    marker = get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"]
    assert result["status"] == expected_status
    assert marker["delivery_generation"] == 7
    assert marker["attempt_sequence"] == expected_sequence
    assert len(fake.provisions) == expected_calls


def test_restart_same_recovery_sequence_rearms_durable_marker_time(
    temp_home, chronos,
):
    from cron.jobs import get_job_raw, save_jobs
    from tests.cron.test_calendar_delivery_v2 import managed_job

    provider, fake = chronos
    job = managed_job(next_run_at="2026-07-15T01:00:00Z")
    save_jobs([job])
    original_provision = fake.provision

    def provision_and_observe(**kwargs):
        result = original_provision(**kwargs)
        fake._armed = [{"job_id": kwargs["job_id"], "fire_at": kwargs["fire_at"]}]
        return result

    fake.provision = provision_and_observe
    marker_retry = "2026-07-15T01:03:00Z"
    intent = {
        "job_id": job["id"], "projection_revision": 2,
        "delivery_generation": 1, "attempt_sequence": 2, "dedupe_key": "2" * 64,
        "retry_at": marker_retry, "deadline_at": "2026-07-15T01:05:00Z",
    }
    provider.reconcile_calendar_recovery_arm(intent)
    marker = get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"]

    fake._armed = []
    restarted = type(provider)()
    restarted._client = fake
    replay = dict(intent, retry_at="2026-07-15T01:02:00Z")
    result = restarted.reconcile_calendar_recovery_arm(replay)

    assert result == {
        "status": "armed",
        "provider": "chronos",
        "observed_fire_at": marker_retry,
    }
    assert fake.provisions[-1]["fire_at"] == marker_retry
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == marker


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("2026-07-22T04:34:56.123456789Z", "2026-07-22T04:34:56.123456Z"),
        ("2026-07-22T04:34:56.123400Z", "2026-07-22T04:34:56.123400Z"),
        ("2026-07-22T04:34:56Z", "2026-07-22T04:34:56Z"),
    ],
)
def test_calendar_recovery_retry_at_uses_wire_microsecond_canonical(raw, canonical):
    from plugins.cron_providers.chronos import (
        _calendar_recovery_state,
        _canonical_calendar_retry_at,
    )
    assert _canonical_calendar_retry_at(raw) == canonical
    assert _calendar_recovery_state({
        "provider_state": {"chronos": {"calendar_recovery": {
            "projection_revision": 2,
            "delivery_generation": 1,
            "retry_at": raw,
            "attempt_sequence": 4,
        }}},
    }) == (2, 1, canonical, 4)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("2026-07-22T04:34:56.123400+00:00", "2026-07-22T04:34:56.123400Z"),
        ("2026-07-22T04:34:56.123400+08:00", None),
        ("2026-07-22T04:34:56.123400", None),
    ],
)
def test_calendar_recovery_retry_at_requires_utc_offset(raw, canonical):
    from plugins.cron_providers.chronos import (
        _calendar_recovery_state,
        _canonical_calendar_retry_at,
    )

    assert _canonical_calendar_retry_at(raw) == canonical
    state = _calendar_recovery_state({
        "provider_state": {"chronos": {"calendar_recovery": {
            "projection_revision": 2,
            "delivery_generation": 1,
            "retry_at": raw,
            "attempt_sequence": 4,
        }}},
    })
    assert state == ((2, 1, canonical, 4) if canonical is not None else None)


@pytest.mark.parametrize("attempt_sequence", [None, 0, True, -1, 1.5, 2**64])
def test_calendar_recovery_marker_requires_bounded_uint64_sequence(attempt_sequence):
    from plugins.cron_providers.chronos import _calendar_recovery_state

    assert _calendar_recovery_state({
        "provider_state": {"chronos": {"calendar_recovery": {
            "projection_revision": 2,
            "delivery_generation": 1,
            "retry_at": "2026-07-22T04:34:56Z",
            "attempt_sequence": attempt_sequence,
        }}},
    }) is None


@pytest.mark.parametrize("delivery_generation", [None, 0, True, -1, 1.5, 2**64])
def test_calendar_recovery_marker_requires_bounded_uint64_generation(delivery_generation):
    from plugins.cron_providers.chronos import _calendar_recovery_state

    assert _calendar_recovery_state({
        "provider_state": {"chronos": {"calendar_recovery": {
            "projection_revision": 2,
            "delivery_generation": delivery_generation,
            "retry_at": "2026-07-22T04:34:56Z",
            "attempt_sequence": 1,
        }}},
    }) is None


def test_calendar_recovery_marker_accepts_uint64_max_and_rejects_extra_fields():
    from plugins.cron_providers.chronos import _calendar_recovery_state

    marker = {
        "projection_revision": 2,
        "delivery_generation": 2**64 - 1,
        "retry_at": "2026-07-22T04:34:56Z",
        "attempt_sequence": 2**64 - 1,
    }
    job = {"provider_state": {"chronos": {"calendar_recovery": marker}}}
    assert _calendar_recovery_state(job) == (2, 2**64 - 1, marker["retry_at"], 2**64 - 1)
    marker["unexpected"] = "reject"
    assert _calendar_recovery_state(job) is None
    marker.pop("unexpected")
    marker.pop("delivery_generation")
    assert _calendar_recovery_state(job) is None


def test_resolved_chronos_instances_flow_serializes_same_revision_recovery_cas(
    temp_home, monkeypatch,
):
    """Real resolve/load/register calls must share the recovery critical section."""
    from cron.jobs import get_job_raw, save_jobs
    from cron.scheduler_provider import resolve_cron_scheduler
    from plugins.cron_providers.chronos import ChronosCronScheduler
    from tests.cron.test_calendar_delivery_v2 import managed_job

    config = {
        "cron": {
            "provider": "chronos",
            "chronos": {
                "portal_url": "https://portal.test",
                "callback_url": "https://agent.example/",
            },
        },
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(ChronosCronScheduler, "_have_nous_token", lambda self: True)

    class ConcurrentClient:
        def __init__(self):
            self._guard = threading.Lock()
            self._active = 0
            self.max_active = 0
            self.provisions = []
            self.armed = {}

        def provision(self, *, job_id, fire_at, agent_callback_url, dedup_key):
            with self._guard:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
                self.provisions.append({
                    "job_id": job_id,
                    "fire_at": fire_at,
                    "dedup_key": dedup_key,
                })
            time.sleep(0.03)
            with self._guard:
                self.armed[job_id] = fire_at
                self._active -= 1
            return {"schedule_id": f"sched-{job_id}"}

        def list_armed(self):
            with self._guard:
                return [
                    {"job_id": job_id, "fire_at": fire_at}
                    for job_id, fire_at in self.armed.items()
                ]

    client = ConcurrentClient()
    monkeypatch.setattr(ChronosCronScheduler, "_get_client", lambda self: client)
    job = managed_job(next_run_at="2026-07-22T04:30:00Z")
    save_jobs([job])
    retries = [
        "2026-07-22T04:34:56.123400Z",
        "2026-07-22T04:35:56.123400+00:00",
    ]
    barrier = threading.Barrier(len(retries))
    provider_ids = []
    provider_ids_guard = threading.Lock()

    def reconcile(index):
        provider = resolve_cron_scheduler()
        with provider_ids_guard:
            provider_ids.append(id(provider))
        barrier.wait()
        return provider.reconcile_calendar_recovery_arm({
            "job_id": job["id"],
            "projection_revision": 2,
            "delivery_generation": 1,
            "attempt_sequence": index + 1,
            "dedupe_key": str(index + 1) * 64,
            "retry_at": retries[index],
            "deadline_at": "2026-07-22T04:40:00Z",
        })

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reconcile, range(2)))

    assert len(set(provider_ids)) == 2
    assert client.max_active == 1
    assert results[1]["observed_fire_at"] == "2026-07-22T04:35:56.123400Z"
    assert results[0]["status"] in {"armed", "superseded"}
    assert client.provisions[-1]["dedup_key"] == "2" * 64
    assert get_job_raw(job["id"])["provider_state"]["chronos"]["calendar_recovery"] == {
        "projection_revision": 2,
        "delivery_generation": 1,
        "retry_at": client.provisions[-1]["fire_at"],
        "attempt_sequence": 2,
    }


def test_provider_state_cas_caps_rebuilt_map_and_skips_malformed_siblings(
    temp_home,
):
    from cron.jobs import load_jobs, save_jobs, update_job_provider_state
    job_id = "cal-alert-" + "a" * 32
    base = {"id": job_id, "calendar_projection_revision": 2}

    sixteen = {f"provider_{i}": {"v": "x"} for i in range(16)}
    save_jobs(["malformed-sibling", {**base, "provider_state": sixteen}])
    with pytest.raises(ValueError, match="provider cap"):
        update_job_provider_state(
            job_id, "chronos", {"v": "x"},
            revision_field="calendar_projection_revision", expected_revision=2,
        )

    near_cap = {f"provider_{i}": {"v": "x" * 850} for i in range(15)}
    save_jobs(["malformed-sibling", {**base, "provider_state": near_cap}])
    with pytest.raises(ValueError, match="total cap"):
        update_job_provider_state(
            job_id, "chronos", {"v": "y" * 4000},
            revision_field="calendar_projection_revision", expected_revision=2,
        )

    save_jobs(["malformed-sibling", base])
    assert update_job_provider_state(
        job_id, "chronos", {"calendar_recovery": {
            "projection_revision": 2,
            "delivery_generation": 1,
            "retry_at": "2026-07-22T04:34:56Z",
            "attempt_sequence": 4,
        }},
        revision_field="calendar_projection_revision", expected_revision=2,
    ) == "updated"
    assert "chronos" in load_jobs()[1]["provider_state"]


@pytest.mark.parametrize("suffix", ["recovery", "A" * 32, "a_b", "a-b"])
def test_calendar_recovery_arm_rejects_noncanonical_planner_job_ids(
    chronos, suffix,
):
    prov, _fake = chronos
    with pytest.raises(ValueError, match="invalid recovery identity"):
        prov.reconcile_calendar_recovery_arm({
            "job_id": "cal-alert-" + suffix,
            "projection_revision": 2,
            "delivery_generation": 1,
            "attempt_sequence": 4,
            "dedupe_key": "d" * 64,
            "retry_at": "2026-07-15T01:02:03Z",
            "deadline_at": "2026-07-15T01:05:00Z",
        })


@pytest.mark.parametrize("attempt_sequence", [0, True, -1, 1.5, 2**64])
def test_calendar_recovery_arm_rejects_unbounded_attempt_sequence(
    chronos, attempt_sequence,
):
    provider, _fake = chronos
    with pytest.raises(ValueError, match="invalid recovery identity"):
        provider.reconcile_calendar_recovery_arm({
            "job_id": "cal-alert-" + "a" * 32,
            "projection_revision": 2,
            "delivery_generation": 1,
            "attempt_sequence": attempt_sequence,
            "dedupe_key": "d" * 64,
            "retry_at": "2026-07-15T01:02:03Z",
            "deadline_at": "2026-07-15T01:05:00Z",
        })


@pytest.mark.parametrize("delivery_generation", [0, True, -1, 1.5, 2**64])
def test_calendar_recovery_arm_rejects_unbounded_delivery_generation(
    chronos, delivery_generation,
):
    provider, _fake = chronos
    with pytest.raises(ValueError, match="invalid recovery identity"):
        provider.reconcile_calendar_recovery_arm({
            "job_id": "cal-alert-" + "a" * 32,
            "projection_revision": 2,
            "delivery_generation": delivery_generation,
            "attempt_sequence": 1,
            "dedupe_key": "d" * 64,
            "retry_at": "2026-07-15T01:02:03Z",
            "deadline_at": "2026-07-15T01:05:00Z",
        })


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
