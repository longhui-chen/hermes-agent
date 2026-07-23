"""Chronos — NAS-mediated managed cron provider (scale-to-zero).

Chronos (the Greek god of time, alongside Hermes) is the first non-default
``CronScheduler``. It lets a hosted gateway scale to zero while idle and still
fire cron jobs: instead of a 60s in-process ticker, it asks NAS to arm exactly
one external one-shot per job at that job's real next-fire time. NAS calls the
agent back at fire time over an authenticated webhook (``/api/cron/fire``); the
agent runs the job via the shared ``run_one_job`` body and re-arms the next
one-shot.

The external scheduler NAS uses is an internal NAS implementation detail —
Chronos names no vendor, holds no scheduler credentials, and speaks only to
NAS's ``agent-cron`` endpoints with the agent's existing Nous token.

Design constraints (see the plan's DQ-1):
  - start() arms all enabled jobs and RETURNS; it never blocks and never spawns
    a periodic wake. Between fires the machine is truly at zero.
  - reconcile runs only on a warm process (start / on_jobs_changed / piggybacked
    on a fire), never as a periodic wake of a sleeping machine.

Inert unless ``cron.provider: chronos``. ``resolve_cron_scheduler`` falls back
to the built-in if Chronos is unavailable, so cron never loses its trigger.

Wire contract: ``docs/chronos-managed-cron-contract.md``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import threading
from typing import Any, Dict, Optional

from cron.scheduler_provider import CronScheduler

logger = logging.getLogger("cron.chronos")
_CALENDAR_JOB_ID_RE = re.compile(r"cal-alert-[a-f0-9]{32}")
_CALENDAR_RECONCILE_LOCK = threading.Lock()
_MAX_RECOVERY_UINT64 = 2**64 - 1


def _valid_recovery_uint64(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 < value <= _MAX_RECOVERY_UINT64
    )


def _canonical_calendar_retry_at(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    try:
        retry = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if retry.tzinfo is None or retry.utcoffset() != dt.timedelta(0):
        return None
    return retry.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _calendar_recovery_state(job: Any) -> Optional[tuple[int, int, str, int]]:
    if not isinstance(job, dict):
        return None
    provider_state = job.get("provider_state")
    chronos_state = provider_state.get("chronos") if isinstance(provider_state, dict) else None
    recovery = chronos_state.get("calendar_recovery") if isinstance(chronos_state, dict) else None
    if not isinstance(recovery, dict) or len(recovery) != 4:
        return None
    revision = recovery.get("projection_revision")
    delivery_generation = recovery.get("delivery_generation")
    retry_at = recovery.get("retry_at")
    attempt_sequence = recovery.get("attempt_sequence")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision <= 0
    ):
        return None
    if not _valid_recovery_uint64(delivery_generation):
        return None
    if not _valid_recovery_uint64(attempt_sequence):
        return None
    canonical_retry = _canonical_calendar_retry_at(retry_at)
    if canonical_retry is None:
        return None
    return revision, delivery_generation, canonical_retry, attempt_sequence


def _persist_calendar_recovery_state(
    job_id: str,
    projection_revision: int,
    retry_at: Optional[str],
    delivery_generation: Optional[int] = None,
    attempt_sequence: Optional[int] = None,
) -> str:
    from cron.jobs import update_job_provider_state
    state = None
    if retry_at is not None:
        if not _valid_recovery_uint64(delivery_generation):
            raise ValueError("invalid recovery delivery generation")
        if not _valid_recovery_uint64(attempt_sequence):
            raise ValueError("invalid recovery attempt sequence")
        state = {
            "calendar_recovery": {
                "projection_revision": projection_revision,
                "delivery_generation": delivery_generation,
                "retry_at": retry_at,
                "attempt_sequence": attempt_sequence,
            },
        }
    return update_job_provider_state(
        job_id,
        "chronos",
        state,
        revision_field="calendar_projection_revision",
        expected_revision=projection_revision,
    )


def _cfg(*keys: str, default: Any = "") -> Any:
    """Read a cron.chronos.* config value (no network)."""
    try:
        from hermes_cli.config import cfg_get, load_config
        return cfg_get(load_config(), *keys, default=default)
    except Exception:
        return default


class ChronosCronScheduler(CronScheduler):
    """NAS-mediated external cron provider."""

    def __init__(self) -> None:
        # In-memory map of job_id → fire_at we've asked NAS to arm. Best-effort
        # cache; reconcile rebuilds desired state from jobs.json, so a cold
        # process simply re-arms (idempotent via dedup_key).
        self._armed: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._client = None  # lazily constructed (no network in is_available)

    # -- identity / availability -----------------------------------------

    @property
    def name(self) -> str:
        return "chronos"

    def is_available(self) -> bool:
        """Config presence only — NO network.

        Chronos needs a portal base URL, the agent's own publicly-reachable
        callback URL (for NAS→agent fires), and a usable Nous token (the agent
        is logged into the portal). If any is missing, resolve_cron_scheduler
        falls back to the built-in ticker.
        """
        if not (_cfg("cron", "chronos", "portal_url") and _cfg("cron", "chronos", "callback_url")):
            return False
        return self._have_nous_token()

    def _have_nous_token(self) -> bool:
        """True if the agent has a Nous Portal login (no network call).

        Checks the stored auth state for a Nous access token — does NOT refresh
        or hit the network (is_available must stay offline). The actual
        refresh-aware token is resolved lazily at provision time.
        """
        try:
            from hermes_cli.auth import get_provider_auth_state
            state = get_provider_auth_state("nous") or {}
            return bool(state.get("access_token"))
        except Exception:
            return False

    # -- client -----------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from ._nas_client import NasCronClient
            self._client = NasCronClient(_cfg("cron", "chronos", "portal_url"))
        return self._client

    def _callback_url(self) -> str:
        return str(_cfg("cron", "chronos", "callback_url") or "")

    # -- lifecycle --------------------------------------------------------

    def start(self, stop_event, *, adapters=None, loop=None, interval=60):
        """Arm all enabled jobs via NAS, then RETURN immediately.

        Does NOT block and does NOT spawn a 60s wake (DQ-1) — that is the whole
        point of scale-to-zero. The machine wakes only on a NAS→agent fire.
        """
        try:
            self.reconcile()
        except Exception as e:
            logger.warning("Chronos start() reconcile failed: %s", e)
        # Intentionally return — no loop, no periodic wake.

    def stop(self) -> None:
        return None

    def on_jobs_changed(self) -> None:
        """A job was created/updated/removed/paused/resumed — reconcile the NAS
        registry so the affected one-shot is (re-)armed or cancelled."""
        try:
            self.reconcile()
        except Exception as e:
            logger.debug("Chronos on_jobs_changed reconcile failed: %s", e)

    # -- arming -----------------------------------------------------------

    def _arm_one_shot(self, job: Dict[str, Any]) -> None:
        """Ask NAS to arm exactly one one-shot at the job's next_run_at.

        The agent computes the time; NAS+its scheduler are the dumb executor.
        Idempotent per (job_id, fire_at) via dedup_key, so re-arming the same
        fire is a no-op NAS-side.
        """
        job_id = job["id"]
        fire_at = job.get("next_run_at")
        if not fire_at:
            return
        dedup_key = f"{job_id}:{fire_at}"
        self._get_client().provision(
            job_id=job_id,
            fire_at=fire_at,
            agent_callback_url=self._callback_url(),
            dedup_key=dedup_key,
        )
        with self._lock:
            self._armed[job_id] = fire_at

    def _cancel(self, job_id: str) -> None:
        try:
            self._get_client().cancel(job_id=job_id)
        finally:
            with self._lock:
                self._armed.pop(job_id, None)

    def _rearm_calendar_recovery(
        self,
        job_id: str,
        revision: int,
        delivery_generation: int,
        retry_at: str,
        attempt_sequence: int,
    ) -> None:
        """Reassert a durable recovery intent after a crash or lost NAS ACK."""
        dedup_key = hashlib.sha256(
            f"calendar-recovery\x00{job_id}\x00{revision}\x00{delivery_generation}\x00{attempt_sequence}\x00{retry_at}".encode("utf-8")
        ).hexdigest()
        self._get_client().provision(
            job_id=job_id,
            fire_at=retry_at,
            agent_callback_url=self._callback_url(),
            dedup_key=dedup_key,
        )
        with self._lock:
            self._armed[job_id] = retry_at

    def _list_armed(self, *, force_remote: bool = False) -> Dict[str, str]:
        """Observed armed one-shots: job_id → fire_at.

        Prefer the in-memory map (warm process); on a cold/empty map, ask NAS
        (best-effort). If NAS list fails, return what we have — reconcile then
        re-arms desired jobs idempotently.
        """
        with self._lock:
            if self._armed and not force_remote:
                return dict(self._armed)
        try:
            observed = {
                item["job_id"]: item.get("fire_at", "")
                for item in self._get_client().list_armed()
                if item.get("job_id")
            }
            with self._lock:
                self._armed.update(observed)
            return observed
        except Exception as e:
            logger.debug("Chronos _list_armed failed (will re-arm idempotently): %s", e)
            return {}

    def calendar_capabilities(self) -> dict:
        return {
            "provider": self.name,
            "contract_version": 1,
            "calendar_external_fire_v1": True,
            "reliable_job_reconcile_v1": True,
        }

    def reconcile_calendar_job(self, job_id: str, expected_action: str, projection_revision: int) -> dict:
        from cron.calendar_delivery import is_managed_calendar_event_alert
        from cron.jobs import get_job_raw

        if expected_action not in {"delete", "upsert"} or projection_revision <= 0:
            raise ValueError("invalid calendar reconcile request")
        with _CALENDAR_RECONCILE_LOCK:
            if expected_action == "delete":
                current = get_job_raw(job_id)
                current_revision = current.get("calendar_projection_revision") if isinstance(current, dict) else None
                if (isinstance(current_revision, int) and not isinstance(current_revision, bool)
                        and current_revision > projection_revision):
                    return {"status": "superseded", "provider": self.name}
                self._cancel(job_id)
                if job_id in self._list_armed(force_remote=True):
                    raise RuntimeError("Chronos cancel not yet observed")
                return {"status": "cancelled", "provider": self.name}
            job = get_job_raw(job_id)
            if not is_managed_calendar_event_alert(job):
                raise RuntimeError("managed calendar job missing or invalid")
            current_revision = job.get("calendar_projection_revision")
            if current_revision > projection_revision:
                return {"status": "superseded", "provider": self.name}
            if current_revision < projection_revision:
                raise RuntimeError("calendar projection revision mismatch")
            recovery = _calendar_recovery_state(job)
            observed = self._list_armed(force_remote=True)
            if recovery is not None:
                recovery_revision, delivery_generation, retry_at, attempt_sequence = recovery
                if recovery_revision > projection_revision:
                    return {
                        "status": "superseded",
                        "provider": self.name,
                        "observed_fire_at": retry_at,
                    }
                if recovery_revision == projection_revision:
                    if observed.get(job_id) != retry_at:
                        self._rearm_calendar_recovery(
                            job_id,
                            recovery_revision,
                            delivery_generation,
                            retry_at,
                            attempt_sequence,
                        )
                        observed = self._list_armed(force_remote=True)
                        if observed.get(job_id) != retry_at:
                            raise RuntimeError("Chronos recovery fire time not durably observed")
                    return {
                        "status": "recovery_preserved",
                        "provider": self.name,
                        "observed_fire_at": retry_at,
                    }
            persisted = _persist_calendar_recovery_state(job_id, projection_revision, None)
            if persisted != "updated":
                if persisted == "superseded":
                    return {"status": "superseded", "provider": self.name}
                raise RuntimeError("calendar projection revision changed before arm")
            self._arm_one_shot(job)
            observed = self._list_armed(force_remote=True)
            if observed.get(job_id) != job.get("next_run_at"):
                raise RuntimeError("Chronos fire time not durably observed")
            return {"status": "armed", "provider": self.name, "observed_fire_at": observed[job_id]}

    def reconcile_calendar_recovery_arm(self, intent: dict) -> dict:
        """Provision and observe one exact planner recovery attempt.

        The provider dedupe key includes delivery generation, monotonic attempt
        sequence and canonical retry time, so replaying a lost HTTP response is
        idempotent without suppressing a later legitimate attempt.
        """
        if not isinstance(intent, dict):
            raise ValueError("invalid recovery intent")
        job_id = str(intent.get("job_id") or "")
        dedupe_key = str(intent.get("dedupe_key") or "")
        retry_at = str(intent.get("retry_at") or "")
        deadline_at = str(intent.get("deadline_at") or "")
        revision = intent.get("projection_revision")
        generation = intent.get("delivery_generation")
        sequence = intent.get("attempt_sequence")
        if (not _CALENDAR_JOB_ID_RE.fullmatch(job_id)
                or not re.fullmatch(r"[0-9a-f]{64}", dedupe_key)
                or not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0
                or not _valid_recovery_uint64(generation)
                or not _valid_recovery_uint64(sequence)):
            raise ValueError("invalid recovery identity")
        try:
            retry = dt.datetime.fromisoformat(retry_at.replace("Z", "+00:00"))
            deadline = dt.datetime.fromisoformat(deadline_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid recovery time") from exc
        if retry.tzinfo is None or deadline.tzinfo is None or retry >= deadline:
            raise ValueError("invalid recovery window")
        canonical_retry = _canonical_calendar_retry_at(retry_at)
        if canonical_retry is None:
            raise ValueError("invalid recovery time")
        with _CALENDAR_RECONCILE_LOCK:
            from cron.calendar_delivery import is_active_managed_calendar_event_alert
            from cron.jobs import get_job_raw
            job = get_job_raw(job_id)
            if job is None:
                return {"status": "cancelled", "provider": self.name}
            current_revision = job.get("calendar_projection_revision")
            if (
                not isinstance(current_revision, int)
                or isinstance(current_revision, bool)
                or current_revision <= 0
            ):
                return {"status": "cancelled", "provider": self.name}
            if current_revision > revision:
                return {"status": "superseded", "provider": self.name}
            if current_revision < revision:
                raise RuntimeError("managed calendar recovery projection not materialized yet")
            if not is_active_managed_calendar_event_alert(job):
                return {"status": "cancelled", "provider": self.name}
            recovery = _calendar_recovery_state(job)
            if recovery is not None:
                stored_revision, stored_generation, stored_retry, stored_sequence = recovery
                if stored_revision > revision or (
                    stored_revision == revision
                    and (stored_generation, stored_sequence) > (generation, sequence)
                ):
                    return {
                        "status": "superseded",
                        "provider": self.name,
                    }
                if (
                    stored_revision == revision
                    and (stored_generation, stored_sequence) == (generation, sequence)
                ):
                    observed = self._list_armed(force_remote=True)
                    if observed.get(job_id) != stored_retry:
                        self._rearm_calendar_recovery(
                            job_id,
                            revision,
                            stored_generation,
                            stored_retry,
                            stored_sequence,
                        )
                        observed = self._list_armed(force_remote=True)
                        if observed.get(job_id) != stored_retry:
                            raise RuntimeError("Chronos recovery fire time not durably observed")
                    return {
                        "status": "armed",
                        "provider": self.name,
                        "observed_fire_at": stored_retry,
                    }
            persisted = _persist_calendar_recovery_state(
                job_id,
                revision,
                canonical_retry,
                generation,
                sequence,
            )
            if persisted != "updated":
                raise RuntimeError("managed calendar recovery projection changed")
            self._get_client().provision(
                job_id=job_id,
                fire_at=canonical_retry,
                agent_callback_url=self._callback_url(),
                dedup_key=dedupe_key,
            )
            with self._lock:
                self._armed[job_id] = canonical_retry
            observed = self._list_armed(force_remote=True)
            if observed.get(job_id) != canonical_retry:
                raise RuntimeError("Chronos recovery fire time not durably observed")
            return {"status": "armed", "provider": self.name, "observed_fire_at": observed[job_id]}

    # -- reconcile --------------------------------------------------------

    def reconcile(self) -> None:
        """Converge the NAS-armed one-shots toward jobs.json (desired state):
        arm missing / re-arm changed-time, cancel orphaned."""
        from cron.calendar_delivery import is_active_managed_calendar_event_alert
        from cron.jobs import load_jobs

        jobs = load_jobs()
        managed_calendar_jobs = {
            str(j.get("id")): j for j in jobs
            if is_active_managed_calendar_event_alert(j)
        }
        managed_calendar_ids = set(managed_calendar_jobs)
        recovery_desired = {
            job_id: recovery
            for job_id, job in managed_calendar_jobs.items()
            if (recovery := _calendar_recovery_state(job)) is not None
        }
        desired: Dict[str, str] = {
            j["id"]: j["next_run_at"]
            for j in jobs
            if j.get("source") != "calendar"
            and j.get("enabled") and j.get("next_run_at") and j.get("state") != "paused"
        }
        desired.update({
            job_id: job["next_run_at"]
            for job_id, job in managed_calendar_jobs.items()
            if job_id not in recovery_desired and job.get("next_run_at")
        })
        # Managed calendar jobs must be observed directly: a warm in-memory
        # entry can be stale after NAS loses an arm. Recovery markers remain
        # the sole desired arm for their job and are replayed below.
        observed = self._list_armed(force_remote=bool(managed_calendar_jobs))

        # Arm missing or changed-time.
        for job_id, fire_at in desired.items():
            if observed.get(job_id) != fire_at:
                # Re-fetch the full job dict to arm (need the whole record).
                from cron.jobs import get_job
                job = get_job(job_id)
                if job:
                    try:
                        self._arm_one_shot(job)
                    except Exception as e:
                        logger.warning("Chronos failed to arm job %s: %s", job_id, e)

        # A persisted recovery marker is authoritative across process restarts
        # and lost NAS arms, so replay that exact immutable attempt rather than
        # next_run_at.
        for job_id, recovery in recovery_desired.items():
            revision, generation, retry_at, sequence = recovery
            if observed.get(job_id) != retry_at:
                try:
                    self._rearm_calendar_recovery(
                        job_id, revision, generation, retry_at, sequence,
                    )
                except Exception as e:
                    logger.warning(
                        "Chronos failed to re-arm calendar recovery %s: %s",
                        job_id,
                        e,
                    )

        # Cancel orphans (armed but no longer desired).
        for job_id in list(observed.keys()):
            if job_id not in desired and job_id not in managed_calendar_ids:
                try:
                    self._cancel(job_id)
                except Exception as e:
                    logger.warning("Chronos failed to cancel orphan %s: %s", job_id, e)

    # -- fire -------------------------------------------------------------

    def fire_due(
        self,
        job_id: str,
        *,
        adapters: Any = None,
        loop: Any = None,
        fire_at: Optional[str] = None,
    ) -> bool:
        """Run the due job (claim + run_one_job via the ABC default), then
        re-arm the NEXT one-shot through NAS.

        Re-arm happens AFTER the run so next_run_at reflects the completed fire.
        If the job is gone (one-shot completed / repeat-N exhausted), get_job
        returns None → nothing to re-arm (the schedule naturally stops).
        """
        ran = super().fire_due(job_id, adapters=adapters, loop=loop, fire_at=fire_at)
        if ran:
            from cron.jobs import get_job
            job = get_job(job_id)
            if job and job.get("enabled") and job.get("next_run_at"):
                try:
                    self._arm_one_shot(job)
                except Exception as e:
                    logger.warning("Chronos failed to re-arm job %s after fire: %s", job_id, e)
        return ran


def register(ctx) -> None:
    """Plugin entrypoint — register the Chronos provider with the loader.

    Mirrors the memory-plugin shape; plugins/cron_providers discovery calls this and
    collects the provider via register_cron_scheduler.
    """
    ctx.register_cron_scheduler(ChronosCronScheduler())
