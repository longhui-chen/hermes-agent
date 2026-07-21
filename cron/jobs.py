"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import contextlib
import copy
from bisect import bisect_left
import json
import logging
import math
import shutil
import tempfile
import threading
import time
import os
import re
import uuid

# Cross-process advisory file locking for jobs.json critical sections.
# fcntl is Unix-only; on Windows fall back to msvcrt. Either may be absent,
# in which case _jobs_lock() degrades to in-process locking only (the old
# behaviour) rather than failing.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
from datetime import datetime, timedelta, timezone
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any, Set, Tuple, Union

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_replace

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:
    from backports.zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # type: ignore[no-redef]

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


_MAX_TIMEZONE_NAME_LENGTH = 255
_MAX_OUTPUT_LANGUAGE_TAG_LENGTH = 63
_MAX_CRON_NEXT_RUN_ATTEMPTS = 8
_MAX_OCCURRENCE_HISTORY_FILES = 5_000
_MAX_OCCURRENCE_STATUS_BYTES = 4_096
_MAX_OCCURRENCE_ITERATIONS = 10_000
_MAX_OCCURRENCE_JOURNAL_RECORDS = 5_000
_MAX_OCCURRENCE_JOURNAL_BYTES = 2 * 1024 * 1024
_MAX_OCCURRENCE_QUERY_JOBS = 256
_IN_FLIGHT_OCCURRENCE_DEFAULT_TTL_SECONDS = 15 * 60
_IN_FLIGHT_OCCURRENCE_MIN_TTL_SECONDS = 2 * 60
_IN_FLIGHT_OCCURRENCE_MAX_TTL_SECONDS = 60 * 60
_MAX_EXTERNAL_FIRE_AGE_SECONDS = 366 * 24 * 60 * 60
_MAX_EXTERNAL_FIRE_FUTURE_SECONDS = 5 * 60


def normalize_output_language_tag(value: Any) -> Optional[str]:
    """Return a canonical, bounded BCP 47 tag, or ``None`` if invalid.

    This is the fail-closed reader for an untrusted persisted field.  It never
    raises, so a hand-edited legacy job cannot break the scheduler, and it
    imports the IANA-backed validator only when a non-empty value is present.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if (
        not text
        or len(text) > _MAX_OUTPUT_LANGUAGE_TAG_LENGTH
        or not text.isascii()
        or not re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text)
    ):
        return None

    try:
        from langcodes import standardize_tag, tag_is_valid

        if not tag_is_valid(text):
            return None
        normalized = standardize_tag(text)
    except (ImportError, LookupError, TypeError, ValueError):
        return None

    # These ISO 639 codes deliberately describe an unknown, multiple, or
    # non-linguistic language and therefore cannot establish an output
    # language. Pure private-use tags have the same ambiguity.
    primary = normalized.split("-", 1)[0].lower()
    if primary in {"und", "mul", "zxx", "x"}:
        return None
    # A one-character subtag after the primary language is an extension
    # singleton (including private-use ``x``). Output-language preferences do
    # not need extensions, and allowing their arbitrary payload would turn
    # this persisted field into a durable system-prompt injection surface.
    if any(len(subtag) == 1 for subtag in normalized.split("-")[1:]):
        return None
    return normalized


def validate_output_language_tag(value: Any) -> Optional[str]:
    """Validate a create/update value, raising a bounded generic error."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    normalized = normalize_output_language_tag(value)
    if normalized is None:
        raise ValueError(
            "output_language must be a valid BCP 47 tag of at most 63 ASCII "
            "characters (for example 'zh-CN', 'ja', 'ar', or 'sr-Latn-RS')"
        )
    return normalized


def _normalized_iana_timezone_name(name: Any) -> Optional[str]:
    """Return a bounded valid IANA timezone name, or ``None``.

    Persisted jobs may predate timezone validation or may have been edited by
    hand.  Treat that field as untrusted: reject non-strings and oversized
    values before asking ``ZoneInfo`` to resolve them.
    """
    if not isinstance(name, str):
        return None
    text = name.strip()
    if not text or len(text) > _MAX_TIMEZONE_NAME_LENGTH:
        return None
    try:
        ZoneInfo(text)
    except (OSError, ZoneInfoNotFoundError, TypeError, ValueError):
        return None
    return text


def _validate_tz_name(name: Optional[str]) -> Optional[str]:
    """Validate IANA timezone name; return canonical string or None.

    Empty / None means "no per-job timezone" — falls back to the hermes
    instance's configured timezone (HERMES_TIMEZONE / config.yaml / system).
    """
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("Invalid timezone: expected an IANA timezone name")
    text = name.strip()
    if not text:
        return None
    normalized = _normalized_iana_timezone_name(text)
    if normalized is None:
        display = text[:128] + ("..." if len(text) > 128 else "")
        raise ValueError(f"Invalid timezone {display!r}")
    return normalized

# =============================================================================
# Configuration
# =============================================================================

# Cron is per-profile by design (issue #4707). Each profile owns its own cron
# store under its own HERMES_HOME, and a profile-scoped gateway runs that
# profile's jobs under that same HERMES_HOME — so a job authored in profile
# `coder` lives in `~/.hermes/profiles/coder/cron/jobs.json` and executes with
# `coder`'s `.env`, `config.yaml`, and skills. We deliberately anchor on
# `get_hermes_home()` (the active profile home), NOT `get_default_hermes_root()`
# (the shared root). Anchoring at the root would funnel every profile's jobs
# into one shared `jobs.json` and run them under whatever HERMES_HOME the
# ticker process happens to have — leaking config/credentials/skills across
# profiles (the security boundary #4707 was filed for). Do NOT change this to
# the default root: that re-breaks per-profile isolation. See also the dynamic
# `_get_hermes_home()` / `_get_lock_paths()` resolution in cron/scheduler.py.
HERMES_DIR = get_hermes_home().resolve()
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
# Heartbeat file the in-process ticker touches on every loop iteration. The
# gateway process and the (separate) ``hermes cron status`` process share it
# so status can tell whether the ticker THREAD is alive, not just whether the
# gateway PROCESS exists — a ticker that dies silently inside a live gateway
# would otherwise report healthy (#32612, #32895).
TICKER_HEARTBEAT_FILE = CRON_DIR / "ticker_heartbeat"
# Last tick that completed WITHOUT raising. Distinguishing this from the plain
# heartbeat lets status detect a ticker that is alive but failing every tick.
TICKER_SUCCESS_FILE = CRON_DIR / "ticker_last_success"
# Default ticker loop interval (seconds). The single source of truth shared by
# the in-process ticker (cron/scheduler_provider.py) and the staleness
# threshold in `hermes cron status` (hermes_cli/cron.py), so the two never
# drift apart.
TICKER_INTERVAL_SECONDS = 60

# In-process lock protecting load_jobs→modify→save_jobs cycles.
# Required when tick() runs jobs in parallel threads — without this,
# concurrent mark_job_run / advance_next_run calls can clobber each other.
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()
OUTPUT_DIR = CRON_DIR / "output"
ONESHOT_GRACE_SECONDS = 120


def _cron_dir() -> Path:
    return get_hermes_home().resolve() / "cron"


def _jobs_file() -> Path:
    return _cron_dir() / "jobs.json"


def _output_dir() -> Path:
    return _cron_dir() / "output"


def _jobs_lock_file() -> Path:
    """Return the advisory lock path for the current cron directory."""
    return _cron_dir() / ".jobs.lock"


@contextlib.contextmanager
def _jobs_lock():
    """Serialize a load_jobs→modify→save_jobs critical section.

    Combines the in-process threading lock (cheap mutual exclusion between
    the gateway's parallel tick threads) with a cross-process advisory file
    lock on ``<cron dir>/.jobs.lock`` (mutual exclusion between the gateway process
    and standalone ``hermes`` CLI invocations, which previously shared no lock
    at all — a `cron pause` could be silently clobbered by a concurrent
    gateway write, leaving a "paused" job still firing).

    The flock is blocking, but every critical section that uses it is short
    (field updates only — no agent execution), so contention resolves in
    milliseconds. If neither fcntl nor msvcrt is available the manager still
    provides in-process locking, matching the historical behaviour.

    Nested calls in the same thread reuse the held lock so legacy callers that
    invoke save_jobs() inside a broader mutation section don't deadlock or try
    to reacquire the advisory file lock.
    """
    depth = getattr(_jobs_lock_state, "depth", 0)
    if depth:
        _jobs_lock_state.depth = depth + 1
        try:
            yield
        finally:
            _jobs_lock_state.depth -= 1
        return

    with _jobs_file_lock:
        _jobs_lock_state.depth = 1
        lock_fd = None
        try:
            try:
                ensure_dirs()
                lock_fd = open(_jobs_lock_file(), "a+", encoding="utf-8")
                lock_fd.seek(0)
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                elif msvcrt is not None:
                    getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
            except (OSError, IOError) as e:
                # Never let a locking failure take down cron writes — fall back to
                # in-process-only protection (still held via _jobs_file_lock).
                logger.warning("jobs.json cross-process lock unavailable (%s); "
                               "proceeding with in-process lock only", e)
            try:
                yield
            finally:
                if lock_fd is not None:
                    try:
                        if fcntl is not None:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        elif msvcrt is not None:
                            getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
                    except (OSError, IOError):
                        pass
                    finally:
                        lock_fd.close()
        finally:
            _jobs_lock_state.depth = 0

# Fields on a cron job that must never change after creation. ``id`` is used
# as a filesystem path component under ``OUTPUT_DIR``; allowing it to be
# updated lets an unsafe value (``../escape``, absolute path, nested) leak
# into output writes/deletes.
_IMMUTABLE_JOB_FIELDS = frozenset({"id"})


def _job_output_dir(job_id: str) -> Path:
    """Resolve a job's output directory, rejecting any path-escape attempt.

    Job IDs are filesystem path components under ``OUTPUT_DIR``. A legacy or
    crafted ID containing ``..``, absolute paths, or nested separators would
    allow output writes/deletes to escape the cron output sandbox. Reject
    anything that isn't a single safe path component.
    """
    text = str(job_id or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    if Path(text).is_absolute() or Path(text).drive:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    return _output_dir() / text


def _parse_occurrence_instant(value: Any) -> Optional[datetime]:
    """Parse a persisted ISO instant into an aware UTC datetime.

    Job files are user-editable and therefore untrusted.  A malformed or naive
    timestamp must not make the read-only calendar preview endpoint fail or
    silently inherit the gateway host's local timezone.
    """
    if not isinstance(value, str) or len(value) > 128:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def normalize_external_fire_at(value: Any, *, now: Optional[datetime] = None) -> Optional[str]:
    """Validate and canonicalize Chronos' protocol-owned ``fire_at`` value."""
    if value is None:
        return None
    instant = _parse_occurrence_instant(value)
    if instant is None:
        raise ValueError("fire_at must be a bounded aware ISO timestamp")
    reference = (now or _hermes_now()).astimezone(timezone.utc)
    delta = (instant - reference).total_seconds()
    if delta < -_MAX_EXTERNAL_FIRE_AGE_SECONDS or delta > _MAX_EXTERNAL_FIRE_FUTURE_SECONDS:
        raise ValueError("fire_at is outside the accepted execution window")
    return instant.isoformat()


def _cron_output_occurrence_status(head: str) -> str:
    """Best-effort legacy status parser scoped to producer-owned metadata.

    Response and prompt bodies are untrusted LLM/user text.  Never interpret
    headings inside them as execution status. New executions use the
    structured sidecar below; this parser exists only for pre-sidecar output.
    """
    metadata = re.split(r"^##\s+(?:Prompt|Response)\s*$", head, maxsplit=1, flags=re.MULTILINE)[0]
    if re.search(r"^#\s*Cron Job:[^\n]*\(FAILED\)", metadata, re.MULTILINE):
        return "failed"
    match = re.search(r"\*\*Status:\*\*\s*([^\n]+)", metadata)
    if match:
        status = match.group(1).strip().lower()
        if status.startswith("script failed") or status.startswith("blocked"):
            return "failed"
        if status.startswith("silent"):
            return "completed"
    if re.search(r"^##\s*Error\b", metadata, re.MULTILINE):
        return "failed"
    return "completed"


def _occurrence_journal_path(job_id: str) -> Path:
    return _job_output_dir(job_id) / ".occurrences.json"


def _occurrence_output_filename(value: Any) -> Optional[str]:
    """Return a bounded producer-owned markdown basename, never a path."""
    if not isinstance(value, str) or not value or len(value) > 255:
        return None
    if value in {".", ".."} or Path(value).name != value or not value.endswith(".md"):
        return None
    return value


def _read_occurrence_journal(job: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """Read the bounded structured execution journal without following links."""
    try:
        job_dir = _job_output_dir(job.get("id"))
        path = job_dir / ".occurrences.json"
    except ValueError:
        return None, False
    if job_dir.is_symlink():
        return [], True
    try:
        if job_dir.resolve(strict=True).parent != _output_dir().resolve(strict=True):
            return [], True
    except OSError:
        return None, False
    if path.is_symlink() or not path.is_file():
        return None, False
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
        try:
            stat = os.fstat(fd)
            if stat.st_size > _MAX_OCCURRENCE_JOURNAL_BYTES:
                return [], True
            raw = os.read(fd, _MAX_OCCURRENCE_JOURNAL_BYTES + 1)
        finally:
            os.close(fd)
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], True
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return [], True
    records = [item for item in payload["records"] if isinstance(item, dict)]
    return records[:_MAX_OCCURRENCE_JOURNAL_RECORDS], bool(payload.get("truncated"))


def _append_job_occurrence(
    job: Dict[str, Any],
    actual_run_at: datetime,
    *,
    scheduled_at: Optional[datetime] = None,
    output_filename: Optional[str] = None,
    success: bool,
    delivery_error: Optional[str],
) -> None:
    """Persist compact execution truth independently of verbose output retention."""
    try:
        output_root = _output_dir()
        output_root.mkdir(parents=True, exist_ok=True)
        _secure_dir(output_root)
        job_dir = _job_output_dir(job.get("id"))
        if job_dir.is_symlink():
            raise ValueError("cron occurrence output directory is a symlink")
        job_dir.mkdir(parents=True, exist_ok=True)
        _secure_dir(job_dir)
        if job_dir.resolve(strict=True).parent != output_root.resolve(strict=True):
            raise ValueError("cron occurrence output directory escaped sandbox")
        path = _occurrence_journal_path(job.get("id"))
        existing, was_truncated = _read_occurrence_journal(job)
        records = existing or []
        actual_instant = actual_run_at.astimezone(timezone.utc).isoformat()
        scheduled_instant = (scheduled_at or actual_run_at).astimezone(timezone.utc).isoformat()
        status = "delivery_failed" if success and delivery_error else "completed" if success else "failed"
        record = {
            # Use the same identity as the scheduled preview so a completion
            # updates the existing calendar node in place instead of moving it
            # to the finish time and remounting it.
            "id": f"{job.get('id')}:scheduled:{scheduled_instant}",
            "job_id": str(job.get("id") or ""),
            "scheduled_at": scheduled_instant,
            "actual_run_at": actual_instant,
            "status": status,
        }
        safe_output_filename = _occurrence_output_filename(output_filename)
        if safe_output_filename is not None:
            record["output_filename"] = safe_output_filename
        records.append(record)
        truncated = was_truncated or len(records) > _MAX_OCCURRENCE_JOURNAL_RECORDS
        records = records[-_MAX_OCCURRENCE_JOURNAL_RECORDS:]
        fd, tmp_path = tempfile.mkstemp(dir=str(job_dir), suffix=".tmp", prefix=".occurrences_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"records": records, "truncated": truncated}, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_path, path)
            _secure_file(path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        # Calendar history must never turn a successful scheduled side effect
        # into a failed execution.
        logger.warning("Failed to persist occurrence journal for job %r: %s", job.get("id"), exc)


def _historical_job_occurrences(
    job: Dict[str, Any],
    from_at: datetime,
    to_at: datetime,
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Read real execution outputs for one job, without following symlinks."""
    journal, journal_truncated = _read_occurrence_journal(job)
    result: List[Dict[str, Any]] = []
    journal_timestamps: List[float] = []
    journal_output_filenames: Set[str] = set()
    if journal:
        for item in journal:
            output_filename = _occurrence_output_filename(item.get("output_filename"))
            if output_filename is not None:
                journal_output_filenames.add(output_filename)
            instant = _parse_occurrence_instant(item.get("scheduled_at"))
            actual_instant = _parse_occurrence_instant(item.get("actual_run_at"))
            if actual_instant is not None:
                journal_timestamps.append(actual_instant.timestamp())
            elif instant is not None:
                journal_timestamps.append(instant.timestamp())
            # Build de-dup indexes from the full journal before applying the
            # requested scheduled-time window. This prevents a run scheduled
            # before midnight but completed after midnight from resurfacing as
            # a phantom markdown-only occurrence on the next day.
            if instant is None or not from_at <= instant < to_at:
                continue
            if item.get("status") not in {"completed", "failed", "delivery_failed"}:
                continue
            result.append(dict(item))
    journal_timestamps.sort()
    try:
        job_dir = _job_output_dir(job.get("id"))
    except ValueError:
        return result[:limit], journal_truncated
    if job_dir.is_symlink() or not job_dir.is_dir():
        return result[:limit], journal_truncated
    try:
        output_root = _output_dir().resolve(strict=True)
        resolved_job_dir = job_dir.resolve(strict=True)
    except OSError:
        return result[:limit], journal_truncated
    if resolved_job_dir.parent != output_root:
        return result[:limit], journal_truncated

    candidates: List[Tuple[int, str, str]] = []
    scan_truncated = False
    try:
        with os.scandir(resolved_job_dir) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_OCCURRENCE_HISTORY_FILES:
                    scan_truncated = True
                    break
                if not entry.name.endswith(".md") or entry.name.startswith("."):
                    continue
                try:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                instant = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                if from_at <= instant < to_at:
                    candidates.append((stat.st_mtime_ns, entry.name, entry.path))
    except OSError:
        return result[:limit], journal_truncated

    candidates.sort(reverse=True)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    for mtime_ns, filename, path in candidates[:limit]:
        if filename in journal_output_filenames:
            continue
        try:
            fd = os.open(path, os.O_RDONLY | no_follow)
            try:
                head = os.read(fd, _MAX_OCCURRENCE_STATUS_BYTES).decode("utf-8", errors="replace")
                actual = datetime.fromtimestamp(os.fstat(fd).st_mtime, tz=timezone.utc)
            finally:
                os.close(fd)
        except OSError:
            continue
        # New executions have both a verbose markdown output and a compact
        # journal row. Keep the structured row, but still merge older legacy
        # markdown so the first post-upgrade execution cannot erase history.
        actual_timestamp = actual.timestamp()
        journal_index = bisect_left(journal_timestamps, actual_timestamp - 2.0)
        if (
            journal_index < len(journal_timestamps)
            and journal_timestamps[journal_index] <= actual_timestamp + 2.0
        ):
            continue
        result.append({
            "id": f"{job.get('id')}:run:{mtime_ns}:{filename}",
            "job_id": str(job.get("id") or ""),
            "scheduled_at": actual.isoformat(),
            "actual_run_at": actual.isoformat(),
            "status": _cron_output_occurrence_status(head),
        })
    result.sort(key=lambda item: item.get("scheduled_at", ""), reverse=True)
    keep = _cron_output_keep()
    legacy_truncated = keep > 0 and len(candidates) >= keep
    query_truncated = len(candidates) > limit or len(result) > limit
    return result[:limit], journal_truncated or legacy_truncated or scan_truncated or query_truncated


def _next_preview_instant(job: Dict[str, Any], base: datetime) -> Optional[datetime]:
    schedule = job.get("schedule")
    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind == "interval":
        minutes = schedule.get("minutes")
        if (
            not isinstance(minutes, (int, float))
            or isinstance(minutes, bool)
            or not math.isfinite(minutes)
            or minutes <= 0
            or minutes > 525_600_000
        ):
            return None
        try:
            return base + timedelta(minutes=float(minutes))
        except (OverflowError, ValueError):
            return None
    if kind != "cron" or not HAS_CRONITER:
        return None
    expr = schedule.get("expr")
    if not isinstance(expr, str) or len(expr) > 256:
        return None
    tz_name = _normalized_iana_timezone_name(job.get("timezone"))
    zoned_base = base.astimezone(ZoneInfo(tz_name)) if tz_name else base
    try:
        iterator = croniter(expr, zoned_base)
        base_timestamp = zoned_base.timestamp()
        for _ in range(_MAX_CRON_NEXT_RUN_ATTEMPTS):
            candidate = iterator.get_next(datetime)
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=zoned_base.tzinfo)
            if candidate.timestamp() > base_timestamp:
                return candidate.astimezone(timezone.utc)
            folded_candidate = candidate.replace(fold=1)
            if (
                folded_candidate.utcoffset() != candidate.utcoffset()
                and folded_candidate.timestamp() > base_timestamp
            ):
                return folded_candidate.astimezone(timezone.utc)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _in_flight_occurrence(job: Dict[str, Any], now: datetime) -> Optional[Dict[str, datetime]]:
    """Return a fresh, bounded execution claim from the persisted job row.

    ``jobs.json`` can be edited externally, so the claim is treated as
    untrusted input: only short ISO timestamps are accepted and stale/future
    claims are ignored.  The claim contains no paths or user-controlled text.
    """
    raw = job.get("in_flight_occurrence")
    if not isinstance(raw, dict):
        return None
    values: Dict[str, datetime] = {}
    for key in ("scheduled_at", "claimed_at", "original_scheduled_at"):
        value = raw.get(key)
        if value is None and key == "original_scheduled_at":
            continue
        if not isinstance(value, str) or len(value) > 128:
            return None
        parsed = _parse_occurrence_instant(value)
        if parsed is None:
            return None
        values[key] = parsed
    claimed_at = values["claimed_at"]
    age = (now.astimezone(timezone.utc) - claimed_at).total_seconds()
    try:
        configured_ttl = int(os.getenv(
            "HERMES_CRON_OCCURRENCE_LEASE_SECONDS",
            str(_IN_FLIGHT_OCCURRENCE_DEFAULT_TTL_SECONDS),
        ))
    except (TypeError, ValueError):
        configured_ttl = _IN_FLIGHT_OCCURRENCE_DEFAULT_TTL_SECONDS
    ttl_seconds = max(
        _IN_FLIGHT_OCCURRENCE_MIN_TTL_SECONDS,
        min(configured_ttl, _IN_FLIGHT_OCCURRENCE_MAX_TTL_SECONDS),
    )
    if age < -300 or age > ttl_seconds:
        return None
    return values


def _set_in_flight_occurrence(
    job: Dict[str, Any],
    *,
    scheduled_at: datetime,
    claimed_at: datetime,
    original_scheduled_at: Optional[datetime] = None,
) -> None:
    claim = {
        "scheduled_at": scheduled_at.astimezone(timezone.utc).isoformat(),
        "claimed_at": claimed_at.astimezone(timezone.utc).isoformat(),
    }
    if original_scheduled_at is not None:
        claim["original_scheduled_at"] = original_scheduled_at.astimezone(timezone.utc).isoformat()
    job["in_flight_occurrence"] = claim


def _future_job_occurrences(
    job: Dict[str, Any],
    from_at: datetime,
    to_at: datetime,
    now: datetime,
    limit: int,
) -> List[Dict[str, Any]]:
    if not job.get("enabled") or job.get("state") in {"paused", "staged", "completed", "error"}:
        return []
    schedule = job.get("schedule")
    if not isinstance(schedule, dict):
        return []
    if schedule.get("kind") == "interval":
        minutes = schedule.get("minutes")
        if (
            not isinstance(minutes, (int, float))
            or isinstance(minutes, bool)
            or not math.isfinite(minutes)
            or minutes <= 0
            or minutes > 525_600_000
        ):
            return []
    candidate = _parse_occurrence_instant(job.get("next_run_at"))
    in_flight = _in_flight_occurrence(job, now)
    frozen_trigger = in_flight["scheduled_at"] if in_flight else _parse_occurrence_instant(
        job.get("_occurrence_triggered_at")
    )
    if candidate is None and frozen_trigger is None:
        return []
    original_stale_at: Optional[datetime] = None
    if frozen_trigger is not None:
        if in_flight and in_flight.get("original_scheduled_at") is not None:
            original_stale_at = in_flight["original_scheduled_at"]
        elif candidate is not None and candidate < frozen_trigger:
            original_stale_at = candidate
    elif candidate is not None and candidate < now:
        missed_by_seconds = (now - candidate).total_seconds()
        kind = schedule.get("kind")
        should_catch_up_now = (
            kind == "once"
            or (kind in {"cron", "interval"} and missed_by_seconds > _compute_grace_seconds(schedule))
        )
        if should_catch_up_now:
            original_stale_at = candidate
            candidate = now

    repeat = job.get("repeat") if isinstance(job.get("repeat"), dict) else {}
    times = repeat.get("times")
    completed = repeat.get("completed", 0)
    remaining: Optional[int] = None
    if isinstance(times, int) and not isinstance(times, bool):
        done = completed if isinstance(completed, int) and not isinstance(completed, bool) else 0
        remaining = max(0, times - done)
        # Finite one-shots are claimed before their side effect. During that
        # execution window completed==times, but the task is still active and
        # must not blink out of calendar clients before mark_job_run records
        # the real terminal occurrence.
        if (
            remaining == 0
            and schedule.get("kind") == "once"
            and job.get("state") == "scheduled"
            and job.get("enabled")
            and not job.get("last_run_at")
        ):
            remaining = 1

    result: List[Dict[str, Any]] = []
    if frozen_trigger is not None and (remaining is None or remaining > 0):
        if from_at <= frozen_trigger < to_at:
            iso = frozen_trigger.isoformat()
            occurrence = {
                "id": f"{job.get('id')}:scheduled:{iso}",
                "job_id": str(job.get("id") or ""),
                "scheduled_at": iso,
                "status": "scheduled",
            }
            if original_stale_at is not None:
                occurrence["original_scheduled_at"] = original_stale_at.isoformat()
            result.append(occurrence)
        if remaining is not None:
            remaining -= 1
        # The store may already point at the following run (after a ticker or
        # external CAS claim), or an ephemeral due object may still point at
        # this run.  Never project the in-flight identity twice.
        if candidate is not None and candidate <= frozen_trigger:
            candidate = _next_preview_instant(job, frozen_trigger)

    if candidate is None:
        return result
    first = frozen_trigger is None
    iterations = 0
    while (
        candidate < to_at
        and len(result) < limit
        and (remaining is None or remaining > 0)
        and iterations < _MAX_OCCURRENCE_ITERATIONS
    ):
        iterations += 1
        if candidate >= from_at:
            iso = candidate.isoformat()
            occurrence = {
                "id": f"{job.get('id')}:scheduled:{iso}",
                "job_id": str(job.get("id") or ""),
                "scheduled_at": iso,
                "status": "scheduled",
            }
            if original_stale_at is not None and first:
                occurrence["original_scheduled_at"] = original_stale_at.isoformat()
            result.append(occurrence)
        if remaining is not None:
            remaining -= 1
        if schedule.get("kind") == "once":
            break
        # A stale next_run_at represents one catch-up execution, not every tick
        # missed while the gateway was offline.  Hermes re-anchors the following
        # run to the actual catch-up time, so previews must do the same.
        base = candidate
        candidate = _next_preview_instant(job, base)
        if candidate is None:
            break
        first = False
    return result


def _collect_job_occurrences(
    jobs: List[Dict[str, Any]],
    from_at: datetime,
    to_at: datetime,
    *,
    now: Optional[datetime] = None,
    limit: int = 2_000,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Collect occurrences and completeness metadata in one bounded scan."""
    if from_at.tzinfo is None or to_at.tzinfo is None:
        raise ValueError("occurrence bounds must include a timezone")
    start = from_at.astimezone(timezone.utc)
    end = to_at.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("occurrence 'to' must be after 'from'")
    bounded_limit = max(1, min(int(limit), 2_000))
    current = (now or _hermes_now()).astimezone(timezone.utc)
    bounded_jobs = jobs[:_MAX_OCCURRENCE_QUERY_JOBS]
    history_truncated = len(jobs) > len(bounded_jobs)
    per_job_occurrences: List[List[Dict[str, Any]]] = []
    for job in bounded_jobs:
        history, truncated = _historical_job_occurrences(job, start, end, bounded_limit)
        history_truncated = history_truncated or truncated
        projected = history + _future_job_occurrences(job, start, end, current, bounded_limit)
        projected.sort(key=lambda item: (item.get("scheduled_at", ""), item.get("id", "")))
        per_job_occurrences.append(projected)

    # Select in rounds before the final chronological sort. This prevents one
    # every-minute task from consuming the entire global cap while other jobs
    # still have occurrences in the requested range.
    occurrences: List[Dict[str, Any]] = []
    round_index = 0
    while len(occurrences) < bounded_limit:
        added = False
        for projected in per_job_occurrences:
            if round_index < len(projected):
                occurrences.append(projected[round_index])
                added = True
                if len(occurrences) >= bounded_limit:
                    break
        if not added:
            break
        round_index += 1
    occurrences.sort(key=lambda item: (item.get("scheduled_at", ""), item.get("id", "")))
    if sum(len(projected) for projected in per_job_occurrences) > len(occurrences):
        history_truncated = True
    return occurrences, history_truncated


def list_job_occurrences(
    jobs: List[Dict[str, Any]],
    from_at: datetime,
    to_at: datetime,
    *,
    now: Optional[datetime] = None,
    limit: int = 2_000,
) -> List[Dict[str, Any]]:
    """Return real past runs plus read-only future previews for calendar UIs.

    This function never mutates jobs.json and never calls ``compute_next_run``;
    the scheduler's state transition path remains completely untouched.
    """
    occurrences, _ = _collect_job_occurrences(jobs, from_at, to_at, now=now, limit=limit)
    return occurrences


def job_occurrence_projection(
    jobs: List[Dict[str, Any]],
    from_at: datetime,
    to_at: datetime,
    *,
    now: Optional[datetime] = None,
    limit: int = 2_000,
) -> Dict[str, Any]:
    """Occurrence payload with an explicit history completeness contract."""
    occurrences, history_truncated = _collect_job_occurrences(
        jobs,
        from_at,
        to_at,
        now=now,
        limit=limit,
    )
    return {
        "occurrences": occurrences,
        "history_truncated": history_truncated,
    }


def _normalize_skill_list(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    """Normalize legacy/single-skill and multi-skill inputs into a unique ordered list."""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _apply_skill_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a job dict with canonical `skills` and legacy `skill` fields aligned."""
    normalized = dict(job)
    skills = _normalize_skill_list(normalized.get("skill"), normalized.get("skills"))
    normalized["skills"] = skills
    normalized["skill"] = skills[0] if skills else None
    return normalized


def _coerce_job_text(value: Any, fallback: str = "") -> str:
    """Coerce legacy/hand-edited nullable cron fields to strings for readers."""
    if value is None:
        return fallback
    return str(value)


def _schedule_display_for_job(job: Dict[str, Any]) -> str:
    display = _coerce_job_text(job.get("schedule_display")).strip()
    if display:
        return display

    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        for key in ("display", "value", "expr", "run_at"):
            text = _coerce_job_text(schedule.get(key)).strip()
            if text:
                return text
    elif schedule is not None:
        return str(schedule)

    return "?"


def _normalize_job_record(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a read-safe cron job shape for UI/API/tool/scheduler consumers.

    Older or hand-edited jobs can have nullable fields like ``prompt``,
    ``name``, or ``schedule_display``.  Keep storage untouched on read, but
    ensure consumers never crash while formatting or running those records.
    """
    normalized = _apply_skill_fields(job)
    job_id = _coerce_job_text(normalized.get("id"), "unknown")
    prompt = _coerce_job_text(normalized.get("prompt"))
    normalized["id"] = job_id
    normalized["prompt"] = prompt

    name = _coerce_job_text(normalized.get("name")).strip()
    if not name:
        script = _coerce_job_text(normalized.get("script")).strip()
        label_source = (
            prompt
            or (normalized["skills"][0] if normalized.get("skills") else "")
            or script
            or job_id
            or "cron job"
        )
        name = label_source[:50].strip() or "cron job"
    normalized["name"] = name
    normalized["schedule_display"] = _schedule_display_for_job(normalized)

    state = _coerce_job_text(normalized.get("state")).strip()
    if not state:
        state = "scheduled" if normalized.get("enabled", True) else "paused"
    normalized["state"] = state

    raw_output_language = normalized.get("output_language")
    output_language = normalize_output_language_tag(raw_output_language)
    if output_language is not None:
        normalized["output_language"] = output_language
    else:
        # Missing and invalid values both mean "legacy fallback". In
        # particular, never pass hand-edited arbitrary text to a system prompt.
        normalized.pop("output_language", None)
        if raw_output_language is not None and raw_output_language != "":
            logger.warning(
                "Ignoring invalid output_language on cron job %r",
                normalized.get("id"),
            )

    return normalized


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass  # Windows or other platforms where chmod is not supported


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def ensure_dirs():
    """Ensure cron directories exist with secure permissions."""
    cron_dir = _cron_dir()
    output_dir = _output_dir()
    cron_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(cron_dir)
    _secure_dir(output_dir)


# =============================================================================
# Schedule Parsing
# =============================================================================

def parse_duration(s: str) -> int:
    """
    Parse duration string into minutes.
    
    Examples:
        "30m" → 30
        "2h" → 120
        "1d" → 1440
    """
    s = s.strip().lower()
    match = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(f"Invalid duration: '{s}'. Use format like '30m', '2h', or '1d'")
    
    value = int(match.group(1))
    unit = match.group(2)[0]  # First char: m, h, or d
    
    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    return value * multipliers[unit]


def parse_schedule(schedule: str, *, tz_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse schedule string into structured format.

    Returns dict with:
        - kind: "once" | "interval" | "cron"
        - For "once": "run_at" (ISO timestamp)
        - For "interval": "minutes" (int)
        - For "cron": "expr" (cron expression)

    ``tz_name`` is an optional IANA timezone (e.g. ``"Asia/Shanghai"``) used to
    anchor *naive* ISO timestamps. With ``tz_name`` set, ``"2026-05-25T10:30"``
    is interpreted as 10:30 wall-clock in that zone instead of the system local
    zone — without it, the result depends on where hermes happens to run.

    Examples:
        "30m"              → once in 30 minutes
        "2h"               → once in 2 hours
        "every 30m"        → recurring every 30 minutes
        "every 2h"         → recurring every 2 hours
        "0 9 * * *"        → cron expression
        "2026-02-03T14:00" → once at timestamp
    """
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()
    
    # "every X" pattern → recurring interval
    if schedule_lower.startswith("every "):
        duration_str = schedule[6:].strip()
        minutes = parse_duration(duration_str)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m"
        }
    
    # Check for cron expression (5 or 6 space-separated fields)
    # Cron fields: minute hour day month weekday [year]
    parts = schedule.split()
    if len(parts) >= 5 and all(
        re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]
    ):
        if not HAS_CRONITER:
            raise ValueError("Cron expressions require 'croniter' package. Install with: pip install croniter")
        # Validate cron expression
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{schedule}': {e}")
        return {
            "kind": "cron",
            "expr": schedule,
            "display": schedule
        }
    
    # ISO timestamp (contains T or looks like date)
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            # Parse and validate
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            # Make naive timestamps timezone-aware at parse time so the stored
            # value doesn't depend on the system timezone matching at check time.
            # When the caller supplied a tz_name (per-job timezone), interpret
            # the naive wall-clock in that zone. Otherwise anchor to the
            # CONFIGURED Hermes timezone, not the server's local timezone. The
            # due-check (`get_due_jobs`) compares `next_run_at` against
            # `hermes_time.now()`, which uses the configured zone (#51021).
            if dt.tzinfo is None:
                anchor_tz = None
                if tz_name:
                    try:
                        anchor_tz = ZoneInfo(tz_name)
                    except (ZoneInfoNotFoundError, ValueError):
                        logger.warning(
                            "parse_schedule: invalid tz_name %r, falling back "
                            "to configured Hermes timezone; "
                            "create_job._validate_tz_name "
                            "should normally catch this earlier",
                            tz_name,
                        )
                        anchor_tz = None
                if anchor_tz is None:
                    anchor_tz = _hermes_now().tzinfo
                dt = dt.replace(tzinfo=anchor_tz)
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"
            }
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")
    
    # Duration like "30m", "2h", "1d" → one-shot from now
    try:
        minutes = parse_duration(schedule)
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {original}"
        }
    except ValueError:
        pass
    
    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Duration: '30m', '2h', '1d' (one-shot)\n"
        f"  - Interval: 'every 30m', 'every 2h' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _ensure_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in Hermes configured timezone.

    Backward compatibility:
    - Older stored timestamps may be naive.
    - Naive values are interpreted as *system-local wall time* (the timezone
      `datetime.now()` used when they were created), then converted to the
      configured Hermes timezone.

    This preserves relative ordering for legacy naive timestamps across
    timezone changes and avoids false not-due results.
    """
    target_tz = _hermes_now().tzinfo
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz).astimezone(target_tz)
    return dt.astimezone(target_tz)


def _timezone_offset_mismatch(stored: datetime, current: datetime) -> bool:
    """Return True when a stored aware timestamp uses a different UTC offset.

    Naive stored timestamps return False: they carry no offset to compare, and
    are normalized by ``_ensure_aware`` instead — they intentionally never take
    the offset-repair path.
    """
    if stored.tzinfo is None or current.tzinfo is None:
        return False
    return stored.utcoffset() != current.utcoffset()


def _stored_wall_clock_is_future(stored: datetime, current: datetime) -> bool:
    """Return True when the stored local wall-clock time has not arrived yet.

    Cron schedules express local wall-clock intent. If Hermes/system local time
    changes after next_run_at was persisted, an old offset can make a future
    wall-clock run look due at the converted absolute time (for example
    21:00+10 becomes 13:00+02). Comparing naive wall-clock values lets us
    distinguish that migration case from a genuinely missed run whose scheduled
    wall time has already passed.
    """
    return stored.replace(tzinfo=None) > current.replace(tzinfo=None)


def _recoverable_oneshot_run_at(
    schedule: Dict[str, Any],
    now: datetime,
    *,
    last_run_at: Optional[str] = None,
) -> Optional[str]:
    """Return a one-shot run time if it is still eligible to fire.

    One-shot jobs get a small grace window so jobs created a few seconds after
    their requested minute still run on the next tick. Once a one-shot has
    already run, it is never eligible again.
    """
    if schedule.get("kind") != "once":
        return None
    if last_run_at:
        return None

    run_at = schedule.get("run_at")
    if not run_at:
        return None

    run_at_dt = _ensure_aware(datetime.fromisoformat(run_at))
    if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


def _stale_oneshot_error(schedule: Dict[str, Any], fallback: Any) -> ValueError:
    run_at = schedule.get("run_at") or schedule.get("display") or fallback
    return ValueError(
        f"One-shot schedule is in the past and cannot be scheduled: {run_at}"
    )


def _compute_grace_seconds(schedule: dict) -> int:
    """Compute the lateness threshold used to classify a missed recurring run.

    Uses half the schedule period, clamped between 120 seconds and 2 hours.
    Stale runs beyond this threshold are still caught up once; the threshold is
    kept for diagnostics/logging so operators can distinguish a normal late tick
    from a gateway-down or device-sleep catch-up.
    """
    MIN_GRACE = 120
    MAX_GRACE = 7200  # 2 hours

    kind = schedule.get("kind")

    if kind == "interval":
        period_seconds = schedule.get("minutes", 1) * 60
        grace = period_seconds // 2
        return max(MIN_GRACE, min(grace, MAX_GRACE))

    if kind == "cron" and HAS_CRONITER:
        try:
            now = _hermes_now()
            cron = croniter(schedule["expr"], now)
            first = cron.get_next(datetime)
            second = cron.get_next(datetime)
            period_seconds = int((second - first).total_seconds())
            grace = period_seconds // 2
            return max(MIN_GRACE, min(grace, MAX_GRACE))
        except Exception:
            pass

    return MIN_GRACE


def compute_next_run(
    schedule: Dict[str, Any],
    last_run_at: Optional[str] = None,
    *,
    tz_name: Optional[str] = None,
) -> Optional[str]:
    """
    Compute the next run time for a schedule.

    Returns ISO timestamp string, or None if no more runs.

    ``tz_name`` is a per-job IANA timezone (e.g. ``"Asia/Shanghai"``).
    Only the cron branch is timezone-sensitive — ``"6 23 * * *"`` means
    different wall-clock instants in different zones. Interval/once jobs
    operate on absolute datetimes, so the job-level tz doesn't change
    their behaviour. When ``tz_name`` is None, fall back to the hermes
    instance's configured timezone via _hermes_now().
    """
    now = _hermes_now()

    if schedule["kind"] == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)

    elif schedule["kind"] == "interval":
        minutes = schedule["minutes"]
        if last_run_at:
            # Next run is last_run + interval
            last = _ensure_aware(datetime.fromisoformat(last_run_at))
            next_run = last + timedelta(minutes=minutes)
        else:
            # First run is now + interval
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()

    elif schedule["kind"] == "cron":
        if not HAS_CRONITER:
            logger.warning(
                "Cannot compute next run for cron schedule %r: 'croniter' is "
                "not installed. croniter is a core dependency as of v0.9.x; "
                "reinstall hermes-agent or run 'pip install croniter' in your "
                "runtime env.",
                schedule.get("expr"),
            )
            return None

        # Resolve the timezone to evaluate the cron expression in.
        # Per-job tz wins; otherwise inherit the hermes instance's tz.
        job_tz = None
        normalized_tz_name = _normalized_iana_timezone_name(tz_name)
        if normalized_tz_name is not None:
            job_tz = ZoneInfo(normalized_tz_name)
        elif tz_name not in (None, ""):
            invalid_tz = (
                tz_name[:128] + ("..." if len(tz_name) > 128 else "")
                if isinstance(tz_name, str)
                else f"<{type(tz_name).__name__}>"
            )
            logger.warning(
                "Invalid per-job timezone %r; falling back to hermes default.",
                invalid_tz,
            )

        # Use last_run_at as the croniter base when available, consistent
        # with interval jobs.  This ensures that after a crash/restart,
        # the next run is anchored to the actual last execution time
        # rather than to an arbitrary restart time.
        if last_run_at:
            base_time = _ensure_aware(datetime.fromisoformat(last_run_at))
        else:
            base_time = now
        if job_tz is not None:
            base_time = base_time.astimezone(job_tz)

        cron = croniter(schedule["expr"], base_time)
        base_timestamp = base_time.timestamp()
        for _ in range(_MAX_CRON_NEXT_RUN_ATTEMPTS):
            next_run = cron.get_next(datetime)
            if next_run.timestamp() > base_timestamp:
                return next_run.isoformat()

            # During a DST fall-back, croniter can return the first occurrence
            # of an ambiguous wall time (fold=0) even when the base is already
            # in the repeated hour (fold=1).  The wall time looks later, but its
            # absolute timestamp is in the past.  Prefer the second occurrence
            # when it is both genuinely ambiguous and strictly after the base.
            folded_next_run = next_run.replace(fold=1)
            if (
                folded_next_run.utcoffset() != next_run.utcoffset()
                and folded_next_run.timestamp() > base_timestamp
            ):
                return folded_next_run.isoformat()

        cron_expr = schedule.get("expr")
        cron_expr_display = (
            cron_expr[:128] + ("..." if len(cron_expr) > 128 else "")
            if isinstance(cron_expr, str)
            else f"<{type(cron_expr).__name__}>"
        )
        logger.error(
            "Cron schedule %r did not produce a next run strictly after %s "
            "within %d attempts.",
            cron_expr_display,
            base_time.isoformat(),
            _MAX_CRON_NEXT_RUN_ATTEMPTS,
        )
        return None

    return None


# =============================================================================
# Ticker heartbeat (liveness signal for `hermes cron status`)
# =============================================================================

def _atomic_write_epoch(path: Path) -> None:
    """Atomically write the current epoch time to ``path``.

    Uses the same tmpfile + ``atomic_replace`` pattern as ``save_jobs`` so a
    concurrent reader in another process (``hermes cron status``) never sees a
    torn/truncated file. Best-effort: failures are swallowed by callers.
    """
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=str(CRON_DIR), suffix=".tmp", prefix=".hb_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(time.time()))
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def record_ticker_heartbeat(success: bool = False) -> None:
    """Record a ticker liveness signal, and optionally a successful-tick signal.

    The ticker calls this once per loop iteration. ``success=True`` additionally
    bumps the *last successful tick* marker. We track two distinct signals so
    `hermes cron status` can tell a thread that is merely *alive and looping*
    (heartbeat fresh, success stale) from one that is actually *firing jobs*
    (both fresh) — a ticker stuck failing every tick would otherwise keep the
    plain heartbeat fresh and falsely report healthy (#32612, #32895).

    Best-effort: a write failure must never disrupt the tick loop.
    """
    try:
        _atomic_write_epoch(TICKER_HEARTBEAT_FILE)
    except Exception:
        pass
    if success:
        try:
            _atomic_write_epoch(TICKER_SUCCESS_FILE)
        except Exception:
            pass


def _epoch_file_age(path: Path) -> Optional[float]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return max(0.0, time.time() - float(raw))
    except Exception:
        return None


def get_ticker_heartbeat_age() -> Optional[float]:
    """Seconds since the ticker loop last iterated, or None if unknown.

    None = heartbeat file missing/unreadable (older build, never ran, or a
    torn read). Callers treat None as "cannot determine", not "dead".
    """
    return _epoch_file_age(TICKER_HEARTBEAT_FILE)


def get_ticker_success_age() -> Optional[float]:
    """Seconds since the ticker last completed a tick WITHOUT raising, or None."""
    return _epoch_file_age(TICKER_SUCCESS_FILE)


# =============================================================================
# Job CRUD Operations
# =============================================================================

def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from storage."""
    ensure_dirs()
    jobs_file = _jobs_file()
    if not jobs_file.exists():
        return []

    _strict_retry = False  # track whether we used the strict=False fallback

    try:
        with open(jobs_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Retry with strict=False to handle bare control chars in string values
        _strict_retry = True
        try:
            with open(jobs_file, 'r', encoding='utf-8') as f:
                data = json.loads(f.read(), strict=False)
        except Exception as e:
            logger.error("Failed to auto-repair jobs.json: %s", e)
            raise RuntimeError(f"Cron database corrupted and unrepairable: {e}") from e
    except IOError as e:
        logger.error("IOError reading jobs.json: %s", e)
        raise RuntimeError(f"Failed to read cron database: {e}") from e

    # Validate the top-level JSON shape: accept a dict (expected) or a bare
    # list (auto-repair). Anything else (str/number/null) is corruption that
    # would otherwise raise an uncaught AttributeError on ``.get()`` and take
    # down the whole cron subsystem.
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        if _strict_retry and jobs:
            # Hit control-character corruption — rewrite with proper escaping.
            save_jobs(jobs)
            logger.warning("Auto-repaired jobs.json (had invalid control characters)")
        return jobs
    if isinstance(data, list):
        # Bare array — likely saved/edited outside save_jobs(). Wrap it back
        # into the expected {"jobs": [...]} structure.
        if data:
            save_jobs(data)
            logger.warning("Auto-repaired jobs.json (bare list wrapped as dict)")
        return data

    raise RuntimeError(
        f"Cron database corrupted: expected {{'jobs': [...]}}, got {type(data).__name__}"
    )


def _save_jobs_unlocked(jobs: List[Dict[str, Any]]):
    """Save all jobs to storage. Caller must hold _jobs_lock()."""
    ensure_dirs()
    jobs_file = _jobs_file()
    fd, tmp_path = tempfile.mkstemp(dir=str(jobs_file.parent), suffix='.tmp', prefix='.jobs_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"jobs": jobs, "updated_at": _hermes_now().isoformat()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, jobs_file)
        _secure_file(jobs_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_jobs(jobs: List[Dict[str, Any]]):
    """Save all jobs to storage."""
    with _jobs_lock():
        _save_jobs_unlocked(jobs)


def _normalize_workdir(workdir: Optional[str]) -> Optional[str]:
    """Normalize and validate a cron job workdir.

    Rules:
      - Empty / None → None (feature off, preserves old behaviour).
      - ``~`` is expanded.  Relative paths are rejected — cron jobs run detached
        from any shell cwd, so relative paths have no stable meaning.
      - The path must exist and be a directory at create/update time.  We do
        NOT re-check at run time (a user might briefly unmount the dir; the
        scheduler will just fall back to old behaviour with a logged warning).

    Returns the absolute path string, or None when disabled.
    Raises ValueError on invalid input.
    """
    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r}). "
            f"Cron jobs run detached from any shell cwd, so relative paths are ambiguous."
        )
    resolved = expanded.resolve()
    if not resolved.exists():
        raise ValueError(f"Cron workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Cron workdir is not a directory: {resolved}")
    return str(resolved)


def _resolve_default_model_snapshot() -> Optional[str]:
    """Resolve the global default model the same way the cron ticker does.

    Mirrors the unpinned-model resolution in ``cron/scheduler.py`` ``run_job``:
    read ``config.yaml`` ``model.default`` (or the ``model`` alias / bare string
    form), applying the managed-scope overlay and env expansion. Used by
    ``create_job`` to snapshot the default model for unpinned jobs so a later
    swap of the global default is detected at fire time (#44585).

    Returns the resolved model string, or ``None`` if config is missing/empty
    or resolution fails (fail-open — caller treats ``None`` as "no snapshot").
    """
    try:
        import yaml
        from hermes_cli.config import _expand_env_vars

        cfg_path = get_hermes_home() / "config.yaml"
        if not cfg_path.exists():
            return None
        with cfg_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
        cfg = _expand_env_vars(cfg)
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            return model_cfg.strip() or None
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default") or model_cfg.get("model")
            if isinstance(default, str):
                return default.strip() or None
        return None
    except Exception:
        return None


def _normalize_job_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _compute_provider_model_snapshots(
    *,
    provider: Any,
    model: Any,
    base_url: Any,
    no_agent: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Snapshot unpinned inference axes for the provider/model drift guard.

    Agent cron jobs with unpinned provider/model follow global config at fire
    time. Capture the current resolution for each unpinned axis so a later
    global switch fails closed instead of silently changing spend. Pinned axes
    and no-agent script jobs intentionally carry no snapshot.
    """
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_model = _normalize_job_optional_text(model)
    normalized_base_url = _normalize_job_optional_text(
        base_url,
        strip_trailing_slash=True,
    )
    if bool(no_agent):
        return None, None

    provider_snapshot: Optional[str] = None
    model_snapshot: Optional[str] = None
    if normalized_provider is None:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime_kwargs = {"requested": None}
            if normalized_base_url:
                runtime_kwargs["explicit_base_url"] = normalized_base_url
            snap = resolve_runtime_provider(**runtime_kwargs)
            snap_provider = str(snap.get("provider") or "").strip().lower()
            provider_snapshot = snap_provider or None
        except Exception:
            provider_snapshot = None
    if normalized_model is None:
        try:
            model_snapshot = _resolve_default_model_snapshot() or None
        except Exception:
            model_snapshot = None
    return provider_snapshot, model_snapshot


def _normalized_inference_axes(job: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Return the stored inference-routing fields in their semantic form."""
    return (
        _normalize_job_optional_text(job.get("provider")),
        _normalize_job_optional_text(job.get("model")),
        _normalize_job_optional_text(job.get("base_url"), strip_trailing_slash=True),
        bool(job.get("no_agent")),
    )


def create_job(
    prompt: Optional[str],
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    script: Optional[str] = None,
    context_from: Optional[Union[str, List[str]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: bool = False,
    attach_to_session: Optional[bool] = None,
    timezone: Optional[str] = None,
    output_language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new cron job.

    Args:
        prompt: The prompt to run (must be self-contained, or a task instruction when skill is set).
                Ignored when ``no_agent=True`` except as an optional name hint.
        schedule: Schedule string (see parse_schedule)
        name: Optional friendly name
        repeat: How many times to run (None = forever, 1 = once)
        deliver: Where to deliver output ("origin", "local", "telegram", etc.)
        origin: Source info where job was created (for "origin" delivery)
        skill: Optional legacy single skill name to load before running the prompt
        skills: Optional ordered list of skills to load before running the prompt
        model: Optional per-job model override
        provider: Optional per-job provider override
        base_url: Optional per-job base URL override
        script: Optional path to a script whose stdout feeds the job. With
                ``no_agent=True`` the script IS the job — its stdout is
                delivered verbatim. Without ``no_agent``, its stdout is
                injected into the agent's prompt as context (data-collection /
                change-detection pattern). Paths resolve under
                ~/.hermes/scripts/; ``.sh`` / ``.bash`` files run via bash,
                anything else via Python.
        context_from: Optional job ID (or list of job IDs) whose most recent output
                      is injected into the prompt as context before each run.
                      Useful for chaining cron jobs: job A finds data, job B processes it.
        enabled_toolsets: Optional list of toolset names to restrict the agent to.
                          When set, only tools from these toolsets are loaded, reducing
                          token overhead. When omitted, all default tools are loaded.
                          Ignored when ``no_agent=True``.
        workdir: Optional absolute path.  When set, the job runs as if launched
                from that directory: AGENTS.md / CLAUDE.md / .cursorrules from
                that directory are injected into the system prompt, and the
                terminal/file/code_exec tools use it as their working directory
                (via TERMINAL_CWD).  When unset, the old behaviour is preserved
                (no context files injected, tools use the scheduler's cwd).
                With ``no_agent=True``, ``workdir`` is still applied as the
                script's cwd so relative paths inside the script behave
                predictably.
        no_agent: When True, skip the agent entirely — run ``script`` on schedule
                and deliver its stdout directly. Empty stdout = silent (no
                delivery). Requires ``script`` to be set. Ideal for classic
                watchdogs and periodic alerts that don't need LLM reasoning.
        output_language: Optional canonical BCP 47 language tag captured when
                         an LLM creates an agent job. Direct/legacy callers may
                         omit it; script-only jobs ignore it.

    Returns:
        The created job dict
    """
    # Validate the per-job timezone up-front so parse_schedule can honour it
    # for naive ISO timestamps (e.g. "2026-05-25T10:30" → 10:30 wall in tz).
    normalized_tz = _validate_tz_name(timezone)
    if normalized_tz is None:
        # All-fixed policy (ZET-1258): pin the device's current timezone at
        # creation so wall-clock is deterministic no matter which path created
        # the job (LLM cronjob tool / HTTP) — both converge here. To restore
        # follow-live later, thread an opt-out param to skip this.
        from hermes_time import get_timezone_name
        normalized_tz = get_timezone_name()
    parsed_schedule = parse_schedule(schedule, tz_name=normalized_tz)

    # Normalize repeat: treat 0 or negative values as None (infinite)
    if repeat is not None and repeat <= 0:
        repeat = None

    # Auto-set repeat=1 for one-shot schedules if not specified
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1

    # Default delivery to origin if available, otherwise local
    if deliver is None:
        deliver = "origin" if origin else "local"

    initial_next_run_at = compute_next_run(parsed_schedule, tz_name=normalized_tz)
    if parsed_schedule["kind"] == "once" and initial_next_run_at is None:
        raise _stale_oneshot_error(parsed_schedule, schedule)

    job_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()

    normalized_skills = _normalize_skill_list(skill, skills)
    normalized_model = _normalize_job_optional_text(model)
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_base_url = _normalize_job_optional_text(base_url, strip_trailing_slash=True)
    normalized_script = str(script).strip() if isinstance(script, str) else None
    normalized_script = normalized_script or None
    normalized_toolsets = [str(t).strip() for t in enabled_toolsets if str(t).strip()] if enabled_toolsets else None
    normalized_toolsets = normalized_toolsets or None
    normalized_workdir = _normalize_workdir(workdir)
    normalized_no_agent = bool(no_agent)
    normalized_attach = attach_to_session if isinstance(attach_to_session, bool) else None
    normalized_output_language = (
        None
        if normalized_no_agent
        else validate_output_language_tag(output_language)
    )

    # no_agent jobs are meaningless without a script — the script IS the job.
    # Surface this as a clear ValueError at create time so bad configs never
    # reach the scheduler.
    if normalized_no_agent and not normalized_script:
        raise ValueError(
            "no_agent=True requires a script — with no agent and no script "
            "there is nothing for the job to run."
        )

    # Normalize context_from: accept str or list of str, store as list or None
    if isinstance(context_from, str):
        context_from = [context_from.strip()] if context_from.strip() else None
    elif isinstance(context_from, list):
        context_from = [str(j).strip() for j in context_from if str(j).strip()] or None
    else:
        context_from = None

    prompt_text = _coerce_job_text(prompt)

    # Reject cron jobs that schedule gateway-lifecycle commands. Prevents
    # agent-driven SIGTERM-respawn loops under launchd/systemd KeepAlive
    # (#30719). Enforced here (not only in the CLI layer) so the agent's
    # `cronjob` model tool — which calls create_job directly — is also
    # covered, not just `hermes cron create`.
    from cron.lifecycle_guard import check_gateway_lifecycle
    check_gateway_lifecycle(prompt_text, normalized_script)

    label_source = (prompt_text or (normalized_skills[0] if normalized_skills else None) or (normalized_script if normalized_no_agent else None)) or "cron job"

    provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
        provider=normalized_provider,
        model=normalized_model,
        base_url=normalized_base_url,
        no_agent=normalized_no_agent,
    )

    job = {
        "id": job_id,
        "name": name or label_source[:50].strip(),
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": normalized_model,
        "provider": normalized_provider,
        # Provider/model resolution captured at creation for unpinned jobs
        # (#44585). None for pinned axes, no_agent jobs, resolution failures, and
        # any pre-existing job written before these fields existed (back-compat).
        "provider_snapshot": provider_snapshot,
        "model_snapshot": model_snapshot,
        "base_url": normalized_base_url,
        "script": normalized_script,
        "no_agent": normalized_no_agent,
        "context_from": context_from,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {
            "times": repeat,  # None = forever
            "completed": 0
        },
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": initial_next_run_at,
        "last_run_at": None,
        "timezone": normalized_tz,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        # Delivery configuration
        "deliver": deliver,
        "origin": origin,  # Tracks where job was created for "origin" delivery
        "enabled_toolsets": normalized_toolsets,
        "workdir": normalized_workdir,
    }
    # Only persist attach_to_session when explicitly set, so existing jobs and
    # the common case stay byte-identical (absent key => fall back to the
    # global cron.mirror_delivery config, default off).
    if normalized_attach is not None:
        job["attach_to_session"] = normalized_attach
    if normalized_output_language is not None:
        job["output_language"] = normalized_output_language

    with _jobs_lock():
        jobs = load_jobs()
        jobs.append(job)
        save_jobs(jobs)

    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return _normalize_job_record(job)
    return None


def get_job_raw(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the persisted job shape without reader normalization.

    This is intentionally narrow: wire-contract validators sometimes need to
    distinguish JSON ``null`` from an empty string.  Callers must treat the
    returned record as untrusted persisted input and validate its full shape
    before acting on it.
    """
    for job in load_jobs():
        if isinstance(job, dict) and job.get("id") == job_id:
            return job
    return None


class AmbiguousJobReference(LookupError):
    """Raised when a job name matches more than one job."""

    def __init__(self, ref: str, matches: List[Dict[str, Any]]):
        self.ref = ref
        self.matches = matches
        ids = ", ".join(m["id"] for m in matches)
        super().__init__(
            f"Job name '{ref}' is ambiguous — matches {len(matches)} jobs: {ids}. "
            f"Use the job ID instead."
        )


def resolve_job_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a job reference (ID or name) to a job record.

    - Exact ID match wins (works even if a different job's name equals this ID).
    - Otherwise, case-insensitive name match.
    - If a name matches more than one job, raises AmbiguousJobReference so the
      caller can surface the matching IDs rather than silently picking one.
    """
    if not ref:
        return None
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == ref:
            return _normalize_job_record(job)
    ref_lower = ref.lower()
    name_matches = [j for j in jobs if (j.get("name") or "").lower() == ref_lower]
    if not name_matches:
        return None
    if len(name_matches) > 1:
        raise AmbiguousJobReference(
            ref, [_normalize_job_record(j) for j in name_matches]
        )
    return _normalize_job_record(name_matches[0])


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """List all jobs, optionally including disabled ones."""
    jobs = [_normalize_job_record(j) for j in load_jobs()]
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    return jobs


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a job by ID, refreshing derived schedule fields when needed."""
    # Block mutation of immutable fields. ``id`` in particular is a filesystem
    # path component under OUTPUT_DIR — letting an update change it leaks
    # path-escape values into output writes/deletes.
    bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
    if bad_fields:
        raise ValueError(
            f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}"
        )

    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            # Validate / normalize workdir if present in updates.  Empty string
            # or None both mean "clear the field" (restore old behaviour).
            if "workdir" in updates:
                _wd = updates["workdir"]
                if _wd in {None, "", False}:
                    updates["workdir"] = None
                else:
                    updates["workdir"] = _normalize_workdir(_wd)

            previous_inference_axes = _normalized_inference_axes(job)
            # Validate timezone if present.  Empty / None clears the per-job tz
            # and falls back to the hermes instance's configured tz.
            if "timezone" in updates:
                updates["timezone"] = _validate_tz_name(updates["timezone"])
            if "output_language" in updates:
                updates["output_language"] = validate_output_language_tag(
                    updates["output_language"]
                )

            updated = _apply_skill_fields({**job, **updates})
            schedule_changed = "schedule" in updates
            inference_fields_changed = bool(
                {"provider", "model", "base_url", "no_agent"}.intersection(updates)
            ) and _normalized_inference_axes(updated) != previous_inference_axes
            # A bare timezone change must also recompute next_run_at — otherwise
            # the user fixes their tz and the next firing still uses the old
            # wall-clock until the next mark_job_run.
            timezone_changed = (
                "timezone" in updates and updates["timezone"] != job.get("timezone")
            )

            if "skills" in updates or "skill" in updates:
                normalized_skills = _normalize_skill_list(updated.get("skill"), updated.get("skills"))
                updated["skills"] = normalized_skills
                updated["skill"] = normalized_skills[0] if normalized_skills else None

            if schedule_changed:
                updated_schedule = updated["schedule"]
                # The API may pass schedule as a raw string (e.g. "every 10m")
                # instead of a pre-parsed dict.  Normalize it the same way
                # create_job() does so downstream code can call .get() safely.
                if isinstance(updated_schedule, str):
                    updated_schedule = parse_schedule(
                        updated_schedule, tz_name=updated.get("timezone")
                    )
                    updated["schedule"] = updated_schedule
                updated["schedule_display"] = updates.get(
                    "schedule_display",
                    updated_schedule.get("display", updated.get("schedule_display")),
                )

            # interval/once next_run 与 tz 无关：纯 tz 变更只该让 cron 重算，否则
            # compute_next_run(无 last_run_at) 会把 interval 重置成 now+间隔。
            tz_only_recompute = timezone_changed and updated["schedule"].get("kind") == "cron"
            if (schedule_changed or tz_only_recompute) and updated.get("state") != "paused":
                updated["next_run_at"] = compute_next_run(
                    updated["schedule"], tz_name=updated.get("timezone")
                )
                if updated["schedule"].get("kind") == "once" and not updated["next_run_at"]:
                    fallback_schedule = updates.get("schedule_display") or updates.get("schedule")
                    raise _stale_oneshot_error(updated["schedule"], fallback_schedule)

            if inference_fields_changed:
                provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
                    provider=updated.get("provider"),
                    model=updated.get("model"),
                    base_url=updated.get("base_url"),
                    no_agent=updated.get("no_agent"),
                )
                updated["provider_snapshot"] = provider_snapshot
                updated["model_snapshot"] = model_snapshot

            if updated.get("enabled", True) and updated.get("state") != "paused" and not updated.get("next_run_at"):
                updated["next_run_at"] = compute_next_run(
                    updated["schedule"], tz_name=updated.get("timezone")
                )
                if updated["schedule"].get("kind") == "once" and not updated["next_run_at"]:
                    fallback_schedule = updates.get("schedule_display") or updates.get("schedule")
                    raise _stale_oneshot_error(updated["schedule"], fallback_schedule)

            trigger_identity_changed = any(
                key in updates and updates.get(key) != job.get(key)
                for key in ("schedule", "timezone", "next_run_at", "enabled", "state")
            )
            if trigger_identity_changed:
                # A claim identifies the exact schedule occurrence that was
                # active when it was created. User pause/resume/reschedule must
                # invalidate that identity before a delayed provider retry can
                # recover it.
                updated["fire_claim"] = None
                updated["in_flight_occurrence"] = None

            jobs[i] = updated
            save_jobs(jobs)
            return _normalize_job_record(jobs[i])
    return None


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pause a job without deleting it. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": False,
            "state": "paused",
            "paused_at": _hermes_now().isoformat(),
            "paused_reason": reason,
        },
    )


def resume_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Resume a paused job and compute the next future run from now. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None

    next_run_at = compute_next_run(job["schedule"], tz_name=job.get("timezone"))
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": next_run_at,
        },
    )


def trigger_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Schedule a job to run on the next scheduler tick. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": _hermes_now().isoformat(),
        },
    )


def remove_job(job_id: str) -> bool:
    """Remove a job by ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with _jobs_lock():
        jobs = load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) < original_len:
            # Resolve the output dir BEFORE saving so a legacy unsafe ID (e.g.
            # left over from before the create-time guard) fails closed without
            # half-applying the removal.
            job_output_dir = _job_output_dir(canonical_id)
            save_jobs(jobs)
            # Clean up output directory to prevent orphaned dirs accumulating
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
            return True
    return False


def mark_job_run(job_id: str, success: bool, error: Optional[str] = None,
                 delivery_error: Optional[str] = None,
                 scheduled_at: Optional[str] = None,
                 output_filename: Optional[str] = None):
    """
    Mark a job as having been run.
    
    Updates last_run_at, last_status, increments completed count,
    computes next_run_at, and preserves a terminal record when the repeat
    limit is reached so calendar/history clients can still render the run.

    ``delivery_error`` is tracked separately from the agent error — a job
    can succeed (agent produced output) but fail delivery (platform down).
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                now_dt = _hermes_now()
                now = now_dt.isoformat()
                job["last_run_at"] = now
                job["last_status"] = "ok" if success else "error"
                job["last_error"] = error if not success else None
                # Track delivery failures separately — cleared on successful delivery
                job["last_delivery_error"] = delivery_error
                fire_claim = job.get("fire_claim")
                if isinstance(fire_claim, dict):
                    completed_fire_at = _parse_occurrence_instant(fire_claim.get("fire_at"))
                    if completed_fire_at is not None:
                        previous = _parse_occurrence_instant(job.get("last_completed_external_fire_at"))
                        if previous is None or completed_fire_at > previous:
                            # Chronos arms one one-shot at a time per job, so
                            # fire_at is a monotonic watermark. This rejects an
                            # arbitrarily delayed A even after B..N complete.
                            job["last_completed_external_fire_at"] = completed_fire_at.isoformat()
                # Clear any external-fire claim so a re-armed recurring job can
                # be claimed again on its next fire (Phase 4C CAS).
                job["fire_claim"] = None
                scheduled_instant = _parse_occurrence_instant(scheduled_at)
                _append_job_occurrence(
                    job,
                    now_dt,
                    scheduled_at=scheduled_instant,
                    output_filename=output_filename,
                    success=success,
                    delivery_error=delivery_error,
                )
                # Terminal history now owns this stable occurrence identity.
                # Clear the bounded execution claim in the same locked write
                # so API readers never observe both forms after completion.
                job["in_flight_occurrence"] = None
                
                # Increment completed count.  Finite one-shot jobs are
                # pre-claimed by claim_dispatch() BEFORE the side effect runs
                # (issue #38758), which already incremented completed — do not
                # double-count them here.  Recurring jobs and direct callers
                # with no pre-run claim still get the legacy increment.
                if job.get("repeat"):
                    repeat = job["repeat"]
                    times = repeat.get("times")
                    completed = repeat.get("completed", 0)
                    kind = job.get("schedule", {}).get("kind")
                    preclaimed_oneshot = (
                        kind == "once"
                        and times is not None
                        and times > 0
                        and completed > 0
                    )
                    if not preclaimed_oneshot:
                        completed += 1
                        repeat["completed"] = completed

                    # Check if we've hit the repeat limit
                    if times is not None and times > 0 and completed >= times:
                        # Keep the exhausted job as an inert terminal record.
                        # The execution output is stored separately under the
                        # same stable job id; retaining this row lets read-only
                        # calendar clients join the real run back to its task
                        # metadata without weakening at-most-times delivery.
                        job["enabled"] = False
                        job["state"] = "completed"
                        job["next_run_at"] = None
                        save_jobs(jobs)
                        return
                
                # Compute next run
                job["next_run_at"] = compute_next_run(
                    job["schedule"], now, tz_name=job.get("timezone")
                )

                # If no next run, decide whether this is terminal completion
                # (one-shot) or a transient failure (recurring schedule couldn't
                # compute — e.g. 'croniter' missing from the runtime env).
                # Recurring jobs must NEVER be silently disabled: that turns a
                # missing runtime dep into "job completed" and the user's
                # schedule quietly goes off. See issue #16265.
                if job["next_run_at"] is None:
                    kind = job.get("schedule", {}).get("kind")
                    if kind in {"cron", "interval"}:
                        job["state"] = "error"
                        if not job.get("last_error"):
                            job["last_error"] = (
                                "Failed to compute next run for recurring "
                                "schedule (is the 'croniter' package "
                                "installed in the gateway's Python env?)"
                            )
                        logger.error(
                            "Job '%s' (%s) could not compute next_run_at; "
                            "leaving enabled and marking state=error so the "
                            "job is not silently disabled.",
                            job.get("name", job["id"]),
                            kind,
                        )
                    else:
                        job["enabled"] = False
                        job["state"] = "completed"
                elif job.get("state") != "paused":
                    job["state"] = "scheduled"

                save_jobs(jobs)
                return

        logger.warning("mark_job_run: job_id %s not found, skipping save", job_id)


def claim_dispatch(job_id: str) -> bool:
    """Atomically claim a finite one-shot job dispatch BEFORE execution.

    Increments ``repeat.completed`` under the cross-process jobs lock and
    persists the claim immediately, so that if the tick dies mid-execution
    (gateway kill, OOM, segfault, hard-timeout) the dispatch is not lost.
    This converts finite one-shot jobs from *at-least-once* to *at-most-times*
    semantics — a job fires at most ``repeat.times`` times instead of
    infinitely (issue #38758).

    Returns ``True`` if the caller may proceed to run the job, ``False`` if the
    dispatch limit is already reached (in which case the stale job is made
    inert and retained for diagnosis/history).

    Only claims jobs with ``schedule.kind == "once"`` and ``repeat.times > 0``.
    Recurring jobs (they use ``advance_next_run``) and infinite-repeat / no-repeat
    jobs are left unchanged and always allowed to proceed.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
                return True  # recurring jobs use advance_next_run(), not dispatch claims
            repeat = job.get("repeat")
            if not repeat:
                return True  # no repeat limit — always dispatch
            times = repeat.get("times")
            if times is None or times <= 0:
                return True  # infinite — always dispatch
            completed = repeat.get("completed", 0)
            if completed >= times:
                # Already dispatched the max number of times (e.g. a prior
                # tick claimed then died before mark_job_run could terminalize
                # it). Keep the record but make it impossible to re-fire.
                job["enabled"] = False
                job["state"] = "error"
                job["next_run_at"] = None
                job["last_status"] = "error"
                job["last_error"] = job.get("last_error") or (
                    "Dispatch was claimed but no completed run was recorded"
                )
                save_jobs(jobs)
                logger.info(
                    "Job '%s': dispatch limit reached (%d/%d) — retaining inert terminal record",
                    job.get("name", job["id"]),
                    completed,
                    times,
                )
                return False
            # Claim this dispatch before the side effect runs.
            repeat["completed"] = completed + 1
            save_jobs(jobs)
            logger.debug(
                "Job '%s': claimed dispatch %d/%d",
                job.get("name", job["id"]),
                repeat["completed"],
                times,
            )
            return True

        logger.debug(
            "claim_dispatch: job_id %s not in store — proceeding without claim "
            "(handed-in job dict; nothing to persist a claim against)",
            job_id,
        )
        return True


def advance_next_run(job_id: str) -> bool:
    """Preemptively advance next_run_at for a recurring job before execution.

    Call this BEFORE run_job() so that if the process crashes mid-execution,
    the job won't re-fire on the next gateway restart.  This converts the
    scheduler from at-least-once to at-most-once for recurring jobs — missing
    one run is far better than firing dozens of times in a crash loop.

    One-shot jobs are left unchanged so they can still retry on restart.

    Returns True if next_run_at was advanced, False otherwise.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] == job_id:
                kind = job.get("schedule", {}).get("kind")
                if kind not in {"cron", "interval"}:
                    return False
                now = _hermes_now().isoformat()
                new_next = compute_next_run(
                    job["schedule"], now, tz_name=job.get("timezone")
                )
                if new_next and new_next != job.get("next_run_at"):
                    job["next_run_at"] = new_next
                    save_jobs(jobs)
                    return True
                return False
        return False


def _machine_id() -> str:
    """Stable-ish identifier for claim attribution/debugging (NOT correctness).

    Uses ``HERMES_MACHINE_ID`` if set, else hostname + pid. The CAS correctness
    comes from the file lock + the fresh-claim check, not from this value.
    """
    explicit = os.getenv("HERMES_MACHINE_ID", "").strip()
    if explicit:
        return explicit
    try:
        import socket
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def claim_job_for_fire(
    job_id: str,
    *,
    claim_ttl_seconds: int = 300,
    triggered_at: Optional[str] = None,
    fire_at: Optional[str] = None,
) -> bool:
    """Atomically claim a job for a single external 'fire' (multi-machine
    at-most-once). Returns True iff THIS caller won the claim.

    Used by the external-provider fire path (``CronScheduler.fire_due``) when an
    external scheduler (Chronos) signals a job is due across N gateway replicas:
    exactly one wins. Single-machine deployments always win.

    Under the file lock: reject if the job is missing/disabled/paused. If a
    fresh claim (younger than ``claim_ttl_seconds``) already exists, lose.
    Otherwise stamp a ``fire_claim`` and, for recurring jobs, advance
    ``next_run_at`` (mirrors ``advance_next_run``'s at-most-once bump so a stale
    re-delivery for the old time can't re-fire). One-shots keep ``next_run_at``
    but the fresh ``fire_claim`` blocks a duplicate retry for the same fire.
    ``mark_job_run`` clears the claim on completion so a re-armed recurring job
    is claimable again next fire.

    The stale-claim TTL means a machine that crashed after claiming but before
    completing doesn't wedge the job forever — after the TTL another fire can
    reclaim it.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            if not job.get("enabled", True) or job.get("state") == "paused":
                return False
            now = _hermes_now()
            recovering_external_claim = False
            canonical_fire_at = normalize_external_fire_at(fire_at, now=now)
            if canonical_fire_at is not None:
                requested_fire = _parse_occurrence_instant(canonical_fire_at)
                completed_watermark = _parse_occurrence_instant(job.get("last_completed_external_fire_at"))
                if completed_watermark is not None and requested_fire <= completed_watermark:
                    return False
                planned_fire = _parse_occurrence_instant(job.get("next_run_at"))
                in_flight = _in_flight_occurrence(job, now)
                in_flight_instants = set()
                if in_flight is not None:
                    in_flight_instants.add(in_flight["scheduled_at"])
                    if in_flight.get("original_scheduled_at") is not None:
                        in_flight_instants.add(in_flight["original_scheduled_at"])
                existing = job.get("fire_claim")
                recoverable_claim_fire = None
                if isinstance(existing, dict):
                    existing_fire = _parse_occurrence_instant(existing.get("fire_at"))
                    existing_at = _parse_occurrence_instant(existing.get("at"))
                    if (
                        existing_fire == requested_fire
                        and existing_at is not None
                        and (now - existing_at).total_seconds() >= claim_ttl_seconds
                    ):
                        recoverable_claim_fire = existing_fire
                        recovering_external_claim = True
                if (
                    requested_fire != planned_fire
                    and requested_fire not in in_flight_instants
                    and requested_fire != recoverable_claim_fire
                ):
                    # A provider callback is scoped to the exact occurrence it
                    # was armed for.  An old arm that races a user reschedule
                    # must not execute the job at the abandoned time.
                    return False
            existing = job.get("fire_claim")
            if existing:
                try:
                    claimed_at = _ensure_aware(datetime.fromisoformat(existing["at"]))
                    if (now - claimed_at).total_seconds() < claim_ttl_seconds:
                        return False  # someone holds a fresh claim
                except Exception:
                    pass  # malformed claim → overwrite
            explicit_trigger: Optional[datetime] = None
            if triggered_at is not None and canonical_fire_at is not None:
                raise ValueError("triggered_at and fire_at are mutually exclusive")
            if canonical_fire_at is not None:
                explicit_trigger = _parse_occurrence_instant(canonical_fire_at)
            elif triggered_at is not None:
                if not isinstance(triggered_at, str) or len(triggered_at) > 128:
                    raise ValueError("triggered_at must be a bounded ISO timestamp")
                explicit_trigger = _parse_occurrence_instant(triggered_at)
                if explicit_trigger is None:
                    raise ValueError("triggered_at must be an aware ISO timestamp")

            # A stale external-fire retry is still the SAME occurrence. The
            # first claim already advanced next_run_at, so recomputing from it
            # would consume tomorrow's slot. Manual Run Now is explicit and
            # intentionally starts a new identity at its requested instant.
            prior_in_flight = _in_flight_occurrence(job, now) if explicit_trigger is None else None
            kind = job.get("schedule", {}).get("kind")
            should_advance = prior_in_flight is None and not recovering_external_claim
            if explicit_trigger is not None:
                effective_trigger = explicit_trigger
                original_trigger = None
            elif prior_in_flight is not None:
                effective_trigger = prior_in_flight["scheduled_at"]
                original_trigger = prior_in_flight.get("original_scheduled_at")
            else:
                planned = _parse_occurrence_instant(job.get("next_run_at"))
                effective_trigger = now
                original_trigger: Optional[datetime] = None
                if planned is not None:
                    lateness = (now - planned).total_seconds()
                    if lateness <= 0:
                        effective_trigger = planned
                    elif kind in {"cron", "interval"}:
                        if lateness <= _compute_grace_seconds(job.get("schedule", {})):
                            effective_trigger = planned
                        else:
                            original_trigger = planned
                    elif kind == "once":
                        if lateness <= ONESHOT_GRACE_SECONDS:
                            effective_trigger = planned
                        else:
                            original_trigger = planned
            job["fire_claim"] = {
                "at": now.isoformat(),
                "by": _machine_id(),
                "scheduled_at": effective_trigger.isoformat(),
            }
            if canonical_fire_at is not None:
                job["fire_claim"]["fire_at"] = canonical_fire_at
            _set_in_flight_occurrence(
                job,
                scheduled_at=effective_trigger,
                claimed_at=now,
                original_scheduled_at=original_trigger,
            )
            if should_advance and kind in {"cron", "interval"}:
                nxt = compute_next_run(
                    job["schedule"], now.isoformat(), tz_name=job.get("timezone")
                )
                if nxt:
                    job["next_run_at"] = nxt
            save_jobs(jobs)
            return True
        return False


def get_due_jobs() -> List[Dict[str, Any]]:
    """Get all jobs that are due to run now.

    For recurring jobs (cron/interval), if the scheduled time is stale (more
    than one period in the past, e.g. because the gateway was down OR because a
    long-running previous execution overran the interval), the accumulated
    missed runs are collapsed — ``next_run_at`` is fast-forwarded to the next
    future occurrence so a backlog does NOT burst-fire on restart — but the job
    still fires ONCE now. This prevents the perpetual-defer loop (#33315) where
    a job whose runtime exceeds ``interval + grace`` would be skipped forever.

    Note: firing once on catch-up flows through ``mark_job_run``, so a job with
    a ``repeat.times`` limit consumes one of its runs on that catch-up fire.
    """
    with _jobs_lock():
        return _get_due_jobs_locked()


def _get_due_jobs_locked() -> List[Dict[str, Any]]:
    """Inner implementation of get_due_jobs(); must be called with _jobs_lock held."""
    now = _hermes_now()
    raw_jobs = load_jobs()
    jobs = [_apply_skill_fields(j) for j in copy.deepcopy(raw_jobs)]
    due = []
    needs_save = False

    for job in jobs:
        if not job.get("enabled", True):
            continue

        next_run = job.get("next_run_at")
        if not next_run:
            schedule = job.get("schedule", {})
            kind = schedule.get("kind")

            # One-shot jobs use a small grace window via the dedicated helper.
            recovered_next = _recoverable_oneshot_run_at(
                schedule,
                now,
                last_run_at=job.get("last_run_at"),
            )
            recovery_kind = "one-shot" if recovered_next else None

            # Recurring jobs reach here only when something — typically a
            # direct jobs.json edit that bypassed add_job() — left
            # next_run_at unset.  Without this branch, such jobs are
            # silently skipped forever; recompute next_run_at from the
            # schedule so they pick up at their next scheduled tick.
            if not recovered_next and kind in {"cron", "interval"}:
                recovered_next = compute_next_run(
                    schedule, now.isoformat(), tz_name=job.get("timezone")
                )
                if recovered_next:
                    recovery_kind = kind

            if not recovered_next:
                continue

            job["next_run_at"] = recovered_next
            next_run = recovered_next
            logger.info(
                "Job '%s' had no next_run_at; recovering %s run at %s",
                job.get("name", job["id"]),
                recovery_kind,
                recovered_next,
            )
            for rj in raw_jobs:
                if rj["id"] == job["id"]:
                    rj["next_run_at"] = recovered_next
                    needs_save = True
                    break

        raw_next_run_dt = datetime.fromisoformat(next_run)
        schedule = job.get("schedule", {})
        kind = schedule.get("kind")

        next_run_dt = _ensure_aware(raw_next_run_dt)
        has_fixed_job_timezone = (
            _normalized_iana_timezone_name(job.get("timezone")) is not None
        )
        # Migration repair: a cron job persists next_run_at as an absolute
        # instant, but the cron expr describes local wall-clock intent. If the
        # configured/system timezone changed after persistence, the stored
        # instant's offset no longer matches now's, and its converted time can
        # look due hours early (21:00+10 -> 13:00+02). When the stored *wall
        # clock* is still in the future, recompute from the schedule so we fire
        # at the intended local time instead of early-then-again.
        #
        # This heuristic applies only to legacy/malformed jobs without a valid
        # pinned IANA timezone. A pinned timezone owns its offset (including
        # normal differences from the Hermes runtime), so its persisted aware
        # timestamp must be judged by absolute time instead.
        if (
            kind == "cron"
            and not has_fixed_job_timezone
            and next_run_dt <= now
            and _timezone_offset_mismatch(raw_next_run_dt, now)
            and _stored_wall_clock_is_future(raw_next_run_dt, now)
        ):
            new_next = compute_next_run(
                schedule, now.isoformat(), tz_name=job.get("timezone")
            )
            if new_next:
                logger.info(
                    "Job '%s' next_run_at offset changed (%s -> %s). "
                    "Recomputing cron run to preserve local wall-clock intent: %s",
                    job.get("name", job["id"]),
                    raw_next_run_dt.utcoffset(),
                    now.utcoffset(),
                    new_next,
                )
                for rj in raw_jobs:
                    if rj["id"] == job["id"]:
                        rj["next_run_at"] = new_next
                        needs_save = True
                        break
                continue

        if next_run_dt <= now:

            # For recurring jobs, a next_run_at far in the past means the gateway
            # was down (or not resident) across the scheduled time. We still run
            # it ONCE to catch up — tick() calls advance_next_run() before
            # execution, which recomputes next_run_at from *now*, so this stays
            # at-most-once with no backlog burst no matter how many periods were
            # missed. Previously a >grace miss was silently fast-forwarded and
            # dropped: e.g. a daily 09:00 job missed by >2h never ran that day and
            # left no failure record (ZET-1413; same root cause as ZET-1334).
            grace = _compute_grace_seconds(schedule)
            missed_by_seconds = (now - next_run_dt).total_seconds()
            stale_recurring_catchup = kind in {"cron", "interval"} and missed_by_seconds > grace
            late_oneshot = kind == "once" and missed_by_seconds > 0
            effective_trigger = now if stale_recurring_catchup or late_oneshot else next_run_dt
            original_trigger = next_run_dt if effective_trigger != next_run_dt else None
            # Keep the dispatch context on the returned object for the runner,
            # and persist the same bounded claim for occurrence API readers.
            # Without the persisted form, advance_next_run moves the store to
            # tomorrow and today's node disappears while the task is running.
            job["_occurrence_triggered_at"] = effective_trigger.isoformat()
            for rj in raw_jobs:
                if rj["id"] == job["id"]:
                    _set_in_flight_occurrence(
                        rj,
                        scheduled_at=effective_trigger,
                        claimed_at=now,
                        original_scheduled_at=original_trigger,
                    )
                    needs_save = True
                    break
            if stale_recurring_catchup:
                # Job is past its catch-up grace window — skip accumulated
                # missed runs but still execute once now to avoid deferring
                # indefinitely (e.g. a long-running job just finished).
                new_next = compute_next_run(
                    schedule, now.isoformat(), tz_name=job.get("timezone")
                )
                if new_next:
                    logger.info(
                        "Job '%s' missed its scheduled time (%s, grace=%ds). "
                        "Running now; next run provisionally set to: %s "
                        "(re-anchored on completion)",
                        job.get("name", job["id"]),
                        next_run,
                        grace,
                        new_next,
                    )
                    # Persist the fast-forward to storage now (skip accumulated
                    # slots). In the built-in ticker path this is shortly
                    # overwritten by advance_next_run + mark_job_run, but it is
                    # NOT redundant: it (a) protects the crash window between
                    # here and mark_job_run, and (b) covers the external
                    # fire_due provider path, which does not call
                    # advance_next_run. mark_job_run re-anchors next_run_at off
                    # the actual completion time, so this value is provisional.
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["next_run_at"] = new_next
                            needs_save = True
                            break
                    # Fall through to due.append(job) — execute once now

            # One-shot dispatch-limit guard (issue #38758): a finite one-shot
            # claimed via claim_dispatch() but whose tick died before
            # mark_job_run could terminalize it will have completed >= times while
            # still looking due (last_run_at was never written, so the
            # recovery helper re-armed it). Remove it instead of re-firing.
            if kind == "once":
                repeat = job.get("repeat")
                if repeat:
                    times = repeat.get("times")
                    completed = repeat.get("completed", 0)
                    if times is not None and times > 0 and completed >= times:
                        logger.info(
                            "Job '%s': one-shot dispatch limit reached (%d/%d) "
                            "— retaining inert stale entry",
                            job.get("name", job["id"]),
                            completed,
                            times,
                        )
                        for rj in raw_jobs:
                            if rj["id"] == job["id"]:
                                rj["enabled"] = False
                                rj["state"] = "error"
                                rj["next_run_at"] = None
                                rj["last_status"] = "error"
                                rj["last_error"] = rj.get("last_error") or (
                                    "Dispatch was claimed but no completed run was recorded"
                                )
                                needs_save = True
                                break
                        continue

            due.append(job)

    if needs_save:
        save_jobs(raw_jobs)

    return due


# Per-run cron output (`cron/output/<job>/<timestamp>.md`) is written once per
# execution. Unlike the quick-snapshot store (`hermes_cli.backup`, capped at 20)
# it had no retention, so a frequently-scheduled job on a long-running deploy
# accumulated one file per run forever and could fill the disk (#52383). Keep the
# most recent N files per job; a non-positive value disables pruning (opt-out).
_CRON_OUTPUT_DEFAULT_KEEP = 50


def _cron_output_keep() -> int:
    """Resolve the per-job output-file retention cap from config (``cron.output_retention``)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return int(cron_cfg.get("output_retention", _CRON_OUTPUT_DEFAULT_KEEP))
    except Exception:
        return _CRON_OUTPUT_DEFAULT_KEEP


def _prune_job_output(job_output_dir: Path, keep: int) -> int:
    """Remove the oldest ``*.md`` run-output files beyond *keep*. Returns count deleted.

    Mirrors the quick-snapshot retention in ``hermes_cli.backup._prune_quick_snapshots``:
    output filenames are timestamp-based (``%Y-%m-%d_%H-%M-%S.md``) so a reverse
    lexical sort orders newest-first, and everything past *keep* is the tail to
    drop. A non-positive *keep* disables pruning. Pruning failures are swallowed
    so they can never break output saving.
    """
    if keep <= 0:
        return 0
    try:
        files = sorted(
            (f for f in job_output_dir.glob("*.md") if f.is_file()),
            key=lambda f: f.name,
            reverse=True,
        )
    except OSError:
        return 0
    deleted = 0
    for stale in files[keep:]:
        try:
            stale.unlink()
            deleted += 1
        except OSError as exc:
            logger.debug("Failed to prune cron output %s: %s", stale.name, exc)
    return deleted


def save_job_output(job_id: str, output: str):
    """Save job output to file."""
    ensure_dirs()
    job_output_dir = _job_output_dir(job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(job_output_dir)

    timestamp = _hermes_now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = job_output_dir / f"{timestamp}.md"

    fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix='.tmp', prefix='.output_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, output_file)
        _secure_file(output_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Bound per-job output growth so long-running deploys don't fill the disk (#52383).
    _prune_job_output(job_output_dir, _cron_output_keep())

    return output_file


# =============================================================================
# Skill reference rewriting (curator integration)
# =============================================================================

def referenced_skill_names() -> Set[str]:
    """Return the set of skill names referenced by ANY cron job.

    Includes paused and disabled jobs deliberately: a paused job never
    fires, so its skills never get a ``bump_use`` from the scheduler, yet
    resuming it must still find its skills present. The curator uses this
    set to protect referenced skills from inactivity archival — a skill a
    live job depends on is "in use" regardless of when it was last loaded.

    Best-effort: a corrupt/unreadable jobs store returns an empty set
    rather than raising, so a cron issue can never break the curator.
    """
    try:
        jobs = load_jobs()
    except Exception:
        logger.debug("referenced_skill_names: failed to load cron jobs", exc_info=True)
        return set()

    names: Set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for name in _normalize_skill_list(job.get("skill"), job.get("skills")):
            cleaned = str(name).strip().lstrip("/")
            if cleaned:
                names.add(cleaned)
    return names


def rewrite_skill_refs(
    consolidated: Optional[Dict[str, str]] = None,
    pruned: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite cron job skill references after a curator consolidation pass.

    When the curator consolidates a skill X into umbrella Y (or archives X
    as pruned), any cron job that lists ``X`` in its ``skills`` field will
    fail to load ``X`` at run time — the scheduler logs a warning and
    skips the skill, so the job runs without the instructions it was
    scheduled to follow. See cron/scheduler.py where ``skill_view`` is
    called per skill name.

    This function repairs cron jobs in-place:

    - A skill listed in ``consolidated`` is replaced with its umbrella
      target (the ``into`` value). If the umbrella is already in the
      job's skill list, the stale name is dropped without duplication.
    - A skill listed in ``pruned`` is dropped outright — there is no
      forwarding target.
    - Ordering and other skills in the list are preserved.
    - The legacy ``skill`` field is realigned via ``_apply_skill_fields``.

    Args:
        consolidated: mapping of ``old_skill_name -> umbrella_skill_name``.
        pruned: list of skill names that were archived with no forwarding
            target.

    Returns a report dict::

        {
            "rewrites": [
                {
                    "job_id": ...,
                    "job_name": ...,
                    "before": [...],
                    "after": [...],
                    "mapped": {"old": "new", ...},
                    "dropped": ["old", ...],
                },
                ...
            ],
            "jobs_updated": N,
            "jobs_scanned": M,
        }

    Best-effort: exceptions from loading/saving propagate to the caller so
    tests can assert behaviour; the curator invocation site wraps this
    call in a try/except so a failure here never breaks the curator.
    """
    consolidated = dict(consolidated or {})
    pruned_set = set(pruned or [])
    # A skill listed in both wins as "consolidated" — it has a target,
    # which is the more useful of the two outcomes.
    pruned_set -= set(consolidated.keys())

    if not consolidated and not pruned_set:
        return {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    with _jobs_lock():
        jobs = load_jobs()
        rewrites: List[Dict[str, Any]] = []
        changed = False

        for job in jobs:
            skills_before = _normalize_skill_list(job.get("skill"), job.get("skills"))
            if not skills_before:
                continue

            mapped: Dict[str, str] = {}
            dropped: List[str] = []
            new_skills: List[str] = []

            for name in skills_before:
                if name in consolidated:
                    target = consolidated[name]
                    mapped[name] = target
                    if target and target not in new_skills:
                        new_skills.append(target)
                elif name in pruned_set:
                    dropped.append(name)
                elif name not in new_skills:
                    new_skills.append(name)

            if not mapped and not dropped:
                continue

            job["skills"] = new_skills
            job["skill"] = new_skills[0] if new_skills else None
            changed = True

            rewrites.append({
                "job_id": job.get("id"),
                "job_name": job.get("name") or job.get("id"),
                "before": list(skills_before),
                "after": list(new_skills),
                "mapped": mapped,
                "dropped": dropped,
            })

        if changed:
            save_jobs(jobs)
            logger.info(
                "Curator rewrote skill references in %d cron job(s)", len(rewrites)
            )

        return {
            "rewrites": rewrites,
            "jobs_updated": len(rewrites),
            "jobs_scanned": len(jobs),
        }
