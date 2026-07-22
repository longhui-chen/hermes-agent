"""Recoverable execution for planner-managed calendar event alerts."""

from __future__ import annotations

import datetime as dt
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import stat
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_TERMINAL = frozenset({"fired", "expired", "cancelled", "superseded"})
_SOURCE_LABEL = {
    "device_calendar": "系统日历",
    "google_calendar": "Google",
    "outlook_calendar": "Outlook",
    "lark_calendar": "飞书",
}


def _scoped_env(name: str, default: str = "") -> str:
    try:
        from cron.scheduler import _cron_env
        return _cron_env(name, default)
    except Exception:
        try:
            from agent.secret_scope import is_multiplex_active
            if is_multiplex_active():
                return default
        except Exception:
            pass
        return os.environ.get(name, default)


def _parse_rfc3339(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def is_managed_calendar_event_alert(job: Any) -> bool:
    """Exact three-end wire predicate; near-misses must never deliver text."""
    if not isinstance(job, dict):
        return False
    schedule = job.get("schedule")
    revision = job.get("calendar_projection_revision")
    return bool(
        job.get("source") == "calendar"
        and job.get("calendar_job_kind") == "event_alert"
        and job.get("no_agent") is True
        and isinstance(schedule, dict)
        and schedule.get("kind") == "once"
        and isinstance(job.get("calendar_delivery_key"), str)
        and _HEX64.fullmatch(job["calendar_delivery_key"])
        and isinstance(revision, int)
        and not isinstance(revision, bool)
        and revision > 0
        and _parse_rfc3339(job.get("calendar_materialized_at")) is not None
        and _parse_rfc3339(job.get("calendar_notification_deadline_at")) is not None
        and job.get("calendar_message_type") == "calendar_notification"
        and job.get("calendar_llm_visible") is False
        and job.get("prompt") is None
    )


def is_invalid_calendar_job(job: Any) -> bool:
    return isinstance(job, dict) and job.get("source") == "calendar" and not is_managed_calendar_event_alert(job)


def quarantine_invalid_calendar_job(job: dict) -> None:
    job_id = str(job.get("id") or "")
    if not job_id:
        return
    try:
        from cron.jobs import update_job
        update_job(job_id, {
            "enabled": False,
            "state": "quarantined",
            "last_error": "invalid managed calendar event-alert contract",
        })
    except Exception:
        # A malformed externally-written job is already silent because the
        # caller returns without run_job/_deliver_result. Persistence of the
        # diagnostic is best-effort; delivery safety does not depend on it.
        return


def _planner_url(path: str) -> str:
    append_url = _scoped_env("ZET_CHAT_APPEND_URL").strip()
    parsed = urllib.parse.urlsplit(append_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("calendar delivery: local-server URL unavailable")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/internal/v1/planner/calendar-deliveries" + path, "", ""))


def _planner_post(path: str, payload: dict) -> dict:
    token = _scoped_env("ZETTLAB_AGENT_ACTION_TOKEN").strip()
    if not token or len(token) > 512:
        raise RuntimeError("calendar delivery: main action token unavailable")
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > 16 * 1024:
        raise RuntimeError("calendar delivery: request exceeded cap")
    request = urllib.request.Request(
        _planner_url(path), data=raw,
        headers={"Content-Type": "application/json", "X-Zettlab-Agent-Action-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body = response.read(64 * 1024)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"calendar delivery: planner unavailable: {exc}") from exc
    decoded = json.loads(body or b"{}")
    data = decoded.get("data") if isinstance(decoded, dict) else None
    if not isinstance(data, dict):
        raise RuntimeError("calendar delivery: malformed planner response")
    return data


def _planner_get(path: str) -> dict:
    token = _scoped_env("ZETTLAB_AGENT_ACTION_TOKEN").strip()
    if not token or len(token) > 512:
        raise RuntimeError("calendar delivery: main action token unavailable")
    request = urllib.request.Request(
        _planner_url(path), headers={"X-Zettlab-Agent-Action-Token": token}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            body = response.read(16 * 1024)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"calendar delivery: planner unavailable: {exc}") from exc
    decoded = json.loads(body or b"{}")
    data = decoded.get("data") if isinstance(decoded, dict) else None
    if not isinstance(data, dict):
        raise RuntimeError("calendar delivery: malformed planner response")
    return data


def _worker_id() -> str:
    return f"{socket.gethostname()[:32]}:{os.getpid()}:{threading.get_ident()}"[:80]


def _fence_payload(state: dict, job: dict, worker_id: str) -> dict:
    return {
        "delivery_generation": state.get("delivery_generation"),
        "fence_token": state.get("fence_token"),
        "projection_revision": job["calendar_projection_revision"],
        "worker_id": worker_id,
    }


def _clean_display_text(value: Any, limit: int) -> str:
    text = "".join(ch for ch in str(value or "") if ch >= " " or ch in "\t\n")
    return text.strip()[:limit]


def _require_delivery_generation(value: Any) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value >= 2**64
    ):
        raise RuntimeError("calendar delivery: invalid delivery generation")
    return value


def _stage_message(job: dict, state: dict) -> tuple[Any, str, int, str]:
    from hermes_state import SessionDB

    owner = _clean_display_text(job.get("calendar_owner_user_id"), 768)
    if not owner:
        raise RuntimeError("calendar delivery: missing owner identity")
    expected_session = "zettlab:oh_" + hashlib.sha256(owner.encode("utf-8")).hexdigest() + ":main:calendar-reminders"
    session_id = state.get("session_id")
    if not isinstance(session_id, str) or not hmac.compare_digest(session_id, expected_session):
        raise RuntimeError("calendar delivery: authoritative session binding mismatch")
    content = state.get("content")
    if not isinstance(content, str) or not content or len(content.encode("utf-8")) > 16 * 1024:
        raise RuntimeError("calendar delivery: invalid authoritative content")
    delivery_generation = _require_delivery_generation(state.get("delivery_generation"))
    db = SessionDB()
    try:
        existing_session = db.get_session(session_id)
        if existing_session is None:
            db.create_session(session_id, source="zet_agent", user_id=owner)
            db.set_session_title(session_id, "日程提醒")
        elif existing_session.get("user_id") != owner:
            raise RuntimeError("calendar delivery: existing session owner mismatch")
        message_id = db.stage_calendar_notification(
            session_id=session_id,
            content=content,
            delivery_key=job["calendar_delivery_key"],
            delivery_generation=delivery_generation,
        )
    except Exception:
        db.close()
        raise
    return db, session_id, message_id, content


def _private_key_file(db: Any) -> Path:
    return Path(db.db_path).resolve().parent / ".calendar-delivery-nonce.key"


def _load_or_create_nonce_key(db: Any) -> bytes:
    path = _private_key_file(db)
    path.parent.mkdir(parents=True, exist_ok=True)

    def load_existing() -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if path.is_symlink():
                raise RuntimeError("calendar delivery: unsafe nonce key") from exc
            raise
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise RuntimeError("calendar delivery: unsafe nonce key")
            chunks = []
            remaining = 33
            while remaining:
                chunk = os.read(fd, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(fd)
        if len(raw) != 32:
            raise RuntimeError("calendar delivery: invalid nonce key")
        return raw

    try:
        return load_existing()
    except FileNotFoundError:
        pass

    raw = secrets.token_bytes(32)
    tmp = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(8)}.tmp"
    )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(tmp, flags, 0o600)
    try:
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            written = 0
            while written < len(view):
                count = os.write(fd, view[written:])
                if count <= 0:
                    raise OSError("calendar delivery: nonce key short write")
                written += count
            if os.fstat(fd).st_size != len(raw):
                raise OSError("calendar delivery: incomplete nonce key write")
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp, path, follow_symlinks=False)
        except FileExistsError:
            return load_existing()
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return raw
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _delivery_nonce(db: Any, job: dict, state: dict) -> bytes:
    material = (
        f"{job['calendar_delivery_key']}\n{state['delivery_generation']}\n"
        f"{state['fence_token']}"
    ).encode("ascii")
    return hmac.new(_load_or_create_nonce_key(db), material, hashlib.sha256).digest()


def _pin_file(db: Any) -> Path:
    return Path(db.db_path).resolve().parent / "calendar-finalize-public-keys.json"


def _load_pins(db: Any) -> dict[str, str]:
    path = _pin_file(db)
    try:
        info = path.lstat()
        if path.is_symlink() or not path.is_file() or info.st_mode & 0o077 or info.st_size > 16 * 1024:
            raise RuntimeError("calendar delivery: unsafe finalize key pins")
        value = json.loads(path.read_text("utf-8"))
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}


def _save_pins(db: Any, pins: dict[str, str]) -> None:
    path = _pin_file(db)
    raw = json.dumps(pins, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


def _resolve_finalize_key(db: Any, key_id: str) -> bytes:
    pins = _load_pins(db)
    encoded = pins.get(key_id)
    if encoded is None:
        response = _planner_get("/finalize-keys")
        keys = response.get("keys")
        if not isinstance(keys, list) or len(keys) > 8:
            raise RuntimeError("calendar delivery: malformed finalize key set")
        for item in keys:
            if not isinstance(item, dict) or item.get("algorithm") != "Ed25519":
                continue
            candidate = item.get("public_key")
            candidate_id = item.get("key_id")
            if not isinstance(candidate, str) or not isinstance(candidate_id, str):
                continue
            try:
                raw = base64.urlsafe_b64decode(candidate + "==")
            except Exception:
                continue
            expected_id = "cal-v1-" + hashlib.sha256(raw).hexdigest()[:16]
            if len(raw) == 32 and hmac.compare_digest(candidate_id, expected_id):
                pins.setdefault(candidate_id, candidate)
        _save_pins(db, pins)
        encoded = pins.get(key_id)
    if not isinstance(encoded, str):
        raise RuntimeError("calendar delivery: unknown finalize key")
    raw = base64.urlsafe_b64decode(encoded + "==")
    if len(raw) != 32:
        raise RuntimeError("calendar delivery: invalid pinned finalize key")
    return raw


def _verify_finalize_receipt(db: Any, job: dict, state: dict, session_id: str, message_id: int) -> None:
    receipt = state.get("receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("calendar delivery: missing finalize receipt")
    expected = {
        "delivery_key": job["calendar_delivery_key"],
        "delivery_generation": state.get("delivery_generation"),
        "fence_token": state.get("fence_token"),
        "message_id": str(message_id),
        "session_id": session_id,
    }
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError("calendar delivery: finalize receipt binding mismatch")
    key_id = receipt.get("key_version")
    queued_at = receipt.get("queued_at")
    signature = receipt.get("signature")
    if not all(isinstance(v, str) and v for v in (key_id, queued_at, signature)):
        raise RuntimeError("calendar delivery: malformed finalize receipt")
    canonical = (
        f"{receipt['delivery_key']}\n{receipt['delivery_generation']}\n"
        f"{receipt['fence_token']}\n{receipt['message_id']}\n{receipt['session_id']}\n"
        f"{queued_at}\n{key_id}"
    ).encode("utf-8")
    try:
        sig = base64.urlsafe_b64decode(signature + "==")
        Ed25519PublicKey.from_public_bytes(_resolve_finalize_key(db, key_id)).verify(sig, canonical)
    except Exception as exc:
        raise RuntimeError("calendar delivery: invalid finalize receipt") from exc


def _notify_visible_append(session_id: str, message_id: int, content: str) -> None:
    url = _scoped_env("ZET_CHAT_APPEND_URL").strip()
    if not url:
        return
    payload = {
        "agent_id": _scoped_env("ZET_AGENT_ID", "main").strip() or "main",
        "session_id": session_id,
        "msg_id": int(message_id),
        "role": "assistant",
        "content": content,
        "kind": "calendar_notification",
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > 16 * 1024:
        return
    try:
        request = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=2.0):
            pass
    except Exception:
        # SessionDB is authoritative for APP history; WS wakeup is best effort.
        return


def begin_external_calendar_fire(
    job: dict,
    *,
    provider_name: str,
    provider_contract_version: int,
    provider_fire_id: str = "",
) -> dict:
    """Persist an external fire attempt before the webhook is acknowledged."""
    if not is_managed_calendar_event_alert(job):
        raise RuntimeError("calendar delivery: invalid external fire contract")
    if job.get("enabled", True) is not True or job.get("state") in {
        "paused", "staged", "completed", "error", "quarantined",
    }:
        # An already-armed callback can race pause/delete convergence. Treat
        # it as terminally consumed without contacting Planner or staging a
        # user-visible notification.
        return {"state": "cancelled", "reason": "job_not_active"}
    worker_id = _worker_id()
    state = _planner_post("/begin-external-fire", {
        "delivery_key": job["calendar_delivery_key"],
        "job_id": job["id"],
        "projection_revision": job["calendar_projection_revision"],
        "materialized_at": job["calendar_materialized_at"],
        "worker_id": worker_id,
        "provider_name": _clean_display_text(provider_name, 32),
        "provider_contract_version": int(provider_contract_version),
        "provider_fire_id": _clean_display_text(provider_fire_id, 160),
    })
    state["_worker_id"] = worker_id
    return state


def _external_retry_at(result: dict | None) -> str:
    candidate = (result or {}).get("retry_after")
    parsed = _parse_rfc3339(candidate)
    if parsed is None:
        parsed = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=5)
    return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def run_external_calendar_delivery(job: dict, begin_state: dict) -> dict:
    """Execute a preflighted fire and durably close/re-arm its attempt."""
    attempt = begin_state.get("attempt_sequence")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
        raise RuntimeError("calendar delivery: missing external attempt sequence")
    try:
        generation = _require_delivery_generation(begin_state.get("delivery_generation"))
    except RuntimeError:
        raise RuntimeError("calendar delivery: missing external delivery generation")
    result: dict | None = None
    failure = ""
    try:
        result = run_calendar_delivery(job, initial_state=begin_state)
        return result
    except Exception as exc:
        failure = _clean_display_text(exc, 500)
        raise
    finally:
        terminal = bool(result and result.get("terminal"))
        payload = {
            "delivery_key": job["calendar_delivery_key"],
            "delivery_generation": generation,
            "attempt_sequence": attempt,
            "terminal": terminal,
            "retry_at": _external_retry_at(result),
            "last_error": failure or ("" if terminal else "calendar delivery remains non-terminal"),
        }
        try:
            _planner_post("/complete-external-fire", payload)
        except Exception:
            # The local-server execution-ack timeout is authoritative. If this
            # callback is lost, its durable worker creates the next recovery
            # arm; the Hermes process is not required to survive.
            pass


def run_calendar_delivery(job: dict, initial_state: dict | None = None) -> dict:
    """Resume the planner delivery saga without generic cron claims/marking."""
    if not is_managed_calendar_event_alert(job):
        quarantine_invalid_calendar_job(job)
        return {"terminal": True, "ledger_state": "quarantined"}
    worker_id = str((initial_state or {}).get("_worker_id") or _worker_id())
    state = dict(initial_state) if initial_state is not None else _planner_post("/claim", {
        "delivery_key": job["calendar_delivery_key"],
        "job_id": job["id"],
        "projection_revision": job["calendar_projection_revision"],
        "materialized_at": job["calendar_materialized_at"],
        "worker_id": worker_id,
    })
    if state.get("state") in _TERMINAL:
        return {"terminal": True, "ledger_state": state.get("state"), **state}
    if state.get("retry_after"):
        return {"terminal": False, "ledger_state": state.get("state"), **state}
    delivery_generation = _require_delivery_generation(state.get("delivery_generation"))
    fence = _fence_payload(state, job, worker_id)
    if state.get("state") == "claimed":
        state = _planner_post(f"/{job['calendar_delivery_key']}/commit", fence)
    if state.get("state") in _TERMINAL:
        return {"terminal": True, "ledger_state": state.get("state"), **state}
    if state.get("state") not in {"committed", "prepared", "queued"}:
        return {"terminal": False, "ledger_state": state.get("state"), **state}
    if _require_delivery_generation(state.get("delivery_generation")) != delivery_generation:
        raise RuntimeError("calendar delivery: delivery generation changed during commit")
    db, session_id, message_id, content = _stage_message(job, state)
    try:
        nonce = _delivery_nonce(db, job, state)
        prepare_payload = {
            **fence,
            "message_id": str(message_id),
            "nonce": base64.urlsafe_b64encode(nonce).rstrip(b"=").decode("ascii"),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        if state.get("state") == "committed":
            state = _planner_post(f"/{job['calendar_delivery_key']}/prepare", prepare_payload)
            if state.get("state") in _TERMINAL:
                return {"terminal": True, "ledger_state": state.get("state"), **state}
            if state.get("state") not in {"prepared", "queued"} or state.get("content") != content:
                raise RuntimeError("calendar delivery: prepare content/state mismatch")
            if _require_delivery_generation(state.get("delivery_generation")) != delivery_generation:
                raise RuntimeError("calendar delivery: delivery generation changed during prepare")
        state = _planner_post(f"/{job['calendar_delivery_key']}/finalize", {
            **fence, "nonce": prepare_payload["nonce"],
        })
        if _require_delivery_generation(state.get("delivery_generation")) != delivery_generation:
            raise RuntimeError("calendar delivery: delivery generation changed during finalize")
        _verify_finalize_receipt(db, job, state, session_id, message_id)
        if not db.activate_calendar_notification(
            job["calendar_delivery_key"], delivery_generation, message_id
        ):
            raise RuntimeError("calendar delivery: staged SessionDB activation failed")
    finally:
        db.close()
    state = _planner_post(f"/{job['calendar_delivery_key']}/ack-fired", fence)
    if state.get("state") == "fired":
        _notify_visible_append(session_id, message_id, content)
    return {"terminal": state.get("state") in _TERMINAL, "ledger_state": state.get("state"), **state}
