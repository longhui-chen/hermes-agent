"""
Zet Agent cron event hook — persists cron summaries to the originating
chat's hermes SessionDB so they show up in the APP's regular chat
history (PRD §6.3 "不丢失：所有 cron 推送都进对话流持久化").

Design choice: persist via SessionDB.append_message → APP's existing
hermes /history call returns it as a regular assistant message → no
new endpoints, no outbox, no extra APP fetch path needed. The cron
summary is encoded as a typed markdown message: a fenced ``cron-summary``
code block carrying JSON metadata, followed by the cron output prose
as the message body. APP's message-bubble detects the fence and
renders CronSummaryCard; webui/other clients see plain markdown.

Why a monkey patch instead of editing cron/scheduler.py
--------------------------------------------------------
hermes-agent is a fork that periodically syncs from upstream. Editing
cron/scheduler.py directly creates merge conflicts every release.

We patch ``cron.scheduler.{save_job_output, mark_job_run}`` at module
import time (triggered by zet_agent_cron.pth in the venv site-packages).

Auto-install: ``install()`` runs on module import. The .pth line forces
import at Python startup, so patches are in place before any cron job
fires.

Activation: opt-in via env vars CRON_WEBHOOK_URL / CRON_PERSIST_TO_SESSION
— if neither is set this module is a no-op (preserves vanilla hermes
behaviour for non-ZettClaw deployments).

Failure modes
-------------
- SessionDB unavailable / append_message raises: log warn, return; cron
  loop continues uninterrupted. md output already on disk so APP can
  recover via /cron/runs fanout.
- Cached content missing (race / restart): persist with empty body —
  CronSummaryCard still renders from the JSON metadata alone.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "__zet_agent_cron_patched__"

# job_id → most recent saved markdown content. Populated by the
# save_job_output wrapper, drained by the mark_job_run wrapper.
_LATEST_OUTPUT: Dict[str, str] = {}


def _dbg(msg: str) -> None:
    """Best-effort debug log to /tmp/zet_agent_cron.log — useful during
    development since hermes child stdout/stderr land in zerolog inside
    local-server console. Production deployments can ignore this file."""
    try:
        import datetime as _dt
        with open("/tmp/zet_agent_cron.log", "a") as _f:
            _f.write(f"{_dt.datetime.now().isoformat()} pid={os.getpid()} {msg}\n")
    except Exception:
        pass


def install() -> None:
    """Patch cron.scheduler.{save_job_output, mark_job_run} +
    APIServerAdapter._create_agent. Idempotent."""
    _dbg("install() entered")
    try:
        import cron.scheduler as _sched
    except ImportError as _ie:
        _dbg(f"install() ABORT: cron.scheduler ImportError: {_ie}")
        return

    if getattr(_sched.mark_job_run, _PATCH_SENTINEL, False):
        _dbg("install() already patched, skip")
        return

    _orig_save = _sched.save_job_output
    _orig_mark = _sched.mark_job_run

    def _wrapped_save(job_id: str, output: str):
        _LATEST_OUTPUT[job_id] = output
        return _orig_save(job_id, output)

    def _wrapped_mark(
        job_id: str,
        success: bool,
        error: Optional[str] = None,
        delivery_error: Optional[str] = None,
    ):
        _dbg(f"_wrapped_mark CALLED job={job_id} success={success}")
        # Snapshot job BEFORE delegating — _orig_mark auto-deletes once+
        # repeat=1 jobs and any repeat=N where completed reaches N.
        try:
            from cron.jobs import get_job
            job_snapshot = get_job(job_id)
        except Exception:
            job_snapshot = None

        will_be_auto_deleted = _will_hit_repeat_limit(job_snapshot)

        result = _orig_mark(job_id, success, error, delivery_error=delivery_error)

        # PRD UX：once+repeat=N 跑满后 hermes 默认把 job 从 jobs.json pop
        # 掉 → APP 列表空。我们把它"复活"成 enabled=False/state=completed
        # 状态保留住，APP 列表能继续看到+查看运行历史。Id / created_at /
        # origin 等元数据来自 snapshot，保证用户在列表里看到的是同一条任
        # 务记录。
        if will_be_auto_deleted and job_snapshot is not None:
            try:
                _restore_as_completed(job_snapshot, success, error, delivery_error)
            except Exception as e:
                _dbg(f"_wrapped_mark restore FAILED: {e!r}")
                logger.warning(
                    "zet_agent_cron: restore-as-completed for %s failed: %s",
                    job_id, e,
                )

        try:
            _try_persist_to_session(job_id, success, error, delivery_error, job_snapshot)
        except Exception as e:
            _dbg(f"_wrapped_mark persist FAILED: {e!r}")
            logger.warning(
                "zet_agent_cron: persist for job %s failed (non-fatal): %s",
                job_id, e,
            )
        finally:
            _LATEST_OUTPUT.pop(job_id, None)

        return result

    setattr(_wrapped_mark, _PATCH_SENTINEL, True)
    setattr(_wrapped_save, _PATCH_SENTINEL, True)

    _sched.save_job_output = _wrapped_save
    _sched.mark_job_run = _wrapped_mark
    _dbg("install() patched mark_job_run + save_job_output OK")

    # ── _create_agent patch — set HERMES_SESSION_* contextvars ────────
    #
    # ZetAgentAdapter._create_agent already calls set_session_vars locally,
    # but contextvars are task-local: if the agent's tool calls run in a
    # task spawned BEFORE _create_agent ran, they won't see the values.
    # Patching at the parent class level (APIServerAdapter._create_agent)
    # gives us a second safety net + diagnostic log so we can prove the
    # values are set right before super() builds the agent.
    try:
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.session_context import set_session_vars

        if not getattr(APIServerAdapter._create_agent, _PATCH_SENTINEL, False):
            _orig_create = APIServerAdapter._create_agent

            def _wrapped_create(self, *args, **kwargs):
                session_id = kwargs.get("session_id")
                if session_id:
                    try:
                        set_session_vars(
                            platform="zettlab",
                            chat_id=session_id,
                            chat_name="",
                            thread_id="",
                            user_id="",
                            user_name="",
                            session_key=session_id,
                        )
                        _dbg(f"_create_agent: set_session_vars chat_id={session_id}")
                    except Exception as _e:
                        _dbg(f"_create_agent: set_session_vars FAILED: {_e!r}")
                else:
                    _dbg("_create_agent: no session_id, skip set_session_vars")
                return _orig_create(self, *args, **kwargs)

            setattr(_wrapped_create, _PATCH_SENTINEL, True)
            APIServerAdapter._create_agent = _wrapped_create
            _dbg("install() patched APIServerAdapter._create_agent OK")
    except ImportError as _ie:
        _dbg(f"install() _create_agent patch SKIP (ImportError): {_ie}")

    logger.info("zet_agent_cron: installed cron persistence hooks on cron.scheduler")


# ── Persist to hermes SessionDB ─────────────────────────────────────

def _try_persist_to_session(
    job_id: str,
    success: bool,
    error: Optional[str],
    delivery_error: Optional[str],
    job_snapshot: Optional[dict],
) -> None:
    """Append cron summary as an assistant message to the originating chat's
    hermes SessionDB. APP's regular hermes /history call surfaces it.

    Skips when:
      - job has no origin (deliver=local or job created via REST without
        origin) — cron output md is on disk, user can review there
      - job_snapshot missing AND get_job returns None (job already gone +
        no pre-mark snapshot — extremely rare race)
    """
    job = job_snapshot
    if job is None:
        try:
            from cron.jobs import get_job
            job = get_job(job_id)
        except Exception:
            job = None
    if not job:
        _dbg(f"_try_persist: job {job_id} unavailable, skip")
        return

    origin = job.get("origin") or {}
    origin_chat_id = origin.get("chat_id", "").strip()
    if not origin_chat_id:
        _dbg(f"_try_persist: job {job_id} no origin.chat_id, skip (deliver={job.get('deliver')!r})")
        return

    content = _build_typed_message_content(job, job_id, success, error, delivery_error)
    _dbg(f"_try_persist: appending to session={origin_chat_id} job={job_id} content_len={len(content)}")

    try:
        from hermes_state import SessionDB
    except ImportError as _ie:
        _dbg(f"_try_persist: SessionDB ImportError: {_ie}")
        return

    db = SessionDB()
    try:
        msg_id = db.append_message(
            session_id=origin_chat_id,
            role="assistant",
            content=content,
        )
        _dbg(f"_try_persist: appended msg_id={msg_id} session={origin_chat_id}")
    finally:
        try:
            db.close()
        except Exception:
            pass

    # PRD §6.3 在线实时显示：写完 SessionDB 立刻 POST 给 local-server，让
    # 当前在线的 chat WS 立即收到 message.appended 事件并插条。失败仅 log
    # —— SessionDB 已落盘，APP 下次进 chat 走 /history 兜底。
    _try_notify_chat_append(origin_chat_id, msg_id, content)


def _try_notify_chat_append(session_id: str, msg_id: int, content: str) -> None:
    url = os.environ.get("ZET_CHAT_APPEND_URL", "").strip()
    if not url:
        _dbg("_try_notify_chat_append: ZET_CHAT_APPEND_URL unset, skip")
        return
    agent_id = os.environ.get("ZET_AGENT_ID", "").strip()
    payload = {
        "agent_id": agent_id,
        "session_id": session_id,
        "msg_id": int(msg_id) if msg_id is not None else 0,
        "role": "assistant",
        "content": content,
    }
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read(256).decode("utf-8", errors="replace")
            _dbg(f"_try_notify_chat_append: POST status={resp.status} body={body!r}")
    except Exception as e:
        _dbg(f"_try_notify_chat_append: POST FAILED: {e!r}")


def _build_typed_message_content(
    job: dict,
    job_id: str,
    success: bool,
    error: Optional[str],
    delivery_error: Optional[str],
) -> str:
    """Build markdown content with a typed code fence.

    Format:

        ```cron-summary
        {"job_id":"...","name":"...","schedule":"...","prompt":"...",
         "deliver":{...},"repeat":{...},"last_run_result":"success",
         "scheduled_at":"...","delivery_error":""}
        ```

        <body — agent's final response or error summary>

    APP message-bubble detects the leading ``cron-summary`` fence,
    parses the JSON to render CronSummaryCard, then renders the body
    as regular markdown below the card.

    Other markdown clients (webui, plaintext readers) see the JSON
    fence as a regular code block + body — degrades gracefully.
    """
    schedule_display = ""
    sched = job.get("schedule")
    if isinstance(sched, dict):
        schedule_display = sched.get("display", "") or ""
    if not schedule_display:
        schedule_display = job.get("schedule_display", "") or ""

    deliver_str = job.get("deliver", "origin")
    deliver_payload = _deliver_to_payload(deliver_str, job.get("origin"))

    metadata = {
        "job_id": job_id,
        "name": job.get("name", ""),
        "schedule": schedule_display,
        "prompt": job.get("prompt", ""),
        "deliver": deliver_payload,
        "repeat": job.get("repeat"),
        "last_run_result": "success" if success else "failed",
        "scheduled_at": _now_iso(),
    }
    if delivery_error:
        metadata["delivery_error"] = delivery_error

    body = _extract_response_body(_LATEST_OUTPUT.get(job_id, "")) or (error or "").strip()

    parts = [
        "```cron-summary",
        json.dumps(metadata, ensure_ascii=False),
        "```",
    ]
    if body:
        parts.append("")  # blank line
        parts.append(body)

    return "\n".join(parts)


def _will_hit_repeat_limit(job: Optional[dict]) -> bool:
    """Predict whether _orig_mark will auto-delete this job (cron.jobs:707)."""
    if not job:
        return False
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    completed = repeat.get("completed", 0)
    return (
        times is not None
        and isinstance(times, int)
        and times > 0
        and completed + 1 >= times
    )


def _restore_as_completed(
    snapshot: dict,
    success: bool,
    error: Optional[str],
    delivery_error: Optional[str],
) -> None:
    """Re-add the auto-deleted job in disabled+completed state, preserving id."""
    try:
        from cron.jobs import load_jobs, save_jobs, _jobs_file_lock
        from hermes_time import now as _hermes_now
    except ImportError as _ie:
        _dbg(f"_restore_as_completed: import FAILED: {_ie}")
        return

    job = dict(snapshot)
    now_iso = _hermes_now().isoformat()

    repeat = dict(job.get("repeat") or {})
    if isinstance(repeat.get("times"), int):
        repeat["completed"] = repeat["times"]
    job["repeat"] = repeat or None

    job["last_run_at"] = now_iso
    job["last_status"] = "ok" if success else "error"
    job["last_error"] = None if success else error
    job["last_delivery_error"] = delivery_error
    job["next_run_at"] = None
    job["enabled"] = False
    job["state"] = "completed"

    with _jobs_file_lock:
        jobs = load_jobs()
        # Idempotent: skip if some other path already re-added the id.
        if any(j.get("id") == job["id"] for j in jobs):
            _dbg(f"_restore_as_completed: id={job['id']} already present, skip")
            return
        jobs.append(job)
        save_jobs(jobs)
        _dbg(f"_restore_as_completed: id={job['id']} restored as completed")


def _extract_response_body(full_md: str) -> str:
    """Strip the cron output md template and return only the agent's response.

    cron/scheduler.py builds the on-disk md as:

        # Cron Job: <name>
        **Job ID:** ...
        ## Prompt
        <prompt>
        ## Response
        <agent response>          ← this is what we want

    The header / Job ID / Schedule / Prompt sections are already encoded in
    the cron-summary JSON metadata; repeating them in the body is just noise.
    Falls back to the full md if the marker isn't found (failed jobs use a
    different template — body shows full text + error).
    """
    if not full_md:
        return ""
    marker = "## Response\n"
    idx = full_md.find(marker)
    if idx < 0:
        return full_md.strip()
    return full_md[idx + len(marker):].strip()


def _deliver_to_payload(deliver: str, origin: Optional[dict]) -> dict:
    """Convert hermes deliver string to APP CronDeliver shape (kept here
    so any deliver-mode evolution stays one helper change)."""
    if not deliver or deliver == "origin":
        return {
            "mode": "origin",
            "chat_name": (origin or {}).get("chat_name") or "",
        }
    if deliver == "new_session":
        return {"mode": "new_session"}
    if deliver == "local":
        return {"mode": "origin", "chat_name": ""}
    if ":" in deliver:
        platform, chat = deliver.split(":", 1)
        return {
            "mode": "specified",
            "send_to": {
                "channel": platform,
                "chat_name": chat,
                "chat_type": "group",
            },
        }
    return {"mode": "origin"}


def _now_iso() -> str:
    try:
        from hermes_time import now as _hermes_now
        return _hermes_now().isoformat()
    except Exception:
        import datetime as _dt
        return _dt.datetime.utcnow().isoformat() + "Z"


# Auto-install on import. Idempotent — install() guards via sentinel attr.
# Triggered by zet_agent_cron.pth in the venv site-packages.
_dbg("module imported")
try:
    install()
except Exception as _e:
    _dbg(f"install() RAISED: {_e!r}")
    import traceback as _tb
    _tb.print_exc()
    logger.warning("zet_agent_cron auto-install failed (non-fatal): %s", _e)
