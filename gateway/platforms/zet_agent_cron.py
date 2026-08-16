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

We patch ``cron.scheduler.{save_job_output, mark_job_run, run_job,
_deliver_result, _resolve_origin}`` at module import time (triggered by
zet_agent_cron.pth in the venv site-packages).

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

import atexit
import json
import logging
import mimetypes
import os
import re
import stat as _stat
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PATCH_SENTINEL = "__zet_agent_cron_patched__"

# job_id → most recent saved markdown content. Populated by the
# save_job_output wrapper, drained by the mark_job_run wrapper.
_LATEST_OUTPUT: Dict[str, str] = {}

# job_id → most recent saved markdown file path. Lets the mark_job_run
# wrapper append a "## Delivery Error" section to the very file the run just
# wrote, so the App's per-run history can show delivery failures distinctly.
_LATEST_OUTPUT_PATH: Dict[str, Any] = {}

# Tool calls whose successful execution we treat as "produced a file this
# turn". Keep in sync with zettlab-local-server/internal/chat/handler/
# produced_files.go (App reuses the same shape for in-chat file cards).
_PRODUCE_TOOL_NAMES = frozenset({
    "Write", "Edit", "MultiEdit", "NotebookEdit",
    "write_file", "edit_file",
})
_PATH_KEYS = ("file_path", "notebook_path", "path")
_CRON_ATTACHMENT_LIMIT = 32
_CRON_ATTACHMENT_SCAN_FILE_LIMIT = 2000
_CRON_ATTACHMENT_MTIME_SLACK_S = 60.0
_CRON_ATTACHMENT_EXCLUDED_SUFFIXES = frozenset({
    ".py", ".pyc", ".pyo",
})
_CRON_ATTACHMENT_EXTERNAL_SUFFIXES = frozenset({
    ".csv", ".doc", ".docx", ".gif", ".htm", ".html", ".jpeg", ".jpg",
    ".json", ".md", ".mp3", ".mp4", ".pdf", ".png", ".ppt", ".pptx",
    ".txt", ".wav", ".webm", ".xls", ".xlsx", ".xml", ".zip",
})
_CRON_ATTACHMENT_TEMP_DIRS = frozenset({
    "tmp", "var/tmp", "private/tmp",
})
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w./-])/{1,3}[^\s\"'<>`|)]{2,}")
# Profile directory name (= agent id). Anything else never becomes a path.
_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _scoped_env(name: str, default: str = "") -> str:
    """Read an env value through the cron profile secret scope when available.

    Under the multiplex gateway there is no per-agent child process:
    ZET_CHAT_APPEND_URL / ZET_AGENT_ID / ZETTLAB_AGENT_ACTION_TOKEN live in
    the profile's ``.env`` (written by zettlab-local-server), NOT in
    ``os.environ``. ``cron.scheduler._cron_env`` resolves through the active
    profile secret scope — including a fresh ``.env`` re-read for values
    written after gateway startup — and falls back to ``os.environ`` in
    legacy per-profile processes.

    On resolution failure (missing scope / upstream helper renamed) the
    fallback is mode-dependent: legacy processes read ``os.environ`` (their
    values were injected per-process, so it is safe), but under an ACTIVE
    multiplexer we return the default instead — gateway startup loads the
    active profile's ``.env`` into the process environment, so reading
    ``os.environ`` here could deliver this cron's result with ANOTHER
    profile's ZET_CHAT_APPEND_URL / action token. A skipped report beats a
    cross-profile delivery.
    """
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


def _is_zet_agent_platform(platform: Any) -> bool:
    return str(platform or "").lower() in {"zet_agent", "zettlab"}


def _target_to_deliver_value(target: dict) -> str:
    value = f"{target.get('platform')}:{target.get('chat_id')}"
    thread_id = target.get("thread_id")
    if thread_id is not None:
        value += f":{thread_id}"
    return value


_CHANNEL_DELIVER_PREFIX = "channel:"
_CHANNEL_SEND_PATH = "/api/v1/internal/agent/channels/send"
_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"


def _split_channel_targets(deliver):
    """Split a cron ``deliver`` string into (channel_kinds, remaining, invalid).

    ``deliver`` is comma-separated (e.g. "origin,channel:wechat"). channel:<kind>
    targets are pulled out here — NOT via scheduler._resolve_delivery_targets,
    because channel:<kind> is not a known platform and would be dropped.

    Returns:
      - channel_kinds: valid kinds (e.g. ["wechat"])
      - remaining: non-channel tokens re-joined, handed back to the original path
      - invalid: malformed channel tokens (e.g. "channel:" with no kind). The
        caller turns these into a delivery error — they are NOT silently dropped,
        so a misconfigured deliver reaches last_delivery_error.
    """
    if not deliver:
        return [], "", []
    kinds = []
    remaining = []
    invalid = []
    for raw in str(deliver).split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok.startswith(_CHANNEL_DELIVER_PREFIX):
            kind = tok[len(_CHANNEL_DELIVER_PREFIX):].strip()
            if kind:
                kinds.append(kind)
            else:
                invalid.append(tok)  # "channel:" with empty kind — surface it
            continue
        remaining.append(tok)
    return kinds, ",".join(remaining), invalid


def _resolve_channel_send_url():
    """Derive local-server's channel-send endpoint from ZET_CHAT_APPEND_URL."""
    raw = _scoped_env("ZET_CHAT_APPEND_URL").strip()
    if not raw:
        return None
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, _CHANNEL_SEND_PATH, "", ""))


def _post_channel_chunk(url: str, token: str, kind: str, text: str, job_id: str):
    """POST one (already-chunked) text to local-server's channel send endpoint.
    Returns None on success or an error string on failure."""
    payload = json.dumps(
        {"target_ref": _CHANNEL_DELIVER_PREFIX + kind, "text": text, "source": "cron", "job_id": job_id}
    ).encode("utf-8")
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            data=payload,
            headers={_ACTION_TOKEN_HEADER: token, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            # Read the full body — local-server's error envelope is tiny, and a
            # truncated read could split the JSON and hide the real "detail".
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
    except Exception as e:
        return f"channel:{kind} delivery failed: {e}"
    if isinstance(parsed, dict) and parsed.get("code") == 200:
        return None
    detail = ""
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
        detail = parsed["data"].get("detail", "")
    return f"channel:{kind} delivery failed: {detail or body[:200]}"


_REFRESH_PERMIT_PATH = "/api/v1/internal/apps/refresh_permit"
# How long a "this agent has no bound app" answer is trusted. Bindings change
# rarely, but a profile can BECOME a maintainer's after app creation, so the
# negative answer must expire rather than last the gateway's lifetime.
_REFRESH_PERMIT_NEG_TTL = 600.0
_REFRESH_PERMIT_TIMEOUT = 3.0
# The permit answer is a tiny JSON decision; a larger body means local-server
# is misbehaving or the request was misrouted, so fail open instead of buffering
# an unbounded stream on a 2 GB device.
_REFRESH_PERMIT_MAX_BODY = 8192
_refresh_permit_neg_cache: dict = {}


def _resolve_local_server_origin() -> str:
    """Derive local-server's origin from ZET_CHAT_APPEND_URL (same derivation
    as _resolve_channel_send_url, minus the path)."""
    raw = _scoped_env("ZET_CHAT_APPEND_URL").strip()
    if not raw:
        return ""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _governor_refresh_defer(job: Optional[dict]) -> Optional[tuple]:
    """Ask local-server whether this maintainer profile may refresh now.

    Runs BEFORE an agent turn is spent: under memory pressure local-server's
    governor answers defer with a retry interval, and the caller postpones
    the job (defer_job) instead of running into a wall. Returns
    (retry_seconds, reason) when the run should be deferred, None to run.

    Fail-open everywhere — no token/URL, unreachable server, unparsable
    answer all mean "run": the gate is an optimization for pressured
    devices, never a dependency. An agent with no bound app (404) is cached
    negatively so ordinary agents pay one lookup per TTL, not one per fire.
    """
    if not isinstance(job, dict) or not job.get("id"):
        return None
    token = _scoped_env("ZETTLAB_AGENT_ACTION_TOKEN").strip()
    origin = _resolve_local_server_origin()
    if not token or not origin:
        return None
    agent_key = _scoped_env("ZET_AGENT_ID").strip() or "?"
    try:
        import time as _time
        cached = _refresh_permit_neg_cache.get(agent_key)
        if cached and (_time.monotonic() - cached) < _REFRESH_PERMIT_NEG_TTL:
            return None
    except Exception:
        pass
    try:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            origin + _REFRESH_PERMIT_PATH,
            data=b"{}",
            headers={_ACTION_TOKEN_HEADER: token, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_REFRESH_PERMIT_TIMEOUT) as resp:
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    try:
                        if int(cl) > _REFRESH_PERMIT_MAX_BODY:
                            return None
                    except ValueError:
                        pass
                # urlopen's timeout is a per-recv socket timeout, NOT a total
                # deadline: a server that drips one byte per window keeps recv
                # from ever timing out and could hold this cron worker open for
                # hours. Bound the whole read against a wall-clock deadline and
                # fail open once it passes.
                import time as _time
                deadline = _time.monotonic() + _REFRESH_PERMIT_TIMEOUT
                raw = b""
                while len(raw) <= _REFRESH_PERMIT_MAX_BODY:
                    if _time.monotonic() >= deadline:
                        return None
                    chunk = resp.read(min(4096, _REFRESH_PERMIT_MAX_BODY + 1 - len(raw)))
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) > _REFRESH_PERMIT_MAX_BODY:
                    return None
                body = raw.decode("utf-8", errors="replace")
                status = resp.status
        except urllib.error.HTTPError as http_err:
            body = ""
            try:
                body = http_err.read(_REFRESH_PERMIT_MAX_BODY + 1).decode("utf-8", errors="replace")
            except Exception:
                pass
            status = http_err.code
        if status == 404:
            # Not a maintainer profile: nothing to gate, remember for a while.
            # Sweep expired entries on every 404 write and cap the dict so
            # profile churn cannot grow this resident cache without bound.
            import time as _time
            now = _time.monotonic()
            for k in [k for k, ts in _refresh_permit_neg_cache.items()
                      if now - ts >= _REFRESH_PERMIT_NEG_TTL]:
                _refresh_permit_neg_cache.pop(k, None)
            _refresh_permit_neg_cache[agent_key] = now
            while len(_refresh_permit_neg_cache) > 256:
                _refresh_permit_neg_cache.pop(next(iter(_refresh_permit_neg_cache)), None)
            return None
        if status != 200:
            return None
        parsed = json.loads(body)
        if not isinstance(parsed, dict) or parsed.get("decision") != "defer":
            return None
        retry_s = parsed.get("retry_in_seconds") or 60
        reason = str(parsed.get("reason") or "governor")
        return (max(30.0, float(retry_s)), reason)
    except Exception:
        return None


def _gate_run_one_job(orig_run_one_job, job, **kwargs):
    """Governor refresh gate around one job firing.

    On a defer decision the run is skipped entirely — no agent turn, no
    output file, no chat card — and the next slot is pushed out via
    defer_job. Returns True ("processed") for the skip; otherwise the
    original firing body decides. Extracted so the gate is unit-testable
    against a stub original.
    """
    # Only the app-refresh job is governed. An AppDedicated profile can also
    # host ordinary reminder/report cron jobs (created via the standard cronjob
    # tool) that must never be postponed by a profile-level governor decision —
    # a defer would silently skip time-sensitive one-shots. The refresh job is
    # tagged ``source="app_refresh"`` by local-server at creation.
    if not (isinstance(job, dict) and job.get("source") == "app_refresh"):
        return orig_run_one_job(job, **kwargs)
    defer_info = _governor_refresh_defer(job)
    if defer_info is not None:
        retry_s, reason = defer_info
        try:
            from cron.jobs import defer_job as _defer_job

            deferred = _defer_job(
                job["id"],
                seconds=retry_s,
                reason=f"governor:{reason}",
                # This gate consumed the occurrence without running it: the
                # claim must be terminated here (not deferred to update_job's
                # trigger-identity check, which is a no-op when next_run_at
                # doesn't move).
                clear_claim=True,
            )
            if deferred is None:
                _dbg(
                    f"governor defer: job {job.get('id')} vanished; "
                    "slot folds away"
                )
        except Exception as defer_err:
            # The gate decided to defer but the push failed (lock/CAS contention
            # or a transient jobs.json write error). Fail open: run the job via
            # the original body — its execution was already claimed, so the run
            # completes normally instead of losing a one-shot to a stale
            # next_run_at + reused dedup key.
            _dbg(
                f"governor defer: defer_job failed for "
                f"{job.get('id')}: {defer_err!r}; failing open"
            )
            return orig_run_one_job(job, **kwargs)
        # The execution was already created (claimed) before the gate; a
        # defer skips the run entirely, so terminalize it here or it stays
        # claimed forever (the ledger only prunes terminal rows, and
        # same-process recovery skips our own claimed rows).
        execution_id = job.get("execution_id")
        if execution_id:
            try:
                from cron.executions import finish_execution
                finish_execution(execution_id, success=True, delivery_outcome="suppressed")
            except Exception:
                _dbg(f"governor defer: finish_execution failed for {execution_id!r}")
        return True  # processed: nothing ran, nothing to deliver
    return orig_run_one_job(job, **kwargs)


def _send_to_channel(kind: str, content: str, job_id: str):
    """Deliver a cron result to a bound IM channel via local-server. Returns an
    error string on failure, or None on success — the caller folds the error
    into the cron job's last_delivery_error.

    Long content is chunked under local-server's per-message rune cap (mirroring
    how native send_message chunks before delivery) and sent piece by piece; each
    chunk is a separate send (separate channel message + cron_send audit). Any
    chunk failure is surfaced (never silently dropped).

    Reuses Phase A's send endpoint + verified-owner gate; the recipient is the
    channel's verified owner (resolved server-side). source="cron" tags the audit.
    """
    url = _resolve_channel_send_url()
    if not url:
        return "channel delivery: ZET_CHAT_APPEND_URL unset"
    token = _scoped_env("ZETTLAB_AGENT_ACTION_TOKEN").strip()
    if not token:
        return "channel delivery: action token unavailable"
    from tools.channel_text import chunk_channel_text
    chunks = chunk_channel_text(content)
    errors = []
    for idx, chunk in enumerate(chunks):
        err = _post_channel_chunk(url, token, kind, chunk, job_id)
        if err:
            label = f" (part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            errors.append(f"{err}{label}")
    return "; ".join(errors) if errors else None


def _handle_channel_delivery(job: dict, content: str):
    """Process channel: targets in a cron job's deliver string.

    Returns (error_or_none, remaining_deliver_str, had_channel_targets).
    - error: combined error string if any channel send failed OR any channel
      token was malformed, else None
    - remaining_deliver_str: deliver tokens minus channel: ones (for the
      original deliver path to handle)
    - had_channel_targets: whether any channel: token (valid OR invalid) present
    """
    kinds, remaining, invalid = _split_channel_targets(job.get("deliver"))
    if not kinds and not invalid:
        return None, remaining, False
    job_id = str(job.get("id", ""))
    errors = []
    # malformed tokens (e.g. "channel:") become explicit errors — never silently
    # dropped, so a misconfigured deliver lands in last_delivery_error.
    for bad in invalid:
        errors.append(f"invalid channel target {bad!r} (expected channel:<kind>)")
    for kind in kinds:
        err = _send_to_channel(kind, content, job_id)
        if err:
            errors.append(err)
    combined = "; ".join(errors) if errors else None
    return combined, remaining, True


def _is_public_zet_agent_session_id(value: str) -> bool:
    """Return whether value has the public zettlab user/agent/session shape."""
    parts = value.split(":", 3)
    return len(parts) == 4 and parts[0] == "zettlab" and all(parts[1:])


def _normalize_zet_agent_chat_id(value: str) -> str:
    """Strip this profile's internal multiplex prefix from a public chat id.

    Older ZetAgent builds accidentally persisted
    ``<profile_home>|zettlab:<user>:<agent>:<session>`` as ``origin.chat_id``.
    Only accept the exact active profile home and a valid public id tail; a
    foreign/arbitrary prefix remains unchanged so routing stays fail-closed.
    """
    raw = str(value or "").strip()
    if _is_public_zet_agent_session_id(raw):
        return raw
    profile_home, separator, candidate = raw.rpartition("|")
    if not separator or not _is_public_zet_agent_session_id(candidate):
        return raw
    try:
        from hermes_constants import get_hermes_home

        if Path(profile_home) != get_hermes_home():
            return raw
    except Exception as exc:
        _dbg(f"_normalize_zet_agent_chat_id: profile resolution FAILED: {exc!r}")
        return raw
    return candidate


def _resolve_zet_agent_chat_id(job: dict) -> str:
    origin = job.get("origin") or {}
    if isinstance(origin, dict):
        chat_id = str(origin.get("chat_id", "") or "").strip()
        platform = origin.get("platform")
        if chat_id and (not platform or _is_zet_agent_platform(platform)):
            return _normalize_zet_agent_chat_id(chat_id)
    try:
        import cron.scheduler as _sched
        for target in _sched._resolve_delivery_targets(job):
            if _is_zet_agent_platform(target.get("platform")):
                return _normalize_zet_agent_chat_id(target.get("chat_id", ""))
    except Exception as _e:
        _dbg(f"_resolve_zet_agent_chat_id: target resolution FAILED: {_e!r}")
    return ""


def _combine_delivery_errors(existing: Optional[str], added: str) -> str:
    if existing:
        return f"{existing}; {added}"
    return added


def _user_id_from(session_id: str) -> str:
    """zettlab:<userID>:<agentID>:<suffix> → userID（取不到返回空串）。"""
    parts = session_id.split(":", 3)
    return parts[1] if len(parts) == 4 and parts[0] == "zettlab" else ""


def _agent_id_from(session_id: str) -> str:
    """zettlab:<userID>:<agentID>:<suffix> → agentID（取不到返回空串）。"""
    parts = session_id.split(":", 3)
    return parts[2] if len(parts) == 4 and parts[0] == "zettlab" else ""


def _profile_state_db_candidates(agent_id: str) -> List[Path]:
    """``<root>/profiles/<agent_id>/state.db``；agent_id 来自 job.origin，只认纯 profile 名。"""
    if not agent_id or not _PROFILE_ID_RE.match(agent_id):
        return []
    try:
        from hermes_constants import get_hermes_home, get_process_hermes_home
        homes = [Path(get_hermes_home()), Path(get_process_hermes_home())]
    except Exception as _e:
        _dbg(f"_profile_state_db_candidates: home resolution FAILED: {_e!r}")
        return []

    out: List[Path] = []
    seen: set = set()
    for root in _state_db_roots(homes):
        base = root / "profiles"
        db_path = base / agent_id / "state.db"
        try:
            resolved = db_path.resolve()
            base_prefix = str(base.resolve()) + os.sep
        except OSError:
            continue
        if not str(resolved).startswith(base_prefix):
            continue
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        out.append(resolved)
    return out


def _state_db_roots(homes: Iterable[Path]) -> List[Path]:
    """每个 home 归一到它的 multiprofile root：``<root>/profiles/<name>`` → ``<root>``。

    home 自己**不能**同时当 root —— 否则会产出 ``<root>/profiles/<A>/profiles/<A>``
    这种嵌套假路径，而它排在候选第一位，新会话会被建进一个没人读的库。
    """
    roots: List[Path] = []
    for home in homes:
        root = home.parent.parent if home.parent.name == "profiles" else home
        if root not in roots:
            roots.append(root)
    return roots


def _root_state_db_candidates() -> List[Path]:
    """各 multiprofile root 自己的 ``state.db``（不是 ``profiles/`` 下的）。

    存量会话可能就住在 root 库里；而 cron 线程的 home override 常指向
    ``<root>/profiles/<name>``，此时 ``get_hermes_home()/state.db`` 是 profile 库、
    永远探不到 root 库 —— 漏了这一层，root-only 会话每次 run 都会被判成已删而派生 handoff。
    """
    try:
        from hermes_constants import get_hermes_home, get_process_hermes_home
        homes = [Path(get_hermes_home()), Path(get_process_hermes_home())]
    except Exception as _e:
        _dbg(f"_root_state_db_candidates: home resolution FAILED: {_e!r}")
        return []

    out: List[Path] = []
    seen: set = set()
    for root in _state_db_roots(homes):
        db_path = root / "state.db"
        key = str(db_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(db_path)
    return out


def _db_has_session(db_path: Path, session_id: str) -> Optional[bool]:
    """Read-only probe: does this state.db own ``session_id``?

    三态：``True`` 拥有 / ``False`` 确认不拥有 / ``None`` 探测失败（库读不出来）。
    ``None`` 绝不能被当成 ``False`` —— 那会把"库打不开"降级成"会话已删"，
    进而派生 handoff 并永久改写 ``job.origin.chat_id``（无回滚路径）。
    """
    try:
        if not db_path.exists():
            return False
    except OSError as _e:
        _dbg(f"_db_has_session: stat FAILED db={db_path}: {_e!r}")
        return None
    try:
        from hermes_state import SessionDB
    except ImportError as _e:
        _dbg(f"_db_has_session: SessionDB unavailable: {_e!r}")
        return None
    db = None
    try:
        db = SessionDB(db_path=db_path, read_only=True)
        return db.get_session(session_id) is not None
    except Exception as _e:
        _dbg(f"_db_has_session: probe FAILED db={db_path} sid={session_id}: {_e!r}")
        logger.warning(
            "cron persist: state.db ownership probe failed db=%s session=%s err=%r",
            db_path, session_id, _e,
        )
        return None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def _same_db_key(db_path: Path) -> str:
    try:
        return str(db_path.resolve())
    except OSError:
        return str(db_path)


def _job_store_agent_id() -> str:
    """执行中 job 所属 profile：当前 cron store 是 ``<root>/profiles/<X>/cron/jobs.json`` 时返回 X。

    这是服务端事实（store 路径来自 ContextVar override / 模块常量 / active home，
    不来自 job 内容），不依赖 profile secret scope 是否绑上；root/legacy store 返回空串。
    """
    try:
        from cron.jobs import _current_cron_store
        jobs_file = Path(_current_cron_store().jobs_file).resolve()
    except Exception as _e:
        _dbg(f"_job_store_agent_id: store resolution FAILED: {_e!r}")
        return ""
    parts = jobs_file.parts
    if len(parts) >= 4 and parts[-4] == "profiles" and parts[-2] == "cron" and parts[-1] == "jobs.json":
        candidate = parts[-3]
        if _PROFILE_ID_RE.match(candidate):
            return candidate
    return ""


def _resolve_persist_db_path(session_id: str) -> Tuple[Path, Optional[str]]:
    """``(持有 session_id 的 state.db, 未决原因)``；见 zettlab-local-server/docs/cron-run-history.md §3.3。

    顺序不能换：存量 root-only 会话若只看 profile 库会被判成已删 → 每次 run 派生 handoff 会话。
    第二个返回值非空 = 有候选库探测失败、归属未决，调用方必须 fail-closed，
    **不许**据此判定会话已删而新建会话。
    """
    from hermes_constants import get_hermes_home
    current = Path(get_hermes_home()) / "state.db"
    origin_agent_id = _agent_id_from(session_id)
    # job 所属 profile 优先于 .env 身份：前者是服务端事实，后者在 scope 未绑时读的是根 .env
    exec_agent_id = _job_store_agent_id() or _scoped_env("ZET_AGENT_ID").strip()
    if origin_agent_id and exec_agent_id and origin_agent_id != exec_agent_id:
        # session_id 来自调用方可控的 job.origin —— 跨 agent 的库一律不落
        return current, (
            f"origin agent {origin_agent_id!r} does not match executing agent "
            f"{exec_agent_id!r} for session {session_id} — cross-agent persist refused"
        )
    agent_id = origin_agent_id or exec_agent_id
    candidates = _profile_state_db_candidates(agent_id)

    probed: Dict[str, Optional[bool]] = {}

    def _owns(db_path: Path) -> Optional[bool]:
        key = _same_db_key(db_path)
        if key not in probed:
            probed[key] = _db_has_session(db_path, session_id)
        return probed[key]

    unknown: List[str] = []

    def _record_unknown(db_path: Path) -> None:
        key = _same_db_key(db_path)
        if key not in unknown:
            unknown.append(key)

    root_candidates = _root_state_db_candidates()

    for db_path in candidates:
        owned = _owns(db_path)
        if owned is None:
            _record_unknown(db_path)
        elif owned and not unknown:
            _warn_if_split_session(session_id, db_path, root_candidates, _owns)
            return db_path, None

    for db_path in root_candidates:
        owned = _owns(db_path)
        if owned is None:
            _record_unknown(db_path)
        elif owned and not unknown:
            return db_path, None

    owned = _owns(current)
    if owned is None:
        _record_unknown(current)
    elif owned and not unknown:
        return current, None

    if unknown:
        return current, (
            "state.db ownership undetermined for session "
            f"{session_id} (unreadable candidates: {', '.join(unknown)})"
        )

    for db_path in candidates:
        if db_path.parent.is_dir():
            return db_path, None
    return current, None


def _warn_if_split_session(session_id, profile_db, root_candidates, owns) -> None:
    """§6.3: 同一会话同时存在于 profile 库和 root 库时必须告警，不许静默取 profile。"""
    profile_key = _same_db_key(profile_db)
    for db_path in root_candidates:
        if _same_db_key(db_path) == profile_key:
            continue
        if owns(db_path) is True:
            logger.warning(
                "cron persist: split session detected — session=%s exists in both "
                "profile store %s and root store %s; writing to the profile store "
                "(metric=cron_persist_split_session)",
                session_id, profile_db, db_path,
            )
            return


def _cron_session_db():
    """cron session 所在库。bare ``SessionDB()`` 是故意的：cron/scheduler.py 也是 bare，两边必须同库。"""
    from hermes_state import SessionDB
    return SessionDB()


def _handoff_session_id(old_id: str) -> Optional[str]:
    """源对话已删时，基于旧 id 派生一个同 user/agent、新后缀的会话 id。

    仅对标准 ``zettlab:<userID>:<agentID>:<suffix>`` 形状生效；其它形状返回
    None（无法安全派生 → 退回原行为，让上层记为投递失败）。
    """
    import uuid
    parts = old_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "zettlab":
        return None
    return f"zettlab:{parts[1]}:{parts[2]}:{uuid.uuid4().hex[:12]}"


def _is_calendar_reminders_session(session_id: str) -> bool:
    """APP/local-server use this fixed synthetic chat for imported calendar reminders."""
    parts = session_id.split(":", 3)
    return len(parts) == 4 and parts[0] == "zettlab" and parts[3] == "calendar-reminders"


def _fence_safe(text: str) -> str:
    """把 error 串里的 ``` 折成 `` —— 防它破坏外层 fenced block / 让 App parser
    （parseCronRunStatus 的 `[\\s\\S]*?(?:\\n```|$)`）提前截断。两个 backtick
    不构成 fence，技术 error 串的字面 backtick 数量无语义损失。"""
    return re.sub(r"`{3,}", "``", text)


def _append_delivery_error_to_output(job_id: str, delivery_error: str) -> None:
    """Append a "## Delivery Error" section to the run's saved markdown.

    A run that executed fine but failed to deliver otherwise looks like a plain
    success in the App's per-run history (the job record's last_delivery_error
    only ever holds the latest run). Best-effort — never raises into the cron
    loop. App parser: zettlab-app services/cron-jobs.ts::parseCronRunStatus.
    """
    path = _LATEST_OUTPUT_PATH.get(job_id)
    if not path or not delivery_error:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Delivery Error\n\n```\n{_fence_safe(delivery_error)}\n```\n")
    except OSError as e:
        _dbg(f"_append_delivery_error_to_output FAILED job={job_id}: {e!r}")


def _append_run_error_to_output(job_id: str, reason: str) -> None:
    """Append a "## Error" section to the run's saved markdown so the App's
    per-run history (parseCronRunStatus) reads this run as failed — matching
    the cron-summary card / job last_status. Needed when mark_job_run downgrades
    a "fake success": scheduler already wrote the .md as a success doc before we
    flipped success, so without this the run history would still show success.
    Best-effort — never raises into the cron loop.
    """
    path = _LATEST_OUTPUT_PATH.get(job_id)
    if not path or not reason:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n## Error\n\n```\n{_fence_safe(reason)}\n```\n")
    except OSError as e:
        _dbg(f"_append_run_error_to_output FAILED job={job_id}: {e!r}")


def _dbg(msg: str) -> None:
    """Best-effort debug log to /tmp/zet_agent_cron.log — useful during
    development since hermes child stdout/stderr land in zerolog inside
    local-server console. Production deployments can ignore this file."""
    try:
        import datetime as _dt
        with open("/tmp/zet_agent_cron.log", "a", encoding="utf-8") as _f:
            _f.write(f"{_dt.datetime.now().isoformat()} pid={os.getpid()} {msg}\n")
    except Exception:
        pass


def install() -> None:
    """Patch Zettlab cron persistence/delivery seams. Idempotent."""
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
        saved = _orig_save(job_id, output)
        _LATEST_OUTPUT_PATH[job_id] = saved
        return saved

    def _wrapped_mark(
        job_id: str,
        success: bool,
        error: Optional[str] = None,
        delivery_error: Optional[str] = None,
        scheduled_at: Optional[str] = None,
        output_filename: Optional[str] = None,
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

        # 把"没干活的假成功"降级成 failed（在下方生成 App 卡片之前）。
        # 同时给 run .md 补 "## Error"，让详情页运行历史也读成 failed —— scheduler
        # 早把 .md 当成功文档写盘了，不补这刀运行历史仍显示成功，三个面不一致。
        if success:
            _fake_reason = _detect_fake_success(job_id)
            if _fake_reason:
                _dbg(f"_wrapped_mark: fake-success job={job_id}: {_fake_reason}")
                success = False
                error = error or _fake_reason
                _append_run_error_to_output(job_id, _fake_reason)

        effective_delivery_error = delivery_error
        try:
            persist_error = _try_persist_to_session(
                job_id, success, error, delivery_error, job_snapshot
            )
            if persist_error:
                effective_delivery_error = _combine_delivery_errors(
                    effective_delivery_error, persist_error
                )
        except Exception as e:
            _dbg(f"_wrapped_mark persist FAILED: {e!r}")
            logger.warning(
                "zet_agent_cron: persist for job %s failed (non-fatal): %s",
                job_id, e,
            )
            effective_delivery_error = _combine_delivery_errors(
                effective_delivery_error,
                f"zet_agent session persist failed: {e}",
            )

        if effective_delivery_error:
            _append_delivery_error_to_output(job_id, effective_delivery_error)

        try:
            result = _orig_mark(
                job_id,
                success,
                error,
                delivery_error=effective_delivery_error,
                scheduled_at=scheduled_at,
                output_filename=output_filename,
            )

            # PRD UX：once+repeat=N 跑满后 hermes 默认把 job 从 jobs.json pop
            # 掉 → APP 列表空。我们把它"复活"成 enabled=False/state=completed
            # 状态保留住，APP 列表能继续看到+查看运行历史。Id / created_at /
            # origin 等元数据来自 snapshot，保证用户在列表里看到的是同一条任
            # 务记录。
            if will_be_auto_deleted and job_snapshot is not None:
                try:
                    _restore_as_completed(
                        job_snapshot, success, error, effective_delivery_error
                    )
                except Exception as e:
                    _dbg(f"_wrapped_mark restore FAILED: {e!r}")
                    logger.warning(
                        "zet_agent_cron: restore-as-completed for %s failed: %s",
                        job_id, e,
                    )

            return result
        finally:
            _LATEST_OUTPUT.pop(job_id, None)
            _LATEST_OUTPUT_PATH.pop(job_id, None)

    setattr(_wrapped_mark, _PATCH_SENTINEL, True)
    setattr(_wrapped_save, _PATCH_SENTINEL, True)

    _sched.save_job_output = _wrapped_save
    _sched.mark_job_run = _wrapped_mark
    _dbg("install() patched mark_job_run + save_job_output OK")

    # ── run_job retry wrapper — auto-retry clean transient failures.
    try:
        if not getattr(_sched.run_job, _PATCH_SENTINEL, False):
            _orig_run_job = _sched.run_job

            def _wrapped_run_job(job, *, defer_agent_teardown=None):
                # Upstream scheduler.run_one_job() passes this holder so agent
                # async resources stay alive through delivery. A retry does not
                # need the previous failed attempt's agent, though: release it
                # immediately before the next attempt, and hand only the final
                # attempt back for post-delivery teardown.
                attempt_agents = [] if defer_agent_teardown is not None else None

                def _run_once(retry_job):
                    return _orig_run_job(
                        retry_job,
                        defer_agent_teardown=attempt_agents,
                    )

                def _release_failed_attempt():
                    while attempt_agents:
                        _sched._teardown_cron_agent(
                            attempt_agents.pop(),
                            job.get("id", ""),
                        )

                run_result = None
                try:
                    run_result = _run_job_with_retry(
                        _run_once,
                        job,
                        before_retry=(
                            _release_failed_attempt
                            if attempt_agents is not None
                            else None
                        ),
                    )
                    return run_result
                finally:
                    # Zettlab file-change protection: a cron run is one turn, so
                    # release its protection snapshot pin here. No-op when the
                    # run never touched protected files.
                    try:
                        from tools.zettlab_snapshot_guard import finish_turn

                        succeeded = (
                            isinstance(run_result, tuple)
                            and len(run_result) > 0
                            and bool(run_result[0])
                        )
                        # 指名收本次 run 的轮（agent 运行时的 _current_turn_id）；
                        # 拿不到时 guard 一律不收（空 id 收「唯一余轮」会错收并发
                        # 轮的 pin），留给服务端 TTL 自愈。
                        guard_turn = ""
                        interrupted = False
                        if attempt_agents:
                            final_agent = attempt_agents[-1]
                            guard_turn = str(
                                getattr(final_agent, "_current_turn_id", "") or ""
                            )
                            interrupted = bool(
                                getattr(final_agent, "_interrupt_requested", False)
                            )
                        if interrupted:
                            # inactivity timeout 路径：scheduler 先 interrupt 再
                            # shutdown(wait=False) 返回失败，executor / 工具线程
                            # 未必已退出，立刻 finish 会在后台写入完成前关掉恢复
                            # 窗口（Codex review P1）。这里不收 pin，留给 LS 侧
                            # PinTTL + reconcile 自愈（终态 reconcile_timeout）。
                            _dbg(
                                "snapshot guard finish skipped: interrupted turn, "
                                "pin left to server-side TTL"
                            )
                        else:
                            finish_turn(
                                "completed" if succeeded else "failed",
                                turn_id=guard_turn,
                            )
                    except Exception:
                        _dbg("snapshot guard finish failed")
                    # Success, terminal failure, or an exception: preserve the
                    # upstream contract for whichever attempt is still live.
                    if attempt_agents:
                        defer_agent_teardown.extend(attempt_agents)

            setattr(_wrapped_run_job, _PATCH_SENTINEL, True)
            _sched.run_job = _wrapped_run_job
            _dbg("install() patched run_job OK")
    except Exception as _e:
        _dbg(f"install() run_job patch FAILED: {_e!r}")

    # ── governor refresh gate — defer BEFORE an agent turn is spent ────────
    # run_one_job is the shared firing body for both the built-in ticker and
    # an external provider's fire_due, so gating here covers every dispatch
    # path. A deferral skips the whole run (no agent, no output file, no
    # chat card — a postponed refresh is invisible by design) and pushes the
    # next slot out; the already-advanced current slot folds away, so a
    # month of deferrals is one catch-up run later, never a backlog burst.
    try:
        if not getattr(_sched.run_one_job, _PATCH_SENTINEL, False):
            _orig_run_one_job = _sched.run_one_job

            def _gated_run_one_job(job, **kwargs):
                return _gate_run_one_job(_orig_run_one_job, job, **kwargs)

            setattr(_gated_run_one_job, _PATCH_SENTINEL, True)
            _sched.run_one_job = _gated_run_one_job
            _dbg("install() patched run_one_job (governor gate) OK")
    except Exception as _e:
        _dbg(f"install() run_one_job patch FAILED: {_e!r}")

    # ── scheduler delivery patch — keep Zettlab-specific delivery out of
    # upstream cron/scheduler.py. App cron output is persisted below in
    # _wrapped_mark via SessionDB + ZET_CHAT_APPEND_URL; scheduler's generic
    # live/standalone send path cannot deliver to a HTTP request/response
    # platform and would otherwise report a false delivery error.
    try:
        if not getattr(_sched._resolve_origin, _PATCH_SENTINEL, False):
            _orig_resolve_origin = _sched._resolve_origin

            def _wrapped_resolve_origin(job):
                origin = _orig_resolve_origin(job)
                if (
                    isinstance(origin, dict)
                    and str(origin.get("platform", "")).lower() == "zettlab"
                ):
                    origin = dict(origin)
                    origin["platform"] = "zet_agent"
                return origin

            setattr(_wrapped_resolve_origin, _PATCH_SENTINEL, True)
            _sched._resolve_origin = _wrapped_resolve_origin
            _dbg("install() patched scheduler._resolve_origin OK")

        if not getattr(_sched._deliver_result, _PATCH_SENTINEL, False):
            _orig_deliver_result = _sched._deliver_result

            def _wrapped_deliver_result(job, content, adapters=None, loop=None):
                # De-identify the upstream failure template before delivery.
                content = _redact_channel_failure(job, content)
                channel_err = None
                had_channel = False
                try:
                    channel_err, remaining_deliver, had_channel = _handle_channel_delivery(job, content)
                    if had_channel:
                        if not remaining_deliver:
                            # only channel targets — no original delivery to run
                            return channel_err
                        # strip channel tokens, let the rest flow through below
                        job = dict(job)
                        job["deliver"] = remaining_deliver
                except Exception as _e:
                    # do NOT swallow — the whole point of this is no silent
                    # delivery failures. Surface the pre-handle crash as an error.
                    _dbg(f"_deliver_result channel pre-handle FAILED: {_e!r}")
                    channel_err = f"channel delivery pre-handle failed: {_e}"

                # Existing zet_agent delivery logic — preserved, EXCEPT the
                # zet-only "bypass" branch becomes a flag instead of `return None`,
                # so a channel send error can never be discarded by an early return.
                orig_err = None
                bypassed = False
                try:
                    targets = _sched._resolve_delivery_targets(job)
                    zet_targets = [
                        t for t in targets if _is_zet_agent_platform(t.get("platform"))
                    ]
                    if zet_targets:
                        other_targets = [
                            t for t in targets
                            if not _is_zet_agent_platform(t.get("platform"))
                        ]
                        if not other_targets:
                            _dbg(
                                f"_deliver_result: bypass zet_agent delivery job={job.get('id')}"
                            )
                            bypassed = True  # was: return None — now a flag so channel_err survives
                        else:
                            job = dict(job)
                            job["origin"] = None
                            job["deliver"] = ",".join(
                                _target_to_deliver_value(t) for t in other_targets
                            )
                except Exception as _e:
                    _dbg(f"_deliver_result patch pre-check FAILED: {_e!r}")

                if not bypassed:
                    orig_err = _orig_deliver_result(job, content, adapters=adapters, loop=loop)

                # combine channel + original delivery errors — neither is ever
                # silently dropped, including when the zet_agent path bypassed.
                if channel_err and orig_err:
                    return f"{channel_err}; {orig_err}"
                return channel_err or orig_err

            setattr(_wrapped_deliver_result, _PATCH_SENTINEL, True)
            _sched._deliver_result = _wrapped_deliver_result
            _dbg("install() patched scheduler._deliver_result OK")
    except Exception as _e:
        _dbg(f"install() scheduler delivery patch FAILED: {_e!r}")

    # Warn if the upstream failure template friendly-ize depends on has drifted.
    try:
        import inspect as _inspect
        _warn_if_failure_template_drifted(_inspect.getsource(_sched))
    except Exception as _e:
        _dbg(f"install() failure-template self-check skipped: {_e!r}")

    # ── _flush_messages_to_session_db patch — fix user-message-drop bug ──
    #
    # 上游 _flush_messages_to_session_db 用 len(conversation_history) 当
    # flush 起点，假设 messages[:N] 跟 conversation_history 一对一对应。
    # 但 _repair_message_sequence (run_agent.py:3891) 会把 messages 里
    # 相邻 user 合并成一条，导致 messages 比 conversation_history 短，
    # flush_from 越过新 turn 的 user message —— assistant 入库但 user 丢失。
    #
    # 现象只出现在 SessionDB 里历史包含相邻 user message 的 session（最常
    # 见于 channel session：一次 turn 被中断 / clarify timeout 后留下两条
    # 没有 assistant 隔开的 user）。一旦埋下"种子"，所有后续 turn 都被 hit。
    #
    # 上游修过 _compress_context 路径（line 11055 显式把 conversation_history
    # 设为 None），但 _repair_message_sequence 路径没补。这里 wrap 一下，把
    # conversation_history 切到 _persist_user_message_idx 前（_run_agent 维
    # 护好的当前-turn user 锚点），让原 flush_from 公式落到 user 位置。
    try:
        from run_agent import AIAgent

        if not getattr(AIAgent._flush_messages_to_session_db, _PATCH_SENTINEL, False):
            _orig_flush = AIAgent._flush_messages_to_session_db

            def _wrapped_flush(self, messages, conversation_history=None):
                # 修正 _repair_message_sequence 合并相邻 user 导致的 flush 错位。
                # 边界条件设计：
                #  - persist_idx = _persist_user_message_idx 是 _run_agent 设的
                #    "期望 user 位置"（line 10949）；如果没设，跳过 patch。
                #  - walk messages 末尾向前找真实 user 位置 turn_user_idx；
                #    无 user（cron 等）→ 跳过。
                #  - 只有 turn_user_idx < persist_idx 时才修（说明 merge 把 user
                #    向前推了），其他情况一律走原逻辑，避免误伤。
                try:
                    persist_idx = getattr(self, "_persist_user_message_idx", None)
                    if persist_idx is None or not isinstance(messages, list):
                        return _orig_flush(self, messages, conversation_history)

                    turn_user_idx = None
                    for _i in range(len(messages) - 1, -1, -1):
                        _m = messages[_i]
                        if isinstance(_m, dict) and _m.get("role") == "user":
                            turn_user_idx = _i
                            break

                    if (
                        turn_user_idx is not None
                        and turn_user_idx < persist_idx
                        and isinstance(conversation_history, list)
                        and turn_user_idx < len(conversation_history)
                    ):
                        _dbg(
                            f"_flush patch: adjust hist_len={len(conversation_history)} "
                            f"→ {turn_user_idx} (persist_idx={persist_idx}, msgs_len={len(messages)})"
                        )
                        conversation_history = conversation_history[:turn_user_idx]
                except Exception as _e:
                    _dbg(f"_flush patch: pre-adjust FAILED (falling back): {_e!r}")
                return _orig_flush(self, messages, conversation_history)

            setattr(_wrapped_flush, _PATCH_SENTINEL, True)
            AIAgent._flush_messages_to_session_db = _wrapped_flush
            _dbg("install() patched AIAgent._flush_messages_to_session_db OK")
    except ImportError as _ie:
        _dbg(f"install() _flush patch SKIP (ImportError): {_ie}")

    logger.info("zet_agent_cron: installed cron persistence hooks on cron.scheduler")


# ── Fake-success detection (ZET-1048) ───────────────────────────────
# 意图宣告句（"我要去做X"），须配合"0 工具执行"才用于降级。中英覆盖。
_INTENT_ANNOUNCE_RE = re.compile(
    r"^\s*"
    r"(?:[^\n。.!?！？]{0,12}[,，、:：]\s*)?"  # 可选短开场白，如 "好的，" / "Sure,"
    r"("
    r"i['’]?ll\b|i\s+will\b|i['’]?m\s+going\s+to\b|i\s+am\s+going\s+to\b|"
    r"let\s+me\b|let['’]?s\b|sure[,.\s]|one\s+moment\b|hold\s+on\b|"
    r"give\s+me\s+a\s+moment\b|"
    r"我(将|来|这就|马上|现在)|让我|稍等|请稍候|马上(去|为)|现在(就)?(去|来|帮|为)"
    r")",
    re.IGNORECASE,
)
_SCRIPT_FRAGMENT_RE = re.compile(r"^[`'\"]?[A-Za-z0-9_.-]+\.(?:py|pyc|pyo)[`'\"]?$", re.IGNORECASE)


def _detect_malformed_response_fragment(body: str) -> Optional[str]:
    """Return a fake-success reason for tiny truncated/code-fragment bodies.

    Cron summaries are user-facing notifications. A body like "`.app" is not a
    valid result; it is a broken markdown/code fragment from an interrupted or
    derailed model response. Keep this intentionally narrow so short real
    reminders ("OK", "Done", "已同步") stay deliverable.
    """
    compact = (body or "").strip()
    if not compact or len(compact) > 120:
        return None
    if "\n" in compact:
        return None
    if compact.count("`") % 2 == 1:
        return (
            "agent produced no user-facing result; got a malformed markdown "
            "fragment"
        )
    if _SCRIPT_FRAGMENT_RE.match(compact):
        return (
            "agent produced no user-facing result; got a helper script "
            "filename"
        )
    return None


def _cron_session_like(job_id: str) -> str:
    """LIKE pattern (ESCAPE '\\') matching this job's cron sessions: cron_<id>_*."""
    prefix = f"cron_{job_id}_"
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _tool_activity_in_messages(messages) -> int:
    """tool_call + tool-result count across one session's message list."""
    count = 0
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                count += len(tcs)
        elif role == "tool":
            count += 1
    return count


def _count_tool_activity(job_id: str) -> Optional[int]:
    """最近一次 cron session 的 tool_call + tool 结果数；读不到返回 None（fail-open）。"""
    if not job_id:
        return None
    try:
        db = _cron_session_db()  # 换 profile 库 → 查不到 session → fake-success 检测静默失效
    except ImportError:
        return None
    try:
        try:
            with db._lock:
                cursor = db._conn.execute(
                    "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                    "ORDER BY started_at DESC LIMIT 1",
                    (_cron_session_like(job_id),),
                )
                row = cursor.fetchone()
        except Exception as _e:
            _dbg(f"_count_tool_activity: session lookup FAILED: {_e!r}")
            return None
        if not row:
            return None
        try:
            messages = db.get_messages(row["id"])
        except Exception as _e:
            _dbg(f"_count_tool_activity: get_messages FAILED: {_e!r}")
            return None
        return _tool_activity_in_messages(messages)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _list_cron_session_ids(job_id: str) -> Optional[set]:
    """All session ids for this job (cron_<id>_*); None if unreadable (fail-open)."""
    if not job_id:
        return None
    try:
        db = _cron_session_db()  # 换 profile 库 → 返回 None → retry 的 silent 判定永远跳过
    except ImportError:
        return None
    try:
        with db._lock:
            cursor = db._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\'",
                (_cron_session_like(job_id),),
            )
            return {r["id"] for r in cursor.fetchall()}
    except Exception as _e:
        _dbg(f"_list_cron_session_ids FAILED: {_e!r}")
        return None
    finally:
        try:
            db.close()
        except Exception:
            pass


def _count_session_tool_activity(session_id: str) -> Optional[int]:
    """tool_call + tool-result count for one session; None if unreadable."""
    if not session_id:
        return None
    try:
        db = _cron_session_db()  # 换 profile 库 → get_messages 返回 [] → 误判「没跑过工具」而重复 retry
    except ImportError:
        return None
    try:
        try:
            messages = db.get_messages(session_id)
        except Exception as _e:
            _dbg(f"_count_session_tool_activity: get_messages FAILED: {_e!r}")
            return None
        return _tool_activity_in_messages(messages)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _attempt_tool_activity(job_id: str, baseline: Optional[set]) -> Optional[int]:
    """Total tool activity in cron sessions THIS run produced — i.e. sessions
    whose id is not in ``baseline`` (the id set captured ONCE before the first
    attempt). Measuring against the fixed pre-run baseline (rather than
    re-snapshotting per attempt) keeps a retry that reuses the same
    second-precision session id counted: ``INSERT OR IGNORE`` lets the retry's
    tool calls land in the existing row, but that id is still absent from
    ``baseline`` so its tools are still summed. Reads messages only for this
    run's sessions, so cost is O(attempts), not O(historical sessions). 0 →
    nothing ran, safe to retry; None → unresolved, caller skips."""
    if baseline is None:
        return None
    now = _list_cron_session_ids(job_id)
    if now is None:
        return None
    total = 0
    for sid in (now - baseline):
        c = _count_session_tool_activity(sid)
        if c is None:
            return None
        total += c
    return total


def _detect_fake_success(job_id: str) -> Optional[str]:
    """Return a failure reason when a successful run has no user-facing result."""
    try:
        body = _extract_response_body(_LATEST_OUTPUT.get(job_id, "")).strip()
    except Exception:
        return None
    if not body:
        return None  # 空 body 上游已 soft-fail (#8585)
    malformed = _detect_malformed_response_fragment(body)
    if malformed:
        return malformed
    if len(body) > 400 or not _INTENT_ANNOUNCE_RE.search(body):
        return None  # 实质内容 / 非意图句 = 真答案
    tool_activity = _count_tool_activity(job_id)
    if tool_activity is None or tool_activity > 0:
        return None  # 读不到，或工具跑了 → 保持成功
    return (
        "agent announced an action but executed no tools and produced no "
        "result (model stream likely interrupted mid tool-call, or no tool "
        "call was emitted)"
    )


# ── Silent-run detection ────────────────────────────────────────────
# 镜像 cron.scheduler.SILENT_MARKER；本地常量避免在 .pth 早期 import 时拉 scheduler。
_SILENT_MARKER = "[SILENT]"
_SILENT_STATUS_RE = re.compile(r"^\*\*Status:\*\*\s*silent\b", re.MULTILINE)


def _is_silent_run(job_id: str) -> bool:
    """本轮 tick 是否为 scheduler 已跳过推送的「静默运行」。

    两种来源（都被 scheduler.tick() 用 SILENT_MARKER 跳过推送）：
      - no_agent 脚本无输出（窗口外提醒 / wakeAgent=false）→ doc 带 "**Status:** silent (...)"
      - agent 回复 [SILENT] → 落在 doc 的 ## Response body

    落卡路径必须同样跳过，否则每个窗口外 tick 都会往聊天泄漏一张空卡；
    per-run .md 仍由 save_job_output 落盘，详情页历史不受影响。
    """
    doc = _LATEST_OUTPUT.get(job_id, "")
    if not doc.strip():
        return False
    metadata = re.split(r"^##\s+(?:Prompt|Response)\s*$", doc, maxsplit=1, flags=re.MULTILINE)[0]
    if _SILENT_STATUS_RE.search(metadata):
        return True
    body = _extract_response_body(doc).strip()
    # Reuse the scheduler's canonical rule. A real response may discuss the
    # sentinel in prose (or in model reasoning) and still end with useful
    # content; substring matching silently dropped those completed jobs from
    # the APP conversation while scheduler delivery considered them non-silent.
    try:
        from cron.scheduler import _is_cron_silence_response
        return _is_cron_silence_response(body)
    except Exception:
        # Import-cycle/startup fallback: fail open for any substantive output.
        return body.strip().upper() == _SILENT_MARKER


# ── Failure classification / friendly messaging ────────────────────
# Stable user-facing categories. Retryability is derived from the category,
# rather than from one broad regex, so clients can explain the actual problem
# without exposing provider error strings.
_NETWORK_ERROR_RE = re.compile(
    r"(?:temporary failure in name resolution|name or service not known"
    r"|nodename nor servname|connecterror|getaddrinfo|\bdns\b"
    r"|network is unreachable|connection (?:reset|refused|aborted|error)"
    r"|read tcp|^\s*EOF\s*$"
    r"|(?:http|response|connection|socket|tcp)[^;\n]{0,80}\bEOF\b)",
    re.IGNORECASE,
)
_TIMEOUT_ERROR_RE = re.compile(
    r"(?:context deadline exceeded|Client\.Timeout|timeout|timed out)",
    re.IGNORECASE,
)
_RATE_LIMIT_ERROR_RE = re.compile(
    r"(?:rate[ _-]?limit|\b429\b|too many requests)",
    re.IGNORECASE,
)
_UPSTREAM_ERROR_RE = re.compile(
    r"(?:\b50[234]\b|temporarily unavailable|overloaded)",
    re.IGNORECASE,
)
_INSUFFICIENT_CREDITS_RE = re.compile(
    r"(?:\b402\b|insufficient_credits|insufficient credits"
    r"|credit balance[^;\n]{0,40}"
    r"(?:insufficient|empty|exhausted|zero|too low|not enough)"
    r"|(?:insufficient|empty|exhausted|zero|too low|not enough)"
    r"[^;\n]{0,40}credit balance)",
    re.IGNORECASE,
)
# Ran but produced nothing usable (#8585 / fake-success).
_EMPTY_RESULT_RE = re.compile(
    r"empty response|produced no|executed no tools|no response generated",
    re.IGNORECASE,
)

def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    """Env int clamped to [lo, hi]; falls back to default on missing/garbage."""
    try:
        v = int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# Auto-retry budget for clean (zero-tool-activity) transient failures.
# ZET_CRON_RETRY_MAX=0 disables retries.
_MAX_RUN_RETRIES = _env_int("ZET_CRON_RETRY_MAX", 2, lo=0, hi=10)
_RETRY_BACKOFF_S = _env_int("ZET_CRON_RETRY_BACKOFF_S", 45, lo=0, hi=600)

# Set at exit so a backoff wait aborts instead of stalling the pool's wait=True drain.
_shutdown = threading.Event()
atexit.register(_shutdown.set)

# The two literal fragments _redact_channel_failure depends on — the job name
# sits between them ("⚠️ Cron '<name>' failed:\n" is the upstream template).
# BOTH must survive an upstream sync, or the redaction silently stops matching
# and raw errors leak through channel delivery.
_CHANNEL_FAILURE_FRAGMENTS = ("⚠️ Cron '", "' failed:")


def _warn_if_failure_template_drifted(scheduler_source: str) -> bool:
    """True if upstream's failure template is intact; else _dbg-warn and False.

    Checks BOTH literal fragments the redaction needs, not just the prefix: a
    wording/case change to "failed:" alone would otherwise pass silently while
    _redact_channel_failure quietly stops matching and raw errors start leaking."""
    if all(frag in scheduler_source for frag in _CHANNEL_FAILURE_FRAGMENTS):
        return True
    _dbg("WARNING: upstream cron failure template changed — "
         "_redact_channel_failure may no longer match; raw errors could leak")
    return False

# code → default-locale reason (clients localize off the code).
_FAILURE_REASON = {
    "network_unavailable": "暂时无法连接服务，请检查网络后重试",
    "upstream_unavailable": "AI 服务暂时繁忙",
    "timeout": "任务执行超时",
    "rate_limited": "AI 服务请求过于频繁",
    "insufficient_credits": "积分不足，无法生成内容",
    "empty_response": "本次未产出有效结果",
    "agent_error": "执行出错",
    "unknown": "执行失败",
}
_RETRY_STATE_LIMIT = 512
_LAST_RETRY_STATE: Dict[str, Dict[str, Any]] = {}


def _new_retry_state() -> Dict[str, Any]:
    return {
        "attempts": 0,
        "max_attempts": _MAX_RUN_RETRIES,
        "retryable": False,
        "skipped_reason": None,
        "tool_activity": None,
    }


def _remember_retry_state(job_id: str, state: Dict[str, Any]) -> None:
    if not job_id:
        return
    _LAST_RETRY_STATE[job_id] = dict(state)
    if len(_LAST_RETRY_STATE) <= _RETRY_STATE_LIMIT:
        return
    for key in list(_LAST_RETRY_STATE)[: len(_LAST_RETRY_STATE) - _RETRY_STATE_LIMIT]:
        _LAST_RETRY_STATE.pop(key, None)


def _failure_metadata(job_id: str, error: Optional[str]) -> Dict[str, Any]:
    code, retryable = _classify_failure(error)
    failure: Dict[str, Any] = {"code": code, "retryable": retryable}
    state = _LAST_RETRY_STATE.get(job_id) or {}
    if retryable and state:
        retry = {
            "attempts": int(state.get("attempts") or 0),
            "max_attempts": int(state.get("max_attempts") or 0),
        }
        skipped_reason = state.get("skipped_reason")
        if skipped_reason:
            retry["skipped_reason"] = skipped_reason
        if "tool_activity" in state:
            retry["tool_activity"] = state.get("tool_activity")
        failure["retry"] = retry
    return failure


def _is_retryable_error(error: Optional[str]) -> bool:
    code, retryable = _classify_failure(error)
    return retryable and code in {
        "network_unavailable",
        "upstream_unavailable",
        "timeout",
        "rate_limited",
    }


def _classify_failure(error: Optional[str]) -> tuple[str, bool]:
    """Raw error → stable user-facing code plus retryability."""
    if not error or not error.strip():
        return ("unknown", False)
    # Empty-result BEFORE retryable: upstream's #8585 sentinel ("...produced
    # empty response (model error, timeout, or misconfiguration)") contains
    # "timeout", which would otherwise be misread as transient/retryable.
    if _EMPTY_RESULT_RE.search(error):
        return ("empty_response", False)
    if _INSUFFICIENT_CREDITS_RE.search(error):
        return ("insufficient_credits", False)
    if _NETWORK_ERROR_RE.search(error):
        return ("network_unavailable", True)
    if _RATE_LIMIT_ERROR_RE.search(error):
        return ("rate_limited", True)
    if _UPSTREAM_ERROR_RE.search(error):
        return ("upstream_unavailable", True)
    if _TIMEOUT_ERROR_RE.search(error):
        return ("timeout", True)
    return ("agent_error", False)


def _friendly_failure(
    job_name: str,
    error: Optional[str],
    retry_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Default-locale failure line; never leaks the raw error."""
    name = (job_name or "").strip() or "定时任务"
    code, _ = _classify_failure(error)
    reason = _FAILURE_REASON[code]
    if retry_state and code in {
        "network_unavailable",
        "upstream_unavailable",
        "timeout",
        "rate_limited",
    }:
        skipped = retry_state.get("skipped_reason")
        attempts = int(retry_state.get("attempts") or 0)
        if skipped == "tool_activity":
            reason = f"{reason}。本次可能已执行部分步骤，为避免重复操作未自动重试"
        elif skipped == "activity_unknown":
            reason = f"{reason}。无法确认本次是否已执行操作，已跳过自动重试"
        elif skipped == "retry_exhausted":
            reason = f"{reason}。已自动重试 {attempts} 次仍失败"
        elif skipped in {"no_agent", "job_script"}:
            reason = f"{reason}。该任务包含脚本步骤，为避免重复副作用未自动重试"
        elif skipped == "retries_disabled":
            reason = f"{reason}。自动重试当前已关闭"
        elif skipped == "shutdown":
            reason = f"{reason}。系统正在停止，已取消自动重试"
    return f"⚠️ 定时任务「{name}」执行失败：{reason}。"


def _redact_channel_failure(job: dict, content: Optional[str]):
    """De-identify upstream's raw failure template before channel delivery;
    pass anything else through unchanged.

    Detects the template by reconstructing the EXACT prefix upstream builds
    (``⚠️ Cron '<name>' failed:\\n``, name = ``job.get('name', job['id'])``)
    from this job's own name/id and slicing by its length — NOT a regex. Job
    names are stored verbatim (cron/jobs.py only end-strips), so an interior
    newline in the name would defeat a ``.*?`` regex and leak the raw error to
    channel bridges; an exact-prefix match handles it."""
    if not content:
        return content
    template_name = job.get("name", job.get("id", ""))
    for prefix in (
        f"⚠️ Cron '{template_name}' failed:\n",
        f"⚠️ Cron job '{template_name}' failed:\n",
    ):
        if content.startswith(prefix):
            return _friendly_failure(job.get("name", ""), content[len(prefix):])
    return content


def _is_retryable_failure_result(result) -> bool:
    """True only for a well-formed FAILED run whose error reads transient.

    Defensive on the upstream contract: ``run_job`` returns
    ``(success, doc, output, error)``, but if a future sync ever returns a
    shorter tuple we treat it as non-retryable rather than let an IndexError
    propagate out and crash the scheduler's job-runner thread."""
    if not isinstance(result, (tuple, list)) or len(result) < 4:
        return False
    return not result[0] and _is_retryable_error(result[3])


def _run_job_with_retry(orig_run_job, job, *, before_retry=None):
    """Re-run a transient failure, but only while this run has produced zero
    tool activity (vs a fixed pre-run baseline) so side effects never repeat.

    ``before_retry`` runs only after the backoff completes and a new attempt is
    definitely about to start. The installed scheduler wrapper uses it to
    release the previous failed attempt's deferred agent while retaining the
    final attempt through delivery.
    """
    job_id = job.get("id", "")
    state = _new_retry_state()
    _LAST_RETRY_STATE.pop(job_id, None)
    # Run once (never retry) when retries are off, or when the job has side
    # effects the tool-activity guard can't see:
    #   • _MAX_RUN_RETRIES == 0 — retries disabled; also skips the baseline DB
    #     query on every cron tick for deployments that set RETRY_MAX=0.
    #   • no_agent / script jobs — a pre-run script (cron/scheduler.py runs
    #     job["script"] BEFORE the LLM, for both no_agent and ordinary agent
    #     jobs) can write files / call webhooks / rotate state. That work is not
    #     Hermes tool activity, so _attempt_tool_activity would read 0 ("safe")
    #     and a retry after an LLM 502 would re-execute the script's side effects.
    if _MAX_RUN_RETRIES == 0 or job.get("no_agent") or job.get("script"):
        result = orig_run_job(job)
        if _is_retryable_failure_result(result):
            state["retryable"] = True
            if _MAX_RUN_RETRIES == 0:
                state["skipped_reason"] = "retries_disabled"
            elif job.get("no_agent"):
                state["skipped_reason"] = "no_agent"
            else:
                state["skipped_reason"] = "job_script"
            _remember_retry_state(job_id, state)
        return result
    # Pre-run id snapshot, captured ONCE (ids only — no message reads). The guard
    # counts tools only in sessions absent from this set, so it never re-reads
    # the job's history and a reused same-second session id stays counted.
    baseline = _list_cron_session_ids(job_id)
    result = orig_run_job(job)
    attempts = 0
    # _is_retryable_failure_result bounds-checks the tuple; result[3] is the
    # error for agent jobs (script jobs are short-circuited above).
    while attempts < _MAX_RUN_RETRIES and _is_retryable_failure_result(result):
        state["retryable"] = True
        activity = _attempt_tool_activity(job_id, baseline)
        state["tool_activity"] = activity
        if activity != 0:  # None (unresolved) or >0 (work ran) → don't repeat
            state["skipped_reason"] = (
                "activity_unknown" if activity is None else "tool_activity"
            )
            _dbg(f"run_job: skip retry job={job_id} run_activity={activity}")
            break
        attempts += 1
        state["attempts"] = attempts
        _dbg(
            f"run_job: transient failure job={job_id} "
            f"retry {attempts}/{_MAX_RUN_RETRIES} after {_RETRY_BACKOFF_S}s"
        )
        # Interruptible backoff: True iff shutdown signalled mid-wait.
        if _shutdown.wait(_RETRY_BACKOFF_S):
            state["skipped_reason"] = "shutdown"
            _dbg(f"run_job: shutdown during backoff — abort retry job={job_id}")
            break
        if before_retry is not None:
            before_retry()
        result = orig_run_job(job)
    if _is_retryable_failure_result(result):
        state["retryable"] = True
        if not state.get("skipped_reason") and attempts >= _MAX_RUN_RETRIES:
            state["skipped_reason"] = "retry_exhausted"
        _remember_retry_state(job_id, state)
    return result


# ── Persist to hermes SessionDB ─────────────────────────────────────

def _try_persist_to_session(
    job_id: str,
    success: bool,
    error: Optional[str],
    delivery_error: Optional[str],
    job_snapshot: Optional[dict],
) -> Optional[str]:
    """Append cron summary as an assistant message to the originating chat's
    hermes SessionDB. APP's regular hermes /history call surfaces it.

    Skips when:
      - the run was silent (no_agent emitted nothing / agent replied [SILENT]) —
        scheduler already skipped its push, so we skip the App card to match;
        per-run md is still on disk
      - job has no origin (deliver=local or job created via REST without
        origin) — cron output md is on disk, user can review there
      - job_snapshot missing AND get_job returns None (job already gone +
        no pre-mark snapshot — extremely rare race)
    """
    if _is_silent_run(job_id):
        _dbg(f"_try_persist: job {job_id} silent run, skip card")
        return None

    job = job_snapshot
    if job is None:
        try:
            from cron.jobs import get_job
            job = get_job(job_id)
        except Exception:
            job = None
    if not job:
        _dbg(f"_try_persist: job {job_id} unavailable, skip")
        return None

    origin_chat_id = _resolve_zet_agent_chat_id(job)
    if not origin_chat_id:
        _dbg(f"_try_persist: job {job_id} no origin.chat_id, skip (deliver={job.get('deliver')!r})")
        return None

    # Self-heal jobs written by the old multiplex binding.  Persistence can
    # proceed even if the best-effort rewrite fails, because resolution above
    # already produced the public SessionDB id for this run.
    origin = job.get("origin") or {}
    if isinstance(origin, dict):
        stored_chat_id = str(origin.get("chat_id", "") or "").strip()
        if stored_chat_id and stored_chat_id != origin_chat_id:
            normalized = _normalize_zet_agent_chat_id(stored_chat_id)
            if normalized == origin_chat_id:
                try:
                    from cron.jobs import update_job

                    healed_origin = dict(origin)
                    healed_origin["chat_id"] = origin_chat_id
                    update_job(job_id, {"origin": healed_origin})
                    _dbg(
                        f"_try_persist: healed scoped origin for job {job_id} "
                        f"to {origin_chat_id}"
                    )
                except Exception as exc:
                    _dbg(f"_try_persist: heal job.origin FAILED: {exc!r}")

    try:
        from hermes_state import SessionDB
    except ImportError as _ie:
        _dbg(f"_try_persist: SessionDB ImportError: {_ie}")
        return f"zet_agent session persist unavailable: {_ie}"

    # 落库跟着会话走：get_hermes_home() 的 profile override 在 cron 线程里不一定绑上
    db_path, unresolved = _resolve_persist_db_path(origin_chat_id)
    if unresolved:
        # fail-closed：归属未决时新建会话会永久改写 job.origin.chat_id，没有回滚路径
        _dbg(f"_try_persist: {unresolved}")
        logger.warning("cron persist: job=%s fail-closed: %s", job_id, unresolved)
        return f"zet_agent session persist deferred: {unresolved}"
    db = SessionDB(db_path=db_path)
    try:
        # deliver=origin 但源对话已被 App 删除：直接 append 会撞 messages→sessions
        # 外键、cron 输出静默丢失。改为新建一个同 user/agent 的承接会话，把本次及
        # 后续输出投到它，并在 cron-summary 打 origin_recreated 标记让 App 渲染
        # 本地化提示。再把 job.origin 重指过去，避免下周期反复新建。
        target_id = origin_chat_id
        origin_recreated = False
        if db.get_session(origin_chat_id) is None:
            if _is_calendar_reminders_session(origin_chat_id):
                db.create_session(origin_chat_id, source="zet_agent", user_id=_user_id_from(origin_chat_id))
                _dbg(f"_try_persist: created synthetic calendar reminder session {origin_chat_id}")
            else:
                new_id = _handoff_session_id(origin_chat_id)
                if not new_id:
                    raise RuntimeError(f"cannot derive handoff session for missing origin {origin_chat_id!r}")
                # source 与正常 App 会话一致（run_agent 用 platform 名），让承接会话
                # 跟用户手建的对话同档，避免别处按 source 的隐性差异。
                db.create_session(new_id, source="zet_agent", user_id=_user_id_from(origin_chat_id))
                target_id = new_id
                origin_recreated = True
                _dbg(f"_try_persist: origin {origin_chat_id} gone → handoff session {new_id}")
                logger.warning(
                    "cron persist: origin session %s not found in %s — derived handoff "
                    "session %s and rewriting job=%s origin.chat_id (irreversible)",
                    origin_chat_id, db_path, new_id, job_id,
                )
                try:
                    from cron.jobs import update_job
                    _new_origin = dict(job.get("origin") or {})
                    _new_origin["chat_id"] = new_id
                    update_job(job_id, {"origin": _new_origin})
                except Exception as _ue:
                    _dbg(f"_try_persist: rewrite job.origin FAILED: {_ue!r}")

        content = _build_typed_message_content(
            job, job_id, success, error, delivery_error, origin_recreated=origin_recreated
        )
        msg_id = db.append_message(
            session_id=target_id,
            role="assistant",
            content=content,
        )
        _dbg(f"_try_persist: appended msg_id={msg_id} session={target_id} recreated={origin_recreated}")
        # gateway.log line so "cron ran but chat empty" is a grep: db = the state.db it landed in.
        logger.info(
            "cron persist: job=%s session=%s db=%s msg_id=%s recreated=%s",
            job_id, target_id, getattr(db, "db_path", "?"), msg_id, origin_recreated,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    # PRD §6.3 在线实时显示：写完 SessionDB 立刻 POST 给 local-server，让
    # 当前在线的 chat WS 立即收到 message.appended 事件并插条。失败仅 log
    # —— SessionDB 已落盘，APP 下次进 chat 走 /history 兜底。
    _try_notify_chat_append(target_id, msg_id, content)
    return None


def _try_notify_chat_append(session_id: str, msg_id: int, content: str) -> None:
    url = _scoped_env("ZET_CHAT_APPEND_URL").strip()
    if not url:
        _dbg("_try_notify_chat_append: ZET_CHAT_APPEND_URL unset, skip")
        return
    agent_id = _scoped_env("ZET_AGENT_ID").strip()
    payload = {
        "agent_id": agent_id,
        "session_id": session_id,
        "msg_id": int(msg_id) if msg_id is not None else 0,
        "role": "assistant",
        "content": content,
        # App 据此识别 cron 消息：turn 忙时排队、turn done 后再插入对话流末尾。
        "kind": "cron_summary",
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
        # best-effort realtime nudge; summary is already persisted so /history recovers — log, don't fail delivery.
        logger.warning(
            "cron realtime chat-append failed (best-effort): session=%s msg_id=%s err=%r",
            session_id, msg_id, e,
        )


def _build_typed_message_content(
    job: dict,
    job_id: str,
    success: bool,
    error: Optional[str],
    delivery_error: Optional[str],
    origin_recreated: bool = False,
) -> str:
    """Build markdown content with a typed code fence.

    Format:

        ```cron-summary
        {"job_id":"...","name":"...","schedule":"...","prompt":"...",
         "deliver":{...},"repeat":{...},"last_run_result":"success",
         "scheduled_at":"...","delivery_error":"",
         "failure":{"code":"upstream_unavailable","retryable":true}}
        ```

    `failure` is set only on failed runs; clients localize off `code`.

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
    if job.get("source") == "calendar":
        metadata["source"] = "calendar"
        for _key in (
            "calendar_source_type",
            "calendar_source_instance_id",
            "calendar_source_platform",
            "calendar_provider",
            "calendar_connection_id",
            "calendar_id",
            "calendar_event_id",
            "calendar_series_id",
            "calendar_original_start",
            "calendar_etag",
            "content",
        ):
            _value = job.get(_key)
            if _value:
                metadata[_key] = _value
    # next_run_at 是绝对时刻（带 offset），App 据此把循环任务的展示时间换算到设备
    # 本地时区——绕开"cron 表达式按哪个时区写的"歧义（旧 job 存 UTC 表达式 +
    # timezone=None，按字面显示会差 8 小时）。timezone 一并带上：App 用它区分
    # 字面可信（tz 显式）还是要靠 next_run_at 兜底（tz 缺失）。
    next_run_at = job.get("next_run_at")
    if next_run_at:
        metadata["next_run_at"] = next_run_at
    job_tz = job.get("timezone")
    if job_tz:
        metadata["timezone"] = job_tz
    if delivery_error:
        metadata["delivery_error"] = delivery_error
    if origin_recreated:
        # 源对话已删、本会话是系统新建来承接 cron 输出的标记。App 暂不渲染横幅，
        # 仅作为数据标记保留（便于后续区分/排查这类承接会话）。
        metadata["origin_recreated"] = True

    if not success:
        # Structured failure for client-side i18n (clients localize off `code`).
        metadata["failure"] = _failure_metadata(job_id, error)

    attachments = _collect_produced_files(job_id, job)
    if attachments:
        metadata["attachments"] = attachments

    if success and job.get("source") == "calendar":
        body = str(job.get("content") or job.get("name") or "").strip()
    elif success:
        body = _extract_response_body(_LATEST_OUTPUT.get(job_id, ""))
    else:
        # Friendly fallback; raw FAILED doc stays in the run .md.
        body = _friendly_failure(job.get("name", ""), error, _LAST_RETRY_STATE.get(job_id))

    parts = [
        "```cron-summary",
        json.dumps(metadata, ensure_ascii=False),
        "```",
    ]
    if body:
        parts.append("")  # blank line
        parts.append(body)

    return "\n".join(parts)


def _collect_produced_files(job_id: str, job: Optional[dict] = None) -> List[Dict[str, Any]]:
    """Enumerate files produced by this job's most recent cron session.

    The authoritative source is the session's agent-scoped output bucket,
    filtered to this run's time window. Write/Edit-class tool calls are used as
    hints and as a direct fast path only after the same output-root validation.
    This catches terminal/execute_code side-effect deliverables while refusing
    helper scripts, temp files, and old files in a reused shared bucket.

    Why look up the cron session by prefix instead of capturing it at
    save_job_output time: hermes upstream owns cron/scheduler.py and we
    avoid editing it to keep merge churn down. The cron session id format
    `cron_{job_id}_{YYYYMMDD_HHMMSS}` is stable enough to query by.

    Silent degradation: any failure (no SessionDB, no matching session,
    unreadable path, malformed tool_calls) yields an empty list — the cron
    summary still ships without an attachments field.
    """
    if not job_id:
        return []
    try:
        db = _cron_session_db()  # 换 profile 库 → 查不到 cron session → cron 附件永远为空
    except ImportError as _ie:
        _dbg(f"_collect_produced_files: SessionDB ImportError: {_ie}")
        return []
    try:
        try:
            with db._lock:
                cursor = db._conn.execute(
                    "SELECT id, started_at FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                    "ORDER BY started_at DESC LIMIT 1",
                    (_cron_session_like(job_id),),
                )
                row = cursor.fetchone()
        except Exception as _e:
            _dbg(f"_collect_produced_files: cron session lookup FAILED: {_e!r}")
            return []
        if not row:
            _dbg(
                f"_collect_produced_files: no cron session matching prefix=cron_{job_id}_"
            )
            return []
        sid = row["id"]
        try:
            started_at = float(row["started_at"])
        except (TypeError, ValueError):
            started_at = None
        try:
            messages = db.get_messages(sid)
        except Exception as _e:
            _dbg(f"_collect_produced_files: get_messages FAILED sid={sid}: {_e!r}")
            return []
    finally:
        try:
            db.close()
        except Exception:
            pass

    agent_ids = _cron_agent_ids(job)
    path_hints = _cron_path_hints(messages)
    output_roots = _cron_output_bucket_roots(path_hints, agent_ids)
    anchored = _anchored_session_root(job)
    if anchored is not None and all(anchored != r for r in output_roots):
        output_roots.append(anchored)

    seen: set[str] = set()
    produced: List[Dict[str, Any]] = []
    min_mtime = None
    if started_at is not None:
        min_mtime = max(0.0, started_at - _CRON_ATTACHMENT_MTIME_SLACK_S)

    def _add(raw_path: str, *, require_mtime: bool, allow_external: bool = False) -> None:
        if len(produced) >= _CRON_ATTACHMENT_LIMIT:
            return
        item = _cron_attachment_for_path(
            raw_path,
            output_roots,
            seen,
            min_mtime if require_mtime else None,
            allow_external=allow_external,
        )
        if item:
            produced.append(item)

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            # Schema in SessionDB.append_message (run_agent.py:4011-4015):
            # {"name": "...", "arguments": "..."}. Older snapshots may still
            # carry OpenAI-shaped {"function": {"name", "arguments"}} —
            # accept both rather than gate on a schema version.
            name = tc.get("name") or ((tc.get("function") or {}).get("name") or "")
            if name not in _PRODUCE_TOOL_NAMES:
                continue
            args_raw = tc.get("arguments")
            if args_raw is None:
                args_raw = (tc.get("function") or {}).get("arguments")
            if not args_raw:
                continue
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            else:
                args = args_raw
            if not isinstance(args, dict):
                continue
            path = ""
            for k in _PATH_KEYS:
                v = args.get(k)
                if v:
                    path = str(v).strip()
                    break
            if path:
                _add(path, require_mtime=True, allow_external=True)
    for root in output_roots:
        if len(produced) >= _CRON_ATTACHMENT_LIMIT:
            break
        for path in _iter_cron_output_files(root):
            _add(str(path), require_mtime=True)
            if len(produced) >= _CRON_ATTACHMENT_LIMIT:
                break
    for path in path_hints:
        _add(path, require_mtime=True, allow_external=True)
        if len(produced) >= _CRON_ATTACHMENT_LIMIT:
            break
    return produced


def _cron_agent_ids(job: Optional[dict]) -> set[str]:
    ids: set[str] = set()
    env_agent_id = _scoped_env("ZET_AGENT_ID").strip()
    if env_agent_id:
        ids.add(env_agent_id)
    origin = (job or {}).get("origin") if isinstance(job, dict) else None
    chat_id = ""
    if isinstance(origin, dict):
        chat_id = str(origin.get("chat_id", "") or "")
    parts = chat_id.split(":", 3)
    if len(parts) == 4 and parts[0] == "zettlab" and parts[2]:
        ids.add(parts[2])
    return ids


def _cron_path_hints(messages: Iterable[dict]) -> List[str]:
    hints: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for value in (msg.get("content"), msg.get("tool_calls")):
            _extend_path_hints(hints, value)
            if len(hints) >= 256:
                return hints
    return hints


def _extend_path_hints(out: List[str], value: Any) -> None:
    if value in (None, "", [], {}):
        return
    try:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    for match in _ABSOLUTE_PATH_RE.finditer(text[:200_000]):
        raw = _clean_cron_path_hint(match.group(0))
        if raw:
            out.append(raw)
            if len(out) >= 256:
                return


def _clean_cron_path_hint(raw: str) -> str:
    value = (raw or "").strip().strip("`*.,;:，。；：")
    value = value.rstrip("\\")
    if value.startswith("//") and not value.startswith("///"):
        return ""
    if value.startswith("///"):
        value = "/" + value.lstrip("/")
    return value


def _anchored_session_root(job: Optional[dict]) -> Optional[Path]:
    """run_job 锚定的 session 桶（与其同一纯函数重建），无 hint 也能扫到。

    纯相对路径写入不会在消息里留下绝对路径 hint，桶扫描就不会启动；这里用
    origin.chat_id + 平台目录确定性重建同一个桶。只收 session 桶，agent 根
    （suffix 不可推导时的回落值）不做扫描根。multiplex 下 delivery 阶段无
    secret scope 时平台目录不可得，静默退回 hint 行为。
    """
    try:
        from tools.runtime_workdir import (
            agent_output_dir,
            prepare_cron_session_output_dir,
        )

        base = agent_output_dir()
        if not base:
            return None
        anchor = prepare_cron_session_output_dir(
            ((job or {}).get("origin") or {}).get("chat_id")
        )
        if not anchor or anchor == base:
            return None
        return Path(anchor).resolve()
    except Exception:
        return None


def _cron_output_bucket_roots(path_hints: Iterable[str], agent_ids: set[str]) -> List[Path]:
    if not agent_ids:
        return []
    roots: List[Path] = []
    seen: set[str] = set()
    for raw in path_hints:
        root = _cron_output_bucket_root(raw, agent_ids)
        if root is None:
            continue
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
        if len(roots) >= 8:
            break
    return roots


def _cron_output_bucket_root(raw_path: str, agent_ids: set[str]) -> Optional[Path]:
    path = _clean_cron_path_hint(raw_path)
    if not path.startswith("/"):
        return None
    parts = Path(path).parts
    if "agents" not in parts or "data" not in parts or "output" not in parts:
        return None
    try:
        agents_idx = parts.index("agents")
        data_idx = agents_idx + 1
        if parts[data_idx] != "data":
            return None
        output_idx = parts.index("output", data_idx + 1)
    except (IndexError, ValueError):
        return None
    if output_idx + 1 >= len(parts):
        return None
    agent_id = parts[data_idx + 1] if data_idx + 1 < len(parts) else ""
    if agent_ids and agent_id not in agent_ids:
        return None
    root = Path(*parts[:output_idx + 2])
    try:
        st = os.stat(root)
    except OSError:
        return None
    if not _stat.S_ISDIR(st.st_mode):
        return None
    return root


def _iter_cron_output_files(root: Path) -> Iterable[Path]:
    scanned = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            scanned += 1
            if scanned > _CRON_ATTACHMENT_SCAN_FILE_LIMIT:
                return
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def _cron_attachment_for_path(
    raw_path: str,
    roots: List[Path],
    seen: set[str],
    min_mtime: Optional[float],
    *,
    allow_external: bool = False,
) -> Optional[Dict[str, Any]]:
    path = _clean_cron_path_hint(raw_path)
    if not path.startswith("/"):
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not _stat.S_ISREG(st.st_mode):
        return None
    if min_mtime is not None and st.st_mtime < min_mtime:
        return None
    if Path(path).suffix.lower() in _CRON_ATTACHMENT_EXCLUDED_SUFFIXES:
        return None
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    under_output_root = _is_under_any_root(resolved, roots)
    if not under_output_root and not (
        allow_external and _is_external_cron_deliverable(resolved)
    ):
        return None
    key = str(resolved)
    if key in seen:
        return None
    seen.add(key)
    mime, _enc = mimetypes.guess_type(str(resolved))
    return {
        "path": str(resolved),
        "name": resolved.name,
        "size": st.st_size,
        "mime": mime or "application/octet-stream",
    }


def _is_external_cron_deliverable(path: Path) -> bool:
    """Allow user-facing files created outside the canonical output bucket.

    This intentionally admits reports like /root/nas-youtube-report.html while
    keeping temporary scripts, sandbox scratch files, and extensionless system
    paths out of chat attachments.
    """
    suffix = path.suffix.lower()
    if suffix not in _CRON_ATTACHMENT_EXTERNAL_SUFFIXES:
        return False
    if _looks_like_agent_output_path(path):
        return False
    path_s = str(path)
    for marker in ("/hermes_sandbox_", "/hermes_exec_"):
        if marker in path_s:
            return False
    try:
        parts = path.relative_to(path.anchor).parts
    except ValueError:
        parts = path.parts
    if parts:
        first = parts[0]
        second = f"{parts[0]}/{parts[1]}" if len(parts) > 1 else ""
        if first in _CRON_ATTACHMENT_TEMP_DIRS or second in _CRON_ATTACHMENT_TEMP_DIRS:
            return False
    return True


def _looks_like_agent_output_path(path: Path) -> bool:
    parts = path.parts
    for i, part in enumerate(parts):
        if part != "agents":
            continue
        if i + 3 < len(parts) and parts[i + 1] == "data" and parts[i + 3] == "output":
            return True
    return False


def _is_under_any_root(path: Path, roots: List[Path]) -> bool:
    path_s = str(path)
    for root in roots:
        root_s = str(root)
        try:
            if os.path.commonpath([root_s, path_s]) == root_s:
                return True
        except ValueError:
            continue
    return False


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
