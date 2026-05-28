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
import mimetypes
import os
import re
import stat as _stat
from typing import Any, Dict, List, Optional

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


def _is_zet_agent_platform(platform: Any) -> bool:
    return str(platform or "").lower() in {"zet_agent", "zettlab"}


def _target_to_deliver_value(target: dict) -> str:
    value = f"{target.get('platform')}:{target.get('chat_id')}"
    thread_id = target.get("thread_id")
    if thread_id is not None:
        value += f":{thread_id}"
    return value


def _resolve_zet_agent_chat_id(job: dict) -> str:
    origin = job.get("origin") or {}
    if isinstance(origin, dict):
        chat_id = str(origin.get("chat_id", "") or "").strip()
        platform = origin.get("platform")
        if chat_id and (not platform or _is_zet_agent_platform(platform)):
            return chat_id
    try:
        import cron.scheduler as _sched
        for target in _sched._resolve_delivery_targets(job):
            if _is_zet_agent_platform(target.get("platform")):
                return str(target.get("chat_id", "") or "").strip()
    except Exception as _e:
        _dbg(f"_resolve_zet_agent_chat_id: target resolution FAILED: {_e!r}")
    return ""


def _combine_delivery_errors(existing: Optional[str], added: str) -> str:
    if existing:
        return f"{existing}; {added}"
    return added


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
        saved = _orig_save(job_id, output)
        _LATEST_OUTPUT_PATH[job_id] = saved
        return saved

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
                job_id, success, error, delivery_error=effective_delivery_error
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
                            return None
                        job = dict(job)
                        job["origin"] = None
                        job["deliver"] = ",".join(
                            _target_to_deliver_value(t) for t in other_targets
                        )
                except Exception as _e:
                    _dbg(f"_deliver_result patch pre-check FAILED: {_e!r}")
                return _orig_deliver_result(job, content, adapters=adapters, loop=loop)

            setattr(_wrapped_deliver_result, _PATCH_SENTINEL, True)
            _sched._deliver_result = _wrapped_deliver_result
            _dbg("install() patched scheduler._deliver_result OK")
    except Exception as _e:
        _dbg(f"install() scheduler delivery patch FAILED: {_e!r}")

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
                            platform="zet_agent",
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


def _count_tool_activity(job_id: str) -> Optional[int]:
    """最近一次 cron session 的 tool_call + tool 结果数；读不到返回 None（fail-open）。"""
    if not job_id:
        return None
    try:
        from hermes_state import SessionDB
    except ImportError:
        return None
    db = SessionDB()
    try:
        prefix = f"cron_{job_id}_"
        like = (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
        )
        try:
            with db._lock:
                cursor = db._conn.execute(
                    "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                    "ORDER BY started_at DESC LIMIT 1",
                    (like,),
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
    finally:
        try:
            db.close()
        except Exception:
            pass


def _detect_fake_success(job_id: str) -> Optional[str]:
    """短意图句 + 本轮 0 工具执行 → 返回失败原因，否则 None（fail-open，不误降真成功）。"""
    try:
        body = _extract_response_body(_LATEST_OUTPUT.get(job_id, "")).strip()
    except Exception:
        return None
    if not body:
        return None  # 空 body 上游已 soft-fail (#8585)
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
        return None

    origin_chat_id = _resolve_zet_agent_chat_id(job)
    if not origin_chat_id:
        _dbg(f"_try_persist: job {job_id} no origin.chat_id, skip (deliver={job.get('deliver')!r})")
        return None

    content = _build_typed_message_content(job, job_id, success, error, delivery_error)
    _dbg(f"_try_persist: appending to session={origin_chat_id} job={job_id} content_len={len(content)}")

    try:
        from hermes_state import SessionDB
    except ImportError as _ie:
        _dbg(f"_try_persist: SessionDB ImportError: {_ie}")
        return f"zet_agent session persist unavailable: {_ie}"

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
    return None


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

    attachments = _collect_produced_files(job_id)
    if attachments:
        metadata["attachments"] = attachments

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


def _collect_produced_files(job_id: str) -> List[Dict[str, Any]]:
    """Enumerate files written by Write/Edit-class tool calls in this job's
    most recent cron session. Result is embedded into cron-summary metadata
    so downstream clients (App, channel bridges) can render attachments
    without re-deriving them from raw history.

    Why look up the cron session by prefix instead of capturing it at
    save_job_output time: hermes upstream owns cron/scheduler.py and we
    avoid editing it to keep merge churn down. The cron session id format
    `cron_{job_id}_{YYYYMMDD_HHMMSS}` is stable enough to query by.

    Silent degradation: any failure (no SessionDB, no matching session,
    unreadable path, malformed tool_calls) yields an empty list — the
    cron summary still ships without an attachments field.
    """
    if not job_id:
        return []
    try:
        from hermes_state import SessionDB
    except ImportError as _ie:
        _dbg(f"_collect_produced_files: SessionDB ImportError: {_ie}")
        return []

    db = SessionDB()
    try:
        prefix = f"cron_{job_id}_"
        like = (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
        )
        try:
            with db._lock:
                cursor = db._conn.execute(
                    "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' "
                    "ORDER BY started_at DESC LIMIT 1",
                    (like,),
                )
                row = cursor.fetchone()
        except Exception as _e:
            _dbg(f"_collect_produced_files: cron session lookup FAILED: {_e!r}")
            return []
        if not row:
            _dbg(
                f"_collect_produced_files: no cron session matching prefix={prefix}"
            )
            return []
        sid = row["id"]
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

    seen: set = set()
    produced: List[Dict[str, Any]] = []
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
            if not path or path in seen:
                continue
            seen.add(path)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if not _stat.S_ISREG(st.st_mode):
                continue
            mime, _enc = mimetypes.guess_type(path)
            produced.append({
                "path": path,
                "name": os.path.basename(path),
                "size": st.st_size,
                "mime": mime or "application/octet-stream",
            })
    return produced


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
