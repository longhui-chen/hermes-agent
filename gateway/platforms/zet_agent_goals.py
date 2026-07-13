"""Zettlab goal-loop driver — the zet_agent host for ``hermes_cli.goals``.

This is the fifth host driver for the persistent-goal core (precedents:
CLI ``process_loop``, ``gateway/run._post_turn_goal_continuation``,
``tui_gateway/server``, ``hermes_cli.goals.run_kanban_goal_loop``). The
core (``GoalManager`` / ``judge_goal`` / SessionDB ``state_meta``
persistence) is reused verbatim; only the drive mechanics differ:

    App → local-server /ws/chat → POST /v1/chat/completions   (each round)
                 ▲                        │
                 │                        ▼  post-turn hook (this module)
        POST ZET_GOAL_ADVANCE_URL  ◄──  evaluate_after_turn()

Because the chat-completions surface is stateless HTTP (no resident agent,
no message queue), the "fire the next turn" step cannot be an in-process
loop. Instead every turn's outcome is REPORTED to zettlab-local-server's
loopback endpoint (``ZET_GOAL_ADVANCE_URL``, mirroring the cron
``ZET_CHAT_APPEND_URL`` channel); when the judge says *continue*, the
report carries the continuation prompt and local-server starts the next
round as a first-class turn. Every round therefore gets streaming, tool
events, cancellation and persistence for free, and each round enjoys its
own MaxTurnDuration budget.

Identity sidecar: ``GoalState`` has no goal-id concept (goals are keyed by
session), and ``hermes_cli/goals.py`` is upstream code we must not modify.
The wire protocol's ``goal_id`` lives in our own ``state_meta`` keys
(``zet_goal:{session_id}`` + a ``zet_goal_index`` list for restart
reconcile) via the public ``SessionDB.get_meta``/``set_meta`` API only.

Failure model (HR#2): every callback POST is fail-open — goal state is
already durable in ``state_meta`` before any report is attempted, so a
lost callback degrades to "the loop stalls until the next user message,
manual resume, or the reconcile-on-start pass after a respawn".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# All GoalManager continuation prompt templates share this prefix — used to
# tell a self-fed continuation turn from a genuine user turn (the judge's
# ``user_initiated`` flag).
CONTINUATION_MARKER = "[Continuing toward"

_SIDECAR_KEY = "zet_goal:{sid}"
_INDEX_KEY = "zet_goal_index"

_REPORT_TIMEOUT_S = 5.0
_RECONCILE_STARTUP_DELAY_S = 5.0
# Barrier wakeups poll at most this often for pid/session barriers whose
# clear time we cannot predict.
_BARRIER_POLL_S = 30.0


def _session_db():
    """Return the (per-HERMES_HOME cached) SessionDB used by the goal core.

    Reuses ``hermes_cli.goals._get_session_db`` so the sidecar rows land in
    the same sqlite file as the goal rows themselves; falls back to a plain
    ``SessionDB()`` if the private helper moves upstream.
    """
    try:
        from hermes_cli.goals import _get_session_db
        return _get_session_db()
    except Exception:
        from hermes_state import SessionDB
        return SessionDB()


def _scoped_env(name: str, default: str = "") -> str:
    """Env resolution that honours multiplex profile scopes (see
    ``zet_agent_cron._scoped_env`` for the full rationale)."""
    try:
        from gateway.platforms.zet_agent_cron import _scoped_env as scoped
        return scoped(name, default)
    except Exception:
        import os
        return os.environ.get(name, default)


class ZetGoalDriver:
    """Owns goal HTTP routes, the post-turn evaluation hook, interrupt /
    interaction projections, barrier wakeups and restart reconcile for one
    ``ZetAgentAdapter`` instance."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._lock = threading.Lock()
        # session_id → threading.Timer for barrier wakeups (bounded: one per
        # waiting goal; cancelled/replaced on every reschedule).
        self._barrier_timers: Dict[str, threading.Timer] = {}
        # session_id → threading.Lock serialising every GoalManager
        # read-modify-write for that session. SessionDB writes are atomic
        # per call, but "load state → judge → save" spans several calls;
        # without this, a user pause landing mid-evaluate gets clobbered by
        # evaluate's own save (the goal would crawl back up after an
        # explicit stop). Bounded: one entry per session that ever ran a
        # goal in this process; pruned on clear/done.
        self._session_locks: Dict[str, threading.Lock] = {}
        # session_id → wall-clock time of the user's stop press. Consulted
        # (and consumed) by the post-turn hook so a cancel that lands while
        # the judge is already evaluating still wins. TTL'd so a stale mark
        # can never pause a future goal round.
        self._cancel_marks: Dict[str, float] = {}

    _CANCEL_MARK_TTL_S = 180.0

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lk = self._session_locks.get(session_id)
            if lk is None:
                lk = threading.Lock()
                self._session_locks[session_id] = lk
            return lk

    def _prune_session_lock(self, session_id: str) -> None:
        with self._lock:
            self._session_locks.pop(session_id, None)
            self._cancel_marks.pop(session_id, None)

    def _mark_user_cancel(self, session_id: str) -> None:
        with self._lock:
            self._cancel_marks[session_id] = time.time()

    def _consume_user_cancel(self, session_id: str) -> bool:
        with self._lock:
            at = self._cancel_marks.pop(session_id, None)
        return at is not None and (time.time() - at) <= self._CANCEL_MARK_TTL_S

    def _spawn(self, fn, *args, **kwargs) -> None:
        """Run fn on a daemon thread WITH the caller's contextvars.

        Plain threading.Thread/Timer does NOT inherit contextvars — under the
        multiplex gateway the profile secret scope lives in ContextVars, so a
        bare thread's ``_scoped_env`` resolves to "" (fail-closed) and every
        report is silently dropped. copy_context().run keeps the scope."""
        import contextvars
        import functools

        ctx = contextvars.copy_context()
        t = threading.Thread(
            target=ctx.run,
            args=(functools.partial(fn, *args, **kwargs),),
            daemon=True,
            name="zet-goal",
        )
        t.start()

    # ------------------------------------------------------------------
    # Sidecar (goal_id + restart index) — public SessionDB meta API only
    # ------------------------------------------------------------------

    def _load_sidecar(self, session_id: str) -> Dict[str, Any]:
        try:
            raw = _session_db().get_meta(_SIDECAR_KEY.format(sid=session_id))
            if raw:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.debug("[zet_goal] load sidecar failed", exc_info=True)
        return {}

    def _save_sidecar(self, session_id: str, data: Dict[str, Any]) -> None:
        try:
            _session_db().set_meta(_SIDECAR_KEY.format(sid=session_id), json.dumps(data))
        except Exception:
            logger.debug("[zet_goal] save sidecar failed", exc_info=True)

    def _index(self) -> List[str]:
        try:
            raw = _session_db().get_meta(_INDEX_KEY)
            if raw:
                data = json.loads(raw)
                if isinstance(data, list):
                    return [s for s in data if isinstance(s, str)]
        except Exception:
            logger.debug("[zet_goal] load index failed", exc_info=True)
        return []

    def _index_write(self, sids: List[str]) -> None:
        try:
            _session_db().set_meta(_INDEX_KEY, json.dumps(sids))
        except Exception:
            logger.debug("[zet_goal] write index failed", exc_info=True)

    def _index_add(self, session_id: str) -> None:
        with self._lock:
            sids = self._index()
            if session_id not in sids:
                sids.append(session_id)
                self._index_write(sids)

    def _index_remove(self, session_id: str) -> None:
        with self._lock:
            sids = self._index()
            if session_id in sids:
                sids.remove(session_id)
                self._index_write(sids)

    # ------------------------------------------------------------------
    # Projection — GoalState → wire dict (local-server goal.status shape)
    # ------------------------------------------------------------------

    def _cumulative_round(self, session_id: str, turns_used: Any, side: Optional[Dict[str, Any]] = None) -> int:
        """App-facing round number: budget resets (resume after "turn budget
        exhausted") zero GoalState.turns_used, but the user sees one
        continuous loop — accumulate pre-reset rounds in the sidecar so the
        displayed round stays monotonic."""
        if side is None:
            side = self._load_sidecar(session_id)
        try:
            offset = int(side.get("rounds_offset") or 0)
        except Exception:
            offset = 0
        try:
            return offset + int(turns_used)
        except Exception:
            return offset

    def projection(self, session_id: str, mgr: Any = None) -> Dict[str, Any]:
        from hermes_cli.goals import GoalManager
        if mgr is None:
            mgr = GoalManager(session_id)
        side = self._load_sidecar(session_id)
        # Fallback id for goals created by another host (CLI /goal): must be
        # STABLE across process restarts — hash() is per-process randomized
        # (PYTHONHASHSEED), so derive from md5 instead.
        goal_id = side.get("goal_id") or "g_cli_" + hashlib.md5(session_id.encode("utf-8")).hexdigest()[:10]
        st = mgr.state
        if st is None:
            return {
                "goal_id": goal_id,
                "session_id": session_id,
                "state": "cleared",
            }
        if st.status == "active":
            state = "waiting" if mgr.is_waiting() else "running"
        elif st.status in ("paused", "done", "cleared"):
            state = st.status
        else:
            state = st.status
        summary = st.last_reason or st.paused_reason or ""
        if st.status == "active" and mgr.is_waiting():
            summary = st.waiting_reason or summary
        return {
            "goal_id": goal_id,
            "session_id": session_id,
            "state": state,
            "round": self._cumulative_round(session_id, st.turns_used, side),
            "max_rounds": int(st.max_turns),
            "goal_text": st.goal,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Advance reporting (fail-open POST to local-server)
    # ------------------------------------------------------------------

    def _wire_session_id(self, session_id: str) -> str:
        """The session id local-server must key on: the STABLE app-level id
        recorded at create time. Hermes' own session id rotates on context
        compaction; reporting the rotated id would make local-server start
        the next round under a registry/transcript key the App never sees."""
        side = self._load_sidecar(session_id)
        app_sid = str(side.get("app_session_id") or "").strip()
        return app_sid or session_id

    def report(
        self,
        session_id: str,
        proj: Dict[str, Any],
        *,
        continuation: Optional[str] = None,
    ) -> None:
        url = _scoped_env("ZET_GOAL_ADVANCE_URL").strip()
        if not url:
            logger.debug("[zet_goal] ZET_GOAL_ADVANCE_URL unset, skip report")
            return
        payload = dict(proj)
        payload["agent_id"] = _scoped_env("ZET_AGENT_ID").strip()
        payload["session_id"] = self._wire_session_id(session_id)
        if continuation:
            payload["continuation"] = continuation
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_REPORT_TIMEOUT_S) as resp:
                resp.read(256)
        except Exception as e:
            # Fail-open: state_meta already holds the truth; reconcile or the
            # next user message re-drives the loop.
            logger.warning("[zet_goal] advance report failed: %r", e)

    def report_in_thread(self, session_id: str, proj: Dict[str, Any], **kw: Any) -> None:
        """POST from a short-lived daemon thread — used by callers on the
        agent thread (approval notify) that must not block on network.
        Context-preserving (see _spawn): mux profile scopes ride ContextVars."""
        self._spawn(self.report, session_id, proj, **kw)

    # ------------------------------------------------------------------
    # HTTP route — POST/GET /v1/sessions/{sid}/goal
    # ------------------------------------------------------------------

    async def handle_goal_route(self, request: Any) -> Any:
        from aiohttp import web

        auth_err = self.adapter._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info.get("session_id", "")
        if not session_id:
            return web.json_response({"error": "session_id required"}, status=400)

        if request.method == "GET":
            proj = await asyncio.to_thread(self.projection, session_id)
            return web.json_response(proj)

        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        action = str(body.get("action", "") or "").strip()
        if action not in ("create", "pause", "resume", "clear", "status"):
            return web.json_response(
                {"error": "action must be create|pause|resume|clear|status"}, status=400
            )
        try:
            proj = await asyncio.to_thread(self._apply_action_sync, session_id, action, body)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            logger.warning("[zet_goal] %s failed for %s: %r", action, session_id, e)
            return web.json_response({"error": f"goal {action} failed: {e}"}, status=500)
        return web.json_response(proj)

    def _apply_action_sync(self, session_id: str, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
        from hermes_cli.goals import GoalManager, parse_contract

        with self._session_lock(session_id):
            mgr = GoalManager(session_id)
            if action == "create":
                text = str(body.get("text", "") or "").strip()
                if not text:
                    raise ValueError("text required for create")
                goal_id = str(body.get("goal_id", "") or "").strip() or f"g_{uuid.uuid4()}"
                max_rounds = body.get("max_rounds")
                try:
                    max_rounds = int(max_rounds) if max_rounds else None
                except (TypeError, ValueError):
                    max_rounds = None
                side = {"goal_id": goal_id, "created_at": time.time()}
                app_sid = str(body.get("app_session_id", "") or "").strip()
                if app_sid:
                    side["app_session_id"] = app_sid
                else:
                    # 缺 app_session_id 意味着 compaction 轮转后上报会漂移到
                    # 新 hermes sid，local-server 侧对不上 registry/transcript。
                    logger.warning("[zet_goal] create without app_session_id for %s", session_id)
                # Crash-ordering: sidecar + index FIRST, activation LAST — a
                # crash mid-create must leave "goal not yet active", never
                # "active but untracked" (an orphan reconcile can't see).
                self._save_sidecar(session_id, side)
                self._index_add(session_id)
                headline, contract = parse_contract(text)
                mgr.set(headline or text, max_turns=max_rounds, contract=contract)
            elif action == "pause":
                reason = str(body.get("reason", "") or "").strip() or "user-paused"
                mgr.pause(reason)
                self._cancel_barrier_timer(session_id)
            elif action == "resume":
                # 用户在第 N 轮暂停后点继续，必须从第 N+1 轮接着数 —— 上游
                # resume() 默认重置轮数预算（turns_used=0），会让 App 轮次
                # 显示跳回第 1 轮。只有"预算耗尽"的暂停才真正需要重置预算
                # （否则 resume 后立刻再次触发预算暂停），重置前把已用轮数
                # 累进 sidecar 偏移，App 侧轮次保持单调递增。
                st0 = mgr.state
                budget_paused = bool(
                    st0 is not None
                    and st0.status == "paused"
                    and "budget exhausted" in str(st0.paused_reason or "")
                )
                if budget_paused and int(getattr(st0, "turns_used", 0) or 0) > 0:
                    side = self._load_sidecar(session_id)
                    side["rounds_offset"] = self._cumulative_round(session_id, st0.turns_used, side)
                    self._save_sidecar(session_id, side)
                mgr.resume(reset_budget=budget_paused)
            elif action == "clear":
                self._cancel_barrier_timer(session_id)
                mgr.clear()
                self._index_remove(session_id)
                self._prune_session_lock(session_id)
                # Minimal terminal shape — consistent with projection()'s
                # no-goal branch so callers see one "cleared" contract.
                return self.projection(session_id)
            # "status" and mutations fall through to a fresh projection.
            proj = self.projection(session_id, mgr=mgr)
            if action == "resume" and proj.get("state") == "running":
                # A resume must actually restart the loop: hand local-server
                # the continuation so the next round fires without a user
                # message. Direct call — we're already off the event loop
                # (asyncio.to_thread) and the context carries the mux scope.
                cont = mgr.next_continuation_prompt()
                if cont:
                    proj = dict(proj)
                    proj["round"] = self._cumulative_round(session_id, mgr.state.turns_used) + 1
                    self.report(session_id, proj, continuation=cont)
            return proj

    # ------------------------------------------------------------------
    # Post-turn hook — the actual loop driver
    # ------------------------------------------------------------------

    def schedule_after_turn(
        self,
        session_id: str,
        user_message: str,
        final_response: str,
        effective_session_id: str = "",
    ) -> None:
        """Called from ``_run_agent`` right after a turn completes. Runs the
        judge off the event loop; cheap no-goal fast path inside.

        ``effective_session_id`` is the post-turn session id from the run
        result — it differs from ``session_id`` when context compaction
        rotated the session mid-turn (the goal row was migrated by
        ``conversation_compression``; the sidecar must follow)."""
        if not session_id:
            return
        try:
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._after_turn_sync,
                    session_id,
                    user_message or "",
                    final_response or "",
                    effective_session_id or "",
                )
            )
            tasks = getattr(self.adapter, "_background_tasks", None)
            if tasks is not None:
                try:
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                except Exception:
                    pass
        except RuntimeError:
            # No running loop (unit tests calling the sync path) — run inline.
            self._after_turn_sync(session_id, user_message or "", final_response or "", effective_session_id or "")

    def _migrate_sidecar(self, old_sid: str, new_sid: str) -> None:
        """Follow a compaction-driven session rotation: the goal row was
        already moved by ``migrate_goal_to_session``; move our sidecar +
        index entry alongside so goal_id / app_session_id survive."""
        side = self._load_sidecar(old_sid)
        if side:
            self._save_sidecar(new_sid, side)
        self._index_remove(old_sid)
        self._index_add(new_sid)

    def _after_turn_sync(
        self,
        session_id: str,
        user_message: str,
        final_response: str,
        effective_session_id: str = "",
    ) -> None:
        from hermes_cli.goals import GoalManager

        if effective_session_id and effective_session_id != session_id:
            self._migrate_sidecar(session_id, effective_session_id)
            session_id = effective_session_id
        try:
            with self._session_lock(session_id):
                # Consume a stop pressed BEFORE this hook ran: the pause may
                # have already landed (status!=active → early return below),
                # or we're first — either way the goal must not continue.
                cancelled_pre = self._consume_user_cancel(session_id)
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    logger.debug("[zet_goal] GoalManager init failed", exc_info=True)
                    return
                st = mgr.state
                if st is None or st.status != "active":
                    # 软暂停落在轮次运行中：这轮 continuation 已完整跑完、模型
                    # 预算已消耗，必须计轮并同步投影 —— 不计的话 resume 会用
                    # 同一轮次号重跑（用户视角：第二轮暂停，继续后又是第二轮）。
                    # 暂停期间的用户插话（无 marker）不计；stop 中断的轮次是
                    # 半成品，留给 resume 重做，也不计。
                    if (
                        st is not None
                        and st.status == "paused"
                        and user_message.startswith(CONTINUATION_MARKER)
                    ):
                        from hermes_cli.goals import save_goal

                        st.turns_used += 1
                        st.last_turn_at = time.time()
                        save_goal(session_id, st)
                        self.report(session_id, self.projection(session_id, mgr=mgr))
                    return
                # Self-heal the restart index: a goal created by another host
                # (CLI) or a missed create-path write must stay reconcilable.
                self._index_add(session_id)
                if cancelled_pre:
                    mgr.pause("user stopped the running turn")
                    self._cancel_barrier_timer(session_id)
                    self.report(session_id, self.projection(session_id, mgr=mgr))
                    return
                user_initiated = not user_message.startswith(CONTINUATION_MARKER)
                try:
                    decision = mgr.evaluate_after_turn(final_response, user_initiated=user_initiated)
                except Exception:
                    logger.warning("[zet_goal] evaluate_after_turn failed", exc_info=True)
                    return
                # A stop pressed WHILE the judge was evaluating: evaluate's own
                # save just wrote status=active over the (racing) pause — the
                # cancel mark is the tiebreaker that stops the crawl-back.
                if self._consume_user_cancel(session_id):
                    mgr.pause("user stopped the running turn")
                    self._cancel_barrier_timer(session_id)
                    self.report(session_id, self.projection(session_id, mgr=mgr))
                    return
                verdict = str(decision.get("verdict") or "")
                proj = self.projection(session_id, mgr=mgr)
                if decision.get("should_continue") and decision.get("continuation_prompt"):
                    proj["state"] = "running"
                    proj["round"] = self._cumulative_round(session_id, mgr.state.turns_used) + 1
                    proj["summary"] = str(decision.get("reason") or "")
                    self.report(session_id, proj, continuation=str(decision["continuation_prompt"]))
                    return
                if verdict == "done":
                    self._index_remove(session_id)
                    self._cancel_barrier_timer(session_id)
                    self._prune_session_lock(session_id)
                    self.report(session_id, proj)
                    return
                if verdict in ("wait", "waiting"):
                    proj["state"] = "waiting"
                    self.report(session_id, proj)
                    self._schedule_barrier_wakeup(session_id)
                    return
                # paused (budget / judge parse failures) or anything else: sync
                # the projection so the App banner shows the true state + reason.
                proj["summary"] = str(decision.get("message") or decision.get("reason") or proj.get("summary") or "")
                self.report(session_id, proj)
        except Exception:
            # Fire-and-forget task: an uncaught exception here would only
            # surface as asyncio "exception was never retrieved" noise.
            logger.warning("[zet_goal] post-turn hook failed", exc_info=True)

    # ------------------------------------------------------------------
    # Barrier wakeups — replace the CLI's ticking REPL for parked goals
    # ------------------------------------------------------------------

    def _cancel_barrier_timer(self, session_id: str) -> None:
        with self._lock:
            t = self._barrier_timers.pop(session_id, None)
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _schedule_barrier_wakeup(self, session_id: str) -> None:
        """Arm a timer that re-checks a parked goal's barrier. Time barriers
        wake exactly at the deadline; pid/session barriers poll at
        ``_BARRIER_POLL_S`` (the CLI equivalent is its always-running REPL
        tick — a gateway has no such loop, hence the timers)."""
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id)
        st = mgr.state
        if st is None or st.status != "active" or not mgr.is_waiting():
            return
        delay = _BARRIER_POLL_S
        if st.waiting_until and st.waiting_until > time.time():
            delay = max(1.0, st.waiting_until - time.time() + 1.0)
        # Timer callbacks run on a bare thread with NO contextvars — capture
        # the current context (mux profile scope) so the wakeup's report can
        # still resolve ZET_GOAL_ADVANCE_URL (see _spawn).
        import contextvars

        ctx = contextvars.copy_context()
        timer = threading.Timer(delay, lambda: ctx.run(self._barrier_wakeup, session_id))
        timer.daemon = True
        with self._lock:
            old = self._barrier_timers.pop(session_id, None)
            self._barrier_timers[session_id] = timer
        if old is not None:
            try:
                old.cancel()
            except Exception:
                pass
        timer.start()

    def _barrier_wakeup(self, session_id: str) -> None:
        from hermes_cli.goals import GoalManager

        with self._lock:
            self._barrier_timers.pop(session_id, None)
        try:
            with self._session_lock(session_id):
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    return
                st = mgr.state
                if st is None or st.status != "active":
                    return
                if mgr.is_waiting():
                    # Barrier still holding (pid/session) — keep polling.
                    self._schedule_barrier_wakeup(session_id)
                    return
                cont = mgr.next_continuation_prompt()
                if not cont:
                    return
                proj = self.projection(session_id, mgr=mgr)
                proj["state"] = "running"
                proj["round"] = self._cumulative_round(session_id, st.turns_used) + 1
                proj["summary"] = "wait barrier cleared; continuing"
                self.report(session_id, proj, continuation=cont)
        except Exception:
            logger.warning("[zet_goal] barrier wakeup failed", exc_info=True)

    # ------------------------------------------------------------------
    # Interrupt + interaction projections
    # ------------------------------------------------------------------

    def on_user_interrupt(self, session_id: str) -> None:
        """User pressed stop (local-server POSTs interrupt with
        reason=user_cancel): pause the goal so it never "crawls back up"
        after an explicit stop. Timeout/disconnect interrupts carry no such
        reason and leave the loop free to continue.

        Two-step: (1) drop a cancel mark synchronously — the post-turn hook
        consumes it even when the pause write below loses the race against
        evaluate_after_turn's own save; (2) do the pause + report on a
        context-preserving thread so the interrupt HTTP response is never
        blocked behind an in-flight judge call holding the session lock."""
        self._mark_user_cancel(session_id)
        self._spawn(self._pause_after_interrupt, session_id)

    def _pause_after_interrupt(self, session_id: str) -> None:
        from hermes_cli.goals import GoalManager

        try:
            with self._session_lock(session_id):
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    return
                if not mgr.is_active():
                    return
                mgr.pause("user stopped the running turn")
                self._cancel_barrier_timer(session_id)
                self.report(session_id, self.projection(session_id, mgr=mgr))
        except Exception:
            logger.debug("[zet_goal] pause after interrupt failed", exc_info=True)

    def on_interaction_pending(self, session_id: str) -> None:
        """An approval/clarify card is blocking the turn: project 'waiting'
        so the App banner explains the stall. GoalManager state itself is
        untouched — the turn is still running from the loop's viewpoint."""
        from hermes_cli.goals import GoalManager

        try:
            mgr = GoalManager(session_id)
        except Exception:
            return
        if not mgr.is_active():
            return
        proj = self.projection(session_id, mgr=mgr)
        proj["state"] = "waiting"
        proj["summary"] = "waiting for user confirmation"
        self.report_in_thread(session_id, proj)

    def on_interaction_resolved(self, session_id: str) -> None:
        from hermes_cli.goals import GoalManager

        try:
            mgr = GoalManager(session_id)
        except Exception:
            return
        if not mgr.is_active():
            return
        self.report_in_thread(session_id, self.projection(session_id, mgr=mgr))

    # ------------------------------------------------------------------
    # Reconcile-on-start (HR#2 pair of local-server's goal keepalive)
    # ------------------------------------------------------------------

    async def reconcile_on_start(self) -> None:
        try:
            await asyncio.sleep(_RECONCILE_STARTUP_DELAY_S)
            await asyncio.to_thread(self._reconcile_sync)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[zet_goal] reconcile-on-start failed", exc_info=True)

    def _reconcile_sync(self) -> None:
        from hermes_cli.goals import GoalManager

        for sid in self._index():
            try:
                mgr = GoalManager(sid)
            except Exception:
                continue
            st = mgr.state
            if st is None or st.status in ("cleared", "done"):
                self._index_remove(sid)
                continue
            proj = self.projection(sid, mgr=mgr)
            if st.status == "active" and not mgr.is_waiting():
                # A live loop was cut mid-flight (crash / OOM respawn).
                # Skip if this process already has an active turn for the
                # session — the loop is running, no kick needed.
                active = {}
                try:
                    with self.adapter._session_run_lock:
                        active = dict(self.adapter._active_session_agents)
                except Exception:
                    pass
                if sid in active:
                    continue
                cont = mgr.next_continuation_prompt()
                if cont:
                    proj["state"] = "running"
                    proj["round"] = int(st.turns_used) + 1
                    proj["summary"] = "resumed after gateway restart"
                    self.report(sid, proj, continuation=cont)
                    continue
            if st.status == "active" and mgr.is_waiting():
                self._schedule_barrier_wakeup(sid)
            self.report(sid, proj)
