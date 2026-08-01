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
        # scope key → callbacks that have atomically detached their Timer
        # from _barrier_timers but have not returned yet. Profile unload must
        # wait for these callbacks before closing the goal DB; Timer.cancel()
        # cannot stop a callback that has already begun (codex P1).
        self._barrier_callbacks: Dict[str, int] = {}
        self._barrier_callbacks_drained = threading.Condition(self._lock)
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
        # resolved profile home → epoch，unload 时递增：排队中尚未建锁/代际
        # 键的后置任务（CLI 建的 goal 首次经本进程跑等）也要能被 unload 失效
        # —— 只翻已存在的 session 键盖不到它们（codex P1）。
        self._home_epochs: Dict[str, int] = {}
        # adapter 级 epoch，disconnect（gateway 重启/adapter 替换）时递增：
        # 已进 executor 的 judge 线程躲得过 cancel_background_tasks（取消的
        # 只是 asyncio wrapper），若不失效，旧后置任务会在新进程 reconcile
        # 接管后并发 report continuation 双驱同一 goal（codex P1）。并入
        # _lock_generation 的 epoch 分量（只增不减，和求和后 != 判定兼容）。
        self._adapter_epoch: int = 0
        self._home_resolve_cache: Dict[str, str] = {}
        # scope key → generation counter, bumped on clear/done prune and on
        # create-over-existing. A post-turn hook that was QUEUED on the old
        # session lock while the user cleared + recreated the goal acquires
        # a lock that no longer guards anything — the generation check makes
        # it exit instead of evaluating the NEW goal with the OLD turn's
        # final_response (codex P1). Bounded at _LOCK_GEN_CAP.
        self._lock_gens: Dict[str, int] = {}

    _CANCEL_MARK_TTL_S = 180.0
    _LOCK_GEN_CAP = 1024

    def _scope_key(self, session_id: str) -> str:
        """Key the in-memory per-session state (locks / cancel marks / barrier
        timers) by profile home + sid: the driver is a singleton on the
        adapter while ``/p/{profile}`` only swaps the runtime scope — two
        profiles can legitimately own the same bare session_id (CLI-created
        sessions share naming), and cross-profile timer replacement/cancel
        would strand the other profile's waiting goal (codex P1)."""
        try:
            from hermes_constants import get_hermes_home

            return f"{get_hermes_home()}|{session_id}"
        except Exception:
            return session_id

    def _session_lock(self, session_id: str) -> threading.Lock:
        key = self._scope_key(session_id)
        with self._lock:
            lk = self._session_locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._session_locks[key] = lk
            return lk

    def _resolved_home(self, home: str) -> str:
        if not home:
            return ""
        cached = self._home_resolve_cache.get(home)
        if cached is not None:
            return cached
        try:
            from pathlib import Path

            resolved = str(Path(home).resolve())
        except Exception:
            resolved = home
        # home 数量 = profile 数，天然有界；GIL 下 str 赋值原子。
        self._home_resolve_cache[home] = resolved
        return resolved

    def _lock_generation_for_key_locked(self, key: str):
        """Return a generation while ``self._lock`` is held."""
        home = key.split("|", 1)[0] if "|" in key else ""
        rhome = self._resolved_home(home)
        return (
            self._lock_gens.get(key, 0),
            self._home_epochs.get(rhome, 0) + self._adapter_epoch,
        )

    def _lock_generation(self, session_id: str):
        """失效代际 = (session 代际, home epoch + adapter epoch)。home epoch
        由 profile unload 递增（codex P1）：覆盖「排队中尚未建 session 键」
        的后置任务 —— 它们捕获的 epoch 在 unload 后必然过期。adapter epoch
        由 disconnect 递增（codex P1）：整个 adapter 被替换/关停时全量失效。
        两者都只增，求和后任何一次翻转都让 != 复核失效。"""
        key = self._scope_key(session_id)
        with self._lock:
            return self._lock_generation_for_key_locked(key)

    def invalidate_all_generations(self) -> None:
        """Adapter teardown hook（codex P1）：disconnect 时让本 driver 名下
        所有已排队/在途的 goal 后置任务在 report 前的代际复核中失效 ——
        cancel_background_tasks 只能取消 asyncio wrapper，进了 executor 的
        judge 线程会继续跑完并上报，与替换者（新 adapter reconcile / 新进程）
        并发自驱同一 goal。"""
        with self._lock:
            self._adapter_epoch += 1

    def _bump_lock_generation_locked_key(self, key: str) -> None:
        """Caller holds self._lock."""
        self._lock_gens[key] = self._lock_gens.get(key, 0) + 1
        while len(self._lock_gens) > self._LOCK_GEN_CAP:
            self._lock_gens.pop(next(iter(self._lock_gens)))

    def bump_lock_generation(self, session_id: str) -> None:
        with self._lock:
            self._bump_lock_generation_locked_key(self._scope_key(session_id))

    def _prune_session_lock(self, session_id: str) -> None:
        key = self._scope_key(session_id)
        with self._lock:
            self._session_locks.pop(key, None)
            self._cancel_marks.pop(key, None)
            # 翻代：正排队等旧锁对象的 post-turn hook 拿到锁后必须失效退出
            # —— 旧锁已不在表里，它与后续新锁持有者不再互斥（codex P1）。
            self._bump_lock_generation_locked_key(key)

    def _mark_user_cancel(self, session_id: str) -> None:
        with self._lock:
            self._cancel_marks[self._scope_key(session_id)] = time.time()

    def _consume_user_cancel(self, session_id: str) -> bool:
        with self._lock:
            at = self._cancel_marks.pop(self._scope_key(session_id), None)
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

    def _barrier_holding(self, st: Any) -> bool:
        """Read-only twin of ``GoalManager.is_waiting()``: judges whether the
        wait barrier still holds WITHOUT the lazy auto-clear side effect
        (``stop_waiting()`` writes state back to the DB). ``projection()``
        runs on lock-free paths (GET /goal status, interaction hooks) — a
        write there races a concurrent pause/clear and can clobber the fresh
        state with a stale active snapshot (codex P1). Barrier clearing only
        happens on the lock-held driving paths (_kick_after_barrier /
        _barrier_wakeup / evaluate_after_turn)."""
        if st is None:
            return False
        try:
            from hermes_cli.goals import _pid_alive, _session_waiting

            if st.waiting_on_session is not None:
                return bool(_session_waiting(st.waiting_on_session))
            if st.waiting_on_pid is not None:
                return bool(_pid_alive(st.waiting_on_pid))
            if st.waiting_until:
                return time.time() < st.waiting_until
            return False
        except Exception:
            # 上游私有 helper 改名等极端情况：退化为「有 barrier 字段即视为
            # waiting」——保守显示，不写库。
            return bool(st.waiting_on_session or st.waiting_on_pid or st.waiting_until)

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
        barrier_holding = self._barrier_holding(st) if st.status == "active" else False
        if st.status == "active":
            state = "waiting" if barrier_holding else "running"
        elif st.status in ("paused", "done", "cleared"):
            state = st.status
        else:
            state = st.status
        summary = st.last_reason or st.paused_reason or ""
        if barrier_holding:
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
        cause: str = "",
    ) -> None:
        url = _scoped_env("ZET_GOAL_ADVANCE_URL").strip()
        if not url:
            logger.debug("[zet_goal] ZET_GOAL_ADVANCE_URL unset, skip report")
            return
        payload = dict(proj)
        payload["agent_id"] = _scoped_env("ZET_AGENT_ID").strip()
        payload["session_id"] = self._wire_session_id(session_id)
        if cause:
            # additive 可选字段（HR#4）：local-server 用它区分「用户 resume
            # 触发的 running」与「pause 前旧判定迟到的 running」——后者不得
            # 撤销用户的暂停（codex P2）。
            payload["cause"] = cause
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
            proj = await asyncio.to_thread(self._projection_at_tip, session_id)
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

    def _follow_migration(self, session_id: str) -> str:
        """沿 sidecar 的 ``migrated_to`` 指针解析到 goal 当前所在的 sid。

        压缩轮转后旧 sidecar 只剩指针、goal 行是 cleared tombstone ——
        local-server 仍可能拿 pre-rotation id 发控制请求（interrupt 分支
        同款时序），不解析的话 pause/clear/status 会打在 tombstone 上
        「成功」返回，新 sid 下真正 active 的 goal 继续被自驱（codex P1）。
        深度上限与 reconcile 的指针恢复一致，防指针环。"""
        sid = session_id
        for _ in range(4):
            try:
                dest = str(self._load_sidecar(sid).get("migrated_to") or "").strip()
            except Exception:
                return sid
            if not dest or dest == sid:
                return sid
            sid = dest
        return sid

    def _projection_at_tip(self, session_id: str) -> Dict[str, Any]:
        return self.projection(self._follow_migration(session_id))

    def _apply_action_sync(self, session_id: str, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
        from hermes_cli.goals import GoalManager, parse_contract

        # 控制动作一律先解析到迁移后的 sid（codex P1）。create 也解析：
        # 轮转后旧 hermes sid 已无会话内容，goal 建在那里会立刻失联。
        session_id = self._follow_migration(session_id)
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
                # 新 goal 不承接历史 stop：丢弃残留 cancel mark（无 goal 会话
                # 上按过的停止 / 旧 goal 清除前的停止，TTL 180s 内仍在）——
                # 否则新 goal 第一轮 post-turn 就会消费旧标记误暂停（codex P1）。
                self._consume_user_cancel(session_id)
                # 同理不承接旧 goal 的 barrier timer（codex P1）：旧 goal 在
                # WAIT 时被 create 覆盖，残留 timer 之后触发会加载到新 goal
                # （已无 barrier）并下发 continuation，与新 goal 并发起轮。
                self._cancel_barrier_timer(session_id)
                # 翻代：还在旧锁上排队的旧轮 post-turn hook 不得评估新 goal
                # （codex P1）。
                self.bump_lock_generation(session_id)
                # 覆盖已有 active goal 时先把旧状态置为不可自驱（codex P1）：
                # 下面「sidecar/index 先行、mgr.set() 最后」的崩溃顺序在覆盖
                # 场景有反例 —— 新 goal_id sidecar 已写、set() 未跑时崩溃，
                # 重启 reconcile 会拿旧 active GoalState 配新 goal_id 继续
                # 自驱旧目标。先 pause：中间态崩溃后旧 goal 是可见的 paused
                # （不自驱、可手动处理），set() 完成即被新 goal 整体覆盖。
                try:
                    if mgr.is_active():
                        mgr.pause("superseded by a new goal")
                except Exception:
                    logger.debug("[zet_goal] pre-create pause failed", exc_info=True)
                # Crash-ordering: sidecar + index FIRST, activation LAST — a
                # crash mid-create must leave "goal not yet active", never
                # "active but untracked" (an orphan reconcile can't see).
                self._save_sidecar(session_id, side)
                self._index_add(session_id)
                headline, contract = parse_contract(text)
                mgr.set(headline or text, max_turns=max_rounds, contract=contract)
            elif action == "pause":
                st0 = mgr.state
                if st0 is None or st0.status in ("cleared", "done"):
                    # 与 resume 对称（codex P1）：上游 pause() 不校验状态，会把
                    # 终态行改成 paused —— 过期的 pause 一到，已终结的 goal 就
                    # 能被后续合法 resume 复活自驱。终态只回投影，不动 DB。
                    return self.projection(session_id, mgr=mgr)
                reason = str(body.get("reason", "") or "").strip() or "user-paused"
                mgr.pause(reason)
                self._cancel_barrier_timer(session_id)
            elif action == "resume":
                st0 = mgr.state
                if st0 is None or st0.status in ("cleared", "done"):
                    # 上游 resume() 不校验状态，会把 cleared/done 直接置回
                    # active（codex P1）——local-server 的重试/过期 resume 一到，
                    # 用户已终结的 goal 就重新自驱。终态只回投影，不动状态。
                    return self.projection(session_id, mgr=mgr)
                if st0.status == "active" and self._barrier_holding(st0):
                    # WAIT barrier 仍成立的 active goal：过期/重试的 resume
                    # 不得清 barrier（上游 resume() 会抹掉 waiting_on_*）再
                    # 立即续轮 —— 那会绕过 judge 设下的等待、重复触发长任务
                    # （codex P1）。幂等处理：只确保 wakeup timer 在，回投影。
                    self._schedule_barrier_wakeup(session_id)
                    return self.projection(session_id, mgr=mgr)
                # 显式恢复 = 宣告此前的 stop 作废：丢弃残留 cancel mark（stop
                # 中断的轮次不跑 post-turn hook，标记不会被正常消费），否则
                # 恢复后第一轮结束时旧标记会把 goal 又暂停回去（codex P1）。
                self._consume_user_cancel(session_id)
                side0 = self._load_sidecar(session_id)
                if side0.get(self._INTERACTION_FLAG):
                    if st0 is not None and st0.status == "active":
                        # 自动 resume（local-server 的 error 重踢）撞上「approval/
                        # clarify 等待中 gateway 挂掉」：确认卡片已随 turn 消亡，
                        # 继续自驱等于替用户跳过确认（HR#3）—— park 成 paused，
                        # 留给用户显式恢复。用户手动 resume 作用于 paused 态，
                        # 不进此分支。
                        side0.pop(self._INTERACTION_FLAG, None)
                        self._save_sidecar(session_id, side0)
                        mgr.pause(
                            "waiting for your confirmation when the round was cut — resume to retry"
                        )
                        self._cancel_barrier_timer(session_id)
                        return self.projection(session_id, mgr=mgr)
                    # paused 态上的残留 flag（等待确认期间用户按了停止）：
                    # 用户显式 resume 即视为放弃那次确认，清掉再正常恢复。
                    side0.pop(self._INTERACTION_FLAG, None)
                    self._save_sidecar(session_id, side0)
                # 用户在第 N 轮暂停后点继续，必须从第 N+1 轮接着数 —— 上游
                # resume() 默认重置轮数预算（turns_used=0），会让 App 轮次
                # 显示跳回第 1 轮。只有"预算耗尽"的暂停才真正需要重置预算
                # （否则 resume 后立刻再次触发预算暂停），重置前把已用轮数
                # 累进 sidecar 偏移，App 侧轮次保持单调递增。
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
                #
                # 幂等防抖（codex P1）：resume 被重试时，上一条 resume 的续轮
                # 可能已在跑 —— 有在途 turn 就不再下发，否则同 goal 并发起轮
                # 重复烧预算。无在途 turn 才重踢（LS 的 error 重踢正是此场景，
                # 它下发前已确认会话空闲）；LS 侧另有 one-turn-per-session +
                # stale-anchor 兜底毫秒级窗口。
                if not self._session_turn_active(session_id):
                    cont = mgr.next_continuation_prompt()
                    if cont and not self._consume_user_cancel(session_id):
                        proj = dict(proj)
                        proj["round"] = self._cumulative_round(session_id, mgr.state.turns_used) + 1
                        self.report(session_id, proj, continuation=cont, cause="resume")
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
        # 代际在「排队时」而不是「executor 调度后」捕获（codex P1）：排队
        # 延迟内 clear+create 会让 hook 里读到新代际、复核形同虚设 —— 旧轮
        # 的 final_response 被拿去评估新 goal。
        scheduled_gen = self._lock_generation(session_id)
        try:
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._after_turn_sync,
                    session_id,
                    user_message or "",
                    final_response or "",
                    effective_session_id or "",
                    scheduled_gen,
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
            self._after_turn_sync(session_id, user_message or "", final_response or "", effective_session_id or "", scheduled_gen)

    def note_compaction_rotation(self, old_sid: str, new_sid: str) -> None:
        """Migrate the sidecar/index AT compaction time (codex P1). The goal
        row itself is moved by ``migrate_goal_to_session`` the moment the
        rotation happens (the old row is archived as cleared); waiting for
        the post-turn hook leaves a crash window — gateway dies between
        compaction and turn end → reconcile follows the old index entry to a
        cleared row, prunes it, and the active goal under the new sid is
        invisible to self-heal forever."""
        if not old_sid or not new_sid or old_sid == new_sid:
            return
        try:
            self._migrate_sidecar(old_sid, new_sid)
        except Exception:
            logger.debug("[zet_goal] compaction sidecar migration failed", exc_info=True)

    def _migrate_sidecar(self, old_sid: str, new_sid: str) -> None:
        """Follow a compaction-driven session rotation: the goal row was
        already moved by ``migrate_goal_to_session``; move our sidecar +
        index entry alongside so goal_id / app_session_id survive. A cancel
        mark dropped on the pre-rotation id (local-server keeps addressing
        it) must follow too, or the post-turn hook only consults the new id
        and a user stop gets silently outraced (codex P1).

        Idempotent by design: both the compaction-time migration
        (note_compaction_rotation) and the post-turn fallback call this.
        The first caller moves the data; later calls only fill gaps — the
        merge never overwrites keys already present under the new sid, and
        the old row is tombstoned so a stale copy can't clobber fresh
        new-sid state.

        Crash-ordering (codex P1 两轮收敛)：第 1 步先在旧 sidecar 上打
        ``migrated_to`` forward pointer（保留全部原字段）——此后任意点崩溃，
        reconcile 都能沿指针恢复新 sid 的 index/sidecar（app_session_id 不
        丢，上报不漂移）；新 sid 尽早入索引；tombstone 只留指针（防旧快照
        回盖新数据）；最后才摘旧索引。"""
        side = self._load_sidecar(old_sid)
        payload = {k: v for k, v in side.items() if k != "migrated_to"}
        if payload:
            self._save_sidecar(old_sid, {**payload, "migrated_to": new_sid})
            existing = self._load_sidecar(new_sid)
            merged = {**payload, **{k: v for k, v in existing.items() if k != "migrated_to"}}
            if merged != existing:
                self._save_sidecar(new_sid, merged)
        self._index_add(new_sid)
        if payload:
            self._save_sidecar(old_sid, {"migrated_to": new_sid})
        self._index_remove(old_sid)
        old_key, new_key = self._scope_key(old_sid), self._scope_key(new_sid)
        with self._lock:
            at = self._cancel_marks.pop(old_key, None)
            if at is not None and at > self._cancel_marks.get(new_key, 0.0):
                self._cancel_marks[new_key] = at

    def _after_turn_sync(
        self,
        session_id: str,
        user_message: str,
        final_response: str,
        effective_session_id: str = "",
        scheduled_gen: Any = None,
    ) -> None:
        from hermes_cli.goals import GoalManager

        rotated = bool(effective_session_id and effective_session_id != session_id)
        if rotated:
            self._migrate_sidecar(session_id, effective_session_id)
            session_id = effective_session_id
        try:
            # 未轮转（绝大多数轮）：用排队时捕获的代际，堵住排队窗口的
            # clear+create（codex P1）；轮转轮的代际键随 sid 变，退回进锁前
            # 快照（双低概率叠加，接受）。
            if scheduled_gen is not None and not rotated:
                gen0 = scheduled_gen
            else:
                gen0 = self._lock_generation(session_id)
            with self._session_lock(session_id):
                if self._lock_generation(session_id) != gen0:
                    # 排队等锁期间 goal 被 clear（可能又 create 了新 goal）：
                    # 本 hook 所属的旧轮世界已不存在，且此刻拿到的旧锁对象
                    # 已被摘除、与新锁持有者不再互斥 —— 只读代际后立刻退出，
                    # 绝不用旧轮的 final_response 评估新 goal（codex P1）。
                    return
                # Any pending-interaction flag died with the turn that raised
                # it (approved / denied / timed out inline) — clear it so it
                # can't park a later legitimate resume.
                self._clear_interaction_flag_locked(session_id)
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
                # Judge visibility into live background processes (CI/build/
                # watch launched by this turn): the WAIT verdict keys off the
                # snapshot — omitting it makes the judge continue immediately
                # and re-launch long tasks (codex P1). Same no-arg gather as
                # gateway/run.py's goal driver.
                try:
                    from hermes_cli.goals import gather_background_processes

                    bg_procs = gather_background_processes()
                except Exception:
                    bg_procs = None
                try:
                    decision = mgr.evaluate_after_turn(
                        final_response,
                        user_initiated=user_initiated,
                        background_processes=bg_procs,
                    )
                except Exception:
                    logger.warning("[zet_goal] evaluate_after_turn failed", exc_info=True)
                    return
                # judge 是本函数里唯一的长阻塞（LLM 调用，几十秒量级）——
                # 期间 profile 可能被 unload（bump_lock_generations_for_home
                # 翻代）：带着旧 profile context 上报 continuation 会重新驱动
                # 用户刚删掉的 agent（codex P1）。report 前复核一次代际。
                if self._lock_generation(session_id) != gen0:
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
                    # report 是网络调用且在锁内 —— stop 的 pause 线程会被本锁
                    # 挡住，continuation 却已发出（codex P1）。发送前最后一刻
                    # 再让 cancel mark 获胜（_mark_user_cancel 不等锁，同步可见）。
                    if self._consume_user_cancel(session_id):
                        mgr.pause("user stopped the running turn")
                        self._cancel_barrier_timer(session_id)
                        self.report(session_id, self.projection(session_id, mgr=mgr))
                        return
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
            t = self._barrier_timers.pop(self._scope_key(session_id), None)
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def cancel_all_barrier_timers(self) -> None:
        """Adapter teardown hook (codex P1): daemon Timers live outside
        ``_background_tasks``, so ``disconnect()``/reload would otherwise
        leave the OLD adapter's timers armed — they'd fire after teardown
        and issue continuations concurrently with the replacement adapter's
        reconcile, double-driving the same goal."""
        with self._lock:
            timers = list(self._barrier_timers.values())
            self._barrier_timers.clear()
        for t in timers:
            try:
                t.cancel()
            except Exception:
                pass

    def bump_lock_generations_for_home(self, profile_home: Any) -> None:
        """Profile-unload hook (codex P1): a post-turn judge task queued in
        ``_background_tasks`` outlives the profile's active-run count — after
        unload it would still evaluate and report a continuation with the
        stale profile context, re-driving the removed agent. Bumping every
        generation under the profile's home makes those tasks fail the
        re-check they run before reporting."""
        from pathlib import Path

        try:
            target = str(Path(str(profile_home)).resolve())
        except Exception:
            target = str(profile_home)
        with self._lock:
            # home epoch 先行：连「还没建 session 键」的排队任务也一并失效
            # （codex P1）；显式逐键翻转保留（同 home 的既有键立即过期）。
            self._home_epochs[target] = self._home_epochs.get(target, 0) + 1
            for key in set(self._lock_gens) | set(self._session_locks):
                home = key.split("|", 1)[0] if "|" in key else ""
                try:
                    matched = bool(home) and str(Path(home).resolve()) == target
                except Exception:
                    matched = home == str(profile_home)
                if matched:
                    self._bump_lock_generation_locked_key(key)

    def close_goal_db_for_home(self, profile_home: Any) -> None:
        """Profile-unload hook（codex P1）：goal sidecar 读写复用
        ``hermes_cli.goals._DB_CACHE``（按 hermes_home 缓存 SessionDB），
        adapter 只关自己的 ``_session_dbs`` 盖不到它 —— profile 删除/重建后
        旧连接仍指向已删 inode，reconcile/control 读到旧 goal 状态继续自驱，
        新 profile 的 goal 又不可见。按 home pop 并 close。"""
        from pathlib import Path

        try:
            from hermes_cli import goals as _goals_mod

            cache = getattr(_goals_mod, "_DB_CACHE", None)
        except Exception:
            return
        if not isinstance(cache, dict):
            return
        try:
            target = str(Path(str(profile_home)).resolve())
        except Exception:
            target = str(profile_home)
        victims = []
        for home in list(cache.keys()):
            try:
                matched = str(Path(home).resolve()) == target
            except Exception:
                matched = home == str(profile_home)
            if matched:
                victims.append(cache.pop(home, None))
        for db in victims:
            close = getattr(db, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def cancel_barrier_timers_for_home(self, profile_home: Any) -> None:
        """Profile-unload hook (codex P1): a WAIT barrier's daemon Timer
        captured the profile's runtime scope — after the profile is unloaded
        (and its directory deleted) the timer would still fire, read the
        goal via the stale in-memory scope and re-drive an agent the user
        just removed. Keys are ``{hermes_home}|{sid}`` — cancel the ones
        whose home resolves to the unloaded profile."""
        from pathlib import Path

        try:
            target = str(Path(str(profile_home)).resolve())
        except Exception:
            target = str(profile_home)
        victims = []
        with self._lock:
            for key in list(self._barrier_timers):
                home = key.split("|", 1)[0] if "|" in key else ""
                try:
                    matched = bool(home) and str(Path(home).resolve()) == target
                except Exception:
                    matched = home == str(profile_home)
                if matched:
                    victims.append(self._barrier_timers.pop(key))
        for t in victims:
            try:
                t.cancel()
            except Exception:
                pass

    def invalidate_barrier_callbacks_for_home(
        self, profile_home: Any, timeout_s: float = 10.0
    ) -> None:
        """Invalidate one profile's goal work and drain detached callbacks.

        A Timer removes itself from ``_barrier_timers`` before waiting for the
        per-session lock. Therefore cancellation alone cannot prove that no
        callback still owns the unloaded profile. Bump the non-reusable home
        epoch first, cancel timers still in the table, then wait for callbacks
        that already detached themselves. The unload handler calls this before
        runtime/DB teardown (codex P1).
        """
        target = self._resolved_home(str(profile_home))
        self.bump_lock_generations_for_home(profile_home)
        self.cancel_barrier_timers_for_home(profile_home)

        def belongs_to_target(key: str) -> bool:
            home = key.split("|", 1)[0] if "|" in key else ""
            return bool(home) and self._resolved_home(home) == target

        deadline = time.monotonic() + max(0.1, timeout_s)
        with self._barrier_callbacks_drained:
            while any(
                count > 0 and belongs_to_target(key)
                for key, count in self._barrier_callbacks.items()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "profile barrier callbacks did not drain before unload"
                    )
                self._barrier_callbacks_drained.wait(remaining)

    def _kick_after_barrier(
        self, session_id: str, mgr: Any, expected_generation: Any = None
    ) -> None:
        """Barrier satisfied → issue the continuation (wakeup + schedule 共用
        的续跑路径)。Caller holds the session lock. 下发前复核在途 turn
        （codex P1）：barrier 等待期间用户可能发起了普通 turn，与它并发自驱
        会重复执行工具 —— 让位，其 post-turn 评估接管续轮。"""
        if (
            expected_generation is not None
            and self._lock_generation(session_id) != expected_generation
        ):
            return
        if self._session_turn_active(session_id):
            return
        cont = mgr.next_continuation_prompt()
        if not cont:
            return
        if (
            expected_generation is not None
            and self._lock_generation(session_id) != expected_generation
        ):
            return
        # 发送前最后一刻让 cancel mark 获胜（codex P1，同 post-turn continue
        # 分支）。
        if self._consume_user_cancel(session_id):
            mgr.pause("user stopped the running turn")
            self.report(session_id, self.projection(session_id, mgr=mgr))
            return
        st = mgr.state
        proj = self.projection(session_id, mgr=mgr)
        proj["state"] = "running"
        proj["round"] = self._cumulative_round(session_id, st.turns_used) + 1
        proj["summary"] = "wait barrier cleared; continuing"
        if (
            expected_generation is not None
            and self._lock_generation(session_id) != expected_generation
        ):
            return
        self.report(session_id, proj, continuation=cont)

    def _schedule_barrier_wakeup(
        self, session_id: str, expected_generation: Any = None
    ) -> None:
        """Arm a timer that re-checks a parked goal's barrier. Time barriers
        wake exactly at the deadline; pid/session barriers poll at
        ``_BARRIER_POLL_S`` (the CLI equivalent is its always-running REPL
        tick — a gateway has no such loop, hence the timers)."""
        from hermes_cli.goals import GoalManager

        scheduled_gen = (
            expected_generation
            if expected_generation is not None
            else self._lock_generation(session_id)
        )
        if self._lock_generation(session_id) != scheduled_gen:
            return
        mgr = GoalManager(session_id)
        st = mgr.state
        if st is None or st.status != "active":
            return
        if not mgr.is_waiting():
            # Barrier 在设置与排定时器之间就满足了（被等的 pid 秒退等）：
            # is_waiting() 已顺手清掉 barrier —— 静默 return 会让 goal 卡在
            # active 无人续跑直到重启 reconcile（codex P1），走 wakeup 同款
            # 续跑路径。
            self._kick_after_barrier(
                session_id, mgr, expected_generation=scheduled_gen
            )
            return
        delay = _BARRIER_POLL_S
        if st.waiting_until and st.waiting_until > time.time():
            delay = max(1.0, st.waiting_until - time.time() + 1.0)
        # Timer callbacks run on a bare thread with NO contextvars — capture
        # the current context (mux profile scope) so the wakeup's report can
        # still resolve ZET_GOAL_ADVANCE_URL (see _spawn).
        import contextvars

        ctx = contextvars.copy_context()
        key = self._scope_key(session_id)
        timer = threading.Timer(
            delay,
            lambda: ctx.run(
                self._barrier_wakeup, session_id, scheduled_gen, key, timer
            ),
        )
        timer.daemon = True
        with self._lock:
            # Unload may have bumped the home epoch while GoalManager read the
            # barrier. Never publish a timer carrying the post-unload epoch.
            if self._lock_generation_for_key_locked(key) != scheduled_gen:
                return
            old = self._barrier_timers.pop(key, None)
            self._barrier_timers[key] = timer
        if old is not None:
            try:
                old.cancel()
            except Exception:
                pass
        timer.start()

    def _barrier_wakeup(
        self,
        session_id: str,
        scheduled_gen: Any,
        key: str,
        timer: threading.Timer,
    ) -> None:
        from hermes_cli.goals import GoalManager

        with self._barrier_callbacks_drained:
            # A replacement/cancelled timer must not pop or execute the timer
            # currently registered for the same session.
            if self._barrier_timers.get(key) is not timer:
                return
            self._barrier_timers.pop(key, None)
            self._barrier_callbacks[key] = self._barrier_callbacks.get(key, 0) + 1
        try:
            if self._lock_generation(session_id) != scheduled_gen:
                return
            with self._session_lock(session_id):
                # The callback may have waited here while profile unload
                # invalidated its home epoch. Re-check before touching the DB.
                if self._lock_generation(session_id) != scheduled_gen:
                    return
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    return
                st = mgr.state
                if st is None or st.status != "active":
                    return
                if mgr.is_waiting():
                    # Barrier still holding (pid/session) — keep polling.
                    self._schedule_barrier_wakeup(
                        session_id, expected_generation=scheduled_gen
                    )
                    return
                self._kick_after_barrier(
                    session_id, mgr, expected_generation=scheduled_gen
                )
        except Exception:
            logger.warning("[zet_goal] barrier wakeup failed", exc_info=True)
        finally:
            with self._barrier_callbacks_drained:
                remaining = self._barrier_callbacks.get(key, 0) - 1
                if remaining > 0:
                    self._barrier_callbacks[key] = remaining
                else:
                    self._barrier_callbacks.pop(key, None)
                self._barrier_callbacks_drained.notify_all()

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

    _INTERACTION_FLAG = "interaction_pending"

    def _live_session_id(self, session_id: str) -> str:
        """Resolve the CURRENT hermes sid for a live turn's request-time sid.

        Mid-turn compaction rotates ``agent.session_id`` (goal row + sidecar
        follow immediately via note_compaction_rotation), while approval/
        clarify callbacks keep the sid captured at ``_create_agent`` time —
        writing waiting flags/pauses under the stale id would park them on a
        tombstoned sidecar reconcile never scans (codex P1)."""
        try:
            key = self._scope_key(session_id)
            with self.adapter._session_run_lock:
                ref = self.adapter._active_session_agents.get(key) or self.adapter._active_session_agents.get(session_id)
            rotated = str(getattr(ref[0], "session_id", "") or "") if ref else ""
            if rotated and rotated != session_id:
                return rotated
        except Exception:
            pass
        return session_id

    def _session_turn_active(self, session_id: str) -> bool:
        """Whether this process has an in-flight turn belonging to the goal's
        session. Three-way match: the active table is keyed by the REQUEST-
        time App sid while compaction rotates ``agent.session_id`` and the
        goal index moves to the new sid immediately — any single-key lookup
        misses one direction (codex P1).

        Profile-filtered (codex P1 两轮收敛): the active table is keyed
        ``{hermes_home}|{sid}`` (adapter._active_turn_key — same shape as our
        _scope_key), so two profiles running SAME-NAMED sessions concurrently
        hold separate entries and neither overwrites the other. Entries from
        another profile's home never count as this goal's in-flight turn;
        bare-sid entries (legacy / single-profile) stay permissive."""
        active = {}
        try:
            with self.adapter._session_run_lock:
                active = dict(self.adapter._active_session_agents)
        except Exception:
            return False
        current_home = ""
        try:
            from hermes_constants import get_hermes_home

            current_home = str(get_hermes_home())
        except Exception:
            pass

        keys = {session_id}
        try:
            app_sid = str(self._load_sidecar(session_id).get("app_session_id") or "").strip()
            if app_sid:
                keys.add(app_sid)
        except Exception:
            pass
        for key, ref in active.items():
            if "|" in key:
                entry_home, entry_sid = key.split("|", 1)
            else:
                entry_home, entry_sid = "", key
            if entry_home and current_home and entry_home != current_home:
                continue  # 另一 profile 的同名会话（codex P1）
            if entry_sid in keys:
                return True
            try:
                if str(getattr(ref[0], "session_id", "") or "") == session_id:
                    return True
            except Exception:
                continue
        return False

    def _interaction_flag_set(self, session_id: str) -> bool:
        return bool(self._load_sidecar(session_id).get(self._INTERACTION_FLAG))

    def _clear_interaction_flag_locked(self, session_id: str) -> None:
        """Drop the pending-interaction flag. Caller holds the session lock."""
        try:
            side = self._load_sidecar(session_id)
            if side.pop(self._INTERACTION_FLAG, None) is not None:
                self._save_sidecar(session_id, side)
        except Exception:
            logger.debug("[zet_goal] clear interaction flag failed", exc_info=True)

    def on_interaction_pending(self, session_id: str) -> None:
        """An approval/clarify card is blocking the turn: project 'waiting'
        so the App banner explains the stall. GoalManager state itself is
        untouched — the turn is still running from the loop's viewpoint.
        The flag IS persisted to the sidecar: if the gateway dies before the
        user confirms, reconcile/resume must park the goal instead of
        self-driving past a confirmation nobody gave (codex P1, HR#3)."""
        from hermes_cli.goals import GoalManager

        # 回调闭包捕获的是 _create_agent 时的 sid —— 本轮若已压缩轮转，goal
        # 行/sidecar 都在新 sid 下（旧行 cleared、旧 sidecar tombstone），
        # 不解析的话 is_active() 直接 False，等待态既不投影也不落盘。
        session_id = self._live_session_id(session_id)
        # 判定 + flag 写入 + 投影构造全部在 session lock 内、状态锁内新载
        # （codex P1）：锁外判定 active 与锁内写 flag 之间 clear/create 可能
        # 换代 —— 旧回调会把等待标记写到新 goal 的 sidecar 并用旧投影上报
        # waiting，随后 reconcile/resume 把新目标无故 park。代际校验与
        # post-turn hook 同款。产品链路上 create 接管被 one-turn-per-session
        # 挡在旧 turn 结束之后，这里主要兜 clear 竞态与 CLI 旁路。
        gen0 = self._lock_generation(session_id)
        try:
            with self._session_lock(session_id):
                if self._lock_generation(session_id) != gen0:
                    return
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    return
                if not mgr.is_active():
                    return
                side = self._load_sidecar(session_id)
                if not side.get(self._INTERACTION_FLAG):
                    side[self._INTERACTION_FLAG] = time.time()
                    self._save_sidecar(session_id, side)
                proj = self.projection(session_id, mgr=mgr)
        except Exception:
            logger.debug("[zet_goal] persist interaction flag failed", exc_info=True)
            return
        proj["state"] = "waiting"
        proj["summary"] = "waiting for user confirmation"
        self.report_in_thread(session_id, proj)

    def _interaction_still_pending(self, *sids: str) -> bool:
        """本 session 是否还有未回应的 approval/clarify 卡片。approval/
        clarify 都是 per-session FIFO、没有 request_id —— respond 只解掉
        最老的一张，剩下的卡片仍在阻塞 turn，此时不能清等待标记。"""
        seen = set()
        for sid in sids:
            if not sid or sid in seen:
                continue
            seen.add(sid)
            try:
                from tools.approval import has_blocking_approval

                if has_blocking_approval(sid):
                    return True
            except Exception:
                pass
            try:
                # Clarify queues are keyed by the same profile-scoped
                # session identity as ZetAgentAdapter's active turns. A
                # bare sid here would make coder's outstanding card keep
                # main's goal projection in waiting (or vice versa).
                clarify_key = self.adapter._active_turn_key(sid)
                with self.adapter._clarify_state_lock:
                    if self.adapter._clarify_queues.get(clarify_key):
                        return True
            except Exception:
                pass
        return False

    def on_interaction_resolved(self, session_id: str) -> None:
        from hermes_cli.goals import GoalManager

        # 与 on_interaction_pending 同款轮转解析：flag 写在哪个 sid 就得从
        # 哪个 sid 清。pending 队列键的是回调原始 sid，两个都查。
        req_sid = session_id
        session_id = self._live_session_id(session_id)
        # 旧卡的响应不代表 turn 不再阻塞（codex P1）：clear+create 换代后
        # 新 goal 自己的卡片可能还挂着 —— per-session FIFO 会先解旧卡，
        # 此时清掉 sidecar 标记会让 gateway 重启后的 reconcile 跳过一个
        # 用户从未给出的确认（HR#3）。还有卡片在等就保留标记。
        if self._interaction_still_pending(req_sid, session_id):
            return
        # 判定 + 清 flag + 投影全部锁内、带代际（codex P1，与
        # on_interaction_pending 对称）：锁外窗口内 clear/create 换代时，
        # 旧回调不得动新 goal 的 sidecar。
        gen0 = self._lock_generation(session_id)
        try:
            with self._session_lock(session_id):
                if self._lock_generation(session_id) != gen0:
                    return
                try:
                    mgr = GoalManager(session_id)
                except Exception:
                    return
                self._clear_interaction_flag_locked(session_id)
                if not mgr.is_active():
                    return
                proj = self.projection(session_id, mgr=mgr)
        except Exception:
            logger.debug("[zet_goal] interaction resolved handling failed", exc_info=True)
            return
        self.report_in_thread(session_id, proj)

    # ------------------------------------------------------------------
    # Reconcile-on-start (HR#2 pair of local-server's goal keepalive)
    # ------------------------------------------------------------------

    async def reconcile_on_start(self) -> None:
        try:
            await asyncio.sleep(_RECONCILE_STARTUP_DELAY_S)
            await asyncio.to_thread(self._reconcile_all_scopes)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[zet_goal] reconcile-on-start failed", exc_info=True)

    def _reconcile_all_scopes(self) -> None:
        """Run the reconcile pass in every runtime scope that owns goal state.

        Under the multiplex gateway each profile keeps its goal rows /
        sidecars / index in its own HERMES_HOME state.db (SessionDB caches
        per home), and reports resolve ZET_GOAL_ADVANCE_URL through the
        profile's ``.env`` (``_scoped_env`` fails closed without a scope) —
        a single default-scope pass would neither see profile goals nor be
        able to report them, so profile goals never self-heal after an OOM
        respawn (codex P1). Single-profile processes keep the plain pass."""
        mux = False
        try:
            from agent.secret_scope import is_multiplex_active

            mux = is_multiplex_active()
        except Exception:
            mux = False
        if not mux:
            self._reconcile_sync()
            return
        try:
            from pathlib import Path

            from gateway.run import _profile_runtime_scope

            homes = self.adapter._multiplex_profile_homes()
        except Exception:
            logger.warning("[zet_goal] multiplex reconcile scaffolding unavailable", exc_info=True)
            return
        seen: set = set()
        for name, home in homes.items():
            try:
                key = str(Path(home).resolve())
            except Exception:
                key = str(home)
            if key in seen:
                continue
            seen.add(key)
            try:
                with _profile_runtime_scope(Path(home)):
                    self._reconcile_sync()
            except Exception:
                logger.warning("[zet_goal] reconcile failed for profile %s", name, exc_info=True)

    def _reconcile_sync(self) -> None:
        for sid in self._index():
            try:
                self._reconcile_one(sid)
            except Exception:
                logger.debug("[zet_goal] reconcile failed for %s", sid, exc_info=True)

    def _reconcile_one(self, sid: str, _depth: int = 0) -> None:
        """Reconcile a single indexed goal. The verdict AND the report happen
        under the same session lock with freshly loaded state (codex P1):
        the 5s startup reconcile can race a user pause/clear or the post-turn
        hook — a lock-free snapshot taken before the race would re-kick a
        goal the user just stopped. Mirrors _after_turn_sync, whose report
        also runs inside the lock."""
        followup = self._reconcile_one_locked(sid, _depth)
        if followup and followup != sid and _depth < 4:
            # 轮转迁移中途崩溃的恢复（codex P1）：旧行的 migrated_to 指针
            # 把我们带到新 sid —— 在旧 sid 的锁外接着 reconcile 它（指针链
            # 每次压缩加一层，深度上限防御环）。
            self._reconcile_one(followup, _depth + 1)

    def _reconcile_one_locked(self, sid: str, _depth: int = 0) -> str:
        from hermes_cli.goals import GoalManager

        with self._session_lock(sid):
            try:
                mgr = GoalManager(sid)
            except Exception:
                return ""
            st = mgr.state
            if st is None or st.status in ("cleared", "done"):
                dest = ""
                try:
                    side_old = self._load_sidecar(sid)
                    dest = str(side_old.get("migrated_to") or "").strip()
                    if dest and dest != sid:
                        # 迁移在半途崩溃：先把新 sid 恢复进 index，缺 sidecar
                        # 时用旧行保留的字段补齐（app_session_id 必须跟过去，
                        # 否则上报漂移到轮转后的 hermes sid，App 稳定会话收
                        # 不到下一轮 —— codex P1）。
                        self._index_add(dest)
                        if not self._load_sidecar(dest):
                            restored = {k: v for k, v in side_old.items() if k != "migrated_to"}
                            if restored:
                                self._save_sidecar(dest, restored)
                except Exception:
                    logger.debug("[zet_goal] migration pointer follow failed", exc_info=True)
                self._index_remove(sid)
                return dest
            if st.status == "active" and self._interaction_flag_set(sid):
                if self._session_turn_active(sid):
                    # 确认轮还活着（进程启动后 5s 内已有本 goal 的轮在跑并卡
                    # 在 approval/clarify）：卡片就在内存里等用户，不是重启
                    # 丢失的 —— park 会让用户确认后 post-turn 只见 paused、
                    # 循环无故卡住（codex P1）。什么都不动，交给交互钩子。
                    return
                # The gateway died while an approval/clarify card was blocking
                # a round — the card died with the turn. Blindly continuing
                # would self-drive past a confirmation the user never gave
                # (codex P1, HR#3): park it and let the user resume explicitly.
                self._clear_interaction_flag_locked(sid)
                if mgr.is_active():
                    mgr.pause(
                        "gateway restarted while waiting for your confirmation — resume to retry"
                    )
                self.report(sid, self.projection(sid))
                return
            proj = self.projection(sid, mgr=mgr)
            if st.status == "active" and not mgr.is_waiting():
                # A live loop was cut mid-flight (crash / OOM respawn).
                # Skip if this process already has an active turn for the
                # session — the loop is running, no kick needed（三路比对
                # 见 _session_turn_active，codex P1）。
                if self._session_turn_active(sid):
                    return
                cont = mgr.next_continuation_prompt()
                if cont:
                    proj["state"] = "running"
                    proj["round"] = int(st.turns_used) + 1
                    proj["summary"] = "resumed after gateway restart"
                    self.report(sid, proj, continuation=cont)
                    return
            if st.status == "active" and mgr.is_waiting():
                self._schedule_barrier_wakeup(sid)
            self.report(sid, proj)
