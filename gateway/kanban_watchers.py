"""Kanban board watcher methods for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3).
These are the background-loop methods that subscribe to kanban boards, deliver
notifications/artifacts, and drive the multi-agent dispatcher. They use only
``self`` state, so they live on a mixin that ``GatewayRunner`` inherits — the
``self._kanban_*`` call sites resolve identically via the MRO, making this a
behavior-neutral move that lifts ~1,000 LOC out of run.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import NamedTuple

from gateway.platforms.base import safe_exc, safe_traceback


class ArtifactFailure(NamedTuple):
    """一个交付物送不到 —— ``path`` 是它，``reason`` 是**为什么**。

    ⭐ 三种成因给用户的话完全不同,压成一个 ``List[str]`` 就只能说出
    三者的最小公约数（原先是「文件仍在设备上，可稍后重新获取」——
    对 ``missing`` 是**假的**）。
    """

    path: str
    reason: str


#: reason ⇒ 给用户的一句话。⛔ 每种成因各自可行动。
_ARTIFACT_FAILURE_TEXT = {
    # ⛔ 不说「仍在设备上」—— 它已经不在了
    "missing": "{n} 个附件在设备上已找不到，任务可能已清理它们。",
    # 🔴 ⛔ 一个 basename 都不许回显:回显等于向聊天对方**确认这个路径存在**。
    "policy_blocked": "{n} 个附件因安全策略未发送。",
    "upload_failed": "{n} 个附件未能送达：{names}。文件仍在设备上，可稍后重新获取。",
}
from typing import Any, Callable, List, Optional

from agent.i18n import t

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")


def _resolve_auto_decompose_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int]":
    """Resolve the live (enabled, per_tick) auto-decompose settings.

    Read fresh from config on every dispatcher tick (#49638) so that flipping
    ``kanban.auto_decompose: false`` to STOP runaway fan-out takes effect on the
    next tick instead of requiring a gateway restart. Auto-decompose is a
    safety toggle — a user who sees it create and launch tasks they didn't
    intend reaches for this flag to halt it, and a stale boot-captured value
    silently ignoring that change is the bug reported in #49638.

    Fails **safe**: if the config read raises, return ``(False, 3)`` — a
    transient read error must never re-enable a feature the user turned off,
    nor fall back to the burst-prone default-on behaviour. ``per_tick`` is
    clamped to ``>= 1``.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("auto_decompose", True))
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    if per_tick < 1:
        per_tick = 1
    return enabled, per_tick


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take an exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway process machine-wide may run the embedded kanban
    dispatcher: concurrent dispatchers double the reclaim frequency (each
    runs its own ``release_stale_claims`` → promote → dispatch loop), double
    claim-attempt events in the event log, and — with ``wal_autocheckpoint=0`` —
    concurrent manual WAL checkpoints can corrupt index pages. The
    ``dispatch_in_gateway`` config flag is the primary control; this lock is the
    backstop that survives config drift and same-profile restart races.

    Delegates to :func:`gateway.status._try_acquire_file_lock` (``fcntl`` on
    POSIX, ``msvcrt`` on Windows) so the guard is cross-platform.

    Returns ``(handle, "held")`` on success — the caller keeps the file handle
    for the process lifetime and **must** release it via
    :func:`_release_singleton_lock` when done. ``(None, "contended")`` when
    another process holds the lock (caller must NOT dispatch). ``(None,
    "unavailable")`` when locking cannot be performed (non-POSIX filesystem
    without flock, or the status.py helpers are unimportable) — caller falls
    back to config-only control.
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a dispatcher singleton lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock
        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        """Return whether this gateway currently owns the singleton lock."""
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership before releasing the OS lock."""
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        _release_singleton_lock(handle)

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        For each subscription row, fetches ``task_events`` newer than the
        stored cursor with kind in the terminal set (``completed``,
        ``blocked``, ``gave_up``, ``crashed``, ``timed_out``). Sends one
        message per new event to ``(platform, chat_id, thread_id)``,
        then advances the cursor. When a task reaches a terminal state
        (``completed`` / ``archived``), the subscription is removed.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Each gateway polls only subscriptions owned by profiles whose
        adapters it hosts. The dispatch-owning gateway also handles legacy
        subscriptions without a profile stamp.
        """
        # Dispatch and delivery have separate ownership. A deployment may run
        # one dispatcher while each profile has its own gateway credentials;
        # those adapter-owning gateways must still poll and deliver their own
        # subscriptions. Legacy rows without a notifier_profile are visible
        # only while this process holds the actual singleton dispatcher lock.
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        # "status" covers dashboard drag-drop and `_set_status_direct()`
        # writes — surface those transitions to subscribers too.
        TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out", "status", "archived", "unblocked", "block_loop_detected")
        # Subscriptions are removed only when the task reaches a truly final
        # status (done / archived). We used to also unsub on any terminal
        # event kind (gave_up / crashed / timed_out / blocked), but that
        # silently dropped the user out of the loop whenever the dispatcher
        # respawned the task: a worker that crashes, gets reclaimed, runs
        # again, and crashes a second time would only notify on the first
        # crash because the subscription was deleted after the first event.
        # Same shape as the reblock-after-unblock cycle that PR #22941
        # fixed for `blocked`. Keeping the subscription alive until the
        # task is genuinely done lets the cursor (advanced atomically by
        # claim_unseen_events_for_sub) handle dedup, and any retry-loop
        # event reaches the user.
        # Per-subscription send-failure counter. Adapter.send raising
        # means the chat is dead (deleted, bot kicked, etc.) — after N
        # consecutive send failures the sub is dropped so we don't spin
        # against a dead chat every 5 seconds forever.
        # Raised from 3 to 12 (~60s at the 5s tick cadence): now that a
        # reported SendResult(success=False) also lands here (see the
        # delivery loop below), a transient Telegram/API outage of a few
        # ticks must NOT permanently unsubscribe a live review-gate channel.
        # A genuinely dead chat still drops, just ~60s later — a fine trade
        # for an unattended gate where a false drop means silent work pileup.
        MAX_SEND_FAILURES = 12
        sub_fail_counts: dict[tuple, int] = getattr(
            self, "_kanban_sub_fail_counts", {}
        )
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None)
        if not notifier_profile:
            notifier_profile = self._active_profile_name()
            self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        while self._running:
            try:
                def _collect():
                    deliveries: list[dict] = []
                    include_unowned = self._owns_kanban_dispatcher_lock()
                    notifier_profiles = {notifier_profile}
                    notifier_profiles.update(
                        str(profile).strip()
                        for profile in getattr(self, "_profile_adapters", {})
                        if str(profile).strip()
                    )
                    active_platforms = {
                        getattr(platform, "value", str(platform)).lower()
                        for platform in self.adapters.keys()
                    }
                    # Widen to every platform any secondary profile has live,
                    # not just the default profile's. This is only a coarse
                    # pre-filter to skip claiming events for subs nobody can
                    # possibly deliver — the precise per-profile check (via
                    # gateway/authz_mixin.py::_authorization_adapter, which
                    # forbids default-profile fallback) still runs at delivery
                    # time below, rewinding the claim if it resolves to None.
                    # Without this, a subscription owned by a secondary
                    # profile on a platform the DEFAULT profile never
                    # connected (e.g. beta owns discord, default doesn't) was
                    # dropped here before ever being claimed — no rewind
                    # applies to an unclaimed event, so it silently never
                    # retries.
                    for _profile_adapter_map in getattr(self, "_profile_adapters", {}).values():
                        active_platforms.update(
                            getattr(platform, "value", str(platform)).lower()
                            for platform in _profile_adapter_map.keys()
                        )
                    if not active_platforms:
                        logger.debug("kanban notifier: no connected adapters; skipping tick")
                        return deliveries

                    # Enumerate every board on disk, but poll each resolved DB
                    # path once. Multiple slugs can point at the same DB when
                    # HERMES_KANBAN_DB pins the board path; without this guard
                    # one gateway could collect the same subscription/event
                    # more than once before advancing the cursor.
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: set[str] = set()
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            logger.debug(
                                "kanban notifier: skipping duplicate board slug %s for DB %s",
                                slug, resolved_db_path,
                            )
                            continue
                        seen_db_paths.add(resolved_db_path)
                        # Zero-subscription early exit: probe the board with a
                        # cheap read-only connection BEFORE the writable
                        # `connect()`. A board with no subscriptions has
                        # nothing to notify, and the writable open (schema
                        # init/migration on first open, WAL/-shm sidecars,
                        # checkpoint traffic) is exactly the per-tick cost
                        # this skip avoids.
                        try:
                            if _kb.count_notify_subs(
                                board=slug,
                                notifier_profiles=notifier_profiles,
                                include_unowned=include_unowned,
                            ) == 0:
                                logger.debug(
                                    "kanban notifier: board %s has no subscriptions owned by %s; skipping open",
                                    slug, sorted(notifier_profiles),
                                )
                                continue
                        except Exception as exc:
                            logger.debug(
                                "kanban notifier: read-only subscription probe failed "
                                "for board %s (%s); falling back to writable open",
                                slug, safe_exc(exc),
                            )
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, safe_exc(exc))
                            continue
                        try:
                            # `connect()` runs the schema + idempotent migration
                            # on first open per process, so an explicit
                            # `init_db()` here would be redundant. Worse:
                            # `init_db()` deliberately busts the per-process
                            # cache and re-runs the migration on a *second*
                            # connection, which races the first and used to
                            # log a benign but noisy `duplicate column name`
                            # traceback (and intermittent "database is locked"
                            # — issue #21378) on every gateway start against
                            # a legacy DB. `_add_column_if_missing` now
                            # tolerates that race, but we still skip the
                            # redundant call to avoid the wasted work.
                            subs = _kb.list_notify_subs(
                                conn,
                                notifier_profiles=notifier_profiles,
                                include_unowned=include_unowned,
                            )
                            if not subs:
                                logger.debug("kanban notifier: board %s has no subscriptions", slug)
                            for sub in subs:
                                try:
                                    owner_profile = sub.get("notifier_profile") or None
                                    if owner_profile and owner_profile != notifier_profile:
                                        _owner_adapters = getattr(self, "_profile_adapters", {}).get(owner_profile)
                                        if not _owner_adapters:
                                            logger.debug(
                                                "kanban notifier: subscription for %s owned by profile %s; current profile %s has no adapter for it, skipping",
                                                sub.get("task_id"), owner_profile, notifier_profile,
                                            )
                                            continue
                                    platform = (sub.get("platform") or "").lower()
                                    if platform not in active_platforms:
                                        logger.debug(
                                            "kanban notifier: subscription for %s on %s skipped; adapter not connected",
                                            sub.get("task_id"), platform or "<missing>",
                                        )
                                        continue
                                    old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                        conn,
                                        task_id=sub["task_id"],
                                        platform=sub["platform"],
                                        chat_id=sub["chat_id"],
                                        thread_id=sub.get("thread_id") or "",
                                        kinds=TERMINAL_KINDS,
                                    )
                                    if not events:
                                        continue
                                    task = _kb.get_task(conn, sub["task_id"])
                                    logger.debug(
                                        "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                        len(events), sub["task_id"], slug, old_cursor, cursor,
                                    )
                                    deliveries.append({
                                        "sub": sub,
                                        "old_cursor": old_cursor,
                                        "cursor": cursor,
                                        "events": events,
                                        "task": task,
                                        "board": slug,
                                    })
                                except Exception as sub_exc:
                                    # Isolate per-subscription failures so one
                                    # bad subscription cannot block delivery for
                                    # all other subscriptions in this tick.
                                    logger.warning(
                                        "kanban notifier: subscription for %s on board %s failed: %s",
                                        sub.get("task_id"), slug, safe_exc(sub_exc),
                                    )
                        finally:
                            conn.close()
                    return deliveries

                deliveries = await asyncio.to_thread(_collect)
                for d in deliveries:
                    sub = d["sub"]
                    task = d["task"]
                    board_slug = d.get("board")
                    platform_str = (sub["platform"] or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string; skip and advance cursor so
                        # we don't replay forever.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        continue
                    sub_profile = sub.get("notifier_profile") or ""
                    # Route via the SAME chokepoint the authorization path uses
                    # (gateway/authz_mixin.py::_authorization_adapter): a stamped
                    # profile with its own adapter-registry entry must be served
                    # by THAT profile's same-platform adapter and must NOT silently
                    # fall back to the default profile's adapter — otherwise a
                    # secondary profile's task notification is delivered by the
                    # wrong bot (the cross-profile mis-delivery this whole change
                    # exists to fix). The helper returns None only when the profile
                    # (or default) genuinely has no adapter for the platform.
                    adapter = self._authorization_adapter(plat, sub_profile or None)
                    if adapter is None:
                        logger.debug(
                            "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                            platform_str, sub["task_id"],
                        )
                        await asyncio.to_thread(
                            self._kanban_rewind,
                            sub,
                            d["cursor"],
                            d.get("old_cursor", 0),
                            board_slug,
                        )
                        continue
                    title = (task.title if task else sub["task_id"])[:120]
                    board_tag = f"[{board_slug}] " if board_slug else ""
                    # Per-subscription failure-counter key. Hoisted out of the
                    # event loop: the wake self-post path (in the loop's
                    # ``else`` clause) needs it even when every event in the
                    # claim was skipped before reaching the send site.
                    sub_key = (
                        sub["task_id"], sub["platform"],
                        sub["chat_id"], sub.get("thread_id") or "",
                    )
                    for ev in d["events"]:
                        kind = ev.kind
                        # Identity prefix: attribute terminal pings to the
                        # worker that did the work. Makes fleets (where one
                        # chat subscribes to many tasks) legible at a glance.
                        who = (task.assignee if task and task.assignee else None)
                        tag = f"@{who} " if who else ""
                        if kind == "completed":
                            # Prefer the run's summary (the worker's
                            # intentional human-facing handoff, carried
                            # in the event payload), then fall back to
                            # task.result for legacy rows written before
                            # runs shipped.
                            handoff = ""
                            payload_summary = None
                            if ev.payload and ev.payload.get("summary"):
                                payload_summary = str(ev.payload["summary"])
                            if payload_summary:
                                lines = payload_summary.strip().splitlines()
                                h = lines[0][:200] if lines else payload_summary[:200]
                                handoff = f"\n{h}"
                            elif task and task.result:
                                lines = task.result.strip().splitlines()
                                r = lines[0][:160] if lines else task.result[:160]
                                handoff = f"\n{r}"
                            msg = (
                                f"✔ {board_tag}{tag}Kanban {sub['task_id']} done"
                                f" — {title}{handoff}"
                            )
                        elif kind == "blocked":
                            reason = ""
                            if ev.payload and ev.payload.get("reason"):
                                reason = f": {str(ev.payload['reason'])[:160]}"
                            msg = f"⏸ {board_tag}{tag}Kanban {sub['task_id']} blocked{reason}"
                        elif kind == "gave_up":
                            err = ""
                            if ev.payload and ev.payload.get("error"):
                                err = f"\n{str(ev.payload['error'])[:200]}"
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} gave up "
                                f"after repeated spawn failures{err}"
                            )
                        elif kind == "crashed":
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} worker crashed "
                                f"(pid gone); dispatcher will retry"
                            )
                        elif kind == "timed_out":
                            limit = 0
                            if ev.payload and ev.payload.get("limit_seconds"):
                                limit = int(ev.payload["limit_seconds"])
                            msg = (
                                f"⏱ {board_tag}{tag}Kanban {sub['task_id']} timed out "
                                f"(max_runtime={limit}s); will retry"
                            )
                        elif kind == "status":
                            new_status = ""
                            if ev.payload and ev.payload.get("status"):
                                new_status = str(ev.payload["status"])
                            msg = f"🔄 {board_tag}{tag}Kanban {sub['task_id']} → {new_status}"
                        elif kind == "block_loop_detected":
                            # A task re-blocked for the same cause past the
                            # recurrence limit and was routed to `triage` for a
                            # human decision. This is the ONE transition that
                            # exists to force human attention, yet it emits no
                            # `blocked`/`status` event — so before adding it to
                            # TERMINAL_KINDS it produced zero notification and
                            # the task stalled in triage silently. Ping loudly.
                            reason = ""
                            recurrences = None
                            if ev.payload:
                                if ev.payload.get("reason"):
                                    reason = f": {str(ev.payload['reason'])[:160]}"
                                recurrences = ev.payload.get("recurrences")
                            rc = f" (blocked {recurrences}x for the same cause)" if recurrences else ""
                            msg = (
                                f"🛑 {board_tag}{tag}Kanban {sub['task_id']} routed to TRIAGE"
                                f" — needs a human decision{rc}{reason}"
                            )
                        else:
                            # archived / unblocked are claimed by TERMINAL_KINDS
                            # (so the cursor advances past them and they can't
                            # wedge a later completed/blocked event behind an
                            # unclaimed row) but are intentionally SILENT: an
                            # archive needs no user ping, and unblocked is an
                            # internal transition. They are also excluded from
                            # _WAKE_KINDS below, so they never wake the creator.
                            continue
                        delivery_metadata = sub.get("delivery_metadata")
                        metadata: dict[str, Any] = (
                            dict(delivery_metadata)
                            if isinstance(delivery_metadata, dict)
                            else {}
                        )
                        if sub.get("thread_id") and not metadata.get("thread_id"):
                            metadata["thread_id"] = sub["thread_id"]
                        # Adapters with no push channel (the API server —
                        # ``supports_async_delivery = False``) can NEVER
                        # satisfy a text-send: ``send()`` always reports
                        # SendResult(success=False) by design (see
                        # ApiServerAdapter.send()). Treating that as a
                        # delivery failure would rewind/drop the subscription
                        # forever and — because the wake dispatch below lives
                        # in this loop's ``else`` clause — would also make the
                        # wake-on-completion path (the actual fix for the
                        # api_server wrong-session bug) unreachable. So for
                        # non-push adapters, skip the doomed send attempt
                        # entirely: there is nothing to text-notify, the
                        # creator is woken via the self-post below instead.
                        from gateway.wake import adapter_supports_push

                        if not adapter_supports_push(adapter):
                            logger.debug(
                                "kanban notifier: adapter %s has no push "
                                "channel; skipping text ping for %s, relying "
                                "on wake self-post instead",
                                platform_str, sub["task_id"],
                            )
                            # Do NOT reset the failure counter here: on this
                            # path the wake self-post below IS the delivery,
                            # so the counter is resolved (reset or bumped) by
                            # the self-post outcome, not by skipping the send.
                            continue
                        try:
                            _send_res = await adapter.send(
                                sub["chat_id"], msg, metadata=metadata,
                            )
                            # A SendResult(success=False) without an exception
                            # (returned by push-capable adapters on a genuine
                            # transient failure) must count as a FAILED
                            # delivery — otherwise the cursor advances and the
                            # event is permanently lost. Adapters returning
                            # None (or anything non-SendResult shaped) keep
                            # the legacy "no exception == delivered" contract.
                            if getattr(_send_res, "success", True) is False:
                                raise RuntimeError(
                                    "adapter send() reported failure: "
                                    f"{getattr(_send_res, 'error', None) or 'unknown error'}"
                                )
                            logger.debug(
                                "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                                kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                            )
                            # After delivering the text notification, surface
                            # any artifact paths the worker referenced in
                            # ``kanban_complete(summary=..., artifacts=[...])``
                            # (or the legacy ``result`` field) as native
                            # uploads. ``extract_local_files`` finds bare
                            # absolute paths in the summary;
                            # ``send_document`` / ``send_image_file`` uploads
                            # them. Only fires on the ``completed`` event so
                            # we never spam attachments on retries.
                            if kind == "completed":
                                try:
                                    _failed_artifacts = await self._deliver_kanban_artifacts(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        event_payload=getattr(ev, "payload", None),
                                        task=task,
                                    )
                                except Exception as art_exc:
                                    # ⛔ 原先这里是 logger.debug（默认不输出）——
                                    # 附件整批投递炸了，线上一个字都看不到。
                                    # ⛔ 也必须是 ArtifactFailure —— 裸字符串
                                    # 会让下游 ``f.path`` / ``f.reason`` 炸,
                                    # 而这里正是**异常路径**,再崩一次就彻底没声音了。
                                    _failed_artifacts = [
                                        ArtifactFailure("<delivery raised>", "upload_failed")]
                                    logger.error(
                                        "kanban notifier: artifact delivery for %s failed: %s",
                                        sub["task_id"], safe_traceback(art_exc),
                                    )
                                if _failed_artifacts:
                                    # 🔴 用户刚收到「任务完成」，却拿不到文件。
                                    # ⛔ 不能沉默：他会以为文件根本没生成，
                                    # 回头把整个任务重跑一遍。
                                    # ⚠️ 只发**一条汇总**，⛔ 不是每个文件一条。
                                    await self._notify_artifact_delivery_failure(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        task_id=sub["task_id"],
                                        failed=_failed_artifacts,
                                    )
                            # Reset the failure counter on success.
                            sub_fail_counts.pop(sub_key, None)
                        except Exception as exc:
                            fails = sub_fail_counts.get(sub_key, 0) + 1
                            sub_fail_counts[sub_key] = fails
                            logger.warning(
                                "kanban notifier: send failed for %s on %s "
                                "(attempt %d/%d): %s",
                                sub["task_id"], platform_str, fails,
                                MAX_SEND_FAILURES, safe_exc(exc),
                            )
                            if fails >= MAX_SEND_FAILURES:
                                logger.warning(
                                    "kanban notifier: dropping subscription "
                                    "%s on %s after %d consecutive send failures",
                                    sub["task_id"], platform_str, fails,
                                )
                                await asyncio.to_thread(self._kanban_unsub, sub, board_slug)
                                sub_fail_counts.pop(sub_key, None)
                            else:
                                await asyncio.to_thread(
                                    self._kanban_rewind,
                                    sub,
                                    d["cursor"],
                                    d.get("old_cursor", 0),
                                    board_slug,
                                )
                            # Rewind the pre-send claim on transient failure so
                            # a later tick can retry. After too many failures,
                            # dropping the subscription is the terminal action.
                            break
                    else:
                        # All text pings delivered (or intentionally skipped
                        # for non-push adapters, whose delivery is the wake
                        # self-post below). Whether the cursor may advance now
                        # depends on the adapter class:
                        #
                        # * push-capable: the text send WAS the delivery, so
                        #   advance immediately (pre-existing behavior); the
                        #   wake injection below stays best-effort.
                        # * non-push (api_server): the wake self-post IS the
                        #   delivery. Advancing first would let a failed /
                        #   retry-exhausted self-post (swallowed by the
                        #   best-effort except) permanently lose the event.
                        #   So the self-post runs FIRST and the cursor only
                        #   advances after it succeeds — a failure rewinds the
                        #   claim exactly like a failed send() above, so the
                        #   next tick retries.
                        task_terminal = task and task.status in {"done", "archived"}
                        _WAKE_KINDS = ("completed", "gave_up", "crashed", "timed_out", "blocked")
                        _wake_kinds = {ev.kind for ev in d["events"] if ev.kind in _WAKE_KINDS}
                        from gateway.wake import adapter_supports_push as _adapter_push_ok

                        _is_push_adapter = _adapter_push_ok(adapter)
                        _session_key = ""
                        _synth = ""
                        if _wake_kinds:
                            _session_key = getattr(task, "session_id", None) or ""
                        if _wake_kinds and _session_key:
                            _title = (task.title if task else sub["task_id"])[:120]
                            _assignee = task.assignee if task else ""
                            _parts = []
                            if "completed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.completed"))
                            if "gave_up" in _wake_kinds: _parts.append(t("gateway.kanban.wake.gave_up"))
                            if "crashed" in _wake_kinds: _parts.append(t("gateway.kanban.wake.crashed"))
                            if "timed_out" in _wake_kinds: _parts.append(t("gateway.kanban.wake.timed_out"))
                            if "blocked" in _wake_kinds: _parts.append(t("gateway.kanban.wake.blocked"))
                            _status = t("gateway.kanban.wake.status_joiner").join(_parts) or t("gateway.kanban.wake.status_default")
                            _synth = t(
                                "gateway.kanban.wake.message",
                                task_id=sub["task_id"],
                                status=_status,
                                title=_title,
                                assignee=_assignee,
                                board=board_slug,
                            )

                        if not _is_push_adapter and _wake_kinds and _session_key:
                            # Wake self-post IS the delivery on this path —
                            # it must succeed BEFORE the cursor advances.
                            from gateway.wake import deliver_wake

                            try:
                                await deliver_wake(
                                    adapter,
                                    text=_synth,
                                    session_id=_session_key,
                                )
                                logger.info(
                                    "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                    sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                                )
                                sub_fail_counts.pop(sub_key, None)
                            except Exception as _wk_err:
                                fails = sub_fail_counts.get(sub_key, 0) + 1
                                sub_fail_counts[sub_key] = fails
                                logger.warning(
                                    "kanban notifier: wake self-post failed "
                                    "for %s (attempt %d/%d): %s",
                                    sub["task_id"], fails,
                                    MAX_SEND_FAILURES, safe_traceback(_wk_err),
                                )
                                if fails >= MAX_SEND_FAILURES:
                                    logger.warning(
                                        "kanban notifier: dropping subscription "
                                        "%s on %s after %d consecutive wake failures",
                                        sub["task_id"], platform_str, fails,
                                    )
                                    await asyncio.to_thread(self._kanban_unsub, sub, board_slug)
                                    sub_fail_counts.pop(sub_key, None)
                                else:
                                    # Rewind the pre-send claim so the next
                                    # tick retries the self-post — the event
                                    # is NOT lost.
                                    await asyncio.to_thread(
                                        self._kanban_rewind,
                                        sub,
                                        d["cursor"],
                                        d.get("old_cursor", 0),
                                        board_slug,
                                    )
                                continue

                        # Delivery complete (text ping for push adapters, wake
                        # self-post for non-push): advance cursor. The cursor
                        # is the dedup mechanism — it prevents re-delivery
                        # of the same event on subsequent ticks.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        if not _is_push_adapter:
                            # Nothing left to deliver on this path (the wake,
                            # if any, already succeeded above).
                            sub_fail_counts.pop(sub_key, None)
                        # Unsubscribe only when the task has reached a truly
                        # final status (done / archived). For blocked /
                        # gave_up / crashed / timed_out the subscription is
                        # kept alive so the user gets notified again if the
                        # dispatcher respawns the task and it cycles into the
                        # same state. See the longer comment on TERMINAL_KINDS
                        # above for the failure mode this prevents.
                        if _is_push_adapter and _wake_kinds and _session_key:
                            try:
                                from gateway.session import SessionSource
                                from gateway.wake import deliver_wake
                                # Rebuild the creator's real session scope from
                                # the chat_type persisted on the subscription
                                # row (#56580). build_session_key() keys DMs
                                # (":dm:<chat_id>") on a wholly different shape
                                # from group/thread, so the old hardcoded
                                # "group" mis-routed DM/thread creators into a
                                # fresh session. Legacy rows written before the
                                # column existed may still carry chat_type in
                                # delivery_metadata (#60600 rows) — fall back
                                # to that, then to "group" (the historical
                                # default that suits the dashboard/group flows).
                                # handle_message() get_or_create_session's the
                                # target, so a mismatch only ever degrades to a
                                # fresh session, never an exception.
                                _chat_type = str(sub.get("chat_type") or "").strip()
                                if not _chat_type:
                                    _delivery_meta = sub.get("delivery_metadata")
                                    if isinstance(_delivery_meta, dict):
                                        _chat_type = str(
                                            _delivery_meta.get("chat_type") or ""
                                        ).strip()
                                _chat_type = _chat_type or "group"
                                _source = SessionSource(
                                    platform=plat,
                                    chat_id=sub["chat_id"],
                                    chat_type=_chat_type,
                                    thread_id=sub.get("thread_id") or None,
                                    user_id=sub.get("user_id"),
                                    profile=sub_profile or None,
                                )
                                # deliver_wake preserves the synthetic
                                # MessageEvent/handle_message path for
                                # push-capable adapters (the non-push /
                                # self-post branch is handled BEFORE the
                                # cursor advance above).
                                await deliver_wake(
                                    adapter,
                                    text=_synth,
                                    session_id=_session_key,
                                    source=_source,
                                )
                                logger.info(
                                    "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                    sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                                )
                            except Exception as _wk_err:
                                # Best-effort: the notification itself already
                                # delivered and the cursor has advanced, so a
                                # broken wake path must not wedge the tick — but
                                # log at WARNING with a traceback rather than
                                # DEBUG so a persistently-failing wake is visible
                                # in normal logs instead of silently no-op'ing.
                                logger.warning(
                                    "kanban notifier: wakeup injection failed for %s: %s",
                                    sub["task_id"], safe_traceback(_wk_err),
                                )
                        if task_terminal:
                            await asyncio.to_thread(
                                self._kanban_unsub, sub, board_slug,
                            )
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", safe_exc(exc))
            # Sleep with cancellation checks.
            for _ in range(int(max(1, interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)

    def _kanban_advance(
        self, sub: dict, cursor: int, board: Optional[str] = None,
    ) -> None:
        """Sync helper: advance a subscription's cursor. Runs in to_thread.

        ``board`` scopes the DB connection to the board that owns this
        subscription. Unsub cursors in one board can't touch another's.
        """
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.advance_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                new_cursor=cursor,
            )
        finally:
            conn.close()

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.remove_notify_sub(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
            )
        finally:
            conn.close()

    def _kanban_rewind(
        self,
        sub: dict,
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        """Sync helper: undo a claimed notification cursor after send failure."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.rewind_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
            )
        finally:
            conn.close()

    async def _notify_artifact_delivery_failure(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        task_id: str,
        failed: List[str],
    ) -> None:
        """告诉用户「任务完成了，但这几个文件没送到」。

        ⭐ 判据是「不写它用户会不会做错事」——**会**:他刚收到「任务完成」,
        看不到附件就会认为**文件没生成**,于是把整个任务重跑一遍。
        ⇒ 这条属于「错误必须可行动」,⛔ 不在「别堆文案」要砍的范围里。

        ⛔ 只发**一条汇总**,⛔ 不是每个文件一条 —— 一次失败往往是整批失败
        (鉴权过期/超限),逐条发就是刷屏。
        ⛔ 这条自己失败时**不再往上抛**:文本通知已经送达,若因为这条提示
        失败就让整个事件重试,用户会**再收到一遍「任务完成」**。
        """
        # 🔴 按 reason 分组 —— ⛔ 不再一句话通吃。
        # ``missing`` 说「仍在设备上」是**假的**;``policy_blocked`` 回显
        # basename 等于向对方确认路径存在。⭐ 与 wecom/weixin 那条
        # 「成因决定建议」是同一把尺。
        by_reason: dict[str, List[ArtifactFailure]] = {}
        for f in failed:
            by_reason.setdefault(getattr(f, "reason", "upload_failed"), []).append(f)

        lines: List[str] = []
        for reason in ("upload_failed", "missing", "policy_blocked"):
            group = by_reason.get(reason)
            if not group:
                continue
            tmpl = _ARTIFACT_FAILURE_TEXT.get(
                reason, _ARTIFACT_FAILURE_TEXT["upload_failed"])
            names = ""
            if "{names}" in tmpl:
                # ⛔ 只有 upload_failed 才列名字:那些文件确实还在设备上,
                # 用户需要知道找哪几个。另两类⛔ 不列。
                names = ", ".join(os.path.basename(f.path) for f in group[:5])
                if len(group) > 5:
                    names += f" 等 {len(group)} 个"
            lines.append(tmpl.format(n=len(group), names=names))
        try:
            res = await adapter.send(
                chat_id=chat_id,
                content="⚠️ 任务已完成，但 " + "".join(lines),
                metadata=metadata,
            )
        except Exception as exc:
            logger.error(
                "kanban notifier: 连「附件未送达」的提示都没发出去 "
                "(task=%s, %d 个附件): %s",
                task_id, len(failed), safe_exc(exc),
            )
            return
        # 🔴 判据是 ``SendResult.success``,⛔ 不是「没抛异常」。
        # 讽刺的是:这个函数正是为了修「三个上传点丢掉返回值」而写的,
        # 而我在它自己身上又犯了同一个错 —— 平台**拒收**(不抛异常、返回
        # success=False)时它静默返回,用户既没拿到附件、也没被告知。
        # ⭐ 修一类缺陷时,新写的代码要先过一遍同一条判据。
        if getattr(res, "success", True) is False:
            logger.error(
                "kanban notifier: 「附件未送达」提示被平台拒收 "
                "(task=%s, %d 个附件): %s",
                task_id, len(failed), getattr(res, "error", None) or "unknown",
            )

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> List[str]:
        """Upload artifact files referenced by a completed kanban task.

        返回**未能送达**的路径清单（空 = 全部送达）。

        🔴 为什么返回清单而不是像文本通知那样 ``raise``:
        同一函数 ``:540`` 的先例是「``SendResult.success is False`` ⇒ raise ⇒
        游标不推进 ⇒ 整个事件重试」。附件这里**⛔ 不能照抄**——完成文本
        **已经发出去了**,整体重试会让用户**再收到一遍"任务完成"**。
        ⇒ 差异是刻意的:附件失败不回滚事件,但必须①被检出 ②可诊断
        ③让用户知道少了什么(否则他会以为文件没生成,回头重跑整个任务)。

        Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
        file paths through the completion event so downstream humans get
        the deliverable as a native upload instead of a path printed in
        chat.

        Sources scanned, in priority order:
          1. ``event_payload['artifacts']`` (explicit list — preferred)
          2. ``event_payload['summary']`` (truncated first line)
          3. ``task.result`` (legacy fallback)

        路径去重按**声明 identity**（已展开的原路径）做，⛔ 不是"存在的才去重"
        —— 后者会让同一个缺失交付物声明两次被计成两个失败。
        **显式声明**的交付物若缺失 / 被安全策略拒绝 / 上传失败，都会带**各自的
        原因**回到调用方；只有自由文本里扫出来的路径才可以静默忽略（它可能
        只是顺口提到）。投递错误只降级为消息级失败，⛔ 不打断 notifier 循环。
        """
        candidates: list[str] = []
        seen: set[str] = set()
        # ZET-2473：从自由文本里扫出来、但 producer 没有显式声明的路径。
        # 只用于记日志，⛔ 不投递。
        unclaimed: list[str] = []

        # 🔴 producer **显式声明**了、却拿不到的交付物（RH 复审第五轮 P1）。
        # 原先 `_add` 在 `isfile=False` 时直接 return、安全过滤也静默丢弃 ⇒
        # 用户只看到「任务完成」,既没有文件、也没有失败提示,会以为文件
        # 根本没生成然后把整个任务重跑一遍。
        # ⭐ 又一次「空列表二义」:candidates 为空分不清「没声明」和「声明了但没了」。
        # ⚠️ 作用域上界:只有 **claimed**（显式 artifacts 列表）才登记;
        #    自由文本扫出来的路径本来就允许是「顺手提到的引用」,⛔ 不许报。
        # 🔴 三种「送不到」是**三件不同的事**,⛔ 不许压成一个 List[str]:
        #   · missing        文件不在了 —— ⛔ 不能说「文件仍在设备上」
        #   · policy_blocked 安全策略拒绝 —— ⛔ 连 basename 都不许回显给聊天
        #                    （回显等于向对方确认「这个路径存在」）
        #   · upload_failed  平台没收 —— 文件确实还在设备上,可稍后重取
        # ⭐ 又一次「一个载体承担多个语义」。
        undeliverable: list[ArtifactFailure] = []

        def _add(path: str, *, claimed: bool) -> None:
            if not path:
                return
            expanded = os.path.expanduser(path)
            # 🔴 去重按**声明 identity**,⛔ 不是"存在才记" ——
            # 原先 seen 只在 isfile 之后写入,同一个缺失路径声明两次会被
            # 计成两个失败,给用户的数量直接说错。
            if expanded in seen:
                return
            seen.add(expanded)
            if not os.path.isfile(expanded):
                if claimed:
                    undeliverable.append(ArtifactFailure(expanded, "missing"))
                return
            (candidates if claimed else unclaimed).append(expanded)

        # 1. Explicit artifacts list in payload —— 唯一会被投递的一路。
        if isinstance(event_payload, dict):
            # ⭐ 用生产方导出的常量，⛔ 不写字面量：这一跳两端各自都有测试
            # 钉住，唯一的漂移方式就是「一端改了 key、另一端没跟上」。共用常量
            # 让改名必然同时影响两端 —— 把可能漂移变成结构上不可能。
            from hermes_cli.kanban_db import COMPLETED_EVENT_ARTIFACTS_KEY

            raw = event_payload.get(COMPLETED_EVENT_ARTIFACTS_KEY)
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, str):
                        _add(item, claimed=True)

            # 2. Paths embedded in the payload summary.
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                paths, _ = adapter.extract_local_files(summary)
                for p in paths:
                    _add(p, claimed=False)

        # 3. Legacy: paths embedded in task.result.
        if task is not None and getattr(task, "result", None):
            result_text = str(task.result)
            paths, _ = adapter.extract_local_files(result_text)
            for p in paths:
                _add(p, claimed=False)

        # ZET-2473：②③ 这两路是从**自由文本里扫出所有本地路径**，代码无从区分
        # 「用户交付物」和「agent 顺手读过的内部状态文件」。现场实证：用户被
        # .card_data.json / *_state.json / *_index.json 等九个内部文件刷屏，
        # 零个真交付物。
        # ⛔ 不许用扩展名 / 文件名黑名单收紧 —— 那是**开集**：不能黑 .json（真
        # 交付物也可能是 JSON），黑 *_state.json / *_index.json 则换个命名
        # （meta.json、cache.json）就漏。
        # 闭集只有一个：**producer 显式声明**。prompt_builder 已把
        # kanban_complete(artifacts=[...]) 定为 top-level 契约，所以这两路降级
        # 为只记日志 —— 万一真有「模型没声明却确实产出了交付物」的场景，日志会
        # 把它暴露出来，可以去推动 producer 补声明，而不是继续刷屏。
        if unclaimed:
            logger.info(
                "kanban notifier: task %s has %d unclaimed path(s) in "
                "summary/result; not delivered (producer must declare them via "
                "kanban_complete(artifacts=[...]))",
                getattr(task, "id", "?"),
                len(unclaimed),
            )
            logger.debug(
                "kanban notifier: unclaimed basenames = %s",
                [os.path.basename(p) for p in unclaimed],
            )

        if not candidates:
            return list(undeliverable)

        from gateway.platforms.base import BasePlatformAdapter
        # 🔴 ``filter_local_delivery_paths`` 返回的是**规范化后**的路径
        # （符号链接被解析成 target）。原先我拿它跟**原始**路径做字符串差集 ⇒
        # 一个合法的符号链接:target 被成功上传,alias 却同时进了失败清单,
        # 用户收到自相矛盾的「附件未送达」。
        # ⭐ 判据必须是 **canonical identity**,⛔ 不是字符串是否相等。
        _accepted = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        _accepted_ids = set()
        for p_ in _accepted:
            try:
                _accepted_ids.add(os.path.realpath(p_))
            except OSError:
                _accepted_ids.add(p_)
        _rejected = []
        for p_ in candidates:
            try:
                canon = os.path.realpath(p_)
            except OSError:
                canon = p_
            if canon not in _accepted_ids and p_ not in set(_accepted):
                _rejected.append(p_)
        candidates = _accepted
        if _rejected:
            # ⛔ 只记条数与 task,⛔ 不记 basename —— 见下方 basename 回显说明。
            logger.warning(
                "kanban notifier: %d 个已声明交付物被投递安全过滤拒绝 (task %s)",
                len(_rejected), getattr(task, "id", "?"),
            )
            undeliverable.extend(
                ArtifactFailure(p_, "policy_blocked") for p_ in _rejected)
        if not candidates:
            return list(undeliverable)

        # 这条投递路径原先**只在失败时记日志**，成功投递零留痕 —— ZET-2473 查不
        # 出来自哪一路，根源就是它。补上条数与 task 关联。
        # ⛔ 文件名不进 info 级：可能带用户内容，且对判断毫无帮助，只增加泄漏面。
        # 要看文件名请开 debug（只打 basename）。
        logger.info(
            "kanban notifier: delivering %d declared artifact(s) for task %s",
            len(candidates),
            getattr(task, "id", "?"),
        )
        logger.debug(
            "kanban notifier: declared basenames = %s",
            [os.path.basename(p) for p in candidates],
        )

        failed: List[ArtifactFailure] = list(undeliverable)
        # ⭐ 预算是**整批共享**的一个可变累加器,⛔ 不是墙钟 deadline。
        # 危害是「退避把 tick 睡死」⇒ 判据就该是「一共睡了多久」;
        # 挂在墙钟上会顺带把**上传本身耗时**也算进来,于是大文件传得慢
        # 就等于取消了后面所有文件的重试 —— 判据和危害没对齐。
        budget = [self._ARTIFACT_RETRY_BUDGET_S]

        for path in candidates:
            if await self._upload_artifact_with_retry(
                adapter=adapter, chat_id=chat_id, metadata=metadata,
                path=path, budget=budget,
            ):
                continue
            failed.append(ArtifactFailure(path, "upload_failed"))

        return failed

    #: 交付物上传的重试次数。⛔ 不照抄 ``_send_with_retry`` 的默认值 ——
    #: 这里跑在 notifier tick 里，**串行**挡着别的订阅(``:560`` 直接 await)，
    #: 退避多久就是别人等多久。⭐ 参数按这个约束定，⛔ 不按「别处写了几」。
    _ARTIFACT_MAX_RETRIES = 2
    _ARTIFACT_RETRY_BASE_DELAY = 1.0
    #: 整批交付物**共享**的退避总预算(秒)。
    #: ⭐ 只有 per-file 上限而没有总预算 = 20 个文件各退避 3 秒 ⇒ 一个 tick
    #: 被拖住一分钟。「无上限的重试」在小设备上是整机级问题的同一形状。
    _ARTIFACT_RETRY_BUDGET_S = 8.0
    # ⛔ 不许拍脑袋:取本文件既有的「一次交付」量纲 —— 重试总预算 8s 覆盖的是
    # **等待**,单次上传要能容纳一个真实的大附件传输,故取其一个数量级以上;
    # 与 gateway 侧 ``_POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS``(30s)同阶。
    _ARTIFACT_UPLOAD_TIMEOUT_S = 120.0
    #: 图片扩展名 —— 只用来选对上传 API，⛔ 不再用于批量分组(见下)。
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

    async def _send_one_artifact(self, *, adapter, chat_id, metadata, path):
        """按类型选上传 API。返回适配器的原始结果。

        🔴 图片这里**逐张** ``send_image_file``，⛔ 不用 ``send_multiple_images``。
        原因不是风格偏好，是**失败可观测性**:
          · ``send_multiple_images`` 在基类与 7 个 adapter 上都声明 ``-> None``，
            feishu 返回 ``bool``;
          · 于是 ``getattr(res, "success", True) is not False`` 对**全部 8 个**
            恒为 ``True`` —— 平台拒收(超限/鉴权过期)时 ``failed`` 是空的，
            用户收到「任务完成」而图片永远缺失，链路上一层报警都没有。
          · ⚠️ 上一版我在这里写了注释声称「判据是 SendResult.success」——
            **那是假闭集**:判据写对了，可这条路上根本没有 SendResult。
            ⭐ 假闭集比没有门更坏，它让人以为这一面守住了。

        ⛔ 为什么不给 ``send_multiple_images`` 加返回值(那才是根治):
        ``base.py:6368/6416`` 把它的返回值当**真值**用
        (``_consume_feishu_batch_quote(meta, image_sent)``)。今天 7 个
        adapter 返回 ``None``(falsy)、feishu 返回 ``bool`` —— 改成返回
        ``SendResult`` 会让那 7 个从 falsy 变成 truthy，**行为当场翻转**。
        ⇒ 改动作用域必须刚好等于缺陷:缺陷在 kanban 这一个调用点上。

        代价:用户看到 N 条图片消息而不是一个相册。⭐ 这里是**交付物投递**，
        「拿到文件」压倒「排版好看」;而且失败时能精确说出是哪几张没送到。
        """
        ext = Path(path).suffix.lower()
        # 🔴 音频先问**基座** —— ⛔ 不再用本文件自造的扩展名表。
        # ``should_send_media_as_audio``(base.py) 是仓内**已有**的共用分派层,
        # cron/scheduler、run.py 的两条路径、base 自己都在用;唯独我这条
        # kanban 路径自己造了第二套 ⇒ 同一个 .mp3 交付物,走对话发出去是语音、
        # 走 kanban 发出去是文件。⭐ 而且 Telegram 的特殊规则（只有 is_voice
        # 才发语音气泡）我这套完全没有。
        from gateway.platforms.base import should_send_media_as_audio

        if should_send_media_as_audio(
            getattr(adapter, "platform", None), ext, is_voice=False
        ):
            return await adapter.send_voice(
                chat_id=chat_id, audio_path=path, metadata=metadata)
        if ext in self._IMAGE_EXTS:
            return await adapter.send_image_file(
                chat_id=chat_id, image_path=path, metadata=metadata)
        if ext in self._VIDEO_EXTS:
            return await adapter.send_video(
                chat_id=chat_id, video_path=path, metadata=metadata)
        return await adapter.send_document(
            chat_id=chat_id, file_path=path, metadata=metadata)

    async def _upload_artifact_with_retry(
        self, *, adapter, chat_id, metadata, path: str, budget: List[float],
    ) -> bool:
        """上传一个交付物，瞬时故障下重试。返回是否送达。

        ⭐ 退避照抄 ``BasePlatformAdapter._send_with_retry``
        (``base.py:5256-5265``):服务端给的 ``retry_after`` 优先且只认一次，
        否则 ``base * 2**(n-1) + jitter``。⛔ 不自造第二套。

        三处**刻意的偏离**，各有理由:
          ① 次数与总预算更小 —— 见 ``_ARTIFACT_MAX_RETRIES`` 上的注释。
          ② ⛔ **超时不重试**。照抄先例那条判据:超时时请求**可能已经送达**，
             重传会让用户收到**两份**同样的文件。
          ③ 没有「降级成纯文本」那一路 —— 文件发不出去就是发不出去，
             循环结束即失败,由调用方汇总告诉用户。
        """
        from gateway.platforms.base import BasePlatformAdapter

        server_retry_after: Optional[float] = None
        for attempt in range(self._ARTIFACT_MAX_RETRIES + 1):
            if attempt:
                if server_retry_after is not None:
                    delay = server_retry_after + random.uniform(0, 1)
                    server_retry_after = None
                else:
                    delay = (self._ARTIFACT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                             + random.uniform(0, 1))
                # ⭐ 总预算是**整批共享**的:预算花完就不再等，直接判失败。
                # ⛔ 不许「反正只剩一点，睡完再说」—— 那正是无上限。
                if delay > budget[0]:
                    logger.warning(
                        "kanban notifier: 交付物重试预算用尽，放弃 %s",
                        os.path.basename(path),
                    )
                    return False
                budget[0] -= delay
                await asyncio.sleep(delay)

            try:
                # 🔴 **每次上传都要硬期限。** ``_ARTIFACT_RETRY_BUDGET_S`` 只
                # 限制**重试前的 sleep**,⛔ 不限制上传本身;而 Feishu 等适配器
                # 最终经**无超时**的 ``_run_blocking()`` 等 SDK。notifier 是
                # **串行**处理订阅的 ⇒ 一次卡死会阻断该 profile **后续所有**
                # 任务完成通知和附件交付。
                # ⛔ 超时后**不重传** —— 上传非幂等,重传 = 用户收到重复文件
                # (与上面「超时优先于可重试」同一条判据)。
                res = await asyncio.wait_for(
                    self._send_one_artifact(
                        adapter=adapter, chat_id=chat_id,
                        metadata=metadata, path=path),
                    timeout=self._ARTIFACT_UPLOAD_TIMEOUT_S,
                )
                # ⭐ ⛔ **不写专门的 ``except asyncio.TimeoutError``。**
                # 逆改实证:Python 3.11 里 ``asyncio.TimeoutError`` 就是内建
                # ``TimeoutError``,会落进下面的通用 ``except Exception``,
                # 而那里 ``_is_timeout_error`` **已经**判死不重传(与 SDK 自报
                # 超时同一条路)。删掉专门分支后门仍绿 ⇒ 它**不承重**。
                # 承重的只有上面这个 ``wait_for``(删掉它门会挂死)。
            except Exception as exc:
                # 🔴 ``_is_retryable_error`` 匹配的是 ``connectionreset`` /
                # ``connecterror`` 这类**异常类名**(无空格),⛔ 不是人类可读
                # 的消息文本 —— ``str(ConnectionResetError("connection reset"))``
                # 是 ``"connection reset"``,带空格,**匹配不上**。
                # ⇒ 必须把类名拼进来,与适配器填 ``SendResult.error`` 的做法一致。
                # ⭐ 「传 str(exc) 就能判」这个前提我原本没查,实查才发现是错的。
                err = f"{type(exc).__name__}: {exc}"
                # 🔴 **超时必须先判,且优先级高于「可重试」。**
                #
                # ``ConnectionError("Read timed out")`` 这类异常里,``err`` **同时**
                # 含 ``ConnectionError``(类名 ⇒ ``_is_retryable_error`` 为真)和
                # 超时文本。上一版这条分支**只问了 ``_is_retryable_error``**,于是
                # 同一个**非幂等**的附件被再次上传 ⇒ **用户收到重复文件**。
                #
                # ⭐ 「照抄」三问 —— 先例是下面的 ``SendResult`` 分支(:1288):
                #   ① 先例每个分支做什么:先算 ``transient``(retryable ∪
                #      _is_retryable_error),**再用 ``_is_timeout_error`` 把它压回
                #      False**,注释写着「超时:可能已送达 ⇒ ⛔ 不重传,否则用户
                #      收到两份」。
                #   ② 我这个分支做什么:同序 —— 先超时判死,再问可重试。
                #   ③ 差异:先例还有 ``res.retryable``(平台显式给的),异常分支
                #      拿不到那个字段 ⇒ 少这一项,其余逐字相同。
                # ⛔ 没有自造第二套判据。
                if BasePlatformAdapter._is_timeout_error(err):
                    logger.warning(
                        "kanban notifier: artifact upload (%s) 超时 —— 可能**已经**"
                        "送达,⛔ 不重传以免用户收到两份: %s",
                        os.path.basename(path), safe_exc(exc),
                    )
                    return False
                if not BasePlatformAdapter._is_retryable_error(err):
                    logger.warning(
                        "kanban notifier: artifact upload (%s) failed: %s", path, safe_exc(exc))
                    return False
                logger.warning(
                    "kanban notifier: artifact upload (%s) 瞬时失败 "
                    "(第 %d/%d 次): %s",
                    os.path.basename(path), attempt + 1,
                    self._ARTIFACT_MAX_RETRIES + 1, safe_exc(exc),
                )
                continue

            # ⛔ 判据是 ``SendResult.success``,不是「没抛异常」。
            # 适配器返回 ``None`` 的沿用旧契约(无异常即送达),与 ``:540``
            # 文本通知那条判据逐字一致。
            if getattr(res, "success", True) is not False:
                if attempt:
                    logger.info(
                        "kanban notifier: %s 第 %d 次重试后送达",
                        os.path.basename(path), attempt)
                return True

            err = getattr(res, "error", None) or "unknown"
            retry_after = getattr(res, "retry_after", None)
            if retry_after is not None:
                server_retry_after = float(retry_after)
            transient = (
                bool(getattr(res, "retryable", False))
                or BasePlatformAdapter._is_retryable_error(err)
            )
            # ② 超时:可能已送达 ⇒ ⛔ 不重传,否则用户收到两份。
            if BasePlatformAdapter._is_timeout_error(err):
                transient = False
            logger.warning(
                "kanban notifier: artifact (%s) rejected by platform "
                "(第 %d/%d 次, transient=%s): %s",
                path, attempt + 1, self._ARTIFACT_MAX_RETRIES + 1, transient, err,
            )
            if not transient:
                return False

        return False

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
        When true, the gateway hosts the single dispatcher for this profile:
        no separate `hermes kanban daemon` process needed. When false, the
        loop exits immediately and an external daemon is expected.

        Each tick calls :func:`kanban_db.dispatch_once` inside
        ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
        event loop. Failures in one tick don't stop subsequent ticks —
        same pattern as `_kanban_notifier_watcher`.

        Shutdown: the loop checks ``self._running`` between ticks; gateway
        stop() flips it to False and cancels pending tasks, and the
        in-flight ``to_thread`` returns on its own after the current
        ``dispatch_once`` call finishes (typically <1ms on an idle board).
        """
        # Read config once at boot. If the user flips the flag later, they
        # restart the gateway; same pattern as every other background
        # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
        # as an escape hatch (false-y value disables without editing YAML).
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", safe_exc(exc))
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return

        # Single-dispatcher backstop. dispatch_in_gateway defaults to true, so a
        # new profile gateway (or a same-profile restart race) can silently
        # start a second dispatcher; concurrent dispatchers double reclaim
        # frequency, double claim-attempt events, and — with
        # wal_autocheckpoint=0 — concurrent manual WAL checkpoints can corrupt
        # index pages. The lock lives at the machine-global kanban root
        # (shared across profiles by design), so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info(
                "kanban dispatcher: another gateway already holds the dispatcher "
                "lock (%s); this gateway will NOT dispatch.", _lock_path,
            )
            return
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning(
                "kanban dispatcher: advisory lock unavailable at %s; proceeding "
                "on config control alone.", _lock_path,
            )

        try:
            interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
        except (ValueError, TypeError):
            logger.warning(
                "kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                kanban_cfg.get("dispatch_interval_seconds"),
            )
            interval = 60.0
        interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

        # Read max_spawn config to limit concurrent kanban tasks
        max_spawn = kanban_cfg.get("max_spawn", None)
        if max_spawn is not None:
            logger.info("kanban dispatcher: max_spawn=%s", max_spawn)

        # Cap the number of simultaneously running tasks so slow workers
        # (local LLMs, resource-constrained hosts) don't pile up and time
        # out. When set, the dispatcher skips spawning when the board
        # already has this many tasks in 'running' status.
        raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
        max_in_progress = None
        if raw_max_in_progress is not None:
            try:
                max_in_progress = int(raw_max_in_progress)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                if max_in_progress < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                        raw_max_in_progress,
                    )
                    max_in_progress = None
                else:
                    logger.info("kanban dispatcher: max_in_progress=%s", max_in_progress)

        raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
        try:
            failure_limit = int(raw_failure_limit)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT
        if failure_limit < 1:
            logger.warning(
                "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT

        # Read stale_timeout_seconds — 0 disables stale detection.
        raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
        try:
            stale_timeout_seconds = int(raw_stale or 0)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                "disabling stale detection",
                raw_stale,
            )
            stale_timeout_seconds = 0

        # Read kanban.default_assignee — fallback profile for tasks
        # created without an explicit assignee (e.g. via the dashboard).
        # When set, the dispatcher applies it to unassigned ready tasks
        # instead of skipping them indefinitely (#27145). Empty string
        # (the schema default) means "no fallback, keep skipping" —
        # backward-compatible with existing installs.
        default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
        if default_assignee:
            logger.info(
                "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                "will route to this profile)",
                default_assignee,
            )

        # Read kanban.max_in_progress_per_profile — per-profile concurrency
        # cap (#21582). When set, no single profile gets more than N
        # workers running at once, even if the global max_in_progress
        # would allow it. Prevents one profile's local model / API quota
        # / browser pool from being overwhelmed by a fan-out.
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
        max_in_progress_per_profile = None
        if raw_per_profile is not None:
            try:
                max_in_progress_per_profile = int(raw_per_profile)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                if max_in_progress_per_profile < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                        raw_per_profile,
                    )
                    max_in_progress_per_profile = None
                else:
                    logger.info(
                        "kanban dispatcher: max_in_progress_per_profile=%d",
                        max_in_progress_per_profile,
                    )

        # Initial delay so the gateway finishes wiring adapters before the
        # dispatcher spawns workers (those workers may hit gateway notify
        # subscriptions etc.). Matches the notifier watcher's delay.
        await asyncio.sleep(5)

        # Health telemetry mirrored from `_cmd_daemon`: warn when ready
        # queue is non-empty but spawns are 0 for N consecutive ticks —
        # usually means broken PATH, missing venv, or credential loss.
        HEALTH_WINDOW = 6
        bad_ticks = 0
        last_warn_at = 0
        # Avoid hot-looping corrupt-looking board DBs, but do not suppress
        # same-fingerprint retries forever: transient WAL/open races can
        # surface as "database disk image is malformed" for one tick.
        CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
        disabled_corrupt_boards: dict[
            str, tuple[tuple[str, int | None, int | None], float]
        ] = {}

        def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
            path = _kb.kanban_db_path(slug)
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            try:
                stat = path.stat()
            except OSError:
                return (resolved, None, None)
            return (resolved, stat.st_mtime_ns, stat.st_size)

        def _is_corrupt_board_db_error(exc: Exception) -> bool:
            corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
            if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
                return True
            if not isinstance(exc, sqlite3.DatabaseError):
                return False
            msg = str(exc).lower()
            return (
                "file is not a database" in msg
                or "database disk image is malformed" in msg
            )

        def _tick_once_for_board(slug: str) -> "Optional[object]":
            """Run one dispatch_once for a specific board.

            Runs in a worker thread via `asyncio.to_thread`. `board=slug`
            is passed through `dispatch_once` so `resolve_workspace` and
            `_default_spawn` see the right paths. The per-board DB is
            opened explicitly so concurrent boards never share a
            connection handle or accidentally claim across each other.
            """
            conn = None
            fingerprint = _board_db_fingerprint(slug)
            disabled_entry = disabled_corrupt_boards.get(slug)
            if disabled_entry is not None:
                disabled_fingerprint, disabled_at = disabled_entry
                age = time.monotonic() - disabled_at
                if (
                    disabled_fingerprint == fingerprint
                    and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
                ):
                    return None
                if disabled_fingerprint == fingerprint:
                    logger.info(
                        "kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch",
                        slug,
                        age,
                    )
                else:
                    logger.info(
                        "kanban dispatcher: board %s database changed; retrying dispatch",
                        slug,
                    )
                disabled_corrupt_boards.pop(slug, None)
            try:
                conn = _kb.connect(board=slug)
                # `connect()` runs the schema + idempotent migration on
                # first open per process; the previous explicit
                # `init_db()` call here busted the per-process cache and
                # re-ran the migration on a second connection, racing
                # the first. See the matching comment in
                # `_kanban_notifier_watcher` and issue #21378.
                return _kb.dispatch_once(
                    conn,
                    board=slug,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    stale_timeout_seconds=stale_timeout_seconds,
                    default_assignee=default_assignee,
                    max_in_progress_per_profile=max_in_progress_per_profile,
                )
            except sqlite3.DatabaseError as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            except Exception as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _tick_once() -> "list[tuple[str, Optional[object]]]":
            """Run one dispatch_once per board. Returns (slug, result) pairs.

            Enumerating boards on every tick keeps the dispatcher honest
            when users create a new board mid-run: no restart required,
            the next tick picks it up automatically.
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            out: list[tuple[str, "Optional[object]"]] = []
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                out.append((slug, _tick_once_for_board(slug)))
            return out

        def _ready_nonempty() -> bool:
            """Cheap probe: is there at least one ready+assigned+unclaimed
            task on ANY board whose assignee maps to a real Hermes profile
            (i.e. one the dispatcher would actually spawn for)?

            Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
            ``orion-research``) are pulled by terminals via
            ``claim_task`` directly and never spawnable, so a queue full
            of those is "correctly idle", not "stuck". Filtering them out
            here keeps the stuck-warn fire only on real failures (broken
            PATH, missing venv, credential loss for a real Hermes profile).
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                conn = None
                try:
                    conn = _kb.connect(board=slug)
                    if _kb.has_spawnable_ready(conn):
                        return True
                    if _kb.has_spawnable_review(conn):
                        return True
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
            return False

        # Auto-decompose: turn fresh triage tasks into ready workgraphs
        # before the dispatcher fans out workers. Gated by
        # ``kanban.auto_decompose`` (default True). Capped by
        # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
        # of triage tasks doesn't burst-spend the aux LLM in one tick;
        # remainder defers to subsequent ticks.
        #
        # The flag is re-read from config EVERY tick (#49638) rather than
        # captured once at boot. Auto-decompose is a safety toggle: a user who
        # sees it fan out and run tasks they didn't intend reaches for
        # ``kanban.auto_decompose: false`` to STOP it — and that must take
        # effect on the next tick, not require a gateway restart. (Reported:
        # auto-decompose created and launched destructive tasks while the user
        # was still typing the task description, and the flag "couldn't be
        # disabled" because the gateway had captured its boot-time value.)
        def _read_auto_decompose_settings() -> tuple[bool, int]:
            """Re-resolve (enabled, per_tick) from current config each tick."""
            return _resolve_auto_decompose_settings(_load_config)

        def _auto_decompose_tick(auto_decompose_per_tick: int) -> int:
            """Run the auto-decomposer for up to N triage tasks across all
            boards. Returns the number of triage tasks that were
            successfully decomposed or specified this tick.
            """
            try:
                from hermes_cli import kanban_decompose as _decomp
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "kanban auto-decompose: import failed (%s); skipping", safe_exc(exc),
                )
                return 0
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            attempted = 0
            successes = 0
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin this board for the duration of the call — same
                # pattern as the dashboard specify endpoint. The
                # decomposer module connects with no board kwarg and
                # relies on the env var.
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug(
                            "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                            slug, safe_exc(exc),
                        )
                        triage_ids = []
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        attempted += 1
                        try:
                            outcome = _decomp.decompose_task(
                                tid, author="auto-decomposer",
                            )
                        except Exception:
                            logger.exception(
                                "kanban auto-decompose: decompose_task crashed on %s",
                                tid,
                            )
                            continue
                        if outcome.ok:
                            successes += 1
                            if outcome.fanout and outcome.child_ids:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → %d children",
                                    slug, tid, len(outcome.child_ids),
                                )
                            else:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                    slug, tid,
                                )
                        else:
                            # Common no-op reasons (no aux client configured) shouldn't
                            # spam logs every tick. Log at debug.
                            logger.debug(
                                "kanban auto-decompose [%s]: %s skipped: %s",
                                slug, tid, outcome.reason,
                            )
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
            return successes

        logger.info(
            "kanban dispatcher: embedded in gateway (interval=%.1fs)", interval
        )
        while self._running:
            try:
                # Reap zombie children before per-board work so a board DB
                # failure cannot block cleanup of unrelated workers.
                pids = await asyncio.to_thread(_kb.reap_worker_zombies)
                if pids:
                    logger.info(
                        "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                        len(pids),
                        pids,
                    )
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Re-read the auto-decompose toggle live each tick so a user
                # flipping kanban.auto_decompose=false to STOP runaway fan-out
                # takes effect on the next tick, not on gateway restart (#49638).
                _ad_enabled, _ad_per_tick = _read_auto_decompose_settings()
                if _ad_enabled:
                    await asyncio.to_thread(_auto_decompose_tick, _ad_per_tick)
                results = await asyncio.to_thread(_tick_once)
                any_spawned = False
                for slug, res in (results or []):
                    if res is not None and getattr(res, "spawned", None):
                        any_spawned = True
                        # Quiet by default — only log when something actually
                        # happened, so an idle gateway stays silent.
                        logger.info(
                            "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                            "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                            slug,
                            len(res.spawned),
                            res.reclaimed,
                            len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
                            len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
                            res.promoted,
                            len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
                        )
                # Health telemetry (aggregate across boards)
                ready_pending = await asyncio.to_thread(_ready_nonempty)
                if ready_pending and not any_spawned:
                    bad_ticks += 1
                else:
                    bad_ticks = 0
                if bad_ticks >= HEALTH_WINDOW:
                    now = int(time.time())
                    if now - last_warn_at >= 300:
                        logger.warning(
                            "kanban dispatcher stuck: ready queue non-empty for "
                            "%d consecutive ticks but 0 workers spawned. Check "
                            "profile health (venv, PATH, credentials) and "
                            "`hermes kanban list --status ready`.",
                            bad_ticks,
                        )
                        last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                self._release_kanban_dispatcher_lock()
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

        self._release_kanban_dispatcher_lock()
