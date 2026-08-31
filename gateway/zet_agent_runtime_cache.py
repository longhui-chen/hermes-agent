"""Bounded, exclusive cache for ordinary Zet ``AIAgent`` runtime shells.

The cache deliberately owns only lifecycle bookkeeping.  It never reads
profile files, SessionDB, credentials, or agent fields while holding its lock;
callers prepare signatures/revisions before entry and release retired agents
after the lock is gone.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class RuntimeShellCacheKey:
    """Identity of one reusable ordinary Zet conversation runtime."""

    profile_home: str
    profile_name: str
    gateway_session_key: str
    session_id: str


@dataclass(frozen=True)
class RuntimeShellLease:
    """Opaque proof that one caller exclusively owns a cached entry."""

    key: RuntimeShellCacheKey
    agent: Any
    token: object


@dataclass(frozen=True)
class RuntimeShellCacheDecision:
    agent: Optional[Any]
    lease: Optional[RuntimeShellLease]
    retired_agents: tuple[Any, ...]
    reason: str


@dataclass
class _RuntimeShellEntry:
    agent: Any
    signature: str
    message_count: Optional[int]
    last_used: float
    lease_token: Optional[object] = None
    retire_on_finish: bool = False


class RuntimeShellCache:
    """Small LRU/TTL cache whose entries are never shared concurrently."""

    def __init__(
        self,
        *,
        capacity: int,
        idle_ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity < 1:
            raise ValueError("runtime shell cache capacity must be positive")
        if idle_ttl_seconds <= 0:
            raise ValueError("runtime shell cache idle TTL must be positive")
        self._capacity = int(capacity)
        self._idle_ttl_seconds = float(idle_ttl_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: "OrderedDict[RuntimeShellCacheKey, _RuntimeShellEntry]" = (
            OrderedDict()
        )
        self._accepting = True

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def idle_ttl_seconds(self) -> float:
        return self._idle_ttl_seconds

    def _expire_idle_locked(self, now: float) -> list[Any]:
        retired: list[Any] = []
        for key, entry in list(self._entries.items()):
            if entry.lease_token is not None:
                continue
            if now - entry.last_used <= self._idle_ttl_seconds:
                continue
            self._entries.pop(key, None)
            retired.append(entry.agent)
        return retired

    def acquire(
        self,
        key: RuntimeShellCacheKey,
        *,
        signature: str,
        message_count: Optional[int],
    ) -> RuntimeShellCacheDecision:
        """Lease a matching idle entry, or return a fail-closed miss reason."""

        now = self._clock()
        with self._lock:
            retired = self._expire_idle_locked(now)
            if not self._accepting:
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "stopped"
                )
            entry = self._entries.get(key)
            if entry is None:
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "miss"
                )
            if entry.lease_token is not None:
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "busy"
                )
            if entry.signature != signature:
                self._entries.pop(key, None)
                retired.append(entry.agent)
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "signature_changed"
                )
            if message_count is None or entry.message_count is None:
                self._entries.pop(key, None)
                retired.append(entry.agent)
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "message_count_unavailable"
                )
            if entry.message_count != message_count:
                self._entries.pop(key, None)
                retired.append(entry.agent)
                return RuntimeShellCacheDecision(
                    None, None, tuple(retired), "message_count_changed"
                )

            token = object()
            entry.lease_token = token
            entry.last_used = now
            self._entries.move_to_end(key)
            return RuntimeShellCacheDecision(
                entry.agent,
                RuntimeShellLease(key=key, agent=entry.agent, token=token),
                tuple(retired),
                "hit",
            )

    def reserve_new(
        self,
        key: RuntimeShellCacheKey,
        *,
        signature: str,
        message_count: Optional[int],
        agent: Any,
    ) -> RuntimeShellCacheDecision:
        """Publish a newly built agent as leased, respecting the hard cap."""

        now = self._clock()
        with self._lock:
            retired = self._expire_idle_locked(now)
            if not self._accepting:
                return RuntimeShellCacheDecision(
                    agent, None, tuple(retired), "stopped"
                )
            if key in self._entries:
                # A defensive concurrent miss won the publication race.  The
                # incoming agent may finish its turn, but never enters cache.
                return RuntimeShellCacheDecision(
                    agent, None, tuple(retired), "publish_race"
                )

            while len(self._entries) >= self._capacity:
                idle_key = next(
                    (
                        candidate_key
                        for candidate_key, candidate in self._entries.items()
                        if candidate.lease_token is None
                    ),
                    None,
                )
                if idle_key is None:
                    return RuntimeShellCacheDecision(
                        agent, None, tuple(retired), "capacity_busy"
                    )
                idle_entry = self._entries.pop(idle_key)
                retired.append(idle_entry.agent)

            token = object()
            self._entries[key] = _RuntimeShellEntry(
                agent=agent,
                signature=signature,
                message_count=message_count,
                last_used=now,
                lease_token=token,
            )
            return RuntimeShellCacheDecision(
                agent,
                RuntimeShellLease(key=key, agent=agent, token=token),
                tuple(retired),
                "reserved",
            )

    def finish(
        self,
        lease: RuntimeShellLease,
        *,
        reusable: bool,
        message_count: Optional[int],
    ) -> RuntimeShellCacheDecision:
        """Return a lease to IDLE, or detach it for caller-owned cleanup."""

        now = self._clock()
        with self._lock:
            entry = self._entries.get(lease.key)
            if entry is not None and entry.agent is lease.agent:
                if entry.lease_token is not lease.token:
                    # Duplicate/late cleanup for an older lease must never
                    # release the same Agent after a newer turn acquired it.
                    return RuntimeShellCacheDecision(
                        entry.agent, None, (), "stale_lease"
                    )
            elif entry is None or entry.agent is not lease.agent:
                return RuntimeShellCacheDecision(
                    None, None, (lease.agent,), "lease_lost"
                )
            if (
                not reusable
                or message_count is None
                or entry.retire_on_finish
                or not self._accepting
            ):
                self._entries.pop(lease.key, None)
                return RuntimeShellCacheDecision(
                    None, None, (entry.agent,), "retired"
                )

            entry.message_count = message_count
            entry.last_used = now
            entry.lease_token = None
            self._entries.move_to_end(lease.key)
            return RuntimeShellCacheDecision(
                entry.agent, None, (), "released"
            )

    def agents_for_prompt_invalidation(
        self, *, profile_home: Optional[str] = None
    ) -> tuple[Any, ...]:
        """Snapshot matching agents; callers invalidate them outside the lock."""

        with self._lock:
            return tuple(
                entry.agent
                for key, entry in self._entries.items()
                if profile_home is None or key.profile_home == profile_home
            )

    def detach_profile(self, profile_home: str) -> tuple[Any, ...]:
        """Detach one profile's idle entries for hard profile-unload cleanup."""

        with self._lock:
            matching = [
                (key, entry)
                for key, entry in self._entries.items()
                if key.profile_home == profile_home
            ]
            if any(entry.lease_token is not None for _, entry in matching):
                raise RuntimeError(
                    "profile unload cannot close a leased Zet runtime shell"
                )
            for key, _ in matching:
                self._entries.pop(key, None)
            return tuple(entry.agent for _, entry in matching)

    def stop(self) -> tuple[Any, ...]:
        """Stop new leases, detach idle entries, retire active ones on finish."""

        with self._lock:
            self._accepting = False
            detached: list[Any] = []
            for key, entry in list(self._entries.items()):
                if entry.lease_token is None:
                    self._entries.pop(key, None)
                    detached.append(entry.agent)
                else:
                    entry.retire_on_finish = True
            return tuple(detached)

    def counts(self) -> dict[str, int]:
        with self._lock:
            leased = sum(
                1 for entry in self._entries.values()
                if entry.lease_token is not None
            )
            return {
                "entries": len(self._entries),
                "idle": len(self._entries) - leased,
                "leased": leased,
            }
