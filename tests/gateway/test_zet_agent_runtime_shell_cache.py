"""Lifecycle contracts for the ordinary Zet Agent runtime-shell cache."""

import pytest

from gateway.zet_agent_runtime_cache import RuntimeShellCache, RuntimeShellCacheKey


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _key(session: str, *, profile: str = "main") -> RuntimeShellCacheKey:
    return RuntimeShellCacheKey(
        profile_home=f"/profiles/{profile}",
        profile_name=profile,
        gateway_session_key=f"/profiles/{profile}|{session}",
        session_id=session,
    )


def _reserve(
    cache: RuntimeShellCache,
    key: RuntimeShellCacheKey,
    agent: object,
    *,
    signature: str = "sig",
    message_count: int = 0,
):
    decision = cache.reserve_new(
        key,
        signature=signature,
        message_count=message_count,
        agent=agent,
    )
    assert decision.agent is agent
    assert decision.lease is not None
    assert decision.reason == "reserved"
    return decision.lease


def test_exact_revision_and_signature_reuses_one_exclusive_shell():
    cache = RuntimeShellCache(capacity=2, idle_ttl_seconds=60)
    key = _key("session-1")
    agent = object()

    first_lease = _reserve(cache, key, agent)
    assert cache.acquire(key, signature="sig", message_count=0).reason == "busy"

    released = cache.finish(first_lease, reusable=True, message_count=2)
    assert released.reason == "released"
    assert cache.counts() == {"entries": 1, "idle": 1, "leased": 0}

    hit = cache.acquire(key, signature="sig", message_count=2)
    assert hit.reason == "hit"
    assert hit.agent is agent
    assert hit.lease is not None


@pytest.mark.parametrize(
    ("signature", "message_count", "reason"),
    [
        ("new-sig", 4, "signature_changed"),
        ("sig", 5, "message_count_changed"),
        ("sig", None, "message_count_unavailable"),
    ],
)
def test_mismatch_fails_closed_and_retires_old_shell(
    signature, message_count, reason
):
    cache = RuntimeShellCache(capacity=2, idle_ttl_seconds=60)
    key = _key("session-1")
    agent = object()
    lease = _reserve(cache, key, agent, message_count=4)
    cache.finish(lease, reusable=True, message_count=4)

    decision = cache.acquire(
        key,
        signature=signature,
        message_count=message_count,
    )

    assert decision.reason == reason
    assert decision.agent is None
    assert decision.retired_agents == (agent,)
    assert cache.counts() == {"entries": 0, "idle": 0, "leased": 0}


def test_concurrent_same_session_uses_temporary_shell_without_replacing_owner():
    cache = RuntimeShellCache(capacity=2, idle_ttl_seconds=60)
    key = _key("session-1")
    owner = object()
    temporary = object()
    owner_lease = _reserve(cache, key, owner)

    assert cache.acquire(key, signature="sig", message_count=0).reason == "busy"
    race = cache.reserve_new(
        key,
        signature="sig",
        message_count=0,
        agent=temporary,
    )

    assert race.reason == "publish_race"
    assert race.agent is temporary
    assert race.lease is None
    cache.finish(owner_lease, reusable=True, message_count=2)
    hit = cache.acquire(key, signature="sig", message_count=2)
    assert hit.agent is owner

    duplicate_finish = cache.finish(
        owner_lease,
        reusable=True,
        message_count=2,
    )
    assert duplicate_finish.reason == "stale_lease"
    assert duplicate_finish.retired_agents == ()
    assert cache.counts() == {"entries": 1, "idle": 0, "leased": 1}


def test_lru_evicts_only_idle_shells_and_hard_cap_never_evicts_leases():
    cache = RuntimeShellCache(capacity=2, idle_ttl_seconds=60)
    key_a = _key("a")
    key_b = _key("b")
    key_c = _key("c")
    agent_a, agent_b, agent_c = object(), object(), object()

    lease_a = _reserve(cache, key_a, agent_a)
    cache.finish(lease_a, reusable=True, message_count=1)
    lease_b = _reserve(cache, key_b, agent_b)
    cache.finish(lease_b, reusable=True, message_count=1)

    # Touch A, making B the least recently used idle entry.
    hit_a = cache.acquire(key_a, signature="sig", message_count=1)
    cache.finish(hit_a.lease, reusable=True, message_count=1)
    lease_c = cache.reserve_new(
        key_c,
        signature="sig",
        message_count=0,
        agent=agent_c,
    )
    assert lease_c.retired_agents == (agent_b,)

    # With both remaining entries leased, another session stays temporary.
    hit_a = cache.acquire(key_a, signature="sig", message_count=1)
    busy = cache.reserve_new(
        _key("d"),
        signature="sig",
        message_count=0,
        agent=object(),
    )
    assert busy.reason == "capacity_busy"
    assert busy.lease is None
    assert cache.counts() == {"entries": 2, "idle": 0, "leased": 2}
    assert hit_a.lease is not None
    assert lease_c.lease is not None


def test_ttl_expires_idle_but_never_an_active_lease():
    clock = _Clock()
    cache = RuntimeShellCache(
        capacity=2,
        idle_ttl_seconds=10,
        clock=clock,
    )
    idle_agent, active_agent = object(), object()
    idle_lease = _reserve(cache, _key("idle"), idle_agent)
    cache.finish(idle_lease, reusable=True, message_count=1)
    active_lease = _reserve(cache, _key("active"), active_agent)

    clock.advance(11)
    decision = cache.acquire(
        _key("missing"),
        signature="sig",
        message_count=0,
    )

    assert decision.reason == "miss"
    assert decision.retired_agents == (idle_agent,)
    assert cache.counts() == {"entries": 1, "idle": 0, "leased": 1}
    cache.finish(active_lease, reusable=True, message_count=1)


def test_profile_detach_is_isolated_and_rejects_active_target_shell():
    cache = RuntimeShellCache(capacity=3, idle_ttl_seconds=60)
    main_agent, coder_agent = object(), object()
    main_lease = _reserve(cache, _key("main-1"), main_agent)
    coder_lease = _reserve(cache, _key("coder-1", profile="coder"), coder_agent)
    cache.finish(coder_lease, reusable=True, message_count=1)

    with pytest.raises(RuntimeError, match="leased"):
        cache.detach_profile("/profiles/main")
    assert cache.detach_profile("/profiles/coder") == (coder_agent,)
    assert cache.counts() == {"entries": 1, "idle": 0, "leased": 1}

    cache.finish(main_lease, reusable=True, message_count=1)
    assert cache.detach_profile("/profiles/main") == (main_agent,)


def test_stop_detaches_idle_and_retires_active_shell_on_finish():
    cache = RuntimeShellCache(capacity=2, idle_ttl_seconds=60)
    idle_agent, active_agent = object(), object()
    idle_lease = _reserve(cache, _key("idle"), idle_agent)
    cache.finish(idle_lease, reusable=True, message_count=1)
    active_lease = _reserve(cache, _key("active"), active_agent)

    assert cache.stop() == (idle_agent,)
    assert cache.acquire(
        _key("new"), signature="sig", message_count=0
    ).reason == "stopped"
    finished = cache.finish(active_lease, reusable=True, message_count=1)
    assert finished.reason == "retired"
    assert finished.retired_agents == (active_agent,)
    assert cache.counts() == {"entries": 0, "idle": 0, "leased": 0}
