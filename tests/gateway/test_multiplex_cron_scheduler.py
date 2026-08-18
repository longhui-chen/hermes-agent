"""Multiplex gateway cron scheduler scoping."""

import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.run import (
    _ProfileAdapterCleanupError,
    _ensure_profile_cron_adapters,
    _multiplex_cron_profiles,
    _start_gateway_cron_schedulers,
    _unload_profile_cron_adapters,
    _wait_future_interruptibly,
)


def test_cron_does_not_publish_scheduler_when_profile_cleanup_is_pending(
    tmp_path, monkeypatch
):
    """cleanup ledger 存在时不能用空 map 登记假成功 scheduler。"""
    import gateway.run as gateway_run

    profile_home = tmp_path / "profiles" / "worker"
    sibling = object()

    async def failing_start(*_args):
        raise _ProfileAdapterCleanupError("cleanup failed")

    runner = SimpleNamespace(
        adapters={},
        _profile_adapters={"worker": {"slack": sibling}},
        _partial_adapter_cleanup_retry={
            ("worker", "feishu"): object(),
        },
        _profile_runtime_unload_retry=set(),
        _profile_runtime_unloads={},
        _start_one_profile_adapters=failing_start,
    )

    scheduled = []

    def schedule(coro, *_args, **_kwargs):
        scheduled.append(coro)
        coro.close()
        future = concurrent.futures.Future()
        future.set_exception(_ProfileAdapterCleanupError("cleanup failed"))
        return future

    monkeypatch.setattr(gateway_run, "_multiplex_active_profile_name", lambda: "default")
    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", schedule)

    with pytest.raises(_ProfileAdapterCleanupError, match="cleanup failed"):
        _ensure_profile_cron_adapters(
            runner, "worker", profile_home, loop=object()
        )

    assert scheduled
    assert runner._profile_adapters["worker"]["slack"] is sibling


def test_cron_does_not_use_published_fatal_owner_while_cleanup_is_pending(
    tmp_path, monkeypatch
):
    """published fatal owner 仍在 cleanup 时不能被 cron 当成 ready。"""
    import gateway.run as gateway_run

    profile_home = tmp_path / "profiles" / "worker"
    stale = object()
    owner_map = {"feishu": stale}
    runner = SimpleNamespace(
        adapters={},
        _profile_adapters={"worker": owner_map},
        _partial_adapter_cleanup_retry={},
        _published_adapter_cleanup_retry={
            ("worker", "feishu"): (stale, owner_map),
        },
        _retiring_adapter_cleanups={},
        _profile_adapter_operations={},
        _profile_runtime_unload_retry=set(),
        _profile_runtime_unloads={},
    )

    monkeypatch.setattr(gateway_run, "_multiplex_active_profile_name", lambda: "default")

    assert (
        _ensure_profile_cron_adapters(
            runner, "worker", profile_home, loop=object()
        )
        is None
    )
    assert runner._profile_adapters["worker"]["feishu"] is stale


def test_cron_does_not_start_while_profile_operation_is_pending(tmp_path, monkeypatch):
    """独立 startup operation owner 未结束时不能并发起第二份 adapter。"""
    import gateway.run as gateway_run

    stale = object()
    runner = SimpleNamespace(
        adapters={},
        _profile_adapters={"worker": {"feishu": stale}},
        _partial_adapter_cleanup_retry={},
        _published_adapter_cleanup_retry={},
        _retiring_adapter_cleanups={},
        _profile_adapter_operations={"worker": {object()}},
        _profile_runtime_unload_retry=set(),
        _profile_runtime_unloads={},
    )
    monkeypatch.setattr(gateway_run, "_multiplex_active_profile_name", lambda: "default")

    assert (
        _ensure_profile_cron_adapters(
            runner, "worker", tmp_path / "profiles" / "worker", loop=object()
        )
        is None
    )
    assert runner._profile_adapters["worker"]["feishu"] is stale


def test_cron_does_not_start_scheduler_after_adapter_ensure_failure(
    tmp_path, monkeypatch
):
    """adapter ensure 失败时 reconciler 必须留空 entry 供下一轮重试。"""
    import gateway.run as gateway_run

    started = []
    monkeypatch.setattr(
        gateway_run, "_ensure_profile_cron_adapters", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        gateway_run,
        "_run_profile_cron_scheduler",
        lambda *_args, **_kwargs: started.append(True),
    )

    runner = SimpleNamespace(_draining=False, _external_drain_active=False)
    assert (
        gateway_run._start_profile_cron_scheduler_thread(
            runner,
            "worker",
            tmp_path / "profiles" / "worker",
            threading.Event(),
            loop=object(),
        )
        is None
    )
    assert started == []


@pytest.mark.parametrize("failure_mode", ["blocked", "no_loop", "future_none", "startup_error"])
def test_cron_adapter_ensure_failure_returns_no_scheduler_adapters(
    tmp_path, monkeypatch, failure_mode
):
    """各类 adapter ensure 失败都必须让 reconciler 留空 entry 重试。"""
    import gateway.run as gateway_run

    profile_home = tmp_path / "profiles" / "worker"
    runner = SimpleNamespace(
        adapters={},
        _profile_adapters={},
        _profile_runtime_unload_retry=set(),
        _profile_runtime_unloads=(
            {"worker": object()} if failure_mode == "blocked" else {}
        ),
        _start_one_profile_adapters=None,
    )
    loop = None if failure_mode == "no_loop" else object()
    if failure_mode in {"future_none", "startup_error"}:
        async def start_one(*_args):
            return 1

        runner._start_one_profile_adapters = start_one

        def schedule(coro, *_args, **_kwargs):
            coro.close()
            if failure_mode == "future_none":
                return None
            future = concurrent.futures.Future()
            future.set_exception(RuntimeError("startup error"))
            return future

        monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", schedule)

    monkeypatch.setattr(gateway_run, "_multiplex_active_profile_name", lambda: "default")
    assert (
        _ensure_profile_cron_adapters(
            runner, "worker", profile_home, loop=loop
        )
        is None
    )


def test_multiplex_cron_profiles_skip_legacy_default_when_main_exists(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    family_home = default_home / "profiles" / "family-manager"

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", default_home),
            ("family-manager", family_home),
            ("main", main_home),
        ],
    )

    assert _multiplex_cron_profiles() == [
        ("family-manager", family_home),
        ("main", main_home),
    ]


def test_multiplex_cron_scheduler_runs_each_profile_under_profile_home(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    seen = []
    seen_event = threading.Event()
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    family_home = default_home / "profiles" / "family-manager"
    main_adapters = {"api_server": object()}
    family_adapters = {"api_server": object()}

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            seen.append((get_hermes_home(), adapters))
            if len(seen) >= 2:
                seen_event.set()
            stop_event.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", default_home),
            ("family-manager", family_home),
            ("main", main_home),
        ],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={
            "family-manager": family_adapters,
            "main": main_adapters,
        },
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        assert seen_event.wait(timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(seen, key=lambda item: str(item[0])) == sorted(
        [
            (family_home, family_adapters),
            (main_home, main_adapters),
        ],
        key=lambda item: str(item[0]),
    )


def test_multiplex_inprocess_cron_pauses_dispatch_while_gateway_drains(
    tmp_path, monkeypatch
):
    from cron.scheduler_provider import InProcessCronScheduler

    main_home = tmp_path / ".hermes" / "profiles" / "main"
    dispatch_states = []
    started = threading.Event()

    class RecordingScheduler(InProcessCronScheduler):
        def start(
            self,
            stop_event,
            *,
            adapters=None,
            loop=None,
            interval=60,
            can_dispatch=None,
        ):
            dispatch_states.append(can_dispatch())
            runner._external_drain_active = True
            dispatch_states.append(can_dispatch())
            started.set()

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("main", main_home)],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "main")
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingScheduler(),
    )

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={},
        _draining=False,
        _external_drain_active=False,
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        assert started.wait(timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert dispatch_states == [True, False]


def test_nonmultiplex_inprocess_cron_pauses_dispatch_while_gateway_drains(monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler

    dispatch_states = []

    class RecordingScheduler(InProcessCronScheduler):
        def start(
            self,
            stop_event,
            *,
            adapters=None,
            loop=None,
            interval=60,
            can_dispatch=None,
        ):
            dispatch_states.append(can_dispatch())
            runner._draining = True
            dispatch_states.append(can_dispatch())

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingScheduler(),
    )

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=False),
        adapters={},
        _draining=False,
        _external_drain_active=False,
    )
    threads = _start_gateway_cron_schedulers(runner, threading.Event())
    for thread in threads:
        thread.join(timeout=2)

    assert dispatch_states == [True, False]


def test_multiplex_cron_reconciler_starts_new_profiles(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    worker_home = default_home / "profiles" / "worker"
    served = {"main": main_home}
    seen_homes = []
    seen_worker = threading.Event()

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            home = get_hermes_home()
            seen_homes.append(home)
            if home == worker_home:
                seen_worker.set()
            stop_event.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: list(served.items()),
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={
            "main": {"api_server": object()},
            "worker": {"api_server": object()},
        },
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        served["worker"] = worker_home
        assert seen_worker.wait(timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert main_home in seen_homes
    assert worker_home in seen_homes


def test_multiplex_cron_reconciler_does_not_restart_returning_provider(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    starts = 0
    unloaded = []

    class ReturningScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            nonlocal starts
            starts += 1

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("main", main_home)],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: ReturningScheduler())

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={"main": {"api_server": object()}},
        unload_profile_runtime=lambda profile_name, **_kwargs: unloaded.append(profile_name),
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        assert _wait_until(lambda: starts == 1, timeout=2)
        threading.Event().wait(timeout=0.15)
        assert starts == 1
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)
    assert unloaded == []


def test_multiplex_cron_reconciler_restarts_failed_provider(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    starts = 0

    class FailingThenBlockingScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            nonlocal starts
            starts += 1
            if starts == 1:
                raise RuntimeError("boom")
            stop_event.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("main", main_home)],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: FailingThenBlockingScheduler(),
    )

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={"main": {"api_server": object()}},
        unload_profile_runtime=lambda profile_name, **_kwargs: None,
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        assert _wait_until(lambda: starts >= 2, timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)


def test_multiplex_cron_reconciler_uses_primary_adapters_for_active_profile(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    main_adapters = {"api_server": object()}
    seen = []
    seen_event = threading.Event()

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            seen.append((get_hermes_home(), adapters))
            seen_event.set()
            stop_event.wait(timeout=5)

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("main", main_home)],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "main")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    def fail_start_one_profile_adapters(*args, **kwargs):
        raise AssertionError("active profile must use runner.adapters")

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters=main_adapters,
        _profile_adapters={},
        _start_one_profile_adapters=fail_start_one_profile_adapters,
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event, reconcile_interval=0.05)
    try:
        assert seen_event.wait(timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert seen == [(main_home, main_adapters)]


@pytest.mark.asyncio
async def test_multiplex_cron_reconciler_starts_adapters_for_new_profiles(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    worker_home = default_home / "profiles" / "worker"
    served = {"main": main_home}
    worker_adapters = {"api_server": object()}
    seen_worker = threading.Event()
    seen = []

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            seen.append((get_hermes_home(), adapters))
            if get_hermes_home() == worker_home and adapters is worker_adapters:
                seen_worker.set()
            stop_event.wait(timeout=5)

    async def start_one_profile_adapters(profile_name, profile_home, claimed):
        assert profile_name == "worker"
        assert profile_home == worker_home
        runner._profile_adapters[profile_name] = worker_adapters
        return 1

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: list(served.items()),
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={"main": {"api_server": object()}},
        _adapter_credential_fingerprint=lambda adapter: None,
        _start_one_profile_adapters=start_one_profile_adapters,
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(
        runner,
        stop_event,
        loop=asyncio.get_running_loop(),
        reconcile_interval=0.05,
    )
    try:
        served["worker"] = worker_home
        assert await _wait_thread_event(seen_worker, timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        thread.join(timeout=2)

    assert (worker_home, worker_adapters) in seen


@pytest.mark.asyncio
async def test_multiplex_cron_reconciler_stops_removed_profiles(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    retired_home = default_home / "profiles" / "retired"
    served = {"main": main_home, "retired": retired_home}
    seen_retired = threading.Event()
    stopped_retired = threading.Event()
    unload_retried = threading.Event()
    unloaded = []

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            home = get_hermes_home()
            if home == retired_home:
                seen_retired.set()
            stop_event.wait(timeout=5)
            if home == retired_home:
                stopped_retired.set()

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: list(served.items()),
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    async def unload_profile_runtime(profile_name, *, profile_home):
        unloaded.append((profile_name, profile_home))
        if len(unloaded) == 1:
            return {"blocked": True, "active_sessions": 1}
        unload_retried.set()
        return {}

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={
            "main": {"api_server": object()},
            "retired": {"api_server": object()},
        },
        unload_profile_runtime=unload_profile_runtime,
    )
    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(
        runner,
        stop_event,
        loop=asyncio.get_running_loop(),
        reconcile_interval=0.05,
    )
    try:
        assert await _wait_thread_event(seen_retired, timeout=2)
        served.pop("retired")
        assert await _wait_thread_event(stopped_retired, timeout=2)
        assert await _wait_thread_event(unload_retried, timeout=2)
    finally:
        stop_event.set()
    for thread in threads:
        await asyncio.to_thread(thread.join, 2)
    assert unloaded == [
        ("retired", retired_home),
        ("retired", retired_home),
    ]


def _wait_until(predicate, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(timeout=0.01)
    return predicate()


async def _wait_thread_event(event: threading.Event, *, timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if event.is_set():
            return True
        await asyncio.sleep(0.01)
    return event.is_set()


def test_wait_future_interruptibly_preserves_inner_timeout_error():
    future = concurrent.futures.Future()
    future.set_exception(TimeoutError("inner timeout"))

    try:
        _wait_future_interruptibly(
            future,
            stop_event=None,
            timeout=1,
            profile_name="main",
            action="start adapters",
        )
    except TimeoutError as exc:
        assert str(exc) == "inner timeout"
    else:
        raise AssertionError("inner TimeoutError should be preserved")


def test_unload_profile_cron_adapters_treats_blocked_result_as_failure(
    tmp_path, monkeypatch
):
    """blocked 是合法失败态，必须保留 reconciler owner 供重试。"""
    future = concurrent.futures.Future()
    future.set_result({"blocked": True, "active_sessions": 2})

    async def unload_profile_runtime(_profile_name, *, profile_home):
        return {"blocked": True, "active_sessions": 2}

    def schedule(coro, _loop, **_kwargs):
        coro.close()
        return future

    monkeypatch.setattr(
        "gateway.run._multiplex_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", schedule)
    runner = SimpleNamespace(unload_profile_runtime=unload_profile_runtime)

    assert _unload_profile_cron_adapters(
        runner,
        "coder",
        profile_home=tmp_path / "profiles" / "coder",
        loop=object(),
    ) is False


def test_unload_profile_cron_adapters_treats_stop_cancellation_as_failure(
    tmp_path, monkeypatch
):
    class _Future:
        def __init__(self):
            self.result_calls = 0
            self.cancel_calls = 0

        def result(self, timeout=None):
            self.result_calls += 1
            raise TimeoutError("future still running")

        def cancel(self):
            self.cancel_calls += 1
            return True

        def done(self):
            return False

    future = _Future()
    stop_event = threading.Event()
    stop_event.set()

    async def unload_profile_runtime(_profile_name, *, profile_home):
        return {}

    def schedule(coro, _loop, **_kwargs):
        coro.close()
        return future

    monkeypatch.setattr(
        "gateway.run._multiplex_active_profile_name", lambda: "default"
    )
    monkeypatch.setattr("gateway.run.safe_schedule_threadsafe", schedule)
    clock = iter((0.0, 0.5, 31.0))
    monkeypatch.setattr("gateway.run.time.monotonic", lambda: next(clock))
    runner = SimpleNamespace(unload_profile_runtime=unload_profile_runtime)

    assert _unload_profile_cron_adapters(
        runner,
        "coder",
        profile_home=tmp_path / "profiles" / "coder",
        loop=object(),
        stop_event=stop_event,
    ) is False
    assert future.result_calls == 0
    assert future.cancel_calls == 1


def test_cron_env_reads_active_profile_secret_scope(monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler

    monkeypatch.setenv("HERMES_MODEL", "outer-model")
    set_multiplex_active(True)
    token = set_secret_scope({"HERMES_MODEL": "profile-model"})
    try:
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model"
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_cron_env_refreshes_profile_dotenv_each_read(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    env_file = profile_home / ".env"
    env_file.write_text("HERMES_MODEL=profile-model-a\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_MODEL", "outer-model")
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(profile_home))
    stale_scope_token = set_secret_scope({"HERMES_MODEL": "stale-profile-model"})
    try:
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model-a"
        env_file.write_text("HERMES_MODEL=profile-model-b\n", encoding="utf-8")
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model-b"
    finally:
        reset_secret_scope(stale_scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_cron_dotenv_refresh_runs_for_legacy_process(tmp_path, monkeypatch):
    from agent.secret_scope import set_multiplex_active
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    calls = []
    home_token = set_hermes_home_override(str(tmp_path / ".hermes"))
    set_multiplex_active(False)
    monkeypatch.setattr(scheduler, "_load_hermes_dotenv", lambda **kwargs: calls.append(kwargs))
    try:
        scheduler._refresh_cron_dotenv_for_legacy_process()
    finally:
        reset_hermes_home_override(home_token)

    assert calls == [{"hermes_home": tmp_path / ".hermes"}]


def test_cron_dotenv_refresh_skips_multiplex_process(tmp_path, monkeypatch):
    from agent.secret_scope import set_multiplex_active
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    calls = []
    home_token = set_hermes_home_override(str(tmp_path / ".hermes"))
    set_multiplex_active(True)
    monkeypatch.setattr(scheduler, "_load_hermes_dotenv", lambda **kwargs: calls.append(kwargs))
    try:
        scheduler._refresh_cron_dotenv_for_legacy_process()
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)

    assert calls == []


def test_cron_env_keeps_process_global_cron_limits(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("HERMES_CRON_MAX_PARALLEL=99\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(profile_home))
    stale_scope_token = set_secret_scope({"HERMES_CRON_MAX_PARALLEL": "2"})
    try:
        assert scheduler._cron_env("HERMES_CRON_MAX_PARALLEL", "4") == "1"
    finally:
        reset_secret_scope(stale_scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_running_job_key_is_profile_qualified(tmp_path, monkeypatch):
    from cron import scheduler

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)

    monkeypatch.setattr(scheduler, "_hermes_home", home_a)
    key_a = scheduler._running_job_key({"id": "same-job"})
    monkeypatch.setattr(scheduler, "_hermes_home", home_b)
    key_b = scheduler._running_job_key({"id": "same-job"})

    assert key_a != key_b
    assert key_a[1] == key_b[1] == "same-job"


def test_context_from_reads_active_profile_output_dir(tmp_path, monkeypatch):
    from cron.scheduler import _build_job_prompt
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "worker"
    source_job_id = "abcdef123456"
    default_output = default_home / "cron" / "output" / source_job_id
    profile_output = profile_home / "cron" / "output" / source_job_id
    default_output.mkdir(parents=True)
    profile_output.mkdir(parents=True)
    (default_output / "2026-01-01_00-00-00.md").write_text(
        "default output must not leak",
        encoding="utf-8",
    )
    (profile_output / "2026-01-01_00-00-00.md").write_text(
        "profile scoped output",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    token = set_hermes_home_override(str(profile_home))
    try:
        prompt = _build_job_prompt(
            {
                "id": "feed00000000",
                "name": "profile-job",
                "prompt": "Summarize it",
                "context_from": [source_job_id],
            }
        )
    finally:
        reset_hermes_home_override(token)

    assert "profile scoped output" in prompt
    assert "default output must not leak" not in prompt
