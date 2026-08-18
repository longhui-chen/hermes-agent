"""Tests for service-singleton lifecycle: atexit handler, idempotent shutdown.

These cover the exit-cleanup behavior added to plug the language-server
process leak — without the atexit hook, ``hermes chat`` exits while
pyright/gopls/etc. are still alive on the host.
"""
from __future__ import annotations

import atexit
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import lsp as lsp_module
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Force a clean module state before each test.

    Tests in this file share process-global state (the lazy
    singleton + atexit registration flag); reset both before and
    after every test so order doesn't matter.
    """
    lsp_module._services.clear()
    lsp_module._atexit_registered = False
    yield
    lsp_module._services.clear()
    lsp_module._atexit_registered = False


def test_get_service_registers_atexit_handler_once(monkeypatch):
    """First call to ``get_service`` must register an atexit handler;
    subsequent calls must NOT register another one (Python's ``atexit``
    runs every registered callable, so a duplicate would shutdown
    twice — harmless but wasteful)."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )

    registrations = []

    def fake_register(fn):
        registrations.append(fn)

    monkeypatch.setattr(atexit, "register", fake_register)

    a = lsp_module.get_service()
    b = lsp_module.get_service()
    c = lsp_module.get_service()

    assert a is fake_svc
    assert b is fake_svc
    assert c is fake_svc
    assert len(registrations) == 1
    # The registered callable must be our internal shutdown wrapper.
    assert registrations[0] is lsp_module._atexit_shutdown




def test_atexit_shutdown_swallows_exceptions(monkeypatch):
    service = MagicMock()
    service.shutdown.side_effect = RuntimeError("server already dead")
    lsp_module._services["broken"] = service

    # Must not raise.
    lsp_module._atexit_shutdown()


def test_shutdown_service_idempotent(monkeypatch):
    """Calling shutdown twice must be safe — first call cleans up,
    second call no-ops (nothing to shut down)."""
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    fake_svc.shutdown = MagicMock()
    monkeypatch.setattr(
        lsp_module.LSPService, "create_from_config", classmethod(lambda cls: fake_svc)
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    lsp_module.shutdown_service()
    lsp_module.shutdown_service()  # must not raise

    assert fake_svc.shutdown.call_count == 1


def test_strict_shutdown_failure_keeps_service_for_retry(monkeypatch):
    fake_svc = MagicMock()
    fake_svc.is_active.return_value = True
    fake_svc.shutdown.side_effect = RuntimeError("still running")
    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        classmethod(lambda cls: fake_svc),
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    lsp_module.get_service()
    identity = lsp_module._profile_identity()
    with pytest.raises(RuntimeError, match="still running"):
        lsp_module.shutdown_service(raise_on_error=True)
    assert lsp_module._services[identity] is fake_svc


@pytest.mark.asyncio
async def test_client_shutdown_failure_keeps_exact_client_ownership():
    service = object.__new__(lsp_module.LSPService)
    key = ("pyright", "/workspace")
    client = MagicMock(shutdown=AsyncMock(side_effect=RuntimeError("still alive")))
    service._idle_reaper_task = None
    service._clients = {key: client}
    service._last_used = {key: 1.0}
    service._broken = {key}
    service._retiring_clients = {}
    service._cleanup_retry_clients = {}
    service._shutdown_in_progress = False
    service._spawning = {}
    service._state_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="LSP shutdown failed"):
        await service._shutdown_async()

    assert service._clients[key] is client
    assert service._last_used[key] == 1.0
    assert key in service._broken


@pytest.mark.asyncio
async def test_strict_shutdown_commits_successful_clients_before_retry():
    """批量关闭部分失败时，成功 owner 不得在下一轮被重复关闭。"""
    service = object.__new__(lsp_module.LSPService)
    good_key = ("pyright", "/good")
    flaky_key = ("gopls", "/flaky")
    good = MagicMock(shutdown=AsyncMock())
    flaky = MagicMock(
        shutdown=AsyncMock(side_effect=[RuntimeError("still alive"), None])
    )
    service._idle_reaper_task = None
    service._clients = {good_key: good, flaky_key: flaky}
    service._last_used = {good_key: 1.0, flaky_key: 1.0}
    service._broken = {good_key, flaky_key}
    service._retiring_clients = {}
    service._cleanup_retry_clients = {}
    service._shutdown_in_progress = False
    service._spawning = {}
    service._state_lock = threading.Lock()

    with pytest.raises(RuntimeError, match="LSP shutdown failed for 1 client"):
        await service._shutdown_async()

    assert good_key not in service._clients
    assert flaky_key in service._clients
    assert service._cleanup_retry_clients[flaky_key] is flaky

    await service._shutdown_async()
    assert good.shutdown.await_count == 1
    assert flaky.shutdown.await_count == 2
    assert service._clients == {}


@pytest.mark.asyncio
async def test_cancelled_shutdown_waits_for_cleanup_before_committing_owner():
    """shutdown 调用方取消后，inner cleanup 完成前不得释放 fence。"""
    service = object.__new__(lsp_module.LSPService)
    key = ("pyright", "/workspace")
    entered = asyncio.Event()
    blocker = asyncio.Event()
    calls = 0

    async def shutdown():
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await blocker.wait()

    client = SimpleNamespace(
        server_id="pyright",
        workspace_root="/workspace",
        shutdown=shutdown,
        _proc=None,
        _stopping=False,
    )
    service._idle_reaper_task = None
    service._clients = {key: client}
    service._last_used = {key: 1.0}
    service._broken = {key}
    service._retiring_clients = {}
    service._cleanup_retry_clients = {}
    service._shutdown_in_progress = False
    service._spawning = {}
    service._state_lock = threading.Lock()

    first = asyncio.create_task(service._shutdown_async())
    await entered.wait()
    first.cancel()
    cancellation_delivered = asyncio.Event()
    asyncio.get_running_loop().call_soon(cancellation_delivered.set)
    await cancellation_delivered.wait()
    assert not first.done()
    assert service._clients[key] is client
    assert service._retiring_clients[key] is client

    blocker.set()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert key not in service._clients
    assert key not in service._cleanup_retry_clients
    assert key not in service._retiring_clients
    assert service._shutdown_in_progress is True
    assert calls == 1
    assert service._broken == set()


@pytest.mark.asyncio
async def test_cancelled_while_waiting_for_reaper_keeps_terminal_fence():
    """等待 reaper 时调用方取消，完整 shutdown 后仍不得重开 spawn fence。"""
    service = object.__new__(lsp_module.LSPService)
    cancelled_reaper_entered = asyncio.Event()
    release_reaper = asyncio.Event()

    async def stubborn_reaper():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_reaper_entered.set()
            try:
                await release_reaper.wait()
            except asyncio.CancelledError:
                await release_reaper.wait()

    reaper = asyncio.create_task(stubborn_reaper())
    service._idle_reaper_task = reaper
    service._clients = {}
    service._last_used = {}
    service._broken = set()
    service._retiring_clients = {}
    service._cleanup_retry_clients = {}
    service._shutdown_in_progress = False
    service._spawning = {}
    service._state_lock = threading.Lock()

    first = asyncio.create_task(service._shutdown_async())
    await cancelled_reaper_entered.wait()
    first.cancel()
    release_reaper.set()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert service._shutdown_in_progress is True
    assert service._idle_reaper_task is None or reaper.done()


def test_lsp_service_is_reused_only_within_one_profile(tmp_path, monkeypatch):
    created = []

    def create_service(_cls):
        service = MagicMock()
        service.is_active.return_value = True
        created.append(service)
        return service

    monkeypatch.setattr(
        lsp_module.LSPService,
        "create_from_config",
        classmethod(create_service),
    )
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    token = set_hermes_home_override(tmp_path / "profiles" / "a")
    try:
        service_a = lsp_module.get_service()
        assert lsp_module.get_service() is service_a
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(tmp_path / "profiles" / "b")
    try:
        service_b = lsp_module.get_service()
    finally:
        reset_hermes_home_override(token)

    assert service_b is not service_a
    assert len(created) == 2


def test_disabled_lsp_result_is_cached_per_profile(tmp_path, monkeypatch):
    create = MagicMock(return_value=None)
    monkeypatch.setattr(lsp_module.LSPService, "create_from_config", create)
    monkeypatch.setattr(atexit, "register", lambda fn: None)

    token = set_hermes_home_override(tmp_path / "profiles" / "disabled")
    try:
        assert lsp_module.get_service() is None
        assert lsp_module.get_service() is None
    finally:
        reset_hermes_home_override(token)

    create.assert_called_once_with()
