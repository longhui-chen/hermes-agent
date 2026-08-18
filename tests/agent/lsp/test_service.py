"""Tests for the synchronous LSPService wrapper.

Drives the service through ``snapshot_baseline`` →
``get_diagnostics_sync`` against the mock LSP server, exercising the
delta filter that ``tools/file_operations._check_lint_delta`` relies
on.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.client import LSPClient
from agent.lsp.servers import (
    SERVERS,
    ServerContext,
    ServerDef,
    SpawnSpec,
)


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _install_mock_server(monkeypatch, script: str = "errors", server_id: str = "pyright"):
    """Replace one registered server with a wrapper that spawns the mock.

    We reuse ``pyright`` so .py files route to it.  This keeps the
    test free of any LSP toolchain dependency.
    """
    target_index = next(i for i, s in enumerate(SERVERS) if s.server_id == server_id)
    original = SERVERS[target_index]

    def _spawn(root: str, ctx: ServerContext) -> SpawnSpec:
        env = {"MOCK_LSP_SCRIPT": script}
        return SpawnSpec(
            command=[sys.executable, MOCK_SERVER],
            workspace_root=root,
            cwd=root,
            env=env,
            initialization_options={},
        )

    replacement = ServerDef(
        server_id=server_id,
        extensions=original.extensions,
        resolve_root=lambda fp, ws: ws,  # always use workspace root
        build_spawn=_spawn,
        seed_first_push=False,
        description="mock " + server_id,
    )
    # Patch the SERVERS list element directly + restore on teardown.
    SERVERS[target_index] = replacement

    yield

    SERVERS[target_index] = original


@pytest.fixture
def mock_pyright(monkeypatch, tmp_path):
    """Install the mock as ``pyright`` and create a fake git workspace."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("")  # so pyright's root resolver finds it
    monkeypatch.chdir(str(repo))
    gen = _install_mock_server(monkeypatch, "errors", "pyright")
    next(gen)
    yield repo
    try:
        next(gen)
    except StopIteration:
        pass






def test_service_e2e_delta_filter(mock_pyright):
    """End-to-end: snapshot baseline → wait → delta returned."""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        assert svc.enabled_for(str(f))
        # Baseline first — server pushes 1 error.
        svc.snapshot_baseline(str(f))
        # Re-poll: same error is in baseline, so delta is empty.
        new_diags = svc.get_diagnostics_sync(str(f))
        assert new_diags == []
    finally:
        svc.shutdown()


def test_service_e2e_delta_filter_with_line_shift(mock_pyright):
    """End-to-end: an edit that shifts the diagnostic's line still
    filters correctly when ``line_shift`` is supplied.

    The mock LSP server emits a fixed error at line 0; for this test
    we don't need to actually shift the server's output — we just
    need to prove that supplying a line_shift through the API works
    and doesn't break the existing delta path.  The unit tests in
    test_delta_key.py cover the shift semantics in detail.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("print('hi')\n")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
    )
    try:
        svc.snapshot_baseline(str(f))
        # Identity shift — should behave exactly like no shift.
        new_diags = svc.get_diagnostics_sync(str(f), line_shift=lambda L: L)
        assert new_diags == []
    finally:
        svc.shutdown()






def test_reused_client_refreshes_last_used_and_survives_reap(mock_pyright):
    """A client re-acquired from the cache must have its ``_last_used``
    timestamp refreshed so a subsequent sweep does NOT evict it.

    Covers the timestamp refresh on the existing-client fast path in
    ``_get_or_spawn`` — without it, a client in constant use would be
    reaped ``idle_timeout`` seconds after its FIRST use.
    """
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=60.0,  # sweeps manually below; loop never fires
    )
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        first_used = svc._last_used[key]

        # Age the timestamp past the cutoff, then re-acquire the client.
        svc._last_used[key] = first_used - 120.0
        svc.get_diagnostics_sync(str(f))
        assert svc._last_used[key] > first_used - 120.0, (
            "re-acquiring a cached client must refresh _last_used"
        )

        # A sweep right after reuse must keep the client.
        svc._loop.run(svc._reap_idle_once(), timeout=5.0)
        assert key in svc._clients
        assert svc.get_status()["clients"]
    finally:
        svc.shutdown()


def test_reaper_keeps_owner_and_retries_failed_client_shutdown(
    mock_pyright, monkeypatch
):
    """一次 shutdown 失败要保留 owner，reaper 存活并在后续轮次重试。"""
    repo = mock_pyright
    f = repo / "x.py"
    f.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=3.0,
        install_strategy="manual",
        idle_timeout=0.1,
    )
    retry_allowed = threading.Event()
    try:
        svc.get_diagnostics_sync(str(f))
        key = next(iter(svc._clients))
        client = svc._clients[key]
        real_shutdown = client.shutdown
        calls = {"n": 0}
        first_failed = threading.Event()

        async def _flaky_shutdown():
            calls["n"] += 1
            if calls["n"] == 1:
                first_failed.set()
                raise RuntimeError("shutdown sabotage")
            await asyncio.to_thread(retry_allowed.wait, 2)
            await real_shutdown()

        monkeypatch.setattr(client, "shutdown", _flaky_shutdown)
        svc._last_used[key] = 0

        assert first_failed.wait(timeout=2)
        assert svc._clients[key] is client
        assert key in svc._last_used
        retry_allowed.set()
        deadline = time.monotonic() + 3.0
        while svc.get_status()["clients"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert calls["n"] >= 2, "reaper 没有重试关闭失败的 client"
        assert svc.get_status()["clients"] == []
        assert svc._idle_reaper_task is not None
        assert not svc._idle_reaper_task.done()
    finally:
        retry_allowed.set()
        svc.shutdown()


def test_reaper_restores_real_lsp_client_process_for_cleanup_retry(tmp_path):
    """真实 LSPClient 首次 terminate 失败后，process handle 必须可重试。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    client = LSPClient(
        server_id="pyright",
        workspace_root=str(repo),
        command=[sys.executable, "-c", "pass"],
    )

    class _Process:
        def __init__(self):
            self.returncode = None
            self.terminate_calls = 0

        def terminate(self):
            self.terminate_calls += 1
            if self.terminate_calls == 1:
                raise PermissionError("terminate denied")

        async def wait(self):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    process = _Process()
    client._proc = process
    key = (client.server_id, client.workspace_root)
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )
    with svc._state_lock:
        svc._clients[key] = client
        svc._last_used[key] = 0
    try:
        with pytest.raises(RuntimeError, match="idle reaper failed"):
            svc._loop.run(svc._reap_idle_once(), timeout=2.0)
        assert svc._clients[key] is client
        assert client._proc is process
        assert client._stopping is False
        assert process.returncode is None

        src = repo / "x.py"
        src.write_text("")
        server_def = SimpleNamespace(
            server_id="pyright",
            resolve_root=lambda _path, _root: str(repo),
        )
        with patch(
            "agent.lsp.manager.find_server_for_file", return_value=server_def
        ), patch(
            "agent.lsp.manager.resolve_workspace_for_file",
            return_value=(str(repo), True),
        ):
            assert svc._loop.run(
                svc._get_or_spawn(str(src)), timeout=2.0
            ) is None
        assert svc._clients[key] is client
        assert client._proc is process

        svc._loop.run(svc._reap_idle_once(), timeout=2.0)
        assert key not in svc._clients
        assert client._proc is None
        assert process.returncode == 0
        assert process.terminate_calls == 2
    finally:
        svc.shutdown()


def test_profile_shutdown_rejects_real_lsp_cleanup_already_in_flight(tmp_path):
    """同一真实 client 的 reaper 与 profile shutdown 不得并发清理。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    client = LSPClient(
        server_id="pyright",
        workspace_root=str(repo),
        command=[sys.executable, "-c", "pass"],
    )

    class _Process:
        def __init__(self):
            self.returncode = None
            self.wait_calls = 0
            self.wait_started = None
            self.release_wait = None

        def terminate(self):
            return None

        async def wait(self):
            self.wait_calls += 1
            if self.wait_calls == 1:
                self.wait_started.set()
                await self.release_wait.wait()
                raise PermissionError("wait denied")
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    process = _Process()
    client._proc = process
    key = (client.server_id, client.workspace_root)
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )
    with svc._state_lock:
        svc._clients[key] = client
        svc._last_used[key] = 0

    async def _exercise():
        process.wait_started = asyncio.Event()
        process.release_wait = asyncio.Event()
        reaper = asyncio.create_task(svc._reap_idle_once())
        await process.wait_started.wait()
        assert client._proc is None
        assert client._stopping is True
        with pytest.raises(RuntimeError, match="shutdown already in progress"):
            await svc._shutdown_async()
        assert svc._clients[key] is client

        process.release_wait.set()
        with pytest.raises(RuntimeError, match="idle reaper failed"):
            await reaper
        assert svc._clients[key] is client
        assert client._proc is process
        assert client._stopping is False

        await svc._shutdown_async()
        assert key not in svc._clients
        assert process.returncode == 0

    try:
        svc._loop.run(_exercise(), timeout=5.0)
    finally:
        svc.shutdown()


@pytest.mark.parametrize(
    "cancel_shutdown", [False, True], ids=["normal", "cancelled"]
)
def test_profile_shutdown_waits_for_client_start_in_flight(
    tmp_path, monkeypatch, cancel_shutdown
):
    """client.start 已进入后，shutdown 必须持有并严格清理该未就绪 owner。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "x.py"
    src.write_text("")
    start_entered = None
    release_start = None
    shutdown_entered = None
    release_shutdown = None

    class _Client:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self._proc = None
            self._stopping = False
            self.stopped = False

        @property
        def is_running(self):
            return False

        async def start(self):
            start_entered.set()
            await release_start.wait()
            if self.stopped:
                raise RuntimeError("stopped during initialize")

        async def shutdown(self):
            self.stopped = True
            shutdown_entered.set()
            await release_shutdown.wait()

    server_def = SimpleNamespace(
        server_id="pyright",
        seed_first_push=False,
        resolve_root=lambda _path, _root: str(repo),
        build_spawn=lambda _root, _ctx: SimpleNamespace(
            workspace_root=str(repo),
            command=["pyright"],
            env=None,
            cwd=str(repo),
            initialization_options=None,
            seed_diagnostics_on_first_push=False,
        ),
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", _Client)
    monkeypatch.setattr(
        "agent.lsp.manager.find_server_for_file", lambda _path: server_def
    )
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (str(repo), True),
    )
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=0,
    )

    async def exercise():
        nonlocal start_entered, release_start, shutdown_entered, release_shutdown
        start_entered = asyncio.Event()
        release_start = asyncio.Event()
        shutdown_entered = asyncio.Event()
        release_shutdown = asyncio.Event()
        spawn = asyncio.create_task(svc._get_or_spawn(str(src)))
        await start_entered.wait()
        key = ("pyright", str(repo))
        assert key in svc._clients

        shutdown = asyncio.create_task(svc._shutdown_async())
        try:
            shutdown_waiting = asyncio.Event()
            asyncio.get_running_loop().call_soon(shutdown_waiting.set)
            await shutdown_waiting.wait()
            assert svc._shutdown_in_progress is True
            assert not shutdown_entered.is_set()
            assert not shutdown.done()
            if cancel_shutdown:
                shutdown.cancel()
                cancellation_delivered = asyncio.Event()
                asyncio.get_running_loop().call_soon(
                    cancellation_delivered.set
                )
                await cancellation_delivered.wait()
                assert not shutdown.done()
                assert svc._shutdown_in_progress is True
            release_start.set()
            await shutdown_entered.wait()
            assert not shutdown.done()
            assert svc._clients[key] is not None
            release_shutdown.set()
            if cancel_shutdown:
                with pytest.raises(asyncio.CancelledError):
                    await shutdown
            else:
                await shutdown
            assert await spawn is None
            assert key not in svc._clients
        finally:
            release_start.set()
            release_shutdown.set()
            await asyncio.gather(spawn, shutdown, return_exceptions=True)

    try:
        svc._loop.run(exercise(), timeout=5.0)
    finally:
        svc._loop.stop()


def test_idle_reaper_skips_client_start_in_flight(tmp_path, monkeypatch):
    """reaper 扫描时不得把仍在 initialize 的 client 当成 idle。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "x.py"
    src.write_text("")
    start_entered = None
    release_start = None
    created = []

    class _Client:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.started = False
            self.shutdown_calls = 0
            created.append(self)

        @property
        def is_running(self):
            return self.started

        async def start(self):
            start_entered.set()
            await release_start.wait()
            self.started = True

        async def shutdown(self):
            self.shutdown_calls += 1
            self.started = False

    server_def = SimpleNamespace(
        server_id="pyright",
        seed_first_push=False,
        resolve_root=lambda _path, _root: str(repo),
        build_spawn=lambda _root, _ctx: SimpleNamespace(
            workspace_root=str(repo),
            command=["pyright"],
            env=None,
            cwd=str(repo),
            initialization_options=None,
            seed_diagnostics_on_first_push=False,
        ),
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", _Client)
    monkeypatch.setattr(
        "agent.lsp.manager.find_server_for_file", lambda _path: server_def
    )
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (str(repo), True),
    )
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )

    async def exercise():
        nonlocal start_entered, release_start
        start_entered = asyncio.Event()
        release_start = asyncio.Event()
        spawn = asyncio.create_task(svc._get_or_spawn(str(src)))
        await start_entered.wait()
        key = ("pyright", str(repo))
        assert key in svc._spawning
        assert key in svc._clients

        await svc._reap_idle_once()
        assert created[0].shutdown_calls == 0
        assert svc._clients[key] is created[0]

        release_start.set()
        assert await spawn is created[0]
        assert key not in svc._spawning
        assert key in svc._last_used
        await svc._shutdown_async()
        assert created[0].shutdown_calls == 1

    try:
        svc._loop.run(exercise(), timeout=5.0)
    finally:
        svc._loop.stop()


def test_cancelled_spawn_follower_does_not_cancel_shared_owner(
    tmp_path, monkeypatch
):
    """一个 follower 取消不能破坏同 root 的共享 spawn Future。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "x.py"
    src.write_text("")
    start_entered = None
    release_start = None
    created = []

    class _Client:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.started = False
            created.append(self)

        @property
        def is_running(self):
            return self.started

        async def start(self):
            start_entered.set()
            await release_start.wait()
            self.started = True

        async def shutdown(self):
            self.started = False

    server_def = SimpleNamespace(
        server_id="pyright",
        seed_first_push=False,
        resolve_root=lambda _path, _root: str(repo),
        build_spawn=lambda _root, _ctx: SimpleNamespace(
            workspace_root=str(repo),
            command=["pyright"],
            env=None,
            cwd=str(repo),
            initialization_options=None,
            seed_diagnostics_on_first_push=False,
        ),
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", _Client)
    monkeypatch.setattr(
        "agent.lsp.manager.find_server_for_file", lambda _path: server_def
    )
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (str(repo), True),
    )
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )

    async def exercise():
        nonlocal start_entered, release_start
        start_entered = asyncio.Event()
        release_start = asyncio.Event()
        leader = asyncio.create_task(svc._get_or_spawn(str(src)))
        await start_entered.wait()
        key = ("pyright", str(repo))
        joined = asyncio.Event()
        joined_count = 0
        spawn_future = svc._spawning[key]

        class _ObservedSpawnAwaitable:
            def __await__(self):
                nonlocal joined_count
                joined_count += 1
                if joined_count == 2:
                    joined.set()
                return (yield from spawn_future.__await__())

        # shield 与错误的直接 await 都会从这里点亮；因此逆改会红在 owner
        # 被取消的断言，而不是因 marker 消失而超时。
        svc._spawning[key] = _ObservedSpawnAwaitable()
        cancelled_follower = asyncio.create_task(svc._get_or_spawn(str(src)))
        surviving_follower = asyncio.create_task(svc._get_or_spawn(str(src)))
        await joined.wait()
        svc._spawning[key] = spawn_future

        cancelled_follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_follower
        assert not svc._spawning[key].cancelled()

        release_start.set()
        client = await leader
        assert client is created[0]
        assert await surviving_follower is client
        await svc._shutdown_async()

    try:
        svc._loop.run(exercise(), timeout=5.0)
    finally:
        svc._loop.stop()


def test_cancelled_spawn_leader_cleanup_failure_keeps_owner_for_retry(
    tmp_path, monkeypatch
):
    """leader 取消后的 cleanup 必须完成后才报错，并保留失败 owner。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "x.py"
    src.write_text("")
    start_entered = None
    cleanup_entered = None
    release_cleanup = None
    created = []

    class _Client:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self.shutdown_calls = 0
            created.append(self)

        @property
        def is_running(self):
            return False

        async def start(self):
            start_entered.set()
            await asyncio.Event().wait()

        async def shutdown(self):
            self.shutdown_calls += 1
            cleanup_entered.set()
            await release_cleanup.wait()
            if self.shutdown_calls == 1:
                raise RuntimeError("still alive")

    server_def = SimpleNamespace(
        server_id="pyright",
        seed_first_push=False,
        resolve_root=lambda _path, _root: str(repo),
        build_spawn=lambda _root, _ctx: SimpleNamespace(
            workspace_root=str(repo),
            command=["pyright"],
            env=None,
            cwd=str(repo),
            initialization_options=None,
            seed_diagnostics_on_first_push=False,
        ),
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", _Client)
    monkeypatch.setattr(
        "agent.lsp.manager.find_server_for_file", lambda _path: server_def
    )
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (str(repo), True),
    )
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )

    async def exercise():
        nonlocal start_entered, cleanup_entered, release_cleanup
        start_entered = asyncio.Event()
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        leader = asyncio.create_task(svc._get_or_spawn(str(src)))
        await start_entered.wait()
        key = ("pyright", str(repo))

        leader.cancel()
        await cleanup_entered.wait()
        assert not leader.done()
        assert svc._clients[key] is created[0]
        leader.cancel()
        second_cancel_delivered = asyncio.Event()
        asyncio.get_running_loop().call_soon(second_cancel_delivered.set)
        await second_cancel_delivered.wait()
        assert not leader.done()
        assert svc._retiring_clients[key] is created[0]
        assert not svc._cleanup_tasks[key].done()

        release_cleanup.set()
        with pytest.raises(RuntimeError, match="cancelled spawn cleanup failed"):
            await leader
        assert svc._clients[key] is created[0]
        assert svc._cleanup_retry_clients[key] is created[0]
        assert await svc._get_or_spawn(str(src)) is None
        assert len(created) == 1

        await svc._shutdown_async()
        assert created[0].shutdown_calls == 2
        assert key not in svc._clients

    try:
        svc._loop.run(exercise(), timeout=5.0)
    finally:
        svc._loop.stop()


def test_spawn_init_failure_cleanup_failure_keeps_owner_for_shutdown_retry(
    tmp_path, monkeypatch
):
    """普通 start/init 异常也必须保留 cleanup 失败的 client owner。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    src = repo / "x.py"
    src.write_text("")
    created = []

    class _Process:
        def __init__(self):
            self.returncode = None

    class _Client:
        def __init__(self, **kwargs):
            self.server_id = kwargs["server_id"]
            self.workspace_root = kwargs["workspace_root"]
            self._proc = None
            self._stopping = False
            self.cleanup_calls = 0
            self.process = _Process()
            created.append(self)

        @property
        def is_running(self):
            return False

        async def start(self):
            self._proc = self.process
            await self._cleanup_process()
            raise RuntimeError("initialize failed")

        async def _cleanup_process(self):
            self.cleanup_calls += 1
            process = self._proc
            self._proc = None
            if self.cleanup_calls <= 2:
                raise RuntimeError("cleanup failed")
            process.returncode = 0

        async def shutdown(self):
            await self._cleanup_process()

    server_def = SimpleNamespace(
        server_id="pyright",
        seed_first_push=False,
        resolve_root=lambda _path, _root: str(repo),
        build_spawn=lambda _root, _ctx: SimpleNamespace(
            workspace_root=str(repo),
            command=["pyright"],
            env=None,
            cwd=str(repo),
            initialization_options=None,
            seed_diagnostics_on_first_push=False,
        ),
    )
    monkeypatch.setattr("agent.lsp.manager.LSPClient", _Client)
    monkeypatch.setattr(
        "agent.lsp.manager.find_server_for_file", lambda _path: server_def
    )
    monkeypatch.setattr(
        "agent.lsp.manager.resolve_workspace_for_file",
        lambda _path: (str(repo), True),
    )
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
        idle_timeout=60.0,
    )

    async def exercise():
        key = ("pyright", str(repo))
        with pytest.raises(RuntimeError, match="LSP spawn cleanup failed"):
            await svc._get_or_spawn(str(src))
        assert svc._clients[key] is created[0]
        assert svc._cleanup_retry_clients[key] is created[0]
        assert created[0].cleanup_calls == 2
        assert created[0]._proc is created[0].process

        await svc._shutdown_async()
        assert created[0].cleanup_calls == 3
        assert created[0].process.returncode == 0
        assert key not in svc._clients

    try:
        svc._loop.run(exercise(), timeout=5.0)
    finally:
        svc._loop.stop()
