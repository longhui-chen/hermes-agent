"""Tests for the broken-set short-circuit added to handle outer-timeout failures.

When ``snapshot_baseline`` or ``get_diagnostics_sync`` time out from the
service layer (because a language server hangs during initialize, or
the binary is wedged), the inner spawn task is cancelled — but the
inner exception handler that adds to ``_broken`` never runs.  Without
the service-layer fallback added in this module, every subsequent
edit re-pays the full timeout cost until the process exits.

This module verifies:
- ``_mark_broken_for_file`` adds the right key
- ``enabled_for`` short-circuits on broken keys
- a missing binary is broken-set'd after one snapshot attempt
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.lsp.manager import LSPService
from agent.lsp.workspace import clear_cache


@pytest.fixture(autouse=True)
def _clear_workspace_cache():
    clear_cache()
    yield
    clear_cache()


def _make_git_workspace(tmp_path: Path) -> Path:
    """Build a minimal git repo with a pyproject so pyright's root resolver fires."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='t'\n")
    return repo








def test_unrelated_project_not_affected_by_broken(tmp_path, monkeypatch):
    """Marking pyright broken for project A must NOT affect project B."""
    repo_a = _make_git_workspace(tmp_path)
    repo_b = tmp_path / "repo-b"
    repo_b.mkdir()
    (repo_b / ".git").mkdir()
    (repo_b / "pyproject.toml").write_text("[project]\nname='b'\n")
    a_src = repo_a / "x.py"
    a_src.write_text("")
    b_src = repo_b / "x.py"
    b_src.write_text("")

    monkeypatch.chdir(str(repo_a))
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        svc._mark_broken_for_file(str(a_src), RuntimeError("simulated"))
        # Project A skipped.
        assert svc.enabled_for(str(a_src)) is False
        # Project B still enabled — the broken key is per-project.
        monkeypatch.chdir(str(repo_b))
        assert svc.enabled_for(str(b_src)) is True
    finally:
        svc.shutdown()




def test_mark_broken_handles_no_workspace_silently(tmp_path):
    """File outside any git worktree → no workspace → no key to add."""
    src = tmp_path / "orphan.py"
    src.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        svc._mark_broken_for_file(str(src), RuntimeError("x"))
        assert len(svc._broken) == 0
    finally:
        svc.shutdown()


def test_mark_broken_shutdown_failure_keeps_exact_client_for_retry(
    tmp_path, monkeypatch
):
    """半初始化 client 关闭失败时不得丢失 owner。"""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    key = ("pyright", str(repo))
    client = MagicMock(
        server_id="pyright",
        workspace_root=str(repo),
        shutdown=AsyncMock(side_effect=RuntimeError("still alive")),
    )
    svc._clients[key] = client
    svc._last_used[key] = 1.0
    try:
        with pytest.raises(RuntimeError, match="broken-client shutdown failed"):
            svc._mark_broken_for_file(str(src), RuntimeError("timed out"))
        assert svc._clients[key] is client
        assert svc._last_used[key] == 1.0
        assert key in svc._broken
    finally:
        client.shutdown.side_effect = None
        svc.shutdown()


def test_mark_broken_timeout_keeps_retiring_fence_until_cleanup_terminal(
    tmp_path, monkeypatch
):
    """外层 1 秒超时后，后台 cleanup 未终态前不得释放 retiring owner。"""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")
    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    key = ("pyright", str(repo))
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    cleanup_done = threading.Event()

    async def blocked_shutdown():
        cleanup_entered.set()
        await asyncio.to_thread(release_cleanup.wait)
        cleanup_done.set()

    client = MagicMock(
        server_id="pyright",
        workspace_root=str(repo),
        shutdown=AsyncMock(side_effect=blocked_shutdown),
    )
    svc._clients[key] = client
    svc._last_used[key] = 1.0
    errors = []

    def mark():
        try:
            svc._mark_broken_for_file(str(src), RuntimeError("timed out"))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=mark)
    worker.start()
    assert cleanup_entered.wait(2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert errors and "broken-client shutdown failed" in str(errors[0])
    assert svc._retiring_clients[key] is client
    assert key in svc._cleanup_tasks

    release_cleanup.set()
    assert cleanup_done.wait(2)
    deadline = time.monotonic() + 2
    while (
        (key in svc._cleanup_tasks or key in svc._retiring_clients)
        and time.monotonic() < deadline
    ):
        threading.Event().wait(0.01)
    assert key not in svc._cleanup_tasks
    assert key not in svc._retiring_clients
    assert key not in svc._clients
    svc.shutdown()


def test_snapshot_failure_marks_broken_via_outer_timeout(tmp_path, monkeypatch):
    """End-to-end: ``snapshot_baseline``'s outer ``_loop.run`` timeout
    triggers ``_mark_broken_for_file``, so a second call to
    ``enabled_for`` returns False."""
    repo = _make_git_workspace(tmp_path)
    monkeypatch.chdir(str(repo))
    src = repo / "x.py"
    src.write_text("")

    svc = LSPService(
        enabled=True,
        wait_mode="document",
        wait_timeout=2.0,
        install_strategy="manual",
    )
    try:
        # Force the inner snapshot coroutine to raise.
        async def boom(_path):
            raise RuntimeError("outer-timeout simulated")

        with patch.object(svc, "_snapshot_async", boom):
            assert svc.enabled_for(str(src)) is True
            svc.snapshot_baseline(str(src))

        # After the failure, the file's pair is in the broken-set and
        # ``enabled_for`` skips it.
        assert ("pyright", str(repo)) in svc._broken
        assert svc.enabled_for(str(src)) is False
    finally:
        svc.shutdown()
