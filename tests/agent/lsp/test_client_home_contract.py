"""The LSP server spawn must flow through the shared subprocess HOME contract.

``LSPClient._spawn`` builds ``env = dict(os.environ)`` (+ any per-server
overrides) and hands it to ``asyncio.create_subprocess_exec``. On a
systemd/cron host launched with no HOME anywhere (ZET-1938), the language
server (and anything it shells out to) resolves ``~`` nowhere. Routing the
env through ``hermes_constants.apply_subprocess_home_env`` falls the child's
HOME back to ``{HERMES_HOME}/home`` in exactly that case, while leaving a
real host HOME untouched.

These tests mock ``asyncio.create_subprocess_exec`` so nothing real is
spawned; they only assert the ``env`` the client would launch with.
"""

from __future__ import annotations

import asyncio

import pytest

import hermes_constants
from agent.lsp.client import LSPClient


class _FakeProc:
    """Minimal asyncio subprocess stand-in so ``_spawn`` can finish."""

    def __init__(self):
        self.stdout = None
        self.stderr = None
        self.stdin = None
        self.returncode = None
        self.pid = 1

    def kill(self):
        pass

    async def wait(self):
        return 0


def _capture_spawn_env(monkeypatch, tmp_path):
    """Run ``LSPClient._spawn`` with a mocked exec and return the child env."""
    captured: dict = {}

    async def fake_exec(*args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    # The reader/stderr drain tasks the client kicks off are harmless against
    # a proc with stdout/stderr = None (they early-return); do not await them.
    client = LSPClient(
        server_id="mock",
        workspace_root=str(tmp_path),
        command=["true"],
        cwd=str(tmp_path),
    )
    asyncio.run(client._spawn())
    return captured["env"]


def _host_mode(monkeypatch):
    monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
    monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)


def test_missing_home_falls_back_to_profile_home(tmp_path, monkeypatch):
    """No HOME anywhere (systemd/cron) → inject ``{HERMES_HOME}/home``."""
    _host_mode(monkeypatch)
    hermes_home = tmp_path / ".hermes"
    profile_home = hermes_home / "home"
    profile_home.mkdir(parents=True)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = _capture_spawn_env(monkeypatch, tmp_path)
    assert env.get("HOME") == str(profile_home)


def test_real_home_is_preserved(tmp_path, monkeypatch):
    """Host with a real HOME → auto mode keeps it untouched."""
    _host_mode(monkeypatch)
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    env = _capture_spawn_env(monkeypatch, tmp_path)
    assert env.get("HOME") == str(real_home)


def test_contextvar_override_bridges_hermes_home(tmp_path, monkeypatch):
    """set_hermes_home_override(A) with process HERMES_HOME=B: the child env
    must carry HERMES_HOME=A (the override the contract used to resolve HOME),
    keeping HERMES_HOME and HOME on the same profile."""
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    _host_mode(monkeypatch)
    a = tmp_path / "profileA" / ".hermes"
    b = tmp_path / "profileB" / ".hermes"
    (a / "home").mkdir(parents=True)
    (b / "home").mkdir(parents=True)
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(b))

    token = set_hermes_home_override(str(a))
    try:
        env = _capture_spawn_env(monkeypatch, tmp_path)
    finally:
        reset_hermes_home_override(token)

    assert env.get("HERMES_HOME") == str(a)
    assert env.get("HOME") == str(a / "home")
