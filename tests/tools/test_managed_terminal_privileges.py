import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.code_execution_tool as code_execution_module
import tools.environments.local as local_module
import tools.process_registry as process_registry_module
from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry


def test_managed_terminal_drops_identity_changing_capabilities(monkeypatch):
    captured = {}
    info = SimpleNamespace(st_mode=stat.S_IFREG | 0o755, st_uid=0)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(local_module.os, "lstat", lambda _path: info)
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_identity",
        lambda _env=None: (65534, 65534),
    )
    monkeypatch.setattr(local_module, "_find_bash", lambda: "/bin/bash")
    monkeypatch.setattr(local_module, "_make_run_env", lambda _env: {})
    monkeypatch.setattr(local_module, "_resolve_safe_cwd", lambda cwd: cwd)
    monkeypatch.setattr(local_module.os, "getpgid", lambda _pid: 42)

    class Process:
        pid = 42

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(local_module.subprocess, "Popen", fake_popen)
    environment = LocalEnvironment.__new__(LocalEnvironment)
    environment.env = {}
    environment.cwd = "/tmp"
    environment._run_bash("id")

    assert captured["argv"] == [
        "/usr/bin/setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
        "/bin/bash",
        "-c",
        "id",
    ]


def test_managed_terminal_fails_closed_without_trusted_setpriv(monkeypatch):
    monkeypatch.setattr(
        local_module.os,
        "lstat",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError()),
    )
    with pytest.raises(OSError, match="privilege drop is unavailable"):
        local_module._managed_terminal_privilege_drop_prefix()


def test_managed_execute_code_drops_identity_capabilities(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_privilege_drop_prefix",
        lambda _env=None: [
            "/usr/bin/setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--bounding-set=-all",
            "--",
        ],
    )

    argv = code_execution_module._managed_execute_code_argv(
        "/app/venv/bin/python",
        "/tmp/hermes-execute/script.py",
        env={},
    )

    assert argv == [
        "/usr/bin/setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--bounding-set=-all",
        "--",
        "/app/venv/bin/python",
        "/tmp/hermes-execute/script.py",
    ]


def test_managed_service_mounts_system_read_only_with_scoped_writes():
    service = Path("zpk/init.d/zettlab-claw.service").read_text(
        encoding="utf-8"
    )
    assert "ProtectSystem=strict" in service
    assert "ReadWritePaths=__APP_BASE__/data" in service
    assert "ReadWritePaths=-/volume1/subvol/agents/data" in service
    assert "ReadWritePaths=-/volume1/agents/data" in service
    assert "ReadOnlyPaths=-/volume1/subvol/agents/zettlab-presets" in service
    assert (
        "Environment=HERMES_LAZY_INSTALL_TARGET="
        "__APP_BASE__/data/lazy-packages"
    ) in service
    assert "Environment=HERMES_DISABLE_LAZY_INSTALLS=1" in service
    assert "MemoryHigh=768M" in service
    assert "MemoryMax=1G" in service
    assert "MemorySwapMax=0" in service
    assert "TasksMax=512" in service
    assert "OOMPolicy=continue" in service
    assert (
        "Environment=PATH="
        "/zettos/main/apps/com.zettlab.local-server/current/sbin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ) in service
    launcher = Path("zpk/libexec/hermes-secure-launcher.py").read_text(
        encoding="utf-8"
    )
    assert '"memory.high": "805306368"' in launcher
    assert '"memory.max": "1073741824"' in launcher
    assert '"memory.swap.max": "0"' in launcher
    assert '"pids.max": "512"' in launcher
    assert "_verify_managed_service_limits(service)" in launcher
    prepare = Path("zpk/prepare-claw-service.sh").read_text(encoding="utf-8")
    assert "secure_profile_secret_files" in prepare
    assert 'chmod 0600 "$path"' in prepare


def _background_registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_write_checkpoint", lambda: None)
    monkeypatch.setattr(
        registry,
        "_safe_host_start_time",
        lambda _pid: 1,
    )
    monkeypatch.setattr(
        process_registry_module.threading.Thread,
        "start",
        lambda _thread: None,
    )
    monkeypatch.setattr(
        process_registry_module,
        "_sanitize_subprocess_env",
        lambda _base, _extra: {},
    )
    monkeypatch.setattr(
        process_registry_module,
        "_find_shell",
        lambda: "/bin/bash",
    )
    monkeypatch.setattr(
        process_registry_module,
        "_resolve_safe_cwd",
        lambda cwd: cwd,
    )
    return registry


def test_managed_background_pipe_drops_identity_capabilities(monkeypatch):
    captured = {}
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_privilege_drop_prefix",
        lambda _env=None: [
            "/usr/bin/setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--bounding-set=-all",
            "--",
        ],
    )
    registry = _background_registry(monkeypatch)

    class Process:
        pid = 42
        stdout = None

        def poll(self):
            return None

    def fake_popen(argv, **_kwargs):
        captured["argv"] = argv
        return Process()

    monkeypatch.setattr(
        process_registry_module.subprocess,
        "Popen",
        fake_popen,
    )
    registry.spawn_local("sleep 1", cwd="/tmp")
    assert captured["argv"][:6] == [
        "/usr/bin/setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--bounding-set=-all",
        "--",
    ]


def test_managed_background_pty_drops_identity_capabilities(monkeypatch):
    captured = {}
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_privilege_drop_prefix",
        lambda _env=None: [
            "/usr/bin/setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--bounding-set=-all",
            "--",
        ],
    )
    registry = _background_registry(monkeypatch)

    class PtyProcess:
        pid = 43

        @classmethod
        def spawn(cls, argv, **_kwargs):
            captured["argv"] = argv
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "ptyprocess",
        SimpleNamespace(PtyProcess=PtyProcess),
    )
    registry.spawn_local("sleep 1", cwd="/tmp", use_pty=True)
    assert captured["argv"][:6] == [
        "/usr/bin/setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--bounding-set=-all",
        "--",
    ]


def test_managed_terminal_identity_is_profile_scoped(monkeypatch):
    monkeypatch.setattr(local_module.os, "geteuid", lambda: 0)
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    first = local_module._managed_terminal_identity(
        {"HERMES_HOME": "/profiles/first"}
    )
    repeated = local_module._managed_terminal_identity(
        {"HERMES_HOME": "/profiles/first"}
    )
    second = local_module._managed_terminal_identity(
        {"HERMES_HOME": "/profiles/second"}
    )
    assert first == repeated
    assert first[0] == first[1]
    assert second[0] == second[1]
    assert first != second
    assert first[0] >= local_module._MANAGED_TERMINAL_UID_MIN


def test_generic_subprocess_scrubs_managed_gateway_key(monkeypatch):
    monkeypatch.setattr(
        "tools.env_passthrough.is_env_passthrough",
        lambda _key: False,
    )
    sanitized = local_module._sanitize_subprocess_env({
        "PATH": "/usr/bin",
        "ZET_AGENT_KEY": "never-inherit",
    })
    assert sanitized["PATH"] == "/usr/bin"
    assert "ZET_AGENT_KEY" not in sanitized
