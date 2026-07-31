import io
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from tools import trusted_direct_runner

_REAL_SELECT_MANAGED_RUNNER_IDENTITY = (
    trusted_direct_runner._select_managed_runner_identity
)


@pytest.fixture(autouse=True)
def _managed_identity_on_non_procfs_hosts(monkeypatch):
    if not Path("/proc").is_dir():
        monkeypatch.setattr(
            trusted_direct_runner,
            "_select_managed_runner_identity",
            lambda: (
                trusted_direct_runner._MANAGED_RUNNER_UID_MIN,
                trusted_direct_runner._MANAGED_RUNNER_UID_MIN,
            ),
        )


class _RecordingStdin:
    def __init__(self, stream, captured: bytearray) -> None:
        self._stream = stream
        self._captured = captured

    @property
    def closed(self):
        return self._stream.closed

    def write(self, data):
        self._captured.extend(data)
        return self._stream.write(data)

    def close(self):
        return self._stream.close()


def _write_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "trusted.py"
    script.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return script


def _managed_delegation_fixture(monkeypatch, tmp_path):
    cgroup_root = tmp_path / "cgroup"
    service_relative = "/system.slice/zettlab-claw.service"
    service = cgroup_root / service_relative.lstrip("/")
    supervisor = service / trusted_direct_runner._MANAGED_SUPERVISOR_CGROUP
    supervisor.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.subtree_control").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.procs").write_text("", encoding="ascii")
    (service / "cgroup.kill").write_text("", encoding="ascii")
    (supervisor / "cgroup.procs").write_text(
        f"{os.getpid()}\n",
        encoding="ascii",
    )
    proc_self = tmp_path / "proc-self-cgroup"
    proc_self.write_text(
        f"0::{service_relative}/{trusted_direct_runner._MANAGED_SUPERVISOR_CGROUP}\n",
        encoding="ascii",
    )
    monkeypatch.setattr(trusted_direct_runner, "_CGROUP2_ROOT", cgroup_root)
    monkeypatch.setattr(trusted_direct_runner, "_PROC_SELF_CGROUP", proc_self)
    monkeypatch.setattr(trusted_direct_runner.sys, "platform", "linux")
    monkeypatch.setattr(trusted_direct_runner.os, "geteuid", lambda: 0)
    monkeypatch.setenv(
        trusted_direct_runner._MANAGED_CGROUP_ROOT_ENV,
        service_relative,
    )
    monkeypatch.setenv(
        trusted_direct_runner._MANAGED_CGROUP_UNIT_ENV,
        "zettlab-claw.service",
    )
    return cgroup_root, service, supervisor, proc_self, service_relative


@pytest.mark.skipif(os.name == "nt", reason="secret FD transport is POSIX-only")
def test_secret_uses_one_shot_fd_not_popen_or_control_env(monkeypatch, tmp_path):
    script = _write_script(
        tmp_path,
        """
        import json
        import os
        import sys

        descriptor = int(os.environ.pop("TEST_TOKEN_FD"))
        descriptor_inheritable = os.get_inheritable(descriptor)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            token = stream.read(4097).decode("utf-8")
        print(json.dumps({
            "argv": sys.argv[1:],
            "descriptor_inheritable": descriptor_inheritable,
            "metadata": os.environ.get("TEST_METADATA"),
            "secret": token == "one-shot-secret",
            "secret_env_absent": "TEST_TOKEN" not in os.environ,
        }))
        """,
    )
    real_popen = trusted_direct_runner.subprocess.Popen
    captured = {"control": bytearray()}

    def capture_popen(*args, **kwargs):
        captured["argv"] = args[0]
        captured["env"] = kwargs["env"]
        process = real_popen(*args, **kwargs)
        process.stdin = _RecordingStdin(process.stdin, captured["control"])
        return process

    monkeypatch.setattr(
        trusted_direct_runner.subprocess,
        "Popen",
        capture_popen,
    )
    result = trusted_direct_runner.run_trusted_python_script(
        script=script,
        argv=[str(script), "preflight"],
        cwd=tmp_path,
        base_env=os.environ,
        injected_env={"TEST_METADATA": "turn-1"},
        injected_secrets={"TEST_TOKEN": "one-shot-secret"},
        timeout=5,
        stdlib_only=True,
    )

    assert result.returncode == 0
    assert json.loads(result.output) == {
        "argv": ["preflight"],
        "descriptor_inheritable": False,
        "metadata": "turn-1",
        "secret": True,
        "secret_env_absent": True,
    }
    captured_text = json.dumps(
        {
            "argv": captured["argv"],
            "env": captured["env"],
            "control": captured["control"].decode("utf-8"),
        }
    )
    assert "one-shot-secret" not in captured_text
    assert "TEST_TOKEN" not in captured["env"]
    assert "TEST_TOKEN_FD" not in captured["env"]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="prctl hardening is Linux-specific",
)
def test_control_wrapper_is_nondumpable_and_no_new_privileges(tmp_path):
    script = _write_script(
        tmp_path,
        """
        import ctypes
        import json

        libc = ctypes.CDLL(None, use_errno=True)
        print(json.dumps({
            "dumpable": libc.prctl(3, 0, 0, 0, 0),
            "no_new_privs": libc.prctl(39, 0, 0, 0, 0),
        }))
        """,
    )

    result = trusted_direct_runner.run_trusted_python_script(
        script=script,
        argv=[str(script)],
        cwd=tmp_path,
        base_env=os.environ,
        injected_env={},
        timeout=5,
        stdlib_only=True,
    )

    assert result.returncode == 0
    assert json.loads(result.output) == {
        "dumpable": 0,
        "no_new_privs": 1,
    }


@pytest.mark.parametrize(
    ("metadata", "secrets"),
    [
        ({"TOKEN": "metadata"}, {"TOKEN": "secret"}),
        ({"TOKEN_FD": "metadata"}, {"TOKEN": "secret"}),
        ({}, {"TOKEN": "secret", "TOKEN_FD": "other-secret"}),
        ({}, {"INVALID-KEY": "secret"}),
        ({}, {"TOKEN": "x" * 4097}),
    ],
)
def test_secret_descriptor_contract_rejects_ambiguous_or_unbounded_values(
    tmp_path,
    metadata,
    secrets,
):
    script = _write_script(tmp_path, "print('must not run')\n")

    with pytest.raises(ValueError):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env=metadata,
            injected_secrets=secrets,
            timeout=5,
        )


def test_managed_wrapper_attaches_and_drops_identity_before_reading_control():
    source = trusted_direct_runner._CONTROL_WRAPPER
    bootstrap_start = source.index("def enter_managed_cgroup():")
    bootstrap_end = source.index("\nenter_managed_cgroup()", bootstrap_start)
    bootstrap = source[bootstrap_start:bootstrap_end]

    assert bootstrap.index("write_cgroup_file(") < bootstrap.index("os.setgroups([])")
    assert bootstrap.index("os.setgroups([])") < bootstrap.index("os.setresgid(")
    assert bootstrap.index("os.setresgid(") < bootstrap.index("os.setresuid(")
    assert bootstrap.index("os.setresuid(") < bootstrap.rindex(
        "harden_linux_process()"
    )
    assert source.index("\nenter_managed_cgroup()") < source.index(
        "control = json.loads(sys.stdin.read()"
    )


def test_create_managed_cgroup_uses_delegation_root_sibling(
    monkeypatch,
    tmp_path,
):
    (
        _cgroup_root,
        service,
        supervisor,
        _proc_self,
        service_relative,
    ) = _managed_delegation_fixture(monkeypatch, tmp_path)
    real_mkdir = os.mkdir

    def materialize_cgroup(path, mode):
        real_mkdir(path, mode)
        path = Path(path)
        (path / "cgroup.procs").write_text("", encoding="ascii")
        (path / "cgroup.kill").write_text("", encoding="ascii")
        (path / "cgroup.events").write_text("populated 0\n", encoding="ascii")
        (path / "memory.max").write_text("", encoding="ascii")
        (path / "memory.swap.max").write_text("", encoding="ascii")
        (path / "memory.oom.group").write_text("", encoding="ascii")
        (path / "pids.max").write_text("", encoding="ascii")

    monkeypatch.setattr(trusted_direct_runner.os, "mkdir", materialize_cgroup)

    cgroup = trusted_direct_runner._create_managed_invocation_cgroup()

    assert cgroup.path.parent == service
    assert cgroup.path.parent != supervisor
    assert cgroup.path.name.startswith("agentcomputer-")
    assert cgroup.relative_path.startswith(
        "/system.slice/zettlab-claw.service/agentcomputer-"
    )
    assert cgroup.delegation_root_path == service
    assert cgroup.delegation_root_relative_path == service_relative
    service_identity = service.stat()
    assert cgroup.delegation_root_identity == (
        service_identity.st_dev,
        service_identity.st_ino,
    )
    assert (cgroup.path / "memory.max").read_text(encoding="ascii") == str(
        trusted_direct_runner._MANAGED_INVOCATION_MEMORY_MAX_BYTES
    )
    assert (cgroup.path / "memory.swap.max").read_text(
        encoding="ascii"
    ) == str(
        trusted_direct_runner._MANAGED_INVOCATION_MEMORY_SWAP_MAX_BYTES
    )
    assert (cgroup.path / "memory.oom.group").read_text(
        encoding="ascii"
    ).strip() == "1"
    assert (cgroup.path / "pids.max").read_text(encoding="ascii") == str(
        trusted_direct_runner._MANAGED_INVOCATION_PIDS_MAX
    )
    for child in cgroup.path.iterdir():
        child.unlink()
    cgroup.path.rmdir()


def test_managed_cgroup_requires_memory_and_pids_enabled_at_delegation_root(
    monkeypatch,
    tmp_path,
):
    _root, service, _supervisor, _proc_self, _relative = (
        _managed_delegation_fixture(monkeypatch, tmp_path)
    )
    (service / "cgroup.subtree_control").write_text("pids\n", encoding="ascii")

    with pytest.raises(OSError, match="controllers are not enabled"):
        trusted_direct_runner._create_managed_invocation_cgroup()


def test_managed_cgroup_requires_launcher_bound_root(monkeypatch, tmp_path):
    _managed_delegation_fixture(monkeypatch, tmp_path)
    monkeypatch.delenv(
        trusted_direct_runner._MANAGED_CGROUP_ROOT_ENV,
        raising=False,
    )

    with pytest.raises(OSError, match="delegation identity"):
        trusted_direct_runner._create_managed_invocation_cgroup()


def test_managed_cgroup_kill_waits_for_empty_then_removes(monkeypatch, tmp_path):
    cgroup_path = tmp_path / "agentcomputer-test"
    cgroup_path.mkdir()
    (cgroup_path / "cgroup.kill").write_text("", encoding="ascii")
    (cgroup_path / "cgroup.events").write_text("populated 0\n", encoding="ascii")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path="/service/agentcomputer-test",
    )
    waited = []

    class Process:
        def wait(self, timeout):
            waited.append(timeout)
            return 0

    removed = {}
    real_rmdir = os.rmdir

    def remove_virtual_cgroup(path):
        path = Path(path)
        removed["kill"] = (path / "cgroup.kill").read_text(encoding="ascii")
        for child in path.iterdir():
            child.unlink()
        real_rmdir(path)

    monkeypatch.setattr(trusted_direct_runner.os, "rmdir", remove_virtual_cgroup)

    trusted_direct_runner._kill_and_remove_managed_cgroup(cgroup, Process())

    assert removed["kill"] == "1"
    assert waited == [trusted_direct_runner._PROCESS_KILL_GRACE_SECONDS]
    assert not cgroup_path.exists()


def test_managed_cgroup_kill_retries_before_success(monkeypatch, tmp_path):
    cgroup_path = tmp_path / "agentcomputer-retry"
    cgroup_path.mkdir()
    (cgroup_path / "cgroup.kill").write_text("", encoding="ascii")
    (cgroup_path / "cgroup.events").write_text("populated 0\n", encoding="ascii")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path="/service/agentcomputer-retry",
    )
    attempts = []
    real_write = trusted_direct_runner._write_control_file

    def flaky_write(path, payload):
        if Path(path).name == "cgroup.kill":
            attempts.append(payload)
            if len(attempts) < trusted_direct_runner._CGROUP_KILL_RETRY_ATTEMPTS:
                raise OSError("emulated transient cgroup.kill failure")
        return real_write(path, payload)

    real_rmdir = os.rmdir

    def remove_virtual_cgroup(path):
        path = Path(path)
        for child in path.iterdir():
            child.unlink()
        real_rmdir(path)

    monkeypatch.setattr(trusted_direct_runner, "_write_control_file", flaky_write)
    monkeypatch.setattr(trusted_direct_runner.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(trusted_direct_runner.os, "rmdir", remove_virtual_cgroup)

    trusted_direct_runner._kill_and_remove_managed_cgroup(cgroup, None)

    assert attempts == [b"1", b"1", b"1"]
    assert not cgroup_path.exists()


def test_managed_cgroup_kill_failure_escalates_without_numeric_pid_kill(
    monkeypatch,
    tmp_path,
):
    cgroup_path = tmp_path / "agentcomputer-pid-fallback"
    cgroup_path.mkdir()
    (cgroup_path / "cgroup.kill").write_text("", encoding="ascii")
    (cgroup_path / "cgroup.events").write_text("populated 1\n", encoding="ascii")
    (cgroup_path / "cgroup.procs").write_text("424242\n", encoding="ascii")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path="/service/agentcomputer-pid-fallback",
    )

    def reject_cgroup_kill(path, payload):
        assert payload == b"1"
        raise OSError(f"emulated failure: {Path(path).name}")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_write_control_file",
        reject_cgroup_kill,
    )
    monkeypatch.setattr(trusted_direct_runner.time, "sleep", lambda _seconds: None)
    escalated = []

    def escalate(got_cgroup, cause):
        escalated.append((got_cgroup, cause))
        raise OSError("emulated exact-service escalation")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_escalate_managed_service_cleanup",
        escalate,
    )

    with pytest.raises(OSError, match="exact-service escalation"):
        trusted_direct_runner._kill_and_remove_managed_cgroup(cgroup, None)

    assert len(escalated) == 1
    assert escalated[0][0] == cgroup
    assert "cgroup kill failed" not in str(escalated[0][1])
    assert cgroup_path.is_dir()
    assert not hasattr(trusted_direct_runner, "_read_cgroup_pids")


def test_managed_cgroup_cleanup_failure_escalates_service_scope(
    monkeypatch,
    tmp_path,
):
    cgroup_path = tmp_path / "agentcomputer-populated"
    cgroup_path.mkdir()
    (cgroup_path / "cgroup.kill").write_text("", encoding="ascii")
    (cgroup_path / "cgroup.events").write_text("populated 1\n", encoding="ascii")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path="/service/agentcomputer-populated",
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_CGROUP_CLEANUP_TIMEOUT_SECONDS",
        0,
    )
    escalated = []

    def escalate(got_cgroup, cause):
        escalated.append((got_cgroup, cause))
        raise OSError("emulated service-scope cleanup")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_escalate_managed_service_cleanup",
        escalate,
    )

    with pytest.raises(OSError, match="service-scope cleanup"):
        trusted_direct_runner._kill_and_remove_managed_cgroup(cgroup, None)

    assert len(escalated) == 1
    assert escalated[0][0] == cgroup
    assert cgroup_path.is_dir()
    assert (cgroup_path / "cgroup.kill").read_text(encoding="ascii") == "1"


def test_managed_cleanup_escalation_kills_exact_service_scope(
    monkeypatch,
    tmp_path,
):
    (
        _cgroup_root,
        service,
        _supervisor,
        _proc_self,
        service_relative,
    ) = _managed_delegation_fixture(monkeypatch, tmp_path)
    invocation = service / "agentcomputer-stuck"
    invocation.mkdir()
    service_identity = service.stat()
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=invocation,
        relative_path="/system.slice/zettlab-claw.service/agentcomputer-stuck",
        delegation_root_path=service,
        delegation_root_relative_path=service_relative,
        delegation_root_identity=(
            service_identity.st_dev,
            service_identity.st_ino,
        ),
    )
    killed = []
    monkeypatch.setattr(
        trusted_direct_runner.os,
        "kill",
        lambda process_id, kill_signal: killed.append(
            (process_id, kill_signal)
        ),
    )

    with pytest.raises(OSError, match="cleanup escalation returned"):
        trusted_direct_runner._escalate_managed_service_cleanup(
            cgroup,
            OSError("child cgroup remained populated"),
        )

    assert (service / "cgroup.kill").read_text(encoding="ascii") == "1"
    assert killed == [(os.getpid(), signal.SIGKILL)]


@pytest.mark.parametrize("systemctl_succeeds", [True, False])
def test_managed_cleanup_service_kill_failure_requests_exact_systemd_sigkill(
    monkeypatch,
    tmp_path,
    systemctl_succeeds,
):
    (
        _cgroup_root,
        service,
        _supervisor,
        _proc_self,
        service_relative,
    ) = _managed_delegation_fixture(monkeypatch, tmp_path)
    invocation = service / "agentcomputer-stuck"
    invocation.mkdir()
    service_identity = service.stat()
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=invocation,
        relative_path="/system.slice/zettlab-claw.service/agentcomputer-stuck",
        delegation_root_path=service,
        delegation_root_relative_path=service_relative,
        delegation_root_identity=(
            service_identity.st_dev,
            service_identity.st_ino,
        ),
    )
    cgroup_kill_attempts = []

    def reject_cgroup_kill(path, payload):
        cgroup_kill_attempts.append((Path(path), payload))
        raise OSError("emulated service cgroup.kill failure")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_write_control_file",
        reject_cgroup_kill,
    )
    monkeypatch.setattr(
        trusted_direct_runner.time,
        "sleep",
        lambda _seconds: None,
    )
    monkeypatch.setenv("ZET_AGENT_KEY", "must-not-reach-systemctl")
    systemctl_calls = []

    class SystemctlProcess:
        def poll(self):
            return 0 if systemctl_succeeds else None

    def fake_popen(argv, **kwargs):
        systemctl_calls.append((argv, kwargs))
        return SystemctlProcess()

    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", fake_popen)
    if not systemctl_succeeds:
        monkeypatch.setattr(
            trusted_direct_runner,
            "_SYSTEMD_KILL_TIMEOUT_SECONDS",
            0,
        )
    killed = []
    monkeypatch.setattr(
        trusted_direct_runner.os,
        "kill",
        lambda process_id, kill_signal: killed.append(
            (process_id, kill_signal)
        ),
    )

    expected_error = (
        "cgroup cleanup escalation failed"
        if systemctl_succeeds
        else "SIGKILL escalation failed"
    )
    with pytest.raises(OSError, match=expected_error):
        trusted_direct_runner._escalate_managed_service_cleanup(
            cgroup,
            OSError("invocation cgroup.kill failed"),
        )

    assert cgroup_kill_attempts == [
        (service / "cgroup.kill", b"1"),
        (service / "cgroup.kill", b"1"),
        (service / "cgroup.kill", b"1"),
    ]
    assert len(systemctl_calls) == (1 if systemctl_succeeds else 2)
    for argv, kwargs in systemctl_calls:
        assert argv == [
            "/bin/systemctl",
            "--system",
            "kill",
            "--kill-whom=all",
            "--signal=SIGKILL",
            "zettlab-claw.service",
        ]
        assert kwargs.get("shell", False) is False
        assert kwargs["env"] == {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "SYSTEMD_COLORS": "0",
        }
        assert "ZET_AGENT_KEY" not in kwargs["env"]
    assert killed == [(os.getpid(), signal.SIGKILL)]


def test_managed_cleanup_identity_mismatch_only_kills_main_process(
    monkeypatch,
    tmp_path,
):
    (
        _cgroup_root,
        service,
        _supervisor,
        _proc_self,
        service_relative,
    ) = _managed_delegation_fixture(monkeypatch, tmp_path)
    invocation = service / "agentcomputer-stuck"
    invocation.mkdir()
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=invocation,
        relative_path="/system.slice/zettlab-claw.service/agentcomputer-stuck",
        delegation_root_path=service,
        delegation_root_relative_path=service_relative,
        delegation_root_identity=(0, 0),
    )
    killed = []
    monkeypatch.setattr(
        trusted_direct_runner.os,
        "kill",
        lambda process_id, kill_signal: killed.append(
            (process_id, kill_signal)
        ),
    )

    with pytest.raises(OSError, match="identity verification failed"):
        trusted_direct_runner._escalate_managed_service_cleanup(
            cgroup,
            OSError("child cgroup remained populated"),
        )

    assert (service / "cgroup.kill").read_text(encoding="ascii") == ""
    assert killed == [(os.getpid(), signal.SIGKILL)]


def test_managed_cgroup_unavailable_fails_before_popen(monkeypatch, tmp_path):
    script = _write_script(tmp_path, "print('must not run')\n")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")

    def unavailable():
        raise OSError("delegated cgroup unavailable")

    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("Popen must not run without a delegated cgroup")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_create_managed_invocation_cgroup",
        unavailable,
    )
    monkeypatch.setattr(
        trusted_direct_runner.subprocess,
        "Popen",
        forbidden_popen,
    )

    with pytest.raises(OSError, match="delegated cgroup unavailable"):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env={},
            timeout=5,
        )


def test_managed_runner_fails_before_popen_when_identity_pool_is_occupied(
    monkeypatch,
    tmp_path,
):
    script = _write_script(tmp_path, "print('must not run')\n")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_select_managed_runner_identity",
        _REAL_SELECT_MANAGED_RUNNER_IDENTITY,
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_occupied_process_uids",
        lambda: set(
            range(
                trusted_direct_runner._MANAGED_RUNNER_UID_MIN,
                trusted_direct_runner._MANAGED_RUNNER_UID_MAX + 1,
            )
        ),
    )
    monkeypatch.setattr(
        trusted_direct_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Popen must not run with an occupied worker UID")
        ),
    )

    with pytest.raises(OSError, match="no isolated process identity"):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env={},
            injected_secrets={"TEST_TOKEN": "scope-token"},
            timeout=5,
        )


def test_occupied_process_uids_ignores_non_ascii_process_name(
    monkeypatch,
    tmp_path,
):
    proc_root = tmp_path / "proc"
    process_root = proc_root / "4242"
    process_root.mkdir(parents=True)
    (process_root / "status").write_bytes(
        b"Name:\tmodel-\xff\n"
        b"State:\tS (sleeping)\n"
        b"Uid:\t60001\t60002\t60003\t60004\n"
    )
    monkeypatch.setattr(trusted_direct_runner, "_PROC_ROOT", proc_root)

    assert trusted_direct_runner._occupied_process_uids() == {
        60001,
        60002,
        60003,
        60004,
    }


def test_managed_runner_does_not_hold_identity_lock_during_execution(
    monkeypatch,
    tmp_path,
):
    script = _write_script(tmp_path, "print('ok')\n")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_reserve_managed_runner_identity",
        lambda: (60001, 60001),
    )
    released = []

    def execute_without_global_lock(**_kwargs):
        acquired = trusted_direct_runner._MANAGED_RUNNER_LOCK.acquire(
            blocking=False
        )
        assert acquired is True
        trusted_direct_runner._MANAGED_RUNNER_LOCK.release()
        return trusted_direct_runner.TrustedPythonResult("", 0)

    monkeypatch.setattr(
        trusted_direct_runner,
        "_run_trusted_python_script_unlocked",
        execute_without_global_lock,
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_release_managed_runner_identity",
        lambda identity: released.append(identity),
    )

    result = trusted_direct_runner.run_trusted_python_script(
        script=script,
        argv=[str(script)],
        cwd=tmp_path,
        base_env=os.environ,
        injected_env={},
        timeout=5,
    )
    assert result.returncode == 0
    assert released == [(60001, 60001)]


def test_managed_popen_failure_cleans_new_cgroup(monkeypatch, tmp_path):
    script = _write_script(tmp_path, "print('must not run')\n")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=tmp_path / "agentcomputer-spawn-failed",
        relative_path="/service/agentcomputer-spawn-failed",
    )
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_create_managed_invocation_cgroup",
        lambda: cgroup,
    )

    def spawn_failed(*_args, **_kwargs):
        raise OSError("emulated spawn failure")

    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", spawn_failed)
    cleaned = []
    monkeypatch.setattr(
        trusted_direct_runner,
        "_kill_and_remove_managed_cgroup",
        lambda got_cgroup, got_process: cleaned.append(
            (got_cgroup, got_process)
        ),
    )

    with pytest.raises(OSError, match="emulated spawn failure"):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env={},
            timeout=5,
        )

    assert cleaned == [(cgroup, None)]


def test_thread_start_failure_reaps_managed_process_and_closes_fds(
    monkeypatch,
    tmp_path,
):
    script = _write_script(tmp_path, "print('must not run')\n")
    cgroup_path = tmp_path / "agentcomputer-thread-failed"
    cgroup_path.mkdir()
    (cgroup_path / "cgroup.kill").write_text("", encoding="ascii")
    (cgroup_path / "cgroup.events").write_text(
        "populated 0\n",
        encoding="ascii",
    )
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path="/service/agentcomputer-thread-failed",
    )
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_create_managed_invocation_cgroup",
        lambda: cgroup,
    )

    waited = []

    class Process:
        pid = 4242
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            waited.append(timeout)
            self.returncode = -9
            return self.returncode

    process = Process()
    monkeypatch.setattr(
        trusted_direct_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    pipe_fds = []
    real_pipe = os.pipe

    def recording_pipe():
        pair = real_pipe()
        pipe_fds.extend(pair)
        return pair

    monkeypatch.setattr(trusted_direct_runner.os, "pipe", recording_pipe)

    def thread_start_failed(_thread):
        raise RuntimeError("emulated thread start failure")

    monkeypatch.setattr(
        trusted_direct_runner.threading.Thread,
        "start",
        thread_start_failed,
    )
    removed = {}
    real_rmdir = os.rmdir

    def remove_virtual_cgroup(path):
        path = Path(path)
        removed["kill"] = (path / "cgroup.kill").read_text(encoding="ascii")
        for child in path.iterdir():
            child.unlink()
        real_rmdir(path)

    monkeypatch.setattr(
        trusted_direct_runner.os,
        "rmdir",
        remove_virtual_cgroup,
    )

    with pytest.raises(RuntimeError, match="emulated thread start failure"):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env={},
            injected_secrets={"TEST_TOKEN": "scope-token"},
            timeout=5,
        )

    assert removed["kill"] == "1"
    assert waited == [trusted_direct_runner._PROCESS_KILL_GRACE_SECONDS]
    assert not cgroup_path.exists()
    assert process.stdin.closed
    assert process.stdout.closed
    for descriptor in pipe_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    ("outcome", "expected_returncode"),
    [("normal", 0), ("timeout", 124), ("interrupt", 130)],
)
def test_managed_invocation_cleans_cgroup_on_every_outcome(
    monkeypatch,
    tmp_path,
    outcome,
    expected_returncode,
):
    script = _write_script(tmp_path, "print('emulated worker')\n")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=tmp_path / "agentcomputer-emulated",
        relative_path="/service/agentcomputer-emulated",
    )
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_create_managed_invocation_cgroup",
        lambda: cgroup,
    )
    popen_argv = []

    class Process:
        pid = 4242
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        returncode = 0 if outcome == "normal" else None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("emulated", timeout)
            return self.returncode

    process = Process()

    def fake_popen(argv, **_kwargs):
        popen_argv.extend(argv)
        return process

    monkeypatch.setattr(trusted_direct_runner.subprocess, "Popen", fake_popen)
    if outcome == "interrupt":
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: True)
    cleaned = []

    def cleanup(got_cgroup, got_process):
        cleaned.append((got_cgroup, got_process))
        if got_process.returncode is None:
            got_process.returncode = -9

    monkeypatch.setattr(
        trusted_direct_runner,
        "_kill_and_remove_managed_cgroup",
        cleanup,
    )

    result = trusted_direct_runner.run_trusted_python_script(
        script=script,
        argv=[str(script)],
        cwd=tmp_path,
        base_env=os.environ,
        injected_env={},
        timeout=0.01 if outcome == "timeout" else 5,
    )

    assert result.returncode == expected_returncode
    assert "--managed-cgroup" in popen_argv
    assert cleaned == [(cgroup, process)]


def test_managed_cleanup_error_fails_invocation(monkeypatch, tmp_path):
    script = _write_script(tmp_path, "print('emulated worker')\n")
    cgroup = trusted_direct_runner._ManagedInvocationCgroup(
        path=tmp_path / "agentcomputer-emulated",
        relative_path="/service/agentcomputer-emulated",
    )
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(
        trusted_direct_runner,
        "_create_managed_invocation_cgroup",
        lambda: cgroup,
    )

    class Process:
        pid = 4242
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(
        trusted_direct_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: Process(),
    )

    def cleanup_failed(*_args):
        raise OSError("emulated cleanup failure")

    monkeypatch.setattr(
        trusted_direct_runner,
        "_kill_and_remove_managed_cgroup",
        cleanup_failed,
    )

    with pytest.raises(OSError, match="managed Agent Creator cleanup failed"):
        trusted_direct_runner.run_trusted_python_script(
            script=script,
            argv=[str(script)],
            cwd=tmp_path,
            base_env=os.environ,
            injected_env={},
            timeout=5,
        )


def test_posix_fallback_does_not_use_psutil_descendant_snapshots():
    import inspect

    source = inspect.getsource(trusted_direct_runner._terminate_process_tree)
    assert "psutil" not in source
    assert "os.killpg" in source


@pytest.mark.skipif(
    os.environ.get("HERMES_TEST_REAL_DELEGATED_CGROUP") != "1",
    reason="requires an explicitly delegated cgroup v2 test service",
)
@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="managed gateway boundary requires Linux root",
)
@pytest.mark.live_system_guard_bypass
def test_real_managed_cgroup_kills_setsid_double_fork(monkeypatch):
    marker = Path("/tmp") / (
        f"hermes-cgroup-escape-{os.getpid()}-{time.monotonic_ns()}"
    )
    marker.unlink(missing_ok=True)
    service_relative = os.environ.get(
        trusted_direct_runner._MANAGED_CGROUP_ROOT_ENV,
        "",
    )
    assert service_relative.startswith("/")
    service_cgroup = (
        trusted_direct_runner._CGROUP2_ROOT
        / service_relative.lstrip("/")
    )
    before = {path.name for path in service_cgroup.glob("agentcomputer-*")}
    daemon_source = textwrap.dedent(
        f"""
        import os
        import time

        if os.fork():
            os._exit(0)
        os.setsid()
        if os.fork():
            os._exit(0)
        time.sleep(0.8)
        with open({str(marker)!r}, "w", encoding="utf-8") as stream:
            stream.write("escaped")
        """
    )
    script_source = textwrap.dedent(
        f"""
        import ctypes
        import json
        import os
        import subprocess
        import sys
        import time

        libc = ctypes.CDLL(None, use_errno=True)
        print(json.dumps({{
            "uid": os.getuid(),
            "gid": os.getgid(),
            "groups": os.getgroups(),
            "dumpable": libc.prctl(3, 0, 0, 0, 0),
            "no_new_privs": libc.prctl(39, 0, 0, 0, 0),
        }}), flush=True)
        subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", {daemon_source!r}],
            start_new_session=True,
        )
        time.sleep(30)
        """
    ).encode("utf-8")
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")

    try:
        result = trusted_direct_runner.run_trusted_python_script(
            script=Path("/tmp/agent-creator-cgroup-test.py"),
            argv=["/tmp/agent-creator-cgroup-test.py"],
            cwd=Path("/tmp"),
            base_env=os.environ,
            injected_env={},
            injected_secrets={"TEST_TOKEN": "scope-token"},
            timeout=0.2,
            script_bytes=script_source,
            stdlib_only=True,
        )

        assert result.returncode == 124
        assert result.timed_out is True
        identity = json.loads(result.output)
        assert (
            trusted_direct_runner._MANAGED_RUNNER_UID_MIN
            <= identity["uid"]
            <= trusted_direct_runner._MANAGED_RUNNER_UID_MAX
        )
        assert identity == {
            "uid": identity["uid"],
            "gid": identity["uid"],
            "groups": [],
            "dumpable": 0,
            "no_new_privs": 1,
        }
        time.sleep(1.0)
        assert not marker.exists()
        after = {
            path.name for path in service_cgroup.glob("agentcomputer-*")
        }
        assert after == before
    finally:
        marker.unlink(missing_ok=True)


_TRANSIENT_PROBE_PREFIX = "HERMES_TRANSIENT_CGROUP_PROBE="


def _run_transient_systemd_cgroup_probe(unit_name: str) -> None:
    """Entry point executed as the MainPID of a disposable delegated service."""

    import errno
    import runpy

    repo_root = Path(__file__).resolve().parents[2]
    launcher_namespace = runpy.run_path(
        str(repo_root / "zpk/libexec/hermes-secure-launcher.py")
    )
    os.environ["HERMES_MANAGED_CGROUP_UNIT"] = unit_name
    os.environ.pop("HERMES_MANAGED_CGROUP_ROOT", None)
    launcher_namespace["_prepare_managed_service_cgroup"]()
    launcher_namespace["_prepare_managed_service_cgroup"]()
    os.environ["HERMES_MANAGED_GATEWAY"] = "1"

    service_relative = os.environ["HERMES_MANAGED_CGROUP_ROOT"]
    service = Path("/sys/fs/cgroup") / service_relative.lstrip("/")
    before = {path.name for path in service.glob("agentcomputer-*")}

    normal = trusted_direct_runner.run_trusted_python_script(
        script=Path("/tmp/hermes-cgroup-normal.py"),
        argv=["/tmp/hermes-cgroup-normal.py"],
        cwd=Path("/tmp"),
        base_env=os.environ,
        injected_env={},
        timeout=5,
        script_bytes=b"print('normal')\n",
        stdlib_only=True,
    )
    memory = trusted_direct_runner.run_trusted_python_script(
        script=Path("/tmp/hermes-cgroup-memory.py"),
        argv=["/tmp/hermes-cgroup-memory.py"],
        cwd=Path("/tmp"),
        base_env=os.environ,
        injected_env={},
        timeout=5,
        script_bytes=(
            b"payload = bytearray(160 * 1024 * 1024)\n"
            b"for offset in range(0, len(payload), 4096):\n"
            b"    payload[offset] = 1\n"
            b"print(len(payload))\n"
        ),
        stdlib_only=True,
    )
    pids_source = textwrap.dedent(
        """
        import errno
        import json
        import os
        import signal
        import time

        children = []
        failure = None
        try:
            for _ in range(80):
                child = os.fork()
                if child == 0:
                    time.sleep(30)
                    os._exit(0)
                children.append(child)
        except OSError as exc:
            failure = exc.errno
        finally:
            for child in children:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for child in children:
                try:
                    os.waitpid(child, 0)
                except ChildProcessError:
                    pass
        print(json.dumps({"failure": failure, "children": len(children)}))
        """
    ).encode("utf-8")
    pids = trusted_direct_runner.run_trusted_python_script(
        script=Path("/tmp/hermes-cgroup-pids.py"),
        argv=["/tmp/hermes-cgroup-pids.py"],
        cwd=Path("/tmp"),
        base_env=os.environ,
        injected_env={},
        timeout=10,
        script_bytes=pids_source,
        stdlib_only=True,
    )

    marker = Path("/tmp") / f"hermes-cgroup-probe-{os.getpid()}"
    marker.unlink(missing_ok=True)
    daemon_source = textwrap.dedent(
        f"""
        import os
        import time

        if os.fork():
            os._exit(0)
        os.setsid()
        if os.fork():
            os._exit(0)
        time.sleep(0.8)
        with open({str(marker)!r}, "w", encoding="utf-8") as stream:
            stream.write("escaped")
        """
    )
    timeout_result = trusted_direct_runner.run_trusted_python_script(
        script=Path("/tmp/hermes-cgroup-timeout.py"),
        argv=["/tmp/hermes-cgroup-timeout.py"],
        cwd=Path("/tmp"),
        base_env=os.environ,
        injected_env={},
        timeout=0.2,
        script_bytes=textwrap.dedent(
            f"""
            import subprocess
            import sys
            import time

            subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", {daemon_source!r}],
                start_new_session=True,
            )
            time.sleep(30)
            """
        ).encode("utf-8"),
        stdlib_only=True,
    )
    time.sleep(1)
    after = {path.name for path in service.glob("agentcomputer-*")}
    pids_state = json.loads(pids.output)
    state = {
        "root": service_relative,
        "current": trusted_direct_runner._current_unified_cgroup(),
        "enabled": sorted(
            (service / "cgroup.subtree_control").read_text(
                encoding="ascii"
            ).split()
        ),
        "normal": [normal.returncode, normal.output],
        "memory_returncode": memory.returncode,
        "pids": pids_state,
        "timeout": [timeout_result.returncode, timeout_result.timed_out],
        "marker_exists": marker.exists(),
        "residual": sorted(after - before),
    }
    marker.unlink(missing_ok=True)
    assert normal.returncode == 0 and normal.output == "normal"
    assert memory.returncode != 0
    assert pids.returncode == 0
    assert pids_state["failure"] == errno.EAGAIN
    assert 0 < pids_state["children"] < 64
    assert timeout_result.returncode == 124 and timeout_result.timed_out
    assert not state["marker_exists"]
    assert not state["residual"]
    print(_TRANSIENT_PROBE_PREFIX + json.dumps(state, sort_keys=True))


@pytest.mark.skipif(
    os.environ.get("HERMES_TEST_REAL_TRANSIENT_CGROUP") != "1",
    reason="requires explicit disposable systemd/cgroup v2 authorization",
)
@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="transient delegated service requires Linux root",
)
@pytest.mark.live_system_guard_bypass
def test_real_transient_systemd_252_resource_and_cleanup_flow():
    if not Path("/run/systemd/system").is_dir():
        pytest.skip("systemd is not PID 1")
    unit_name = f"hermes-cgroup-test-{uuid.uuid4().hex[:12]}.service"
    source = (
        "import runpy, sys;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
        f"ns=runpy.run_path({str(Path(__file__).resolve())!r});"
        f"ns['_run_transient_systemd_cgroup_probe']({unit_name!r})"
    )
    completed = subprocess.run(
        [
            "systemd-run",
            "--quiet",
            "--wait",
            "--pipe",
            "--collect",
            f"--unit={unit_name}",
            "--property=Delegate=memory pids",
            "--property=KillMode=control-group",
            "--property=Restart=no",
            sys.executable,
            "-I",
            "-c",
            source,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    probe_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith(_TRANSIENT_PROBE_PREFIX)
    ]
    assert len(probe_lines) == 1, completed.stdout
    state = json.loads(probe_lines[0][len(_TRANSIENT_PROBE_PREFIX) :])
    assert state["root"].endswith(f"/{unit_name}")
    assert state["current"].endswith(
        f"/{unit_name}/{trusted_direct_runner._MANAGED_SUPERVISOR_CGROUP}"
    )
    assert {"memory", "pids"}.issubset(state["enabled"])


def _run_transient_systemd_sigkill_probe(
    unit_name: str,
    generation_path: str,
    escaped_path: str,
    restarted_path: str,
) -> None:
    generation = Path(generation_path)
    escaped = Path(escaped_path)
    restarted = Path(restarted_path)
    if generation.exists():
        restarted.touch()
        time.sleep(0.3)
        return

    generation.touch()
    ready_read, ready_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.setsid()
        if os.fork():
            os._exit(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.write(ready_write, b"1")
        os.close(ready_write)
        time.sleep(0.8)
        escaped.touch()
        time.sleep(30)
        os._exit(0)

    os.close(ready_write)
    assert os.read(ready_read, 1) == b"1"
    os.close(ready_read)
    os.environ[trusted_direct_runner._MANAGED_CGROUP_UNIT_ENV] = unit_name
    trusted_direct_runner._request_systemd_unit_sigkill(unit_name)
    raise AssertionError("systemd unit SIGKILL unexpectedly returned")


@pytest.mark.skipif(
    os.environ.get("HERMES_TEST_REAL_TRANSIENT_CGROUP") != "1",
    reason="requires explicit disposable systemd/cgroup v2 authorization",
)
@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.geteuid() != 0,
    reason="transient delegated service requires Linux root",
)
@pytest.mark.live_system_guard_bypass
def test_real_transient_systemd_252_exact_unit_sigkill_fallback():
    if not Path("/run/systemd/system").is_dir():
        pytest.skip("systemd is not PID 1")
    unique = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
    generation = Path("/run") / f"hermes-kill-{unique}.generation"
    escaped = Path("/run") / f"hermes-kill-{unique}.escaped"
    restarted = Path("/run") / f"hermes-kill-{unique}.restarted"
    unit_name = f"hermes-kill-test-{uuid.uuid4().hex[:12]}.service"
    source = (
        "import runpy, sys;"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
        f"ns=runpy.run_path({str(Path(__file__).resolve())!r});"
        "ns['_run_transient_systemd_sigkill_probe']("
        f"{unit_name!r},{str(generation)!r},{str(escaped)!r},"
        f"{str(restarted)!r})"
    )
    try:
        started = time.monotonic()
        completed = subprocess.run(
            [
                "systemd-run",
                "--quiet",
                "--wait",
                "--pipe",
                "--collect",
                f"--unit={unit_name}",
                "--property=KillMode=control-group",
                "--property=Restart=on-failure",
                "--property=RestartSec=0.2",
                "--property=TimeoutStopSec=5",
                "--property=NoNewPrivileges=true",
                "--property=PrivateTmp=true",
                "--property=ProtectProc=invisible",
                "--property=CapabilityBoundingSet=~CAP_SYS_PTRACE CAP_SYS_ADMIN",
                sys.executable,
                "-I",
                "-c",
                source,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.monotonic() - started
        assert completed.returncode == 0, completed.stderr
        assert elapsed < 4
        assert restarted.exists()
        assert not escaped.exists()
    finally:
        generation.unlink(missing_ok=True)
        escaped.unlink(missing_ok=True)
        restarted.unlink(missing_ok=True)
