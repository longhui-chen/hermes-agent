"""Regression coverage for per-invocation execute_code identity retirement."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import pytest

import tools.code_execution_tool as code_execution_module
from tools.environments import local


def test_retirement_terminates_before_releasing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        local, "_terminate_managed_uid", lambda uid: calls.append(("kill", uid)) or 2
    )
    monkeypatch.setattr(
        local,
        "_release_managed_execute_code_identity",
        lambda uid, env, scope: calls.append(("release", uid, env["HERMES_HOME"], scope)),
    )

    killed = local.retire_managed_execute_code_identity(
        61001, {"HERMES_HOME": "/profiles/coder"}, "run-1"
    )

    assert killed == 2
    assert calls == [
        ("kill", 61001),
        ("release", 61001, "/profiles/coder", "run-1"),
    ]


def test_retirement_keeps_reservation_when_termination_fails(monkeypatch):
    released = []
    monkeypatch.setattr(
        local,
        "_terminate_managed_uid",
        lambda _uid: (_ for _ in ()).throw(OSError("descendant survived")),
    )
    monkeypatch.setattr(
        local,
        "_release_managed_execute_code_identity",
        lambda *args: released.append(args),
    )

    with pytest.raises(OSError, match="descendant survived"):
        local.retire_managed_execute_code_identity(
            61002, {"HERMES_HOME": "/profiles/coder"}, "run-2"
        )

    assert released == []


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="requires Linux root cgroup delegation",
)
def test_retirement_kills_root_detached_descendant(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "managed-execute-retirement-test")
    env = {"HERMES_HOME": str(tmp_path / "profile")}
    scope = "detached-descendant"
    workspace = Path(
        tempfile.mkdtemp(prefix="hermes-execute-private-", dir="/tmp")
    )
    uid = local._prepare_managed_execute_code_workspace(
        str(workspace), [], env=env, execution_scope=scope
    )
    descendant_file = workspace / "descendant.pid"
    child_code = (
        "import pathlib,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,start_new_session=True);"
        f"pathlib.Path({str(descendant_file)!r}).write_text(str(p.pid))"
    )
    argv = local._managed_execute_code_sandbox_argv(
        [sys.executable, "-c", child_code],
        env=env,
        execution_scope=scope,
        workspace=str(workspace),
    )

    descendant_pid = None
    try:
        subprocess.run(argv, check=True, timeout=10)
        descendant_pid = int(descendant_file.read_text())
        assert Path(f"/proc/{descendant_pid}").exists()

        assert local.retire_managed_execute_code_identity(uid, env, scope) == 0
        deadline = time.monotonic() + 2
        while (
            Path(f"/proc/{descendant_pid}").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert not Path(f"/proc/{descendant_pid}").exists()
        assert local._managed_uid_processes(uid) == set()
        assert uid not in local._MANAGED_TERMINAL_SCOPE_BY_UID
    finally:
        if uid in local._MANAGED_TERMINAL_SCOPE_BY_UID:
            try:
                local.retire_managed_execute_code_identity(uid, env, scope)
            except OSError:
                pass
        try:
            local._terminate_managed_uid(uid)
        except OSError:
            for pid in local._managed_uid_processes(uid, Path("/proc")):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        local._MANAGED_TERMINAL_SCOPE_BY_UID.pop(uid, None)
        local._MANAGED_EXECUTE_CODE_CGROUP_BY_UID.pop(uid, None)
        if (
            descendant_pid is not None
            and Path(f"/proc/{descendant_pid}").exists()
        ):
            try:
                os.kill(descendant_pid, 9)
            except ProcessLookupError:
                pass
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="requires Linux root cgroup delegation",
)
def test_root_execute_code_preserves_rpc_peer_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "managed-execute-rpc-peer-test")
    env = {"HERMES_HOME": str(tmp_path / "profile")}
    scope = "rpc-peer"
    workspace = Path(
        tempfile.mkdtemp(prefix="hermes-execute-rpc-", dir="/tmp")
    )
    socket_path = workspace / "rpc.sock"
    marker = workspace / "rpc-marker"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(1)
    server.settimeout(10)
    uid = local._prepare_managed_execute_code_workspace(
        str(workspace),
        [str(socket_path)],
        env=env,
        execution_scope=scope,
    )
    child_code = (
        "import os,pathlib,socket;"
        "s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);"
        "s.connect(os.environ['HERMES_RPC_SOCKET']);"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));"
        "s.sendall(b'connected');s.close()"
    )
    argv = local._managed_execute_code_sandbox_argv(
        [sys.executable, "-c", child_code],
        env=env,
        execution_scope=scope,
        workspace=str(workspace),
    )
    child_env = os.environ.copy()
    child_env["HERMES_RPC_SOCKET"] = str(socket_path)
    proc = None
    connection = None

    try:
        proc = subprocess.Popen(argv, env=child_env)
        connection, _ = server.accept()
        code_execution_module._validate_rpc_peer(
            connection,
            (proc.pid, getattr(os, "geteuid", lambda: -1)()),
        )
        assert connection.recv(32) == b"connected"
        assert proc.wait(timeout=10) == 0
        assert marker.read_text() == str(proc.pid)
        local.retire_managed_execute_code_identity(uid, env, scope)
        assert uid not in local._MANAGED_TERMINAL_SCOPE_BY_UID
    finally:
        if connection is not None:
            connection.close()
        server.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        if uid in local._MANAGED_TERMINAL_SCOPE_BY_UID:
            try:
                local.retire_managed_execute_code_identity(uid, env, scope)
            except OSError:
                pass
        try:
            local._terminate_managed_uid(uid)
        except OSError:
            pass
        local._MANAGED_TERMINAL_SCOPE_BY_UID.pop(uid, None)
        local._MANAGED_EXECUTE_CODE_CGROUP_BY_UID.pop(uid, None)
        shutil.rmtree(workspace, ignore_errors=True)
