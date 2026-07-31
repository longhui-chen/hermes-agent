import os
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.code_execution_tool as code_execution_module
import tools.environments.local as local_module
import tools.process_registry as process_registry_module
from tools.environments.local import LocalEnvironment
from tools.process_registry import ProcessRegistry, ProcessSession


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
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_cwd",
        lambda cwd, *, env: cwd,
    )
    monkeypatch.setattr(local_module, "_resolve_safe_cwd", lambda cwd: cwd)
    monkeypatch.setattr(local_module.os, "getpgid", lambda _pid: 42)
    monkeypatch.setattr(
        local_module,
        "_ensure_managed_terminal_cgroup",
        lambda _env=None: Path("/sys/fs/cgroup/unit/terminal-profile-65534"),
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_python", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_unshare", lambda: "/usr/bin/unshare"
    )
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_home_paths",
        lambda _env=None: (
            Path("/run/zettlab-claw/terminal-homes/65534"),
            Path("/run/zettlab-claw/terminal-homes/65534/tmp"),
            Path("/run/zettlab-claw/terminal-homes/65534/var-tmp"),
        ),
    )

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

    assert captured["argv"][:5] == [
        "/usr/bin/python3",
        "-I",
        "-c",
        local_module._MANAGED_TERMINAL_CGROUP_ENTER,
        "/sys/fs/cgroup/unit/terminal-profile-65534",
    ]
    assert captured["argv"][5:14] == [
        "/usr/bin/setpriv",
        "--reuid=65534",
        "--regid=65534",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
    ]
    assert captured["argv"][14:27] == [
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--fork",
        "--kill-child=KILL",
        "--",
        "/usr/bin/python3",
        "-I",
        "-c",
        local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER,
        "/run/zettlab-claw/terminal-homes/65534/tmp",
        "/run/zettlab-claw/terminal-homes/65534/var-tmp",
    ]
    assert captured["argv"][27:] == [
        "/bin/bash",
        "-c",
        "id",
    ]
    assert "mount(None,b'/',16384|262144)" in (
        local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER
    )
    assert "b'/tmp',4096|16384" in local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER
    assert "b'/var/tmp',4096|16384" in local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER


def test_managed_terminal_default_cwd_falls_back_to_profile_home(monkeypatch):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    run_env = {"HERMES_HOME": "/profiles/main"}

    def prepare_home(env):
        env["HOME"] = "/run/zettlab-claw/terminal-homes/100001"
        env["TMPDIR"] = "/tmp"
        env["TMP"] = "/tmp"
        env["TEMP"] = "/tmp"
        return env["HOME"]

    monkeypatch.setattr(
        local_module,
        "_prepare_managed_terminal_home",
        prepare_home,
    )
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_identity",
        lambda _env=None: (100001, 100001),
    )
    monkeypatch.setattr(
        local_module,
        "_managed_identity_can_traverse",
        lambda directory, **_kwargs: directory != "/root",
    )

    assert local_module._managed_terminal_cwd(
        "/root",
        env=run_env,
    ) == "/run/zettlab-claw/terminal-homes/100001"
    assert local_module._managed_terminal_cwd(
        "/workspace",
        env=run_env,
    ) == "/workspace"
    assert run_env["HOME"] == "/run/zettlab-claw/terminal-homes/100001"
    assert run_env["TMPDIR"] == "/tmp"
    assert run_env["TMP"] == "/tmp"
    assert run_env["TEMP"] == "/tmp"


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
        "_managed_execute_code_sandbox_argv",
        lambda argv, *, env, execution_scope: [
            "/usr/bin/setpriv",
            "--reuid=65534",
            "--regid=65534",
            "--clear-groups",
            "--bounding-set=-all",
            "--",
            *argv,
        ],
    )

    argv = code_execution_module._managed_execute_code_argv(
        "/app/venv/bin/python",
        "/tmp/hermes-execute/script.py",
        env={},
        execution_scope="scope-1",
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


def test_managed_execute_code_gets_unique_identity_from_terminal(monkeypatch):
    monkeypatch.setattr(local_module.os, "geteuid", lambda: 0)
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    env = {"HERMES_HOME": "/profiles/main"}

    terminal_uid, _ = local_module._managed_terminal_identity(env)
    first_uid, _ = local_module._managed_execute_code_identity(env, "run-1")
    second_uid, _ = local_module._managed_execute_code_identity(env, "run-2")

    assert len({terminal_uid, first_uid, second_uid}) == 3


def test_managed_execute_code_uses_per_invocation_cgroup(monkeypatch):
    import tools.trusted_direct_runner as trusted_runner

    cgroup = SimpleNamespace(path=Path("/sys/fs/cgroup/unit/execute-code-test"))
    cleaned = []
    calls = []
    info = SimpleNamespace(st_mode=stat.S_IFREG | 0o755, st_uid=0)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(local_module.os, "lstat", lambda _path: info)
    monkeypatch.setattr(
        local_module,
        "_managed_execute_code_identity",
        lambda _env, _scope: (61001, 61001),
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_python", lambda: "/usr/bin/python3"
    )
    monkeypatch.setattr(
        trusted_runner, "_create_managed_invocation_cgroup", lambda: cgroup
    )
    monkeypatch.setattr(
        trusted_runner,
        "_kill_and_remove_managed_cgroup",
        lambda got, process: cleaned.append((got, process)),
    )
    local_module._MANAGED_EXECUTE_CODE_CGROUP_BY_UID.clear()

    argv = local_module._managed_execute_code_sandbox_argv(
        ["/usr/bin/python3", "/tmp/script.py"],
        env={"HERMES_HOME": "/profiles/main"},
        execution_scope="scope-1",
    )

    assert argv[:6] == [
        "/usr/bin/python3",
        "-I",
        "-c",
        local_module._MANAGED_TERMINAL_CGROUP_ENTER,
        str(cgroup.path),
        "/usr/bin/setpriv",
    ]
    assert argv[6:9] == [
        "--reuid=61001",
        "--regid=61001",
        "--clear-groups",
    ]
    monkeypatch.setattr(
        local_module,
        "_terminate_managed_uid",
        lambda uid: calls.append(("kill", uid)) or 0,
    )
    monkeypatch.setattr(
        local_module,
        "_release_managed_execute_code_identity",
        lambda uid, _env, scope: calls.append(("release", uid, scope)),
    )
    local_module.retire_managed_execute_code_identity(
        61001,
        {"HERMES_HOME": "/profiles/main"},
        "scope-1",
    )
    assert cleaned == [(cgroup, None)]
    assert calls == [("kill", 61001), ("release", 61001, "scope-1")]
    assert local_module._MANAGED_EXECUTE_CODE_CGROUP_BY_UID == {}


def test_rpc_peer_must_match_expected_pid_and_uid():
    class Connection:
        def __init__(self, pid, uid):
            self.pid = pid
            self.uid = uid

        def getsockopt(self, _level, _option, _size):
            return struct.pack("3i", self.pid, self.uid, self.uid)

    expected = (1234, 4567)
    original = getattr(code_execution_module.socket, "SO_PEERCRED", None)
    code_execution_module.socket.SO_PEERCRED = 17
    try:
        code_execution_module._validate_rpc_peer(Connection(*expected), expected)
        with pytest.raises(PermissionError, match="identity mismatch"):
            code_execution_module._validate_rpc_peer(
                Connection(1234, 9999),
                expected,
            )
    finally:
        if original is None:
            delattr(code_execution_module.socket, "SO_PEERCRED")
        else:
            code_execution_module.socket.SO_PEERCRED = original


def test_managed_execute_code_preamble_disables_dumpability():
    assert "prctl(4, 0, 0, 0, 0)" in (
        code_execution_module._MANAGED_EXECUTE_CODE_PREAMBLE
    )


def test_managed_service_mounts_system_read_only_with_scoped_writes():
    service = Path("zpk/init.d/zettlab-claw.service").read_text(
        encoding="utf-8"
    )
    assert "ProtectSystem=strict" in service
    assert "RuntimeDirectory=zettlab-claw" in service
    assert "RuntimeDirectoryMode=0755" in service
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
    assert "PrivateTmp=true" in service
    assert "CAP_SYS_ADMIN" in service and "CapabilityBoundingSet=~CAP_SYS_ADMIN" in service
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
    monkeypatch.setattr(
        process_registry_module,
        "_managed_terminal_cwd",
        lambda cwd, *, env: cwd,
    )
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_home_paths",
        lambda _env=None: (
            Path("/run/zettlab-claw/terminal-homes/65534"),
            Path("/run/zettlab-claw/terminal-homes/65534/tmp"),
            Path("/run/zettlab-claw/terminal-homes/65534/var-tmp"),
        ),
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_unshare", lambda: "/usr/bin/unshare"
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
    monkeypatch.setattr(
        local_module,
        "_ensure_managed_terminal_cgroup",
        lambda _env=None: Path("/sys/fs/cgroup/unit/terminal-profile-65534"),
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_python", lambda: "/usr/bin/python3"
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
    assert captured["argv"][5:11] == [
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
    monkeypatch.setattr(
        local_module,
        "_ensure_managed_terminal_cgroup",
        lambda _env=None: Path("/sys/fs/cgroup/unit/terminal-profile-65534"),
    )
    monkeypatch.setattr(
        local_module, "_trusted_managed_python", lambda: "/usr/bin/python3"
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
    assert captured["argv"][5:11] == [
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
    local_module._MANAGED_TERMINAL_RETIRED_UIDS.clear()
    local_module._MANAGED_TERMINAL_RETIRED_SCOPES.clear()
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


def test_managed_terminal_cgroup_enforces_profile_limits(
    monkeypatch, tmp_path
):
    from tools import trusted_direct_runner

    delegation_root = tmp_path / "zettlab-claw.service"
    delegation_root.mkdir()
    real_mkdir = os.mkdir

    def materialize_cgroup(path, mode=0o777):
        real_mkdir(path, mode)
        path = Path(path)
        for control in (
            "cgroup.procs",
            "cgroup.kill",
            "cgroup.events",
            "memory.max",
            "memory.swap.max",
            "memory.oom.group",
            "pids.max",
        ):
            (path / control).write_text(
                "populated 0\n" if control == "cgroup.events" else "",
                encoding="ascii",
            )

    monkeypatch.setattr(local_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(local_module.os, "mkdir", materialize_cgroup)
    monkeypatch.setattr(
        local_module, "_managed_terminal_identity", lambda _env=None: (100001, 100001)
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_resolve_managed_delegation_root",
        lambda: (delegation_root, "/unit", (1, 2)),
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_write_control_file",
        lambda path, value: Path(path).write_bytes(value),
    )
    monkeypatch.setattr(
        trusted_direct_runner,
        "_read_bounded_ascii",
        lambda path, **_kwargs: Path(path).read_text(encoding="ascii"),
    )

    cgroup = local_module._ensure_managed_terminal_cgroup(
        {"HERMES_HOME": "/profiles/main"}
    )

    assert cgroup.name == "terminal-profile-100001"
    assert (cgroup / "memory.max").read_text() == str(256 * 1024 * 1024)
    assert (cgroup / "memory.swap.max").read_text() == "0"
    assert (cgroup / "memory.oom.group").read_text() == "1"
    assert (cgroup / "pids.max").read_text() == "64"


def test_process_registry_kill_all_is_scoped_to_immutable_profile(monkeypatch):
    registry = ProcessRegistry()
    first = str(Path("/profiles/first").resolve())
    second = str(Path("/profiles/second").resolve())
    registry._running = {
        "first": ProcessSession(
            id="first", command="sleep 1", profile_owner=first
        ),
        "second": ProcessSession(
            id="second", command="sleep 1", profile_owner=second
        ),
    }
    killed = []

    def kill_process(session_id, **_kwargs):
        killed.append(session_id)
        registry._running[session_id].exited = True
        return {"status": "killed"}

    monkeypatch.setattr(registry, "kill_process", kill_process)
    assert registry.kill_all(profile_owner=first) == 1
    assert killed == ["first"]
    assert registry._running["second"].exited is False


def test_managed_uid_inventory_ignores_zombies(tmp_path):
    running = tmp_path / "101"
    zombie = tmp_path / "102"
    other = tmp_path / "103"
    for process_dir in (running, zombie, other):
        process_dir.mkdir()
    running.joinpath("status").write_text(
        "State:\tS (sleeping)\nUid:\t100001\t100001\t100001\t100001\n"
    )
    zombie.joinpath("status").write_text(
        "State:\tZ (zombie)\nUid:\t100001\t100001\t100001\t100001\n"
    )
    other.joinpath("status").write_text(
        "State:\tS (sleeping)\nUid:\t100002\t100002\t100002\t100002\n"
    )

    assert local_module._managed_uid_processes(100001, tmp_path) == {101}


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="requires Linux root identity broker",
)
def test_profile_retirement_kills_background_and_rotates_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "profile-retirement-test-key")
    monkeypatch.setattr(
        local_module, "_MANAGED_TERMINAL_HOME_ROOT", tmp_path / "homes"
    )
    monkeypatch.setattr(
        local_module,
        "_managed_terminal_argv",
        lambda argv, *, env=None: (
            local_module._managed_terminal_privilege_drop_prefix(env) + list(argv)
        ),
    )
    monkeypatch.setattr(
        local_module, "_remove_managed_terminal_cgroup", lambda _uid: True
    )
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    local_module._MANAGED_TERMINAL_RETIRED_UIDS.clear()
    local_module._MANAGED_TERMINAL_RETIRED_SCOPES.clear()
    profile_home = str(tmp_path / "profile")
    env = {"HERMES_HOME": profile_home}
    uid, gid = local_module._managed_terminal_identity(env)
    homes = tmp_path / "homes"
    homes.mkdir(mode=0o711)
    home = homes / str(uid)
    home.mkdir(mode=0o700)
    os.chown(home, uid, gid)
    process = subprocess.Popen(
        local_module._managed_terminal_argv(
            ["/bin/sh", "-c", "sleep 60"], env=env
        ),
        start_new_session=True,
    )
    deadline = time.monotonic() + 2
    while process.poll() is None and process.pid not in local_module._managed_uid_processes(uid):
        if time.monotonic() >= deadline:
            process.kill()
            pytest.fail("managed process did not enter its UID domain")
        time.sleep(0.02)

    result = local_module.retire_managed_terminal_profile(profile_home)
    process.wait(timeout=2)
    new_uid, _ = local_module._managed_terminal_identity(env)
    assert result["identity_retired"] is True
    assert result["terminal_home_removed"] is True
    assert result["terminal_cgroup_removed"] is True
    assert result["killed_uid_processes"] >= 1
    assert not home.exists()
    assert new_uid != uid


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
