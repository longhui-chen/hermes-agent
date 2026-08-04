import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
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


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="requires root POSIX ownership semantics",
)
def test_managed_profile_runtime_exposes_only_active_skills_and_output(
    monkeypatch, request
):
    tmp_path = Path(tempfile.mkdtemp(prefix="hermes-runtime-test-", dir="/run"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    os.chmod(tmp_path, 0o755)
    hermes_root = tmp_path / "hermes_home"
    profiles_root = hermes_root / "profiles"
    profile_home = profiles_root / "agent-a"
    skills_root = profile_home / "skills"
    sibling_home = profiles_root / "agent-b"
    output = tmp_path / "agents" / "data" / "agent-a" / "output"
    skills_root.mkdir(parents=True)
    sibling_home.mkdir()
    output.mkdir(parents=True)
    private_skill_dir = skills_root / "support-suite" / "scripts"
    private_skill_dir.mkdir(parents=True)
    private_skill = private_skill_dir / "onboard.py"
    private_skill.write_text("print('ok')")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "must-not-be-chowned.txt"
    outside_file.write_text("protected")
    os.chmod(outside_file, 0o600)
    output_link = output / "outside-link"
    output_link.symlink_to(outside, target_is_directory=True)
    hermes_alias = tmp_path / "hermes-alias"
    hermes_alias.symlink_to(hermes_root, target_is_directory=True)
    lexical_profile_home = hermes_alias / "profiles" / "agent-a"
    lexical_skill = (
        lexical_profile_home / "skills" / "support-suite" / "scripts" / "onboard.py"
    )
    for path in (hermes_root, profiles_root, profile_home, skills_root, sibling_home):
        os.chmod(path, 0o700)
    # Installed board packages may arrive as root-owned 707/607. Preparation
    # must safely tighten those modes instead of rejecting the entrypoint.
    os.chmod(private_skill_dir.parent, 0o707)
    os.chmod(private_skill_dir, 0o707)
    os.chmod(private_skill, 0o607)
    os.chmod(output, 0o755)

    monkeypatch.setattr(local_module, "_IS_WINDOWS", False)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    env = {
        "HERMES_HOME": str(lexical_profile_home),
        "ZET_AGENT_OUTPUT_DIR": str(output),
    }

    local_module._prepare_managed_profile_runtime(env)
    command = f'python3 "{lexical_skill}"'
    local_module._prepare_managed_command_skill_sources(command, env)
    uid, gid = local_module._managed_terminal_identity(env)

    assert stat.S_IMODE(hermes_root.stat().st_mode) == 0o711
    assert stat.S_IMODE(profiles_root.stat().st_mode) == 0o711
    assert profile_home.stat().st_gid == gid
    assert stat.S_IMODE(profile_home.stat().st_mode) == 0o710
    assert skills_root.stat().st_gid == gid
    assert stat.S_IMODE(skills_root.stat().st_mode) == 0o750
    assert private_skill_dir.stat().st_gid == gid
    assert stat.S_IMODE(private_skill_dir.stat().st_mode) == 0o750
    assert private_skill.stat().st_gid == gid
    assert stat.S_IMODE(private_skill.stat().st_mode) == 0o640

    os.chmod(private_skill, 0o600)
    local_module._prepare_managed_command_skill_sources(command, env)
    assert private_skill.stat().st_gid == gid
    assert stat.S_IMODE(private_skill.stat().st_mode) == 0o640
    assert output.stat().st_uid == uid
    assert output.stat().st_gid == gid
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert os.lstat(output_link).st_uid == 0
    assert outside_file.stat().st_uid == 0
    assert stat.S_IMODE(outside_file.stat().st_mode) == 0o600
    assert outside_file.read_text() == "protected"
    assert stat.S_IMODE(sibling_home.stat().st_mode) == 0o700
    assert local_module._managed_identity_can_traverse(
        str(skills_root), uid=uid, gid=gid
    )
    assert not local_module._managed_identity_can_traverse(
        str(sibling_home), uid=uid, gid=gid
    )

    os.chmod(private_skill, 0o622)
    local_module._prepare_managed_command_skill_sources(command, env)
    assert stat.S_IMODE(private_skill.stat().st_mode) == 0o640

    state_dir = output / "support-suite-state"
    state_file = state_dir / "onboarding.json"
    state_dir.mkdir()
    state_file.write_text("{}")
    os.chown(state_dir, uid, gid)
    os.chown(state_file, uid, gid)
    os.chmod(state_dir, 0o700)
    os.chmod(state_file, 0o600)
    monkeypatch.setenv("ZET_AGENT_KEY", "rotated-device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()

    local_module._prepare_managed_profile_runtime(env)
    rotated_uid, rotated_gid = local_module._managed_terminal_identity(env)

    assert rotated_uid == uid
    assert rotated_gid == gid
    assert state_dir.stat().st_uid == uid
    assert state_file.stat().st_uid == uid
    assert state_file.stat().st_gid == gid


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="requires root POSIX ownership semantics",
)
def test_managed_profile_runtime_restores_non_python_skill_packages(
    monkeypatch, request
):
    tmp_path = Path(tempfile.mkdtemp(prefix="hermes-skilltree-test-", dir="/run"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    os.chmod(tmp_path, 0o755)
    hermes_root = tmp_path / "hermes_home"
    profiles_root = hermes_root / "profiles"
    profile_home = profiles_root / "agent-a"
    skills_root = profile_home / "skills"
    output = tmp_path / "agents" / "data" / "agent-a" / "output"
    shell_pkg = skills_root / "shell-suite"
    shell_script = shell_pkg / "run.sh"
    shell_doc = shell_pkg / "SKILL.md"
    bad_pkg = skills_root / "bad-suite"
    bad_file = bad_pkg / "keep.txt"
    shell_pkg.mkdir(parents=True)
    bad_pkg.mkdir()
    output.mkdir(parents=True)
    shell_script.write_text("echo ok")
    shell_doc.write_text("# doc")
    bad_file.write_text("keep")
    for path in (hermes_root, profiles_root, profile_home, skills_root, shell_pkg):
        os.chmod(path, 0o700)
    os.chmod(shell_script, 0o600)
    os.chmod(shell_doc, 0o600)
    # 非 root 属主的包会被 normalize 拒绝，用来验证坏包不连累别人
    os.chown(bad_pkg, 1001, 1001)
    os.chmod(output, 0o755)

    monkeypatch.setattr(local_module, "_IS_WINDOWS", False)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()
    env = {
        "HERMES_HOME": str(profile_home),
        "ZET_AGENT_OUTPUT_DIR": str(output),
    }

    # 不给命令解析任何 python <脚本> 线索：bash/cat 等形态也要拿到放权
    local_module._prepare_managed_profile_runtime(env)
    uid, gid = local_module._managed_terminal_identity(env)

    assert shell_pkg.stat().st_gid == gid
    assert stat.S_IMODE(shell_pkg.stat().st_mode) == 0o750
    assert shell_script.stat().st_gid == gid
    assert stat.S_IMODE(shell_script.stat().st_mode) == 0o640
    assert shell_doc.stat().st_gid == gid
    assert stat.S_IMODE(shell_doc.stat().st_mode) == 0o640
    assert bad_pkg.stat().st_uid == 1001
    assert bad_file.stat().st_gid != gid
    assert output.stat().st_uid == uid
    assert stat.S_IMODE(output.stat().st_mode) == 0o700


def test_managed_skill_tree_cache_skips_repeat_walks(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    (skills_root / "pkg-a").mkdir(parents=True)
    (skills_root / "pkg-b").mkdir()
    calls = []
    monkeypatch.setattr(
        local_module,
        "_normalize_managed_skill_package",
        lambda package, gid: calls.append((package.name, gid)),
    )
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()

    local_module._prepare_managed_skill_tree(skills_root, 100001)
    assert sorted(name for name, _gid in calls) == ["pkg-a", "pkg-b"]

    local_module._prepare_managed_skill_tree(skills_root, 100001)
    assert len(calls) == 2

    # 指纹变化（新装/卸载技能改动 skills 根目录）→ 重新全量放权。
    # +1s 而不是 +1ns：NTFS 时间戳粒度 100ns，会把 +1ns 截没
    info = os.stat(skills_root)
    os.utime(
        skills_root, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000)
    )
    local_module._prepare_managed_skill_tree(skills_root, 100001)
    assert len(calls) == 4

    # 身份轮换后旧的放权结果不可信
    local_module._prepare_managed_skill_tree(skills_root, 100002)
    assert len(calls) == 6
    assert calls[-1][1] == 100002


def test_managed_skill_tree_cache_notices_in_package_updates(
    monkeypatch, tmp_path
):
    """技能热更新通常只落在包内子目录，顶层包的时间戳不会动。"""

    skills_root = tmp_path / "skills"
    scripts = skills_root / "pkg-a" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run.sh").write_text("echo old\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        local_module,
        "_normalize_managed_skill_package",
        lambda package, gid: calls.append((package.name, gid)),
    )
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()

    local_module._prepare_managed_skill_tree(skills_root, 100001)
    assert len(calls) == 1
    package_mtime = os.stat(skills_root / "pkg-a").st_mtime_ns

    (scripts / "extra.py").write_text("print(1)\n", encoding="utf-8")
    # +1s 而不是 +1ns：NTFS 时间戳粒度 100ns，会把 +1ns 截没
    info = os.stat(scripts)
    os.utime(scripts, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    # 顶层包自己没被动过——只按包判定的旧指纹正是在这里恒命中缓存
    assert os.stat(skills_root / "pkg-a").st_mtime_ns == package_mtime

    local_module._prepare_managed_skill_tree(skills_root, 100001)
    assert len(calls) == 2


def test_managed_skill_tree_oversized_tree_never_caches(monkeypatch, tmp_path):
    """指纹装不下就不缓存：每次重新放权很慢，但不会静默停止放权。"""

    skills_root = tmp_path / "skills"
    (skills_root / "pkg-a" / "scripts").mkdir(parents=True)
    monkeypatch.setattr(local_module, "_MANAGED_SKILL_TREE_MAX_ENTRIES", 1)
    calls = []
    monkeypatch.setattr(
        local_module,
        "_normalize_managed_skill_package",
        lambda package, gid: calls.append((package.name, gid)),
    )
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()

    local_module._prepare_managed_skill_tree(skills_root, 100001)
    local_module._prepare_managed_skill_tree(skills_root, 100001)

    assert len(calls) == 2
    assert str(skills_root) not in local_module._MANAGED_SKILL_TREE_PREPARED


def test_managed_skill_tree_bad_package_does_not_block_others(
    monkeypatch, tmp_path
):
    skills_root = tmp_path / "skills"
    (skills_root / "bad-suite").mkdir(parents=True)
    (skills_root / "good-suite").mkdir()
    normalized = []

    def fake_normalize(package, gid):
        if package.name == "bad-suite":
            raise OSError("managed profile skill entry is not trusted")
        normalized.append(package.name)

    monkeypatch.setattr(
        local_module, "_normalize_managed_skill_package", fake_normalize
    )
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()

    local_module._prepare_managed_skill_tree(skills_root, 100001)

    assert normalized == ["good-suite"]


@pytest.mark.skipif(
    os.name == "nt"
    or not hasattr(os, "geteuid")
    or os.geteuid() != 0
    or not Path("/usr/bin/setpriv").is_file()
    or not Path("/usr/bin/unshare").is_file(),
    reason="requires the production root/Linux namespace boundary",
)
def test_managed_terminal_reads_but_cannot_modify_skill_and_writes_output(
    monkeypatch,
):
    root = Path(tempfile.mkdtemp(prefix="hermes-managed-test-", dir="/run"))
    try:
        os.chmod(root, 0o755)
        hermes_root = root / "hermes_home"
        profile_home = hermes_root / "profiles" / "agent-a"
        script = profile_home / "skills" / "support-suite" / "scripts" / "onboard.py"
        output = root / "agents" / "data" / "agent-a" / "output"
        script.parent.mkdir(parents=True)
        output.mkdir(parents=True)
        legacy_output = output / "legacy-owner.txt"
        legacy_output.write_text("historical")
        os.chown(legacy_output, 1001, 1001)
        original = (
            "from pathlib import Path\n"
            "source = Path(__file__)\n"
            "try:\n"
            "    source.write_text('tampered')\n"
            "except OSError:\n"
            "    pass\n"
            "else:\n"
            "    raise SystemExit('skill source was writable')\n"
            "Path(__import__('os').environ['ZET_AGENT_OUTPUT_DIR'], "
            "'state.txt').write_text('ok')\n"
        )
        script.write_text(original)
        for path in (
            hermes_root,
            profile_home.parent,
            profile_home,
            profile_home / "skills",
            script.parent.parent,
            script.parent,
        ):
            os.chmod(path, 0o700)
        os.chmod(script, 0o600)
        os.chmod(output, 0o755)
        monkeypatch.setattr(local_module, "_IS_WINDOWS", False)
        monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
        monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
        local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
        env = os.environ.copy()
        env.update(
            {
                "HERMES_HOME": str(profile_home),
                "ZET_AGENT_OUTPUT_DIR": str(output),
            }
        )
        local_module._prepare_managed_profile_runtime(env)
        local_module._prepare_managed_command_skill_sources(
            f'python3 "{script}"', env
        )
        uid, gid = local_module._managed_terminal_identity(env)
        private_tmp = root / "private-tmp"
        private_var_tmp = root / "private-var-tmp"
        private_tmp.mkdir()
        private_var_tmp.mkdir()
        for path in (private_tmp, private_var_tmp):
            os.chown(path, uid, gid)
            os.chmod(path, 0o700)

        argv = [
            "/usr/bin/setpriv",
            f"--reuid={uid}",
            f"--regid={gid}",
            "--clear-groups",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--no-new-privs",
            "--",
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
            str(private_tmp),
            str(private_var_tmp),
            "/usr/bin/python3",
            str(script),
        ]
        result = subprocess.run(
            argv,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert script.read_text() == original
        assert (output / "state.txt").read_text() == "ok"
        assert legacy_output.read_text() == "historical"
        assert legacy_output.stat().st_uid == uid
        assert legacy_output.stat().st_gid == gid
    finally:
        shutil.rmtree(root)


def test_managed_terminal_mounts_skill_source_read_only():
    helper = local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER

    compile(helper, "<managed-private-tmp-enter>", "exec")
    assert "skill_root=os.path.join(hermes_home,'skills')" in helper
    assert "mount(encoded,encoded,4096|16384)" in helper
    # MS_REMOUNT 不递归：子挂载必须经 mountinfo 收集后从深到浅逐点补只读
    assert "open('/proc/self/mountinfo','rb')" in helper
    assert "point==skill_root or point.startswith(skill_root+'/')" in helper
    assert "sorted(points,key=len,reverse=True)" in helper
    assert "mount(None,os.fsencode(point),32|4096|1|2|4)" in helper


def test_managed_terminal_skill_mount_requires_absolute_hermes_home():
    helper = local_module._MANAGED_TERMINAL_PRIVATE_TMP_ENTER

    # HERMES_HOME 为空/相对路径时不得把子进程 cwd 下的同名目录挂成只读
    assert "hermes_home=os.environ.get('HERMES_HOME','')" in helper
    assert (
        "if os.path.isabs(hermes_home) and os.path.isdir(skill_root):" in helper
    )
    assert "if skill_root and os.path.isdir(skill_root):" not in helper


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
        lambda argv, *, env, execution_scope, workspace: [
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
        workspace="/tmp/hermes-execute",
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
        local_module, "_trusted_managed_unshare", lambda: "/usr/bin/unshare"
    )
    monkeypatch.setattr(
        local_module,
        "_managed_execute_code_private_tmp_paths",
        lambda _workspace, _uid: (
            Path("/run/zettlab-claw/execute-code/61001"),
            Path("/run/zettlab-claw/execute-code/61001/var-tmp"),
        ),
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
        workspace="/run/zettlab-claw/execute-code/61001",
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
    assert argv[14:25] == [
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--",
        "/usr/bin/python3",
        "-I",
        "-c",
        local_module._MANAGED_EXECUTE_CODE_PRIVATE_TMP_ENTER,
        "/run/zettlab-claw/execute-code/61001",
        "/run/zettlab-claw/execute-code/61001/var-tmp",
    ]
    assert argv[25:] == ["/usr/bin/python3", "/tmp/script.py"]
    assert "--fork" not in argv
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
        "HERMES_MANAGED_GATEWAY": "1",
        "HERMES_MANAGED_CGROUP_UNIT": "zettlab-claw.service",
        "HERMES_MANAGED_CGROUP_ROOT": "/system.slice/zettlab-claw.service",
    })
    assert sanitized["PATH"] == "/usr/bin"
    assert "ZET_AGENT_KEY" not in sanitized
    assert "HERMES_MANAGED_GATEWAY" not in sanitized
    assert "HERMES_MANAGED_CGROUP_UNIT" not in sanitized
    assert "HERMES_MANAGED_CGROUP_ROOT" not in sanitized


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="requires root POSIX ownership semantics",
)
def test_managed_profile_runtime_migrates_previous_identity_output(
    monkeypatch, request
):
    """升级板的 output 属于上一版算法派生的旧 UID，必须走迁移而不是被判不可信。

    _migrate_managed_output_tree 本就是为「历史 output 属于另一个数字 UID」写
    的；调用方只认 (0, uid) 会把它挡在门外，agent 从此写不进自己的产出目录。
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="hermes-migrate-test-", dir="/run"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    os.chmod(tmp_path, 0o755)
    hermes_root = tmp_path / "hermes_home"
    profiles_root = hermes_root / "profiles"
    profile_home = profiles_root / "agent-a"
    output = tmp_path / "agents" / "data" / "agent-a" / "output"
    profile_home.mkdir(parents=True)
    output.mkdir(parents=True)
    carried = output / "report.md"
    carried.write_text("earlier run")
    for path in (hermes_root, profiles_root, profile_home):
        os.chmod(path, 0o700)

    # 上一版身份算法派生出来的旧 UID：在受管段内，但既不是 root 也不是新身份。
    stale_uid = local_module._MANAGED_TERMINAL_UID_MIN + 4242
    os.chown(carried, stale_uid, stale_uid)
    os.chown(output, stale_uid, stale_uid)
    os.chmod(output, 0o700)

    monkeypatch.setattr(local_module, "_IS_WINDOWS", False)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()
    env = {
        "HERMES_HOME": str(profile_home),
        "ZET_AGENT_OUTPUT_DIR": str(output),
    }

    local_module._prepare_managed_profile_runtime(env)
    uid, gid = local_module._managed_terminal_identity(env)

    assert uid != stale_uid
    assert output.stat().st_uid == uid
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    # 存量产出跟着换主，不是被删或跳过。
    assert carried.read_text() == "earlier run"
    assert carried.stat().st_uid == uid
    assert carried.stat().st_gid == gid


@pytest.mark.skipif(
    os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="requires root POSIX ownership semantics",
)
def test_managed_profile_runtime_grants_skills_without_output(monkeypatch, request):
    """output 是可选能力，skills 放权不是。

    平台没注入 ZET_AGENT_OUTPUT_DIR 时提前返回会连带跳过 skills 放权，非
    `python <abs>` 的技能入口拿不到 per-command 兜底，在设备上 permission denied。
    """
    tmp_path = Path(tempfile.mkdtemp(prefix="hermes-nooutput-test-", dir="/run"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    os.chmod(tmp_path, 0o755)
    hermes_root = tmp_path / "hermes_home"
    profiles_root = hermes_root / "profiles"
    profile_home = profiles_root / "agent-a"
    skills_root = profile_home / "skills"
    shell_pkg = skills_root / "shell-suite"
    shell_script = shell_pkg / "run.sh"
    shell_pkg.mkdir(parents=True)
    shell_script.write_text("echo ok")
    for path in (hermes_root, profiles_root, profile_home, skills_root, shell_pkg):
        os.chmod(path, 0o700)
    os.chmod(shell_script, 0o600)

    monkeypatch.setattr(local_module, "_IS_WINDOWS", False)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("ZET_AGENT_KEY", "device-key")
    local_module._MANAGED_TERMINAL_SCOPE_BY_UID.clear()
    local_module._MANAGED_SKILL_TREE_PREPARED.clear()
    env = {"HERMES_HOME": str(profile_home)}  # 平台未注入 output

    local_module._prepare_managed_profile_runtime(env)
    _uid, gid = local_module._managed_terminal_identity(env)

    assert stat.S_IMODE(hermes_root.stat().st_mode) == 0o711
    assert stat.S_IMODE(profiles_root.stat().st_mode) == 0o711
    assert profile_home.stat().st_gid == gid
    assert skills_root.stat().st_gid == gid
    assert stat.S_IMODE(skills_root.stat().st_mode) == 0o750
    assert shell_pkg.stat().st_gid == gid
    assert shell_script.stat().st_gid == gid
    assert stat.S_IMODE(shell_script.stat().st_mode) == 0o640
