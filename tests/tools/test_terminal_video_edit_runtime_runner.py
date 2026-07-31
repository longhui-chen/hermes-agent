import ast
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from gateway.session_context import clear_turn_vars, set_turn_vars
from tools import terminal_tool as terminal_tool_module
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _reset_runtime_anchor(monkeypatch):
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)


def _write_trusted_script(tmp_path, name="workflow_state.py"):
    script = (
        tmp_path
        / "presets"
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / name
    )
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(textwrap.dedent(
        """
        import os
        from _zettlab_video_runtime_context import get as runtime_value

        print("execution=" + runtime_value("ZETTLAB_BUSINESS_EXECUTION_TOKEN"))
        print("agent=" + os.environ.get("ZET_AGENT_ID", ""))
        print("turn=" + os.environ.get("HERMES_TURN_ID", ""))
        print("connector-token=" + os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN", ""))
        print("connector-url=" + os.environ.get("ZETTLAB_CONNECTORS_URL", ""))
        """
    ).lstrip())
    return script


def test_managed_runtime_import_keeps_gateway_exec_privilege(tmp_path):
    code = """
import json
import os
import sys

import tools.process_security as process_security

calls = []
process_security.harden_sensitive_process = (
    lambda **kwargs: calls.append(kwargs) or True
)
os.environ["ZETTLAB_PRESETS_DIR"] = sys.argv[1]
sys.platform = "linux"
import tools.terminal_tool
print(json.dumps(calls, sort_keys=True))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "missing-presets")],
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(completed.stdout) == [
        {"drop_ptrace": True, "no_new_privs": False}
    ]


def test_embedded_worker_tree_keeps_no_new_privs_boundary():
    tree = ast.parse(terminal_tool_module._VIDEO_EDIT_WORKER_MEMORY_BOOTSTRAP)
    hardening_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "harden_sensitive_process"
    ]

    assert hardening_calls
    for call in hardening_calls:
        no_new_privs = next(
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "no_new_privs"
        )
        assert isinstance(no_new_privs, ast.Constant)
        assert no_new_privs.value is True


def test_sensitive_runtime_boundary_keeps_gateway_exec_privilege(monkeypatch):
    calls = []
    monkeypatch.setattr(
        terminal_tool_module,
        "_SENSITIVE_PROCESS_OS_BOUNDARY",
        False,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_MODEL_DESCENDANT_PTRACE_BOUNDARY",
        False,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "harden_sensitive_process",
        lambda **kwargs: calls.append(kwargs) or True,
    )

    assert terminal_tool_module._ensure_sensitive_runtime_boundary() is True
    assert calls == [{"no_new_privs": False, "drop_ptrace": True}]


def test_generic_terminal_never_receives_business_execution_token(monkeypatch):
    monkeypatch.setenv("ZETTLAB_BUSINESS_EXECUTION_TOKEN", "stale-global-secret")
    tokens = set_turn_vars(turn_id="turn-1", business_execution_token="capability-secret")
    try:
        result = LocalEnvironment().execute(
            "printf '%s' \"$ZETTLAB_BUSINESS_EXECUTION_TOKEN\""
        )
    finally:
        clear_turn_vars(tokens)
    assert result["output"] == ""


def test_trusted_video_runner_receives_only_current_scoped_capability(monkeypatch, tmp_path):
    _write_trusted_script(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "connector-secret")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://connector.invalid/rpc")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    tokens = set_turn_vars(turn_id="turn-1", business_execution_token="capability-secret")
    try:
        result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        clear_turn_vars(tokens)
    assert result["video_edit_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "execution=[REDACTED]" in result["output"]
    assert "agent=agent-1" in result["output"]
    assert "turn=turn-1" in result["output"]
    assert "connector-token=" in result["output"]
    assert "connector-url=" in result["output"]
    assert "connector-secret" not in result["output"]
    assert "connector.invalid" not in result["output"]


def test_trusted_video_runner_can_import_sibling_from_same_pinned_tree(monkeypatch, tmp_path):
    script = _write_trusted_script(tmp_path, "preference_resolver.py")
    (script.parent / "workflow_state.py").write_text("VALUE = 'trusted-sibling'\n")
    script.write_text("from workflow_state import VALUE\nprint(VALUE)\n")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/scripts/preference_resolver.py"',
        cwd=str(tmp_path),
        timeout=5,
    ))
    assert result["exit_code"] == 0
    assert result["output"] == "trusted-sibling"


def test_video_worker_execute_registers_main_module_for_dataclasses(monkeypatch):
    from tools import process_security

    monkeypatch.setitem(sys.modules, "process_security", process_security)
    from tools import video_edit_runtime_worker as worker

    payload = _worker_payload(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            @dataclass(frozen=True)
            class Preference:
                scene: str

            print(Preference("daily").scene)
            """
        ).lstrip()
    )

    result = worker._execute(payload)

    assert result["returncode"] == 0
    assert result["stdout"] == "daily\n"
    assert result["stderr"] == ""


def test_video_worker_stdlib_closure_includes_secrets_unit(monkeypatch):
    from tools import process_security

    monkeypatch.setitem(sys.modules, "process_security", process_security)
    from tools import video_edit_runtime_worker as worker

    assert worker.secrets is sys.modules["secrets"]


def test_trusted_video_worker_runs_preference_secrets_flow():
    source = textwrap.dedent(
        """
        import secrets

        print(secrets.token_hex(16))
        """
    ).lstrip()
    terminal_tool_module._stop_video_edit_worker()
    try:
        result = terminal_tool_module._run_video_edit_worker(
            _worker_payload(source),
            timeout=5,
        )
    finally:
        terminal_tool_module._stop_video_edit_worker()

    token = result["stdout"].strip()
    assert result["returncode"] == 0
    assert result["stderr"] == ""
    assert len(token) == 32
    assert int(token, 16) >= 0


def test_trusted_video_worker_runs_regex_and_dataclass_startup_flow():
    source = textwrap.dedent(
        r"""
        from dataclasses import dataclass
        import re

        SCENE_RE = re.compile(
            r"^(?:[a-z][a-z0-9_]{0,63}|custom:[a-z][a-z0-9_]{0,63})$"
        )

        @dataclass(frozen=True)
        class Preference:
            scene: str

        preference = Preference("daily")
        print(SCENE_RE.fullmatch(preference.scene) is not None)
        """
    ).lstrip()
    terminal_tool_module._stop_video_edit_worker()
    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS is None
    try:
        result = terminal_tool_module._run_video_edit_worker(
            _worker_payload(source),
            timeout=5,
        )
    finally:
        terminal_tool_module._stop_video_edit_worker()

    assert result["returncode"] == 0
    assert result["stdout"] == "True\n"
    assert result["stderr"] == ""


def test_trusted_video_runner_keeps_capability_out_of_wrapper_process_env(monkeypatch, tmp_path):
    _write_trusted_script(tmp_path, "cloud_render_business.py")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "connector-secret")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://connector.invalid/rpc")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    captured = {}

    def fake_run(payload, *, timeout):
        captured.update(payload)
        captured["timeout"] = timeout
        return {"stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(terminal_tool_module, "_run_video_edit_worker", fake_run)
    tokens = set_turn_vars(turn_id="turn-1", business_execution_token="capability-secret")
    try:
        result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/scripts/cloud_render_business.py"',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        clear_turn_vars(tokens)
    assert result["video_edit_runtime_direct"] is True
    assert "ZETTLAB_BUSINESS_EXECUTION_TOKEN" not in captured["env"]
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in captured["env"]
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["env"]
    assert "ZETTLAB_CONNECTORS_URL" not in captured["env"]
    assert captured["secrets"]["ZETTLAB_BUSINESS_EXECUTION_TOKEN"] == "capability-secret"
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["secrets"]
    assert "ZETTLAB_CONNECTORS_URL" not in captured["secrets"]
    assert "pythonpath" not in captured
    assert captured["source_bundle"]["__main__"]["path"].endswith(
        "cloud_render_business.py"
    )


def test_trusted_video_upload_uses_dedicated_long_timeout_flow(monkeypatch, tmp_path):
    _write_trusted_script(tmp_path, "cloud_render_business.py")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    captured = {}

    def fake_run(payload, *, timeout):
        captured["argv"] = payload["argv"]
        captured["timeout"] = timeout
        return {"stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(terminal_tool_module, "_run_video_edit_worker", fake_run)
    result = json.loads(
        terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/'
            'scripts/cloud_render_business.py" --agent-id agent-1 '
            '--timeout=1800 upload',
            cwd=str(tmp_path),
            timeout=600,
        )
    )

    assert result["exit_code"] == 0, result
    assert captured["argv"][-1] == "upload"
    assert (
        captured["timeout"] == terminal_tool_module._VIDEO_EDIT_UPLOAD_TIMEOUT_SECONDS
    )


def test_trusted_video_upload_timeout_parser_rejects_option_value_named_upload():
    parsed = terminal_tool_module._VideoEditRuntimeCommand(
        argv=[
            sys.executable,
            "/trusted/cloud_render_business.py",
            "--agent-id",
            "upload",
            "assets-batch",
        ],
        root_identity=(1, 2),
        script_identity=(3, 4),
    )

    assert terminal_tool_module._video_edit_runtime_timeout(parsed, 600) == 600


def test_trusted_video_non_upload_keeps_requested_timeout():
    parsed = terminal_tool_module._VideoEditRuntimeCommand(
        argv=[
            sys.executable,
            "/trusted/cloud_render_business.py",
            "assets-batch",
        ],
        root_identity=(1, 2),
        script_identity=(3, 4),
    )

    assert terminal_tool_module._video_edit_runtime_timeout(parsed, 600) == 600


def test_trusted_video_runner_dependency_isolation_flow_ignores_writable_argparse(
    monkeypatch, tmp_path
):
    script = _write_trusted_script(tmp_path, "workflow_state.py")
    attacker_dir = tmp_path / "attacker"
    attacker_dir.mkdir()
    marker = tmp_path / "stolen-token.txt"
    (attacker_dir / "argparse.py").write_text(
        "from _zettlab_video_runtime_context import get\n"
        f"open({str(marker)!r}, 'w').write(get('ZETTLAB_BUSINESS_EXECUTION_TOKEN'))\n"
        "ATTACKER = True\n"
    )
    script.write_text(
        "import argparse\n"
        "print('stdlib=' + str(getattr(argparse, 'ATTACKER', False)))\n"
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("PYTHONPATH", str(attacker_dir))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    tokens = set_turn_vars(turn_id="turn-1", business_execution_token="capability-secret")
    try:
        result = json.loads(
            terminal_tool_module._run_video_edit_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/'
                'video-edit-workflow-mini/scripts/workflow_state.py"',
                cwd=str(attacker_dir),
                timeout=5,
            )
        )
    finally:
        clear_turn_vars(tokens)

    assert result["exit_code"] == 0
    assert result["output"] == "stdlib=False"
    assert not marker.exists()


def _worker_payload(source: str) -> dict:
    return {
        "script": "/trusted/workflow_state.py",
        "argv": ["/trusted/workflow_state.py"],
        "env": {},
        "secrets": {},
        "cwd": "",
        "source_bundle": {
            "__main__": {
                "path": "/trusted/workflow_state.py",
                "source": source,
            }
        },
    }


def _setsid_double_fork_source(
    pid_marker: Path,
    leak_marker: Path,
    *,
    parent_sleep: bool,
) -> str:
    tail = "time.sleep(30)" if parent_sleep else "print(escaped_pid)"
    return textwrap.dedent(
        f"""
        import os
        import pathlib
        import time

        first = os.fork()
        if first == 0:
            os.setsid()
            second = os.fork()
            if second > 0:
                os._exit(0)
            pathlib.Path({str(pid_marker)!r}).write_text(str(os.getpid()))
            time.sleep(4.0)
            pathlib.Path({str(leak_marker)!r}).write_text(
                os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN", "missing")
            )
            os._exit(0)

        os.waitpid(first, 0)
        deadline = time.monotonic() + 5
        while not pathlib.Path({str(pid_marker)!r}).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not pathlib.Path({str(pid_marker)!r}).exists():
            raise RuntimeError("double-fork descendant did not start")
        escaped_pid = int(pathlib.Path({str(pid_marker)!r}).read_text())
        {tail}
        """
    )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.02)
    return not _pid_exists(pid)


def _install_temp_worker_sources(monkeypatch, tmp_path) -> dict[str, Path]:
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    copies: dict[str, Path] = {}
    for name, source in terminal_tool_module._video_edit_worker_source_paths():
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        copies[name] = target
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_source_paths",
        lambda: tuple(copies.items()),
    )
    return copies


def test_video_worker_seed_restart_uses_only_retained_snapshot_unit(monkeypatch):
    snapshot = terminal_tool_module._TrustedWorkerSourceSnapshot(
        modules=(),
        python_executable="/trusted/python",
        python_fingerprint=(),
        worker_path="/trusted/video_edit_runtime_worker.py",
    )

    def fail_disk_capture():
        raise AssertionError("post-terminal seed rebuild must not read worker source")

    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT",
        snapshot,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED",
        True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_capture_trusted_video_edit_worker_snapshot",
        fail_disk_capture,
    )

    assert (
        terminal_tool_module._trusted_video_edit_worker_snapshot_for_seed_start()
        is snapshot
    )

    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT",
        None,
    )
    with pytest.raises(PermissionError, match="snapshot was not captured"):
        terminal_tool_module._trusted_video_edit_worker_snapshot_for_seed_start()


def test_video_worker_factory_seed_start_failure_cleans_both_bounded_attempts_unit(
    monkeypatch,
):
    supervisor = object()
    snapshot = object()
    start_attempts = []
    cleanup_attempts = []

    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR",
        supervisor,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT",
        snapshot,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_PROCESS",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_CHANNEL",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SEED_PROCESS",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_factory_is_ready",
        lambda: False,
    )

    def install_factory_candidate():
        start_attempts.append(len(start_attempts) + 1)
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS = object()
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_CHANNEL = object()

    def reject_seed_start():
        raise PermissionError("trusted video-edit worker factory unavailable")

    def validate_recovery_root(candidate_supervisor, candidate_snapshot):
        assert candidate_supervisor is supervisor
        assert candidate_snapshot is snapshot

    def cleanup_factory_candidate():
        cleanup_attempts.append(len(cleanup_attempts) + 1)
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS = None
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_CHANNEL = None

    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_video_edit_worker_factory_started",
        install_factory_candidate,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_spawn_video_edit_worker_seed_from_factory",
        reject_seed_start,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_validate_trusted_video_edit_worker_factory_supervisor",
        validate_recovery_root,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_shutdown_video_edit_worker_seed",
        cleanup_factory_candidate,
    )

    with pytest.raises(
        PermissionError,
        match="trusted video-edit worker factory unavailable",
    ):
        terminal_tool_module._ensure_video_edit_worker_seed_started()

    assert start_attempts == [1, 2]
    assert cleanup_attempts == [1, 2]


def test_terminal_tool_fresh_import_does_not_require_fcntl_flow(tmp_path):
    marker = tmp_path / "terminal-tool-imported"
    code = textwrap.dedent(
        f"""
        import builtins
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fcntl" and (globals or {{}}).get("__name__") == "tools.terminal_tool":
                raise ModuleNotFoundError("fcntl unavailable")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        import tools.terminal_tool
        open({str(marker)!r}, "w").write("ok")
        """
    )
    env = dict(os.environ)
    env.pop("ZETTLAB_PRESETS_DIR", None)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text() == "ok"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork guard")
def test_terminal_tool_managed_import_preforks_clean_supervisor_flow(
    tmp_path,
):
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    code = textwrap.dedent(
        f"""
        import os

        os.environ["ZETTLAB_PRESETS_DIR"] = {str(presets_dir)!r}

        from tools import terminal_tool

        assert terminal_tool._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT is not None
        assert terminal_tool._VIDEO_EDIT_WORKER_FACTORY_IMAGE is not None
        supervisor = terminal_tool._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        assert supervisor is not None
        assert supervisor.process.poll() is None
        assert terminal_tool._VIDEO_EDIT_WORKER_FACTORY_PROCESS is None
        assert terminal_tool._VIDEO_EDIT_WORKER_SEED_PROCESS is None
        terminal_tool._stop_video_edit_worker()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.live_system_guard_bypass
def test_video_worker_idle_recycle_keeps_clean_supervisor_flow(monkeypatch):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_IDLE_TIMEOUT_SECONDS",
        0.05,
    )

    result = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('idle-recycle')\n"),
        timeout=5,
    )
    supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

    assert result["stdout"].strip() == "idle-recycle"
    assert supervisor is not None
    supervisor_pid = supervisor.process.pid

    deadline = time.monotonic() + 5
    while (
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS is not None
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)

    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR is supervisor
    assert supervisor.process.pid == supervisor_pid
    assert supervisor.process.poll() is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER is None
    terminal_tool_module._stop_video_edit_worker()
    assert _wait_for_pid_exit(supervisor_pid)


@pytest.mark.live_system_guard_bypass
def test_video_worker_idle_recycle_keeps_profile_payloads_isolated_flow(monkeypatch):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_IDLE_TIMEOUT_SECONDS",
        0.05,
    )

    def run_for_profile(profile: str):
        payload = _worker_payload(
            "import os\n"
            "from _zettlab_video_runtime_context import get\n"
            "print(os.environ.get('HERMES_HOME', ''))\n"
            "print(os.environ.get('PROFILE_ONLY_VALUE', ''))\n"
            "print(get('ZETTLAB_BUSINESS_EXECUTION_TOKEN'))\n"
        )
        payload["env"] = {
            "HERMES_HOME": f"/profiles/{profile}",
            "PROFILE_ONLY_VALUE": profile,
        }
        payload["secrets"] = {
            "ZETTLAB_BUSINESS_EXECUTION_TOKEN": f"secret-{profile}",
        }
        result = terminal_tool_module._run_video_edit_worker(payload, timeout=5)
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        assert supervisor is not None
        return (
            result["stdout"],
            supervisor,
            terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_GENERATION,
        )

    first_output, first_supervisor, first_generation = run_for_profile("alpha")
    first_pid = first_supervisor.process.pid
    deadline = time.monotonic() + 5
    while (
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS is not None
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR is first_supervisor
    assert first_supervisor.process.poll() is None

    background_started = threading.Event()
    release_background = threading.Event()

    def hold_gateway_thread():
        background_started.set()
        release_background.wait(timeout=10)

    background = threading.Thread(target=hold_gateway_thread)
    background.start()
    assert background_started.wait(timeout=2)
    try:
        second_output, second_supervisor, second_generation = run_for_profile("beta")
        assert first_output.splitlines() == [
            "/profiles/alpha",
            "alpha",
            "secret-alpha",
        ]
        assert second_output.splitlines() == [
            "/profiles/beta",
            "beta",
            "secret-beta",
        ]
        assert "alpha" not in second_output
        assert second_supervisor is first_supervisor
        assert second_supervisor.process.pid == first_pid
        assert second_generation > first_generation
    finally:
        release_background.set()
        background.join(timeout=2)
        terminal_tool_module._stop_video_edit_worker()


def test_video_worker_rejects_first_supervisor_fork_after_threads_start_unit(
    monkeypatch,
):
    fork_calls = []
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR",
        None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_trusted_video_edit_worker_factory_image",
        lambda _snapshot: object(),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_process_thread_count",
        lambda: 2,
    )
    monkeypatch.setattr(
        terminal_tool_module.os,
        "fork",
        lambda: fork_calls.append(True) or 1,
    )

    with pytest.raises(PermissionError, match="before gateway threads start"):
        terminal_tool_module._trusted_video_edit_worker_factory_bootstrap(object())

    assert fork_calls == []


def test_video_worker_idle_scheduler_keeps_only_one_deadline_unit(monkeypatch):
    created = []

    class _Timer:
        def __init__(self, interval, function, args=()):
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.cancelled = False
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(terminal_tool_module.threading, "Timer", _Timer)
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR",
        object(),
    )
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_IDLE_TIMER", None)

    terminal_tool_module._schedule_video_edit_worker_idle_recycle()
    first = terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER
    terminal_tool_module._schedule_video_edit_worker_idle_recycle()
    second = terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER

    assert len(created) == 2
    assert first is created[0]
    assert first.cancelled is True
    assert second is created[1]
    assert second.started is True
    assert second.cancelled is False
    terminal_tool_module._cancel_video_edit_worker_idle_recycle()


def test_video_worker_timer_start_failure_preserves_supervisor_unit(monkeypatch):
    cleanup_calls = []

    class _BrokenTimer:
        daemon = False

        def __init__(self, _interval, _function, args=()):
            self.args = args
            self.cancelled = False

        def start(self):
            raise RuntimeError("thread limit")

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(terminal_tool_module.threading, "Timer", _BrokenTimer)
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR",
        object(),
    )
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_IDLE_TIMER", None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_terminate_video_edit_worker",
        lambda **_kwargs: cleanup_calls.append("broker"),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_shutdown_video_edit_worker_seed",
        lambda: cleanup_calls.append("seed_factory"),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_discard_trusted_video_edit_worker_factory_supervisor",
        lambda: cleanup_calls.append("supervisor"),
    )

    terminal_tool_module._schedule_video_edit_worker_idle_recycle()

    assert terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER is None
    assert cleanup_calls == ["broker", "seed_factory"]


@pytest.mark.parametrize(
    ("source", "timeout", "expected_returncode"),
    [
        ("raise RuntimeError('boom')\n", 5, 1),
        ("import time\ntime.sleep(2)\n", 0.05, 124),
    ],
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_error_and_timeout_paths_idle_recycle_flow(
    monkeypatch,
    source,
    timeout,
    expected_returncode,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_IDLE_TIMEOUT_SECONDS",
        0.05,
    )

    result = terminal_tool_module._run_video_edit_worker(
        _worker_payload(source),
        timeout=timeout,
    )
    supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

    assert result["returncode"] == expected_returncode
    assert supervisor is not None
    supervisor_pid = supervisor.process.pid
    deadline = time.monotonic() + 5
    while (
        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS is not None
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)

    assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR is supervisor
    assert supervisor.process.poll() is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER is None
    terminal_tool_module._stop_video_edit_worker()
    assert _wait_for_pid_exit(supervisor_pid)


@pytest.mark.live_system_guard_bypass
def test_video_worker_atexit_reaps_lazy_tree_flow():
    code = textwrap.dedent(
        """
        from tools import terminal_tool

        payload = {
            "script": "/trusted/workflow_state.py",
            "argv": ["/trusted/workflow_state.py"],
            "env": {},
            "secrets": {},
            "cwd": "",
            "source_bundle": {
                "__main__": {
                    "path": "/trusted/workflow_state.py",
                    "source": "print('atexit')\\n",
                }
            },
        }
        result = terminal_tool._run_video_edit_worker(payload, timeout=5)
        assert result["returncode"] == 0
        supervisor = terminal_tool._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        assert supervisor is not None
        print(supervisor.process.pid, flush=True)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    supervisor_pid = int(result.stdout.strip().splitlines()[-1])
    assert _wait_for_pid_exit(supervisor_pid)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux memfd seals the interpreter image",
)
def test_video_worker_interpreter_memfd_is_immutable_unit():
    snapshot = terminal_tool_module._capture_trusted_video_edit_worker_snapshot()
    executable, descriptor = (
        terminal_tool_module._sealed_trusted_video_edit_worker_interpreter(snapshot)
    )
    assert descriptor is not None
    try:
        assert executable == f"/proc/self/fd/{descriptor}"
        with pytest.raises(OSError):
            os.write(descriptor, b"tamper")
    finally:
        os.close(descriptor)


def test_video_worker_seed_control_protocol_never_sends_execution_payload(monkeypatch):
    terminal_tool_module._stop_video_edit_worker()
    sent: list[bytes] = []
    broker_channel, peer = socket.socketpair()
    broker_identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=100,
        pidfd=None,
    )

    class _SeedChannel:
        def sendall(self, data):
            sent.append(data)

    class _SeedProcess:
        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
        _SeedChannel(),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_SEED_PROCESS",
        _SeedProcess(),
    )
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_CHANNEL", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_PID", None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_fd_frame",
        lambda *_args, **_kwargs: (
            {"broker_ready": True, "broker_pid": 43210},
            broker_channel,
        ),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_capture_video_edit_worker_process_identity",
        lambda _pid: broker_identity,
    )
    try:
        terminal_tool_module._spawn_video_edit_worker_broker()
    finally:
        active_channel = terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL
        if active_channel is not None:
            active_channel.close()
        peer.close()

    assert sent == [b"S"]
    assert b"secret" not in b"".join(sent)


@pytest.mark.parametrize(
    "first_failure",
    [
        socket.timeout("stopped seed"),
        ValueError("malformed seed broker frame"),
    ],
)
def test_video_worker_broker_handshake_rebuilds_seed_once_unit(
    monkeypatch,
    first_failure,
):
    sent: list[str] = []
    discarded: list[str] = []
    rebuilt: list[bool] = []
    broker_channel, peer = socket.socketpair()
    broker_identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=100,
        pidfd=None,
    )
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_IDENTITY", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_PID", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_CHANNEL", None)

    class _SeedChannel:
        def __init__(self, name: str):
            self.name = name

        def sendall(self, data: bytes) -> None:
            assert data == b"S"
            sent.append(self.name)

    class _SeedProcess:
        @staticmethod
        def poll():
            return None

    def install_seed(name: str) -> None:
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
            _SeedChannel(name),
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_PROCESS",
            _SeedProcess(),
        )

    install_seed("initial")
    responses = [
        first_failure,
        ({"broker_ready": True, "broker_pid": 43210}, broker_channel),
    ]

    def receive(*_args, **_kwargs):
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def discard_seed() -> bool:
        channel = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_CHANNEL
        discarded.append(channel.name)
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
            None,
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_PROCESS",
            None,
        )
        return True

    def rebuild_seed() -> None:
        rebuilt.append(True)
        install_seed("rebuilt")

    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_fd_frame",
        receive,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_discard_video_edit_worker_seed",
        discard_seed,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_video_edit_worker_seed_started",
        rebuild_seed,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_capture_video_edit_worker_process_identity",
        lambda _pid: broker_identity,
    )

    try:
        terminal_tool_module._spawn_video_edit_worker_broker()
    finally:
        active_channel = terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL
        if active_channel is not None:
            active_channel.close()
        terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_IDENTITY = None
        terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID = None
        terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL = None
        peer.close()

    assert sent == ["initial", "rebuilt"]
    assert discarded == ["initial"]
    assert rebuilt == [True]
    assert responses == []


def test_video_worker_broker_handshake_recovery_is_bounded_unit(monkeypatch):
    sent: list[str] = []
    discarded: list[str] = []
    rebuilt: list[bool] = []
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_IDENTITY", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_PID", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_CHANNEL", None)

    class _SeedChannel:
        def __init__(self, name: str):
            self.name = name

        def sendall(self, data: bytes) -> None:
            assert data == b"S"
            sent.append(self.name)

    class _SeedProcess:
        @staticmethod
        def poll():
            return None

    def install_seed(name: str) -> None:
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
            _SeedChannel(name),
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_PROCESS",
            _SeedProcess(),
        )

    install_seed("initial")

    def discard_seed() -> bool:
        channel = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_CHANNEL
        discarded.append(channel.name)
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_CHANNEL",
            None,
        )
        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_SEED_PROCESS",
            None,
        )
        return True

    def rebuild_seed() -> None:
        rebuilt.append(True)
        install_seed("rebuilt")

    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_fd_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            socket.timeout("unresponsive seed")
        ),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_discard_video_edit_worker_seed",
        discard_seed,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_video_edit_worker_seed_started",
        rebuild_seed,
    )

    with pytest.raises(socket.timeout, match="unresponsive seed"):
        terminal_tool_module._spawn_video_edit_worker_broker()

    assert sent == ["initial", "rebuilt"]
    assert discarded == ["initial", "rebuilt"]
    assert rebuilt == [True]
    assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL is None


@pytest.mark.live_system_guard_bypass
def test_video_worker_seed_rejects_second_active_call_invariant(monkeypatch):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    try:
        terminal_tool_module._ensure_video_edit_worker_seed_started()
        terminal_tool_module._spawn_video_edit_worker_broker()
        first_pid = terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID
        seed_channel = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_CHANNEL
        assert first_pid is not None
        assert seed_channel is not None

        seed_channel.sendall(b"S")
        response, second_channel = terminal_tool_module._video_edit_worker_recv_fd_frame(
            seed_channel,
            timeout=5,
        )

        assert second_channel is None
        assert response["broker_ready"] is False
        assert "active call" in response["error"]
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID == first_pid
    finally:
        assert terminal_tool_module._terminate_video_edit_worker(
            close_disk_trust=False
        ) is True
        terminal_tool_module._stop_video_edit_worker()


def test_video_worker_fd_receive_uses_cloexec_and_rejects_truncation(monkeypatch):
    calls: list[int] = []

    class _Channel:
        @staticmethod
        def gettimeout():
            return None

        @staticmethod
        def settimeout(_timeout):
            return None

        @staticmethod
        def recvmsg(_size, _ancillary_size, flags):
            calls.append(flags)
            return b"F", [], getattr(socket, "MSG_CTRUNC", 8), None

    monkeypatch.setattr(
        terminal_tool_module.socket,
        "MSG_CTRUNC",
        getattr(socket, "MSG_CTRUNC", 8),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="truncated broker fd metadata"):
        terminal_tool_module._video_edit_worker_recv_fd_frame(_Channel(), timeout=1)
    assert calls == [getattr(socket, "MSG_CMSG_CLOEXEC", 0)]


def test_video_worker_fd_receive_closes_descriptor_when_frame_decode_fails(monkeypatch):
    descriptor = os.open(os.devnull, os.O_RDONLY)

    class _Channel:
        @staticmethod
        def gettimeout():
            return None

        @staticmethod
        def settimeout(_timeout):
            return None

        @staticmethod
        def recvmsg(_size, _ancillary_size, _flags):
            descriptors = terminal_tool_module.array.array("i", [descriptor])
            return (
                b"F",
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())],
                0,
                None,
            )

    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad frame")),
    )

    with pytest.raises(ValueError, match="bad frame"):
        terminal_tool_module._video_edit_worker_recv_fd_frame(_Channel(), timeout=1)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_video_worker_pidfd_rechecks_start_time_before_signal(monkeypatch):
    from tools import process_security

    monkeypatch.setitem(sys.modules, "process_security", process_security)
    from tools import video_edit_runtime_worker as worker

    record = worker._LinuxProcessRecord(
        pid=43210,
        ppid=123,
        start_time=100,
        rss_bytes=0,
    )
    sent: list[tuple[int, int]] = []
    closed: list[int] = []
    monkeypatch.setattr(worker.os, "pidfd_open", lambda _pid, _flags: 9, raising=False)
    monkeypatch.setattr(
        worker.signal,
        "pidfd_send_signal",
        lambda descriptor, signum, *_args: sent.append((descriptor, signum)),
        raising=False,
    )
    monkeypatch.setattr(worker.os, "sysconf", lambda _name: 4096)
    monkeypatch.setattr(
        worker,
        "_read_linux_process_record",
        lambda *_args, **_kwargs: worker._LinuxProcessRecord(
            pid=43210,
            ppid=1,
            start_time=101,
            rss_bytes=0,
        ),
    )
    monkeypatch.setattr(worker.os, "close", closed.append)

    assert worker._signal_linux_process(record, signal.SIGKILL) is False
    assert sent == []
    assert closed == [9]


def test_video_worker_gateway_fallback_rejects_reused_broker_pid_unit(monkeypatch):
    identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=100,
        pidfd=None,
    )
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        terminal_tool_module,
        "_read_video_edit_worker_process_start_time",
        lambda _pid: 101,
    )
    monkeypatch.setattr(
        terminal_tool_module.os,
        "killpg",
        lambda pid, signum: signals.append(("group", pid, signum)),
    )
    monkeypatch.setattr(
        terminal_tool_module.os,
        "kill",
        lambda pid, signum: signals.append(("pid", pid, signum)),
    )

    assert terminal_tool_module._force_kill_video_edit_worker_group(identity) is False
    assert signals == []


def test_video_worker_seed_handle_rejects_reused_pid_unit(monkeypatch):
    identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=100,
        pidfd=None,
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        terminal_tool_module,
        "_read_video_edit_worker_process_start_time",
        lambda _pid: 101,
    )
    monkeypatch.setattr(
        terminal_tool_module.os,
        "kill",
        lambda pid, signum: signals.append((pid, signum)),
    )
    process = terminal_tool_module._ForkedVideoEditWorkerSeed(
        pid=identity.pid,
        identity=identity,
    )

    process.kill()

    assert signals == []


def test_video_worker_gateway_fallback_without_kernel_identity_never_signals_unit(
    monkeypatch,
):
    identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=None,
        pidfd=None,
    )
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        terminal_tool_module.os,
        "kill",
        lambda pid, signum: signals.append(("pid", pid, signum)),
    )
    monkeypatch.setattr(
        terminal_tool_module.os,
        "killpg",
        lambda pid, signum: signals.append(("group", pid, signum)),
    )
    process = terminal_tool_module._ForkedVideoEditWorkerSeed(
        pid=identity.pid,
        identity=identity,
    )

    assert process.kill() is False
    assert terminal_tool_module._force_kill_video_edit_worker_group(identity) is False
    assert signals == []


def test_video_worker_owned_snapshot_ignores_reused_former_leader_pid(monkeypatch):
    from tools import process_security

    monkeypatch.setitem(sys.modules, "process_security", process_security)
    from tools import video_edit_runtime_worker as worker

    owner_pid = 100
    former_leader_pid = 200
    adopted_pid = 300
    nested_pid = 301
    table = {
        owner_pid: worker._LinuxProcessRecord(owner_pid, 1, 10, 0),
        # Same numeric PID as the reaped leader, now owned by an unrelated process.
        former_leader_pid: worker._LinuxProcessRecord(former_leader_pid, 999, 99, 0),
        adopted_pid: worker._LinuxProcessRecord(adopted_pid, owner_pid, 30, 0),
        nested_pid: worker._LinuxProcessRecord(nested_pid, adopted_pid, 31, 0),
    }
    monkeypatch.setattr(
        worker,
        "_read_linux_process_table",
        lambda: (table, True),
    )

    owned, complete = worker._owned_process_snapshot(owner_pid)

    assert complete is True
    assert set(owned) == {adopted_pid, nested_pid}
    assert former_leader_pid not in owned


@pytest.mark.live_system_guard_bypass
def test_video_worker_rebuilds_from_fork_seed_after_source_paths_change_flow(
    monkeypatch,
    tmp_path,
):
    copies = _install_temp_worker_sources(monkeypatch, tmp_path)
    marker = tmp_path / "tampered-source-loaded"
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('before-crash')\n"),
            timeout=5,
        )
        seed_pid = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS.pid
        first_broker = first["worker"]["pid"]
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
        terminal_tool_module._close_video_edit_worker_disk_trust()

        process_security_copy = copies["process_security"]
        original_inode = process_security_copy.stat().st_ino
        process_security_copy.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('same-inode')\n"
        )
        assert process_security_copy.stat().st_ino == original_inode

        worker_copy = copies["video_edit_runtime_worker"]
        replacement = tmp_path / "replacement-worker.py"
        replacement.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('replacement')\n"
        )
        replaced_inode = worker_copy.stat().st_ino
        os.replace(replacement, worker_copy)
        assert worker_copy.stat().st_ino != replaced_inode

        rebuilt = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-crash')\n"),
            timeout=5,
        )
        second_broker = rebuilt["worker"]["pid"]

        assert first["stdout"].strip() == "before-crash"
        assert rebuilt["stdout"].strip() == "after-crash"
        assert terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS.pid == seed_pid
        assert second_broker != first_broker
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
        assert not marker.exists()
        assert _wait_for_pid_exit(first_broker)
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.live_system_guard_bypass
def test_video_worker_seed_reaps_repeated_inflight_broker_crashes_without_orphans(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    try:
        snapshot = (
            terminal_tool_module._trusted_video_edit_worker_snapshot_for_seed_start()
        )
        terminal_tool_module._trusted_video_edit_worker_factory_bootstrap(snapshot)
        seed_pid = 0
        broker_pids: list[int] = []
        for index in range(2):
            marker = tmp_path / f"inflight-{index}.txt"
            child_source = "import time; time.sleep(30)"
            source = textwrap.dedent(
                f"""
                import os
                import pathlib
                import subprocess
                import sys
                import time

                child = subprocess.Popen([sys.executable, "-I", "-S", "-c", {child_source!r}])
                pathlib.Path({str(marker)!r}).write_text(f"{{os.getpid()}} {{child.pid}}")
                time.sleep(30)
                """
            )
            results: list[dict] = []
            errors: list[BaseException] = []

            def run_worker() -> None:
                try:
                    results.append(
                        terminal_tool_module._run_video_edit_worker(
                            _worker_payload(source),
                            timeout=10,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001 - assert thread failures
                    errors.append(exc)

            thread = threading.Thread(target=run_worker)
            thread.start()
            deadline = time.monotonic() + 5
            marker_parts: list[str] = []
            while time.monotonic() < deadline:
                if marker.exists():
                    marker_parts = marker.read_text().split()
                    if len(marker_parts) == 2:
                        break
                time.sleep(0.01)
            assert len(marker_parts) == 2
            broker_pid, descendant_pid = map(int, marker_parts)
            assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID == broker_pid
            broker_pids.append(broker_pid)
            os.kill(broker_pid, signal.SIGKILL)
            thread.join(timeout=8)
            assert not thread.is_alive()
            assert not errors
            assert results[0]["returncode"] == 1
            assert results[0]["worker"]["reaped"] is True
            assert _wait_for_pid_exit(descendant_pid)
            current_seed_pid = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS.pid
            if seed_pid:
                assert current_seed_pid == seed_pid
            seed_pid = current_seed_pid

            recovered = terminal_tool_module._run_video_edit_worker(
                _worker_payload(f"print('rebuild-{index}')\n"), timeout=5
            )
            assert recovered["stdout"].strip() == f"rebuild-{index}"
            assert recovered["worker"]["reaped"] is True
            assert terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS.pid == seed_pid

        assert len(set(broker_pids)) == 2
        assert all(_wait_for_pid_exit(pid) for pid in broker_pids)
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux subreaper owns escaped daemon descendants",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_broker_sigkill_reaps_setsid_double_fork_and_recovers_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    pid_marker = tmp_path / "broker-kill-daemon.pid"
    leak_marker = tmp_path / "broker-kill-daemon-token"
    payload = _worker_payload(
        _setsid_double_fork_source(pid_marker, leak_marker, parent_sleep=True)
    )
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "broker-kill-secret"}
    results: list[dict] = []
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            results.append(terminal_tool_module._run_video_edit_worker(payload, timeout=10))
        except BaseException as exc:  # noqa: BLE001 - assert thread failures
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_worker)
        thread.start()
        deadline = time.monotonic() + 5
        while not pid_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_marker.exists()
        escaped_pid = int(pid_marker.read_text())
        broker_pid = terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID
        assert broker_pid is not None

        os.kill(broker_pid, signal.SIGKILL)
        thread.join(timeout=8)

        assert not thread.is_alive()
        assert not errors
        assert results[0]["returncode"] == 1
        assert results[0]["worker"]["reaped"] is True
        assert _wait_for_pid_exit(escaped_pid)
        time.sleep(0.1)
        assert not leak_marker.exists()
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-broker-kill-daemon')\n"), timeout=5
        )
        assert recovered["stdout"].strip() == "after-broker-kill-daemon"
        assert recovered["worker"]["reaped"] is True
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux verifies stopped seed recovery through the resident factory",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_stopped_seed_rebuilds_from_resident_factory_flow(
    monkeypatch,
    tmp_path,
):
    copies = _install_temp_worker_sources(monkeypatch, tmp_path)
    disk_marker = tmp_path / "post-terminal-stopped-seed-source-loaded"
    popen_calls: list[tuple] = []
    old_seed_pid = 0
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('before-seed-stop')\n"),
            timeout=5,
        )
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        retained_snapshot = terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT
        retained_image = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_IMAGE
        assert first["stdout"].strip() == "before-seed-stop"
        assert factory is not None
        assert supervisor is not None
        assert seed is not None
        assert retained_snapshot is not None
        assert retained_image is not None
        factory_pid = factory.pid
        supervisor_pid = supervisor.process.pid
        old_seed_pid = seed.pid

        terminal_tool_module._close_video_edit_worker_disk_trust()
        for name, source in copies.items():
            source.write_text(
                "from pathlib import Path\n"
                f"Path({str(disk_marker)!r}).write_text({name!r})\n"
            )
        stopped = LocalEnvironment(cwd=str(tmp_path)).execute(
            f"kill -STOP -- {old_seed_pid}",
            timeout=5,
        )
        assert stopped["returncode"] == 0

        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS",
            0.5,
        )

        def reject_disk_capture():
            disk_marker.write_text("disk-capture")
            raise AssertionError("stopped seed recovery must use retained source")

        def reject_post_trust_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            disk_marker.write_text("fresh-exec")
            raise AssertionError("stopped seed recovery must not exec from disk")

        monkeypatch.setattr(
            terminal_tool_module,
            "_capture_trusted_video_edit_worker_snapshot",
            reject_disk_capture,
        )
        monkeypatch.setattr(
            terminal_tool_module.subprocess,
            "Popen",
            reject_post_trust_exec,
        )

        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-seed-stop')\n"),
            timeout=5,
        )
        rebuilt_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        rebuilt_supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS

        assert recovered["stdout"].strip() == "after-seed-stop"
        assert recovered["worker"]["reaped"] is True
        assert rebuilt_factory is not None
        assert rebuilt_factory.pid == factory_pid
        assert rebuilt_supervisor is supervisor
        assert rebuilt_supervisor.process.pid == supervisor_pid
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != old_seed_pid
        assert _wait_for_pid_exit(old_seed_pid)
        assert terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT is retained_snapshot
        assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_IMAGE is retained_image
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
        assert terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL is None
        assert popen_calls == []
        assert not disk_marker.exists()
    finally:
        if old_seed_pid and _pid_exists(old_seed_pid):
            try:
                os.kill(old_seed_pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.live_system_guard_bypass
def test_video_runtime_seed_loss_rebuilds_from_memory_snapshots_flow(
    monkeypatch,
    tmp_path,
):
    copies = _install_temp_worker_sources(monkeypatch, tmp_path)
    script = _write_trusted_script(tmp_path)
    script.write_text("print('trusted-memory-source')\n")
    marker = tmp_path / "post-terminal-disk-source-loaded"
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    try:
        first_raw = terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/'
            'video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
        assert first_raw is not None
        first = json.loads(first_raw)
        retained_snapshot = terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert retained_snapshot is not None
        assert seed is not None
        old_seed_pid = seed.pid

        terminal_tool_module._close_video_edit_worker_disk_trust()
        script.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('script')\n"
        )
        for name, source in copies.items():
            source.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text({name!r})\n"
            )
        popen_calls: list[tuple] = []

        def reject_fresh_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            marker.write_text("fresh-exec")
            raise AssertionError("seed recovery must fork from the resident factory")

        monkeypatch.setattr(terminal_tool_module.subprocess, "Popen", reject_fresh_exec)
        seed.kill()
        seed.wait(timeout=5)

        rebuilt_raw = terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/'
            'video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
        assert rebuilt_raw is not None
        rebuilt = json.loads(rebuilt_raw)
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        stdlib_marker = tmp_path / "stdlib-shadow-loaded"
        (tmp_path / "uuid.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(stdlib_marker)!r}).write_text('loaded')\n"
        )
        stdlib_probe = _worker_payload(
            "try:\n"
            "    import uuid\n"
            "except ModuleNotFoundError:\n"
            "    print('stdlib-filesystem-sealed')\n"
            "else:\n"
            "    print('stdlib-filesystem-loaded')\n"
        )
        stdlib_probe["cwd"] = str(tmp_path)
        stdlib_result = terminal_tool_module._run_video_edit_worker(
            stdlib_probe,
            timeout=5,
        )

        assert first["output"] == "trusted-memory-source"
        assert rebuilt["output"] == "trusted-memory-source"
        assert rebuilt["exit_code"] == 0
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != old_seed_pid
        assert terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT is retained_snapshot
        assert terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL is None
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
        assert stdlib_result["stdout"].strip() == "stdlib-filesystem-sealed"
        assert not stdlib_marker.exists()
        assert not marker.exists()
        assert popen_calls == []
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux retains a sealed interpreter for factory recovery",
)
@pytest.mark.live_system_guard_bypass
def test_video_runtime_factory_loss_rebuilds_from_memory_snapshots_flow(
    monkeypatch,
    tmp_path,
):
    copies = _install_temp_worker_sources(monkeypatch, tmp_path)
    script = _write_trusted_script(tmp_path)
    script.write_text("print('trusted-factory-memory-source')\n")
    marker = tmp_path / "post-terminal-factory-disk-source-loaded"
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    try:
        first_raw = terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/'
            'video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
        assert first_raw is not None
        first = json.loads(first_raw)
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        retained_snapshot = terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT
        assert first["output"] == "trusted-factory-memory-source"
        assert factory is not None
        assert supervisor is not None
        assert supervisor.process.poll() is None
        assert seed is not None
        assert retained_snapshot is not None
        old_factory_pid = factory.pid
        supervisor_pid = supervisor.process.pid
        old_seed_pid = seed.pid

        terminal_tool_module._close_video_edit_worker_disk_trust()
        script.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('script')\n"
        )
        for name, source in copies.items():
            source.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text({name!r})\n"
            )
        factory.kill()
        factory.wait(timeout=5)
        popen_argv: list[tuple[str, ...]] = []

        def reject_post_trust_exec(*args, **kwargs):
            popen_argv.append(tuple(args[0]))
            marker.write_text("fresh-exec")
            raise AssertionError("factory recovery must fork from resident memory")

        monkeypatch.setattr(terminal_tool_module.subprocess, "Popen", reject_post_trust_exec)

        rebuilt_raw = terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/'
            'video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
        assert rebuilt_raw is not None
        rebuilt = json.loads(rebuilt_raw)
        rebuilt_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        rebuilt_supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS

        assert rebuilt["output"] == "trusted-factory-memory-source"
        assert rebuilt["exit_code"] == 0
        assert rebuilt_factory is not None
        assert rebuilt_factory.pid != old_factory_pid
        assert rebuilt_supervisor is supervisor
        assert rebuilt_supervisor.process.pid == supervisor_pid
        assert rebuilt_supervisor.process.poll() is None
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != old_seed_pid
        assert terminal_tool_module._VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT is retained_snapshot
        assert popen_argv == []
        assert not marker.exists()
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux verifies same-UID terminal signal recovery",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_same_uid_terminal_kill_of_supervisor_rebuilds_from_gateway_flow(
    monkeypatch,
    tmp_path,
):
    copies = _install_temp_worker_sources(monkeypatch, tmp_path)
    disk_marker = tmp_path / "post-terminal-supervisor-source-loaded"
    stdlib_marker = tmp_path / "post-terminal-supervisor-stdlib-loaded"
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('before-supervisor-kill')\n"),
            timeout=5,
        )
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        resident_image = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_IMAGE
        assert first["stdout"].strip() == "before-supervisor-kill"
        assert supervisor is not None
        assert resident_image is not None

        terminal_tool_module._close_video_edit_worker_disk_trust()
        for name, source in copies.items():
            source.write_text(
                "from pathlib import Path\n"
                f"Path({str(disk_marker)!r}).write_text({name!r})\n"
            )
        (tmp_path / "uuid.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(stdlib_marker)!r}).write_text('uuid')\n"
        )
        killed = LocalEnvironment(cwd=str(tmp_path)).execute(
            f"kill -9 -- {supervisor.process.pid}",
            timeout=5,
        )
        assert killed["returncode"] == 0
        supervisor.process.wait(timeout=5)

        popen_calls: list[tuple] = []

        def reject_post_trust_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            raise AssertionError("supervisor recovery must fork from gateway memory")

        monkeypatch.setattr(terminal_tool_module.subprocess, "Popen", reject_post_trust_exec)
        monkeypatch.setattr(
            terminal_tool_module,
            "_sealed_trusted_video_edit_worker_interpreter",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("supervisor recovery must not reopen the interpreter")
            ),
        )
        recovery_probe = _worker_payload(
            "print('after-supervisor-kill')\n"
            "try:\n"
            "    import uuid\n"
            "except ModuleNotFoundError:\n"
            "    print('stdlib-filesystem-sealed')\n"
            "else:\n"
            "    print('stdlib-filesystem-loaded')\n"
        )
        recovery_probe["cwd"] = str(tmp_path)
        recovered = terminal_tool_module._run_video_edit_worker(
            recovery_probe,
            timeout=5,
        )
        rebuilt_supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

        assert recovered["stdout"].splitlines() == [
            "after-supervisor-kill",
            "stdlib-filesystem-sealed",
        ]
        assert rebuilt_supervisor is not None
        assert rebuilt_supervisor.process.pid != supervisor.process.pid
        assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_IMAGE is resident_image
        assert popen_calls == []
        assert not disk_marker.exists()
        assert not stdlib_marker.exists()
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    sys.platform.startswith("linux"),
    reason="non-Linux exercises the same resident fork recovery without memfd",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_factory_loss_after_terminal_reuses_resident_supervisor_flow(
    monkeypatch,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('factory-ready')\n"),
            timeout=5,
        )
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        assert first["stdout"].strip() == "factory-ready"
        assert factory is not None
        assert supervisor is not None
        assert supervisor.process.poll() is None
        supervisor_pid = supervisor.process.pid

        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED",
            True,
        )
        factory.kill()
        factory.wait(timeout=5)
        popen_calls: list[tuple] = []

        def reject_post_trust_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            raise AssertionError("factory recovery must not exec after trust closes")

        monkeypatch.setattr(terminal_tool_module.subprocess, "Popen", reject_post_trust_exec)

        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('resident-recovery')\n"),
            timeout=5,
        )
        rebuilt_supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

        assert recovered["stdout"].strip() == "resident-recovery"
        assert rebuilt_supervisor is supervisor
        assert rebuilt_supervisor.process.pid == supervisor_pid
        assert popen_calls == []
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.live_system_guard_bypass
def test_video_worker_factory_readiness_loss_retries_from_resident_supervisor_flow(
    monkeypatch,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED",
        False,
    )
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('factory-ready-before-probe-loss')\n"),
            timeout=5,
        )
        old_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        assert first["stdout"].strip() == "factory-ready-before-probe-loss"
        assert old_factory is not None
        assert supervisor is not None
        supervisor_pid = supervisor.process.pid

        terminal_tool_module._close_video_edit_worker_disk_trust()
        old_factory.kill()
        old_factory.wait(timeout=5)

        real_factory_is_ready = terminal_tool_module._video_edit_worker_factory_is_ready
        failed_candidates = []

        def lose_first_rebuilt_factory_probe():
            ready = real_factory_is_ready()
            candidate = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
            if (
                ready
                and candidate is not None
                and candidate.pid != old_factory.pid
                and not failed_candidates
            ):
                failed_candidates.append(candidate)
                return False
            return ready

        popen_calls: list[tuple] = []

        def reject_post_trust_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            raise AssertionError("factory recovery must use the resident supervisor")

        monkeypatch.setattr(
            terminal_tool_module,
            "_video_edit_worker_factory_is_ready",
            lose_first_rebuilt_factory_probe,
        )
        monkeypatch.setattr(
            terminal_tool_module.subprocess,
            "Popen",
            reject_post_trust_exec,
        )

        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('factory-ready-after-probe-loss')\n"),
            timeout=5,
        )
        rebuilt_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        rebuilt_supervisor = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

        assert recovered["stdout"].strip() == "factory-ready-after-probe-loss"
        assert len(failed_candidates) == 1
        assert failed_candidates[0].poll() is not None
        assert rebuilt_factory is not None
        assert rebuilt_factory.pid not in {
            old_factory.pid,
            failed_candidates[0].pid,
        }
        assert rebuilt_supervisor is supervisor
        assert rebuilt_supervisor.process.pid == supervisor_pid
        assert popen_calls == []
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux executes the initial factory from a sealed memfd",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_interpreter_path_swap_recovers_without_exec_or_fd_leak_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT", None)
    trusted_interpreter = tmp_path / "trusted-python"
    original_interpreter = Path(sys.executable).resolve(strict=True)
    shutil.copy2(original_interpreter, trusted_interpreter)
    monkeypatch.setattr(
        terminal_tool_module.sys,
        "executable",
        str(trusted_interpreter),
    )
    marker = tmp_path / "replacement-interpreter-ran"
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('sealed-interpreter')\n"),
            timeout=5,
        )
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert first["stdout"].strip() == "sealed-interpreter"
        assert factory is not None
        assert seed is not None
        factory_pid = factory.pid
        old_seed_pid = seed.pid

        replacement = tmp_path / "replacement-python"
        replacement.write_text(
            "#!/bin/sh\n"
            f"printf compromised > {str(marker)!r}\n"
            "exit 99\n"
        )
        replacement.chmod(0o500)
        os.replace(replacement, trusted_interpreter)
        terminal_tool_module._close_video_edit_worker_disk_trust()
        real_popen = subprocess.Popen
        popen_calls: list[tuple] = []

        def record_sealed_exec(*args, **kwargs):
            popen_calls.append((args, kwargs))
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(terminal_tool_module.subprocess, "Popen", record_sealed_exec)
        factory.kill()
        factory.wait(timeout=5)
        fd_probe = _worker_payload(
            "import os\n"
            "targets = []\n"
            "for name in os.listdir('/proc/self/fd'):\n"
            "    try:\n"
            "        targets.append(os.readlink('/proc/self/fd/' + name))\n"
            "    except OSError:\n"
            "        pass\n"
            "print(any('hermes-video-worker-python' in value for value in targets))\n"
        )
        recovered = terminal_tool_module._run_video_edit_worker(fd_probe, timeout=5)
        rebuilt_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS

        assert recovered["stdout"].strip() == "False"
        assert rebuilt_factory is not None
        assert rebuilt_factory.pid != factory_pid
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != old_seed_pid
        assert popen_calls == []
        assert not marker.exists()
        for pid in (rebuilt_factory.pid, rebuilt_seed.pid):
            targets = []
            for name in os.listdir(f"/proc/{pid}/fd"):
                try:
                    targets.append(os.readlink(f"/proc/{pid}/fd/{name}"))
                except OSError:
                    pass
            assert not any(
                "hermes-video-worker-python" in value for value in targets
            )
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork semantics")
@pytest.mark.live_system_guard_bypass
def test_video_worker_fork_child_discards_handles_without_controlling_parent_flow():
    terminal_tool_module._stop_video_edit_worker()
    terminal_tool_module._VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED = False
    read_fd, write_fd = os.pipe()
    try:
        first = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('before-fork')\n"),
            timeout=5,
        )
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert first["stdout"].strip() == "before-fork"
        assert factory is not None
        assert seed is not None

        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                inherited_handles_cleared = all(
                    value is None
                    for value in (
                        terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID,
                        terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL,
                        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS,
                        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_CHANNEL,
                        terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS,
                        terminal_tool_module._VIDEO_EDIT_WORKER_SEED_CHANNEL,
                        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_IMAGE,
                        terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR,
                        terminal_tool_module._VIDEO_EDIT_WORKER_IDLE_TIMER,
                    )
                )
                terminal_tool_module._stop_video_edit_worker()
                os.write(write_fd, b"1" if inherited_handles_cleared else b"0")
            finally:
                os.close(write_fd)
                os._exit(0)

        os.close(write_fd)
        write_fd = -1
        child_state = os.read(read_fd, 1)
        waited_pid, status = os.waitpid(child_pid, 0)

        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
        assert child_state == b"1"
        assert factory.poll() is None
        assert seed.poll() is None
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-fork')\n"),
            timeout=5,
        )
        assert recovered["stdout"].strip() == "after-fork"
        assert terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS.pid == factory.pid
        assert terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS.pid == seed.pid
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux PDEATHSIG closes seed SIGKILL descendants",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_seed_sigkill_cleans_inflight_and_rebuilds_from_memory_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    pid_marker = tmp_path / "seed-loss-daemon.pid"
    leak_marker = tmp_path / "seed-loss-daemon-token"
    payload = _worker_payload(
        _setsid_double_fork_source(pid_marker, leak_marker, parent_sleep=True)
    )
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "seed-loss-secret"}
    results: list[dict] = []
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            results.append(
                terminal_tool_module._run_video_edit_worker(
                    payload,
                    timeout=10,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - assert thread failures
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_worker)
        thread.start()
        deadline = time.monotonic() + 5
        while not pid_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_marker.exists()
        descendant_pid = int(pid_marker.read_text())
        broker_pid = terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert seed is not None
        assert broker_pid is not None
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID == broker_pid

        seed.kill()
        seed.wait(timeout=5)
        thread.join(timeout=8)

        assert not thread.is_alive()
        assert not errors
        assert results[0]["returncode"] == 1
        assert results[0]["worker"]["reaped"] is False
        assert _wait_for_pid_exit(broker_pid)
        assert _wait_for_pid_exit(descendant_pid)
        time.sleep(0.1)
        assert not leak_marker.exists()
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-seed-rebuild')\n"), timeout=5
        )
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert recovered["stdout"].strip() == "after-seed-rebuild"
        assert recovered["worker"]["reaped"] is True
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != seed.pid
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux parent-death and subreaper boundaries contain factory loss",
)
@pytest.mark.live_system_guard_bypass
def test_video_worker_factory_sigkill_cleans_inflight_and_rebuilds_from_memory_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    pid_marker = tmp_path / "factory-loss-daemon.pid"
    leak_marker = tmp_path / "factory-loss-daemon-token"
    payload = _worker_payload(
        _setsid_double_fork_source(pid_marker, leak_marker, parent_sleep=True)
    )
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "factory-loss-secret"}
    results: list[dict] = []
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            results.append(
                terminal_tool_module._run_video_edit_worker(
                    payload,
                    timeout=10,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - assert thread failures
            errors.append(exc)

    try:
        thread = threading.Thread(target=run_worker)
        thread.start()
        deadline = time.monotonic() + 5
        while not pid_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_marker.exists()
        descendant_pid = int(pid_marker.read_text())
        broker_pid = terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID
        factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert broker_pid is not None
        assert factory is not None
        assert seed is not None

        monkeypatch.setattr(
            terminal_tool_module,
            "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED",
            True,
        )
        factory.kill()
        factory.wait(timeout=5)
        thread.join(timeout=8)

        assert not thread.is_alive()
        assert not errors
        assert results[0]["returncode"] == 1
        assert results[0]["worker"]["reaped"] is False
        assert _wait_for_pid_exit(broker_pid)
        assert _wait_for_pid_exit(descendant_pid)
        time.sleep(0.1)
        assert not leak_marker.exists()

        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-factory-rebuild')\n"), timeout=5
        )
        rebuilt_factory = terminal_tool_module._VIDEO_EDIT_WORKER_FACTORY_PROCESS
        rebuilt_seed = terminal_tool_module._VIDEO_EDIT_WORKER_SEED_PROCESS
        assert recovered["stdout"].strip() == "after-factory-rebuild"
        assert recovered["worker"]["reaped"] is True
        assert rebuilt_factory is not None
        assert rebuilt_factory.pid != factory.pid
        assert rebuilt_seed is not None
        assert rebuilt_seed.pid != seed.pid
    finally:
        terminal_tool_module._stop_video_edit_worker()


@pytest.mark.live_system_guard_bypass
def test_video_worker_success_kills_background_descendants_before_reply_flow(tmp_path):
    marker = tmp_path / "descendant-token"
    delayed_marker = tmp_path / "escaped-after-reply"
    business_token = "business-capability-stays-context-only"
    connector_token = "connector-bearer-inherited-by-trusted-helper"
    child_source = (
        "import os, pathlib, time\n"
        f"pathlib.Path({str(marker)!r}).write_text("
        "os.environ.get('ZETTLAB_CONNECTORS_AUTH_TOKEN', ''))\n"
        "time.sleep(0.5)\n"
        f"pathlib.Path({str(delayed_marker)!r}).write_text('escaped')\n"
    )
    source = textwrap.dedent(
        f"""
        import os
        import subprocess
        import sys
        import time
        from _zettlab_video_runtime_context import get

        assert get("ZETTLAB_BUSINESS_EXECUTION_TOKEN") == {business_token!r}
        child = subprocess.Popen([sys.executable, "-I", "-S", "-c", {child_source!r}])
        deadline = time.monotonic() + 5
        while not os.path.exists({str(marker)!r}) and time.monotonic() < deadline:
            time.sleep(0.01)
        if not os.path.exists({str(marker)!r}):
            raise RuntimeError("background child did not start")
        print(child.pid)
        """
    )
    payload = _worker_payload(source)
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": connector_token}
    payload["secrets"] = {"ZETTLAB_BUSINESS_EXECUTION_TOKEN": business_token}
    descendant_pid = 0
    try:
        result = terminal_tool_module._run_video_edit_worker(payload, timeout=8)
        descendant_pid = int(result["stdout"].strip())
        broker_pid = result["worker"]["pid"]
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('next-call')\n"),
            timeout=5,
        )

        assert result["returncode"] == 0
        assert result["worker"]["reaped"] is True
        assert marker.read_text() == connector_token
        assert connector_token not in json.dumps(result)
        assert business_token not in json.dumps(result)
        assert _wait_for_pid_exit(descendant_pid)
        time.sleep(0.6)
        assert not delayed_marker.exists()
        assert recovered["stdout"].strip() == "next-call"
        assert recovered["worker"]["pid"] != broker_pid
        assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
    finally:
        if descendant_pid and _pid_exists(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux subreaper owns escaped daemon descendants",
)
def test_video_worker_success_reaps_setsid_double_fork_before_reply_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    pid_marker = tmp_path / "success-daemon.pid"
    leak_marker = tmp_path / "success-daemon-token"
    payload = _worker_payload(
        _setsid_double_fork_source(pid_marker, leak_marker, parent_sleep=False)
    )
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "success-secret"}
    try:
        result = terminal_tool_module._run_video_edit_worker(payload, timeout=5)
        escaped_pid = int(pid_marker.read_text())
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-success-daemon')\n"), timeout=5
        )

        assert result["returncode"] == 0
        assert result["worker"]["reaped"] is True
        assert _wait_for_pid_exit(escaped_pid)
        time.sleep(0.1)
        assert not leak_marker.exists()
        assert recovered["stdout"].strip() == "after-success-daemon"
        assert recovered["worker"]["reaped"] is True
    finally:
        terminal_tool_module._stop_video_edit_worker()


def test_video_worker_recycles_one_shot_executor_every_call_flow():
    first = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('first')\n"), timeout=5
    )
    second = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('second')\n"), timeout=5
    )

    assert first["returncode"] == 0
    assert second["returncode"] == 0
    assert first["worker"]["one_shot"] is True
    assert second["worker"]["one_shot"] is True
    assert first["worker"]["reaped"] is True
    assert second["worker"]["reaped"] is True
    assert first["worker"]["pid"] != second["worker"]["pid"]
    assert first["worker"]["call_index"] == 1
    assert second["worker"]["call_index"] == 1
    if sys.platform.startswith("linux"):
        assert first["worker"]["applied"] is True
        assert first["worker"]["limit_bytes"] <= 512 * 1024 * 1024
        assert first["worker"]["max_rss_bytes"] <= first["worker"]["limit_bytes"]


def test_video_worker_address_space_headroom_preserves_rss_budget_unit(monkeypatch):
    from tools import process_security

    monkeypatch.setitem(sys.modules, "process_security", process_security)
    from tools import video_edit_runtime_worker as worker

    assert worker._MEMORY_LIMIT_BYTES == 512 * 1024 * 1024
    assert worker._SUBTREE_RSS_LIMIT_BYTES == 256 * 1024 * 1024
    assert (
        terminal_tool_module._VIDEO_EDIT_WORKER_MEMORY_LIMIT_BYTES
        == worker._MEMORY_LIMIT_BYTES
    )

    tree = ast.parse(terminal_tool_module._VIDEO_EDIT_WORKER_MEMORY_BOOTSTRAP)
    memory_limit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "apply_worker_memory_limit"
    ]
    assert len(memory_limit_calls) == 1
    memory_limit_arg = memory_limit_calls[0].args[0]
    assert isinstance(memory_limit_arg, ast.Attribute)
    assert isinstance(memory_limit_arg.value, ast.Name)
    assert memory_limit_arg.value.id == "worker"
    assert memory_limit_arg.attr == "_MEMORY_LIMIT_BYTES"


def test_video_worker_timeout_has_no_stale_frame_on_next_call_flow():
    timed_out = terminal_tool_module._run_video_edit_worker(
        _worker_payload("import time\ntime.sleep(2)\nprint('too-late')\n"),
        timeout=0.1,
    )
    recovered = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('after-timeout')\n"),
        timeout=5,
    )

    assert timed_out["returncode"] == 124
    assert timed_out["worker"]["reaped"] is True
    assert recovered["returncode"] == 0
    assert recovered["stdout"].strip() == "after-timeout"
    assert recovered["worker"]["reaped"] is True
    assert timed_out["worker"]["call_index"] == 1
    assert recovered["worker"]["call_index"] == 1
    assert recovered["worker"]["pid"] != timed_out["worker"]["pid"]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux subreaper owns escaped daemon descendants",
)
def test_video_worker_timeout_reaps_setsid_double_fork_and_recovers_flow(
    monkeypatch,
    tmp_path,
):
    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    pid_marker = tmp_path / "timeout-daemon.pid"
    leak_marker = tmp_path / "timeout-daemon-token"
    payload = _worker_payload(
        _setsid_double_fork_source(pid_marker, leak_marker, parent_sleep=True)
    )
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "timeout-secret"}
    try:
        timed_out = terminal_tool_module._run_video_edit_worker(payload, timeout=0.1)
        escaped_pid = int(pid_marker.read_text())
        recovered = terminal_tool_module._run_video_edit_worker(
            _worker_payload("print('after-timeout-daemon')\n"), timeout=5
        )

        assert timed_out["returncode"] == 124
        assert timed_out["worker"]["reaped"] is True
        assert _wait_for_pid_exit(escaped_pid)
        time.sleep(0.1)
        assert not leak_marker.exists()
        assert recovered["stdout"].strip() == "after-timeout-daemon"
        assert recovered["worker"]["reaped"] is True
    finally:
        terminal_tool_module._stop_video_edit_worker()


def test_video_worker_seed_environment_is_fixed_and_secret_free(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-reach-seed")
    monkeypatch.setenv("CUSTOM_RUNTIME_SECRET", "must-not-reach-seed")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "must-not-reach-seed")

    assert terminal_tool_module._trusted_video_edit_worker_env() == {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "TZ": "UTC",
    }


def test_video_worker_cleanup_ack_failure_cannot_report_reaped(monkeypatch):
    class _Channel:
        def close(self):
            return None

        def sendall(self, _data):
            return None

        def recv(self, *_args):
            raise BlockingIOError

    class _Process:
        pid = 12345

        @staticmethod
        def poll():
            return None

    broker_identity = terminal_tool_module._VideoEditWorkerProcessIdentity(
        pid=43210,
        start_time=None,
        pidfd=None,
    )
    forced: list[terminal_tool_module._VideoEditWorkerProcessIdentity] = []
    discarded: list[bool] = []
    monkeypatch.setattr(
        terminal_tool_module,
        "_VIDEO_EDIT_WORKER_BROKER_IDENTITY",
        broker_identity,
    )
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_PID", 43210)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_CHANNEL", _Channel())
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_SEED_CHANNEL", _Channel())
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_SEED_PROCESS", _Process())
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_frame",
        lambda *_args, **_kwargs: {
            "cleanup": "unknown",
            "pid": 43210,
            "reaped": False,
        },
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_force_kill_video_edit_worker_group",
        forced.append,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_discard_video_edit_worker_seed",
        lambda: discarded.append(True),
    )

    assert terminal_tool_module._terminate_video_edit_worker(
        close_disk_trust=False
    ) is False
    assert terminal_tool_module._VIDEO_EDIT_WORKER_BROKER_PID is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_CHANNEL is None
    assert terminal_tool_module._VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED is True
    assert forced == [broker_identity]
    assert discarded == [True]


def test_video_worker_success_response_fails_closed_without_cleanup_ack(monkeypatch):
    channel = object()
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_video_edit_worker_started",
        lambda: None,
    )

    def spawn() -> None:
        monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_BROKER_PID", 43210)
        monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_CHANNEL", channel)

    monkeypatch.setattr(terminal_tool_module, "_spawn_video_edit_worker_broker", spawn)
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_send_frame",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_video_edit_worker_recv_frame",
        lambda *_args, **_kwargs: {
            "stdout": "must-not-return",
            "stderr": "",
            "returncode": 0,
            "worker": {"one_shot": True, "pid": 43210, "call_index": 1},
        },
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_terminate_video_edit_worker",
        lambda **_kwargs: False,
    )

    result = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('ignored')\n"), timeout=5
    )

    assert result["returncode"] == 1
    assert result["stdout"] == ""
    assert result["worker"]["reaped"] is False


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux /proc watchdog enforces owned-tree fanout",
)
def test_video_worker_owned_tree_fanout_catches_setsid_double_forks_flow(tmp_path):
    pid_dir = tmp_path / "fanout-pids"
    leak_dir = tmp_path / "fanout-leaks"
    pid_dir.mkdir()
    leak_dir.mkdir()
    source = textwrap.dedent(
        f"""
        import os
        import pathlib
        import time

        for index in range(20):
            first = os.fork()
            if first == 0:
                os.setsid()
                second = os.fork()
                if second > 0:
                    os._exit(0)
                pathlib.Path({str(pid_dir)!r}, str(index)).write_text(str(os.getpid()))
                time.sleep(4.0)
                pathlib.Path({str(leak_dir)!r}, str(index)).write_text(
                    os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN", "missing")
                )
                os._exit(0)
            os.waitpid(first, 0)
        time.sleep(30)
        """
    )
    payload = _worker_payload(source)
    payload["env"] = {"ZETTLAB_CONNECTORS_AUTH_TOKEN": "fanout-secret"}
    limited = terminal_tool_module._run_video_edit_worker(
        payload, timeout=8
    )
    escaped_pids = [int(path.read_text()) for path in pid_dir.iterdir()]
    recovered = terminal_tool_module._run_video_edit_worker(
        _worker_payload("print('after-fanout')\n"), timeout=5
    )

    assert limited["returncode"] == 1
    assert limited["worker"]["reaped"] is True
    assert escaped_pids
    assert all(_wait_for_pid_exit(pid) for pid in escaped_pids)
    time.sleep(0.1)
    assert list(leak_dir.iterdir()) == []
    assert recovered["returncode"] == 0
    assert recovered["stdout"].strip() == "after-fanout"
    assert recovered["worker"]["reaped"] is True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is Linux gate")
def test_video_worker_hard_memory_limit_fails_closed_flow():
    result = terminal_tool_module._run_video_edit_worker(
        _worker_payload("payload = bytearray(700 * 1024 * 1024)\nprint(len(payload))\n"),
        timeout=5,
    )

    assert result["returncode"] == 1
    assert "MemoryError" in result["stderr"]
    assert result["worker"]["one_shot"] is True
    assert result["worker"]["limit_bytes"] <= 512 * 1024 * 1024


def test_trusted_runtime_source_cache_never_rereads_disk_after_trust_closes(
    monkeypatch,
    tmp_path,
):
    script = _write_trusted_script(tmp_path)
    original_source = script.read_text()
    presets_root = tmp_path / "presets"
    identity = (123, 456)
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE", {})
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES", 0)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda *_args, **_kwargs: True,
    )

    captured = terminal_tool_module._trusted_video_edit_source_bundle(
        script=script,
        presets_root=presets_root,
        expected_root_identity=identity,
    )
    terminal_tool_module._close_video_edit_worker_disk_trust()
    original_inode = script.stat().st_ino
    script.write_text("raise RuntimeError('tampered same inode')\n")
    assert script.stat().st_ino == original_inode

    def disk_access_forbidden(*_args, **_kwargs):
        raise AssertionError("post-trust runtime source disk access")

    monkeypatch.setattr(Path, "glob", disk_access_forbidden)
    monkeypatch.setattr(
        terminal_tool_module,
        "_read_stable_trusted_worker_source",
        disk_access_forbidden,
    )
    reused = terminal_tool_module._trusted_video_edit_source_bundle(
        script=script,
        presets_root=presets_root,
        expected_root_identity=identity,
    )

    assert captured["__main__"]["source"] == original_source
    assert reused == captured


def test_trusted_runtime_source_cache_miss_fails_before_disk_access_post_trust(
    monkeypatch,
    tmp_path,
):
    script = _write_trusted_script(tmp_path)
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE", {})
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES", 0)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", True)
    disk_accesses: list[bool] = []

    def disk_access_forbidden(*_args, **_kwargs):
        disk_accesses.append(True)
        raise AssertionError("post-trust runtime source disk access")

    monkeypatch.setattr(Path, "glob", disk_access_forbidden)
    monkeypatch.setattr(
        terminal_tool_module,
        "_read_stable_trusted_worker_source",
        disk_access_forbidden,
    )

    with pytest.raises(PermissionError, match="not captured before terminal access"):
        terminal_tool_module._trusted_video_edit_source_bundle(
            script=script,
            presets_root=tmp_path / "presets",
            expected_root_identity=(123, 456),
        )
    assert disk_accesses == []


def test_trusted_runtime_source_directory_reserves_ipc_frame_overhead(
    monkeypatch,
    tmp_path,
):
    scripts = tmp_path / "presets" / "skills" / "linear" / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "connector_runtime.py"
    for index in range(13):
        (scripts / ("connector_runtime.py" if index == 0 else f"module_{index}.py")).write_text(
            "# placeholder\n"
        )
    source = b"#" + (b"x" * (500 * 1024 - 2)) + b"\n"
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE", {})
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES", 0)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_read_stable_trusted_worker_source",
        lambda _path: source,
    )

    with pytest.raises(MemoryError, match="exceeds IPC budget"):
        terminal_tool_module._trusted_video_edit_source_bundle(
            script=script,
            presets_root=tmp_path / "presets",
            expected_root_identity=(123, 456),
        )
    assert terminal_tool_module._TRUSTED_RUNTIME_SOURCE_CACHE == {}


def test_terminal_late_retry_preloads_bundles_and_image_before_trust_closes(
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", "/trusted/presets")
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_preload_trusted_runtime_source_bundles",
        lambda: calls.append("preload"),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_trusted_video_edit_worker_snapshot_for_seed_start",
        lambda: calls.append("snapshot") or object(),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_trusted_video_edit_worker_factory_image",
        lambda _snapshot: calls.append("image"),
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_trusted_video_edit_worker_factory_bootstrap",
        lambda _snapshot: calls.append("supervisor"),
    )

    terminal_tool_module._late_prepare_video_edit_worker_before_terminal()

    assert calls == ["snapshot", "image", "preload", "supervisor"]
    assert terminal_tool_module._VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED is False


def test_generic_terminal_then_first_trusted_video_uses_frozen_image(
    monkeypatch,
    tmp_path,
):
    _write_trusted_script(tmp_path)
    presets_root = tmp_path / "presets"
    boundary_calls = []

    terminal_tool_module._stop_video_edit_worker()
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_root))
    monkeypatch.setenv("ZET_AGENT_ID", "agent-after-terminal")
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT", None)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_FACTORY_IMAGE", None)
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE", {})
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES", 0)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setattr(terminal_tool_module, "_SENSITIVE_PROCESS_OS_BOUNDARY", False)
    monkeypatch.setattr(terminal_tool_module, "_MODEL_DESCENDANT_PTRACE_BOUNDARY", False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "harden_sensitive_process",
        lambda **_kwargs: boundary_calls.append("hardened") or True,
    )

    generic = json.loads(
        terminal_tool_module.terminal_tool(
            "printf generic",
            workdir=str(tmp_path),
        )
    )
    assert generic["exit_code"] == 0
    assert generic["output"] == "generic"
    assert terminal_tool_module._VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED is True

    tokens = set_turn_vars(
        turn_id="turn-after-terminal",
        business_execution_token="capability-after-terminal",
    )
    try:
        video = json.loads(
            terminal_tool_module._run_video_edit_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/'
                'video-edit-workflow-mini/scripts/workflow_state.py"',
                cwd=str(tmp_path),
                timeout=5,
            )
        )
    finally:
        clear_turn_vars(tokens)
        terminal_tool_module._stop_video_edit_worker()

    assert video["video_edit_runtime_direct"] is True
    assert video["exit_code"] == 0
    assert "execution=[REDACTED]" in video["output"]
    assert "agent=agent-after-terminal" in video["output"]
    assert "turn=turn-after-terminal" in video["output"]
    assert boundary_calls == ["hardened"]


def test_trusted_video_runner_rejects_untrusted_sibling_dependency(monkeypatch, tmp_path):
    script = _write_trusted_script(tmp_path, "preference_resolver.py")
    sibling = script.parent / "workflow_state.py"
    sibling.write_text("VALUE = 'tampered'\n")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))

    def trust_only_entry(path, presets_root, **kwargs):
        return Path(path) != sibling

    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        trust_only_entry,
    )
    result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/scripts/preference_resolver.py"',
        cwd=str(tmp_path),
        timeout=5,
    ))
    assert result["video_edit_runtime_direct"] is True
    assert result["exit_code"] == -1
    assert "PermissionError" in result["error"]


def test_trusted_video_runner_rejects_same_inode_mutation_during_snapshot(
    monkeypatch, tmp_path
):
    script = _write_trusted_script(tmp_path, "preference_resolver.py")
    original_inode = script.stat().st_ino
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE", {})
    monkeypatch.setattr(terminal_tool_module, "_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES", 0)
    monkeypatch.setattr(terminal_tool_module, "_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED", False)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    real_fstat = terminal_tool_module.os.fstat
    mutated = False

    def mutate_after_open(fd):
        nonlocal mutated
        opened_stat = real_fstat(fd)
        if not mutated and opened_stat.st_ino == original_inode:
            mutated = True
            script.write_text("print('tampered during snapshot')\n")
        return opened_stat

    monkeypatch.setattr(terminal_tool_module.os, "fstat", mutate_after_open)
    result = json.loads(
        terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/'
            'video-edit-workflow-mini/scripts/preference_resolver.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
    )

    assert mutated is True
    assert script.stat().st_ino == original_inode
    assert result["exit_code"] == -1
    assert "PermissionError" in result["error"]


def test_plan_preparation_command_allows_only_preference_resolver_readiness_steps(
    monkeypatch, tmp_path
):
    _write_trusted_script(tmp_path, "preference_resolver.py")
    _write_trusted_script(tmp_path, "normalize.py")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    prefix = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/'
        'video-edit-workflow-mini/scripts/'
    )
    assert terminal_tool_module._is_video_edit_plan_preparation_command(
        prefix + 'preference_resolver.py" freeze --state-file /safe/state.json'
    )
    assert not terminal_tool_module._is_video_edit_plan_preparation_command(
        prefix + 'preference_resolver.py" authorize --state-file /safe/state.json'
    )
    assert not terminal_tool_module._is_video_edit_plan_preparation_command(
        prefix + 'normalize.py" --input in.mov --output out.mp4'
    )


@pytest.mark.parametrize("suffix", ["; env", "| cat", "&"])
def test_video_runner_rejects_compound_or_background_commands(monkeypatch, tmp_path, suffix):
    _write_trusted_script(tmp_path, "normalize.py")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/video-edit-workflow-mini/scripts/normalize.py" ' + suffix,
        cwd=str(tmp_path),
        timeout=5,
    ))
    assert result["video_edit_runtime_blocked"] is True


def test_video_runner_rejects_same_named_script_outside_pinned_presets(monkeypatch, tmp_path):
    script = tmp_path / "workflow_state.py"
    script.write_text("print('untrusted')")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    result = json.loads(terminal_tool_module._run_video_edit_runtime_command_if_allowed(
        f"python3 {script}", cwd=str(tmp_path), timeout=5,
    ))
    assert result["video_edit_runtime_blocked"] is True


def test_video_runner_rejects_source_layout_path(monkeypatch, tmp_path):
    _write_trusted_script(tmp_path, "workflow_state.py")
    source_script = (
        tmp_path
        / "presets"
        / "skills"
        / "common"
        / "video-edit-workflow-mini"
        / "scripts"
        / "workflow_state.py"
    )
    source_script.parent.mkdir(parents=True, exist_ok=True)
    source_script.write_text("print('source layout must not run')\n")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = json.loads(
        terminal_tool_module._run_video_edit_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/common/'
            'video-edit-workflow-mini/scripts/workflow_state.py"',
            cwd=str(tmp_path),
            timeout=5,
        )
    )

    assert result["video_edit_runtime_blocked"] is True
