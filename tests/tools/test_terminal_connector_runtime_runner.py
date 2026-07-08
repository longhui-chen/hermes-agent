import json
import os
import sys
import textwrap
from io import StringIO

from tools import terminal_tool as terminal_tool_module


def _write_connector_runtime(tmp_path):
    script = tmp_path / "presets" / "skills" / "linear" / "scripts" / "connector_runtime.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
        import os

        print("connector token ok: " + str(os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN") == "runner-token"))
        print("connector_agent=" + str(os.environ.get("ZET_AGENT_ID")))
        """
    ).lstrip())
    return script


def _write_connector_runtime_with_import(tmp_path):
    script = tmp_path / "presets" / "skills" / "linear" / "scripts" / "connector_runtime.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
        import os
        import shadowed_dependency

        print("connector token ok: " + str(os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN") == "runner-token"))
        print("dependency=" + shadowed_dependency.VALUE)
        """
    ).lstrip())
    return script


def _write_connector_runtime_with_global_mutation(tmp_path):
    script = tmp_path / "presets" / "skills" / "linear" / "scripts" / "connector_runtime.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
        import os
        import sys

        os.environ["PARENT_SHOULD_NOT_SEE"] = os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN", "")
        sys.path[:] = ["connector-only-path"]
        os.chdir("/")
        print("worker token ok: " + str(os.environ.get("ZETTLAB_CONNECTORS_AUTH_TOKEN") == "runner-token"))
        """
    ).lstrip())
    return script


def _write_connector_runtime_sleep(tmp_path):
    script = tmp_path / "presets" / "skills" / "linear" / "scripts" / "connector_runtime.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
        import time

        time.sleep(30)
        print("should not finish")
        """
    ).lstrip())
    return script


def test_connector_runtime_direct_runner_flow_receives_profile_scoped_env(monkeypatch, tmp_path):
    """Official presets keep working through the dedicated connector runner."""
    from agent import secret_scope as ss

    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    ss.set_multiplex_active(False)

    result = json.loads(terminal_tool_module.terminal_tool(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        task_id="connector-runtime-direct-test",
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "connector token ok: True" in result["output"]
    assert "connector_agent=agent-1" in result["output"]


def test_connector_runtime_direct_runner_keeps_token_out_of_popen_env(monkeypatch, tmp_path):
    """Connector bearer is delivered over stdin to the allowlisted runner."""
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1/rpc")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env", {})
        captured["input"] = kwargs.get("input", "")
        return terminal_tool_module.subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(terminal_tool_module.subprocess, "run", fake_run)

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        cwd=str(tmp_path),
        timeout=5,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["env"]
    assert "ZETTLAB_CONNECTORS_URL" not in captured["env"]
    assert "runner-token" in captured["input"]
    assert "http://127.0.0.1/rpc" in captured["input"]
    assert captured["argv"][:2] == [sys.executable, "-c"]


def test_connector_runtime_direct_runner_preserves_parent_process_globals(monkeypatch, tmp_path):
    """Worker env/path/cwd/stdout mutations cannot bleed into the gateway process."""
    _write_connector_runtime_with_global_mutation(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    original_cwd = os.getcwd()
    original_path = list(sys.path)
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        captured_parent_stdout = sys.stdout.getvalue()
        sys.stdout = original_stdout

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "worker token ok: True" in result["output"]
    assert "PARENT_SHOULD_NOT_SEE" not in os.environ
    assert os.getcwd() == original_cwd
    assert sys.path == original_path
    assert captured_parent_stdout == ""


def test_connector_runtime_direct_runner_timeout_restores_control(monkeypatch, tmp_path):
    _write_connector_runtime_sleep(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
        cwd=str(tmp_path),
        timeout=1,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 124
    assert "timed out" in result["error"]
    assert "should not finish" not in result["output"]


def test_connector_runtime_isolated_sys_path_preserves_venv_under_root(monkeypatch):
    fake_site_packages = "/root/.hermes/hermes-agent/venv/lib/python3.11/site-packages"
    monkeypatch.setattr(
        terminal_tool_module.sys,
        "path",
        ["/root", fake_site_packages, ""],
    )

    isolated = terminal_tool_module._connector_runtime_isolated_sys_path(
        script=terminal_tool_module.Path("/presets/skills/linear/scripts/connector_runtime.py"),
        cwd=terminal_tool_module.Path("/root"),
    )

    assert "/root" not in isolated
    assert fake_site_packages in isolated


def test_connector_runtime_direct_runner_isolates_pythonpath(monkeypatch, tmp_path):
    """A model-writable cwd/PYTHONPATH module cannot run after token injection."""
    safe_dep = tmp_path / "safe"
    safe_dep.mkdir()
    (safe_dep / "shadowed_dependency.py").write_text('VALUE = "safe"\n')
    attacker_dep = tmp_path / "attacker"
    attacker_dep.mkdir()
    (attacker_dep / "shadowed_dependency.py").write_text(
        "import os\n"
        "VALUE = os.environ.get('ZETTLAB_CONNECTORS_AUTH_TOKEN', 'missing')\n"
    )
    _write_connector_runtime_with_import(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("PYTHONPATH", str(attacker_dep))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    monkeypatch.syspath_prepend(str(safe_dep))

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        cwd=str(attacker_dep),
        timeout=5,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "dependency=safe" in result["output"]
    assert "runner-token" not in result["output"]


def test_connector_runtime_direct_runner_redacts_before_truncating(monkeypatch, tmp_path):
    secret = "SECRET-" + ("x" * 64) + "-END"
    script = tmp_path / "presets" / "skills" / "linear" / "scripts" / "connector_runtime.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        f"""
        import os

        print("prefix-" + os.environ["ZETTLAB_CONNECTORS_AUTH_TOKEN"] + "-suffix")
        """
    ).lstrip())
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", secret)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    from tools import tool_output_limits

    monkeypatch.setattr(tool_output_limits, "get_max_bytes", lambda: 45)

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
        cwd=str(tmp_path),
        timeout=5,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "SECRET-" not in result["output"]
    assert "-END" not in result["output"]
    assert "xxxxxxxx" not in result["output"]


def test_compound_connector_runtime_command_does_not_receive_token(monkeypatch, tmp_path):
    """A command with shell punctuation falls back to generic terminal env."""
    from agent import secret_scope as ss

    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )
    ss.set_multiplex_active(False)

    result = json.loads(terminal_tool_module.terminal_tool(
        (
            'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools; '
            f'{sys.executable} -c "import os; '
            'print(os.environ.get(\\"ZETTLAB_CONNECTORS_AUTH_TOKEN\\", \\"missing\\"))"'
        ),
        task_id="connector-runtime-compound-test",
    ))

    assert result.get("connector_runtime_direct") is not True
    assert result["exit_code"] == 0
    assert "connector token ok: False" in result["output"]
    assert "runner-token" not in result["output"]


def test_parser_rejects_non_presets_or_compound_connector_runtime(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root: True,
    )

    direct = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools'
    )
    relative = terminal_tool_module._parse_connector_runtime_command(
        "python3 skills/linear/scripts/connector_runtime.py list-tools"
    )
    compound = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"; env'
    )
    non_presets = terminal_tool_module._parse_connector_runtime_command(
        f"python3 {script.parent.parent / 'connector_runtime.py'}"
    )

    assert direct is not None
    assert relative is not None
    assert compound is None
    assert non_presets is None


def test_parser_rejects_writable_presets_runner_by_default(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))

    parsed = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools'
    )

    assert parsed is None


def test_connector_runtime_trust_rejects_non_root_owned_tree_for_root_service(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 0, raising=False)

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        tmp_path / "presets",
    ) is False


def test_connector_runtime_trust_rejects_root_tree_modified_after_module_load(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 0, raising=False)
    future_mtime = terminal_tool_module._CONNECTOR_RUNTIME_TRUST_CUTOFF + 10

    class FakeStat:
        st_uid = 0
        st_gid = 0
        st_mode = 0o100644
        st_mtime = future_mtime
        st_ctime = future_mtime

    original_stat = terminal_tool_module.Path.stat

    def fake_stat(path):
        if path == script:
            return FakeStat()
        return original_stat(path)

    monkeypatch.setattr(terminal_tool_module.Path, "stat", fake_stat)

    assert terminal_tool_module._path_writable_by_current_user(script) is True


def test_connector_runtime_trust_rejects_root_tree_ctime_bump(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 0, raising=False)
    old_mtime = terminal_tool_module._CONNECTOR_RUNTIME_TRUST_CUTOFF - 10
    future_ctime = terminal_tool_module._CONNECTOR_RUNTIME_TRUST_CUTOFF + 10

    class FakeStat:
        st_uid = 0
        st_gid = 0
        st_mode = 0o100644
        st_mtime = old_mtime
        st_ctime = future_ctime

    original_stat = terminal_tool_module.Path.stat

    def fake_stat(path):
        if path == script:
            return FakeStat()
        return original_stat(path)

    monkeypatch.setattr(terminal_tool_module.Path, "stat", fake_stat)

    assert terminal_tool_module._path_writable_by_current_user(script) is True


def test_connector_runtime_trust_checks_presets_parent_directories(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    presets_root = tmp_path / "presets"
    writable_parent = tmp_path

    def fake_writable(path):
        return path == writable_parent

    monkeypatch.setattr(
        terminal_tool_module,
        "_path_writable_by_current_user",
        fake_writable,
    )

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        presets_root,
    ) is False


def test_connector_runtime_output_force_redacts_actual_secret_values():
    result = json.loads(terminal_tool_module._connector_runtime_result_json(
        command='python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
        output=(
            'ZETTLAB_CONNECTORS_AUTH_TOKEN=runner-token\n'
            '{"token":"runner-token","url":"http://127.0.0.1/rpc?auth=runtime-url-secret"}'
        ),
        returncode=1,
        secret_values=["runner-token", "http://127.0.0.1/rpc?auth=runtime-url-secret"],
    ))

    assert "runner-token" not in result["output"]
    assert "runtime-url-secret" not in result["output"]
    assert "[REDACTED]" in result["output"]
