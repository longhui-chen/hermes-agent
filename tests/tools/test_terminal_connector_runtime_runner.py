import json
import os
import shlex
import sys
import textwrap
from io import StringIO

import pytest

from tools import terminal_tool as terminal_tool_module


@pytest.fixture(autouse=True)
def _reset_connector_runtime_root_anchor(monkeypatch):
    monkeypatch.setattr(
        terminal_tool_module,
        "_CONNECTOR_RUNTIME_ROOT_ANCHOR",
        None,
    )


def _write_connector_runtime(tmp_path, skill_id="linear"):
    script = tmp_path / "presets" / "skills" / skill_id / "scripts" / "connector_runtime.py"
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
        lambda path, presets_root, **kwargs: True,
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
    from tools import trusted_direct_runner

    script = _write_connector_runtime(tmp_path)
    script.chmod(0o600)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1/rpc")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return trusted_direct_runner.TrustedPythonResult(output="ok", returncode=0)

    monkeypatch.setattr(
        trusted_direct_runner,
        "run_trusted_python_script",
        fake_run,
    )

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        cwd=str(tmp_path),
        timeout=5,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["base_env"]
    assert "ZETTLAB_CONNECTORS_URL" not in captured["base_env"]
    assert captured["injected_env"]["ZETTLAB_CONNECTORS_AUTH_TOKEN"] == "runner-token"
    assert captured["injected_env"]["ZETTLAB_CONNECTORS_URL"] == "http://127.0.0.1/rpc"
    assert captured["argv"][0].endswith("connector_runtime.py")
    assert captured["script_bytes"] == script.read_bytes()


def test_connector_runtime_direct_runner_preserves_parent_process_globals(monkeypatch, tmp_path):
    """Worker env/path/cwd/stdout mutations cannot bleed into the gateway process."""
    _write_connector_runtime_with_global_mutation(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
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


@pytest.mark.live_system_guard_bypass
def test_connector_runtime_direct_runner_timeout_restores_control(monkeypatch, tmp_path):
    _write_connector_runtime_sleep(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
        cwd=str(tmp_path),
        timeout=1,
    ))

    assert result["connector_runtime_direct"] is True
    assert result["exit_code"] == 124, result
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
        lambda path, presets_root, **kwargs: True,
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
        lambda path, presets_root, **kwargs: True,
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


def test_compound_connector_runtime_command_flow_is_blocked_and_retried_singly(monkeypatch, tmp_path):
    """A compound runtime command never falls through to generic terminal."""
    from agent import secret_scope as ss

    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
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

    assert result["connector_runtime_direct"] is False
    assert result["connector_runtime_blocked"] is True
    assert result["exit_code"] == 2
    assert result["errorCode"] == "connector_runtime_compound_command"
    assert result["connector_error"]["nextAction"] == {"type": "retry_single_command"}
    assert "separate terminal tool call" in result["error"]
    assert "connector token ok" not in result["output"]
    assert "runner-token" not in result["output"]


@pytest.mark.parametrize(
    ("connector_kind", "skill_id", "runtime_args"),
    [
        ("first_party", "github", "list-tools --prefix github."),
        ("custom_mcp", "custom-connectors", "call custom_connector.list_tools --args-json '{}'"),
        ("custom_api", "custom-connectors", "call custom_connector.list_tools --args-json '{}'"),
    ],
)
@pytest.mark.parametrize(
    "command_template",
    [
        "timeout 30 {runtime}; true",
        "env FOO=1 {runtime} && true",
        "sudo -n {runtime} || true",
        "command {runtime}; true",
    ],
)
def test_wrapped_connector_runtime_flow_is_blocked_for_all_connector_kinds(
    monkeypatch,
    tmp_path,
    connector_kind,
    skill_id,
    runtime_args,
    command_template,
):
    """Every connector kind keeps runtime context out of wrapper shells."""
    from agent import secret_scope as ss

    _write_connector_runtime(tmp_path, skill_id)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    ss.set_multiplex_active(False)
    runtime = (
        f'python3 "$ZETTLAB_PRESETS_DIR/skills/{skill_id}/scripts/connector_runtime.py" '
        f"{runtime_args}"
    )

    result = json.loads(terminal_tool_module.terminal_tool(
        command_template.format(runtime=runtime),
        task_id=f"connector-runtime-wrapper-{connector_kind}",
    ))

    assert result["connector_runtime_blocked"] is True
    assert result["errorCode"] == "connector_runtime_compound_command"
    assert result["connector_error"]["nextAction"] == {"type": "retry_single_command"}
    assert "runner-token" not in json.dumps(result)


def test_single_wrapped_connector_runtime_command_is_blocked(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = json.loads(terminal_tool_module._run_connector_runtime_command_if_allowed(
        'timeout 30 python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        cwd=str(tmp_path),
        timeout=5,
    ))

    assert result["connector_runtime_blocked"] is True
    assert result["errorCode"] == "connector_runtime_compound_command"


@pytest.mark.parametrize(
    "command_template",
    [
        "FOO=1 {runtime}; true",
        "exec {runtime}; true",
        "nice -n 10 {runtime}; true",
        "nohup {runtime}; true",
        "setsid {runtime}; true",
        "stdbuf -o L {runtime}; true",
        "time {runtime}; true",
        "builtin {runtime}; true",
        "( {runtime} ); true",
        "python3 -u {script} list-tools; true",
    ],
)
def test_connector_runtime_shell_guard_blocks_similar_non_direct_shapes(
    monkeypatch,
    tmp_path,
    command_template,
):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    runtime = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" '
        "list-tools"
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        command_template.format(runtime=runtime, script=script)
    )

    assert result is not None
    assert json.loads(result)["connector_runtime_blocked"] is True


@pytest.mark.parametrize(
    ("connector_kind", "skill_id"),
    [
        ("first_party_linear", "linear"),
        ("first_party_github", "github"),
        ("custom_mcp", "custom-connectors"),
        ("custom_api", "custom-connectors"),
    ],
)
@pytest.mark.parametrize("shell_command", ["sh -c", "bash -lc"])
def test_nested_shell_connector_runtime_flow_is_blocked_for_all_connector_kinds(
    monkeypatch,
    tmp_path,
    connector_kind,
    skill_id,
    shell_command,
):
    _write_connector_runtime(tmp_path, skill_id)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    nested = (
        f'python3 "$ZETTLAB_PRESETS_DIR/skills/{skill_id}/scripts/connector_runtime.py" '
        "list-tools; true"
    )

    result = json.loads(terminal_tool_module.terminal_tool(
        f"{shell_command} {shlex.quote(nested)}",
        task_id=f"nested-shell-{connector_kind}",
    ))

    assert result["connector_runtime_blocked"] is True
    assert result["errorCode"] == "connector_runtime_compound_command"
    assert "runner-token" not in json.dumps(result)


def test_nested_shell_guard_ignores_runtime_path_used_as_command_data(monkeypatch):
    monkeypatch.setattr(
        terminal_tool_module,
        "_resolve_connector_runtime_script",
        lambda path: pytest.fail(f"data-only runtime path was resolved: {path}"),
    )
    nested = (
        'echo python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"; '
        "true"
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        f"bash -lc {shlex.quote(nested)}"
    )

    assert result is None


def test_nested_shell_guard_stops_parsing_options_after_double_dash(monkeypatch):
    monkeypatch.setattr(
        terminal_tool_module,
        "_resolve_connector_runtime_script",
        lambda path: pytest.fail(f"non-command runtime path was resolved: {path}"),
    )
    nested = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"; '
        "true"
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        f"bash -- -c {shlex.quote(nested)}"
    )

    assert result is None


def test_connector_runtime_flow_blocks_three_nested_command_shells(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    command = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" '
        "list-tools; true"
    )
    for shell_command in ("bash -lc", "sh -c", "dash -c"):
        command = f"{shell_command} {shlex.quote(command)}"

    result = json.loads(terminal_tool_module.terminal_tool(
        command,
        task_id="connector-runtime-three-nested-shells",
    ))

    assert result["connector_runtime_blocked"] is True
    assert result["errorCode"] == "connector_runtime_compound_command"


@pytest.mark.parametrize(
    ("background", "pty"),
    [(True, False), (False, True)],
)
def test_connector_runtime_flow_blocks_unsupported_execution_modes(
    monkeypatch,
    tmp_path,
    background,
    pty,
):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "runner-token")
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = json.loads(terminal_tool_module.terminal_tool(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools',
        background=background,
        pty=pty,
        task_id="connector-runtime-unsupported-mode",
    ))

    assert result["connector_runtime_blocked"] is True
    assert result["errorCode"] == "connector_runtime_compound_command"
    assert "foreground non-PTY" in result["error"]
    assert "runner-token" not in json.dumps(result)


def test_connector_runtime_shell_guard_ignores_runtime_path_used_as_command_data(monkeypatch):
    monkeypatch.setattr(
        terminal_tool_module,
        "_resolve_connector_runtime_script",
        lambda path: pytest.fail(f"data-only runtime path was resolved: {path}"),
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        'echo python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"; true'
    )

    assert result is None


def test_parser_rejects_non_presets_or_compound_connector_runtime(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    direct = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools'
    )
    unquoted_braced = terminal_tool_module._parse_connector_runtime_command(
        "python3 ${ZETTLAB_PRESETS_DIR}/skills/linear/scripts/connector_runtime.py list-tools"
    )
    quoted_braced = terminal_tool_module._parse_connector_runtime_command(
        'python3 "${ZETTLAB_PRESETS_DIR}/skills/linear/scripts/connector_runtime.py" list-tools'
    )
    relative = terminal_tool_module._parse_connector_runtime_command(
        "python3 skills/linear/scripts/connector_runtime.py list-tools"
    )
    absolute = terminal_tool_module._parse_connector_runtime_command(
        f"python3 {script} list-tools"
    )
    compound = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"; env'
    )
    newline_compound = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" list-tools\n'
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" call linear.list_issues'
    )
    quoted_punctuation = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" '
        "call linear.list_issues --args-json '{\"query\":\"a;b\"}'"
    )
    non_presets = terminal_tool_module._parse_connector_runtime_command(
        f"python3 {script.parent.parent / 'connector_runtime.py'}"
    )

    assert direct is not None
    assert unquoted_braced is not None
    assert quoted_braced is not None
    assert relative is not None
    assert absolute is not None
    assert compound is None
    assert newline_compound is None
    assert quoted_punctuation is not None
    assert non_presets is None


def test_shell_guard_blocks_wrapped_unquoted_braced_presets_path(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        "timeout 30 python3 ${ZETTLAB_PRESETS_DIR}/skills/linear/scripts/connector_runtime.py "
        "list-tools; true"
    )

    assert result is not None
    assert json.loads(result)["connector_runtime_blocked"] is True


def test_shell_guard_still_blocks_brace_grouped_runtime(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    result = terminal_tool_module._connector_runtime_shell_guard_result(
        '{ python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py" '
        "list-tools; true; }"
    )

    assert result is not None
    assert json.loads(result)["connector_runtime_blocked"] is True


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


def test_connector_runtime_trust_rejects_version_tree_modified_after_start(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(
        terminal_tool_module,
        "_CONNECTOR_RUNTIME_TRUST_CUTOFF",
        terminal_tool_module.time.time(),
    )
    future = terminal_tool_module._CONNECTOR_RUNTIME_TRUST_CUTOFF + 10
    os.utime(script, (future, future))

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        tmp_path / "presets",
    ) is False


def test_connector_runtime_trust_ignores_shared_ancestor_timestamp_changes(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    presets_root = tmp_path / "presets"
    writable_parent = tmp_path

    def fake_writable(path, *, enforce_cutoff=True):
        return path == writable_parent and enforce_cutoff

    monkeypatch.setattr(
        terminal_tool_module,
        "_path_writable_by_current_user",
        fake_writable,
    )

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        presets_root,
    ) is True


def test_connector_runtime_trust_rejects_symlink_inside_version_tree(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    real_script = script.with_name("real_connector_runtime.py")
    script.rename(real_script)
    script.symlink_to(real_script.name)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 424242, raising=False)
    monkeypatch.setattr(terminal_tool_module.os, "getegid", lambda: 424242, raising=False)
    monkeypatch.setattr(terminal_tool_module.os, "getgroups", lambda: [])

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        tmp_path / "presets",
    ) is False


def test_connector_runtime_pins_original_version_when_current_symlink_moves(monkeypatch, tmp_path):
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    _write_connector_runtime(v1)
    _write_connector_runtime(v2)
    current = tmp_path / "current"
    current.symlink_to(v1 / "presets", target_is_directory=True)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(current))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )

    first = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"'
    )
    current.unlink()
    current.symlink_to(v2 / "presets", target_is_directory=True)
    second = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"'
    )

    assert first is not None
    assert second is not None
    assert first.argv[1] == second.argv[1]
    assert str(v1 / "presets") in first.argv[1]


def test_connector_runtime_rejects_version_root_identity_replacement(monkeypatch, tmp_path):
    _write_connector_runtime(tmp_path)
    presets = tmp_path / "presets"
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets))
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    parsed = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"'
    )
    assert parsed is not None

    original = presets.with_name("presets-old")
    presets.rename(original)
    presets.mkdir()

    result = terminal_tool_module._run_connector_runtime_command_if_allowed(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"',
        cwd=str(tmp_path),
        timeout=5,
    )

    assert result is None


def test_connector_runtime_rejection_log_never_contains_token(monkeypatch, tmp_path, caplog):
    _write_connector_runtime(tmp_path)
    token = "canary-connector-token-must-not-appear"
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", token)

    parsed = terminal_tool_module._parse_connector_runtime_command(
        'python3 "$ZETTLAB_PRESETS_DIR/skills/linear/scripts/connector_runtime.py"'
    )

    assert parsed is None
    assert "owned_by_terminal_user" in caplog.text
    assert token not in caplog.text


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
