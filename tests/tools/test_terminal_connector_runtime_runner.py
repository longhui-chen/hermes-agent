import json
import sys
import textwrap

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


def test_connector_runtime_direct_runner_receives_profile_scoped_env(monkeypatch, tmp_path):
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


def test_connector_runtime_trust_rejects_root_service_user(monkeypatch, tmp_path):
    script = _write_connector_runtime(tmp_path)
    monkeypatch.setattr(terminal_tool_module.os, "geteuid", lambda: 0, raising=False)

    assert terminal_tool_module._connector_runtime_path_is_trusted(
        script,
        tmp_path / "presets",
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
