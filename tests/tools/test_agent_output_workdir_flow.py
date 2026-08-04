from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

from tools import registry as registry_module
from tools.registry import ToolRegistry


def test_agent_output_alias_is_resolved_before_snapshot_gate_and_execution_flow(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "agent-output"
    output_dir.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))

    guarded_workdirs: list[str] = []

    def capture_gate(name, args, kwargs):
        if name == "terminal":
            guarded_workdirs.append(args.get("workdir"))
        return None

    monkeypatch.setattr(registry_module, "_zettlab_snapshot_gate", capture_gate)

    received_workdirs: list[str] = []

    def execute_handler(args, **_kwargs):
        received_workdirs.append(args["workdir"])
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('agent-output-marker.txt').write_text('ok')",
            ],
            cwd=args["workdir"],
            check=False,
            capture_output=True,
            text=True,
        )
        return json.dumps(
            {
                "exit_code": completed.returncode,
                "output": completed.stdout,
                "error": completed.stderr,
            }
        )

    registry = ToolRegistry()
    registry.register(
        name="terminal",
        toolset="terminal",
        schema={"name": "terminal", "parameters": {"type": "object"}},
        handler=execute_handler,
    )

    original_args = {"command": "write marker", "workdir": "agent_output"}
    result = json.loads(
        registry.dispatch(
            "terminal",
            original_args,
            task_id="agent-output-workdir-flow",
            turn_id="turn-agent-output-workdir-flow",
        )
    )

    assert result["exit_code"] == 0, result
    assert guarded_workdirs == [str(output_dir)]
    assert received_workdirs == [str(output_dir)]
    assert original_args["workdir"] == "agent_output"
    assert (output_dir / "agent-output-marker.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="terminal process hardening is Linux-only in the current repository",
)
def test_agent_output_alias_rejects_non_local_backend_before_execution(
    monkeypatch, tmp_path
):
    import model_tools
    from tools import terminal_tool

    output_dir = tmp_path / "agent-output"
    output_dir.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(registry_module, "_zettlab_snapshot_gate", lambda *_args: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "docker",
            "cwd": "/root",
            "timeout": 30,
            "docker_image": "unused",
        },
    )

    def fail_if_environment_starts(**_kwargs):
        raise AssertionError("non-local environment must not start")

    monkeypatch.setattr(terminal_tool, "_create_environment", fail_if_environment_starts)

    result = json.loads(
        model_tools.handle_function_call(
            "terminal",
            {"command": "pwd", "workdir": "agent_output"},
            task_id="agent-output-non-local-flow",
            turn_id="agent-output-non-local-flow-turn",
        )
    )

    assert result["error_type"] == "runtime_workdir"
    assert "local terminal backend" in result["error"]


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="terminal process hardening is Linux-only in the current repository",
)
def test_real_terminal_executes_in_agent_output_workdir_flow(monkeypatch, tmp_path):
    import model_tools
    from tools import terminal_tool

    output_dir = tmp_path / "agent-output"
    output_dir.mkdir()
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))

    task_id = "real-agent-output-workdir-flow"
    try:
        result = json.loads(
            model_tools.handle_function_call(
                "terminal",
                {
                    "command": (
                        f"{shlex.quote(sys.executable)} -c \"from pathlib import Path; "
                        "Path('real-terminal-marker.txt').write_text('ok')\""
                    ),
                    "workdir": "agent_output",
                },
                task_id=task_id,
                turn_id="real-agent-output-workdir-flow-turn",
            )
        )
    finally:
        terminal_tool.clear_session_cwd(task_id)
        terminal_tool.cleanup_all_environments()

    assert result["exit_code"] == 0, result
    assert (output_dir / "real-terminal-marker.txt").read_text(encoding="utf-8") == "ok"
