from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent import zet_agent_response_mode as response_mode
from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.session_context import clear_session_vars, clear_turn_vars, set_session_vars, set_turn_vars
from tools import terminal_tool as terminal_tool_module
from tools.environments.local import build_printer3d_runtime_env


ACTION_TOKEN = "a" * 64
HARDWARE_TOKEN = "b" * 64


@pytest.fixture(autouse=True)
def _reset_runtime_anchor(monkeypatch):
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(terminal_tool_module, "_connector_runtime_path_is_trusted", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(terminal_tool_module, "_ensure_sensitive_runtime_boundary", lambda: True)


def _write_runtime(tmp_path: Path, *, control: bool) -> Path:
    root = tmp_path / "presets"
    skill_id = "printer3d-control" if control else "printer3d"
    script_name = "printer3d_control.py" if control else "printer3d_connector.py"
    script = root / "skills" / skill_id / "scripts" / script_name
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent("""
        import json
        import os
        import sys

        def secret(name):
            descriptor = int(os.environ[name + "_FD"])
            return os.read(descriptor, 4096).decode()

        print(json.dumps({
            "argv": sys.argv[1:],
            "action": secret("ZETTLAB_AGENT_ACTION_TOKEN"),
            "action_plain": os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", ""),
        }))
    """).lstrip(), encoding="utf-8")
    manifest = root / "skills" / skill_id / "manifest.yaml"
    if control:
        scopes = "[hardware.printer3d:control, hardware.printer3d:job]"
        capability = "hardware.printer3d.control.v1"
    else:
        scopes = "[hardware.printer3d:read]"
        capability = "zettlab.printer3d.actions.v1"
    manifest.write_text(textwrap.dedent(f"""
        id: {skill_id}
        required_scopes: {scopes}
        runtime_capabilities: [{capability}]
    """).lstrip(), encoding="utf-8")
    return script


def _bind_receipt():
    secret_token = set_secret_scope({"ZET_AGENT_ID": "agent-1", "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN})
    session_tokens = set_session_vars(session_key="zettlab:owner-1:agent-1:stable", session_id="session-1")
    turn_tokens = set_turn_vars(
        turn_id="turn-1",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    turn_identity = response_mode._current_skill_direct_turn_identity()
    assert turn_identity is not None
    receipt = response_mode._capture_trusted_execution_receipt(
        turn_identity,
        "skills/printer3d-control/SKILL.md",
    )
    assert receipt is not None
    receipt_token = response_mode._TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(receipt)
    return secret_token, session_tokens, turn_tokens, receipt_token


def _clear_receipt(tokens) -> None:
    secret_token, session_tokens, turn_tokens, receipt_token = tokens
    response_mode._TRUSTED_HARDWARE_RUNTIME_RECEIPT.reset(receipt_token)
    clear_turn_vars(turn_tokens)
    clear_session_vars(session_tokens)
    reset_secret_scope(secret_token)


def test_printer3d_runtime_parser_accepts_only_fixed_argv(monkeypatch, tmp_path):
    _write_runtime(tmp_path, control=False)
    _write_runtime(tmp_path, control=True)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    accepted = [
        'python3 "$ZETTLAB_PRESETS_DIR/skills/printer3d/scripts/printer3d_connector.py" list',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/printer3d/scripts/printer3d_connector.py" status --printer-id printer-1',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/printer3d-control/scripts/printer3d_control.py" cancel --printer-id printer-1 --idempotency-key idem-1',
    ]
    rejected = [
        accepted[1] + " --host 192.168.1.2",
        accepted[2] + "; id",
        accepted[2].replace("cancel", "start"),
        accepted[2].replace("idem-1", "../../secret"),
    ]
    assert all(terminal_tool_module._parse_printer3d_runtime_command(command) is not None for command in accepted)
    assert all(terminal_tool_module._parse_printer3d_runtime_command(command) is None for command in rejected)

    read_args = {"command": accepted[1]}
    control_args = {"command": accepted[2]}
    assert response_mode._printer3d_command_policy(
        read_args,
        relative_path="skills/printer3d/SKILL.md",
    )
    assert not response_mode._printer3d_command_policy(
        control_args,
        relative_path="skills/printer3d/SKILL.md",
    )
    assert response_mode._printer3d_command_policy(
        control_args,
        relative_path="skills/printer3d-control/SKILL.md",
    )
    assert not response_mode._printer3d_command_policy(
        read_args,
        relative_path="skills/printer3d-control/SKILL.md",
    )


def test_printer3d_runtime_direct_runner_uses_private_fds(monkeypatch, tmp_path):
    _write_runtime(tmp_path, control=True)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        assert build_printer3d_runtime_env()["HERMES_SESSION_ID"] == "session-1"
        result = json.loads(terminal_tool_module._run_printer3d_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/printer3d-control/scripts/printer3d_control.py" pause --printer-id printer-1 --idempotency-key idem-1',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        _clear_receipt(tokens)
    assert result["printer3d_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert ACTION_TOKEN not in result["output"] and HARDWARE_TOKEN not in result["output"]
    payload = json.loads(result["output"])
    assert payload["argv"] == ["pause", "--printer-id", "printer-1", "--idempotency-key", "idem-1"]
    assert payload["action"] == "[REDACTED]"
    assert payload["action_plain"] == ""


def test_printer3d_runtime_rejects_untrusted_manifest(monkeypatch, tmp_path):
    _write_runtime(tmp_path, control=True)
    manifest = tmp_path / "presets" / "skills" / "printer3d-control" / "manifest.yaml"
    manifest.write_text(manifest.read_text().replace("hardware.printer3d.control.v1", "hardware.printer3d.control.v0"))
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        result = json.loads(terminal_tool_module._run_printer3d_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/printer3d-control/scripts/printer3d_control.py" pause --printer-id printer-1 --idempotency-key idem-1',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        _clear_receipt(tokens)
    assert result["exit_code"] == -1
    assert "capability" in result["error"].lower()
