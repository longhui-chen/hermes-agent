from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agent import zet_agent_response_mode as response_mode
from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.session_context import clear_session_vars, clear_turn_vars, set_session_vars, set_turn_vars
from tools import terminal_tool as terminal_tool_module
from tools.environments.local import build_smart_home_runtime_env


ACTION_TOKEN = "a" * 64
HARDWARE_TOKEN = "b" * 64


@pytest.fixture(autouse=True)
def _reset_runtime_anchor(monkeypatch):
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(terminal_tool_module, "_connector_runtime_path_is_trusted", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(terminal_tool_module, "_ensure_sensitive_runtime_boundary", lambda: True)


def _write_runtime(tmp_path: Path) -> None:
    root = tmp_path / "presets"
    skill_root = root / "skills" / "smart-home-light-control"
    script = skill_root / "scripts" / "smart_home_light.py"
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(
        """
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
        """
    ).lstrip(), encoding="utf-8")
    (skill_root / "manifest.yaml").write_text(textwrap.dedent(
        """
        id: smart-home-light-control
        required_scopes: [hardware.smart_home:control]
        runtime_capabilities: [zettlab.smart_home.light.actions.v1]
        """
    ).lstrip(), encoding="utf-8")


def _bind_receipt():
    secret_token = set_secret_scope({"ZET_AGENT_ID": "agent-1", "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN})
    session_tokens = set_session_vars(session_key="zettlab:owner-1:agent-1:stable", session_id="session-1")
    turn_tokens = set_turn_vars(turn_id="turn-1", hardware_execution_token=HARDWARE_TOKEN)
    turn_identity = response_mode._current_skill_direct_turn_identity()
    assert turn_identity is not None
    receipt = response_mode._capture_trusted_execution_receipt(
        turn_identity,
        "skills/smart-home-light-control/SKILL.md",
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


def test_smart_home_runtime_parser_accepts_fixed_light_commands(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    prefix = 'python3 "$ZETTLAB_PRESETS_DIR/skills/smart-home-light-control/scripts/smart_home_light.py"'
    accepted = [
        prefix + " list",
        prefix + " power --target-id light-1 --on --idempotency-key idem-1",
        prefix + " brightness --target-id light-1 --percent 50 --idempotency-key idem-2",
        prefix + " color --target-id light-1 --red 1 --green 2 --blue 3 --idempotency-key idem-3",
        prefix + " color-temperature --target-id light-1 --kelvin 2700 --idempotency-key idem-4",
    ]
    rejected = [
        prefix + " power --target-id light-1 --on --idempotency-key idem-1 --host 127.0.0.1",
        prefix + " power --target-id light-1 --on --idempotency-key idem-1; id",
        prefix + " brightness --target-id light-1 --percent 101 --idempotency-key idem-2",
        prefix + " power --target-id light-1 --on --idempotency-key ../../secret",
    ]
    assert all(terminal_tool_module._parse_smart_home_runtime_command(command) is not None for command in accepted)
    assert all(terminal_tool_module._parse_smart_home_runtime_command(command) is None for command in rejected)
    assert response_mode._smart_home_command_policy({"command": accepted[1]})
    assert not response_mode._smart_home_command_policy({"command": accepted[1], "background": True})


def test_smart_home_runtime_direct_runner_uses_private_fds(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        assert build_smart_home_runtime_env()["HERMES_SESSION_ID"] == "session-1"
        result = json.loads(terminal_tool_module._run_smart_home_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/smart-home-light-control/scripts/smart_home_light.py" power --target-id light-1 --on --idempotency-key idem-1',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        _clear_receipt(tokens)
    assert result["smart_home_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert ACTION_TOKEN not in result["output"] and HARDWARE_TOKEN not in result["output"]
    payload = json.loads(result["output"])
    assert payload["argv"] == ["power", "--target-id", "light-1", "--on", "--idempotency-key", "idem-1"]
    assert payload["action"] == "[REDACTED]"
    assert payload["action_plain"] == ""


def test_smart_home_runtime_requires_attested_manifest(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    manifest = tmp_path / "presets" / "skills" / "smart-home-light-control" / "manifest.yaml"
    manifest.write_text(manifest.read_text().replace("zettlab.smart_home.light.actions.v1", "zettlab.smart_home.light.actions.v0"))
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        result = json.loads(terminal_tool_module._run_smart_home_runtime_command_if_allowed(
            'python3 "$ZETTLAB_PRESETS_DIR/skills/smart-home-light-control/scripts/smart_home_light.py" list',
            cwd=str(tmp_path),
            timeout=5,
        ))
    finally:
        _clear_receipt(tokens)
    assert result["exit_code"] == -1
    assert "capability" in result["error"].lower()
