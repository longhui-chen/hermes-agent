from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import zet_agent_response_mode as response_mode
from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.session_context import clear_session_vars, clear_turn_vars, set_session_vars, set_turn_vars
from tools import terminal_tool as terminal_tool_module
from tools.environments.local import build_plaud_runtime_env


ACTION_TOKEN = "a" * 64
HARDWARE_TOKEN = "b" * 64


@pytest.fixture(autouse=True)
def _reset_runtime_anchor(monkeypatch):
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_sensitive_runtime_boundary",
        lambda: True,
    )


def _write_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    script = root / "skills" / "plaud-recordings" / "scripts" / "plaud_connector.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        textwrap.dedent(
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
        ).lstrip(),
        encoding="utf-8",
    )
    manifest = root / "skills" / "plaud-recordings" / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
            id: plaud-recordings
            required_scopes: [hardware.plaud:read]
            runtime_capabilities: [zettlab.plaud.actions.v1]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return script


def _bind_receipt():
    secret_token = set_secret_scope(
        {"ZET_AGENT_ID": "agent-1", "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN}
    )
    session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:stable",
        session_id="session-1",
    )
    turn_tokens = set_turn_vars(
        turn_id="turn-1",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    turn_identity = response_mode._current_skill_direct_turn_identity()
    assert turn_identity is not None
    receipt = response_mode._capture_trusted_execution_receipt(
        turn_identity,
        "skills/plaud-recordings/SKILL.md",
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


def test_plaud_runtime_parser_accepts_only_fixed_read_argv(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    prefix = 'python3 "$ZETTLAB_PRESETS_DIR/skills/plaud-recordings/scripts/plaud_connector.py"'
    accepted = [
        prefix + " list",
        prefix + " list --page 2 --page-size 20",
        prefix + ' search "planning notes" --from 2026-08-01 --to 2026-08-24 --max 50',
        prefix + " transcript --file-id recording_1",
        prefix + " note --file-id recording-2",
    ]
    rejected = [
        prefix + " audio --file-id recording_1",
        prefix + " transcript --file-id ../../tokens.json",
        prefix + " note --file-id recording_1 --output /tmp/note",
        prefix + " list --page 1 --page 2",
        prefix + " list --page 1 --page-size 2",
        prefix + " search planning --from 2026-02-31",
        prefix + " list; id",
        prefix + " list --account-id account_1",
        prefix + " list --url https://api.plaud.ai",
    ]
    assert all(
        terminal_tool_module._parse_plaud_runtime_command(command) is not None
        for command in accepted
    )
    assert all(
        terminal_tool_module._parse_plaud_runtime_command(command) is None
        for command in rejected
    )
    assert response_mode._plaud_command_policy({"command": accepted[2]})
    assert not response_mode._plaud_command_policy(
        {"command": accepted[0], "background": True}
    )
    blocked = json.loads(
        terminal_tool_module._plaud_runtime_shell_guard_result(
            prefix + " list --page 1 --page-size 2"
        )
    )
    assert "--page-size must be between 10 and 100" in blocked["error"]


def test_plaud_runtime_direct_runner_uses_private_fds(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        assert build_plaud_runtime_env()["HERMES_SESSION_ID"] == "session-1"
        result = json.loads(
            terminal_tool_module._run_plaud_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/plaud-recordings/scripts/plaud_connector.py" list --page 1 --page-size 20',
                cwd=str(tmp_path),
                timeout=5,
            )
        )
    finally:
        _clear_receipt(tokens)
    assert result["plaud_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert ACTION_TOKEN not in result["output"]
    assert HARDWARE_TOKEN not in result["output"]
    payload = json.loads(result["output"])
    assert payload["argv"] == ["list", "--page", "1", "--page-size", "20"]
    assert payload["action"] == "[REDACTED]"
    assert payload["action_plain"] == ""


def test_plaud_runtime_rejects_untrusted_manifest(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    manifest = tmp_path / "presets" / "skills" / "plaud-recordings" / "manifest.yaml"
    manifest.write_text(
        manifest.read_text().replace(
            "zettlab.plaud.actions.v1",
            "zettlab.plaud.actions.v0",
        )
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        result = json.loads(
            terminal_tool_module._run_plaud_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/plaud-recordings/scripts/plaud_connector.py" list',
                cwd=str(tmp_path),
                timeout=5,
            )
        )
    finally:
        _clear_receipt(tokens)
    assert result["exit_code"] == -1
    assert "capability" in result["error"].lower()


def test_plaud_trusted_scope_binds_exact_command_to_current_turn(monkeypatch, tmp_path):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    secret_token = set_secret_scope(
        {"ZET_AGENT_ID": "agent-1", "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN}
    )
    session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:stable",
        session_id="session-1",
    )
    turn_tokens = set_turn_vars(
        turn_id="turn-1",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    try:
        agent = SimpleNamespace(
            platform="zet_agent",
            _zet_agent_execution_policy="",
        )
        response_mode.reset_trusted_skill_execution(
            agent,
            "列出我的 PLAUD 录音",
            explicit_skill_slug="plaud-recordings",
        )
        turn_identity = response_mode._current_skill_direct_turn_identity()
        assert turn_identity is not None
        assert response_mode._activate_trusted_skill_scope(
            agent,
            relative_path="skills/plaud-recordings/SKILL.md",
            attested_turn_identity=turn_identity,
        )
        assert response_mode.trusted_skill_allowed_tool_names(agent) == {"terminal"}
        blocked = response_mode.trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args={"command": "python3 plaud_connector.py audio --file-id recording_1"},
        )
        assert blocked is not None

        assert response_mode._activate_trusted_skill_scope(
            agent,
            relative_path="skills/plaud-recordings/SKILL.md",
            attested_turn_identity=turn_identity,
        )
        exact_args = {
            "command": 'python3 "$ZETTLAB_PRESETS_DIR/skills/plaud-recordings/scripts/plaud_connector.py" list'
        }
        assert response_mode.trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args=exact_args,
        ) is None
        receipt, dispatch_error = response_mode._claim_trusted_terminal_dispatch(
            agent,
            exact_args,
        )
        assert dispatch_error is None
        assert receipt is not None
        assert receipt.action_token == ACTION_TOKEN
        assert not hasattr(receipt, "hardware_execution_token")
    finally:
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
        reset_secret_scope(secret_token)


def test_plaud_retry_continuation_requires_recent_attested_same_session(
    monkeypatch,
    tmp_path,
):
    _write_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    response_mode._PLAUD_RESUME_SESSIONS.clear()
    secret_token = set_secret_scope(
        {"ZET_AGENT_ID": "agent-1", "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN}
    )
    session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:plaud-retry",
        session_id="session-plaud-retry",
    )
    first_turn = set_turn_vars(
        turn_id="turn-plaud-source",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    agent = SimpleNamespace(platform="zet_agent", _zet_agent_execution_policy="")
    try:
        response_mode.reset_trusted_skill_execution(agent, "读取我的 PLAUD 录音")
        turn_identity = response_mode._current_skill_direct_turn_identity()
        assert turn_identity is not None
        assert response_mode._activate_trusted_skill_scope(
            agent,
            relative_path="skills/plaud-recordings/SKILL.md",
            attested_turn_identity=turn_identity,
        )
    finally:
        clear_turn_vars(first_turn)

    retry_turn = set_turn_vars(
        turn_id="turn-plaud-retry",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    try:
        response_mode.reset_trusted_skill_execution(agent, "再次读取下")
        assert agent._zet_agent_skill_direct_task.plaud_applicable
        assert response_mode._trusted_skill_view_refresh_required(
            agent,
            {"name": "plaud-recordings"},
        )
    finally:
        clear_turn_vars(retry_turn)
        clear_session_vars(session_tokens)
        reset_secret_scope(secret_token)

    other_session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:other-session",
        session_id="session-other",
    )
    other_turn = set_turn_vars(
        turn_id="turn-other",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    try:
        response_mode.reset_trusted_skill_execution(agent, "再次读取下")
        assert not agent._zet_agent_skill_direct_task.plaud_applicable
    finally:
        clear_turn_vars(other_turn)
        clear_session_vars(other_session_tokens)
        response_mode._PLAUD_RESUME_SESSIONS.clear()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("获取下 How to use Plaud 的内容", True),
        ("获取下 How to use 的内容", False),
    ],
)
def test_plaud_named_content_intent_requires_explicit_plaud(message, expected):
    task = response_mode._skill_direct_task_context(SimpleNamespace(), message)

    assert task.plaud_applicable is expected


def test_plaud_named_content_followup_forces_fresh_skill_read():
    session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:plaud-content",
        session_id="session-plaud-content",
    )
    turn_tokens = set_turn_vars(
        turn_id="turn-plaud-content",
        hardware_execution_token=HARDWARE_TOKEN,
    )
    agent = SimpleNamespace(platform="zet_agent", _zet_agent_execution_policy="")
    try:
        response_mode.reset_trusted_skill_execution(
            agent,
            "获取下 How to use Plaud 的内容",
        )

        def _dispatch():
            assert response_mode.trusted_skill_view_fresh_read_required()
            return '{"success": true}'

        result = response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="skill_view",
            function_args={"name": "plaud-recordings"},
            dispatch=_dispatch,
        )

        assert result == '{"success": true}'
        assert not response_mode.trusted_skill_view_fresh_read_required()
    finally:
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
