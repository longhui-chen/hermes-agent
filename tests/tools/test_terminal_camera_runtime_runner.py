from __future__ import annotations

import hashlib
import json
import textwrap
import types
from pathlib import Path

import pytest

from agent import zet_agent_response_mode as response_mode
from agent.secret_scope import reset_secret_scope, set_secret_scope
from gateway.session_context import (
    clear_session_vars,
    clear_turn_vars,
    set_session_vars,
    set_turn_vars,
)
from tools import terminal_tool as terminal_tool_module
from tools import zettlab_snapshot_guard
from tools.environments.local import build_camera_runtime_env


ACTION_TOKEN = "a" * 64
BUSINESS_TOKEN = "b" * 64


@pytest.fixture(autouse=True)
def _reset_runtime_anchor(monkeypatch):
    monkeypatch.setattr(terminal_tool_module, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    monkeypatch.setattr(
        terminal_tool_module,
        "_connector_runtime_path_is_trusted",
        lambda path, presets_root, **kwargs: True,
    )
    monkeypatch.setattr(
        terminal_tool_module,
        "_ensure_sensitive_runtime_boundary",
        lambda: True,
    )


def _write_camera_runtime(tmp_path: Path) -> Path:
    root = tmp_path / "presets"
    script = root / "skills" / "camsnap" / "scripts" / "camera_connector.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        textwrap.dedent(
            """
        import json
        import os
        import sys

        def secret(name):
            descriptor = int(os.environ[name + "_FD"])
            return os.read(descriptor, 4096).decode("utf-8")

        print(json.dumps({
            "argv": sys.argv[1:],
            "agent": os.environ.get("ZET_AGENT_ID", ""),
            "turn": os.environ.get("HERMES_TURN_ID", ""),
            "session_id": os.environ.get("HERMES_SESSION_ID", ""),
            "session_key": os.environ.get("HERMES_SESSION_KEY", ""),
            "action": secret("ZETTLAB_AGENT_ACTION_TOKEN"),
            "business": secret("ZETTLAB_BUSINESS_EXECUTION_TOKEN"),
            "action_plain": os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", ""),
            "business_plain": os.environ.get("ZETTLAB_BUSINESS_EXECUTION_TOKEN", ""),
        }, sort_keys=True))
        """
        ).lstrip(),
        encoding="utf-8",
    )
    manifest = root / "skills" / "camsnap" / "manifest.yaml"
    manifest.write_text(
        textwrap.dedent(
            """
        id: camsnap
        version: 0.1.0
        name: Camera
        description: test
        estimated_rss_mb: 128
        required_scopes: [hardware.camera:read]
        fs_allowlist: []
        shell_allowlist: []
        runtime_capabilities: [zettlab.camera.actions.v1]
        external_deps: []
        worst_case: test
        """
        ).lstrip(),
        encoding="utf-8",
    )
    return script


def _bind_receipt():
    secret_token = set_secret_scope({
        "ZET_AGENT_ID": "agent-1",
        "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN,
    })
    session_tokens = set_session_vars(
        session_key="zettlab:owner-1:agent-1:stable-session",
        session_id="api-lineage-session-1",
    )
    turn_tokens = set_turn_vars(
        turn_id="turn-1",
        hardware_execution_token=BUSINESS_TOKEN,
        business_execution_action="c" * 64,
        business_execution_action_version="1",
    )
    turn_identity = response_mode._current_skill_direct_turn_identity()
    assert turn_identity is not None
    receipt = response_mode._capture_trusted_execution_receipt(
        turn_identity,
        response_mode._CAMERA_SKILL_PATH,
    )
    assert receipt is not None
    receipt_token = response_mode._TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.set(receipt)
    return secret_token, session_tokens, turn_tokens, receipt_token


def _clear_receipt(tokens) -> None:
    secret_token, session_tokens, turn_tokens, receipt_token = tokens
    response_mode._TRUSTED_VIDEO_EDIT_RUNTIME_RECEIPT.reset(receipt_token)
    clear_turn_vars(turn_tokens)
    clear_session_vars(session_tokens)
    reset_secret_scope(secret_token)


def test_camera_runtime_parser_accepts_only_fixed_actions(monkeypatch, tmp_path):
    _write_camera_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))

    accepted = [
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" list',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" snap --camera-id cam_front',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" doctor --camera-id cam_front',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" clip --camera-id cam_front --duration 60',
    ]
    rejected = [
        accepted[1] + " --host 192.168.1.2",
        accepted[1] + " --password secret",
        accepted[1] + " --out /tmp/frame.jpg",
        accepted[1] + "; id",
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" watch --camera-id cam_front',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" clip --camera-id cam_front --duration 61',
        'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" snap --camera-id ../../secret',
    ]

    assert all(
        terminal_tool_module._parse_camera_runtime_command(command) is not None
        for command in accepted
    )
    assert all(
        terminal_tool_module._parse_camera_runtime_command(command) is None
        for command in rejected
    )


def test_camera_runtime_flow_bypasses_generic_cwd_snapshot(monkeypatch, tmp_path):
    _write_camera_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:19090/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", ACTION_TOKEN)
    monkeypatch.setenv("TERMINAL_CWD", "/root")
    monkeypatch.setattr(
        zettlab_snapshot_guard,
        "_OPENER",
        types.SimpleNamespace(
            open=lambda *_args, **_kwargs: pytest.fail(
                "trusted camera action must not request a cwd snapshot"
            )
        ),
    )
    zettlab_snapshot_guard.reset_for_test()
    try:
        result = zettlab_snapshot_guard.maybe_require_snapshot(
            "terminal",
            {
                "command": (
                    'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/'
                    'camera_connector.py" list'
                )
            },
            turn_id="turn-1",
        )
    finally:
        zettlab_snapshot_guard.reset_for_test()

    assert result is None


def test_camera_runtime_env_is_request_and_profile_scoped():
    tokens = _bind_receipt()
    try:
        assert build_camera_runtime_env() == {
            "ZET_AGENT_ID": "agent-1",
            "ZETTLAB_AGENT_ACTION_TOKEN": ACTION_TOKEN,
            "ZETTLAB_BUSINESS_EXECUTION_TOKEN": BUSINESS_TOKEN,
            "HERMES_TURN_ID": "turn-1",
            "HERMES_SESSION_ID": "api-lineage-session-1",
            "HERMES_SESSION_KEY": "api-lineage-session-1",
        }
    finally:
        _clear_receipt(tokens)


def test_camera_runtime_direct_runner_flow_uses_secret_fds(monkeypatch, tmp_path):
    script = _write_camera_runtime(tmp_path)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        result = json.loads(
            terminal_tool_module._run_camera_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" snap --camera-id cam_front',
                cwd=str(tmp_path),
                timeout=5,
            )
        )
    finally:
        _clear_receipt(tokens)

    assert result["camera_runtime_direct"] is True
    assert result["exit_code"] == 0
    assert ACTION_TOKEN not in result["output"]
    assert BUSINESS_TOKEN not in result["output"]
    payload = json.loads(result["output"])
    assert payload["argv"] == ["snap", "--camera-id", "cam_front"]
    assert payload["agent"] == "agent-1"
    assert payload["turn"] == "turn-1"
    assert payload["session_id"] == "api-lineage-session-1"
    assert payload["session_key"] == "api-lineage-session-1"
    assert payload["action"] == "[REDACTED]"
    assert payload["business"] == "[REDACTED]"
    assert payload["action_plain"] == ""
    assert payload["business_plain"] == ""
    assert (
        hashlib.sha256(script.read_bytes()).hexdigest()
        == (
            terminal_tool_module._CONNECTOR_RUNTIME_ROOT_ANCHOR.file_digests[
                "skills/camsnap/scripts/camera_connector.py"
            ]
        )
    )


def test_camera_runtime_flow_fails_closed_without_capability(monkeypatch, tmp_path):
    _write_camera_runtime(tmp_path)
    manifest = tmp_path / "presets" / "skills" / "camsnap" / "manifest.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "zettlab.camera.actions.v1",
            "zettlab.camera.actions.v0",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    tokens = _bind_receipt()
    try:
        result = json.loads(
            terminal_tool_module._run_camera_runtime_command_if_allowed(
                'python3 "$ZETTLAB_PRESETS_DIR/skills/camsnap/scripts/camera_connector.py" list',
                cwd=str(tmp_path),
                timeout=5,
            )
        )
    finally:
        _clear_receipt(tokens)
    assert result["camera_runtime_direct"] is True
    assert result["exit_code"] == -1
    assert "capability" in result["error"].lower()
