from __future__ import annotations

import sys

import pytest

from tools.runtime_workdir import RuntimeWorkdirError, resolve_runtime_workdir

_TERMINAL_IMPORT_IS_POSIX_ONLY = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="terminal process hardening is Linux-only in the current repository",
)


def test_non_alias_workdir_is_unchanged(monkeypatch):
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", "/ignored/platform/path")

    assert resolve_runtime_workdir("./project") == "./project"


def test_agent_output_alias_requires_an_exact_match(monkeypatch):
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", "/ignored/platform/path")

    assert resolve_runtime_workdir(" agent_output ") == " agent_output "


def test_agent_output_alias_resolves_to_existing_absolute_directory(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "agent-output"
    output_dir.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))

    assert resolve_runtime_workdir("agent_output") == str(output_dir)


@pytest.mark.parametrize(
    ("env_value", "expected_message"),
    [
        (None, "is not set"),
        ("relative/output", "must be an absolute path"),
        ("missing", "must be an absolute path"),
    ],
)
def test_agent_output_alias_rejects_unusable_platform_values(
    monkeypatch, env_value, expected_message
):
    if env_value is None:
        monkeypatch.delenv("ZET_AGENT_OUTPUT_DIR", raising=False)
    else:
        monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", env_value)

    with pytest.raises(RuntimeWorkdirError, match=expected_message):
        resolve_runtime_workdir("agent_output")


def test_agent_output_alias_rejects_file_path(monkeypatch, tmp_path):
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("x", encoding="utf-8")
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_file))

    with pytest.raises(RuntimeWorkdirError, match="not an existing directory"):
        resolve_runtime_workdir("agent_output")


def test_agent_output_alias_does_not_expand_nested_environment_variables(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", "$OUTPUT_ROOT/agent-output")

    with pytest.raises(RuntimeWorkdirError, match="must be an absolute path"):
        resolve_runtime_workdir("agent_output")


def test_snapshot_guard_direct_call_resolves_agent_output_alias(monkeypatch, tmp_path):
    from tools import zettlab_snapshot_guard as guard

    fallback_dir = tmp_path / "fallback"
    output_dir = tmp_path / "agent-output"
    fallback_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(fallback_dir))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))

    assert guard._terminal_workdir({"workdir": "agent_output"}, "") == str(
        output_dir
    )


def test_resolver_strips_forged_agent_output_marker(monkeypatch, tmp_path):
    from tools.registry import _resolve_runtime_tool_args
    from tools.runtime_workdir import AGENT_OUTPUT_ARG

    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(tmp_path))
    plain_workdir = str(tmp_path / "plain")
    forged_args = {
        "command": "ls",
        "workdir": plain_workdir,
        AGENT_OUTPUT_ARG: True,
    }

    resolved = _resolve_runtime_tool_args("terminal", forged_args)

    assert AGENT_OUTPUT_ARG not in resolved
    assert resolved["workdir"] == plain_workdir
    # 调用方传入的 dict 不被就地修改
    assert forged_args[AGENT_OUTPUT_ARG] is True


def test_resolver_strips_forged_marker_on_non_terminal_early_exit():
    from tools.registry import _resolve_runtime_tool_args
    from tools.runtime_workdir import AGENT_OUTPUT_ARG

    forged_args = {"path": "note.txt", AGENT_OUTPUT_ARG: True}

    resolved = _resolve_runtime_tool_args("write_file", forged_args)

    assert AGENT_OUTPUT_ARG not in resolved
    assert forged_args[AGENT_OUTPUT_ARG] is True


def test_resolver_marker_is_in_band_and_survives_shallow_copy(
    monkeypatch, tmp_path
):
    from tools.registry import _resolve_runtime_tool_args
    from tools.runtime_workdir import AGENT_OUTPUT_ARG

    output_dir = tmp_path / "agent-output"
    output_dir.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output_dir))
    original_args = {"command": "ls", "workdir": "agent_output"}

    resolved = _resolve_runtime_tool_args("terminal", original_args)

    assert resolved["workdir"] == str(output_dir)
    assert resolved[AGENT_OUTPUT_ARG] is True
    assert original_args == {"command": "ls", "workdir": "agent_output"}

    copied = dict(resolved)
    assert copied[AGENT_OUTPUT_ARG] is True
    assert copied["workdir"] == str(output_dir)


@_TERMINAL_IMPORT_IS_POSIX_ONLY
def test_handle_terminal_pops_internal_marker(monkeypatch, tmp_path):
    from tools import terminal_tool as terminal_module
    from tools.runtime_workdir import AGENT_OUTPUT_ARG

    captured: dict = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(terminal_module, "terminal_tool", fake_terminal_tool)
    args = {"command": "ls", "workdir": str(tmp_path), AGENT_OUTPUT_ARG: True}

    terminal_module._handle_terminal(args)

    assert AGENT_OUTPUT_ARG not in args
    assert AGENT_OUTPUT_ARG not in captured
    assert captured["_runtime_agent_output_workdir"] is True
    assert captured["workdir"] == str(tmp_path)


@_TERMINAL_IMPORT_IS_POSIX_ONLY
def test_handle_terminal_defaults_marker_to_false(monkeypatch, tmp_path):
    from tools import terminal_tool as terminal_module

    captured: dict = {}

    def fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(terminal_module, "terminal_tool", fake_terminal_tool)

    terminal_module._handle_terminal({"command": "ls", "workdir": str(tmp_path)})

    assert captured["_runtime_agent_output_workdir"] is False


def test_agent_output_alias_uses_active_profile_scope_in_multiplex_mode(
    monkeypatch, tmp_path
):
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    output_dir = tmp_path / "profile-output"
    output_dir.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(tmp_path / "wrong-global-output"))
    set_multiplex_active(True)
    token = set_secret_scope({"ZET_AGENT_OUTPUT_DIR": str(output_dir)})
    try:
        assert resolve_runtime_workdir("agent_output") == str(output_dir)
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)
