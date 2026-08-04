from __future__ import annotations

import pytest

from tools.runtime_workdir import RuntimeWorkdirError, resolve_runtime_workdir


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
