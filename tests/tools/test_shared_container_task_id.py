"""
Regression tests for the shared-container task_id mapping.

The top-level agent and all delegate_task subagents share a single
terminal sandbox keyed by ``"default"``.  ``_resolve_container_task_id``
is the sole gatekeeper for which tool-call task_ids go to the shared
container vs. get their own isolated sandbox.  RL / benchmark
environments opt in to isolation by calling
``register_task_env_overrides(task_id, {...})`` before the agent loop;
every other task_id collapses back to ``"default"``.

If you change the collapse logic, update both the helper and these
tests -- see `hermes-agent-dev` skill, "Why do subagents get their own
containers?" section, and the Container lifecycle paragraph under
Docker Backend in ``website/docs/user-guide/configuration.md``.
"""

from types import SimpleNamespace

import pytest

from tools import terminal_tool


@pytest.fixture(autouse=True)
def _clean_overrides():
    """Ensure no stray overrides from other tests leak in."""
    before = dict(terminal_tool._task_env_overrides)
    terminal_tool._task_env_overrides.clear()
    yield
    terminal_tool._task_env_overrides.clear()
    terminal_tool._task_env_overrides.update(before)


def test_none_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id(None) == "default"


def test_empty_task_id_maps_to_default():
    assert terminal_tool._resolve_container_task_id("") == "default"


def test_cwd_only_override_collapses_to_default():
    """CWD-only overrides (ACP adapter workspace tracking) must NOT trigger
    container isolation — they should collapse to the shared 'default'
    container so all surfaces (TUI, gateway, dashboard) share one sandbox.
    Regression for #37361."""
    terminal_tool.register_task_env_overrides(
        "acp-session-abc", {"cwd": "/home/user/project"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("acp-session-abc")
            == "default"
        )
    finally:
        terminal_tool.clear_task_env_overrides("acp-session-abc")


def test_env_type_override_keeps_own_id():
    """env_type is an isolation key — must trigger per-task container."""
    terminal_tool.register_task_env_overrides(
        "bench-env", {"env_type": "sandbox", "cwd": "/work"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("bench-env")
            == "bench-env"
        )
    finally:
        terminal_tool.clear_task_env_overrides("bench-env")


def test_managed_profiles_isolate_same_session_terminal_state(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    alpha = tmp_path / "profiles" / "alpha"
    beta = tmp_path / "profiles" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir()
    cleaned = []

    def install(profile, path_value, cwd):
        token = set_hermes_home_override(profile)
        try:
            key = terminal_tool._resolve_container_task_id("default")
            terminal_tool.record_session_cwd("default", cwd)
            terminal_tool.register_task_env_overrides("default", {"cwd": cwd})
            terminal_tool._active_environments[key] = SimpleNamespace(
                env={"PATH": path_value},
                cwd=cwd,
                cleanup=lambda: cleaned.append(str(profile.resolve())),
            )
            terminal_tool._last_activity[key] = 1.0
            return key
        finally:
            reset_hermes_home_override(token)

    alpha_key = install(alpha, "/tmp/alpha:/usr/bin", "/tmp/alpha")
    beta_token = set_hermes_home_override(beta)
    try:
        beta_key = terminal_tool._resolve_container_task_id("default")
        assert beta_key != alpha_key
        assert terminal_tool.get_active_env("default") is None
        assert terminal_tool.get_session_cwd("default") is None
        assert terminal_tool.resolve_task_overrides("default") == {}
    finally:
        reset_hermes_home_override(beta_token)

    beta_key = install(beta, "/tmp/beta:/usr/bin", "/tmp/beta")
    alpha_token = set_hermes_home_override(alpha)
    try:
        assert terminal_tool.get_active_env("default").env["PATH"].startswith(
            "/tmp/alpha:"
        )
        assert terminal_tool.get_session_cwd("default") == "/tmp/alpha"
    finally:
        reset_hermes_home_override(alpha_token)

    assert terminal_tool.cleanup_managed_profile_environments(alpha) == 1
    assert alpha_key not in terminal_tool._active_environments
    assert beta_key in terminal_tool._active_environments
    assert cleaned == [str(alpha.resolve())]
    terminal_tool.cleanup_managed_profile_environments(beta)
