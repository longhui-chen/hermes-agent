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


def test_literal_default_stays_default():
    assert terminal_tool._resolve_container_task_id("default") == "default"


def test_subagent_task_id_collapses_to_default():
    # delegate_task constructs IDs like "subagent-<N>-<uuid_hex>"; these
    # should share the parent's container, not spin up their own.
    assert terminal_tool._resolve_container_task_id("subagent-0-deadbeef") == "default"
    assert terminal_tool._resolve_container_task_id("subagent-42-cafef00d") == "default"


def test_arbitrary_session_id_collapses_to_default():
    # Session UUIDs or anything else without an override still collapse.
    assert terminal_tool._resolve_container_task_id("sess-123e4567-e89b-12d3") == "default"


def test_rl_task_with_override_keeps_its_own_id():
    # RL / benchmark pattern: register a per-task image, then the task_id
    # must survive ``_resolve_container_task_id`` so the rollout lands in
    # its own sandbox.
    terminal_tool.register_task_env_overrides(
        "tb2-task-fix-git", {"docker_image": "tb2:fix-git", "cwd": "/app"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("tb2-task-fix-git")
            == "tb2-task-fix-git"
        )
    finally:
        terminal_tool.clear_task_env_overrides("tb2-task-fix-git")


def test_cleared_override_collapses_again():
    terminal_tool.register_task_env_overrides("tb2-x", {"docker_image": "x:y"})
    assert terminal_tool._resolve_container_task_id("tb2-x") == "tb2-x"
    terminal_tool.clear_task_env_overrides("tb2-x")
    assert terminal_tool._resolve_container_task_id("tb2-x") == "default"


def test_get_active_env_reads_shared_container_from_subagent_id():
    """``get_active_env`` must see the shared ``"default"`` sandbox when
    called with a subagent's task_id, so the agent loop's turn-budget
    enforcement reads the real env (not None) during delegation."""
    sentinel = object()
    terminal_tool._active_environments["default"] = sentinel
    try:
        assert terminal_tool.get_active_env("subagent-7-cafe") is sentinel
        assert terminal_tool.get_active_env(None) is sentinel
        assert terminal_tool.get_active_env("default") is sentinel
    finally:
        terminal_tool._active_environments.pop("default", None)


def test_get_active_env_honours_rl_override():
    rl_env = object()
    default_env = object()
    terminal_tool._active_environments["default"] = default_env
    terminal_tool._active_environments["rl-42"] = rl_env
    terminal_tool.register_task_env_overrides("rl-42", {"docker_image": "x"})
    try:
        # With an override registered, lookup returns the task's own env,
        # not the shared "default" one.
        assert terminal_tool.get_active_env("rl-42") is rl_env
    finally:
        terminal_tool.clear_task_env_overrides("rl-42")
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._active_environments.pop("rl-42", None)


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


def test_cwd_plus_docker_image_keeps_own_id():
    """When overrides include both cwd AND docker_image, isolation must
    still be honoured (RL/benchmark pattern with explicit cwd)."""
    terminal_tool.register_task_env_overrides(
        "rl-with-cwd", {"docker_image": "myimg:latest", "cwd": "/workspace"}
    )
    try:
        assert (
            terminal_tool._resolve_container_task_id("rl-with-cwd")
            == "rl-with-cwd"
        )
    finally:
        terminal_tool.clear_task_env_overrides("rl-with-cwd")


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
