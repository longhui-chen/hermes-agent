"""Cross-session HERMES_SESSION_* leak guard for the local terminal backend.

Regression coverage for the bug where a terminal subprocess could observe a
*different concurrent session's* ``HERMES_SESSION_KEY`` (and the other
``HERMES_SESSION_*`` vars).

Root cause: the session vars have a process-global ``os.environ`` mirror (written
last-writer-wins as a CLI/cron fallback, never cleared), while the
concurrency-safe source of truth is a task-local ``ContextVar``. The subprocess
env was built from ``os.environ`` and only *overrode* the session vars when the
ContextVar was set+truthy. When the subprocess was spawned from a thread/context
that never inherited the agent's copied context (ContextVar ``_UNSET``), the
override no-op'd and the stale, foreign ``os.environ`` value leaked into the
child — so e.g. ``bug_thread.py whoami`` read another session's thread id.

The fix: once the session-context machinery is engaged in this process (any
concurrent host — gateway, ACP, API server, TUI, cron — has called
``set_session_vars``), the session vars are ContextVar-authoritative. The
subprocess-env bridge resolves each ``HERMES_SESSION_*`` from the ContextVar and,
when it is ``_UNSET``, STRIPS the var from the child env rather than inheriting
the process-global value that may belong to another session. A pure
single-process CLI/one-shot that never engaged the session-context system keeps
the ``os.environ`` fallback.
"""

import os

import pytest

import gateway.session_context as sc
from gateway.session_context import (
    _VAR_MAP,
    clear_session_vars,
    clear_turn_vars,
    set_session_vars,
    set_zettlab_connector_route_capability,
    set_turn_vars,
)
from tools.code_execution_tool import _inject_execute_code_session_context_env, _scrub_child_env
from tools.environments import local as local_env_module
from tools.environments.local import (
    LocalEnvironment,
    PROFILE_SCOPED_SUBPROCESS_ENV_KEYS,
    RETIRED_VIDEO_EXECUTION_ENV_KEYS,
    _make_run_env,
    _sanitize_subprocess_env,
    build_connector_runtime_env,
    hermes_subprocess_env,
)

# The full set of session vars the bridge owns.
SESSION_VARS = list(_VAR_MAP.keys())


def test_retired_video_execution_environment_is_scrubbed_without_touching_generic_action_token():
    assert {
        "ZETTLAB_BUSINESS_EXECUTION_GRANT_VERSION",
        "ZETTLAB_BUSINESS_EXECUTION_MODE",
        "ZETTLAB_EXECUTION_REQUEST_DIGEST",
    }.issubset(RETIRED_VIDEO_EXECUTION_ENV_KEYS)
    env = {key: "stale" for key in RETIRED_VIDEO_EXECUTION_ENV_KEYS}
    env["ZETTLAB_AGENT_ACTION_TOKEN"] = "unrelated-skill-token"
    env["PATH"] = "/usr/bin"

    result = local_env_module._sanitize_subprocess_env(env)

    for key in RETIRED_VIDEO_EXECUTION_ENV_KEYS:
        assert key not in result
    assert result.get("ZETTLAB_AGENT_ACTION_TOKEN") != "unrelated-skill-token"
    assert result["PATH"] == "/usr/bin"


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + os.environ + engaged-latch slate per test, restored."""
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(sc._UNSET)
    sc._session_context_engaged = False
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _engage():
    """Mark the session-context machinery engaged, like a concurrent host would."""
    sc._session_context_engaged = True


@pytest.fixture(autouse=True)
def _isolate_secret_scope():
    """Clean multiplex/profile secret scope around env-bridge tests."""
    from agent import secret_scope as ss

    saved_active = ss.is_multiplex_active()
    token = ss.set_secret_scope(None)
    ss.set_multiplex_active(False)
    try:
        yield
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(saved_active)


# --------------------------------------------------------------------------- #
# Foreground path (_make_run_env)
# --------------------------------------------------------------------------- #

def test_engaged_unset_contextvar_strips_foreign_session_key(monkeypatch):
    """Engaged host + UNSET ContextVar must NOT inherit a foreign global.

    This is the production hijack: a concurrent session wrote
    os.environ["HERMES_SESSION_KEY"], this task's ContextVar is unset, and the
    subprocess must see NO key rather than the foreign one.
    """
    _engage()
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN_CONCURRENT:FOREIGN_CONCURRENT",
    )

    env = _make_run_env({})

    assert "HERMES_SESSION_KEY" not in env, (
        "Foreign concurrent session key leaked into subprocess env: "
        f"{env.get('HERMES_SESSION_KEY')!r}"
    )


def test_set_session_vars_engages_and_overrides_foreign_global(monkeypatch):
    """set_session_vars itself engages the latch and the bound value wins.

    Mirrors a real host: calling set_session_vars both marks the process engaged
    and binds the ContextVar, so the bound value overrides the foreign global.
    """
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN:FOREIGN",
    )

    tokens = set_session_vars(
        session_key="agent:main:discord:group:MY_BUGS_ROOT:111",
        platform="discord",
        chat_id="MY_BUGS_ROOT",
    )
    try:
        assert sc.session_context_engaged() is True
        env = _make_run_env({})
    finally:
        clear_session_vars(tokens)

    assert env.get("HERMES_SESSION_KEY") == "agent:main:discord:group:MY_BUGS_ROOT:111"


def test_unengaged_process_preserves_os_environ_fallback(monkeypatch):
    """A process that never engaged the session-context system keeps the fallback.

    Pure single-process CLI/one-shot sets HERMES_SESSION_* directly in os.environ
    and relies on the subprocess inheriting them; there is no concurrency to leak
    across, so the strip must NOT apply.
    """
    # _isolate_session_context already forced engaged=False.
    monkeypatch.setenv("HERMES_SESSION_KEY", "cli-session-key")
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-session-id")

    env = _make_run_env({})

    assert env.get("HERMES_SESSION_KEY") == "cli-session-key"
    assert env.get("HERMES_SESSION_ID") == "cli-session-id"


def test_explicit_empty_thread_id_overrides_stale_value(monkeypatch):
    """A bound-but-empty thread id must override a stale inherited value.

    This is the complementary case (the #38507 scenario): a top-level post with
    no thread id binds HERMES_SESSION_THREAD_ID="" and that empty value must win
    over an older non-empty value left in os.environ.
    """
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "stale-thread-from-prior-turn")

    tokens = set_session_vars(
        session_key="mm:chan",
        platform="mattermost",
        chat_id="chan",
        thread_id="",  # explicitly no thread
    )
    try:
        env = _make_run_env({})
    finally:
        clear_session_vars(tokens)

    assert env.get("HERMES_SESSION_THREAD_ID") == "", (
        "Bound-empty thread id did not override the stale value: "
        f"{env.get('HERMES_SESSION_THREAD_ID')!r}"
    )
    assert env.get("HERMES_SESSION_KEY") == "mm:chan"


# --------------------------------------------------------------------------- #
# Background / PTY path (_sanitize_subprocess_env via process_registry)
# --------------------------------------------------------------------------- #

def test_sanitize_subprocess_env_strips_foreign_session_key_when_engaged(monkeypatch):
    """The background/PTY spawn path gets the same cross-session strip.

    process_registry.spawn_local() builds its env via _sanitize_subprocess_env(
    os.environ, env_vars). A background subprocess spawned with an UNSET
    ContextVar in an engaged process must not inherit a foreign session key.
    """
    _engage()
    stale_base = {
        "PATH": "/usr/bin:/bin",
        "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN_BG:FOREIGN_BG",
        "HERMES_SESSION_THREAD_ID": "FOREIGN_BG",
    }

    sanitized = _sanitize_subprocess_env(stale_base)

    assert "HERMES_SESSION_KEY" not in sanitized, (
        f"Background subprocess inherited foreign key: {sanitized.get('HERMES_SESSION_KEY')!r}"
    )
    assert "HERMES_SESSION_THREAD_ID" not in sanitized


def test_sanitize_subprocess_env_set_contextvar_wins_when_engaged():
    """Background path: a SET ContextVar overrides the foreign global base."""
    stale_base = {
        "PATH": "/usr/bin:/bin",
        "HERMES_SESSION_KEY": "agent:main:discord:thread:FOREIGN_BG:FOREIGN_BG",
    }
    tokens = set_session_vars(
        session_key="agent:main:discord:group:REAL_BG:222",
        platform="discord",
        chat_id="REAL_BG",
    )
    try:
        sanitized = _sanitize_subprocess_env(stale_base)
    finally:
        clear_session_vars(tokens)

    assert sanitized.get("HERMES_SESSION_KEY") == "agent:main:discord:group:REAL_BG:222"


# --------------------------------------------------------------------------- #
# Non-terminal spawn surface (hermes_subprocess_env) — sibling path
# --------------------------------------------------------------------------- #

def test_hermes_subprocess_env_strips_foreign_session_key_when_engaged(monkeypatch):
    """hermes_subprocess_env (browser/ACP/CLI/TUI-host spawns) must not leak a
    foreign session key either. cli.exec spawns via this helper WITHOUT re-binding
    the session identity, so an UNSET ContextVar under an engaged host must strip
    the inherited global rather than hand the child another session's identity.
    """
    _engage()
    monkeypatch.setenv(
        "HERMES_SESSION_KEY",
        "agent:main:discord:thread:FOREIGN_CONCURRENT:FOREIGN_CONCURRENT",
    )

    env = hermes_subprocess_env()

    assert "HERMES_SESSION_KEY" not in env, (
        "Foreign concurrent session key leaked into non-terminal spawn env: "
        f"{env.get('HERMES_SESSION_KEY')!r}"
    )


def test_hermes_subprocess_env_unengaged_preserves_fallback(monkeypatch):
    """A pure single-process CLI (never engaged) keeps the inherited fallback."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "cli-fallback-key")
    # not engaged (autouse fixture leaves _session_context_engaged False)
    env = hermes_subprocess_env()
    assert env.get("HERMES_SESSION_KEY") == "cli-fallback-key"


# --------------------------------------------------------------------------- #
# multiplex profile-scoped connector runtime env
# --------------------------------------------------------------------------- #

def test_make_run_env_keeps_profile_scoped_connector_runtime_out_of_popen_env(monkeypatch):
    """Terminal Popen env must not carry profile-scoped connector bearer.

    The gateway process may have stale globals from another profile, and the
    LocalEnvironment snapshot may also carry stale values. The generic terminal
    path is model-controlled shell, not the connector-specific runner, so it
    must stay free of these keys even when a profile scope exists.
    """
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv(
        "ZETTLAB_CONNECTORS_URL",
        "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=foreign",
    )
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "foreign-token")
    monkeypatch.setenv("ZET_AGENT_ID", "foreign")
    token = ss.set_secret_scope({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=main",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "main-token",
        "ZET_AGENT_ID": "main",
        "ZETTLAB_AGENT_ACTION_TOKEN": "main-action",
        "ZETTLAB_HARDWARE_EXECUTION_TOKEN": "main-hardware",
    })
    try:
        env = _make_run_env({
            "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=snapshot",
            "ZETTLAB_CONNECTORS_AUTH_TOKEN": "snapshot-token",
            "ZET_AGENT_ID": "snapshot",
        })
    finally:
        ss.reset_secret_scope(token)

    assert "ZETTLAB_CONNECTORS_URL" not in env
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert "ZET_AGENT_ID" not in env


def test_terminal_env_injects_only_profile_scoped_agent_output(monkeypatch, tmp_path):
    """Model shell receives the current profile's public output path, not tokens."""
    from agent import secret_scope as ss

    output = tmp_path / "agent-a" / "output"
    output.mkdir(parents=True)
    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(tmp_path / "foreign-output"))
    token = ss.set_secret_scope(
        {
            "ZET_AGENT_OUTPUT_DIR": str(output),
            "ZETTLAB_CONNECTORS_AUTH_TOKEN": "connector-secret",
            "ZETTLAB_AGENT_ACTION_TOKEN": "action-secret",
        }
    )
    try:
        foreground = _make_run_env({})
        background = _sanitize_subprocess_env({})
        local_env_module._apply_profile_secret_scope_env(background, inject=True)
    finally:
        ss.reset_secret_scope(token)

    for env in (foreground, background):
        assert env["ZET_AGENT_OUTPUT_DIR"] == str(output)
        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
        assert "ZETTLAB_AGENT_ACTION_TOKEN" not in env


def test_foreground_terminal_observes_live_profile_output(monkeypatch, tmp_path):
    """Shell snapshot scrubbing must restore the current public profile value."""
    from agent import secret_scope as ss

    output = tmp_path / "agent-a" / "output"
    output.mkdir(parents=True)
    monkeypatch.delenv("HERMES_MANAGED_GATEWAY", raising=False)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"ZET_AGENT_OUTPUT_DIR": str(output)})
    environment = None
    try:
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=15)
        result = environment.execute('printf "%s" "$ZET_AGENT_OUTPUT_DIR"')
    finally:
        if environment is not None:
            environment.cleanup()
        ss.reset_secret_scope(token)

    assert result["returncode"] == 0
    assert result["output"].strip() == str(output)


def test_terminal_child_observes_only_current_profile_catalog_token(monkeypatch, tmp_path):
    """A model terminal child gets the narrow current-profile catalog token only."""
    from agent import secret_scope as ss

    current_catalog_token = "a" * 64
    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZETTLAB_SKILLHUB_CATALOG_TOKEN", "b" * 64)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "foreign-action-token")
    scope_token = ss.set_secret_scope(
        {
            "ZETTLAB_SKILLHUB_CATALOG_TOKEN": current_catalog_token,
            "ZETTLAB_AGENT_ACTION_TOKEN": "current-action-token",
        }
    )
    environment = None
    try:
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=15)
        wrapped = environment._wrap_command("true", str(tmp_path))
        result = environment.execute(
            'printf "%s|%s" "${ZETTLAB_SKILLHUB_CATALOG_TOKEN-absent}" '
            '"${ZETTLAB_AGENT_ACTION_TOKEN-absent}"'
        )
    finally:
        if environment is not None:
            environment.cleanup()
        ss.reset_secret_scope(scope_token)

    assert result["returncode"] == 0
    assert result["output"].strip() == f"{current_catalog_token}|absent"
    assert current_catalog_token not in wrapped


@pytest.mark.parametrize(
    "profile_value",
    [None, "a" * 63, "A" * 64, "a" * 65, "a" * 63 + "g"],
)
def test_terminal_child_rejects_missing_or_invalid_catalog_token(
    monkeypatch,
    profile_value,
    tmp_path,
):
    """A global or another profile cannot supply catalog access to this shell."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZETTLAB_SKILLHUB_CATALOG_TOKEN", "b" * 64)
    scope = (
        {"ZETTLAB_SKILLHUB_CATALOG_TOKEN": profile_value}
        if profile_value is not None
        else {}
    )
    scope_token = ss.set_secret_scope(scope)
    environment = None
    try:
        environment = LocalEnvironment(cwd=str(tmp_path), timeout=15)
        result = environment.execute(
            'printf "%s|%s" "${ZETTLAB_SKILLHUB_CATALOG_TOKEN-absent}" '
            '"${ZETTLAB_AGENT_ACTION_TOKEN-absent}"'
        )
    finally:
        if environment is not None:
            environment.cleanup()
        ss.reset_secret_scope(scope_token)

    assert result["returncode"] == 0
    assert result["output"].strip() == "absent|absent"


def test_single_profile_terminal_does_not_enable_catalog_token(monkeypatch):
    """The narrow token is only a multiplex-gateway happy-path capability."""
    from agent import secret_scope as ss

    monkeypatch.setenv("ZETTLAB_SKILLHUB_CATALOG_TOKEN", "b" * 64)
    scope_token = ss.set_secret_scope(
        {"ZETTLAB_SKILLHUB_CATALOG_TOKEN": "a" * 64}
    )
    try:
        env = _make_run_env({})
    finally:
        ss.reset_secret_scope(scope_token)

    assert "ZETTLAB_SKILLHUB_CATALOG_TOKEN" not in env


def test_catalog_token_stays_out_of_nonterminal_subprocess_envs(monkeypatch):
    """Only the generic terminal builder opts into the catalog bearer."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZETTLAB_SKILLHUB_CATALOG_TOKEN", "b" * 64)
    scope_token = ss.set_secret_scope(
        {"ZETTLAB_SKILLHUB_CATALOG_TOKEN": "a" * 64}
    )
    try:
        envs = (
            _sanitize_subprocess_env({}),
            hermes_subprocess_env(),
        )
    finally:
        ss.reset_secret_scope(scope_token)

    for env in envs:
        assert "ZETTLAB_SKILLHUB_CATALOG_TOKEN" not in env


@pytest.mark.parametrize("value", ["relative/output", "bad\x00output", ""])
def test_terminal_env_rejects_invalid_profile_output(monkeypatch, value):
    """Malformed profile values never replace the scrubbed process-global value."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", "/foreign/output")
    token = ss.set_secret_scope({"ZET_AGENT_OUTPUT_DIR": value})
    try:
        env = _make_run_env({})
    finally:
        ss.reset_secret_scope(token)

    assert "ZET_AGENT_OUTPUT_DIR" not in env


def test_make_run_env_strips_connector_runtime_without_profile_scope(monkeypatch):
    """Multiplex mode must not inherit stale connector env without a scope."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv(
        "ZETTLAB_CONNECTORS_URL",
        "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=foreign",
    )
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "foreign-token")
    monkeypatch.setenv("ZET_AGENT_ID", "foreign")

    env = _make_run_env({})

    assert "ZETTLAB_CONNECTORS_URL" not in env
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert "ZET_AGENT_ID" not in env


def test_make_run_env_strips_connector_runtime_without_multiplex(monkeypatch):
    """Generic terminal commands must not inherit connector bearer in any mode."""
    monkeypatch.setenv(
        "ZETTLAB_CONNECTORS_URL",
        "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=single",
    )
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "single-token")
    monkeypatch.setenv("ZET_AGENT_ID", "single")

    env = _make_run_env({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=extra",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "extra-token",
        "ZET_AGENT_ID": "extra",
    })

    assert "ZETTLAB_CONNECTORS_URL" not in env
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert "ZET_AGENT_ID" not in env


def test_local_snapshot_wrapper_unsets_connector_runtime_before_and_after_command():
    """Sourced shell snapshots must not re-expose or re-persist connector bearer."""
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.env = {}
    env._session_id = "snapshot-test"
    env._snapshot_path = "/tmp/hermes-snapshot-test.sh"
    env._cwd_file = "/tmp/hermes-cwd-test.txt"
    env._cwd_marker = "__HERMES_CWD_snapshot_test__"
    env._snapshot_ready = True
    env._snapshot_passthrough_names = set()

    script = env._wrap_command("printf done", "/tmp")

    source_idx = script.index("source /tmp/hermes-snapshot-test.sh")
    first_unset_idx = script.index("unset ZETTLAB_CONNECTORS_AUTH_TOKEN")
    eval_idx = script.index("eval 'printf done'")
    second_unset_idx = script.index(
        "unset ZETTLAB_CONNECTORS_AUTH_TOKEN",
        first_unset_idx + 1,
    )
    dump_idx = script.index("export -p;")

    assert source_idx < first_unset_idx < eval_idx
    assert eval_idx < second_unset_idx < dump_idx


def test_local_snapshot_wrapper_preserves_unengaged_cli_fallback(monkeypatch):
    """Snapshot wrapping must not erase CLI-only os.environ session identity."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "cli-fallback-key")
    env = LocalEnvironment.__new__(LocalEnvironment)
    env._session_id = "snapshot-cli-test"
    env._snapshot_path = "/tmp/hermes-snapshot-cli-test.sh"
    env._cwd_file = "/tmp/hermes-cwd-cli-test.txt"
    env._cwd_marker = "__HERMES_CWD_snapshot_cli_test__"
    env._snapshot_ready = True
    env._snapshot_passthrough_names = set()
    env.env = {}

    script = env._wrap_command(
        "printf '%s' \"$HERMES_SESSION_KEY\"",
        "/tmp",
    )

    assert "unset HERMES_SESSION_KEY" not in script


def test_local_snapshot_bootstrap_unsets_connector_runtime_before_first_dump():
    """init_session must not publish connector bearer in the initial snapshot."""
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = "/tmp"
    env._session_id = "bootstrap-test"
    env._snapshot_path = "/tmp/hermes-snapshot-bootstrap.sh"
    env._cwd_file = "/tmp/hermes-cwd-bootstrap.txt"
    env._cwd_marker = "__HERMES_CWD_bootstrap_test__"
    env._snapshot_timeout = 5
    env._snapshot_passthrough_names = set()
    captured: dict[str, str] = {}

    class _Proc:
        pass

    env._run_bash = lambda script, **kwargs: captured.setdefault("script", script) or _Proc()
    env._wait_for_process = lambda proc, **kwargs: {"returncode": 0, "output": ""}
    env._update_cwd = lambda result: None

    env.init_session()
    script = captured["script"]

    unset_idx = script.index("unset ZETTLAB_CONNECTORS_AUTH_TOKEN")
    dump_idx = script.index("export -p;")

    assert unset_idx < dump_idx


def test_build_connector_runtime_env_uses_profile_scope(monkeypatch):
    """Only the allowlisted connector runner receives current profile values."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "foreign-token")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "foreign-action")
    monkeypatch.setenv("ZETTLAB_HARDWARE_EXECUTION_TOKEN", "foreign-hardware")
    token = ss.set_secret_scope({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=main",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "main-token",
        "ZET_AGENT_ID": "main",
    })
    try:
        env = build_connector_runtime_env()
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert env["ZETTLAB_CONNECTORS_URL"].endswith("agent_id=main")
    assert env["ZET_AGENT_ID"] == "main"
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in env
    assert "ZETTLAB_HARDWARE_EXECUTION_TOKEN" not in env


def test_build_connector_runtime_env_single_profile_strips_stale_capabilities(
    monkeypatch,
):
    from agent import secret_scope as ss

    ss.set_multiplex_active(False)
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://single.invalid/rpc")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "single-connector")
    monkeypatch.setenv("ZET_AGENT_ID", "single-agent")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "profile-action")
    monkeypatch.setenv("ZETTLAB_HARDWARE_EXECUTION_TOKEN", "stale-hardware")

    env = build_connector_runtime_env()

    assert env["ZETTLAB_CONNECTORS_URL"] == "http://single.invalid/rpc"
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert env["ZET_AGENT_ID"] == "single-agent"
    assert env["ZETTLAB_AGENT_ACTION_TOKEN"] == "profile-action"
    assert "ZETTLAB_HARDWARE_EXECUTION_TOKEN" not in env


def test_connector_route_capability_replaces_only_dedicated_runner_session_key(
    monkeypatch,
):
    """The per-turn capability never replaces generic subprocess routing."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "foreign-session")
    tokens = set_session_vars(
        session_key="real-session",
        platform="api_server",
        chat_id="chat-1",
    )
    set_zettlab_connector_route_capability("c" * 43)
    try:
        connector_env = build_connector_runtime_env()
        generic_env = _sanitize_subprocess_env(os.environ)
    finally:
        set_zettlab_connector_route_capability("")
        clear_session_vars(tokens)

    assert connector_env["HERMES_SESSION_KEY"] == "c" * 43
    assert generic_env["HERMES_SESSION_KEY"] == "real-session"


def test_sanitize_and_nonterminal_spawn_scrub_connector_runtime_env(monkeypatch):
    """Background/PTY and helper spawn paths must not receive connector bearer."""
    from agent import secret_scope as ss

    ss.set_multiplex_active(True)
    monkeypatch.setenv(
        "ZETTLAB_CONNECTORS_URL",
        "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=foreign",
    )
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "foreign-token")
    monkeypatch.setenv("ZET_AGENT_ID", "foreign")
    token = ss.set_secret_scope({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=profile-b",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "profile-b-token",
        "ZET_AGENT_ID": "profile-b",
    })
    try:
        sanitized = _sanitize_subprocess_env({})
        helper = hermes_subprocess_env()
    finally:
        ss.reset_secret_scope(token)

    for env in (sanitized, helper):
        assert "ZETTLAB_CONNECTORS_URL" not in env
        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
        assert "ZET_AGENT_ID" not in env


def test_sanitize_and_nonterminal_spawn_scrub_connector_runtime_env_without_multiplex(monkeypatch):
    """Non-multiplex helper paths still scrub connector bearer from os.environ."""
    monkeypatch.setenv(
        "ZETTLAB_CONNECTORS_URL",
        "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=single",
    )
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "single-token")
    monkeypatch.setenv("ZET_AGENT_ID", "single")

    sanitized = _sanitize_subprocess_env({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:9090/api/v1/internal/connectors/rpc?agent_id=base",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "base-token",
        "ZET_AGENT_ID": "base",
    })
    helper = hermes_subprocess_env()

    for env in (sanitized, helper):
        assert "ZETTLAB_CONNECTORS_URL" not in env
        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
        assert "ZET_AGENT_ID" not in env


# --------------------------------------------------------------------------- #
# execute_code sandbox child env
# --------------------------------------------------------------------------- #

def test_execute_code_child_gets_bound_session_routing_without_connector_bearer(monkeypatch):
    """execute_code should know its session key but not receive connector tokens.

    The sandbox child can route helper RPCs back to the right chat session via
    HERMES_SESSION_KEY, while ZETTLAB_CONNECTORS_AUTH_TOKEN stays out of the
    arbitrary Python environment.
    """
    monkeypatch.setenv("HERMES_SESSION_KEY", "foreign-session")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1:9090/api/v1/internal/connectors/rpc")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "connector-bearer")
    tokens = set_session_vars(
        session_key="real-session",
        platform="api",
        chat_id="chat-1",
    )
    try:
        env = _scrub_child_env(os.environ, is_passthrough=lambda _: False, is_windows=False)
        _inject_execute_code_session_context_env(env)
    finally:
        clear_session_vars(tokens)

    assert env.get("HERMES_SESSION_KEY") == "real-session"
    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert "ZETTLAB_CONNECTORS_URL" not in env


def test_execute_code_scrubs_connector_runtime_even_when_passthrough_allows_it(monkeypatch):
    """Connector runtime keys must not be recoverable through env_passthrough."""
    monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1:9090/rpc")
    monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "connector-bearer")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")

    env = _scrub_child_env(
        os.environ,
        is_passthrough=lambda name: name.startswith("ZET"),
        is_windows=False,
    )

    assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in env
    assert "ZETTLAB_CONNECTORS_URL" not in env
    assert "ZET_AGENT_ID" not in env
