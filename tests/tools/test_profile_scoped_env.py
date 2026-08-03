"""The profile-scoped subprocess env contract.

The bug these pin: nothing pointed the worker's wecom-cli at the session
profile's credential store, so it reported WeCom "not initialised" on a device
where the credentials were present and live API calls worked.

Two shapes reach that outcome and both are covered below. In per-profile-process
mode local-server injects ``WECOM_CLI_CONFIG_DIR`` computed for the GATEWAY's
agent, so a worker on another profile inherits a pointer to the wrong store. In
multiplex_gateway mode -- what the device runs -- there is no per-agent child at
all, so the key never arrives and wecom-cli falls back to one global directory
that every agent shares. Re-pointing a wrong value and supplying a missing one
are different tests; only the second matches the device.

⛔ The assertions below spawn a real child and read the environment from
*inside* it. Checking the dict we built would only prove we wrote a line;
``/proc/<pid>/environ`` would not help either, since it freezes at exec and
cannot see what Python put into ``os.environ`` afterwards. The only honest
observation is the child reporting its own environment.
"""

import json
import subprocess
import sys
from pathlib import Path

from hermes_constants import apply_profile_scoped_env
from tools.environments.local import _sanitize_subprocess_env, hermes_subprocess_env

# Reads the keys under test out of the child's own environment.
_REPORTER = (
    "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
    "('HERMES_HOME', 'WECOM_CLI_CONFIG_DIR', 'WECOM_SKILLS_DIR', 'LARK_SKILLS_DIR')}))"
)


def _child_env_of(env: dict[str, str]) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, "-c", _REPORTER],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"reporter failed: {result.stderr}"
    return json.loads(result.stdout)


def _gateway_env(profile_root: Path) -> dict[str, str]:
    """What local-server hands the gateway: keys computed for the GATEWAY's agent."""
    env = hermes_subprocess_env(inherit_credentials=True)
    env["HERMES_HOME"] = str(profile_root / "gateway-agent")
    env["WECOM_CLI_CONFIG_DIR"] = str(profile_root / "gateway-agent" / "wecom-cli-config")
    # Both skills dirs are one shared global path on the board, not per-agent.
    env["WECOM_SKILLS_DIR"] = "/root/.agents/skills"
    env["LARK_SKILLS_DIR"] = "/root/.agents/skills"
    return env


def test_wecom_config_dir_follows_the_profile_into_the_spawned_child(tmp_path):
    session_profile = tmp_path / "profiles" / "main"
    env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(env, session_profile)

    seen = _child_env_of(env)
    assert seen["HERMES_HOME"] == str(session_profile)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config"), (
        "the child still points at the gateway agent's credential store; wecom-cli "
        "reads this key and nothing else, so it finds no credentials and reports "
        "'not initialised'"
    )


def test_the_key_is_supplied_when_the_gateway_never_had_it(tmp_path):
    """The shape the device is actually in, and it is not the one above.

    multiplex_gateway mode has no per-agent child process, so registry.go's
    gatewayEnv injection never runs; per-profile values go into
    <hermesHome>/profiles/<agent>/.env instead, and the interactive path's
    load_hermes_dotenv reads the LAUNCH home's .env, not the profile's. The key
    therefore never arrives at all -- and an absent WECOM_CLI_CONFIG_DIR makes
    wecom-cli fall back to one global directory, so every agent shares a single
    identity and none see their own credentials.

    ⛔ Re-pointing a wrong value and supplying a missing one are different
    tests; only the second matches the device.
    """
    session_profile = tmp_path / "profiles" / "main"
    env = hermes_subprocess_env(inherit_credentials=True)
    env.pop("WECOM_CLI_CONFIG_DIR", None)
    assert "WECOM_CLI_CONFIG_DIR" not in env, "the premise is that nobody injected it"

    apply_profile_scoped_env(env, session_profile)

    seen = _child_env_of(env)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config")


def test_the_value_survives_the_terminal_hop_the_agent_actually_uses(tmp_path):
    """wecom-cli is not the worker: the agent runs it through the terminal path.

    That path uses a different sanitizer (``_sanitize_subprocess_env``), so the
    worker having the key proves nothing on its own.
    """
    session_profile = tmp_path / "profiles" / "main"
    worker_env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(worker_env, session_profile)

    terminal_env = _sanitize_subprocess_env(worker_env)
    assert "PATH" in terminal_env, "calibration: an empty base makes every key look stripped"

    seen = _child_env_of(terminal_env)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config")


def test_shared_skills_dirs_are_not_rewritten_per_profile(tmp_path):
    """⛔ These two are one global directory on the board, not per-agent.

    Deriving them from the profile would aim wecom-cli and lark-cli at a
    directory that has no skills in it -- a regression that looks like the fix.
    """
    env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(env, tmp_path / "profiles" / "main")

    seen = _child_env_of(env)
    assert seen["WECOM_SKILLS_DIR"] == "/root/.agents/skills"
    assert seen["LARK_SKILLS_DIR"] == "/root/.agents/skills"


def test_no_spawn_site_re_points_hermes_home_on_its_own(tmp_path):
    """The twin guard: HERMES_HOME was set inline, and that is how the sibling
    key got forgotten. Any site that re-points a child at a profile must go
    through the contract, so the next key added there reaches every spawn.
    """
    root = Path(__file__).resolve().parents[2]
    sites = [root / "tui_gateway" / "server.py", root / "hermes_cli" / "web_server.py"]
    for site in sites:
        source = site.read_text(encoding="utf-8")
        assert "apply_profile_scoped_env" in source, (
            f"{site.name} re-points a child at a session profile and must use the contract"
        )
        offenders = [
            line.strip()
            for line in source.splitlines()
            if 'env["HERMES_HOME"] =' in line or "env['HERMES_HOME'] =" in line
        ]
        assert not offenders, (
            f"{site.name} sets HERMES_HOME directly: {offenders}. Every profile-scoped "
            "key has to move together, which is what apply_profile_scoped_env is for"
        )
