"""Regression tests for the config.yaml → env var bridge in gateway/run.py.

Guards against the 60-vs-500 bug where a stale `.env HERMES_MAX_ITERATIONS=60`
entry silently shadowed `agent.max_turns: 500` in config.yaml because the
bridge used `if X not in os.environ` guards. After PR#18413 the bridge
treats config.yaml as authoritative and unconditionally overwrites .env
values for `agent.*`, `display.*`, `timezone`, and `security.*` keys.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_gateway_import(hermes_home: Path, initial_env: dict[str, str]) -> dict[str, str]:
    """Import gateway.run in a clean subprocess and return the post-import env.

    The bridge runs at module-import time, so simply importing is enough
    to exercise it. Running in a subprocess isolates the test from other
    import side effects and makes the "what ends up in os.environ" check
    deterministic.
    """
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        try:
            from gateway import run  # noqa: F401  — module import triggers bridge
        except Exception as exc:
            print(f"IMPORT_ERROR:{{type(exc).__name__}}:{{exc}}", file=sys.stderr)
            sys.exit(2)

        for k in (
            "HERMES_MAX_ITERATIONS",
            "HERMES_AGENT_TIMEOUT",
            "HERMES_AGENT_TIMEOUT_WARNING",
            "HERMES_GATEWAY_BUSY_INPUT_MODE",
            "HERMES_GATEWAY_BUSY_TEXT_MODE",
            "HERMES_TIMEZONE",
        ):
            v = os.environ.get(k)
            if v is not None:
                print(f"{{k}}={{v}}")
        """
    )
    env = dict(initial_env)
    env["HERMES_HOME"] = str(hermes_home)
    # Keep PATH / PYTHONPATH so venv imports resolve.
    for k in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME"):
        if k in os.environ and k not in env:
            env[k] = os.environ[k]

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"gateway.run import failed (rc={result.returncode})\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def _run_resolved_timezone(
    hermes_home: Path,
    initial_env: dict[str, str],
    *,
    rewrite_timezone: str | None = None,
) -> dict[str, str]:
    """Import gateway.run, then resolve the timezone via hermes_time.

    Asserts on the RESOLVED timezone (the user-visible outcome) rather than the
    raw HERMES_TIMEZONE env var, because the fix deliberately stops pinning the
    env var. Returns ``{"first": <tz>}`` and, when ``rewrite_timezone`` is
    given, also ``{"second": <tz>}`` — the timezone resolved AFTER rewriting
    config.yaml's ``timezone`` live, exercising the path env-pinning froze.
    """
    # Build the script as explicit column-0 lines — textwrap.dedent does not
    # compose with interpolating an already-dedented multi-line block.
    lines = [
        "import os, sys",
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})",
        "try:",
        "    from gateway import run  # noqa: F401 — import triggers the bridge",
        "except Exception as exc:",
        "    print('IMPORT_ERROR:' + type(exc).__name__ + ':' + str(exc), file=sys.stderr)",
        "    sys.exit(2)",
        "import hermes_time",
        "hermes_time.reset_cache()",
        "print('FIRST=' + (hermes_time.get_timezone_name() or 'NONE'))",
    ]
    if rewrite_timezone is not None:
        lines += [
            "import yaml as _yaml",
            "_cfg_path = os.path.join(os.environ['HERMES_HOME'], 'config.yaml')",
            "with open(_cfg_path) as _f:",
            "    _data = _yaml.safe_load(_f) or {}",
            f"_data['timezone'] = {rewrite_timezone!r}",
            "with open(_cfg_path, 'w') as _f:",
            "    _yaml.safe_dump(_data, _f)",
            "hermes_time.reset_cache()",
            "print('SECOND=' + (hermes_time.get_timezone_name() or 'NONE'))",
        ]
    script = "\n".join(lines)
    env = dict(initial_env)
    env["HERMES_HOME"] = str(hermes_home)
    for k in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME"):
        if k in os.environ and k not in env:
            env[k] = os.environ[k]

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"gateway.run import/resolve failed (rc={result.returncode})\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("FIRST="):
            out["first"] = line[len("FIRST="):]
        elif line.startswith("SECOND="):
            out["second"] = line[len("SECOND="):]
    return out


def _write_config(home: Path, agent_cfg: dict | None = None, display_cfg: dict | None = None,
                  timezone: str | None = None) -> None:
    import yaml
    cfg: dict = {}
    if agent_cfg:
        cfg["agent"] = agent_cfg
    if display_cfg:
        cfg["display"] = display_cfg
    if timezone:
        cfg["timezone"] = timezone
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _write_env(home: Path, entries: dict[str, str]) -> None:
    lines = [f"{k}={v}\n" for k, v in entries.items()]
    (home / ".env").write_text("".join(lines))


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


def test_config_max_turns_wins_over_stale_env(hermes_home: Path) -> None:
    """Regression: config.yaml:agent.max_turns=500 must beat .env=60."""
    _write_config(hermes_home, agent_cfg={"max_turns": 500})
    _write_env(hermes_home, {"HERMES_MAX_ITERATIONS": "60"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_MAX_ITERATIONS") == "500", (
        f"expected config.yaml max_turns=500 to win; got {env.get('HERMES_MAX_ITERATIONS')!r}. "
        "Stale .env value is shadowing config — the bridge lost its override."
    )


def test_config_gateway_timeout_wins_over_stale_env(hermes_home: Path) -> None:
    """Every agent.* bridge key must be config-authoritative, not .env-authoritative."""
    _write_config(hermes_home, agent_cfg={
        "gateway_timeout": 1800,
        "gateway_timeout_warning": 900,
    })
    _write_env(hermes_home, {
        "HERMES_AGENT_TIMEOUT": "60",
        "HERMES_AGENT_TIMEOUT_WARNING": "30",
    })

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_AGENT_TIMEOUT") == "1800"
    assert env.get("HERMES_AGENT_TIMEOUT_WARNING") == "900"


def test_config_display_busy_input_mode_wins_over_stale_env(hermes_home: Path) -> None:
    _write_config(hermes_home, display_cfg={"busy_input_mode": "interrupt"})
    _write_env(hermes_home, {"HERMES_GATEWAY_BUSY_INPUT_MODE": "queue"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_GATEWAY_BUSY_INPUT_MODE") == "interrupt"


def test_config_display_busy_text_mode_wins_over_stale_env(hermes_home: Path) -> None:
    _write_config(hermes_home, display_cfg={"busy_text_mode": "queue"})
    _write_env(hermes_home, {"HERMES_GATEWAY_BUSY_TEXT_MODE": "interrupt"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_GATEWAY_BUSY_TEXT_MODE") == "queue"


def test_config_timezone_wins_over_stale_env(hermes_home: Path) -> None:
    """config.yaml timezone must win over a stale .env HERMES_TIMEZONE.

    The bridge achieves this by DROPPING the stale .env value (not by pinning
    config into the env, which froze live updates), so hermes_time resolves
    config.yaml live.
    """
    _write_config(hermes_home, timezone="America/Los_Angeles")
    _write_env(hermes_home, {"HERMES_TIMEZONE": "UTC"})

    resolved = _run_resolved_timezone(hermes_home, initial_env={})

    assert resolved["first"] == "America/Los_Angeles"


def test_config_timezone_live_update_not_frozen(hermes_home: Path) -> None:
    """Editing config.yaml's timezone after startup must take effect live.

    Regression guard for the env-pinning bug: pinning HERMES_TIMEZONE froze the
    value for the process lifetime, so later config.yaml edits (and the
    device-mode OS timezone set by the App via timedatectl) were ignored until
    the gateway restarted.
    """
    _write_config(hermes_home, timezone="America/Los_Angeles")
    _write_env(hermes_home, {"HERMES_TIMEZONE": "UTC"})

    resolved = _run_resolved_timezone(
        hermes_home, initial_env={}, rewrite_timezone="America/New_York"
    )

    assert resolved["first"] == "America/Los_Angeles"
    assert resolved["second"] == "America/New_York"  # picked up live, not frozen


def test_operator_env_timezone_survives_config(hermes_home: Path) -> None:
    """A genuine operator override (HERMES_TIMEZONE in the real environment)
    still wins over config.yaml — only a stale .env value is dropped."""
    _write_config(hermes_home, timezone="America/Los_Angeles")
    # No .env timezone; the operator exports it in the real process env.
    resolved = _run_resolved_timezone(
        hermes_home, initial_env={"HERMES_TIMEZONE": "Asia/Tokyo"}
    )

    assert resolved["first"] == "Asia/Tokyo"


def test_invalid_config_timezone_keeps_valid_env(hermes_home: Path) -> None:
    """A typo'd config.yaml timezone must NOT strip a valid .env value.

    Dropping the .env value only helps when something better will win; if the
    config zone is invalid, stripping would downgrade resolution to
    server-local time — worse than the working .env value. So a bad config
    zone leaves the .env value in place.
    """
    _write_config(hermes_home, timezone="Not/A/Zone")
    _write_env(hermes_home, {"HERMES_TIMEZONE": "UTC"})

    resolved = _run_resolved_timezone(hermes_home, initial_env={})

    assert resolved["first"] == "UTC"


def test_device_mode_strips_stale_env_when_os_tz_resolves(monkeypatch) -> None:
    """Device mode + a resolvable OS timezone (set by the App via timedatectl):
    the stale .env HERMES_TIMEZONE is dropped so the OS tz wins, even with no
    config timezone."""
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")  # simulate a .env-injected value
    # Classify it as NOT operator-set (came from .env, not the pre-load snapshot).
    monkeypatch.setattr("gateway.run.env_var_was_operator_set", lambda name: False)
    monkeypatch.setattr("hermes_time._is_zettlab_device_mode", lambda: True)
    monkeypatch.setattr("hermes_time._read_os_timezone", lambda: "Asia/Shanghai")

    gateway_run._apply_config_timezone_authority({})  # config has no timezone

    assert "HERMES_TIMEZONE" not in os.environ


def test_device_mode_keeps_stale_env_when_os_tz_unresolvable(monkeypatch) -> None:
    """Device mode but NO resolvable OS tz (fresh board before the App ran
    timedatectl) and no config tz: the stale .env value is KEPT — stripping
    would leave nothing valid and downgrade to server-local time."""
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    monkeypatch.setattr("gateway.run.env_var_was_operator_set", lambda name: False)
    monkeypatch.setattr("hermes_time._is_zettlab_device_mode", lambda: True)
    monkeypatch.setattr("hermes_time._read_os_timezone", lambda: "")  # OS tz unset

    gateway_run._apply_config_timezone_authority({})

    assert os.environ.get("HERMES_TIMEZONE") == "UTC"


def test_env_value_survives_when_config_omits_key(hermes_home: Path) -> None:
    """If config.yaml doesn't set max_turns, .env value must still pass through.

    The bridge only overwrites when the config key is present — an absent
    config key should NOT clobber the .env value.
    """
    _write_config(hermes_home, agent_cfg={})  # no max_turns
    _write_env(hermes_home, {"HERMES_MAX_ITERATIONS": "123"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_MAX_ITERATIONS") == "123"
