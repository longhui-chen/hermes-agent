"""Tests for Singularity/Apptainer preflight availability check.

Verifies that a clear error is raised when neither apptainer nor
singularity is installed, instead of a cryptic FileNotFoundError.

See: https://github.com/NousResearch/hermes-agent/issues/1511
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.environments.singularity import (
    SingularityEnvironment,
    _find_singularity_executable,
    _ensure_singularity_available,
)


class TestFindSingularityExecutable:
    """_find_singularity_executable resolution tests."""

    def test_prefers_apptainer(self):
        """When both are available, apptainer should be preferred."""
        def which_both(name):
            return f"/usr/bin/{name}" if name in {"apptainer", "singularity"} else None

        with patch("shutil.which", side_effect=which_both):
            assert _find_singularity_executable() == "apptainer"

    def test_falls_back_to_singularity(self):
        """When only singularity is available, use it."""
        def which_singularity_only(name):
            return "/usr/bin/singularity" if name == "singularity" else None

        with patch("shutil.which", side_effect=which_singularity_only):
            assert _find_singularity_executable() == "singularity"

    def test_raises_when_neither_found(self):
        """Must raise RuntimeError with install instructions."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither.*apptainer.*nor.*singularity"):
                _find_singularity_executable()


class TestEnsureSingularityAvailable:
    """_ensure_singularity_available preflight tests."""

    def test_returns_executable_on_success(self):
        """Returns the executable name when version check passes."""
        fake_result = MagicMock(returncode=0, stderr="")

        with patch("shutil.which", side_effect=lambda n: "/usr/bin/apptainer" if n == "apptainer" else None), \
             patch("subprocess.run", return_value=fake_result):
            assert _ensure_singularity_available() == "apptainer"

    def test_raises_on_version_failure(self):
        """Raises RuntimeError when version command fails."""
        fake_result = MagicMock(returncode=1, stderr="unknown flag")

        with patch("shutil.which", side_effect=lambda n: "/usr/bin/apptainer" if n == "apptainer" else None), \
             patch("subprocess.run", return_value=fake_result):
            with pytest.raises(RuntimeError, match="version.*failed"):
                _ensure_singularity_available()

    def test_raises_on_timeout(self):
        """Raises RuntimeError when version command times out."""
        with patch("shutil.which", side_effect=lambda n: "/usr/bin/apptainer" if n == "apptainer" else None), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("apptainer", 10)):
            with pytest.raises(RuntimeError, match="timed out"):
                _ensure_singularity_available()

    def test_raises_when_not_installed(self):
        """Raises RuntimeError when neither executable exists."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Neither.*apptainer.*nor.*singularity"):
                _ensure_singularity_available()


class TestSingularityConnectorEnvScrub:
    """Connector profile env must not leak into Apptainer commands."""

    def test_start_instance_uses_cleanenv_and_sanitized_host_env(self, monkeypatch):
        captured = {}
        env = SingularityEnvironment.__new__(SingularityEnvironment)
        env.executable = "apptainer"
        env.instance_id = "hermes_test"
        env.image = "/tmp/hermes.sif"
        env._persistent = False
        env._overlay_dir = None
        env._memory = 0
        env._cpu = 0
        env._instance_started = False

        monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "host-token")
        monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1/rpc")
        monkeypatch.setenv("ZET_AGENT_ID", "agent-1")

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        env._start_instance()

        assert "--cleanenv" in captured["cmd"]
        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["env"]
        assert "ZETTLAB_CONNECTORS_URL" not in captured["env"]
        assert "ZET_AGENT_ID" not in captured["env"]
        assert env._instance_started is True

    def test_run_bash_uses_cleanenv_and_sanitized_host_env(self, monkeypatch):
        captured = {}
        env = SingularityEnvironment.__new__(SingularityEnvironment)
        env.executable = "apptainer"
        env.instance_id = "hermes_test"
        env._instance_started = True

        monkeypatch.setenv("ZETTLAB_CONNECTORS_AUTH_TOKEN", "host-token")
        monkeypatch.setenv("ZETTLAB_CONNECTORS_URL", "http://127.0.0.1/rpc")
        monkeypatch.setenv("ZET_AGENT_ID", "agent-1")

        def fake_popen(cmd, stdin_data=None, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return MagicMock()

        monkeypatch.setattr("tools.environments.singularity._popen_bash", fake_popen)

        env._run_bash("env", login=True)

        assert "--cleanenv" in captured["cmd"]
        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" not in captured["env"]
        assert "ZETTLAB_CONNECTORS_URL" not in captured["env"]
        assert "ZET_AGENT_ID" not in captured["env"]

    def test_snapshot_ephemeral_env_keys_include_connector_runtime(self):
        env = SingularityEnvironment.__new__(SingularityEnvironment)

        keys = set(env._snapshot_ephemeral_env_keys())

        assert "ZETTLAB_CONNECTORS_AUTH_TOKEN" in keys
        assert "ZETTLAB_CONNECTORS_URL" in keys
        assert "ZET_AGENT_ID" in keys

    def test_snapshot_ephemeral_env_keys_include_engaged_turn_context(self, monkeypatch):
        import gateway.session_context as sc

        monkeypatch.setattr(sc, "_session_context_engaged", True)
        env = SingularityEnvironment.__new__(SingularityEnvironment)

        keys = set(env._snapshot_ephemeral_env_keys())

        assert "HERMES_TURN_ID" in keys
        assert "HERMES_PLAN_ACK_REVISION_REQUESTED" in keys
