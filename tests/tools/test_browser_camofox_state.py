"""Tests for Hermes-managed Camofox state helpers."""

from unittest.mock import patch



def _load_module():
    from tools import browser_camofox_state as state
    return state


class TestCamofoxStatePaths:
    def test_paths_are_profile_scoped(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            assert state.get_camofox_state_dir() == tmp_path / "browser_auth" / "camofox"


class TestCamofoxIdentity:
    def test_identity_is_deterministic(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            first = state.get_camofox_identity("task-1")
            second = state.get_camofox_identity("task-1")
            assert first == second

    def test_identity_differs_by_task(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            a = state.get_camofox_identity("task-a")
            b = state.get_camofox_identity("task-b")
            # Same user (same profile), different session keys
            assert a["user_id"] == b["user_id"]
            assert a["session_key"] != b["session_key"]

    def test_identity_differs_by_profile(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-a"):
            a = state.get_camofox_identity("task-1")
        with patch.object(state, "get_hermes_home", return_value=tmp_path / "profile-b"):
            b = state.get_camofox_identity("task-1")
        assert a["user_id"] != b["user_id"]

    def test_default_task_id(self, tmp_path):
        state = _load_module()
        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            identity = state.get_camofox_identity()
            assert "user_id" in identity
            assert "session_key" in identity
            assert identity["user_id"].startswith("hermes_")
            assert identity["session_key"].startswith("task_")

    def test_session_context_is_stable_and_not_exposed(self, tmp_path, monkeypatch):
        state = _load_module()
        sensitive_session = "agent:main:telegram:dm:user-13800138000"
        monkeypatch.setenv("HERMES_SESSION_ID", sensitive_session)

        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            first = state.get_camofox_identity("transient-tool-task")
            second = state.get_camofox_identity("different-tool-task")

        assert first == second
        assert sensitive_session not in first["session_key"]
        assert "13800138000" not in first["session_key"]

    def test_stable_session_key_takes_precedence_over_rotating_session_id(self, tmp_path, monkeypatch):
        state = _load_module()
        monkeypatch.setenv("HERMES_SESSION_KEY", "stable-channel-session")
        monkeypatch.setenv("HERMES_SESSION_ID", "conversation-before-compression")

        with patch.object(state, "get_hermes_home", return_value=tmp_path):
            before = state.get_camofox_identity("tool-task")
            monkeypatch.setenv("HERMES_SESSION_ID", "conversation-after-compression")
            after = state.get_camofox_identity("tool-task")

        assert before == after


class TestCamofoxConfigDefaults:
    def test_default_config_includes_camofox_controls(self):
        from hermes_cli.config import DEFAULT_CONFIG

        browser_cfg = DEFAULT_CONFIG["browser"]
        assert browser_cfg["camofox"]["managed_persistence"] is False
        assert browser_cfg["camofox"]["user_id"] == ""
        assert browser_cfg["camofox"]["session_key"] == ""
        assert browser_cfg["camofox"]["adopt_existing_tab"] is False
