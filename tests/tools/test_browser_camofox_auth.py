"""Tests that Camofox browser sends Authorization header when CAMOFOX_API_KEY is set.

Regression test for https://github.com/NousResearch/hermes-agent/issues/20476
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_camofox import (
    _auth_headers,
    camofox_back,
    camofox_click,
    camofox_close,
    camofox_navigate,
    camofox_press,
    camofox_scroll,
    camofox_snapshot,
    camofox_type,
    check_camofox_available,
    get_camofox_url,
    get_vnc_url,
    is_camofox_mode,
)


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = b"\x89PNG\r\n\x1a\nfake"
    resp.raise_for_status = MagicMock()
    return resp


class TestAuthHeaders:
    """Unit tests for _auth_headers() helper."""

    def test_empty_when_no_key(self, monkeypatch):
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)
        assert _auth_headers() == {}

    def test_bearer_when_key_set(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_API_KEY", "test-secret-123")
        assert _auth_headers() == {"Authorization": "Bearer test-secret-123"}

    def test_empty_when_key_blank(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_API_KEY", "   ")
        assert _auth_headers() == {}

    def test_action_token_mode_uses_dedicated_header(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_AUTH_MODE", "zettlab_action_token")
        monkeypatch.setenv("CAMOFOX_URL", "http://127.0.0.1:9377/internal/browser/camofox")
        monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "agent-action-token")
        monkeypatch.setenv("CAMOFOX_API_KEY", "direct-api-key")

        assert _auth_headers() == {
            "X-Zettlab-Agent-Action-Token": "agent-action-token",
        }

    @pytest.mark.parametrize(
        "url",
        [
            "http://evil.example:9377",
            "https://127.0.0.1:9377",
            "http://10.0.0.5:9377",
            "",
        ],
    )
    def test_action_token_never_leaves_loopback(self, monkeypatch, url):
        """The token grants local Agent authority and must stay on loopback.

        A profile .env that is misconfigured or overwritten with an external
        address would otherwise hand the credential to whoever answers, with no
        redirect involved.
        """
        monkeypatch.setenv("CAMOFOX_AUTH_MODE", "zettlab_action_token")
        monkeypatch.setenv("CAMOFOX_URL", url)
        monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "agent-action-token")

        with pytest.raises(RuntimeError, match="loopback"):
            _auth_headers()

    def test_action_token_mode_fails_closed_without_token(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_AUTH_MODE", "zettlab_action_token")
        monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
        monkeypatch.setenv("CAMOFOX_API_KEY", "must-not-fall-back")

        with pytest.raises(RuntimeError, match="ZETTLAB_AGENT_ACTION_TOKEN"):
            _auth_headers()

    def test_multiplex_reads_only_active_profile_scope(self, monkeypatch):
        from agent.secret_scope import (
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )

        monkeypatch.setenv("CAMOFOX_URL", "http://wrong-profile:9377")
        monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "wrong-profile-token")
        set_multiplex_active(True)
        scope_token = set_secret_scope({
            "CAMOFOX_URL": "http://127.0.0.1:9420/internal/browser",
            "CAMOFOX_AUTH_MODE": "zettlab_action_token",
            "ZETTLAB_AGENT_ACTION_TOKEN": "active-profile-token",
        })
        try:
            assert get_camofox_url() == "http://127.0.0.1:9420/internal/browser"
            assert _auth_headers() == {
                "X-Zettlab-Agent-Action-Token": "active-profile-token",
            }
        finally:
            reset_secret_scope(scope_token)
            set_multiplex_active(False)

    def test_multiplex_without_profile_scope_fails_closed(self):
        from agent.secret_scope import UnscopedSecretError, set_multiplex_active

        set_multiplex_active(True)
        try:
            with pytest.raises(UnscopedSecretError):
                get_camofox_url()
        finally:
            set_multiplex_active(False)

    def test_multiplex_cdp_override_is_profile_scoped(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

        monkeypatch.setenv("BROWSER_CDP_URL", "http://foreign-profile:9222")
        set_multiplex_active(True)
        scope_token = set_secret_scope({
            "CAMOFOX_URL": "http://127.0.0.1:9420/internal/browser",
            "BROWSER_CDP_URL": "",
        })
        try:
            assert is_camofox_mode() is True
        finally:
            reset_secret_scope(scope_token)
            set_multiplex_active(False)

    def test_multiplex_never_exposes_process_global_vnc(self, monkeypatch):
        from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
        import tools.browser_camofox as mod

        mod._vnc_url = "http://first-profile:6080"
        mod._vnc_url_checked = True
        set_multiplex_active(True)
        token = set_secret_scope({"CAMOFOX_URL": "http://second-profile:9377"})
        try:
            assert get_vnc_url() is None
        finally:
            reset_secret_scope(token)
            set_multiplex_active(False)


class TestAuthHeadersSent:
    """Verify all HTTP call sites include auth headers when CAMOFOX_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def _set_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_API_KEY", "my-api-key")

    @patch("tools.browser_camofox.requests.post")
    def test_ensure_tab_sends_auth(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t1"})
        camofox_navigate("https://example.com", task_id="auth_test_1")
        _, kwargs = mock_post.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    def test_post_sends_auth(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t2"})
        camofox_navigate("https://example.com", task_id="auth_test_2")
        mock_post.return_value = _mock_response(json_data={"ok": True, "url": "https://x.com"})
        camofox_navigate("https://x.com", task_id="auth_test_2")
        # The second call is a POST to /tabs/{tabId}/navigate
        last_call = mock_post.call_args_list[-1]
        assert last_call.kwargs.get("headers") == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.get")
    def test_get_sends_auth(self, mock_get, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t3"})
        camofox_navigate("https://example.com", task_id="auth_test_3")
        mock_get.return_value = _mock_response(json_data={
            "snapshot": '- heading "Hello"',
            "refsCount": 1,
        })
        camofox_snapshot(task_id="auth_test_3")
        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.post")
    @patch("tools.browser_camofox.requests.delete")
    def test_delete_sends_auth(self, mock_delete, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t4"})
        camofox_navigate("https://example.com", task_id="auth_test_4")
        mock_delete.return_value = _mock_response(json_data={"ok": True})
        camofox_close(task_id="auth_test_4")
        _, kwargs = mock_delete.call_args
        assert kwargs["headers"] == {"Authorization": "Bearer my-api-key"}

    @patch("tools.browser_camofox.requests.get")
    def test_health_sends_auth(self, mock_get):
        mock_get.return_value = _mock_response(json_data={"ok": True})

        assert check_camofox_available() is True

        # allow_redirects=False keeps the credential from being replayed to a
        # redirect target chosen by a misconfigured or compromised endpoint.
        mock_get.assert_called_once_with(
            "http://localhost:9377/health",
            timeout=5,
            headers={"Authorization": "Bearer my-api-key"},
            allow_redirects=False,
        )


class TestNoAuthHeadersWhenKeyUnset:
    """Verify HTTP calls send empty headers when CAMOFOX_API_KEY is not set."""

    @pytest.fixture(autouse=True)
    def _unset_key(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.delenv("CAMOFOX_API_KEY", raising=False)

    @patch("tools.browser_camofox.requests.post")
    def test_no_auth_on_tab_creation(self, mock_post):
        mock_post.return_value = _mock_response(json_data={"tabId": "t5"})
        camofox_navigate("https://example.com", task_id="noauth_test_1")
        _, kwargs = mock_post.call_args
        assert kwargs.get("headers") == {}
