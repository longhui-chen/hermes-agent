"""Tests for user-configured ``model.default_headers`` in the auxiliary client.

Companion to ``tests/run_agent/test_provider_attribution_headers.py`` (which
covers the main agent client). The main agent turn and the auxiliary client
(title generation, context compression, vision routing) build separate OpenAI
clients, so a ``custom`` endpoint behind a gateway/WAF that rejects the OpenAI
SDK's identifying headers needs the ``model.default_headers`` override applied
on BOTH paths — otherwise the main turn succeeds but auxiliary calls to the
same endpoint still fail with an opaque 4xx/502. (#40033)
"""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so load_config() reads our test config.yaml."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    import yaml
    (tmp_path / ".hermes" / "config.yaml").write_text(yaml.dump(config_dict))


class TestApplyUserDefaultHeadersHelper:
    """Direct unit tests for the merge helper."""

    def test_user_headers_merged_and_win(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({"User-Agent": "OpenAI/Python 2.24.0"})
        assert merged["User-Agent"] == "curl/8.7.1"  # user wins
        assert merged["X-Extra"] == "1"




    def test_none_values_skipped(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "m", "default_headers": {"User-Agent": "curl/8.7.1", "X-Drop": None}},
        })
        from agent.auxiliary_client import _apply_user_default_headers
        merged = _apply_user_default_headers({})
        assert merged == {"User-Agent": "curl/8.7.1"}
        assert "X-Drop" not in merged


class TestAuxClientHonorsUserDefaultHeaders:
    """Integration: resolve_provider_client must pass overridden headers to OpenAI."""

    def test_custom_provider_overrides_sdk_user_agent(self, tmp_path):
        """The #40033 reproduction on the auxiliary path."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
                "default_headers": {"User-Agent": "curl/8.7.1", "X-Extra": "1"},
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        assert mock_openai.called
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent") == "curl/8.7.1"
        assert headers.get("X-Extra") == "1"

    def test_custom_provider_no_override_sends_no_user_agent(self, tmp_path):
        """Without config, the aux client injects nothing — SDK defaults apply."""
        _write_config(tmp_path, {
            "model": {
                "default": "my-custom-model",
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
            },
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("main", "my-custom-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "User-Agent" not in headers

    def test_named_custom_provider_honors_override(self, tmp_path):
        """A `custom_providers:` entry's aux calls also honor model.default_headers.

        This is a distinct construction path (_extra2) from the config-level
        `model.provider: custom` path — both must apply the global override.
        """
        _write_config(tmp_path, {
            "model": {
                "default": "test-model",
                "default_headers": {"User-Agent": "curl/8.7.1"},
            },
            "custom_providers": [
                {"name": "my-gw", "base_url": "http://my-gw.local/v1", "api_key": "k"},
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("my-gw", "test-model")

        assert client is not None
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert headers.get("User-Agent") == "curl/8.7.1"

    def test_openrouter_provider_does_not_cache_zettlab_session_headers(self, tmp_path, monkeypatch):
        """OpenRouter default_headers are cached, so session headers stay per request."""
        _write_config(tmp_path, {"model": {"default": "test-model"}})
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        from gateway.session_context import set_current_session_id
        set_current_session_id("zettlab:u1:agent-a:abc")
        try:
            with patch("agent.auxiliary_client.OpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                from agent.auxiliary_client import _try_openrouter
                client, model = _try_openrouter(model="openrouter/model")
        finally:
            set_current_session_id("")

        assert client is not None
        assert model == "openrouter/model"
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "X-Task-Id" not in headers
        assert "X-Zettlab-Conversation-ID" not in headers
        assert "X-Scene-Type" not in headers
        assert headers.get("X-Title") == "Hermes Agent"

    def test_nous_provider_does_not_cache_zettlab_session_headers(self, tmp_path):
        """Nous default_headers are cached, so session headers stay per request."""
        _write_config(tmp_path, {"model": {"default": "test-model"}})

        from gateway.session_context import set_current_session_id
        set_current_session_id("zettlab:u1:agent-a:abc")
        try:
            with (
                patch("agent.auxiliary_client._read_nous_auth", return_value={}),
                patch(
                    "agent.auxiliary_client._resolve_nous_runtime_api",
                    return_value=("nous-jwt", "https://inference.example/v1"),
                ),
                patch("agent.auxiliary_client.OpenAI") as mock_openai,
            ):
                mock_openai.return_value = MagicMock()
                from agent.auxiliary_client import _try_nous
                client, model = _try_nous()
        finally:
            set_current_session_id("")

        assert client is not None
        assert model
        headers = mock_openai.call_args.kwargs.get("default_headers", {}) or {}
        assert "X-Task-Id" not in headers
        assert "X-Zettlab-Conversation-ID" not in headers
        assert "X-Scene-Type" not in headers
