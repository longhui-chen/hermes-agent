"""Unit tests for the Zettlab-hosted managed-tool gateway resolver."""

from __future__ import annotations

import pytest

from agent.secret_scope import reset_secret_scope, set_secret_scope
from tools.zettlab_tool_gateway import (
    local_server_gateway_url,
    resolve_zettlab_tool_gateway,
)


_ENV_KEYS = (
    "BROWSER_USE_GATEWAY_URL",
    "ZETTLAB_TOOL_GATEWAY_TOKEN",
    "ZET_CHAT_APPEND_URL",
    "ZETTLAB_AGENT_SHARE_ACTION_URL",
    "ZETTLAB_AGENT_ACTION_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_returns_none_without_local_server_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    assert resolve_zettlab_tool_gateway("browser-use") is None


def test_returns_none_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert resolve_zettlab_tool_gateway("browser-use") is None


def test_resolves_from_local_server_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-browser-use")
    cfg = resolve_zettlab_tool_gateway("browser-use")
    assert cfg is not None
    assert cfg.vendor == "browser-use"
    assert cfg.gateway_origin == "http://127.0.0.1:9090/api/v1/browser-use"
    assert cfg.token == "local-browser-use"


def test_resolves_only_openai_tts_to_local_ai_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-tts-token")
    cfg = resolve_zettlab_tool_gateway("openai-tts")
    assert cfg is not None
    assert cfg.gateway_origin == "http://127.0.0.1:9090/api/v1/ai-proxy"
    assert cfg.token == "local-tts-token"
    assert resolve_zettlab_tool_gateway("openai-audio") is None


def test_can_use_share_action_url_as_local_server_anchor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETTLAB_AGENT_SHARE_ACTION_URL", "http://127.0.0.1:9090/api/v1/internal/action")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-browser-use")
    assert local_server_gateway_url("browser-use") == "http://127.0.0.1:9090/api/v1/browser-use"


def test_returns_none_for_unsupported_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-browser-use")
    assert local_server_gateway_url("firecrawl") == ""
    assert resolve_zettlab_tool_gateway("firecrawl") is None


def test_ignores_old_per_vendor_zettlab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_GATEWAY_URL", "http://127.0.0.1:9090/api/v1/browser-use")
    monkeypatch.setenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "old-placeholder")
    assert local_server_gateway_url("browser-use") == ""
    assert resolve_zettlab_tool_gateway("browser-use") is None


def test_returns_none_for_malformed_callback_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "not-a-url")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-browser-use")
    assert local_server_gateway_url("browser-use") == ""
    assert resolve_zettlab_tool_gateway("browser-use") is None


@pytest.mark.parametrize(
    "callback_url",
    (
        "https://example.com/api/v1/internal/chat/append",
        "http://192.0.2.10:9090/api/v1/internal/chat/append",
        "ftp://127.0.0.1:9090/api/v1/internal/chat/append",
    ),
)
def test_rejects_non_loopback_or_non_http_callback_url(
    monkeypatch: pytest.MonkeyPatch,
    callback_url: str,
) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", callback_url)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "must-not-leave-device")

    assert resolve_zettlab_tool_gateway("openai-tts") is None


def test_resolves_profile_scoped_local_server_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ZET_CHAT_APPEND_URL",
        "http://127.0.0.1:9999/api/v1/internal/chat/append",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "stale-profile-token")
    scope = set_secret_scope(
        {
            "ZET_CHAT_APPEND_URL": (
                "http://127.0.0.1:9420/api/v1/internal/chat/append"
            ),
            "ZETTLAB_AGENT_ACTION_TOKEN": "active-profile-token",
        }
    )
    try:
        cfg = resolve_zettlab_tool_gateway("openai-tts")
    finally:
        reset_secret_scope(scope)

    assert cfg is not None
    assert cfg.gateway_origin == "http://127.0.0.1:9420/api/v1/ai-proxy"
    assert cfg.token == "active-profile-token"


def test_never_falls_back_to_nous_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # No local-server callback → no default (unlike the Nous build_vendor_gateway_url).
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "action-token")
    assert local_server_gateway_url("browser-use") == ""
    assert resolve_zettlab_tool_gateway("browser-use") is None
