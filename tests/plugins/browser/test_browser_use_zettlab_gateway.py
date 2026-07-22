"""browser-use provider integration with the Zettlab managed gateway.

Verifies the device path: with local-server's generic callback/action env and no
commercial key, the provider resolves a managed config pointed at the Zettlab
gateway without any Nous entitlement.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.browser.browser_use.provider import BrowserUseBrowserProvider

_LOOPBACK = "http://127.0.0.1:9090/api/v1/browser-use"

_ENV_KEYS = (
    "BROWSER_USE_API_KEY",
    "BROWSER_USE_GATEWAY_URL",
    "ZETTLAB_TOOL_GATEWAY_TOKEN",
    "ZET_CHAT_APPEND_URL",
    "ZETTLAB_AGENT_SHARE_ACTION_URL",
    "ZETTLAB_AGENT_ACTION_TOKEN",
    "TOOL_GATEWAY_DOMAIN",
    "TOOL_GATEWAY_USER_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _zettlab_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "local-browser-use")


def test_zettlab_gateway_resolves_managed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _zettlab_env(monkeypatch)
    cfg = BrowserUseBrowserProvider()._get_config_or_none()
    assert cfg is not None
    assert cfg["managed_mode"] is True
    assert cfg["base_url"] == _LOOPBACK
    assert cfg["api_key"] == "local-browser-use"


def test_is_available_true_on_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _zettlab_env(monkeypatch)
    assert BrowserUseBrowserProvider().is_available() is True


def test_direct_key_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    # A self-billed key still wins over the gateway (unless use_gateway is set).
    _zettlab_env(monkeypatch)
    monkeypatch.setenv("BROWSER_USE_API_KEY", "sk-direct")
    cfg = BrowserUseBrowserProvider()._get_config_or_none()
    assert cfg is not None
    assert cfg["managed_mode"] is False
    assert cfg["api_key"] == "sk-direct"


def test_unconfigured_device_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # No gateway, no key, no Nous account → browser-use unavailable.
    with patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=False):
        assert BrowserUseBrowserProvider()._get_config_or_none() is None


def test_old_zettlab_env_does_not_configure_device_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_GATEWAY_URL", _LOOPBACK)
    monkeypatch.setenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "old-placeholder")
    with patch("tools.managed_tool_gateway.managed_nous_tools_enabled", return_value=False):
        assert BrowserUseBrowserProvider()._get_config_or_none() is None
