"""Unit tests for the Zettlab-hosted managed-tool gateway resolver."""

from __future__ import annotations

import pytest

from tools.zettlab_tool_gateway import (
    explicit_vendor_gateway_url,
    resolve_zettlab_tool_gateway,
)


_ENV_KEYS = ("BROWSER_USE_GATEWAY_URL", "ZETTLAB_TOOL_GATEWAY_TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_returns_none_without_gateway_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "placeholder")
    assert resolve_zettlab_tool_gateway("browser-use") is None


def test_returns_none_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_GATEWAY_URL", "http://127.0.0.1:9090/api/v1/browser-use")
    assert resolve_zettlab_tool_gateway("browser-use") is None


def test_resolves_from_explicit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_GATEWAY_URL", "http://127.0.0.1:9090/api/v1/browser-use/")
    monkeypatch.setenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "local-browser-use")
    cfg = resolve_zettlab_tool_gateway("browser-use")
    assert cfg is not None
    assert cfg.vendor == "browser-use"
    # trailing slash trimmed.
    assert cfg.gateway_origin == "http://127.0.0.1:9090/api/v1/browser-use"
    assert cfg.token == "local-browser-use"


def test_never_falls_back_to_nous_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # No explicit URL → no default (unlike the Nous build_vendor_gateway_url).
    monkeypatch.setenv("TOOL_GATEWAY_DOMAIN", "nousresearch.com")
    monkeypatch.setenv("ZETTLAB_TOOL_GATEWAY_TOKEN", "placeholder")
    assert explicit_vendor_gateway_url("browser-use") == ""
    assert resolve_zettlab_tool_gateway("browser-use") is None
