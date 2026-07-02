"""Tests for tools/list_my_channels_tool.py."""

import json
from unittest.mock import patch

import pytest

from tools.list_my_channels_tool import (
    _check_list_my_channels,
    _resolve_channels_url,
    list_my_channels_tool,
)


def test_check_requires_both_env(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    assert _check_list_my_channels() is False

    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _check_list_my_channels() is False  # token still missing

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    assert _check_list_my_channels() is True


def test_resolve_channels_url_via_urlsplit(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _resolve_channels_url() == "http://127.0.0.1:9090/api/v1/internal/agent/channels"


def test_resolve_channels_url_missing_env(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    assert _resolve_channels_url() is None


def test_tool_returns_installed_channels(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    fake = {"code": 200, "data": {"installed_channels": [
        {"kind": "wechat", "name": "我的微信", "status": "online"}]}}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        out = list_my_channels_tool({})
    # Tool returns a JSON STRING (same contract as send_message/clarify), not a
    # raw dict — a dict reaches the model provider as non-string content and is
    # rejected (deepseek-v4/MMGPT → 400).
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "installed_channels" in parsed
    assert parsed["installed_channels"][0]["kind"] == "wechat"


def test_tool_errors_when_env_missing(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    out = list_my_channels_tool({})
    assert isinstance(out, str)
    assert "error" in json.loads(out)
