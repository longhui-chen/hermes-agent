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


def test_check_flips_per_missing_scope_secret(monkeypatch):
    """Scope-authoritative gate test: scope filled + os.environ purged, then
    each secret removed in turn must close the gate. A poison-style test
    asserting only `check() is True` cannot tell "read the scope" from "read
    a non-empty poison value in os.environ" — this construction can: with the
    environ purged, an os.environ-reading regression sees nothing and the
    full-scope case goes False."""
    from tests.tools._profile_scope import mux_profile_scope

    full = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    with mux_profile_scope(monkeypatch, full):
        assert _check_list_my_channels() is True
    for missing in full:
        with mux_profile_scope(monkeypatch, {**full, missing: ""}):
            assert _check_list_my_channels() is False, f"gate stayed open without {missing}"


def test_profile_scope_flow_works_with_poisoned_environ(monkeypatch):
    """Shared gateway mode: values live only in the profile secret scope while
    os.environ holds another profile's stale decoys — the tool must build its
    request entirely from the scope."""
    from tests.tools._profile_scope import mux_profile_scope, request_fingerprint

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    fake = {"code": 200, "data": {"installed_channels": [{"kind": "wechat"}]}}
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        return FakeResp()

    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        assert _check_list_my_channels() is True
        with patch("urllib.request.urlopen", fake_urlopen):
            out = list_my_channels_tool({})

    parsed = json.loads(out)
    assert parsed["installed_channels"][0]["kind"] == "wechat"
    req = seen["req"]
    assert req.full_url == "http://127.0.0.1:9420/api/v1/internal/agent/channels"
    assert req.get_header("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)
    assert getattr(_check_list_my_channels, "_profile_scope_sensitive") is True


def test_tool_passes_through_available_kinds(monkeypatch):
    """区域感知可连清单必须透传（governor 库存判定与主模型口径接地都依赖它）；
    老版 local-server 无此字段时输出也不携带（tolerant）。"""
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    fake = {"code": 200, "data": {
        "installed_channels": [{"kind": "feishu", "name": "飞书", "status": "online"}],
        "available_kinds": ["wecom", "wechat"],
    }}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        parsed = json.loads(list_my_channels_tool({}))
    assert parsed["available_kinds"] == ["wecom", "wechat"]

    legacy = {"code": 200, "data": {"installed_channels": []}}

    class LegacyResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(legacy).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=LegacyResp()):
        parsed = json.loads(list_my_channels_tool({}))
    assert "available_kinds" not in parsed
