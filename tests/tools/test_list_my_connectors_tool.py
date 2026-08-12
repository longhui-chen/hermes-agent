"""Tests for tools/list_my_connectors_tool.py."""

import json
from unittest.mock import patch

import pytest

from tools.list_my_connectors_tool import (
    _check_list_my_connectors,
    _resolve_connectors_url,
    list_my_connectors_tool,
)


def test_check_requires_both_env(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    assert _check_list_my_connectors() is False

    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _check_list_my_connectors() is False  # token still missing

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    assert _check_list_my_connectors() is True


def test_resolve_connectors_url_via_urlsplit(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _resolve_connectors_url() == "http://127.0.0.1:9090/api/v1/internal/agent/connectors"


def test_resolve_connectors_url_missing_env(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    assert _resolve_connectors_url() is None


def test_tool_returns_connectors(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    # Pins the local-server response envelope contract being developed in
    # parallel: {code, data:{connectors:[{provider, state, ...}]}}.
    fake = {"code": 200, "data": {"connectors": [
        {"provider": "gmail", "state": "connected"},
        {"provider": "notion", "state": "not_connected"}]}}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        out = list_my_connectors_tool({})
    # Tool returns a JSON STRING (same contract as send_message/clarify), not a
    # raw dict — a dict reaches the model provider as non-string content and is
    # rejected (deepseek-v4/MMGPT → 400).
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "connectors" in parsed
    assert parsed["connectors"][0]["provider"] == "gmail"
    assert parsed["connectors"][0]["state"] == "connected"
    assert parsed["connectors"][1]["state"] == "not_connected"


def test_tool_errors_when_url_missing(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    out = list_my_connectors_tool({})
    assert isinstance(out, str)
    assert "error" in json.loads(out)


def test_tool_errors_when_token_missing(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    out = list_my_connectors_tool({})
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "error" in parsed
    assert "token" in parsed["error"]


def test_tool_errors_on_unexpected_response(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    fake = {"code": 200, "data": {"something_else": []}}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        out = list_my_connectors_tool({})
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "error" in parsed
    assert "unexpected response" in parsed["error"]


def test_tool_errors_on_timeout(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        out = list_my_connectors_tool({})
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert "error" in parsed
    assert "failed to fetch connectors" in parsed["error"]


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
        assert _check_list_my_connectors() is True
    for missing in full:
        with mux_profile_scope(monkeypatch, {**full, missing: ""}):
            assert _check_list_my_connectors() is False, f"gate stayed open without {missing}"


def test_profile_scope_flow_works_with_poisoned_environ(monkeypatch):
    """Shared gateway mode: values live only in the profile secret scope while
    os.environ holds another profile's stale decoys — the tool must build its
    request entirely from the scope."""
    from tests.tools._profile_scope import mux_profile_scope, request_fingerprint

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    fake = {"code": 200, "data": {"connectors": [{"provider": "github", "state": "connected"}]}}
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(fake).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        return FakeResp()

    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        assert _check_list_my_connectors() is True
        with patch("urllib.request.urlopen", fake_urlopen):
            out = list_my_connectors_tool({})

    parsed = json.loads(out)
    assert parsed["connectors"][0]["provider"] == "github"
    req = seen["req"]
    assert req.full_url == "http://127.0.0.1:9420/api/v1/internal/agent/connectors"
    assert req.get_header("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)
    assert getattr(_check_list_my_connectors, "_profile_scope_sensitive") is True


def test_list_my_connectors_in_core_tools_and_catalog():
    """Reachability contract (see test_toolsets.py's profile-scope guard):
    the composite must list the tool AND the catalog must have an entry for
    its toolset, otherwise the real path silently drops it."""
    import toolsets

    assert "list_my_connectors" in toolsets._HERMES_CORE_TOOLS
    assert "list_my_connectors" in toolsets.TOOLSETS["zettlab_connectors"]["tools"]
    assert "list_my_connectors" in toolsets.resolve_toolset("hermes-zet-agent")
