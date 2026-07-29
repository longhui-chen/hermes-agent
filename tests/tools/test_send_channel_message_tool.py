"""Tests for tools/send_channel_message_tool.py."""

import json
from unittest.mock import patch

from tools.send_channel_message_tool import (
    _check_send_channel_message,
    _resolve_send_url,
    send_channel_message_tool,
)


def test_check_requires_both_env(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    assert _check_send_channel_message() is False
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    assert _check_send_channel_message() is True


def test_resolve_send_url(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _resolve_send_url() == "http://127.0.0.1:9090/api/v1/internal/agent/channels/send"


def test_missing_args_returns_error(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    assert "error" in json.loads(send_channel_message_tool({"text": "hi"}))
    assert "error" in json.loads(send_channel_message_tool({"target_ref": "channel:wechat"}))


def test_success_posts_and_returns_message_id(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"code": 200, "data": {"message_id": "m1"}}).encode()

    def fake_urlopen(req, timeout=None):
        captured["method"] = req.get_method()
        captured["url"] = req.full_url
        captured["body"] = req.data
        return FakeResp()

    with patch("urllib.request.urlopen", fake_urlopen):
        out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "你好"}))
    assert out.get("message_id") == "m1"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/v1/internal/agent/channels/send")
    body = json.loads(captured["body"].decode())
    assert body["target_ref"] == "channel:wechat" and body["text"] == "你好"


def test_failure_surfaces_error(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"code": 95007, "data": {"detail": "not a verified owner"}}).encode()

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "你好"}))
    assert "error" in out


def test_exception_does_not_raise(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    with patch("urllib.request.urlopen", side_effect=Exception("boom")):
        out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "hi"}))
    assert "error" in out


def test_long_text_chunked_into_multiple_posts(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    posts = []

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"code": 200, "data": {"message_id": "m"}}).encode()

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode())["text"])
        return FakeResp()

    with patch("urllib.request.urlopen", fake_urlopen):
        out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "水" * 9000}))
    assert "message_id" in out and out.get("parts", 1) >= 2
    assert len(posts) >= 2
    assert all(len(p) < 4000 for p in posts), [len(p) for p in posts]


def test_profile_scope_flow_works_with_poisoned_environ(monkeypatch):
    """Shared gateway mode: values live only in the profile secret scope while
    os.environ holds another profile's stale decoys — the send must be built
    entirely from the scope."""
    from tests.tools._profile_scope import mux_profile_scope, request_fingerprint

    scope = {
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    }
    seen = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"code": 200, "data": {"message_id": "m1"}}).encode()

    def fake_urlopen(req, timeout=None):
        seen["req"] = req
        return FakeResp()

    with mux_profile_scope(monkeypatch, scope, poison_environ=True):
        assert _check_send_channel_message() is True
        with patch("urllib.request.urlopen", fake_urlopen):
            out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "你好"}))

    assert out.get("message_id") == "m1"
    req = seen["req"]
    assert req.full_url == "http://127.0.0.1:9420/api/v1/internal/agent/channels/send"
    assert req.get_header("X-zettlab-agent-action-token") == scope["ZETTLAB_AGENT_ACTION_TOKEN"]
    assert "stale-" not in request_fingerprint(req)
    assert getattr(_check_send_channel_message, "_profile_scope_sensitive") is True


def test_short_text_single_post_no_parts(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    posts = []

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"code": 200, "data": {"message_id": "m1"}}).encode()

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode())["text"])
        return FakeResp()

    with patch("urllib.request.urlopen", fake_urlopen):
        out = json.loads(send_channel_message_tool({"target_ref": "channel:wechat", "text": "你好"}))
    assert out == {"message_id": "m1"}  # no 'parts' key for single send
    assert posts == ["你好"]
