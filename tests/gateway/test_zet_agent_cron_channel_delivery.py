"""Tests for channel: delivery handling in zet_agent_cron."""

import json
from unittest.mock import patch

import gateway.platforms.zet_agent_cron as zc
from gateway.platforms.zet_agent_cron import (
    _handle_channel_delivery,
    _resolve_channel_send_url,
    _send_to_channel,
    _split_channel_targets,
)


# --- Task 1: _split_channel_targets ---

def test_split_only_channel():
    kinds, remaining, invalid = _split_channel_targets("channel:wechat")
    assert kinds == ["wechat"]
    assert remaining == ""
    assert invalid == []


def test_split_mixed():
    kinds, remaining, invalid = _split_channel_targets("origin,channel:wechat")
    assert kinds == ["wechat"]
    assert remaining == "origin"
    assert invalid == []


def test_split_multiple_channels_and_other():
    kinds, remaining, invalid = _split_channel_targets("channel:wechat,telegram:123,channel:feishu")
    assert kinds == ["wechat", "feishu"]
    assert remaining == "telegram:123"
    assert invalid == []


def test_split_no_channel():
    kinds, remaining, invalid = _split_channel_targets("origin,telegram:123")
    assert kinds == []
    assert remaining == "origin,telegram:123"
    assert invalid == []


def test_split_empty_and_whitespace():
    assert _split_channel_targets("") == ([], "", [])
    assert _split_channel_targets(None) == ([], "", [])
    kinds, remaining, invalid = _split_channel_targets(" channel:wechat , origin ")
    assert kinds == ["wechat"]
    assert remaining == "origin"
    assert invalid == []


def test_split_empty_kind_is_invalid_not_dropped():
    kinds, remaining, invalid = _split_channel_targets("channel:,origin")
    assert kinds == []
    assert remaining == "origin"
    assert invalid == ["channel:"]


# --- Task 2: _send_to_channel ---

def test_resolve_channel_send_url(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    assert _resolve_channel_send_url() == "http://127.0.0.1:9090/api/v1/internal/agent/channels/send"


def test_resolve_channel_send_url_missing(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    assert _resolve_channel_send_url() is None


def test_send_to_channel_success(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    captured = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 200, "data": {"message_id": "m1"}}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["token"] = req.get_header("X-zettlab-agent-action-token")
        return FakeResp()

    with patch("urllib.request.urlopen", fake_urlopen):
        err = _send_to_channel("wechat", "hello from cron", "job-7")
    assert err is None
    assert captured["url"].endswith("/api/v1/internal/agent/channels/send")
    assert captured["body"]["target_ref"] == "channel:wechat"
    assert captured["body"]["text"] == "hello from cron"
    assert captured["body"]["source"] == "cron"
    assert captured["body"]["job_id"] == "job-7"
    assert captured["token"] == "tok"


def test_send_to_channel_failure_returns_error(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 95007, "data": {"detail": "not a verified owner"}}).encode()

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        err = _send_to_channel("wechat", "x", "job-7")
    assert err is not None and "verified" in err


def test_send_to_channel_no_token(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    err = _send_to_channel("wechat", "x", "job-7")
    assert err is not None


def test_send_to_channel_exception_returns_error(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    with patch("urllib.request.urlopen", side_effect=Exception("boom")):
        err = _send_to_channel("wechat", "x", "job-7")
    assert err is not None and "boom" in err


# --- Task 3: _handle_channel_delivery ---

def test_handle_channel_delivery_only_channel_success(monkeypatch):
    sent = []
    monkeypatch.setattr(zc, "_send_to_channel",
                        lambda kind, content, job_id: sent.append((kind, content, job_id)) or None)
    job = {"id": "j1", "deliver": "channel:wechat"}
    err, remaining_deliver, had_channel = _handle_channel_delivery(job, "cron body")
    assert had_channel is True
    assert sent == [("wechat", "cron body", "j1")]
    assert err is None
    assert remaining_deliver == ""


def test_handle_channel_delivery_failure_surfaces(monkeypatch):
    monkeypatch.setattr(zc, "_send_to_channel",
                        lambda kind, content, job_id: "channel:wechat delivery failed: not verified")
    job = {"id": "j1", "deliver": "channel:wechat"}
    err, remaining_deliver, had_channel = _handle_channel_delivery(job, "x")
    assert had_channel is True
    assert err is not None and "not verified" in err
    assert remaining_deliver == ""


def test_handle_channel_delivery_no_channel(monkeypatch):
    called = []
    monkeypatch.setattr(zc, "_send_to_channel", lambda *a: called.append(a) or None)
    job = {"id": "j1", "deliver": "origin"}
    err, remaining_deliver, had_channel = _handle_channel_delivery(job, "x")
    assert had_channel is False
    assert called == []
    assert remaining_deliver == "origin"
    assert err is None


def test_handle_channel_delivery_mixed(monkeypatch):
    sent = []
    monkeypatch.setattr(zc, "_send_to_channel", lambda kind, content, job_id: sent.append(kind) or None)
    job = {"id": "j1", "deliver": "origin,channel:feishu"}
    err, remaining_deliver, had_channel = _handle_channel_delivery(job, "x")
    assert had_channel is True
    assert sent == ["feishu"]
    assert remaining_deliver == "origin"
    assert err is None


def test_handle_channel_delivery_invalid_token_is_error(monkeypatch):
    called = []
    monkeypatch.setattr(zc, "_send_to_channel", lambda *a: called.append(a) or None)
    job = {"id": "j1", "deliver": "channel:,origin"}
    err, remaining_deliver, had_channel = _handle_channel_delivery(job, "x")
    assert had_channel is True
    assert called == []
    assert err is not None and "invalid channel target" in err
    assert remaining_deliver == "origin"


# --- chunking: long content split under local-server's rune cap ---

def test_send_to_channel_chunks_long_content(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    posts = []

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 200, "data": {"message_id": "m"}}).encode()

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode())["text"])
        return FakeResp()

    long_content = "水" * 9000  # well over the 4000-rune endpoint cap
    with patch("urllib.request.urlopen", fake_urlopen):
        err = _send_to_channel("wechat", long_content, "job-long")
    assert err is None
    assert len(posts) >= 2, "long content must be split into multiple sends"
    # every posted chunk must be under local-server's 4000-rune cap (no 'text too long')
    assert all(len(p) < 4000 for p in posts), [len(p) for p in posts]


def test_send_to_channel_short_content_single_post(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    posts = []

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 200, "data": {"message_id": "m"}}).encode()

    def fake_urlopen(req, timeout=None):
        posts.append(json.loads(req.data.decode())["text"])
        return FakeResp()

    with patch("urllib.request.urlopen", fake_urlopen):
        err = _send_to_channel("wechat", "该喝水啦 💧", "job-short")
    assert err is None
    assert posts == ["该喝水啦 💧"], "short content must be one unchanged send"


def test_send_to_channel_chunk_failure_surfaces(monkeypatch):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok")
    calls = {"n": 0}

    class OkResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 200, "data": {"message_id": "m"}}).encode()

    class FailResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, *a): return json.dumps({"code": 95005, "data": {"detail": "send failed"}}).encode()

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return FailResp() if calls["n"] == 2 else OkResp()  # 2nd chunk fails

    with patch("urllib.request.urlopen", fake_urlopen):
        err = _send_to_channel("wechat", "水" * 9000, "job-x")
    assert err is not None and "part 2/" in err  # failure surfaced, not silent
