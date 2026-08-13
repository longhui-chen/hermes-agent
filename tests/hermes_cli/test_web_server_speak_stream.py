"""/api/audio/speak-stream — desktop streaming TTS over WebSocket."""

from __future__ import annotations

import json
import time
from urllib.parse import urlencode

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server


@pytest.fixture
def stream_client(monkeypatch, _isolate_hermes_home):
    previous_auth_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False

    client = TestClient(web_server.app)
    try:
        yield client
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        if previous_auth_required is None:
            if hasattr(web_server.app.state, "auth_required"):
                delattr(web_server.app.state, "auth_required")
        else:
            web_server.app.state.auth_required = previous_auth_required


def _url(token: str | None = None, profile: str | None = None) -> str:
    params = {"token": token or web_server._SESSION_TOKEN}
    if profile:
        params["profile"] = profile
    return f"/api/audio/speak-stream?{urlencode(params)}"


class _FakeStreamer:
    sample_rate = 24000
    channels = 1

    def __init__(self, chunks):
        self.chunks = chunks
        self.requests: list[str] = []

    def stream(self, text):
        self.requests.append(text)
        yield from self.chunks


def _patch_provider(monkeypatch, streamer, cap=4000):
    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", lambda cfg: streamer)
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda cfg: "fake")
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda provider, cfg: cap)






def test_streams_pcm_frames_then_end(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x01\x02\x03\x04", b"\x05\x06"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        start = conn.receive_json()
        assert start == {"type": "start", "sample_rate": 24000, "channels": 1}

        conn.send_text(json.dumps({"text": "Hello there.", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02\x03\x04"
        assert conn.receive_bytes() == b"\x05\x06"
        assert conn.receive_json() == {"type": "end"}

    assert streamer.requests == ["Hello there."]


def test_stream_installs_target_profile_secret_scope(stream_client, monkeypatch):
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_beta"
    worker_home.mkdir(parents=True)
    (worker_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (worker_home / ".env").write_text(
        "ZETTLAB_AGENT_ACTION_TOKEN=worker-stream-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    seen = []

    class _ScopedStreamer(_FakeStreamer):
        def stream(self, text):
            scope = current_secret_scope()
            seen.append(scope["ZETTLAB_AGENT_ACTION_TOKEN"])
            yield from super().stream(text)

    streamer = _ScopedStreamer([b"\x01\x02"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url(profile="worker_beta")) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "Hello scoped stream.", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02"
        assert conn.receive_json() == {"type": "end"}

    assert seen == ["worker-stream-token"]


def test_stream_current_profile_preserves_process_env(stream_client, monkeypatch):
    from agent.secret_scope import get_secret

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "process-stream-token")
    seen = []

    class _EnvStreamer(_FakeStreamer):
        def stream(self, text):
            seen.append(get_secret("ZETTLAB_AGENT_ACTION_TOKEN"))
            yield from super().stream(text)

    streamer = _EnvStreamer([b"\x01\x02"])
    _patch_provider(monkeypatch, streamer)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "Hello current stream.", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02"
        assert conn.receive_json() == {"type": "end"}

    assert seen == ["process-stream-token"]


def test_stream_target_config_env_ref_does_not_expand_process_secret(
    stream_client, monkeypatch
):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_beta"
    worker_home.mkdir(parents=True)
    (worker_home / "config.yaml").write_text(
        "tts:\n  provider: openai\n  api_key: ${env:OPENAI_API_KEY}\n",
        encoding="utf-8",
    )
    (worker_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "dashboard-openai-key")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    seen = {}
    streamer = _FakeStreamer([b"\x01\x02"])

    def resolve(cfg):
        seen["api_key"] = cfg.get("api_key")
        return streamer

    monkeypatch.setattr("tools.tts_streaming.resolve_streaming_provider", resolve)
    monkeypatch.setattr("tools.tts_tool._resolve_max_text_length", lambda *_args: 4000)

    with stream_client.websocket_connect(_url(profile="worker_beta")) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(json.dumps({"text": "Hello worker stream.", "done": True}))
        assert conn.receive_bytes() == b"\x01\x02"
        assert conn.receive_json() == {"type": "end"}

    assert seen["api_key"] == "${env:OPENAI_API_KEY}"








def test_long_text_is_split_across_provider_requests(stream_client, monkeypatch):
    streamer = _FakeStreamer([b"\x00\x00"])
    _patch_provider(monkeypatch, streamer, cap=24)

    with stream_client.websocket_connect(_url()) as conn:
        assert conn.receive_json()["type"] == "start"
        conn.send_text(
            json.dumps(
                {"text": "First sentence here. Second sentence here. Third one.", "done": True}
            )
        )
        # One PCM frame per split piece, then end.
        frames = 0
        while True:
            message = conn.receive()
            if message.get("bytes") is not None:
                frames += 1
            else:
                assert json.loads(message["text"]) == {"type": "end"}
                break

    assert len(streamer.requests) > 1
    assert frames == len(streamer.requests)
    # Nothing lost in the split: every sentence reached the provider.
    joined = " ".join(streamer.requests)
    for fragment in ("First sentence here.", "Second sentence here.", "Third one."):
        assert fragment in joined


def test_split_text_respects_cap_and_preserves_content():
    text = "Alpha beta. Gamma delta epsilon. Zeta eta theta iota kappa."
    pieces = web_server._split_text_for_speak_stream(text, 30)
    assert pieces
    assert all(len(piece) <= 30 for piece in pieces)
    joined = " ".join(pieces)
    for word in text.replace(".", "").split():
        assert word in joined
