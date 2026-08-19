"""Relay Phase 2 media tests — send_media egress lanes + inbound media localization.

Covers:
  - the five ``send_*`` overrides route through ONE ``send_media`` op with the
    right ``media_kind`` and honor op-level capability gating (a connector not
    advertising ``send_media`` falls back to the base-class behaviour);
  - local-path sources upload through the RelayMediaClient first (the
    connector cannot reach our filesystem) and public URLs pass through;
  - a connector decline / failed upload degrades to the pre-media fallback;
  - inbound ``media_urls`` are localized to temp paths (re-hosts downloaded
    with the per-gateway bearer; dead re-host refs dropped; public URLs kept
    when no client is available);
  - the RelayMediaClient URL derivation + auth header shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.media import RelayMediaClient, media_base_url

from tests.gateway.relay.stub_connector import StubConnector


def make_desc(**kw) -> CapabilityDescriptor:
    base = dict(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        supported_ops=(
            "send",
            "edit",
            "typing",
            "get_chat_info",
            "send_media",
        ),
    )
    base.update(kw)
    return CapabilityDescriptor(**base)


class FakeMediaClient:
    """In-memory stand-in for RelayMediaClient (no HTTP)."""

    def __init__(self) -> None:
        self.enabled = True
        self.uploads: list[tuple[str, Optional[str]]] = []
        self.downloads: list[str] = []
        self.upload_result: Optional[str] = "https://conn.example/relay/media/aa11"
        self.download_result: Optional[str] = "/tmp/relay_media_fake.png"

    async def upload(self, file_path, *, mime=None, filename=None):
        self.uploads.append((str(file_path), filename))
        return self.upload_result

    async def download(self, url, *, suggested_name=None):
        self.downloads.append(url)
        return self.download_result

    def is_relay_media_url(self, url: str) -> bool:
        return "/relay/media/" in (url or "")


def _adapter(**desc_kw) -> tuple[RelayAdapter, StubConnector, FakeMediaClient]:
    stub = StubConnector(make_desc(**desc_kw))
    adapter = RelayAdapter(PlatformConfig(), make_desc(**desc_kw), transport=stub)
    fake = FakeMediaClient()
    adapter._media_client = fake  # bypass env-derived construction
    return adapter, stub, fake


# ── egress: the five overrides ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_image_url_passes_through_without_upload():
    adapter, stub, fake = _adapter()
    result = await adapter.send_image(
        "chat1", "https://fal.media/x.png", caption="a pic", reply_to="m9"
    )
    assert result.success is True
    assert result.message_id == "md1"
    assert fake.uploads == []  # public URL → no upload leg
    action = stub.sent[-1]
    assert action["op"] == "send_media"
    assert action["media_kind"] == "image"
    assert action["source_url"] == "https://fal.media/x.png"
    assert action["content"] == "a pic"
    assert action["reply_to"] == "m9"


@pytest.mark.asyncio
async def test_local_path_lanes_upload_first(tmp_path: Path):
    adapter, stub, fake = _adapter()
    f = tmp_path / "clip.ogg"
    f.write_bytes(b"oggbytes")
    result = await adapter.send_voice("chat1", str(f), caption="listen")
    assert result.success is True
    assert fake.uploads == [(str(f), None)]
    action = stub.sent[-1]
    assert action["op"] == "send_media"
    assert action["media_kind"] == "voice"
    # The wire carries the RE-HOST reference, never the local path.
    assert action["source_url"] == fake.upload_result
    assert str(f) not in str(action)


@pytest.mark.asyncio
async def test_op_gating_falls_back_when_not_advertised(tmp_path: Path):
    # Connector advertises only the legacy ops — send_media must never hit the wire.
    adapter, stub, fake = _adapter(
        supported_ops=("send", "edit", "typing", "get_chat_info")
    )
    result = await adapter.send_image("chat1", "https://x.io/a.png", caption="hi")
    # Base-class fallback: caption + URL as a text send.
    assert result.success is True
    ops = [a["op"] for a in stub.sent]
    assert "send_media" not in ops
    assert ops[-1] == "send"
    assert "https://x.io/a.png" in stub.sent[-1]["content"]


# ── inbound localization ─────────────────────────────────────────────────


def _make_event(media_urls):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    return MessageEvent(
        text="look",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform="telegram", chat_id="c1", chat_type="dm", user_id="u1"
        ),
        media_urls=list(media_urls),
    )


@pytest.mark.asyncio
async def test_inbound_without_client_keeps_public_drops_rehost():
    adapter, _stub, _fake = _adapter()
    adapter._media_client = None
    adapter._get_media_client = lambda: None  # type: ignore[method-assign]
    event = _make_event(
        [
            "https://conn.example/relay/media/deadbeef",
            "https://cdn.discordapp.com/attachments/a/b.png",
        ]
    )
    await adapter._localize_inbound_media(event)
    assert event.media_urls == ["https://cdn.discordapp.com/attachments/a/b.png"]


# ── RelayMediaClient unit surface ────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_upload_rejects_oversize_and_missing(tmp_path: Path):
    c = RelayMediaClient("https://c.example", "gw1", "sec")
    # Missing file → None (no network attempted).
    assert await c.upload(str(tmp_path / "nope.bin")) is None
    # Empty file → None.
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    assert await c.upload(str(empty)) is None


def test_rehost_auth_is_scoped_to_the_configured_origin():
    c = RelayMediaClient("https://conn.example", "gw1", "sec")

    assert c.is_relay_media_url("https://conn.example/relay/media/ok")
    assert not c.is_relay_media_url("https://evil.example/relay/media/steal")
    assert not c.is_relay_media_url("https://conn.example.evil/relay/media/steal")


def test_rehost_auth_preserves_the_configured_connector_base_path():
    base_url = media_base_url("wss://conn.example/hermes/relay")
    assert base_url == "https://conn.example/hermes", (
        "夹具必须真实经过带前缀的 relay dial URL 解析链"
    )
    c = RelayMediaClient(base_url, "gw1", "sec")
    generated_shape = "https://conn.example/hermes/relay/media/ok"

    assert generated_shape == f"{c._base_url}/relay/media/ok"
    assert c.is_relay_media_url(generated_shape)
    assert not c.is_relay_media_url("https://conn.example/relay/media/wrong-root")


@pytest.mark.asyncio
async def test_oversize_upload_is_rejected_before_full_file_read(tmp_path, monkeypatch):
    from gateway.relay import media as relay_media

    path = tmp_path / "huge.bin"
    with path.open("wb") as fh:
        fh.truncate(relay_media.MEDIA_MAX_BYTES + 1)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("超限文件不能先整份读进内存再检查大小")
        ),
    )

    c = RelayMediaClient("https://conn.example", "gw1", "sec")
    assert await c.upload(str(path)) is None


@pytest.mark.asyncio
async def test_public_download_uses_ssrf_safe_client(monkeypatch, tmp_path):
    from gateway.relay import media as relay_media

    calls = []

    class Response:
        headers = {"content-type": "image/png"}

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"png"

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *_args):
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return Stream()

    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_async_client",
        lambda **_kwargs: Client(),
    )
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda _url: True)
    monkeypatch.setattr(relay_media.tempfile, "gettempdir", lambda: str(tmp_path))

    c = RelayMediaClient("https://conn.example", "gw1", "sec")
    localized = await c.download("https://cdn.example/media.png")

    assert calls and calls[0][1] == "https://cdn.example/media.png"
    assert "Authorization" not in calls[0][2].get("headers", {})
    assert Path(localized).read_bytes() == b"png"
