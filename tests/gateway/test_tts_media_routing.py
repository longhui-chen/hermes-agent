"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

from types import SimpleNamespace
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    FeishuQuoteLease,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.run import (
    GatewayRunner,
    _non_conversational_metadata,
    _non_conversational_reply_to,
)
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


class _FeishuMediaRoutingAdapter(_MediaRoutingAdapter):
    def __init__(self):
        super().__init__()
        self.platform = Platform.FEISHU


def _feishu_event(message_id="om-question"):
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc-group",
        chat_type="group",
        thread_id="om-old-root",
    )
    return MessageEvent(
        text="请处理",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(
        return_value=f"[[audio_as_voice]]\nMEDIA:{media_file}"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_feishu_text_reply_consumes_quote_before_attachment(tmp_path, monkeypatch):
    adapter = _FeishuMediaRoutingAdapter()
    event = _feishu_event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter._message_handler = AsyncMock(return_value=f"已整理\nMEDIA:{media_file}")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    assert "reply_to_message_id" in adapter.send.await_args.kwargs["metadata"]
    assert "reply_to_message_id" not in adapter.send_document.await_args.kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_base_delivery_observes_quote_consumed_by_earlier_turn_send(
    tmp_path, monkeypatch
):
    adapter = _FeishuMediaRoutingAdapter()
    event = _feishu_event()
    lease = FeishuQuoteLease("om-question")
    event.source._feishu_quote_lease = lease
    lease.consume(
        {"reply_to_message_id": "om-question"},
        SendResult(success=True, message_id="interim"),
    )
    adapter._message_handler = AsyncMock(return_value="最终答案")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))

    await adapter._process_message_background(event, build_session_key(event.source))

    assert "reply_to_message_id" not in adapter.send.await_args.kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_media_only_reply_quotes_first_attachment(tmp_path, monkeypatch):
    adapter = _FeishuMediaRoutingAdapter()
    event = _feishu_event()
    first = _allowed_media_path(tmp_path, monkeypatch, "first.pdf")
    second = _allowed_media_path(tmp_path, monkeypatch, "second.pdf")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{first}\nMEDIA:{second}")
    adapter.send_document = AsyncMock(side_effect=[
        SendResult(success=True, message_id="doc-1"),
        SendResult(success=True, message_id="doc-2"),
    ])

    await adapter._process_message_background(event, build_session_key(event.source))

    calls = adapter.send_document.await_args_list
    assert calls[0].kwargs["metadata"]["reply_to_message_id"] == "om-question"
    assert "reply_to_message_id" not in calls[1].kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_media_only_reply_keeps_quote_until_attachment_succeeds(tmp_path, monkeypatch):
    adapter = _FeishuMediaRoutingAdapter()
    event = _feishu_event()
    first = _allowed_media_path(tmp_path, monkeypatch, "first.pdf")
    second = _allowed_media_path(tmp_path, monkeypatch, "second.pdf")
    third = _allowed_media_path(tmp_path, monkeypatch, "third.pdf")
    adapter._message_handler = AsyncMock(
        return_value=f"MEDIA:{first}\nMEDIA:{second}\nMEDIA:{third}"
    )
    adapter.send_document = AsyncMock(side_effect=[
        SendResult(success=False, error="temporary"),
        SendResult(success=True, message_id="doc-2"),
        SendResult(success=True, message_id="doc-3"),
    ])

    await adapter._process_message_background(event, build_session_key(event.source))

    calls = adapter.send_document.await_args_list
    assert calls[0].kwargs["metadata"]["reply_to_message_id"] == "om-question"
    assert calls[1].kwargs["metadata"]["reply_to_message_id"] == "om-question"
    assert "reply_to_message_id" not in calls[2].kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_base_image_batch_releases_failed_reservation_without_returning_result():
    adapter = _FeishuMediaRoutingAdapter()
    lease = FeishuQuoteLease("om-question")
    metadata = {
        "reply_to_message_id": "om-question",
        "_feishu_quote_lease": lease,
    }
    adapter.send_image = AsyncMock(side_effect=[
        RuntimeError("temporary"),
        SendResult(success=True, message_id="image-2"),
        SendResult(success=True, message_id="image-3"),
    ])

    result = await adapter.send_multiple_images(
        "oc-group",
        [("https://example.com/1.png", ""),
         ("https://example.com/2.png", ""),
         ("https://example.com/3.png", "")],
        metadata=metadata,
    )

    first, second, third = adapter.send_image.await_args_list
    assert result is None
    assert first.kwargs["metadata"]["reply_to_message_id"] == "om-question"
    assert second.kwargs["metadata"]["reply_to_message_id"] == "om-question"
    assert "reply_to_message_id" not in third.kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_voice_is_flat_and_text_keeps_quote(tmp_path, monkeypatch):
    adapter = _FeishuMediaRoutingAdapter()
    event = _feishu_event()
    event.message_type = MessageType.VOICE
    adapter._message_handler = AsyncMock(return_value="文字答案")
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="text"))
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    audio_path = tmp_path / "reply.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr("tools.tts_tool.check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        lambda **_kwargs: json.dumps({"file_path": str(audio_path)}),
    )

    await adapter._process_message_background(event, build_session_key(event.source))

    assert "reply_to_message_id" not in adapter.play_tts.await_args.kwargs["metadata"]
    assert "reply_to_message_id" in adapter.send.await_args.kwargs["metadata"]


def test_slack_and_telegram_reply_routing_is_unchanged():
    from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_source

    for platform, chat_type, thread_id in (
        (Platform.SLACK, "channel", "thread-1"),
        (Platform.TELEGRAM, "dm", "topic-1"),
    ):
        source = SessionSource(
            platform=platform,
            chat_id="chat-1",
            chat_type=chat_type,
            thread_id=thread_id,
            message_id="msg-1",
        )
        event = MessageEvent(text="hi", source=source, message_id="msg-1")
        metadata = _thread_metadata_for_source(source, "msg-1")
        assert "reply_to_message_id" not in metadata
        assert _reply_anchor_for_event(event) == "msg-1"


def test_non_conversational_feishu_sends_are_flat_without_changing_thread_route():
    lease = FeishuQuoteLease("om-question")
    metadata = {
        "thread_id": "om-root",
        "reply_to_message_id": "om-question",
        "_feishu_quote_lease": lease,
        "notify": True,
    }

    assert _non_conversational_metadata(
        metadata, platform=Platform.FEISHU
    ) == {"thread_id": "om-root", "notify": True}
    assert _non_conversational_reply_to(
        "om-question", platform=Platform.FEISHU
    ) is None


def test_non_conversational_slack_and_telegram_routing_is_unchanged():
    for platform in (Platform.SLACK, Platform.TELEGRAM):
        metadata = {"thread_id": "thread-1", "notify": True}
        assert _non_conversational_metadata(
            metadata, platform=platform
        ) is metadata
        assert _non_conversational_reply_to(
            "message-1", platform=platform
        ) == "message-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("platform", "chat_type", "thread_id", "expected"),
    [
        (Platform.SLACK, "channel", "thread-1", {"thread_id": "thread-1", "notify": True}),
        (
            Platform.TELEGRAM,
            "dm",
            "topic-1",
            {
                "thread_id": "topic-1",
                "telegram_dm_topic_reply_fallback": True,
                "direct_messages_topic_id": "topic-1",
                "telegram_reply_to_message_id": "msg-1",
                "notify": True,
            },
        ),
    ],
)
async def test_other_platform_media_metadata_stays_unchanged(
    platform, chat_type, thread_id, expected, tmp_path, monkeypatch
):
    adapter = _MediaRoutingAdapter()
    adapter.platform = platform
    source = SessionSource(
        platform=platform,
        chat_id="chat-1",
        chat_type=chat_type,
        thread_id=thread_id,
        message_id="msg-1",
    )
    event = MessageEvent(text="file", source=source, message_id="msg-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.pdf")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.send_document.await_args.kwargs["metadata"] == expected


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_delivery_blocks_media_path_outside_allowed_roots(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    allowed_root = tmp_path / "media-cache"
    allowed_root.mkdir()
    secret = tmp_path / "outside.pdf"
    secret.write_bytes(b"%PDF secret")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (allowed_root,),
    )
    # This test exercises the strict-allowlist path; force strict mode on
    # and disable recency trust so the freshly-written tmp_path file is not
    # auto-accepted by the trust window. (Recency trust is covered separately
    # in test_platform_base.py. The public default flipped to non-strict in
    # 2026-05; this test pins strict on explicitly.)
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{secret}",
        event,
        adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()


class _DiscordMediaFailureAdapter(BasePlatformAdapter):
    """Minimal adapter to exercise non-streaming MEDIA failure notification."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.notices: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        self.notices.append(content or "")
        return SendResult(success=True, message_id="notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_non_streaming_media_failure_notifies_user(tmp_path, monkeypatch):
    """Attachmentless send_video results must surface a user-visible notice (#66797)."""
    adapter = _DiscordMediaFailureAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "clip.mp4")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_video = AsyncMock(
        return_value=SendResult(
            success=False,
            error="Discord accepted the message but attached no files (clip.mp4)",
        )
    )
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_multiple_images = AsyncMock()

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_video.assert_awaited_once()
    assert adapter.notices == ["⚠️ Couldn't deliver the video attachment."]


@pytest.mark.asyncio
async def test_non_streaming_remote_image_batch_failure_notifies_user():
    adapter = _DiscordMediaFailureAdapter()
    event = _event()
    remote_image = "https://cdn.example/chart.png?signature=secret-token"
    adapter._message_handler = AsyncMock(
        return_value=f"![chart]({remote_image})"
    )
    adapter.send_multiple_images = AsyncMock(return_value=False)

    await adapter._process_message_background(event, build_session_key(event.source))

    sent_images = adapter.send_multiple_images.await_args.kwargs["images"]
    assert sent_images == [(remote_image, "chart")], (
        "夹具必须真实进入远程图片批量发送分支"
    )
    assert adapter.notices == [
        "⚠️ Couldn't deliver the file attachment (chart.png)."
    ]
    assert "secret-token" not in adapter.notices[0]


class _DiscordMediaFailureAdapter(BasePlatformAdapter):
    """Minimal adapter to exercise non-streaming MEDIA failure notification."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.DISCORD)
        self.notices: list[str] = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        self.notices.append(content or "")
        return SendResult(success=True, message_id="notice")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}
