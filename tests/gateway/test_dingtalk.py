"""Tests for DingTalk platform adapter."""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


class _FakeDingTalkModel:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeChatbotMessage(SimpleNamespace):
    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            message_id=data.get("msgId") or data.get("messageId") or data.get("message_id") or "",
            conversation_id=data.get("conversationId") or data.get("conversation_id") or "",
            conversation_type=str(data.get("conversationType") or data.get("conversation_type") or "1"),
            sender_id=data.get("senderId") or data.get("sender_id") or "",
            sender_staff_id=data.get("senderStaffId") or data.get("sender_staff_id") or data.get("senderId") or "",
            sender_nick=data.get("senderNick") or data.get("sender_nick") or "",
            text=data.get("text") or "",
            rich_text=data.get("richText") or data.get("rich_text"),
            rich_text_content=data.get("richTextContent") or data.get("rich_text_content"),
            session_webhook=data.get("sessionWebhook") or data.get("session_webhook") or "",
            session_webhook_expired_time=data.get("sessionWebhookExpiredTime") or data.get("session_webhook_expired_time") or 0,
            create_at=data.get("createAt") or data.get("create_at") or 0,
            at_users=data.get("atUsers") or data.get("at_users") or [],
            is_in_at_list=bool(data.get("isInAtList") or data.get("is_in_at_list")),
        )


@pytest.fixture(autouse=True)
def _fake_dingtalk_optional_sdks(monkeypatch):
    """Keep DingTalk adapter tests hermetic when optional SDKs are absent."""
    import plugins.platforms.dingtalk.adapter as dt

    card_models = SimpleNamespace(**{
        name: _FakeDingTalkModel
        for name in (
            "CreateCardRequest",
            "CreateCardRequestCardData",
            "CreateCardRequestImGroupOpenSpaceModel",
            "CreateCardRequestImRobotOpenSpaceModel",
            "CreateCardHeaders",
            "DeliverCardRequest",
            "DeliverCardRequestImGroupOpenDeliverModel",
            "DeliverCardRequestImRobotOpenDeliverModel",
            "DeliverCardHeaders",
            "StreamingUpdateRequest",
            "StreamingUpdateHeaders",
        )
    })
    robot_models = SimpleNamespace(**{
        name: _FakeDingTalkModel
        for name in (
            "RobotReplyEmotionRequestTextEmotion",
            "RobotReplyEmotionRequest",
            "RobotReplyEmotionHeaders",
            "RobotRecallEmotionRequestTextEmotion",
            "RobotRecallEmotionRequest",
            "RobotRecallEmotionHeaders",
            "RobotMessageFileDownloadRequest",
            "RobotMessageFileDownloadHeaders",
        )
    })

    monkeypatch.setattr(dt, "ChatbotMessage", _FakeChatbotMessage, raising=False)
    monkeypatch.setattr(
        dt,
        "AckMessage",
        SimpleNamespace(STATUS_OK=200, STATUS_SYSTEM_EXCEPTION=500),
        raising=False,
    )
    monkeypatch.setattr(dt, "tea_util_models", SimpleNamespace(RuntimeOptions=_FakeDingTalkModel), raising=False)
    monkeypatch.setattr(dt, "dingtalk_card_models", card_models, raising=False)
    monkeypatch.setattr(dt, "dingtalk_robot_models", robot_models, raising=False)


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------


class TestDingTalkRequirements:


    def test_returns_false_when_env_vars_missing(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.platforms.dingtalk.adapter.DINGTALK_STREAM_AVAILABLE", True
        )
        monkeypatch.setattr("plugins.platforms.dingtalk.adapter.HTTPX_AVAILABLE", True)
        monkeypatch.delenv("DINGTALK_CLIENT_ID", raising=False)
        monkeypatch.delenv("DINGTALK_CLIENT_SECRET", raising=False)
        from plugins.platforms.dingtalk.adapter import check_dingtalk_requirements
        assert check_dingtalk_requirements() is False


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


class TestDingTalkAdapterInit:

    def test_reads_config_from_extra(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        config = PlatformConfig(
            enabled=True,
            extra={"client_id": "cfg-id", "client_secret": "cfg-secret"},
        )
        adapter = DingTalkAdapter(config)
        assert adapter._client_id == "cfg-id"
        assert adapter._client_secret == "cfg-secret"
        assert adapter.name == "Dingtalk"  # base class uses .title()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:

    def test_first_message_not_duplicate(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        assert adapter._dedup.is_duplicate("msg-1") is False


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


class TestSend:

    @pytest.mark.asyncio
    async def test_send_posts_to_webhook(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send(
            "chat-123", "Hello!",
            metadata={"session_webhook": "https://dingtalk.example/webhook"}
        )
        assert result.success is True
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://dingtalk.example/webhook"
        payload = call_args[1]["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["title"] == "Hermes"
        assert payload["markdown"]["text"] == "Hello!"


    @pytest.mark.asyncio
    async def test_send_image_renders_markdown_image(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "OK"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        adapter._http_client = mock_client

        result = await adapter.send_image(
            "chat-123",
            "https://example.com/demo.png",
            caption="Screenshot",
            metadata={"session_webhook": "https://dingtalk.example/webhook"},
        )

        assert result.success is True
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["msgtype"] == "markdown"
        assert payload["markdown"]["text"] == "Screenshot\n\n![image](https://example.com/demo.png)"


# ---------------------------------------------------------------------------
# Connect / disconnect
# ---------------------------------------------------------------------------


class TestConnect:


    @pytest.mark.asyncio
    async def test_connect_fails_without_sdk(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.platforms.dingtalk.adapter.DINGTALK_STREAM_AVAILABLE", False
        )
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        result = await adapter.connect()
        assert result is False


    @pytest.mark.asyncio
    async def test_disconnect_finalizes_open_streaming_cards(self):
        """Streaming cards must be finalized before HTTP client closes."""
        from unittest.mock import AsyncMock, patch
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._http_client = AsyncMock()
        adapter._stream_task = None
        adapter._streaming_cards = {
            "chat-1": {"track-a": "last content"},
            "chat-2": {"track-b": "other"},
        }

        close_calls = []

        async def fake_close_siblings(chat_id):
            # HTTP client must still be alive at call time.
            assert adapter._http_client is not None, (
                "HTTP client was already closed before card finalization"
            )
            close_calls.append(chat_id)
            adapter._streaming_cards.pop(chat_id, None)

        with patch.object(adapter, "_close_streaming_siblings", side_effect=fake_close_siblings):
            await adapter.disconnect()

        assert set(close_calls) == {"chat-1", "chat-2"}
        assert adapter._streaming_cards == {}
        assert adapter._http_client is None


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SDK compatibility regression tests (dingtalk-stream >= 0.20 / 0.24)
# ---------------------------------------------------------------------------


class TestWebhookDomainAllowlist:
    """Guard the webhook origin allowlist against regression.

    The SDK started returning reply webhooks on ``oapi.dingtalk.com`` in
    addition to ``api.dingtalk.com``. Both must be accepted, and hostile
    lookalikes must still be rejected (SSRF defence-in-depth).
    """

    def test_api_domain_accepted(self):
        from plugins.platforms.dingtalk.adapter import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://api.dingtalk.com/robot/send?access_token=x"
        )

    def test_oapi_domain_accepted(self):
        from plugins.platforms.dingtalk.adapter import _DINGTALK_WEBHOOK_RE
        assert _DINGTALK_WEBHOOK_RE.match(
            "https://oapi.dingtalk.com/robot/send?access_token=x"
        )


class TestHandlerProcessIsAsync:
    """dingtalk-stream >= 0.20 requires ``process`` to be a coroutine."""

    def test_process_is_coroutine_function(self):
        from plugins.platforms.dingtalk.adapter import _IncomingHandler
        assert asyncio.iscoroutinefunction(_IncomingHandler.process)


class TestExtractText:
    """_extract_text must handle both legacy and current SDK payload shapes.

    Before SDK 0.20 ``message.text`` was a ``dict`` with a ``content`` key.
    From 0.20 onward it is a ``TextContent`` dataclass whose ``__str__``
    returns ``"TextContent(content=...)"`` — falling back to ``str(text)``
    leaks that repr into the agent's input.
    """


    def test_text_as_textcontent_object(self):
        """SDK >= 0.20 shape: object with ``.content`` attribute."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        class FakeTextContent:
            content = "hello from new sdk"

            def __str__(self):  # mimic real SDK repr
                return f"TextContent(content={self.content})"

        msg = MagicMock()
        msg.text = FakeTextContent()
        msg.rich_text_content = None
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert result == "hello from new sdk"
        assert "TextContent(" not in result


    def test_rich_text_content_new_shape(self):
        """SDK >= 0.20 exposes rich text as ``message.rich_text_content.rich_text_list``."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        class FakeRichText:
            rich_text_list = [{"text": "hello "}, {"text": "world"}]

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = FakeRichText()
        msg.rich_text = None
        result = DingTalkAdapter._extract_text(msg)
        assert "hello" in result and "world" in result


    def test_empty_message(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.rich_text = None
        assert DingTalkAdapter._extract_text(msg) == ""

    # --- Card / interactiveCard message handling (文档分享卡片) ---

    def test_card_with_dict_content_and_url(self):
        """card msgtype with extensions.card.content as dict with url key."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "card"
        msg.extensions = {
            "card": {
                "title": "Q3经营分析报告",
                "content": {"url": "https://dingtalk.com/doc/abc123"},
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档] Q3经营分析报告 https://dingtalk.com/doc/abc123"

    def test_card_with_dict_content_docurl(self):
        """card msgtype with extensions.card.content as dict with docUrl key."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "card"
        msg.extensions = {
            "card": {
                "title": "周报模板",
                "content": {"docUrl": "https://docs.dingtalk.com/xyz"},
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档] 周报模板 https://docs.dingtalk.com/xyz"


    def test_interactive_card_with_title_and_url(self):
        """interactiveCard msgtype with both title and biz_custom_action_url."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        msg = MagicMock()
        msg.text = None
        msg.rich_text = None
        msg.message_type = "interactiveCard"
        msg.extensions = {
            "content": {
                "title": "项目看板",
                "biz_custom_action_url": "https://dingtalk.com/doc/kanban",
            }
        }
        assert DingTalkAdapter._extract_text(msg) == "[文档卡片] 项目看板 https://dingtalk.com/doc/kanban"


class TestExtractMedia:
    """_extract_media must split native voice rich-text items (auto-STT)
    from generic audio file uploads (kept as attachments, no STT)."""

    def _msg_with_rich_text(self, items):
        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = items
        return msg


    def test_richtext_reset_does_not_clobber_voice(self):
        """A richText envelope containing a native voice item must stay
        VOICE — the ``msg_type_str == "richText"`` re-derivation used to
        reset it to TEXT, dropping the voice note from the STT path
        (#38211, #38219)."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.base import MessageType

        msg = self._msg_with_rich_text(
            [{"type": "voice", "downloadCode": "dl_voice_rt"}]
        )
        msg.message_type = "richText"
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(
            DingTalkAdapter, msg
        )
        assert msg_type == MessageType.VOICE
        assert urls == ["dl_voice_rt"]
        # 🔴 契约重写(2026-08-16):原断言钉的是 ``["audio"]``。
        # ``media_types`` 的契约是 **MIME**(见 adapter.py 的 EXT_MAP 注释),
        # 而下游 ``gateway.run._event_media_is_audio`` 判的是 ``"audio/"``。
        # 裸类别 "audio" 判 False,且因为**非空**连消息级兜底也跳过 ——
        # 实测 image/audio/video 三个门全 False,只有语音靠
        # ``message_type == VOICE`` 短路才没出事。
        # ⇒ 推不出子类型时如实记「未知」,让消息级类型兜底。
        assert mtypes == ["audio/*"], (
            "逐附件类型没写成 media-range ⇒ 异构列表会被消息级类型压平"
        )


    def test_image_no_filename_still_photo(self):
        """msgtype='image' without fileName → still PHOTO (MIME heuristic)."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.base import MessageType

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "image"
        msg.extensions = {"content": {"downloadCode": "dl_img_noext"}}
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(
            DingTalkAdapter, msg
        )
        assert msg_type == MessageType.PHOTO
        assert urls == ["dl_img_noext"]
        # 🔴 契约重写(2026-08-16)。原断言是
        #     assert mtypes == ["application/octet-stream"]
        #     # ... msg_type_str=="image" still wins
        # 那句注释描述的是**当时的实现恰好怎么跑**,⛔ 不是产品需求 ——
        # 它把一个 bug 钉成了契约:``application/octet-stream`` 非空且非
        # ``image/``,下游关口既判不出图片、又因非空跳过 PHOTO 兜底,
        # ⇒ **这条路的图从来没进过模型**。改对了反而会让这条测试变红。
        # ⇒ 需求是「钉钉说这是图片,它就得能进模型」。
        assert mtypes == [""]


class TestDingTalkMediaReachesTheModel:
    """⭐ 判据换成**下游关口本身**,⛔ 不再钉某个实现值。

    ``media_types`` 里放什么是实现细节;用户要的是「钉钉发的图 Agent 看得到」。
    直接驱动 ``gateway.run._event_media_is_image`` 等三个真实关口来断言。
    """

    def _event(self, mtypes, msg_type):
        from gateway.platforms.base import MessageEvent

        return MessageEvent(text="", message_type=msg_type,
                            media_urls=["x"] * len(mtypes), media_types=list(mtypes))

    def test_image_content_branch_reaches_the_model(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "picture"
        msg.image_content = MagicMock(download_code="dl_a")
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert _event_media_is_image(self._event(mtypes, msg_type), 0), (
            f"钉钉图片进不了模型:media_types={mtypes!r} message_type={msg_type!r}"
        )

    def test_rich_text_image_reaches_the_model(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = [{"type": "picture", "downloadCode": "dl_b"}]
        msg.message_type = "richText"
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert _event_media_is_image(self._event(mtypes, msg_type), 0), (
            f"钉钉富文本图片进不了模型:media_types={mtypes!r} message_type={msg_type!r}"
        )

    def test_image_without_filename_reaches_the_model(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "image"
        msg.extensions = {"content": {"downloadCode": "dl_c"}}
        msg_type, urls, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert _event_media_is_image(self._event(mtypes, msg_type), 0), (
            f"无 fileName 的钉钉图片进不了模型:media_types={mtypes!r}"
        )

    def test_voice_reaches_the_audio_bucket(self):
        """兄弟枚举:同一个函数里 audio 是同一形态的缺陷,一并修。"""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.run import _event_media_is_audio

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = [{"type": "voice", "downloadCode": "dl_v"}]
        msg.message_type = "richText"
        mt, _, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert _event_media_is_audio(self._event(mtypes, mt), 0), (mtypes, mt)

    def test_rich_text_video_is_currently_unreachable(self):
        """⚠️ 如实标注:``mapped == "video"`` 那条分支**今天走不到**。

        ``DINGTALK_TYPE_MAPPING`` 只映射 ``picture→image`` / ``voice→audio``;
        富文本里的 ``type="video"`` 落到默认 ``"file"`` ⇒ DOCUMENT。
        我在那条分支上顺手做的同形修改因此是**防御性的、当前未被执行**,
        ⛔ 不许把它说成「video 也修好了」。
        ⛔ 不在本轮给 mapping 加 video —— 那是改行为,不是修这个缺陷。
        """
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter, DINGTALK_TYPE_MAPPING
        from gateway.platforms.base import MessageType

        assert "video" not in DINGTALK_TYPE_MAPPING, DINGTALK_TYPE_MAPPING
        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = [{"type": "video", "downloadCode": "dl_w"}]
        msg.message_type = "richText"
        mt, _, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert mt == MessageType.DOCUMENT and mtypes == ["application/octet-stream"]

    def test_a_real_filename_still_yields_its_real_mime(self):
        """🔒 必须保持不变:有 fileName 的那条路**一格不动**。"""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "image"
        msg.extensions = {"content": {"downloadCode": "dl_d", "fileName": "shot.png"}}
        msg_type, _, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert mtypes == ["image/png"], mtypes           # ⛔ 没变
        assert msg_type == MessageType.PHOTO
        assert _event_media_is_image(self._event(mtypes, msg_type), 0)

    def test_a_real_document_is_still_a_document(self):
        """🔒 必须保持不变:⛔ 别把非图片一起放进图片桶。"""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "file"
        msg.extensions = {"content": {"downloadCode": "dl_e", "fileName": "report.pdf"}}
        msg_type, _, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert mtypes == ["application/pdf"], mtypes     # ⛔ 没变
        assert msg_type == MessageType.DOCUMENT
        assert not _event_media_is_image(self._event(mtypes, msg_type), 0)

    def test_a_file_without_extension_is_not_promoted_to_image(self):
        """🔒 ⛔ 作用域别外扩:``msgtype='file'`` 且推不出 MIME 时仍是文档。"""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_image

        msg = MagicMock()
        msg.text = None
        msg.image_content = None
        msg.rich_text_content = None
        msg.rich_text = None
        msg.message_type = "file"
        msg.extensions = {"content": {"downloadCode": "dl_f", "fileName": "blob"}}
        msg_type, _, mtypes = DingTalkAdapter._extract_media(DingTalkAdapter, msg)
        assert mtypes == ["application/octet-stream"], mtypes   # ⛔ 没变
        assert msg_type == MessageType.DOCUMENT
        assert not _event_media_is_image(self._event(mtypes, msg_type), 0)


# ---------------------------------------------------------------------------
# Group gating — require_mention + allowed_users (parity with other platforms)
# ---------------------------------------------------------------------------


def _make_gating_adapter(monkeypatch, *, extra=None, env=None):
    """Build a DingTalkAdapter with only the gating fields populated.

    Clears every DINGTALK_* gating env var before applying the caller's
    overrides so individual tests stay isolated.
    """
    for key in (
        "DINGTALK_REQUIRE_MENTION",
        "DINGTALK_MENTION_PATTERNS",
        "DINGTALK_FREE_RESPONSE_CHATS",
        "DINGTALK_ALLOWED_USERS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter
    return DingTalkAdapter(PlatformConfig(enabled=True, extra=extra or {}))


class TestAllowedUsersGate:

    def test_empty_allowlist_allows_everyone(self, monkeypatch):
        adapter = _make_gating_adapter(monkeypatch)
        assert adapter._is_user_allowed("anyone", "any-staff") is True


    def test_matches_sender_id_case_insensitive(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"allowed_users": ["SenderABC"]}
        )
        assert adapter._is_user_allowed("senderabc", "") is True


class TestMentionPatterns:


    def test_pattern_matches_text(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"mention_patterns": ["^hermes"]}
        )
        assert adapter._message_matches_mention_patterns("hermes please help") is True
        assert adapter._message_matches_mention_patterns("please hermes help") is False


    def test_env_var_json_populates_patterns(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            env={"DINGTALK_MENTION_PATTERNS": '["^bot", "^assistant"]'},
        )
        assert len(adapter._mention_patterns) == 2
        assert adapter._message_matches_mention_patterns("bot ping") is True


class TestShouldProcessMessage:

    def test_dm_always_accepted(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch, extra={"require_mention": True}
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=False, chat_id="dm1") is True


    def test_group_accepted_when_chat_in_free_response_list(self, monkeypatch):
        adapter = _make_gating_adapter(
            monkeypatch,
            extra={"require_mention": True, "free_response_chats": ["grp1"]},
        )
        msg = MagicMock(is_in_at_list=False)
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp1") is True
        # Different group still blocked
        assert adapter._should_process_message(msg, "hi", is_group=True, chat_id="grp2") is False


# ---------------------------------------------------------------------------
# _IncomingHandler.process — session_webhook extraction & fire-and-forget
# ---------------------------------------------------------------------------


class TestIncomingHandlerProcess:
    """Verify that _IncomingHandler.process correctly converts callback data
    and dispatches message processing as a background task (fire-and-forget)
    so the SDK ACK is returned immediately."""


    @pytest.mark.asyncio
    async def test_process_returns_ack_immediately(self):
        """process() must not block on _on_message — it should return
        the ACK tuple before the message is fully processed."""
        from plugins.platforms.dingtalk.adapter import _IncomingHandler, DingTalkAdapter

        processing_started = asyncio.Event()
        processing_gate = asyncio.Event()

        async def slow_on_message(msg):
            processing_started.set()
            await processing_gate.wait()  # Block until we release

        adapter = DingTalkAdapter(PlatformConfig(enabled=True))
        adapter._on_message = slow_on_message
        handler = _IncomingHandler(adapter, asyncio.get_running_loop())

        callback = MagicMock()
        callback.data = {
            "msgtype": "text",
            "text": {"content": "test"},
            "senderId": "u",
            "conversationId": "c",
            "sessionWebhook": "https://oapi.dingtalk.com/x",
            "msgId": "m",
        }

        # process() should return immediately even though _on_message blocks
        result = await handler.process(callback)
        assert result[0] == 200

        # Clean up: release the gate so the background task finishes
        processing_gate.set()
        await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# Text extraction — mention preservation + platform sanity
# ---------------------------------------------------------------------------

class TestExtractTextMentions:

    def test_preserves_at_mentions_in_text(self):
        """@mentions are routing signals (via isInAtList), not text to strip.

        Stripping all @handles collateral-damages emails, SSH URLs, and
        literal references the user wrote.
        """
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        cases = [
            ("@bot hello", "@bot hello"),
            ("contact alice@example.com", "contact alice@example.com"),
            ("git@github.com:foo/bar.git", "git@github.com:foo/bar.git"),
            ("what does @openai think", "what does @openai think"),
            ("@机器人 转发给 @老王", "@机器人 转发给 @老王"),
        ]
        for text, expected in cases:
            msg = MagicMock()
            msg.text = text
            msg.rich_text = None
            msg.rich_text_content = None
            assert DingTalkAdapter._extract_text(msg) == expected, (
                f"mangled: {text!r} -> {DingTalkAdapter._extract_text(msg)!r}"
            )


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Concurrency — chat-scoped message context
# ---------------------------------------------------------------------------


class TestMessageContextIsolation:

    def test_contexts_keyed_by_chat_id(self):
        """Two concurrent chats must not clobber each other's context."""
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        adapter = DingTalkAdapter(PlatformConfig(enabled=True))

        msg_a = MagicMock(conversation_id="chat-A", sender_staff_id="user-A")
        msg_b = MagicMock(conversation_id="chat-B", sender_staff_id="user-B")
        adapter._message_contexts["chat-A"] = msg_a
        adapter._message_contexts["chat-B"] = msg_b

        assert adapter._message_contexts["chat-A"] is msg_a
        assert adapter._message_contexts["chat-B"] is msg_b


# ---------------------------------------------------------------------------
# Card lifecycle: finalize via metadata["streaming"]
# ---------------------------------------------------------------------------


class TestCardLifecycle:

    @pytest.fixture
    def adapter_with_card(self):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter
        a = DingTalkAdapter(PlatformConfig(
            enabled=True,
            extra={"card_template_id": "tmpl-1"},
        ))
        a._card_sdk = MagicMock()
        a._card_sdk.create_card_with_options_async = AsyncMock()
        a._card_sdk.deliver_card_with_options_async = AsyncMock()
        a._card_sdk.streaming_update_with_options_async = AsyncMock()
        a._http_client = AsyncMock()
        a._get_access_token = AsyncMock(return_value="token")
        # Minimal message context
        msg = MagicMock(
            conversation_id="chat-1",
            conversation_type="1",
            sender_staff_id="staff-1",
            message_id="user-msg-1",
        )
        a._message_contexts["chat-1"] = msg
        a._session_webhooks["chat-1"] = (
            "https://api.dingtalk.com/x", 9999999999999,
        )
        return a

    @pytest.mark.asyncio
    async def test_final_reply_finalizes_card(self, adapter_with_card):
        """send(reply_to=...) creates a closed card (final response path)."""
        a = adapter_with_card
        result = await a.send("chat-1", "Hello", reply_to="user-msg-1")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is True
        # Not tracked as streaming — it's already closed.
        assert "chat-1" not in a._streaming_cards

    @pytest.mark.asyncio
    async def test_intermediate_send_stays_streaming(self, adapter_with_card):
        """send() without reply_to creates an OPEN card (tool progress /
        commentary / streaming first chunk).  No flicker closed→streaming
        when edit_message follows."""
        a = adapter_with_card
        result = await a.send("chat-1", "💻 terminal: ls")
        assert result.success
        call = a._card_sdk.streaming_update_with_options_async.call_args
        assert call[0][0].is_finalize is False
        # Tracked for sibling cleanup.
        assert result.message_id in a._streaming_cards.get("chat-1", {})


    @pytest.mark.asyncio
    async def test_edit_message_finalize_fires_done(self, adapter_with_card):
        """Stream consumer's final edit_message(finalize=True) fires Done."""
        a = adapter_with_card
        fired: list[str] = []
        a._fire_done_reaction = lambda cid: fired.append(cid)

        await a.send("chat-1", "initial")
        # Reopen via edit_message(finalize=False) then close.
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="streaming...", finalize=False,
        )
        await a.edit_message(
            chat_id="chat-1", message_id="track-X",
            content="final", finalize=True,
        )
        assert "chat-1" in fired


# ---------------------------------------------------------------------------
# AI Card Tests
# ---------------------------------------------------------------------------

class TestDingTalkAdapterAICards:
    @pytest.fixture
    def config(self):
        return PlatformConfig(
            enabled=True,
            extra={
                "client_id": "test_id",
                "client_secret": "test_secret",
                "card_template_id": "test_card_template",
            },
        )

    @pytest.fixture
    def mock_stream_client(self):
        client = MagicMock()
        client.get_access_token = MagicMock(return_value="test_token")
        return client

    @pytest.fixture
    def mock_http_client(self):
        return AsyncMock()

    @pytest.fixture
    def mock_message(self):
        msg = MagicMock()
        msg.message_id = "test_msg_id"
        msg.conversation_id = "test_conv_id"
        msg.conversation_type = "1"
        msg.sender_id = "sender1"
        msg.sender_nick = "Test User"
        msg.sender_staff_id = "staff1"
        msg.text = MagicMock(content="Hello")
        msg.session_webhook = "https://api.dingtalk.com/robot/sendBySession?session=test"
        msg.session_webhook_expired_time = 999999999999
        msg.create_at = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        msg.at_users = []
        return msg

    @pytest.mark.asyncio
    async def test_send_uses_ai_card_if_configured(self, config, mock_stream_client, mock_http_client, mock_message):
        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        adapter = DingTalkAdapter(config)
        adapter._stream_client = mock_stream_client
        adapter._http_client = mock_http_client
        adapter._message_contexts["test_conv_id"] = mock_message
        adapter._session_webhooks = {"test_conv_id": ("https://api.dingtalk.com/robot/sendBySession?session=test", 9999999999999)}
        adapter._card_template_id = "test_card_template"

        # Mock the card SDK with proper async methods
        mock_card_sdk = MagicMock()
        mock_card_sdk.create_card_with_options_async = AsyncMock()
        mock_card_sdk.deliver_card_with_options_async = AsyncMock()
        mock_card_sdk.streaming_update_with_options_async = AsyncMock()
        adapter._card_sdk = mock_card_sdk

        # Mock access token
        adapter._get_access_token = AsyncMock(return_value="test_token")

        result = await adapter.send("test_conv_id", "Hello World")

        mock_card_sdk.create_card_with_options_async.assert_called_once()
        mock_card_sdk.deliver_card_with_options_async.assert_called_once()
        mock_card_sdk.streaming_update_with_options_async.assert_called_once()
        assert result.success is True


class TestDingTalkHeterogeneousMediaIsNowRouted:
    """⭐ 本类原名 ``…IsAKnownLimitation``,断言的是**当时的错误行为**。

    它当时就注明「若转绿说明有人修好了,请同步更新注释」——**现在正是那一刻**:
    机器人第三轮 H⑦ 点名后已改为逐附件 media-range,断言随之翻面。
    """

    def _extract(self, items):
        from types import SimpleNamespace

        from plugins.platforms.dingtalk.adapter import DingTalkAdapter

        msg = SimpleNamespace(
            rich_text_content=SimpleNamespace(rich_text_list=items),
            message_type="richText", image_content=None, extensions={},
        )
        return DingTalkAdapter._extract_media(DingTalkAdapter, msg)

    def test_picture_then_voice_keeps_each_attachment_its_own_kind(self):
        _mt, _urls, types = self._extract([
            {"type": "picture", "downloadCode": "d1"},
            {"type": "voice", "downloadCode": "d2"},
        ])
        assert types[0].startswith("image/")
        assert types[1].startswith("audio/"), "语音仍会被送进视觉模型"

    def test_voice_then_picture_keeps_each_attachment_its_own_kind(self):
        _mt, _urls, types = self._extract([
            {"type": "voice", "downloadCode": "d1"},
            {"type": "picture", "downloadCode": "d2"},
        ])
        assert types[0].startswith("audio/")
        assert types[1].startswith("image/"), "图片仍会进 STT 路径"
