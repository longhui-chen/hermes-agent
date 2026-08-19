"""Tests for video attachment context notes in gateway turns."""

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner() -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._has_setup_skill = lambda: False
    return runner


@pytest.mark.asyncio
async def test_video_attachment_adds_path_note_without_document_wording(tmp_path):
    from gateway.run import _build_media_placeholder

    # 必须是真实存在的文件:可读性契约接线后,不存在的路径不会再被写进
    # 模型提示(那正是族 A 的现场)。本用例钉的是「video 用 video 措辞」,
    # 与可读性无关,所以只需把假路径换成真文件。
    _clip = str(tmp_path / "video_clip.mp4")
    (tmp_path / "video_clip.mp4").write_bytes(b"\x00\x00\x00 ftypmp42")

    runner = _make_runner()
    source = SessionSource(platform=Platform.SLACK, chat_id="D123", chat_type="dm")
    event = MessageEvent(
        text="what happens here?",
        message_type=MessageType.VIDEO,
        source=source,
        media_urls=[_clip],
        media_types=["video/mp4"],
    )

    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda path: path,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert "video attachment" in result
    assert _clip in result
    assert "video analysis or media tool" in result
    assert "The user sent a document" not in result
    # ⚠️ 签名**有意**改成 async;本用例本身就是协程 ⇒ 直接 await,断言逐字不变。
    assert await _build_media_placeholder(event) == f"[User sent a video: {_clip}]"


@pytest.mark.asyncio
async def test_parameterized_mixed_case_video_reaches_model_as_video(tmp_path):
    """合法 MIME 变体不能在真实入站消费链里静默丢掉视频/文本附件。"""
    clip = str(tmp_path / "video_clip.mp4")
    (tmp_path / "video_clip.mp4").write_bytes(b"\x00\x00\x00 ftypmp42")
    note = str(tmp_path / "note.txt")
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    runner = _make_runner()
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:test", chat_type="dm")
    event = MessageEvent(
        text="请分析视频",
        message_type=MessageType.DOCUMENT,
        source=source,
        media_urls=[clip, note],
        media_types=[
            " VIDEO/MP4; charset=binary ",
            " Text/Plain; charset=utf-8 ",
        ],
    )

    assert event.message_type != MessageType.VIDEO, "夹具不能靠消息级 VIDEO 兜底变绿"
    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda path: path,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert "video attachment" in result and clip in result, (
        "合法大小写/带参数 MIME 的视频必须把可读路径交给模型，不能静默丢失"
    )
    assert "text document" in result and note in result, (
        "合法大小写/带参数 MIME 的文本附件不能被错误描述成二进制文档"
    )
