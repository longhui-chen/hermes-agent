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
    assert _build_media_placeholder(event) == f"[User sent a video: {_clip}]"
