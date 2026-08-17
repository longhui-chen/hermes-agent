"""Regression tests for mixed-attachment routing in gateway/run.py.

Issue #25935: when a message mixes a real image with a document (e.g. a .md
brief), Discord types the whole message MessageType.PHOTO. The per-attachment
loops must classify each attachment by its OWN mimetype:

  * A document must NOT be swept into image_paths just because the message-level
    type is PHOTO — mislabelling it as an image sent its bytes to the vision
    endpoint, which rejected them with a non-retryable HTTP 400 and killed the
    whole turn ("Could not process image").
  * That same document must STILL reach the agent as a readable cached file via
    the document context-note path, even though the message-level type isn't
    DOCUMENT.

The message-level fallback (PHOTO/VOICE/AUDIO/VIDEO) is preserved only for
attachments whose per-file mimetype is unknown (empty) — platforms that don't
populate media_types.
"""

from types import SimpleNamespace

from gateway.platforms.base import MessageType
from gateway.run import (
    _build_media_placeholder,
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_video,
)


def _evt(media_urls, media_types, message_type):
    return SimpleNamespace(
        media_urls=media_urls,
        media_types=media_types,
        message_type=message_type,
    )


# ─── per-attachment classification helpers ───────────────────────────────────


def test_image_trusts_own_mime_over_photo_message_type():
    evt = _evt(["/c/pic.png", "/c/brief.md"], ["image/png", "text/markdown"], MessageType.PHOTO)
    assert _event_media_is_image(evt, 0) is True
    # The document must NOT be promoted to an image by the PHOTO fallback.
    assert _event_media_is_image(evt, 1) is False


# ─── _build_media_placeholder ────────────────────────────────────────────────


def test_placeholder_document_in_photo_message_is_not_an_image(tmp_path):
    # ⚠️ 必须是**真实存在**的文件：可读性契约接线后，读不到的路径不再被写进
    # 模型提示（那正是族 A 的现场）。本用例钉的是「文档不许被 PHOTO 消息类型
    # 提升成 image」，与可读性无关 —— 只需把假路径换成真文件。
    png = tmp_path / "product.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    md = tmp_path / "brief.md"
    md.write_bytes(b"# brief")

    evt = _evt([str(png), str(md)], ["image/png", "text/markdown"], MessageType.PHOTO)
    out = _build_media_placeholder(evt)
    assert f"[User sent an image: {png}]" in out
    assert f"[User sent an image: {md}]" not in out
    assert f"[User sent a file: {md}]" in out


