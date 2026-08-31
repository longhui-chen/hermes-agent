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
    _build_media_placeholder as _bmp_async,
    _event_media_is_audio,
    _event_media_is_image,
    _event_media_is_video,
)


def _build_media_placeholder(_e):
    # ⚠️ 签名**有意**改成 async;只改调用方式,断言逐字不变。
    import asyncio

    return asyncio.run(_bmp_async(_e))


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


def test_parameterized_mixed_case_image_mime_reaches_model_as_image(tmp_path):
    """Matrix/Teams 可带大小写和参数；模型最终仍必须看到 image，而不是 file。"""
    image = tmp_path / "actual.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    evt = _evt(
        [str(image)],
        [" Image/PNG; charset=binary "],
        MessageType.DOCUMENT,
    )

    assert evt.media_types[0] != "image/png", "夹具必须真的进入未规范化 MIME 分支"
    out = _build_media_placeholder(evt)

    assert f"[User sent an image: {image}]" in out, (
        "合法的大小写/带参数 MIME 不能让图片退化成普通文件，导致模型看不到像素"
    )

