"""LINE 出站图片:URL 图片必须走原生 ``image`` 消息 —— ⛔ 不再退化成文本链接。

═══════════════════════════════════════════════════════════════════════
缺陷(修之前的真实行为)
═══════════════════════════════════════════════════════════════════════
``plugins/platforms/line/adapter.py`` 已有 ``send_image_file``(本地图片)、
``send_video``、``send_voice``,唯独没有覆写 ``send_image`` ⇒ 落到
``BasePlatformAdapter.send_image`` 的兜底「把 URL 当**文本**发」。
用户在 LINE 里看到的是一条链接,而不是图片 —— **兄弟调用点没跟上**。

═══════════════════════════════════════════════════════════════════════
本门关住 / 仍开集(⭐ 分格声明)
═══════════════════════════════════════════════════════════════════════
关住:①https URL ⇒ 组出 LINE ``image`` 消息(``originalContentUrl`` 是原 URL)
      ②带 caption ⇒ 图片**之后**追一条 text(⛔ 不许把 caption 塞进图片消息)
      ③**非 https ⇒ 既有行为逐字不变**(回退 base)—— 这一条钉的是
        「修复⛔ 不许弄坏原来对的东西」:LINE 只接受公网 https,
        本地路径/http 若硬发原生图会直接失败,还不如原来那条文本。
      ④未连接 ⇒ 明确失败,⛔ 不假成功。

仍开集:LINE 端**真的能不能渲染**那张图,取决于 URL 公网可达与证书 ——
        源码断言不了。**零真机验证**,本轮一条都没在真实 LINE 上跑过。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_line = load_plugin_adapter("line")
LineAdapter = _line.LineAdapter


def _adapter(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("LINE_HOST", raising=False)
    monkeypatch.delenv("LINE_PUBLIC_URL", raising=False)
    ad = LineAdapter(
        PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "tok", "channel_secret": "sec"},
        )
    )
    ad._client = MagicMock()
    return ad


class TestLineSendImageNative:
    @pytest.mark.asyncio
    async def test_force_push_preserves_fresh_reply_token_for_next_turn(self, monkeypatch):
        ad = _adapter(monkeypatch)
        ad._client.push = AsyncMock()
        ad._reply_tokens["C1"] = ("fresh-next-turn-token", float("inf"))

        result = await ad._send_text_chunks("C1", "缓存正文", force_push=True)

        assert result.success and ad._client.push.await_count == 1
        assert ad._reply_tokens.get("C1") == ("fresh-next-turn-token", float("inf")), (
            "旧 turn 的强制 push 不能吞掉新 turn 刚到的 reply token"
        )

    @pytest.mark.asyncio
    async def test_native_image_settles_slow_button_and_next_text_is_delivered(
        self, monkeypatch, tmp_path,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._client.push = AsyncMock()
        ad._client.reply = AsyncMock()
        downloaded = tmp_path / "downloaded-image"
        downloaded.write_bytes(b"\xff\xd8\xff" + b"x" * 16)
        monkeypatch.setattr(
            _line,
            "cache_image_from_url",
            AsyncMock(return_value=str(downloaded)),
        )
        rid = ad._cache.register_pending("C1")
        ad._pending_buttons["C1"] = rid
        assert ad._cache.get(rid).state is _line.State.PENDING, (
            "夹具必须真的进入 slow-response 按钮已创建的 PENDING 状态"
        )

        image_result = await ad.send_image(
            "C1", "https://cdn.example.com/pic.png", caption="图片说明",
        )
        text_result = await ad.send("C1", "后续文本")

        assert image_result.success and text_result.success
        assert ad._client.push.await_count == 2, (
            "原生图片后的文本必须真实送达，不能写进旧 PENDING cache 后假报成功"
        )
        _chat, messages = ad._client.push.await_args_list[1].args
        assert messages == [{"type": "text", "text": "后续文本"}]
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_native_image_flushes_text_cached_by_mixed_response(
        self, monkeypatch, tmp_path,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._client.push = AsyncMock()
        downloaded = tmp_path / "downloaded-image"
        downloaded.write_bytes(b"\xff\xd8\xff" + b"x" * 16)
        monkeypatch.setattr(
            _line,
            "cache_image_from_url",
            AsyncMock(return_value=str(downloaded)),
        )
        rid = ad._cache.register_pending("C1")
        ad._pending_buttons["C1"] = rid

        cached_result = await ad.send("C1", "混合正文")
        assert cached_result.success and ad._cache.get(rid).state is _line.State.READY
        assert ad._client.push.await_count == 0, "夹具必须先把正文缓存而非直接发送"

        image_result = await ad.send_image("C1", "https://cdn.example.com/pic.png")

        assert image_result.success
        assert ad._client.push.await_count == 2, "图片快路径不能吞掉已缓存的混合正文"
        _chat, messages = ad._client.push.await_args_list[1].args
        assert messages == [{"type": "text", "text": "混合正文"}]
        assert "C1" not in ad._pending_buttons
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [
        "https://cdn.example.com/pic.png",
        "https://cdn.example.com/PIC.JPEG?cache=1",
    ])
    async def test_supported_https_url_becomes_native_image_message(
        self, monkeypatch, tmp_path, url,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        downloaded = tmp_path / "downloaded-image"
        downloaded.write_bytes(b"\xff\xd8\xff" + b"x" * 16)
        fetch = AsyncMock(return_value=str(downloaded))
        monkeypatch.setattr(_line, "cache_image_from_url", fetch)

        res = await ad.send_image("C1", url)

        assert res.success is True
        fetch.assert_awaited_once_with(
            url,
            ext=".img",
            retries=0,
            max_bytes=_line.LINE_IMAGE_MAX_BYTES,
        )
        sent.assert_awaited_once()
        chat_id, msgs = sent.await_args.args
        assert chat_id == "C1"
        assert len(msgs) == 1, f"⛔ 只该发一条图片消息,实际 {msgs}"
        assert msgs[0]["type"] == "image", (
            "LINE 收到的仍不是原生 image 消息 —— 用户看到的是一条链接。"
            f"实际: {msgs[0]}"
        )
        assert msgs[0]["originalContentUrl"].startswith(
            "https://tunnel.example.com/line/media/"
        )
        assert msgs[0]["originalContentUrl"] != url, (
            "LINE 必须回拉我们按魔数确认后的快照，不能再次抓内容可协商的源 URL"
        )
        assert len(ad._media_temp_paths) == 1
        assert sum(ad._media_temp_sizes.values()) <= _line.LINE_IMAGE_MAX_BYTES
        assert not downloaded.exists(), "校验下载临时文件必须立即清理"
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_caption_goes_out_as_a_separate_text_message(
        self, monkeypatch, tmp_path,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        downloaded = tmp_path / "downloaded-image"
        downloaded.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)
        monkeypatch.setattr(
            _line, "cache_image_from_url", AsyncMock(return_value=str(downloaded)),
        )

        await ad.send_image("C1", "https://cdn.example.com/pic.png", caption="说明")

        _chat, msgs = sent.await_args.args
        assert [m["type"] for m in msgs] == ["image", "text"], (
            f"caption 必须是**图片之后**单独一条 text,实际 {msgs}"
        )
        assert msgs[1]["text"] == "说明"
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [
        "https://cdn.example.com/pic.webp",
        "https://cdn.example.com/pic.avif",
        "https://cdn.example.com/pic",
    ])
    async def test_unsupported_https_format_keeps_text_fallback(self, monkeypatch, url):
        ad = _adapter(monkeypatch)
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        fetch = AsyncMock()
        monkeypatch.setattr(_line, "cache_image_from_url", fetch)

        await ad.send_image("C1", url)

        assert not sent.await_count, f"{url} 不能作为 LINE 原生 image 发出"
        assert not fetch.await_count, "已声明为不支持的格式不应支付下载成本"
        base_sent.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [
        b"GIF89a" + b"x" * 16,
        b"RIFF" + b"x" * 4 + b"WEBP" + b"x" * 16,
    ])
    async def test_declared_png_with_unsupported_bytes_keeps_text_fallback(
        self, monkeypatch, tmp_path, payload,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        url = "https://cdn.example.com/spoofed.png"
        downloaded = tmp_path / "downloaded-image"
        downloaded.write_bytes(payload)
        fetch = AsyncMock(return_value=str(downloaded))
        monkeypatch.setattr(_line, "cache_image_from_url", fetch, raising=False)

        assert _line._line_native_image_url_mime(url) == "image/png", (
            "夹具必须真的进入声明为 PNG 的旧闸门"
        )
        assert _line._line_native_image_file_mime(downloaded) == "", (
            "夹具实际字节必须是 LINE 不支持的 GIF/WebP"
        )

        await ad.send_image("C1", url, caption="说明")

        assert not sent.await_count, "声明为 PNG 的 GIF/WebP 不能组装成 LINE 原生 image"
        fetch.assert_awaited_once_with(
            url,
            ext=".img",
            retries=0,
            max_bytes=_line.LINE_IMAGE_MAX_BYTES,
        )
        base_sent.assert_awaited_once_with(
            chat_id="C1",
            content=f"说明\n{url}",
            reply_to=None,
            metadata=None,
        )
        assert not downloaded.exists(), "不支持格式的校验下载也必须立即清理"

    @pytest.mark.asyncio
    async def test_gif_via_animation_keeps_text_fallback(self, monkeypatch):
        ad = _adapter(monkeypatch)
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        fetch = AsyncMock()
        monkeypatch.setattr(_line, "cache_image_from_url", fetch)
        url = "https://cdn.example.com/animated.gif"

        assert ad._is_animation_url(url), "量具没进入 bot 点名的 GIF → send_animation 分支"

        await ad.send_animation("C1", url)

        assert not sent.await_count, "GIF 不能被误组装成 LINE 原生静态 image"
        assert not fetch.await_count, "声明为 GIF 时不应支付无意义下载成本"
        base_sent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_public_media_url_keeps_text_fallback_without_download(
        self, monkeypatch,
    ):
        ad = _adapter(monkeypatch)
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        fetch = AsyncMock()
        ad._send_messages = sent
        ad.send = base_sent
        monkeypatch.setattr(_line, "cache_image_from_url", fetch)
        url = "https://cdn.example.com/pic.png"

        await ad.send_image("C1", url, caption="说明")

        assert not sent.await_count and not fetch.await_count
        base_sent.assert_awaited_once_with(
            chat_id="C1",
            content=f"说明\n{url}",
            reply_to=None,
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_validation_download_failure_keeps_text_fallback(
        self, monkeypatch,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        monkeypatch.setattr(
            _line,
            "cache_image_from_url",
            AsyncMock(side_effect=TimeoutError("validation timed out")),
        )
        url = "https://cdn.example.com/pic.png"

        await ad.send_image("C1", url)

        assert not sent.await_count, "无法确认实际字节时不能组装 LINE 原生 image"
        base_sent.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("suffix", [".gif", ".webp", ".avif", ".bin"])
    async def test_unsupported_local_image_keeps_safe_base_fallback(self, monkeypatch, tmp_path, suffix):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._register_media = MagicMock(return_value="token")
        ad._media_url = MagicMock(return_value="https://tunnel.example.com/image.png")
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        image = tmp_path / f"image{suffix}"
        image.write_bytes(b"image")

        await ad.send_image_file("C1", str(image))

        assert not sent.await_count and not ad._register_media.called
        base_sent.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name,payload,mime", [
        ("spoofed.png", b"GIF89a" + b"x" * 8, ""),
        ("extensionless", b"\x89PNG\r\n\x1a\n" + b"x", "image/png"),
        ("misnamed.webp", b"\xff\xd8\xff" + b"x" * 8, "image/jpeg"),
    ])
    async def test_local_image_uses_bytes_not_filename(self, monkeypatch, tmp_path, name, payload, mime):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._register_media = MagicMock(return_value="token")
        ad._media_url = MagicMock(return_value="https://tunnel.example.com/image")
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        ad.send = base_sent
        image = tmp_path / name
        image.write_bytes(payload)

        await ad.send_image_file("C1", str(image))

        assert _line._line_native_image_file_mime(image) == mime
        if mime:
            sent.assert_awaited_once()
            assert ad._register_media.called and not base_sent.await_count
        else:
            assert not sent.await_count and not ad._register_media.called
            base_sent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_local_media_response_uses_detected_image_mime(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        image = tmp_path / "misnamed.webp"
        image.write_bytes(b"\xff\xd8\xff" + b"x" * 8)
        ad._media_tokens["token"] = (str(image), float("inf"))
        request = type("Request", (), {"match_info": {"token": "token"}})()

        response = await ad._handle_media(request)

        assert response.headers["Content-Type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_local_image_snapshot_survives_source_replacement(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._send_messages = AsyncMock(return_value=_line.SendResult(success=True))
        image = tmp_path / "race.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")

        await ad.send_image_file("C1", str(image))

        (token, (snapshot, _expires_at)), = ad._media_tokens.items()
        assert snapshot != str(image) and Path(snapshot).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        image.write_bytes(b"GIF89a" + b"x" * 8)
        app = web.Application()
        app.router.add_get("/media/{token}/{filename}", ad._handle_media)
        async with TestClient(TestServer(app)) as client:
            response = await client.get(f"/media/{token}/race.png")
            body = await response.read()

        assert response.headers["Content-Type"] == "image/png"
        assert body.startswith(b"\x89PNG\r\n\x1a\n")
        assert Path(snapshot).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        ad._discard_media_token(token)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("public_url", [
        "http://not-https.example",
        "https:///missing-host",
    ])
    async def test_invalid_public_url_rejects_before_creating_a_snapshot(self, monkeypatch, tmp_path, public_url):
        ad = _adapter(monkeypatch)
        ad.public_base_url = public_url
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")

        result = await ad.send_image_file("C1", str(image))

        assert not result.success and "HTTPS" in (result.error or "")
        assert not ad._media_tokens and not ad._media_temp_paths

    @pytest.mark.asyncio
    async def test_failed_image_send_discards_the_snapshot(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._send_messages = AsyncMock(return_value=_line.SendResult(success=False, error="LINE rejected"))
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")

        result = await ad.send_image_file("C1", str(image))

        assert not result.success
        assert not ad._media_tokens and not ad._media_temp_paths

    @pytest.mark.asyncio
    async def test_raised_image_send_discards_the_snapshot(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._send_messages = AsyncMock(side_effect=RuntimeError("LINE transport failed"))
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")

        with pytest.raises(RuntimeError, match="LINE transport failed"):
            await ad.send_image_file("C1", str(image))

        assert not ad._media_tokens and not ad._media_temp_paths

    @pytest.mark.asyncio
    async def test_expired_snapshot_is_deleted_on_media_fetch(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._send_messages = AsyncMock(return_value=_line.SendResult(success=True))
        image = tmp_path / "image.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x")

        await ad.send_image_file("C1", str(image))

        (token, (snapshot, _expires_at)), = ad._media_tokens.items()
        ad._media_tokens[token] = (snapshot, 0)
        response = await ad._handle_media(type("Request", (), {"match_info": {"token": token}})())

        assert response.status == 410
        assert token not in ad._media_tokens and snapshot not in ad._media_temp_paths
        assert not Path(snapshot).exists()


class TestLineImageSnapshotBudget:
    def test_limits_follow_repo_precedent_and_real_image_measurement(self):
        count = getattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_COUNT", None)
        total = getattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_TOTAL_BYTES", None)
        assert count == 128, "数量上限必须照抄仓内 runtime source cache 的 128-entry 先例"
        assert total == 64 * 1024 * 1024
        measured_real_image = 403 * 1024  # cloud-gt002 企微真图实测
        assert count * measured_real_image <= total
        assert 6 * _line.LINE_IMAGE_MAX_BYTES <= total < 7 * _line.LINE_IMAGE_MAX_BYTES

    @pytest.mark.asyncio
    async def test_full_pool_preserves_sent_unfetched_snapshot_and_falls_back_new_remote(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_COUNT", 1, raising=False)
        monkeypatch.setattr(
            _line, "LINE_IMAGE_SNAPSHOT_MAX_TOTAL_BYTES", 1024, raising=False,
        )
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        native_send = AsyncMock(return_value=_line.SendResult(success=True))
        text_send = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = native_send
        ad.send = text_send
        first = tmp_path / "already-sent.jpg"
        first.write_bytes(b"\xff\xd8\xff" + b"a" * 32)

        first_result = await ad.send_image_file("C1", str(first))
        (first_token, (first_snapshot, _expiry)), = ad._media_tokens.items()
        assert first_result.success and native_send.await_count == 1, (
            "夹具必须先成功发出第一张 image message"
        )
        assert Path(first_snapshot).exists(), (
            "夹具必须停在消息已发出、LINE 尚未回拉的窗口"
        )

        downloaded = tmp_path / "new-remote.img"
        downloaded.write_bytes(b"\xff\xd8\xff" + b"b" * 32)
        monkeypatch.setattr(
            _line,
            "cache_image_from_url",
            AsyncMock(return_value=str(downloaded)),
        )
        url = "https://cdn.example.com/new.png"
        await ad.send_image("C1", url, caption="新图")

        assert first_token in ad._media_tokens and Path(first_snapshot).exists(), (
            "容量满时不能挤掉已发出但尚未被 LINE 回拉的旧快照"
        )
        assert native_send.await_count == 1, "容量满时新图不能再组装原生 image"
        text_send.assert_awaited_once_with(
            chat_id="C1",
            content=f"新图\n{url}",
            reply_to=None,
            metadata=None,
        )
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_count_limit_preserves_oldest_and_rejects_new_snapshot(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_COUNT", 2, raising=False)
        monkeypatch.setattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_TOTAL_BYTES", 1024, raising=False)
        ad = _adapter(monkeypatch)
        ad._media_ttl = 3600
        paths = []
        tokens = []
        for index in range(3):
            path = tmp_path / f"count-{index}.png"
            path.write_bytes(b"x" * 4)
            paths.append(path)
            tokens.append(ad._register_media(str(path), cleanup=True))

        assert len(ad._media_temp_paths) == 2
        assert tokens[0] and tokens[1] and tokens[2] is None
        assert paths[0].exists() and paths[1].exists() and not paths[2].exists()
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_byte_limit_preserves_oldest_and_rejects_new_snapshot(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_COUNT", 10, raising=False)
        monkeypatch.setattr(_line, "LINE_IMAGE_SNAPSHOT_MAX_TOTAL_BYTES", 10, raising=False)
        ad = _adapter(monkeypatch)
        ad._media_ttl = 3600
        first = tmp_path / "bytes-first.png"
        second = tmp_path / "bytes-second.png"
        first.write_bytes(b"x" * 6)
        second.write_bytes(b"y" * 6)

        first_token = ad._register_media(str(first), cleanup=True)
        second_token = ad._register_media(str(second), cleanup=True)

        assert len(ad._media_temp_paths) == 1
        assert first_token and second_token is None
        assert first.exists() and not second.exists()
        assert sum(Path(path).stat().st_size for path in ad._media_temp_paths) <= 10
        for token in list(ad._media_tokens):
            ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_ttl_actively_deletes_without_another_request(self, monkeypatch, tmp_path):
        ad = _adapter(monkeypatch)
        ad._media_ttl = 0.01
        path = tmp_path / "active-expiry.png"
        path.write_bytes(b"x")

        token = ad._register_media(str(path), cleanup=True)
        assert token in ad._media_tokens and path.exists()
        await asyncio.sleep(0.05)

        assert token not in ad._media_tokens
        assert not path.exists() and str(path.resolve()) not in ad._media_temp_paths

    @pytest.mark.asyncio
    async def test_inflight_snapshot_survives_expiry_before_response_prepare(
        self, monkeypatch, tmp_path,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        ad._send_messages = AsyncMock(return_value=_line.SendResult(success=True))
        image = tmp_path / "inflight.png"
        payload = b"\x89PNG\r\n\x1a\n" + b"x" * 4096
        image.write_bytes(payload)
        await ad.send_image_file("C1", str(image))
        (token, (snapshot, _expiry)), = ad._media_tokens.items()
        assert snapshot in ad._media_temp_paths and Path(snapshot).exists(), (
            "夹具必须真的进入受 TTL/容量回收约束的临时快照分支"
        )

        app = web.Application()
        original_prepare = web.StreamResponse.prepare
        expired_after_open = False

        async def expire_before_prepare(response, request):
            nonlocal expired_after_open
            # 修复后只有已持有文件描述符的精确 StreamResponse 走这里；在真正
            # prepare 前模拟 TTL 回收，目录项消失但当前响应仍必须读完整。
            if type(response) is web.StreamResponse:
                assert Path(snapshot).exists(), "回收前快照必须仍存在"
                ad._discard_media_token(token)
                assert not Path(snapshot).exists(), "夹具必须真的完成 unlink"
                expired_after_open = True
            return await original_prepare(response, request)

        monkeypatch.setattr(web.StreamResponse, "prepare", expire_before_prepare)

        async def route(request):
            response = await ad._handle_media(request)
            if isinstance(response, web.FileResponse):
                assert Path(snapshot).exists(), "夹具没进入 FileResponse 延迟打开窗口"
                ad._discard_media_token(token)
            return response

        app.router.add_get("/media/{token}/{filename}", route)
        async with TestClient(TestServer(app)) as client:
            response = await client.get(f"/media/{token}/inflight.png")
            body = await response.read()

        assert response.status == 200 and body == payload, (
            "已受理的快照请求不能因并发 TTL/容量回收退化成 404 或截断"
        )
        assert expired_after_open, "夹具没有命中打开后、prepare 前的回收窗口"
        ad._discard_media_token(token)

    @pytest.mark.asyncio
    async def test_non_https_keeps_the_previous_behaviour(self, monkeypatch):
        """⭐「修复⛔ 不许弄坏原来对的东西」——非 https 那一支逐字不变。

        本地路径 / http / data: 都走回 base 的兜底(发一条文本),
        ⛔ 不许因为「想发原生图」把原本还能送达的那条消息弄没。
        """
        ad = _adapter(monkeypatch)
        sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad._send_messages = sent
        base_sent = AsyncMock(return_value=_line.SendResult(success=True))
        ad.send = base_sent

        for candidate in (
            "/var/folders/xx/cache/img.png",
            "http://cdn.example.com/pic.png",
            "",
        ):
            base_sent.reset_mock()
            sent.reset_mock()
            await ad.send_image("C1", candidate)
            assert not sent.await_count, (
                f"{candidate!r} 走了原生图片路径 —— LINE 只接受公网 https,"
                "这会把原来还能送达的一条文本变成失败"
            )
            assert base_sent.await_count == 1, (
                f"{candidate!r} 既没走原生、也没回退 base ⇒ 消息**凭空消失**了"
            )

    @pytest.mark.asyncio
    async def test_disconnected_fails_loud(self, monkeypatch):
        """⛔ 不假成功:没连上就明确报失败(fail fast)。"""
        ad = _adapter(monkeypatch)
        ad._client = None
        ad._send_messages = AsyncMock(return_value=_line.SendResult(success=True))

        res = await ad.send_image("C1", "https://cdn.example.com/pic.png")
        assert res.success is False
        assert "not connected" in (res.error or "")


class TestLineOutboundSiblingsUnchanged:
    """飞书/钉钉之外,**LINE 自己**既有的三个出站出口本轮一个字没动。

    ⭐ 逐条显式说「没变」,⛔ 不靠「我记得没动」。
    """

    @pytest.mark.parametrize(
        "name", ["send_image_file", "send_video", "send_voice"]
    )
    def test_existing_outbound_entrypoints_still_defined_here(self, name):
        assert name in LineAdapter.__dict__, (
            f"{name} 不再由 LINE adapter 自己实现 ⇒ 会掉进 base 兜底,"
            "既有能力被弄坏了"
        )

    def test_send_document_still_falls_back_by_design(self):
        """⚠️ LINE Messaging API **没有** file/document 消息类型。

        ⇒ ``send_document`` 落到 base 的「友好提示」是**正确行为**,
        ⛔ 不是缺口,本轮**故意**不实现它。
        (出处:LINE Messaging API 的消息对象类型只有 text/sticker/image/
        video/audio/location/imagemap/template/flex —— 无 file。)
        """
        assert "send_document" not in LineAdapter.__dict__

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raises", [False, True])
    async def test_failed_video_send_discards_auto_preview_and_tokens(
        self, monkeypatch, tmp_path, raises,
    ):
        ad = _adapter(monkeypatch)
        ad.public_base_url = "https://tunnel.example.com"
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video")
        registered = []
        real_register = ad._register_media

        def capture_register(path, *, cleanup=False):
            registered.append((path, cleanup))
            return real_register(path, cleanup=cleanup)

        monkeypatch.setattr(ad, "_register_media", capture_register)
        if raises:
            ad._send_messages = AsyncMock(side_effect=RuntimeError("LINE rejected video"))
            with pytest.raises(RuntimeError, match="LINE rejected video"):
                await ad.send_video("C1", str(video))
        else:
            ad._send_messages = AsyncMock(
                return_value=_line.SendResult(success=False, error="LINE rejected video")
            )
            result = await ad.send_video("C1", str(video))
            assert not result.success

        preview_paths = [Path(path) for path, cleanup in registered if cleanup]
        assert len(preview_paths) == 1, "夹具必须真的注册自动 video preview snapshot"
        assert not ad._media_tokens, "视频未送达时 video/preview token 都必须释放"
        assert not ad._media_temp_paths and not preview_paths[0].exists(), (
            "视频未送达时自动 preview 不能继续占 snapshot 池"
        )
