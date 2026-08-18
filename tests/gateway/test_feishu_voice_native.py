"""飞书语音：非 opus 音频必须转码后发成**原生语音**，⛔ 不许静默变文件。

现场：``send_voice()`` 把路径直接交给 ``_send_uploaded_file_message``，而它的
分派 ``_resolve_outbound_file_routing`` **只看扩展名** —— ``.ogg/.opus`` ⇒
``audio``，其余一律 ⇒ ``file``。于是 ``send_voice("x.mp3")`` **静默降级成一个
普通文件附件**：用户发的是语音、对方收到的是文件，而链路上没有任何一层解释。
⚠️ ``outbound_message_type="audio"`` 这个参数形同虚设 —— 分派函数只在最后两个
分支用到它，而那两个分支返回的是同一个结果。

官方逐字（[上传文件](https://open.feishu.cn/document/server-docs/im-v1/file/create)）：
``"OPUS 音频文件。其他格式的音频文件，请转为 OPUS 格式后上传。"``
⇒ 这**不是**「我们没实现」，是平台确实只收 opus ⇒ 正解是**转码**，
转不了时**显式报错**，⛔ 两者都不许静默降级。
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from plugins.platforms.feishu import adapter as feishu

_HAS_FFMPEG = feishu._FEISHU_FFMPEG_PATH is not None


def _adapter(monkeypatch):
    a = feishu.FeishuAdapter.__new__(feishu.FeishuAdapter)
    sent: list[dict] = []

    async def _fake_send(**kwargs):
        sent.append(kwargs)
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="m1")

    monkeypatch.setattr(a, "_send_uploaded_file_message", _fake_send)
    return a, sent


def _real_mp3(tmp_path):
    """用 ffmpeg 造一个**真的** mp3 —— ⛔ 不用假字节。

    ⭐ 转码门必须喂真音频:假字节会让 ffmpeg 失败，于是「转码失败分支」绿了、
    「转码成功分支」从没跑过 —— 门看起来全绿却只验了一半。
    """
    out = tmp_path / "voice.mp3"
    subprocess.run(
        [feishu._FEISHU_FFMPEG_PATH, "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-c:a", "libmp3lame", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(out)


# ───────── 缺陷本体：非 opus 必须转码成原生语音 ─────────


@pytest.mark.skipif(not _HAS_FFMPEG, reason="本机没有 ffmpeg")
@pytest.mark.asyncio
async def test_mp3_is_transcoded_and_sent_as_native_audio(tmp_path, monkeypatch):
    """🔴 端到端跑**真的 ffmpeg**：mp3 ⇒ opus ⇒ 走 audio 分派。"""
    a, sent = _adapter(monkeypatch)
    mp3 = _real_mp3(tmp_path)

    res = await a.send_voice(chat_id="oc_1", audio_path=mp3)

    assert res.success, f"转码后仍然发失败:{res.error}"
    assert len(sent) == 1
    used = sent[0]["file_path"]
    assert used.endswith(".ogg"), f"没有转码，原样发了:{used}"
    # ⭐ 判据落在**分派结果**上,⛔ 不是「文件名以 .ogg 结尾」——
    # 后者只是手段,真正要钉的是它会被路由成 audio 而不是 file。
    upload_type, message_type = a._resolve_outbound_file_routing(
        file_path=used, requested_message_type="audio")
    assert (upload_type, message_type) == ("opus", "audio"), (
        f"转码后仍被路由成 {message_type}，不是原生语音")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="本机没有 ffmpeg")
@pytest.mark.asyncio
async def test_transcoded_temp_file_is_cleaned_up(tmp_path, monkeypatch):
    """⛔ 每条语音都留一个临时 .ogg 会把磁盘塞满。"""
    a, sent = _adapter(monkeypatch)
    await a.send_voice(chat_id="oc_1", audio_path=_real_mp3(tmp_path))
    assert not os.path.exists(sent[0]["file_path"]), "临时转码文件没删"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="本机没有 ffmpeg")
@pytest.mark.asyncio
async def test_temp_file_is_not_written_next_to_the_source(tmp_path, monkeypatch):
    """⚠️ 刻意偏离先例：⛔ 不许把 ``a.mp3`` 转成同目录 ``a.ogg``。

    源目录可能只读（kanban artifact），且同名会**覆盖用户文件**。
    """
    a, sent = _adapter(monkeypatch)
    mp3 = _real_mp3(tmp_path)
    (tmp_path / "voice.ogg").write_bytes(b"USER-FILE-DO-NOT-TOUCH")

    await a.send_voice(chat_id="oc_1", audio_path=mp3)

    assert (tmp_path / "voice.ogg").read_bytes() == b"USER-FILE-DO-NOT-TOUCH", (
        "转码覆盖了同目录的同名用户文件")
    assert not sent[0]["file_path"].startswith(str(tmp_path)), (
        f"临时文件写在了源目录:{sent[0]['file_path']}")


# ───────── ⛔ 转不了时显式报错，不许静默降级 ─────────


@pytest.mark.asyncio
async def test_missing_ffmpeg_returns_actionable_error(tmp_path, monkeypatch):
    """🔴 没有 ffmpeg ⇒ ⛔ 不许悄悄发成文件，要说清怎么办。

    ⭐ 这里**刻意不照抄** whatsapp_cloud 的「降级发 mp3」：
    它降级后仍是 audio 消息（只差波形气泡），飞书降级会变成 file，
    语义整个丢掉。⇒ 照抄第四问「它的判据在我这边还成立吗」= 不成立。
    """
    monkeypatch.setattr(feishu, "_FEISHU_FFMPEG_PATH", None)
    a, sent = _adapter(monkeypatch)
    src = tmp_path / "voice.mp3"
    src.write_bytes(b"\x00" * 16)

    res = await a.send_voice(chat_id="oc_1", audio_path=str(src))

    assert res.success is False, "没有 ffmpeg 却报告发送成功"
    assert "ffmpeg" in res.error and "opus" in res.error, f"没说清原因:{res.error}"
    assert sent == [], "转不了还是把文件发出去了 —— 静默降级"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="本机没有 ffmpeg")
@pytest.mark.asyncio
async def test_broken_audio_returns_error_not_a_file_attachment(tmp_path, monkeypatch):
    """转码失败（文件根本不是音频）⇒ 显式错误，⛔ 不降级成文件附件。"""
    a, sent = _adapter(monkeypatch)
    bad = tmp_path / "voice.mp3"
    bad.write_bytes(b"this is not audio at all")

    res = await a.send_voice(chat_id="oc_1", audio_path=str(bad))

    assert res.success is False
    assert sent == [], "转码失败还是发出去了"


@pytest.mark.skipif(not _HAS_FFMPEG, reason="本机没有 ffmpeg")
@pytest.mark.asyncio
async def test_error_never_leaks_the_source_path(tmp_path, monkeypatch):
    """⛔ 给用户的错误里不许含绝对路径。"""
    a, _ = _adapter(monkeypatch)
    bad = tmp_path / "secret-recording.mp3"
    bad.write_bytes(b"nope")
    res = await a.send_voice(chat_id="oc_1", audio_path=str(bad))
    assert str(tmp_path) not in (res.error or ""), f"路径泄漏:{res.error}"


# ───────── ⛔ 必须保持不变的行为 ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".ogg", ".opus"])
async def test_opus_input_path_is_byte_for_byte_unchanged(tmp_path, monkeypatch, ext):
    """🔴 已经是 opus ⇒ ⛔ 不转码、⛔ 不进临时文件、原路径原样上传。"""
    a, sent = _adapter(monkeypatch)
    called = {"n": 0}

    async def _never(_src):
        called["n"] += 1
        return None

    monkeypatch.setattr(a, "_transcode_to_opus", _never)
    src = tmp_path / f"voice{ext}"
    src.write_bytes(b"OggS-fake")

    res = await a.send_voice(chat_id="oc_1", audio_path=str(src))

    assert res.success and called["n"] == 0, "opus 输入被多余地转码了"
    assert sent[0]["file_path"] == str(src), "opus 原路径被改写"
    assert sent[0]["outbound_message_type"] == "audio"


@pytest.mark.asyncio
async def test_other_send_methods_are_untouched(tmp_path, monkeypatch):
    """⛔ 只动 send_voice —— 文档/视频/图片三条路径一个字节不变。"""
    a, sent = _adapter(monkeypatch)
    f = tmp_path / "a.pdf"
    f.write_bytes(b"x")
    await a.send_document(chat_id="oc_1", file_path=str(f))
    assert sent[-1]["file_path"] == str(f)

    v = tmp_path / "a.mp4"
    v.write_bytes(b"x")
    await a.send_video(chat_id="oc_1", video_path=str(v))
    assert sent[-1]["file_path"] == str(v)


def test_routing_table_still_maps_opus_and_media(tmp_path):
    """⛔ 分派表本身不许被改坏（本轮没动它，钉住现状）。"""
    r = feishu.FeishuAdapter._resolve_outbound_file_routing
    assert r(file_path="a.ogg", requested_message_type="audio") == ("opus", "audio")
    assert r(file_path="a.mp4", requested_message_type="video") == ("mp4", "media")
    assert r(file_path="a.pdf", requested_message_type="file")[1] == "file"
