"""企微入站媒体失败：**原因**必须带到调用方，用户拿到的建议必须对症（P2-3 / P2-4）。

两条缺陷、同一个面：

**P2-3 —— 已知类型但载荷畸形仍被当真空消息。**
``msgtype="image"`` 而 ``body["image"]`` 不是 dict（缺失 / null / 上游把它
改成了字符串）时，原先所有分支都不接，``refs`` 为空、``failed_kinds`` 为空 ⇒
``_on_message`` 把它当成「用户发了条空消息」直接 return。
⚠️ 兜底那条 ``msgtype not in _WECOM_KNOWN_MSGTYPES`` 挡不住它 —— ``image``
**是**已知类型。⭐ 判据挑错了维度:该问的是「有没有被任何分支接住」，
⛔ 不是「这个 msgtype 我认不认识」。

**P2-4 —— 七种成因被压成一个词。**
``failed_kinds: List[str]`` 只带 kind 不带 reason，于是用户永远收到同一句
「请稍后重新发送；如果是较大的文件，可以先压缩后再试」。对**链接过期**、
**解密失败**、**磁盘写满**全是错误指引。
⭐ **补错的建议比不补更坏**:它让用户去做一件确定无效的事，然后以为是自己的
问题 —— 与「补错字段」「假闭集」同形，错误的确定性会终止追查。
"""
from __future__ import annotations

import base64 as _b64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.wecom import adapter as wecom_adapter
from plugins.platforms.wecom.adapter import (
    _MEDIA_FAILURE_ADVICE,
    MediaFailure,
    WeComAdapter,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _adapter():
    class _A(WeComAdapter):
        name = "wecom"

    return _A.__new__(_A)


def _boom_disk(*_a, **_kw):
    raise OSError(28, "No space left on device")


# ═════════ ① 七种成因全集：每一条 return None 都必须留下 reason ═════════
#
# 🔴 这一节是 ``_extract_media`` 里删掉 ``failed_kinds.append(kind)`` 的**前提**。
# 删掉它以后，记账**唯一**来源是 ``_media_intake_failed``;只要有任何一条
# ``return None`` 路径绕过了它，失败就会重新变成静默 —— 而 diff 里看不出来。
# ⭐ 所以这条门驱动**全部七种成因**，⛔ 不是抽查两三种。


async def _drive(monkeypatch, kind, media, *, dl=None, writer_boom=False):
    a = _adapter()
    if writer_boom:
        for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
            monkeypatch.setattr(wecom_adapter, w, _boom_disk)
    if dl is not None:
        monkeypatch.setattr(a, "_download_remote_bytes", dl)
    failures: list[MediaFailure] = []
    await a._cache_media(kind, media, failures=failures)
    return failures


@pytest.mark.asyncio
async def test_reason_no_media_reference(monkeypatch):
    assert await _drive(monkeypatch, "image", {}) == [
        MediaFailure("image", "no_media_reference")]


@pytest.mark.asyncio
async def test_reason_base64_decode_failed(monkeypatch):
    assert await _drive(monkeypatch, "image", {"base64": "!!!not-base64!!!"}) == [
        MediaFailure("image", "base64_decode_failed")]


@pytest.mark.asyncio
async def test_reason_download_failed(monkeypatch):
    async def _dl(url, max_bytes=None):
        raise RuntimeError("connection reset")

    assert await _drive(
        monkeypatch, "image", {"url": "https://wework.qpic.cn/x"}, dl=_dl
    ) == [MediaFailure("image", "download_failed")]


@pytest.mark.asyncio
async def test_reason_decrypt_failed(monkeypatch):
    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    got = await _drive(
        monkeypatch, "image",
        {"url": "https://wework.qpic.cn/x", "aeskey": "not-a-real-key"}, dl=_dl)
    assert got == [MediaFailure("image", "decrypt_failed")]


@pytest.mark.asyncio
async def test_reason_cache_write_failed(monkeypatch):
    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    got = await _drive(
        monkeypatch, "image", {"url": "https://wework.qpic.cn/x"},
        dl=_dl, writer_boom=True)
    assert got == [MediaFailure("image", "cache_write_failed")]


@pytest.mark.asyncio
async def test_reason_not_an_image(monkeypatch):
    def _reject(*_a, **_kw):
        raise ValueError("not an image")

    monkeypatch.setattr(wecom_adapter, "cache_image_from_bytes", _reject)
    got = await _drive(
        monkeypatch, "image", {"base64": _b64.b64encode(b"plain text").decode()})
    assert got == [MediaFailure("image", "not_an_image")]


@pytest.mark.asyncio
async def test_document_write_failure_also_records(monkeypatch):
    """⭐ 文件分支与图片分支是**两条独立的写盘路径** —— 只测 image
    等于给 document 那条发免检（本仓今晚已经栽过一次同形）。"""
    got = await _drive(
        monkeypatch, "file", {"base64": _b64.b64encode(b"pdf-ish").decode()},
        writer_boom=True)
    assert got == [MediaFailure("file", "cache_write_failed")]


@pytest.mark.asyncio
async def test_one_failed_attachment_is_counted_once(monkeypatch):
    """⛔ 不许重复计数。

    删掉调用方那句 append 正是为了这个:两处都记，一个失败附件会变成
    「2 个附件」，给用户的话直接说错数量。
    """
    a = _adapter()
    for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(wecom_adapter, w, _boom_disk)

    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    monkeypatch.setattr(a, "_download_remote_bytes", _dl)
    _p, _t, failures = await a._extract_media(
        {"msgtype": "image", "image": {"url": "https://wework.qpic.cn/x"}})
    assert failures == [MediaFailure("image", "cache_write_failed")]


# ═════════ ② P2-3：声明了媒体、载荷畸形 —— 四处全集 ═════════


@pytest.mark.asyncio
@pytest.mark.parametrize("msgtype", ["image", "file", "video"])
@pytest.mark.parametrize(
    "bad", [None, "", "a-string", 123, []],
    ids=["absent-none", "empty-str", "str", "int", "list"])
async def test_toplevel_malformed_payload_is_not_an_empty_message(msgtype, bad):
    """🔴 P2-3 本体：类型说有媒体、对象形状不对 ⇒ ⛔ 不许当成真空消息。

    ⭐ 形状用 parametrize 枚举:``None`` / 空串 / 字符串 / 数字 / 列表 ——
    判据是「不是 dict」这一个闭集条件，⛔ 不是「我列到的那几种畸形」。
    """
    a = _adapter()
    _p, _t, failures = await a._extract_media({"msgtype": msgtype, msgtype: bad})
    assert failures == [MediaFailure(msgtype, "payload_malformed")], (
        f"{msgtype} 载荷畸形被静默丢弃了 —— 用户发的东西凭空消失")


@pytest.mark.asyncio
async def test_missing_key_entirely_is_also_malformed():
    """键**整个不在**（不是形状不对）也一样 —— 协议里它是必填的。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media({"msgtype": "image"})
    assert failures == [MediaFailure("image", "payload_malformed")]


@pytest.mark.asyncio
async def test_mixed_item_malformed_payload_is_reported():
    """mixed 子项同形:``msgtype=image`` 但 ``image`` 不是 dict。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media({
        "msgtype": "mixed",
        "mixed": {"msg_item": [
            {"msgtype": "text", "text": {"content": "看图"}},
            {"msgtype": "image", "image": None},
        ]},
    })
    assert failures == [MediaFailure("image", "payload_malformed")]


@pytest.mark.asyncio
async def test_quoted_message_malformed_payload_is_reported():
    """引用消息同形 —— ⛔ 不许因为「它只是引用」就免检。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media({
        "msgtype": "text", "text": {"content": "看这条"},
        "quote": {"msgtype": "file", "file": "oops-a-string"},
    })
    assert failures == [MediaFailure("file", "payload_malformed")]


@pytest.mark.asyncio
async def test_malformed_payload_reaches_the_user(monkeypatch):
    """🔴 从**真实入口**驱动一次:畸形载荷的纯附件消息不许无声无息。

    ⭐ 前面几条驱动的是 ``_extract_media``;缺陷的**后果**发生在
    ``_on_message`` 那一层，那里才是「整条消息消失」的地方。
    """
    a = _adapter()
    a._dedup = SimpleNamespace(is_duplicate=lambda _m: False)
    a._remember_reply_req_id = lambda *_a, **_k: None
    a._remember_chat_req_id = lambda *_a, **_k: None
    a._is_group_allowed = lambda *_a, **_k: True
    a._is_dm_intake_allowed = lambda *_a, **_k: True
    a._payload_req_id = lambda _p: "req-1"

    from gateway.session import Platform, SessionSource

    a.build_source = lambda **kw: SessionSource(platform=Platform.WECOM, **kw)
    a.config = SimpleNamespace(extra={})
    a._pending_text_batches = {}
    a._pending_text_batch_tasks = {}
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    a.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))

    await a._on_message({"body": {
        "msgid": "mid-x", "msgtype": "image", "image": None,
        "from": {"userid": "u1"}, "chatid": "oc_1",
    }})

    assert a.send.await_count == 1, "畸形载荷的图片消息被静默丢弃 —— 发了没反应"
    assert "图片" in a.send.await_args.args[1]


# ═════════ ③ P2-4：给用户的建议必须对症 ═════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason,must_have,must_not_have",
    [
        ("download_failed", "重新发送", "压缩"),
        ("decrypt_failed", "重新发送", "压缩"),
        ("cache_write_failed", "存储空间", "压缩"),
        ("not_an_image", "PNG", "压缩"),
        ("no_media_reference", "重新发送", "压缩"),
        ("payload_malformed", "重新发送", "压缩"),
        ("base64_decode_failed", "重新发送", "压缩"),
    ],
)
async def test_advice_matches_the_actual_cause(reason, must_have, must_not_have):
    """🔴 P2-4 本体:七种成因给七种（去重后仍分得开的）建议。

    ⛔ 「压缩」曾是唯一一句 —— 它对上面**每一种**都无效:
    链接过期要立刻重发、磁盘满要清空间、格式不对要换格式。
    ⭐ 判据不是「文案变长了」，是「照着做能不能解决」。
    """
    a = _adapter()
    body = a._media_failure_reply_text([MediaFailure("image", reason)])
    assert must_have in body, f"{reason} 的建议没说清怎么办:{body}"
    assert must_not_have not in body, f"{reason} 拿到了无效建议:{body}"


def test_unknown_reason_falls_back_without_leaking_the_code():
    """⭐ 未知成因:⛔ 不许伪装成功、⛔ 也不许把 reason 码甩给用户。

    清单天然是开集（以后一定会加新 reason）⇒ 兜底必须闭集。
    """
    a = _adapter()
    body = a._media_failure_reply_text([MediaFailure("image", "brand_new_reason")])
    assert "brand_new_reason" not in body, f"内部 reason 码泄漏给用户:{body}"
    assert "重新发送" in body, f"兜底没给下一步:{body}"


def test_reply_never_leaks_reason_codes():
    """闭集自检：**每一个**已登记 reason 的文案都不许含 reason 码本身。"""
    a = _adapter()
    for reason in _MEDIA_FAILURE_ADVICE:
        body = a._media_failure_reply_text([MediaFailure("image", reason)])
        assert reason not in body, f"{reason} 的码泄漏进了用户文案:{body}"


def test_kind_label_is_chinese_not_the_internal_id():
    """⛔ 用户文案里不许出现内部标识 ``image`` / ``file`` / ``video``。

    🔴 上一轮我就是直接把 kind 拼进去的，用户会看到
    「未能读取你发送的附件（image）」。
    """
    a = _adapter()
    for kind, label in (("image", "图片"), ("file", "文件"), ("video", "视频")):
        body = a._media_failure_reply_text([MediaFailure(kind, "download_failed")])
        assert label in body and kind not in body, f"{kind} 没翻译:{body}"


def test_agent_note_keeps_the_reason():
    """⚠️ 与用户侧相反:给 **Agent** 的提示**要**带 reason —— 它得据此说话。"""
    a = _adapter()
    note = a._media_failure_note([MediaFailure("image", "cache_write_failed")])
    assert "cache_write_failed" in note, f"Agent 拿不到成因:{note}"
    assert "图片" in note


# ═════════ ④ ⛔ 必须保持不变的行为 ═════════


@pytest.mark.asyncio
async def test_legit_voice_is_never_reported_as_failed():
    """⛔ 合法语音不许被判成失败。

    ``VoiceContent`` 只有 ``content``（转写文本），**没有** url / aeskey ——
    它本来就不该走下载。把它算进畸形 = 每条语音都误报。
    """
    a = _adapter()
    _p, _t, failures = await a._extract_media(
        {"msgtype": "voice", "voice": {"content": "你好"}})
    assert failures == [], f"合法语音被误判成失败:{failures}"


@pytest.mark.asyncio
async def test_text_only_mixed_message_is_not_reported():
    """⛔ 纯文本的 mixed 消息不含图片 ⇒ 不是失败。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media({
        "msgtype": "mixed",
        "mixed": {"msg_item": [{"msgtype": "text", "text": {"content": "只有字"}}]},
    })
    assert failures == []


@pytest.mark.asyncio
async def test_appmsg_without_attachment_is_not_reported():
    """⛔ ``appmsg`` 没有 file/image 是**合法**的（链接卡片）⇒ 不许误报。

    ⭐ 这条划出了 P2-3 的作用域上界:只有**协议里必填**的载荷缺失才算畸形。
    改宽了就会对每张链接卡片都回一句「未能读取你发送的附件」。
    """
    a = _adapter()
    _p, _t, failures = await a._extract_media(
        {"msgtype": "appmsg", "appmsg": {"title": "某链接", "url": "https://x"}})
    assert failures == []


@pytest.mark.asyncio
async def test_plain_text_is_not_reported():
    """⛔ 纯文本消息:一条失败都不许有。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media(
        {"msgtype": "text", "text": {"content": "你好"}})
    assert failures == []


@pytest.mark.asyncio
async def test_successful_media_records_no_failure(monkeypatch):
    """⛔ 成功路径 ``failures`` 必须为空 —— 否则每条正常图片都会带上提示。"""
    a = _adapter()
    monkeypatch.setattr(
        wecom_adapter, "cache_image_from_bytes", lambda raw, ext: "/cache/a.png")

    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    monkeypatch.setattr(a, "_download_remote_bytes", _dl)
    paths, _t, failures = await a._extract_media(
        {"msgtype": "image", "image": {"url": "https://wework.qpic.cn/x"}})
    assert paths == ["/cache/a.png"] and failures == []


# ═════════ RH 第五轮：mixed 外层容器 / quoted mixed / 超限 ═════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mixed", [None, "a-string", 123, [], {"msg_item": None},
              {"msg_item": "not-a-list"}, {}],
    ids=["none", "str", "int", "list", "items-none", "items-str", "no-items"])
async def test_mixed_container_malformed_is_reported(mixed):
    """🔴 P2-4：``mixed`` 外层容器畸形仍被当空消息。

    我上一轮只补了**子项**畸形，外层容器（``mixed`` 本身 / ``msg_item``）
    被 `isinstance(...) else {}` **归一成空** ⇒ 纯附件的畸形 mixed 静默消失。
    ⭐ 兄弟调用点没跟上 —— 同一个函数里，两层都是必填的。
    """
    a = _adapter()
    _p, _t, failures = await a._extract_media({"msgtype": "mixed", "mixed": mixed})
    assert failures == [MediaFailure("mixed", "payload_malformed")], (
        f"mixed 外层畸形被静默丢弃:{failures}")


@pytest.mark.asyncio
async def test_legit_empty_mixed_item_list_is_not_malformed():
    """⛔ 作用域上界：``msg_item: []`` 形状是**对的** ⇒ 不算畸形。"""
    a = _adapter()
    _p, _t, failures = await a._extract_media(
        {"msgtype": "mixed", "mixed": {"msg_item": []}})
    assert failures == []


@pytest.mark.asyncio
async def test_quoted_mixed_image_is_visible(monkeypatch):
    """🔴 P2-5：引用一条图文混排消息时，那张图原先**完全不可见**。

    官方 ``QuoteContent`` 允许 ``mixed``，而 quoted 分支只认 image/file。
    用户引用图文消息再问「看这张图」⇒ Agent 既拿不到引用文字也拿不到图。
    """
    a = _adapter()
    monkeypatch.setattr(
        wecom_adapter, "cache_image_from_bytes", lambda raw, ext: "/cache/q.png")

    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    monkeypatch.setattr(a, "_download_remote_bytes", _dl)

    paths, _t, failures = await a._extract_media({
        "msgtype": "text", "text": {"content": "看这张图"},
        "quote": {"msgtype": "mixed", "mixed": {"msg_item": [
            {"msgtype": "text", "text": {"content": "原消息"}},
            {"msgtype": "image", "image": {"url": "https://wework.qpic.cn/q"}},
        ]}},
    })
    assert paths == ["/cache/q.png"], f"引用消息里的图片拿不到:{paths}"
    assert failures == []


def test_quoted_mixed_text_is_visible():
    """引用 mixed 的**文字**同样要拿到 —— ⛔ 只修图片是半条链。"""
    a = _adapter()
    text, reply = a._extract_text({
        "msgtype": "text", "text": {"content": "看这张图"},
        "quote": {"msgtype": "mixed", "mixed": {"msg_item": [
            {"msgtype": "text", "text": {"content": "原消息正文"}},
        ]}},
    })
    assert text == "看这张图"
    assert reply == "原消息正文", f"引用 mixed 的文字丢了:{reply!r}"


@pytest.mark.asyncio
async def test_oversize_is_not_reported_as_download_failure(monkeypatch):
    """🔴 P2-6：超限被归成 ``download_failed`` ⇒ 劝用户重发同一个文件，必然再失败。"""
    a = _adapter()

    async def _too_big(url, max_bytes=None):
        raise wecom_adapter.WeComMediaTooLarge("exceeds limit: 99 > 1")

    monkeypatch.setattr(a, "_download_remote_bytes", _too_big)

    _p, _t, failures = await a._extract_media(
        {"msgtype": "file", "file": {"url": "https://wework.qpic.cn/big"}})
    assert failures == [MediaFailure("file", "too_large")], f"超限被误分类:{failures}"


def test_oversize_advice_says_compress_not_resend():
    """⭐ 「压缩」在这里 —— 也**只在**这里 —— 是正确建议。"""
    a = _adapter()
    body = a._media_failure_reply_text([MediaFailure("file", "too_large")])
    assert "压缩" in body, f"超限没给可行动建议:{body}"
    net = a._media_failure_reply_text([MediaFailure("file", "download_failed")])
    assert "压缩" not in net, f"「压缩」又漏到别的成因上了:{net}"


# ═════════ RH 第六轮：mixed 元素层畸形 ═════════


@pytest.mark.asyncio
async def test_mixed_non_dict_element_is_reported():
    """🔴 ``msg_item`` 是 list 但混了非 dict 元素 ⇒ 仍要留痕。

    ⭐ 容器层和元素层是**两层**。我上一轮修了容器层就以为修完了 ——
    这是同一个缺陷第三次「兄弟点没跟上」。
    """
    a = _adapter()
    _p, _t, failures = await a._extract_media({
        "msgtype": "mixed", "mixed": {"msg_item": ["not-a-dict", 123]}})
    assert failures == [MediaFailure("mixed", "payload_malformed")]


@pytest.mark.asyncio
async def test_mixed_keeps_valid_items_when_one_element_is_broken(monkeypatch):
    """🔴 ⛔ 一个坏元素不许把整条消息里的好图片一起丢掉。

    ⭐ 这条钉的是**修复的作用域**：记账要发生，但合法子项照常投递。
    我第一版在畸形时直接 return，正是「改宽了弄坏原来对的东西」。
    """
    a = _adapter()
    monkeypatch.setattr(
        wecom_adapter, "cache_image_from_bytes", lambda raw, ext: "/cache/ok.png")

    async def _dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    monkeypatch.setattr(a, "_download_remote_bytes", _dl)

    paths, _t, failures = await a._extract_media({
        "msgtype": "mixed", "mixed": {"msg_item": [
            "broken-element",
            {"msgtype": "image", "image": {"url": "https://wework.qpic.cn/ok"}},
        ]}})

    assert paths == ["/cache/ok.png"], f"合法子项被坏元素连坐丢掉:{paths}"
    assert failures == [MediaFailure("mixed", "payload_malformed")]
