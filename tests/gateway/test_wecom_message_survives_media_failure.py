"""企微：附件取不到时，整条消息**不许消失**（RH 复审第二轮 P1-3）。

现场：``_extract_media()`` 返回空后，``_on_message`` 的
``if not text and not media_urls: return`` 直接跳过 —— ``handle_message``
**从不被调用**（RH 实测 ``handle_message_calls=0``）。用户看到的是
「发了图没反应」，而链路上没有任何一层告诉他发生了什么。

⚠️ 这就是**族 A 的企微现场**：不是下载失败，是**根本没接**。
（板端侧同病：企微图片/文件/视频在回调阶段就被整个丢弃。两端合起来才是
用户那句「发图没反应」。）

🔴 **本文件必须从 ``_on_message`` 驱动。**
上一版有个测试叫 ``test_whole_message_survives_a_disk_failure``，
名字这么写，实际只驱动了 ``_extract_media`` —— **没经过真实入口**，
所以缺陷就在它眼皮底下绿着。⭐ 测试名声称的作用域必须等于它真正驱动的作用域。
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.wecom import adapter as wecom_adapter
from plugins.platforms.wecom.adapter import WeComAdapter


def _boom(*_a, **_kw):
    raise OSError(28, "No space left on device")


@pytest.fixture
def adapter(monkeypatch):
    """能真正跑 ``_on_message`` 的最小 adapter —— ⛔ 绕过 __init__，
    但被测路径上的方法全部用真实实现。"""

    class _A(WeComAdapter):
        name = "wecom"

    a = _A.__new__(_A)
    a._dedup = SimpleNamespace(is_duplicate=lambda _m: False)
    a._remember_reply_req_id = lambda *a_, **k_: None
    a._remember_chat_req_id = lambda *a_, **k_: None
    a._is_group_allowed = lambda *a_, **k_: True
    a._is_dm_intake_allowed = lambda *a_, **k_: True
    a._payload_req_id = lambda _p: "req-1"
    # ⭐ 用**真的** SessionSource —— ⛔ 不用 SimpleNamespace 假冒。
    # 上一版假 source 的 ``platform`` 是 str，一进 build_session_key 就炸;
    # 夹具长得不像现场，测出来的就是另一个世界（今晚第二次栽在这上面）。
    from gateway.session import SessionSource, Platform

    a.build_source = lambda **kw: SessionSource(platform=Platform.WECOM, **kw)
    a.config = SimpleNamespace(extra={})
    a._pending_text_batches = {}
    a._pending_text_batch_tasks = {}

    # 文本批处理与本门无关，关掉它走直投路径（⛔ 不改被测逻辑，只是配置）
    # ⚠️ 下面 merge 那两条会自己调高它 —— 那是唯一能逼出合并分支的办法。
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    a.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    return a


def _payload(body):
    return {"body": body}


def _image_msg(text=None, aeskey=None):
    """⚠️ 默认**不带** aeskey：带了就会走 AES 解密分支并因假 key 失败，
    那样「成功路径」用例测的其实是另一种失败 —— 我第一版就踩了这个。"""
    image = {"url": "https://wework.qpic.cn/x"}
    if aeskey:
        image["aeskey"] = aeskey
    body = {
        "msgid": "mid-1",
        "msgtype": "image",
        "image": image,
        "from": {"userid": "u1"},
        "chatid": "oc_1",
    }
    if text:
        body["text"] = {"content": text}
    return _payload(body)


# ───────────────── 缺陷本体：从真实入口驱动 ─────────────────

@pytest.mark.asyncio
async def test_pure_media_failure_still_reaches_the_user(adapter, monkeypatch, caplog):
    """🔴 纯附件消息取不到内容 ⇒ ⛔ 不许静默丢弃整条消息。

    原先 ``handle_message`` 一次都不会被调用，用户等不到任何回应。
    """
    for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(wecom_adapter, w, _boom)

    async def _fake_dl(url, max_bytes=None):
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, {"content-type": "image/png"}

    monkeypatch.setattr(adapter, "_download_remote_bytes", _fake_dl)

    with caplog.at_level(logging.WARNING):
        await adapter._on_message(_image_msg())

    assert adapter.send.await_count == 1, (
        f"用户没有收到任何回应 —— 「发了图没反应」原样复现"
        f"（send 调用 {adapter.send.await_count} 次）")
    body = adapter.send.await_args.args[1]
    # ⭐ 钉的是**用户能看懂的类别 + 下一步**,⛔ 不是「出现『附件』二字」——
    # 后者是措辞,前者才是需求。现在文案会说清是「图片」还是「文件」。
    assert "图片" in body, f"回复没说清是哪类东西没读到:{body}"
    assert "重新" in body or "重试" in body, f"回复没给用户下一步:{body}"
    # 🔴 这条失败是磁盘写满 ⇒ ⛔ 不许再给「先压缩后再试」这种确定无效的建议。
    assert "压缩" not in body, f"给了对本次失败无效的建议:{body}"


@pytest.mark.asyncio
async def test_reply_never_leaks_internals(adapter, monkeypatch):
    """⛔ 给用户的那句话不许含 url / 原始错误 / 内部字段。"""
    for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(wecom_adapter, w, _boom)

    secret = "https://wework.qpic.cn/media?token=SUPER_SECRET"

    async def _fake_dl(url, max_bytes=None):
        return b"\x89PNG\r\n\x1a\n", {"content-type": "image/png"}

    monkeypatch.setattr(adapter, "_download_remote_bytes", _fake_dl)
    msg = _image_msg()
    msg["body"]["image"]["url"] = secret

    await adapter._on_message(msg)

    body = adapter.send.await_args.args[1]
    assert "SUPER_SECRET" not in body and secret not in body, f"凭据泄漏:{body}"
    assert "Errno" not in body and "No space left" not in body, f"原始错误泄漏:{body}"


@pytest.mark.asyncio
async def test_text_plus_failed_media_tells_the_agent(adapter, monkeypatch):
    """有正文 + 附件取不到 ⇒ 正文照常处理，但 **Agent 要知道有附件没取到**。

    否则它连「你发的图我没收到」都说不出来，只会答非所问。
    ⚠️ ⛔ 不额外给用户发消息 —— 正文已经在处理，再发一条就是刷屏。
    """
    for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(wecom_adapter, w, _boom)

    async def _fake_dl(url, max_bytes=None):
        return b"\x89PNG\r\n\x1a\n", {"content-type": "image/png"}

    monkeypatch.setattr(adapter, "_download_remote_bytes", _fake_dl)

    await adapter._on_message(_image_msg(text="看看这张图"))

    assert adapter.handle_message.await_count == 1, "有正文却没进 handle_message"
    event = adapter.handle_message.await_args.args[0]
    # 🔴 正文必须**逐字不变** —— 上一版我把提示拼进了 text，
    # 等于把系统内容伪装成用户原话（会进命令解析、批处理合并、持久化历史）。
    # ⚠️ 而我上一版的测试**反而断言提示必须出现在 text**，
    #    把错误的层级固化成了期望。⭐ 测试会把 bug 钉成契约。
    assert event.text == "看看这张图", f"用户正文被改写了:{event.text!r}"
    assert "未能取到" in (event.channel_prompt or ""), (
        f"Agent 不知道用户发过附件 —— 它会答非所问:{event.channel_prompt!r}")
    assert adapter.send.await_count == 0, (
        "正文已经在处理，还额外发了一条附件失败提示 —— 刷屏")


# ───────────────── ⛔ 必须保持不变的行为 ─────────────────

@pytest.mark.asyncio
async def test_plain_text_path_is_untouched(adapter):
    """🔴 纯文本消息：一个字节都不许变，⛔ 不许多发任何东西。"""
    await adapter._on_message(_payload({
        "msgid": "mid-2", "msgtype": "text",
        "text": {"content": "你好"},
        "from": {"userid": "u1"}, "chatid": "oc_1",
    }))

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "你好"
    assert adapter.send.await_count == 0, "纯文本消息触发了额外发送"


@pytest.mark.asyncio
async def test_genuinely_empty_message_is_still_skipped(adapter, caplog):
    """⛔ 真的空消息（没有任何媒体引用）仍然跳过 —— 行为逐字不变。

    ⭐ 这一条是「作用域刚好等于缺陷」的另一半：缺陷是「有附件却被当成空」，
    ⛔ 不是「空消息也要回复」。修宽了就会对每条空事件都发一句话。
    """
    with caplog.at_level(logging.DEBUG):
        await adapter._on_message(_payload({
            "msgid": "mid-3", "msgtype": "text",
            "from": {"userid": "u1"}, "chatid": "oc_1",
        }))

    assert adapter.handle_message.await_count == 0
    assert adapter.send.await_count == 0, "空消息也回了话 —— 每个空事件都会刷一条"


@pytest.mark.asyncio
async def test_successful_media_is_unaffected(adapter, monkeypatch):
    """⛔ 附件正常取到时，行为与从前一致（⛔ 不发失败提示）。"""
    monkeypatch.setattr(
        wecom_adapter, "cache_image_from_bytes",
        lambda raw, ext: "/cache/images/a.png")

    async def _fake_dl(url, max_bytes=None):
        return b"\x89PNG\r\n\x1a\n", {"content-type": "image/png"}

    monkeypatch.setattr(adapter, "_download_remote_bytes", _fake_dl)

    await adapter._on_message(_image_msg())

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/cache/images/a.png"]
    assert not (event.channel_prompt or ""), "成功路径被插入了失败提示"
    assert adapter.send.await_count == 0


# ───────── 批处理合并：⛔ 提示不许在 merge 分支里蒸发 ─────────


async def _flush_now(a):
    """把挂起的批处理立刻 flush 掉，⛔ 不 sleep 等真实延迟。"""
    a._text_batch_delay_seconds = 0
    a._text_batch_split_delay_seconds = 0
    for key in list(a._pending_text_batches):
        task = a._pending_text_batch_tasks.get(key)
        if task:
            task.cancel()
        a._pending_text_batch_tasks[key] = asyncio.create_task(
            a._flush_text_batch(key))
    tasks = list(a._pending_text_batch_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_batch_merge_keeps_the_media_failure_note(adapter, monkeypatch):
    """🔴 我上一轮修复的**半条链**:新造了 per-event ``channel_prompt``，
    但 ``_enqueue_text_event`` 的合并分支只搬 text/media ⇒ 提示被丢弃。

    ⭐ 与 weixin 同形 —— 兄弟调用点全集:凡是「产出 per-event channel_prompt」
    的 adapter，它自己的批处理合并就必须一起搬这个字段。
    （其余 7 个 batcher 的 prompt 是每会话常量 ⇒ ⛔ 不在作用域内。）
    """
    adapter._text_batch_delay_seconds = 5
    adapter._text_batch_split_delay_seconds = 5

    await adapter._on_message(_payload({
        "msgid": "mid-a", "msgtype": "text",
        "text": {"content": "帮我看下"},
        "from": {"userid": "u1"}, "chatid": "oc_1",
    }))
    assert adapter.handle_message.await_count == 0, "第一条没进批处理，测不到 merge"

    for w in ("cache_image_from_bytes", "cache_document_from_bytes"):
        monkeypatch.setattr(wecom_adapter, w, _boom)

    async def _fake_dl(url, max_bytes=None):
        return b"\x89PNG\r\n\x1a\n", {"content-type": "image/png"}

    monkeypatch.setattr(adapter, "_download_remote_bytes", _fake_dl)

    msg = _image_msg(text="这张图")
    msg["body"]["msgid"] = "mid-b"
    await adapter._on_message(msg)
    await _flush_now(adapter)

    assert adapter.handle_message.await_count == 1, "合并后没有投递"
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "帮我看下\n这张图", f"正文合并被改坏:{event.text!r}"
    assert "未能取到" in (event.channel_prompt or ""), (
        f"合并把附件失败提示丢了 —— Agent 又不知道有附件:{event.channel_prompt!r}")


@pytest.mark.asyncio
async def test_batch_merge_leaves_prompt_none_when_nothing_failed(adapter):
    """⛔ 必须保持不变:两条都没有附件失败 ⇒ 合并结果的 prompt 仍是空。

    ⭐ 作用域检查的另一半 —— 修复不许给正常路径塞出一个非空 prompt。
    """
    adapter._text_batch_delay_seconds = 5
    adapter._text_batch_split_delay_seconds = 5

    for i, t in enumerate(("第一句", "第二句")):
        await adapter._on_message(_payload({
            "msgid": f"mid-{i}", "msgtype": "text",
            "text": {"content": t},
            "from": {"userid": "u1"}, "chatid": "oc_1",
        }))
    await _flush_now(adapter)

    event = adapter.handle_message.await_args.args[0]
    assert event.text == "第一句\n第二句"
    assert not (event.channel_prompt or ""), (
        f"正常路径被塞进了提示:{event.channel_prompt!r}")
