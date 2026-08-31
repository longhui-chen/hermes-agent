"""微信：附件下载失败时，整条消息**不许消失**（RH 复审第四轮 P1-2）。

现场：``_collect_media()`` 四类下载（image / video / file / voice）失败时都只是
「不 append」，调用方只看到空 ``media_paths``，随后在 ``_process_message`` 的
``if not text and not media_paths: return`` 直接跳过 —— 用户发的图片、视频、
文件或语音只要下载失败，就是**「发了但没有任何回复」**。

🔴 这是 wecom 那条缺陷的**同根孪生**。我上一轮只修了 wecom、没扫这里 ——
**兄弟调用点没跟上**。这已经是用户点名那件事的第三处
（板端回调丢 · wecom · weixin）。

⭐ 修法照抄刚在 wecom 做对的那套（把「有没有尝试过」从空列表里拆出来），
但用 weixin 自己的**出参风格**，⛔ 不强行改成返回三元组。

🔴 门从 ``_process_message`` 驱动 —— ⛔ 不从 ``_collect_media`` 内部驱动。
上一轮 wecom 那个叫 ``test_whole_message_survives_a_disk_failure`` 的测试就是
只驱动内部函数，名字说的作用域比它真正驱动的大，缺陷在它眼皮底下绿着。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms import weixin
from tests.gateway.test_weixin import _make_adapter


def _adapter(monkeypatch):
    a = _make_adapter()
    a._poll_session = Mock()
    a._token = None
    a._cdn_base_url = "https://example.invalid"
    # ⚠️ 纯文本（以及「正文 + 附件全失败」，因为它 message_type 仍是 TEXT）
    # 走 debounce 队列，⛔ 不直接进 handle_message。延迟设 0 只是不等待，
    # flush 仍在**另一个 task** 里跑 ⇒ 必须 ``_settle()`` 把它 await 掉。
    a._text_batch_delay_seconds = 0
    a._text_batch_split_delay_seconds = 0
    a.handle_message = AsyncMock()
    a.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    return a


async def _settle(a):
    """把挂起的 debounce flush task 跑完。

    ⛔ 不用 ``asyncio.sleep(0.2)`` 猜时间 —— 那是 flaky 的来源。
    直接 await 真实 task，⭐ 判据与实现同源但**时序确定**。
    """
    tasks = list(a._pending_text_batch_tasks.values())
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _msg(item_type, *, text=None):
    item = {"type": item_type}
    if item_type == weixin.ITEM_IMAGE:
        item["image_item"] = {"media": {"full_url": "https://example.invalid/a.jpg"}}
    elif item_type == weixin.ITEM_VIDEO:
        item["video_item"] = {"media": {"full_url": "https://example.invalid/a.mp4"}}
    elif item_type == weixin.ITEM_FILE:
        item["file_item"] = {"media": {"full_url": "https://example.invalid/a.pdf"}}
    elif item_type == weixin.ITEM_VOICE:
        item["voice_item"] = {"media": {"full_url": "https://example.invalid/a.silk"}}
    items = [item]
    if text:
        items.insert(0, {"type": 1, "text_item": {"text": text}})
    return {
        "from_user_id": "user-123",
        "to_user_id": "test-account",
        "message_id": "msg-1",
        "msg_type": 1,
        "item_list": items,
    }


def _break_downloads(monkeypatch):
    """让四类下载全部失败 —— 模拟网络/解密/CDN 故障。"""
    async def _boom(*_a, **_k):
        raise RuntimeError("download failed")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _boom)


# ───────────── 缺陷本体：四类全集，从真实入口驱动 ─────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type,label",
    [
        (weixin.ITEM_IMAGE, "图片"),
        (weixin.ITEM_VIDEO, "视频"),
        (weixin.ITEM_FILE, "文件"),
        (weixin.ITEM_VOICE, "语音"),
    ],
)
async def test_pure_media_failure_still_reaches_the_user(monkeypatch, item_type, label):
    """🔴 四类下载失败**全集**：纯附件消息不许被静默丢弃。

    ⭐ 四条是同构的，只测 image 等于给另外三条发免检。
    """
    a = _adapter(monkeypatch)
    _break_downloads(monkeypatch)

    await a._process_message(_msg(item_type))

    assert a.send.await_count == 1, (
        f"{label}下载失败后用户没收到任何回应 —— 「发了没反应」原样复现")
    body = a.send.await_args.args[1]
    assert label in body, f"回复没说清是哪类附件:{body}"
    assert "重新发送" in body, f"回复没给下一步:{body}"
    # 🔴 ⛔ 不许给「先压缩」——这条路径的失败是下载/解密，压缩确定无效。
    assert "压缩" not in body, f"给了对本次失败无效的建议:{body}"


@pytest.mark.asyncio
async def test_reply_leaks_nothing(monkeypatch):
    """⛔ 给用户的话不许含 url / 原始错误。"""
    a = _adapter(monkeypatch)
    _break_downloads(monkeypatch)

    await a._process_message(_msg(weixin.ITEM_IMAGE))

    body = a.send.await_args.args[1]
    assert "example.invalid" not in body, f"url 泄漏:{body}"
    assert "download failed" not in body and "RuntimeError" not in body, f"原始错误泄漏:{body}"


@pytest.mark.asyncio
async def test_text_plus_failed_media_uses_channel_prompt(monkeypatch):
    """有正文 + 附件失败 ⇒ 提示走 ``channel_prompt``，⛔ 不许拼进 ``text``。

    拼进 text 等于把系统内容伪装成用户原话（会进命令解析、批处理合并、
    持久化历史）—— 这正是 wecom 那边我做错、被 RH 抓出来的层级问题。
    ⭐ 孪生修复也要把**层级**一起抄对，⛔ 不只抄「加了提示」。
    """
    a = _adapter(monkeypatch)
    _break_downloads(monkeypatch)

    await a._process_message(_msg(weixin.ITEM_IMAGE, text="看看这张"))
    await _settle(a)

    assert a.handle_message.await_count == 1, "有正文却没进 handle_message"
    event = a.handle_message.await_args.args[0]
    assert event.text == "看看这张", f"用户正文被改写了:{event.text!r}"
    assert "未能取到" in (event.channel_prompt or ""), (
        f"Agent 不知道用户发过附件:{event.channel_prompt!r}")
    assert a.send.await_count == 0, "正文已在处理，还额外发了提示 —— 刷屏"


# ───────────── ⛔ 必须保持不变的行为 ─────────────

@pytest.mark.asyncio
async def test_plain_text_is_untouched(monkeypatch):
    """🔴 纯文本消息：一个字节不变，⛔ 不许多发任何东西。"""
    a = _adapter(monkeypatch)

    await a._process_message({
        "from_user_id": "user-123", "to_user_id": "test-account",
        "message_id": "msg-t", "msg_type": 1,
        "item_list": [{"type": 1, "text_item": {"text": "你好"}}],
    })
    await _settle(a)

    assert a.handle_message.await_count == 1
    assert a.handle_message.await_args.args[0].text == "你好"
    assert a.send.await_count == 0
    assert not (a.handle_message.await_args.args[0].channel_prompt or "")


@pytest.mark.asyncio
async def test_genuinely_empty_message_is_still_skipped(monkeypatch):
    """⛔ 真的空消息仍然跳过 —— 缺陷是「有附件却被当成空」，
    ⛔ 不是「空消息也要回复」。修宽了就会对每条空事件都发一句话。"""
    a = _adapter(monkeypatch)

    await a._process_message({
        "from_user_id": "user-123", "to_user_id": "test-account",
        "message_id": "msg-e", "msg_type": 1, "item_list": [],
    })

    assert a.handle_message.await_count == 0
    assert a.send.await_count == 0, "空消息也回了话"


@pytest.mark.asyncio
async def test_successful_media_is_unaffected(monkeypatch, tmp_path):
    """⛔ 下载成功时行为与从前一致（⛔ 不发失败提示）。"""
    a = _adapter(monkeypatch)

    async def _ok(*_a, **_k):
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(
        weixin, "cache_image_from_bytes",
        lambda data, ext=".jpg": str(tmp_path / "a.jpg"))

    await a._process_message(_msg(weixin.ITEM_IMAGE))

    assert a.handle_message.await_count == 1
    event = a.handle_message.await_args.args[0]
    assert event.media_urls == [str(tmp_path / "a.jpg")]
    assert not (event.channel_prompt or ""), "成功路径被插入了失败提示"
    assert a.send.await_count == 0


# ───────── 批处理合并：⛔ 提示不许在 merge 分支里蒸发 ─────────

@pytest.mark.asyncio
async def test_batch_merge_keeps_the_media_failure_note(monkeypatch):
    """🔴 「半条链」:新造了 per-event ``channel_prompt``，但合并分支只搬
    text/media ⇒ 提示被静默丢弃，缺陷在批处理路径上原样复活。

    ⭐ 现场很常见:用户先打一句「帮我看下」，紧接着发「这张图」+ 图片。
    两条落在同一 session key 上 ⇒ 走 merge 分支。
    ⚠️ 缺的那半条**不在 diff 里** —— 只看修复本身永远看不见它。
    """
    a = _adapter(monkeypatch)
    a._text_batch_delay_seconds = 5          # ⭐ 撑住窗口，逼出 merge 分支
    a._text_batch_split_delay_seconds = 5

    await a._process_message({
        "from_user_id": "user-123", "to_user_id": "test-account",
        "message_id": "msg-a", "msg_type": 1,
        "item_list": [{"type": 1, "text_item": {"text": "帮我看下"}}],
    })
    assert a.handle_message.await_count == 0, "第一条没进批处理，本门测不到 merge"

    _break_downloads(monkeypatch)
    await a._process_message(_msg(weixin.ITEM_IMAGE, text="这张图"))

    # 强制立刻 flush，⛔ 不等 5 秒
    a._text_batch_delay_seconds = 0
    a._text_batch_split_delay_seconds = 0
    key = next(iter(a._pending_text_batches))
    a._pending_text_batch_tasks[key].cancel()
    a._pending_text_batch_tasks[key] = asyncio.create_task(a._flush_text_batch(key))
    await _settle(a)

    assert a.handle_message.await_count == 1, "合并后没有投递"
    event = a.handle_message.await_args.args[0]
    assert event.text == "帮我看下\n这张图", f"正文合并被改坏:{event.text!r}"
    assert "未能取到" in (event.channel_prompt or ""), (
        f"合并把附件失败提示丢了 —— Agent 又不知道有附件:{event.channel_prompt!r}")


@pytest.mark.asyncio
async def test_batch_merge_does_not_duplicate_the_same_note(monkeypatch):
    """⛔ 连发两张都失败的图 ⇒ 提示只出现一次，不许堆叠。

    ⭐ 复用 ``_merge_caption`` 正是为了拿到这条去重语义，
    这条门钉的就是「我确实复用了它」的**可观察后果**。
    """
    a = _adapter(monkeypatch)
    a._text_batch_delay_seconds = 5
    a._text_batch_split_delay_seconds = 5
    _break_downloads(monkeypatch)

    for i, t in enumerate(("图一", "图二")):
        m = _msg(weixin.ITEM_IMAGE, text=t)
        m["message_id"] = f"msg-{i}"
        await a._process_message(m)

    a._text_batch_delay_seconds = 0
    a._text_batch_split_delay_seconds = 0
    key = next(iter(a._pending_text_batches))
    a._pending_text_batch_tasks[key].cancel()
    a._pending_text_batch_tasks[key] = asyncio.create_task(a._flush_text_batch(key))
    await _settle(a)

    prompt = a.handle_message.await_args.args[0].channel_prompt or ""
    assert prompt.count("未能取到") == 1, f"提示被重复堆叠:{prompt!r}"
