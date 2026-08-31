"""长文本分块：部分成功必须按【成功】收口（RH 复审 #3）。

现场：``send`` 把长回复切成多块依次发送，原本用**最后一块**的
响应决定整体成败。首块成功时用户**已经看到内容**、引用（quote）也已经消费掉；
若末块失败就按整体失败收口，上层 ``_consume_feishu_quote(metadata, result)``
会把 quote lease 退回，进而可能**重发整段** —— 用户看到的是重复刷屏，比缺
最后一块严重得多。

⭐ 这里钉的是**用户最终看到的结果**，不是"收口逻辑内部算出了什么"：
``SendResult.success`` 与 ``message_id`` 是上层唯一的依据 ——
``gateway/platforms/base.py::_consume_feishu_quote`` 直接把它交给
``lease.consume(metadata, result)``，lease 退不退**完全由这两个字段决定**。
⇒ 钉住它们，就钉住了"部分成功时 lease 不该退"这一格。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


def _ok(message_id: str):
    return SimpleNamespace(
        success=lambda: True, code=0, msg="", data=SimpleNamespace(message_id=message_id)
    )


def _fail():
    return SimpleNamespace(success=lambda: False, code=99991400, msg="boom", data=None)


def _adapter(chunks: list[str]):
    """最小 adapter —— 只绕过 __init__，被测方法全部用真实实现。"""
    a = FeishuAdapter.__new__(FeishuAdapter)
    a._client = object()
    a.truncate_message = lambda _text, _limit: list(chunks)
    a.format_message = lambda text: text
    return a


@pytest.mark.asyncio
async def test_first_chunk_success_last_chunk_failure_reports_success():
    """🔴 用户报的那件事：首块已送达 ⇒ ⛔ 不许整体报失败。"""
    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_ok("om_first"), _fail()])

    result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert a._feishu_send_with_retry.await_count >= 2, (
        "calibration: 没有真的发两块,下面的断言会恒真")
    assert result.success is True, (
        "首块已送达、引用已消费,却按失败收口 ⇒ 上层会退回 quote lease 并重发整段")
    assert result.message_id == "om_first", (
        f"message_id 必须是【用户实际看到的那条】(首个成功块),实际={result.message_id!r}")


@pytest.mark.asyncio
async def test_all_chunks_failing_still_reports_failure():
    """⛔ 正常功能不许被弄坏：一块都没送达时必须是失败。"""
    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_fail(), _fail()])

    result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert result.success is False


@pytest.mark.asyncio
async def test_single_chunk_success_unchanged():
    """⛔ 单块（绝大多数消息）的行为逐字不变。"""
    a = _adapter(["only"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_ok("om_only")])

    result = await a.send("oc_chat", "short", reply_to=None, metadata=None)

    assert result.success is True
    assert result.message_id == "om_only"


@pytest.mark.asyncio
async def test_quote_is_consumed_by_the_first_successful_chunk_only():
    """引用只能挂在首个成功块上 —— 后续块 ⛔ 不许再带 reply_to。

    这是「lease 不该退」的另一面：引用**确实已经用掉了**。若后续块还带
    reply_to，用户会看到同一条消息被引用多次。
    """
    a = _adapter(["part-1", "part-2", "part-3"])
    a._feishu_send_with_retry = AsyncMock(
        side_effect=[_ok("om_first"), _ok("om_second"), _ok("om_third")]
    )

    await a.send("oc_chat", "long", reply_to="om_user", metadata=None)

    replies = [c.kwargs.get("reply_to") for c in a._feishu_send_with_retry.await_args_list]
    assert replies[0] == "om_user", "calibration: 首块没带引用,后面的断言会恒真"
    assert all(r is None for r in replies[1:]), (
        f"引用被重复挂到后续块上 ⇒ 用户看到同一条被引用多次:{replies}")


# ───────── RH 复审补漏：两种情形原先一条测试都没有 ─────────

@pytest.mark.asyncio
async def test_missing_head_is_distinguishable_from_missing_tail(caplog):
    """🔴 首块失败、后块成功 —— 用户看到的内容**从中间开始**。

    原先这条路径完全没测。它比缺尾危险：缺尾用户能感觉到"话没说完"，
    缺头则像"模型答非所问"，用户根本不知道少了东西。
    ⇒ 至少日志必须能区分,否则线上无法定位。
    """
    import logging

    a = _adapter(["head", "tail"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_fail(), _ok("om_tail")])

    with caplog.at_level(logging.WARNING):
        result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert a._feishu_send_with_retry.await_count >= 2, "calibration: 没真发两块"
    assert result.success is True
    assert "缺开头" in caplog.text, (
        f"缺头与缺尾被压成同一条日志,线上无法区分:{caplog.text!r}")
    assert "补发一轮后" in caplog.text, (
        "日志没说明已经补发过一轮 —— 排查的人不知道系统试过没有")


@pytest.mark.asyncio
async def test_missing_tail_says_so(caplog):
    """对照组：缺尾必须报「缺结尾」，⛔ 不许与缺头混为一谈。"""
    import logging

    a = _adapter(["head", "tail"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_ok("om_head"), _fail()])

    with caplog.at_level(logging.WARNING):
        await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert "缺结尾" in caplog.text, f"{caplog.text!r}"
    assert "缺开头" not in caplog.text


@pytest.mark.asyncio
async def test_exception_after_partial_delivery_leaves_a_trace(caplog):
    """🔴 先成功、后抛异常：返回 success=False 且不带已送达信息。

    上层可能重发整段 ⇒ 用户看到前半部分两次。本 lane 闭合不了
    （消费侧不读 partial 元数据），但**必须留痕** —— 否则线上只有一条
    "Send error"，看不出用户其实已经看到了一半。
    """
    import logging

    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(
        side_effect=[_ok("om_first"), RuntimeError("connection reset")]
    )

    with caplog.at_level(logging.ERROR):
        result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert result.success is False, "抛异常仍应按失败收口"
    assert "已送达 1 块后失败" in caplog.text, (
        f"没记下「抛错前已经送达了多少」⇒ 无法判断用户看到了什么:{caplog.text!r}")
    assert "用户会看到重复内容" in caplog.text, (
        "没有提示重发会造成重复 —— 这正是这条路径最大的风险")


# ───────── P1-1：只重发失败的那几块（RH 复审第二轮） ─────────

@pytest.mark.asyncio
async def test_failed_chunk_is_retried_and_can_succeed(monkeypatch):
    """🔴 平台**返回失败响应**的块必须被补发一次。

    ``_feishu_send_with_retry`` 的重试**只覆盖抛异常**（``except Exception``），
    平台返回失败响应（限流 / 临时 5xx）时它直接 ``return response`` ——
    那一块**一次都没重试过**，内容就这么缺了。
    """
    monkeypatch.setattr("asyncio.sleep", AsyncMock())      # ⛔ 别真等 2 秒
    a = _adapter(["part-1", "part-2"])
    # 第一轮：块1 成功、块2 失败；补发轮：块2 成功
    a._feishu_send_with_retry = AsyncMock(
        side_effect=[_ok("om_1"), _fail(), _ok("om_2")])

    result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert a._feishu_send_with_retry.await_count == 3, (
        f"失败的块没有被补发（共调用 {a._feishu_send_with_retry.await_count} 次）")
    assert result.success is True


@pytest.mark.asyncio
async def test_all_chunks_failing_is_not_retried(monkeypatch):
    """⛔ 全失败 ⇒ 不补发。

    全失败说明是全局故障（鉴权失效 / 群被解散 / 网络断），
    补发只会拖延并再失败一轮，而用户在等。
    ⭐ 「至少一块成功」这个前提是刻意的，⛔ 不是省事。
    """
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(side_effect=[_fail(), _fail()])

    result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert a._feishu_send_with_retry.await_count == 2, (
        "全失败时还去补发 —— 会再等一轮退避、再失败一次")
    assert result.success is False


@pytest.mark.asyncio
async def test_retry_does_not_reuse_the_quote(monkeypatch):
    """⛔ 补发不许再带引用 —— 引用已被第一轮消费掉了。"""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(
        side_effect=[_ok("om_1"), _fail(), _ok("om_2")])

    await a.send("oc_chat", "long text", reply_to="om_orig", metadata=None)

    retry_call = a._feishu_send_with_retry.await_args_list[2]
    assert retry_call.kwargs["reply_to"] is None, (
        f"补发又带上了引用:{retry_call.kwargs['reply_to']}")


@pytest.mark.asyncio
async def test_retry_that_still_fails_keeps_the_old_contract(monkeypatch):
    """补发仍失败 ⇒ 回到原来的收口（成功 + 详细日志），⛔ 行为不变。"""
    import logging

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    a = _adapter(["part-1", "part-2"])
    a._feishu_send_with_retry = AsyncMock(
        side_effect=[_ok("om_1"), _fail(), _fail()])

    with caplog_at(logging.WARNING) as cap:
        result = await a.send("oc_chat", "long text", reply_to=None, metadata=None)

    assert result.success is True
    assert result.message_id == "om_1"
    assert "补发一轮后" in cap.text, f"日志没说明补发过:{cap.text!r}"


from contextlib import contextmanager  # noqa: E402


@contextmanager
def caplog_at(level):
    """极简 caplog 替身（本文件没有用 pytest 的 caplog fixture 的用例签名）。"""
    import io
    import logging

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    logger = logging.getLogger("plugins.platforms.feishu.adapter")
    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(level)
    try:
        yield type("C", (), {"text": property(lambda _s: stream.getvalue())})()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old)
