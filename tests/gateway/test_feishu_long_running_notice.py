"""ZET-2111：飞书长任务跑完没有任何推送提醒，用户不知道好了没有。

判据设计的要点（⛔ 这几条是本文件存在的理由，改动前先读）：

1. **「发过心跳」就是判据本身**。gateway 只在「跑够
   `HERMES_AGENT_NOTIFY_INTERVAL`」且「用户没关长任务通知」时才发心跳，所以
   `note_long_running_turn()` 被调到这件事，已经蕴含了这两个条件。
   ⛔ adapter 不许自己重新读阈值 / 开关判一遍 —— 那两个判据长在 gateway 的
   闭包里（`_display_surface_mode` 依赖 `source`/`user_config`/`platform_key`），
   adapter 复用不了，各判一次必然漂移。
2. **⛔ 只发一个语言无关的 ✅**：成功态不配文字，不复制答案（复制答案就变成
   ZET-2473 那种刷屏），零新增翻译文案。
3. **完成提醒独立于 reaction 开关**：reaction 是 UI 徽章，完成提醒是推送，
   两件事。关掉 reaction 的用户仍然需要知道长任务结束了。
4. **平铺**：它是状态消息不是会话正文 ⇒ 不引用、不进话题。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import ProcessingOutcome


def _adapter(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig())
    # reaction 原语与本门无关，桩掉避免碰真 API。
    monkeypatch.setattr(adapter, "_add_reaction", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock(return_value=True))
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    return adapter


def _event(chat_id="oc_1", thread_id=None, message_id="om_1"):
    return SimpleNamespace(
        message_id=message_id,
        source=SimpleNamespace(chat_id=chat_id, thread_id=thread_id),
    )


def _sent_texts(adapter):
    return [call.args[1] for call in adapter.send.await_args_list]


@pytest.mark.asyncio
async def test_long_running_success_gets_exactly_one_flat_tick(monkeypatch):
    adapter = _adapter(monkeypatch)
    event = _event()

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == ["✅"], "长任务成功后应当且仅当补一个 ✅"
    call = adapter.send.await_args_list[0]
    # 平铺：不引用、不进话题。
    assert call.kwargs.get("reply_to") is None
    assert call.kwargs.get("metadata") is None


@pytest.mark.asyncio
async def test_the_tick_is_not_repeated_on_a_second_hook_call(monkeypatch):
    adapter = _adapter(monkeypatch)
    event = _event()

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == ["✅"], "完成标记只能发一次"


@pytest.mark.asyncio
async def test_short_turn_gets_no_tick(monkeypatch):
    """没发过心跳 = 短问答 ⇒ ⛔ 不许多一个气泡。"""
    adapter = _adapter(monkeypatch)

    await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", [ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED]
)
async def test_failed_or_cancelled_long_turn_gets_no_tick(monkeypatch, outcome):
    """失败 / 取消各有自己的反馈，⛔ 不许再盖一个"完成"。"""
    adapter = _adapter(monkeypatch)
    event = _event()

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, outcome)

    assert _sent_texts(adapter) == []


@pytest.mark.asyncio
async def test_tick_survives_reactions_being_disabled(monkeypatch):
    """⭐ 完成提醒是推送，reaction 是徽章 —— 关掉 reaction 不该连提醒一起没。

    旧的 on_processing_complete 开头就 `if not self._reactions_enabled(): return`，
    把提醒挂在它后面会让关了 reaction 的用户永远收不到。
    """
    monkeypatch.setenv("FEISHU_REACTIONS", "false")
    adapter = _adapter(monkeypatch)
    event = _event()

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == ["✅"]


@pytest.mark.asyncio
async def test_tick_survives_a_reaction_removal_failure(monkeypatch):
    """reaction 移除失败会让既有逻辑早退 —— 提醒必须在那之前发出去。"""
    adapter = _adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_remove_reaction", AsyncMock(return_value=False))
    event = _event()
    adapter._remember_processing_reaction(event.message_id, "r1")

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == ["✅"]


@pytest.mark.asyncio
async def test_send_failure_does_not_turn_a_successful_turn_into_a_failure(monkeypatch):
    """完成标记发不出去不许改判已经成功的一轮 —— 正文早已送达。"""
    adapter = _adapter(monkeypatch)
    adapter.send = AsyncMock(side_effect=RuntimeError("transport down"))
    event = _event()

    adapter.note_long_running_turn(event.source)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)  # 不许抛


def test_marker_cache_is_bounded(monkeypatch):
    """turn 异常终止、hook 没跑到时不许无界增长。"""
    from plugins.platforms.feishu.adapter import _FEISHU_LONG_RUNNING_CACHE_SIZE

    adapter = _adapter(monkeypatch)
    for i in range(_FEISHU_LONG_RUNNING_CACHE_SIZE + 50):
        adapter.note_long_running_turn(SimpleNamespace(chat_id=f"oc_{i}", thread_id=None))

    assert len(adapter._long_running_turns) == _FEISHU_LONG_RUNNING_CACHE_SIZE


def test_other_platforms_are_untouched_by_the_new_hook():
    """⛔ 基类默认必须是 no-op：Slack / Telegram 零新增行为。"""
    from gateway.platforms.base import BasePlatformAdapter

    # 默认实现不碰任何状态、不返回任何东西、不抛。
    assert (
        BasePlatformAdapter.note_long_running_turn(
            object(), SimpleNamespace(chat_id="C1", thread_id=None)
        )
        is None
    )


# ═════════ RH 复审 P2-1：标记粒度太粗 + 消费提交在动作之前 ═════════

def test_long_running_key_is_stable_across_both_sides():
    """🔴 calibration：两侧必须拿到**同一类** source，否则 key 永远匹配不上。

    ``note_long_running_turn(source)`` 由 gateway 在心跳成功后调用，参数是
    ``_run_agent_inner`` 的 ``source``；``_notify_long_running_done`` 用的是
    ``event.source``。两者若不同源，加进 key 的 ``user_id`` 一边有一边没有 ⇒
    **完成提醒整个失效**，比原来的串号更坏。
    ⭐ 这条把「它们同源」这个**前提**钉住，⛔ 不靠我读代码时的印象。
    """
    import dataclasses

    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    assert MessageEvent.__annotations__.get("source") is SessionSource, (
        "event.source 不再是 SessionSource —— key 的 user_id 一侧会取不到")
    names = {f.name for f in dataclasses.fields(SessionSource)}
    assert {"chat_id", "thread_id", "user_id"} <= names, (
        f"SessionSource 少了 key 需要的字段:{names}")


@pytest.mark.asyncio
async def test_two_users_in_one_group_do_not_consume_each_other(monkeypatch):
    """🔴 同群并发：A 的长任务标记，⛔ 不许被 B 的短问答消费掉。

    上一版 key 只有 ``(chat_id, thread_id)`` ⇒ B 一完成就 pop 掉 A 的标记：
    **B 收到不属于他的 ✅，而 A 跑完反而没有提醒**（RH 运行时复现）。
    """
    adapter = _adapter(monkeypatch)
    a = SimpleNamespace(
        message_id="om_a",
        source=SimpleNamespace(chat_id="oc_1", thread_id=None, user_id="userA"))
    b = SimpleNamespace(
        message_id="om_b",
        source=SimpleNamespace(chat_id="oc_1", thread_id=None, user_id="userB"))

    adapter.note_long_running_turn(a.source)          # 只有 A 是长任务
    await adapter.on_processing_complete(b, ProcessingOutcome.SUCCESS)
    assert _sent_texts(adapter) == [], (
        "B 的短问答收到了 ✅ —— 它消费了 A 的标记")

    await adapter.on_processing_complete(a, ProcessingOutcome.SUCCESS)
    assert _sent_texts(adapter) == ["✅"], (
        "A 的长任务跑完没有提醒 —— 标记已被别人消费掉")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome_send",
    [("raise", None), ("reject", None)],
    ids=["send_raises", "send_rejected"],
)
async def test_marker_never_survives_into_the_next_turn(monkeypatch, outcome_send):
    """🔴 标记的生命周期就是这一轮 —— 发送失败也**必须**清掉。

    我上一版为了「失败还能重试」把 pop 挪到发送成功之后。**生产里没有下一次
    机会**：每轮只调一次 ``on_processing_complete``。于是失败的标记残留下来，
    被同一用户的**下一轮短问答**消费 —— 那一轮凭空收到一个 ✅，而它压根没
    发过心跳。⭐ 比原 bug 更坏：原来是少一个提醒，改完变成给错人发提醒。

    ⚠️ 上一版的测试用「同一个 event 再调一次」模拟重试 —— 那是**生产不存在
    的时序**，所以它绿着，而真实缺陷就在它眼皮底下。
    """
    mode, _ = outcome_send
    adapter = _adapter(monkeypatch)
    turn1 = _event(message_id="om_1")
    adapter.note_long_running_turn(turn1.source)

    if mode == "raise":
        adapter.send = AsyncMock(side_effect=RuntimeError("network down"))
    else:
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=False, error="rate limited"))
    await adapter.on_processing_complete(turn1, ProcessingOutcome.SUCCESS)

    # 下一轮：同一 chat / 同一 user 的**短问答**，⛔ 没有发过心跳
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m2"))
    turn2 = _event(message_id="om_2")
    await adapter.on_processing_complete(turn2, ProcessingOutcome.SUCCESS)

    assert _sent_texts(adapter) == [], (
        f"短问答收到了上一轮残留的 ✅ —— 它从没发过心跳:{_sent_texts(adapter)}")


@pytest.mark.asyncio
async def test_marker_is_consumed_exactly_once_on_success(monkeypatch):
    """⛔ 不许弄坏原来对的：成功发出后标记必须消费，⛔ 不能重复发。"""
    adapter = _adapter(monkeypatch)
    event = _event()
    adapter.note_long_running_turn(event.source)

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    assert _sent_texts(adapter) == ["✅"], "成功后标记没被消费，✅ 发了不止一次"
