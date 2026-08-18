"""飞书引用租约:等待必须有独立硬期限,但**只在真的等不到时**降级。

🔴 缺陷(机器人第九轮,自带新证据):``_INTERIM_SEND_TIMEOUT`` 只放开了 agent
worker;租约的释放走 ``consume()``,而 ``consume()`` 由发送任务的完成回调驱动。
飞书 SDK 内部永久卡死 ⇒ 回调永不执行 ⇒ 最终正文在 ``metadata()`` 里**无限等**
⇒ **用户的回答永远发不出去**。

⭐ 这道门同时钉两个方向(⛔ 缺一不可):
  · **应该改变**:持有者永不回来 ⇒ 期限到了降级为无引用正文,交付不被卡住。
  · **必须保持不变**:无竞争 / 及时释放 / 失败释放 / 重入 / 取消 / 无租约
    —— 六条既有路径一个字都不许变。⭐ 降级的作用域必须**刚好等于**卡死的作用域。
"""

import asyncio
import threading
import time

import pytest

from gateway.platforms.base import (
    FeishuQuoteLease,
    _FEISHU_QUOTE_LEASE_KEY,
    _FEISHU_QUOTE_RESERVATION_KEY,
    _reserve_feishu_quote_metadata,
)
from gateway.platforms import base as base_mod


@pytest.fixture
def fast_deadline(monkeypatch):
    """把 30s 期限压到 0.15s —— ⛔ 只压时长,不改任何判定分支。"""
    monkeypatch.setattr(base_mod, "_FEISHU_QUOTE_WAIT_TIMEOUT_SECONDS", 0.15)
    return 0.15


def _meta(lease):
    return {_FEISHU_QUOTE_LEASE_KEY: lease, "thread_id": "omt_1"}


# ═══════════════ ① 应该改变:持有者永不回来 ═══════════════

def test_a_holder_that_never_returns_does_not_block_delivery(fast_deadline):
    lease = FeishuQuoteLease("om_anchor")
    holder = lease.metadata(_meta(lease))
    assert holder["reply_to_message_id"] == "om_anchor", "前置:持有者应拿到引用"
    reservation = holder[_FEISHU_QUOTE_RESERVATION_KEY]

    started = time.monotonic()
    late = lease.metadata(_meta(lease))          # 持有者从不 consume
    elapsed = time.monotonic() - started

    assert elapsed < 2.0, f"仍在无限等待({elapsed:.1f}s)⇒ 最终正文永远发不出去"
    assert elapsed >= fast_deadline * 0.5, "根本没等 ⇒ 期限没生效,可能提前放弃"
    assert "reply_to_message_id" not in late, "降级后不该带引用"
    assert late["thread_id"] == "omt_1", "🔴 路由上下文必须保留 —— 降级 ≠ 丢线程"
    assert lease._reservation is reservation, (
        "🔴 抢占/清掉了迟到 interim 的 reservation ⇒ 它若最终发出会造成**重复引用**"
    )


def test_the_deadline_outlasts_the_interim_send_bound():
    """⛔ 期限不许拍脑袋:必须 **> interim 上限**,否则会在 interim 还在正常飞
    的时候提前放弃,把本该带引用的正文降级成无引用。"""
    from gateway.run import _INTERIM_SEND_TIMEOUT

    assert base_mod._FEISHU_QUOTE_WAIT_TIMEOUT_SECONDS > _INTERIM_SEND_TIMEOUT, (
        f"{base_mod._FEISHU_QUOTE_WAIT_TIMEOUT_SECONDS} 不大于 interim 上限 "
        f"{_INTERIM_SEND_TIMEOUT} ⇒ 会误判正常在飞的 interim"
    )
    assert (
        base_mod._FEISHU_QUOTE_WAIT_TIMEOUT_SECONDS
        == base_mod._POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS
    ), "取值应沿用本文件既有的投递路径量纲,⛔ 不另发明一个数"


# ═══════════════ ② 必须保持不变(六条) ═══════════════

def test_uncontended_still_takes_the_quote_immediately(fast_deadline):
    lease = FeishuQuoteLease("om_anchor")
    started = time.monotonic()
    out = lease.metadata(_meta(lease))
    assert time.monotonic() - started < 0.05, "无竞争路径不该产生任何等待"
    assert out["reply_to_message_id"] == "om_anchor"


def test_a_holder_that_succeeds_in_time_still_burns_the_quote(fast_deadline):
    """既有语义:引用用掉后,后续正文是 flat —— ⛔ 不许被降级路径改写。"""
    lease = FeishuQuoteLease("om_anchor")
    holder = lease.metadata(_meta(lease))

    def _release():
        time.sleep(0.02)
        lease.consume(holder, True)

    t = threading.Thread(target=_release); t.start()
    out = lease.metadata(_meta(lease))
    t.join()

    assert "reply_to_message_id" not in out, "引用已被成功发送用掉 ⇒ 后续应 flat"
    assert lease._reservation is None, "成功后 reservation 应已释放"


def test_a_holder_that_fails_in_time_hands_the_quote_over(fast_deadline):
    """既有语义:持有者失败 ⇒ 引用**交还**给下一位,⛔ 不许退化成 flat。"""
    lease = FeishuQuoteLease("om_anchor")
    holder = lease.metadata(_meta(lease))

    def _release():
        time.sleep(0.02)
        lease.consume(holder, False)

    t = threading.Thread(target=_release); t.start()
    out = lease.metadata(_meta(lease))
    t.join()

    assert out["reply_to_message_id"] == "om_anchor", (
        "持有者失败后引用没交还 ⇒ 用户看到的回答丢了引用锚点"
    )


def test_reentrant_holder_returns_immediately_with_the_quote(fast_deadline):
    lease = FeishuQuoteLease("om_anchor")
    holder = lease.metadata(_meta(lease))
    again = lease.metadata(dict(holder))
    assert again["reply_to_message_id"] == "om_anchor"
    assert again[_FEISHU_QUOTE_RESERVATION_KEY] is holder[_FEISHU_QUOTE_RESERVATION_KEY]


def test_cancellation_path_is_untouched(fast_deadline):
    """取消仍走**自己**的分支:摘引用 + 摘 reservation,⛔ 不走降级分支。"""
    lease = FeishuQuoteLease("om_anchor")
    lease.metadata(_meta(lease))                   # 先让 reservation 被占住
    cancelled = threading.Event()
    cancelled.set()
    out = lease.metadata(_meta(lease), cancelled)
    assert "reply_to_message_id" not in out
    assert _FEISHU_QUOTE_RESERVATION_KEY not in out


def test_metadata_without_a_lease_is_returned_verbatim(fast_deadline):
    plain = {"thread_id": "omt_1", "reply_to_message_id": "om_x"}
    assert asyncio.run(_reserve_feishu_quote_metadata(dict(plain))) == plain
    assert asyncio.run(_reserve_feishu_quote_metadata(None)) is None
