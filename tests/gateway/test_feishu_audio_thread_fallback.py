"""飞书音频 99992402 回退**必须留在话题里** —— ⛔ 不许降级到群主时间线。

🔴 催生本门的原始案例(⭐ 门必须抓得住它,抓不住 = 出生即空转):
**线程内音频 + 返回码 99992402 + metadata 里没有 ``reply_to_message_id``**
⇒ 上一版最后一步用 ``metadata=None`` 重试,**把 ``thread_id`` 一起丢掉**
⇒ 只属于话题的语音被发到**群主时间线**:①错位回复 ②**扩大内容可见范围**。

这是 ``bd994a9a2``(恢复通用 thread 路由)的**兄弟调用点**,当时没跟上。
"""

import asyncio
import json
import pathlib
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.feishu.adapter import FeishuAdapter


#: 生产入口会 ``os.path.exists`` 把关 ⇒ 用一个真实存在的文件,
#: ⛔ 不 monkeypatch 掉那道校验(那会把入口的前置条件一起绕过)。
#: ⚠️ 落在系统临时目录 —— ⛔ 测试不许往仓库树里丢文件。
_AUDIO_FILE = pathlib.Path(tempfile.gettempdir()) / "hermes_audio_probe.opus"
_AUDIO_FILE.write_bytes(b"OggS\x00" + b"\x00" * 32)


def _ok():
    return SimpleNamespace(code=0, msg="ok",
                           data=SimpleNamespace(message_id="om_new"))


def _err_99992402():
    return SimpleNamespace(code=99992402, msg="thread routing rejected", data=None)


class _Recorder:
    """记录每次 ``_feishu_send_with_retry`` 的实参 —— 判据落在**真实调用序列**上。"""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.outcomes[min(len(self.calls) - 1, len(self.outcomes) - 1)]


def _adapter(recorder, *, last_in_thread=None):
    # ⚠️ ``last_in_thread`` 只是历史签名残留:``_fetch_last_message_in_thread``
    #    已随「⛔ 不发明引用」的裁定一并删除,这里⛔ 不再桩一个不存在的函数
    #    (桩不存在的东西 = 假路径夹具)。
    a = MagicMock(spec=FeishuAdapter)
    a._client = MagicMock()          # spec 不提供实例属性
    a._feishu_send_with_retry = recorder
    a._response_succeeded = lambda r: getattr(r, "code", None) == 0
    a._resolve_outbound_file_routing = lambda *a_, **k_: ("stream", "audio")
    a._finalize_send_result = lambda resp, msg: SimpleNamespace(
        success=getattr(resp, "code", None) == 0, error=None if
        getattr(resp, "code", None) == 0 else msg)
    # 上传那一段不是本门要测的面 —— 只把它的**产物**钉住,⛔ 不绕过入口本身。
    a._get_audio_duration_ms = lambda _p: 0
    a._build_file_upload_body = lambda **k: object()
    a._build_file_upload_request = lambda b: object()
    a._extract_response_field = lambda _r, _f: "file_key_1"

    async def _run_blocking(fn, *args):
        return object()

    a._run_blocking = _run_blocking
    a._build_media_post_payload = FeishuAdapter._build_media_post_payload.__get__(a)
    return a


async def _send(adapter, metadata, tmp_path=None):
    """驱动**生产入口** ``_send_uploaded_file_message``,⛔ 不测判据函数本身。"""
    return await FeishuAdapter._send_uploaded_file_message(
        adapter, chat_id="oc_group", file_path=str(_AUDIO_FILE),
        caption=None, reply_to=None, metadata=metadata,
        outbound_message_type="audio")


# ═══════════ ① 应该改变:原始案例 ═══════════

def test_the_original_case_never_reaches_the_chat_timeline(monkeypatch):
    """⭐ 线程内音频 + 99992402 + **无** reply_to_message_id ⇒ 每一次重试都必须
    仍然带着 ``thread_id``,⛔ 一次都不许出现 ``metadata=None``。"""
    rec = _Recorder([_err_99992402()])          # 每次都失败,把所有重试都逼出来
    a = _adapter(rec, last_in_thread=None)      # 线程里也取不到锚点
    asyncio.run(_send(a, {"thread_id": "omt_secret"}))

    assert len(rec.calls) >= 2, f"回退路径没被驱动(只调了 {len(rec.calls)} 次)"
    leaked = [i for i, c in enumerate(rec.calls)
              if not (c.get("metadata") or {}).get("thread_id")]
    assert not leaked, (
        f"第 {leaked} 次重试丢了 thread_id ⇒ 只属于话题的语音被发到群主时间线,"
        "错位回复 + 扩大内容可见范围"
    )


def test_it_never_invents_a_quote_from_the_thread():
    """🔴 **这条门我第一版钉反了。**

    我原本按 finding 的建议钉「没有显式锚点就去取线程内最后一条」——
    而本仓另有一条**同样成立**的既有契约:
    ``test_audio_99992402_flat_retry_does_not_invent_reply_from_thread``
    ⛔ 不许从线程里捞一条当 quote(与 H④「显式不要引用就不许冒出引用」同族)。

    ⭐ 两条真需求只在**混淆「引用」与「路由」**时才冲突:
    引用只由**显式锚点**决定,路由由 **thread_id** 决定 —— 各管各的。
    ⇒ finding 的**现象**对,它给的**修法**不能照做。
    """
    rec = _Recorder([_err_99992402(), _ok()])
    a = _adapter(rec, last_in_thread="om_should_not_be_used")
    asyncio.run(_send(a, {"thread_id": "omt_1"}))      # 线程内,⛔ 无显式锚点

    assert len(rec.calls) == 2
    assert rec.calls[1]["reply_to"] is None, (
        "凭空发明了一个引用 —— 用户没要求引用,⛔ 不许从线程里捞一条塞进去"
    )
    assert (rec.calls[1]["metadata"] or {}).get("thread_id") == "omt_1", (
        "路由丢了 ⇒ 语音落到群主时间线"
    )


def test_it_fails_rather_than_leaking_when_every_thread_attempt_fails():
    """✅ **收紧**:线程内所有尝试都失败 ⇒ 返回失败,⛔ 不降级到群顶层。"""
    rec = _Recorder([_err_99992402()])
    a = _adapter(rec, last_in_thread=None)
    res = asyncio.run(_send(a, {"thread_id": "omt_1"}))
    assert res.success is False, "线程内发不出去却报了成功"
    assert all((c.get("metadata") or {}).get("thread_id") for c in rec.calls)


# ═══════════ ② 必须保持不变 ═══════════

def test_a_non_threaded_audio_still_falls_back_to_chat_id():
    """🔴 **收紧不许扩到这一格。** 群聊/私聊里本来就没有 thread 的正常语音,
    行为必须与先例**逐字相同** —— 退到 ``chat_id`` 并且**发得出去**。"""
    rec = _Recorder([_err_99992402(), _ok()])
    a = _adapter(rec, last_in_thread=None)
    res = asyncio.run(_send(a, {}))             # ⛔ 无 thread_id

    assert res.success is True, "非线程语音被这条收紧改成了失败 —— 弄坏了原来对的行为"
    assert len(rec.calls) == 2
    assert rec.calls[1]["reply_to"] is None and rec.calls[1]["metadata"] is None, (
        "非线程回退的实参变了 ⇒ 与先例不再逐字相同"
    )


def test_metadata_none_is_also_treated_as_non_threaded():
    """🔴 **必须保持不变**:``metadata=None`` 与「无 thread_id」同一格。"""
    rec = _Recorder([_err_99992402(), _ok()])
    a = _adapter(rec, last_in_thread=None)
    assert asyncio.run(_send(a, None)).success is True


def test_a_first_try_success_does_not_trigger_any_fallback():
    """🔴 **必须保持不变**:首次就成功 ⇒ 一次调用,⛔ 不多发。"""
    rec = _Recorder([_ok()])
    a = _adapter(rec, last_in_thread="om_x")
    res = asyncio.run(_send(a, {"thread_id": "omt_1"}))
    assert res.success is True and len(rec.calls) == 1


def test_a_non_99992402_failure_does_not_enter_the_branch():
    """🔴 **必须保持不变**:别的错误码⛔不许被这条分支接管。"""
    rec = _Recorder([SimpleNamespace(code=230001, msg="other", data=None)])
    a = _adapter(rec, last_in_thread="om_x")
    asyncio.run(_send(a, {"thread_id": "omt_1"}))
    assert len(rec.calls) == 1, "非 99992402 的失败被误当成 thread 路由问题重试了"


def test_an_explicit_anchor_is_preferred_over_fetching():
    """🔴 **必须保持不变**:有显式 ``reply_to_message_id`` 就直接用,⛔ 不多跑一趟 API。"""
    rec = _Recorder([_err_99992402(), _ok()])
    a = _adapter(rec, last_in_thread="om_fetched")
    asyncio.run(_send(a, {"thread_id": "omt_1", "reply_to_message_id": "om_explicit"}))
    assert rec.calls[1]["reply_to"] == "om_explicit"


# ═══════════ ③ bd994a9a2 的通用回退路径必须同轮仍然有效 ═══════════

class TestGeneralThreadRoutingStillHolds:
    """⛔「它还绿着」不算证据 —— 这几条与音频分支**互不覆盖**,
    专门钉住 ``bd994a9a2`` 恢复的那条通用路由仍在。"""

    @staticmethod
    def _raw_adapter():
        a = MagicMock(spec=FeishuAdapter)
        client = MagicMock()
        client.im.v1.message.create = MagicMock(return_value=SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(message_id="m1")))
        a._client = client
        a._build_create_message_body = FeishuAdapter._build_create_message_body
        a._build_create_message_request = FeishuAdapter._build_create_message_request

        async def _passthrough(fn, *args):
            return fn(*args)

        a._run_blocking = _passthrough
        return a, client

    def test_generic_fallback_create_still_targets_the_thread(self):
        a, client = self._raw_adapter()
        asyncio.run(FeishuAdapter._send_raw_message(
            a, chat_id="oc_group", msg_type="text",
            payload=json.dumps({"text": "hi"}),
            reply_to=None, metadata={"thread_id": "omt_1"}))
        req = client.im.v1.message.create.call_args[0][0]
        assert getattr(req, "receive_id_type", None) == "thread_id", (
            "bd994a9a2 恢复的通用 thread 路由失效了"
        )

    def test_generic_non_threaded_create_still_targets_the_chat(self):
        a, client = self._raw_adapter()
        asyncio.run(FeishuAdapter._send_raw_message(
            a, chat_id="oc_group", msg_type="text",
            payload=json.dumps({"text": "hi"}),
            reply_to=None, metadata={}))
        req = client.im.v1.message.create.call_args[0][0]
        assert getattr(req, "receive_id_type", None) == "chat_id"


def test_the_thread_flat_retry_strips_the_quote_anchor_but_keeps_routing():
    """⭐ 门必须**同时**钉两半 —— 少钉一半就会被改回去。

    ①**仍落在话题里**:``metadata`` 必须带 ``thread_id``。
    ②**不带引用锚点**:``reply_to_message_id`` 必须被摘掉 ——
      ``_send_raw_message`` 会「在线程内从 metadata 恢复引用」,原样回传等于
      把刚失败过的引用**重新装上**,大概率再撞同一个 99992402,这条重试就白设了。
    """
    rec = _Recorder([_err_99992402(), _err_99992402(), _ok()])
    a = _adapter(rec)
    md = {"thread_id": "omt_1", "reply_to_message_id": "om_anchor", "notify": True}
    asyncio.run(_send(a, md))

    assert len(rec.calls) == 3, f"三段式没走全(只调了 {len(rec.calls)} 次)"
    assert rec.calls[1]["reply_to"] == "om_anchor", "第二次应当用显式锚点 reply"
    flat = rec.calls[2]["metadata"] or {}
    assert rec.calls[2]["reply_to"] is None
    assert flat.get("thread_id") == "omt_1", "flat 重试丢了路由 ⇒ 发到群主时间线"
    assert "reply_to_message_id" not in flat, (
        "flat 重试仍带着引用锚点 ⇒ 下游会把引用重新装上,大概率再撞 99992402"
    )
    assert flat.get("notify") is True, (
        "🔴 必须保持不变:⛔ 只摘引用锚点这一个键,其余 metadata 原样保留"
    )
    assert md == {"thread_id": "omt_1", "reply_to_message_id": "om_anchor",
                  "notify": True}, "🔴 调用方传进来的 metadata 被就地改了"
