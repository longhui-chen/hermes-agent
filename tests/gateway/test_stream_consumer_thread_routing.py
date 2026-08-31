"""Regression tests for stream consumer thread/topic routing fix.

Verifies that GatewayStreamConsumer correctly passes reply_to on the first
message send, ensuring messages land in the correct topic/thread instead of
the main group chat.

Covers: #6969, #9916, #7355
"""
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace
import asyncio
import threading

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
)
from gateway.platforms.base import (
    FeishuQuoteLease,
    SendResult,
    _consume_feishu_quote,
    _feishu_quote_metadata,
    _reserve_feishu_quote_metadata,
)


def _make_adapter(send_result=None, edit_result=None, max_length=4096):
    adapter = MagicMock()
    adapter.send = AsyncMock(
        return_value=send_result or SimpleNamespace(success=True, message_id="msg_1")
    )
    adapter.edit_message = AsyncMock(
        return_value=edit_result or SimpleNamespace(success=True)
    )
    adapter.MAX_MESSAGE_LENGTH = max_length
    return adapter


class _SignalingCondition(threading.Condition):
    """在 reservation 等待点发信号，避免并发测试依赖调度时机。"""

    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()
        self.returned = threading.Event()
        self.allow_return = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        result = super().wait(timeout)
        self.returned.set()
        assert self.allow_return.wait(timeout=10)
        return result


class TestInitialReplyToId:
    """Verify initial_reply_to_id is passed as reply_to on first send."""

    @pytest.mark.asyncio
    async def test_first_send_uses_initial_reply_to_id(self):
        """When initial_reply_to_id is set, first adapter.send() should
        include reply_to=initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )
        await consumer._send_or_edit("Hello world")

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_456", (
            "First send should pass initial_reply_to_id as reply_to"
        )
        assert call_kwargs["chat_id"] == "chat_123"


    @pytest.mark.asyncio
    async def test_subsequent_edits_ignore_initial_reply_to_id(self):
        """After first send, edits should use message_id, not initial_reply_to_id."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_456",
        )

        # First send
        await consumer._send_or_edit("Hello world")
        assert adapter.send.call_count == 1

        # Second call should edit, not send
        await consumer._send_or_edit("Hello world updated")
        assert adapter.send.call_count == 1, "Should edit, not send again"
        adapter.edit_message.assert_called_once()
        edit_kwargs = adapter.edit_message.call_args[1]
        assert edit_kwargs["message_id"] == "msg_1"
        assert edit_kwargs["chat_id"] == "chat_123"


class TestOverflowFirstMessage:
    """Verify thread routing is preserved when the first message overflows."""

    @pytest.mark.asyncio
    async def test_overflow_first_send_uses_initial_reply_to_id(self):
        """When first message exceeds platform limit and is split into chunks,
        each chunk should be threaded to initial_reply_to_id, not None."""
        adapter = _make_adapter(max_length=10)
        adapter.truncate_message = MagicMock(
            return_value=["chunk_1", "chunk_2"]
        )
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            metadata={"thread_id": "omt_topic123"},
            initial_reply_to_id="om_user_msg_789",
        )

        # Inject oversized accumulated text to trigger overflow path
        consumer._accumulated = "A" * 100
        consumer._current_edit_interval = 999
        await consumer._send_new_chunk("chunk_1", consumer._message_id or consumer._initial_reply_to_id)

        adapter.send.assert_called_once()
        call_kwargs = adapter.send.call_args[1]
        assert call_kwargs["reply_to"] == "om_user_msg_789", (
            "Overflow first chunk should use initial_reply_to_id"
        )

    @pytest.mark.asyncio
    async def test_feishu_success_without_message_id_consumes_quote(self):
        adapter = _make_adapter(send_result=SimpleNamespace(success=True, message_id=None))
        adapter.platform = "feishu"
        consumer = GatewayStreamConsumer(
            adapter,
            "chat_123",
            initial_reply_to_id="om_user_msg_789",
        )

        result = await consumer._send_new_chunk("chunk_1", "om_user_msg_789")

        assert result == ""
        assert consumer._feishu_quote_available is False


@pytest.mark.asyncio
async def test_feishu_turn_quote_is_shared_across_commentary_and_stream_segments():
    adapter = _make_adapter()
    adapter.platform = "feishu"
    lease = FeishuQuoteLease("om_user")
    metadata = {
        "thread_id": "omt_topic",
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        metadata=metadata,
        initial_reply_to_id="om_user",
    )

    assert await consumer._send_commentary("先说明") is True
    await consumer._send_new_chunk("再回答", "om_user")

    first, second = adapter.send.await_args_list
    assert first.kwargs["metadata"]["reply_to_message_id"] == "om_user"
    assert "reply_to_message_id" not in second.kwargs["metadata"]
    assert second.kwargs["reply_to"] is None


@pytest.mark.asyncio
async def test_feishu_failed_commentary_keeps_turn_quote_for_next_success():
    adapter = _make_adapter()
    adapter.platform = "feishu"
    adapter.send = AsyncMock(side_effect=[
        SimpleNamespace(success=False, message_id=None),
        SimpleNamespace(success=True, message_id="msg-2"),
    ])
    lease = FeishuQuoteLease("om_user")
    metadata = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        metadata=metadata,
        initial_reply_to_id="om_user",
    )

    assert await consumer._send_commentary("发送失败") is False
    await consumer._send_new_chunk("随后成功", "om_user")

    assert all(
        call.kwargs["metadata"]["reply_to_message_id"] == "om_user"
        for call in adapter.send.await_args_list
    )


def test_feishu_quote_reservation_blocks_concurrent_delivery_then_releases():
    lease = FeishuQuoteLease("om_user")
    lease._condition = _SignalingCondition()
    shared = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }

    reserved = _feishu_quote_metadata(shared)
    assert reserved["reply_to_message_id"] == "om_user"

    concurrent = {}
    returned = threading.Event()

    def _reserve_concurrent():
        concurrent["metadata"] = _feishu_quote_metadata(shared)
        returned.set()

    thread = threading.Thread(target=_reserve_concurrent, daemon=True)
    thread.start()
    assert lease._condition.waiting.wait(timeout=10)
    try:
        assert not lease._condition.returned.is_set()
        assert not returned.is_set()
    finally:
        lease._condition.allow_return.set()
        _consume_feishu_quote(reserved, SendResult(success=False, error="temporary"))
        thread.join(timeout=10)
    assert not thread.is_alive()
    retry = concurrent["metadata"]
    assert retry["reply_to_message_id"] == "om_user"
    _consume_feishu_quote(retry, SendResult(success=True, message_id="quoted"))
    assert "reply_to_message_id" not in _feishu_quote_metadata(shared)


@pytest.mark.asyncio
async def test_feishu_inflight_reservation_orders_visible_deliveries():
    lease = FeishuQuoteLease("om_user")
    lease._condition = _SignalingCondition()
    shared = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def _first_send(**_kwargs):
        first_entered.set()
        await release_first.wait()
        return SendResult(success=True, message_id="first")

    first_adapter = _make_adapter()
    first_adapter.platform = "feishu"
    first_adapter.send = AsyncMock(side_effect=_first_send)
    second_adapter = _make_adapter()
    second_adapter.platform = "feishu"
    first = GatewayStreamConsumer(
        first_adapter,
        "chat_123",
        metadata=shared,
        initial_reply_to_id="om_user",
    )
    second = GatewayStreamConsumer(
        second_adapter,
        "chat_123",
        metadata=shared,
        initial_reply_to_id="om_user",
    )

    first_task = asyncio.create_task(first._send_commentary("先到达"))
    await asyncio.wait_for(first_entered.wait(), timeout=10)
    second_task = asyncio.create_task(second._send_commentary("后到达"))
    assert await asyncio.to_thread(lease._condition.waiting.wait, 10)
    try:
        assert not lease._condition.returned.is_set()
        second_adapter.send.assert_not_awaited()
    finally:
        lease._condition.allow_return.set()
        release_first.set()
        results = await asyncio.gather(first_task, second_task)

    assert results == [True, True]
    assert first_adapter.send.await_args.kwargs["metadata"]["reply_to_message_id"] == "om_user"
    assert "reply_to_message_id" not in second_adapter.send.await_args.kwargs["metadata"]


@pytest.mark.asyncio
async def test_feishu_cancelled_waiter_does_not_strand_reservation():
    lease = FeishuQuoteLease("om_user")
    lease._condition = _SignalingCondition()
    shared = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    owner = _feishu_quote_metadata(shared)
    waiter = asyncio.create_task(_reserve_feishu_quote_metadata(shared))
    assert await asyncio.to_thread(lease._condition.waiting.wait, 10)

    waiter.cancel()
    lease._condition.allow_return.set()
    _consume_feishu_quote(owner, SendResult(success=False, error="temporary"))
    with pytest.raises(asyncio.CancelledError):
        await waiter

    retry = _feishu_quote_metadata(shared)
    assert retry["reply_to_message_id"] == "om_user"
    _consume_feishu_quote(retry, SendResult(success=True, message_id="quoted"))


@pytest.mark.asyncio
async def test_feishu_first_send_exception_releases_quote_for_next_success():
    lease = FeishuQuoteLease("om_user")
    metadata = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    adapter = _make_adapter()
    adapter.platform = "feishu"
    adapter.send = AsyncMock(side_effect=RuntimeError("transport failed"))
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        metadata=metadata,
        initial_reply_to_id="om_user",
    )

    assert await consumer._send_or_edit("首次失败") is False
    failed_metadata = adapter.send.await_args.kwargs["metadata"]
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="next"))
    await consumer._send_new_chunk("随后成功", "om_user")

    assert failed_metadata["reply_to_message_id"] == "om_user"
    assert adapter.send.await_args.kwargs["metadata"]["reply_to_message_id"] == "om_user"


@pytest.mark.asyncio
async def test_feishu_fallback_exception_releases_quote_and_stays_visible():
    lease = FeishuQuoteLease("om_user")
    metadata = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    adapter = _make_adapter()
    adapter.platform = "feishu"
    adapter.send = AsyncMock(side_effect=RuntimeError("fallback failed"))
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        metadata=metadata,
        initial_reply_to_id="om_user",
    )

    with pytest.raises(RuntimeError, match="fallback failed"):
        await consumer._send_fallback_final("回退内容")
    failed_metadata = adapter.send.await_args.kwargs["metadata"]
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="next"))
    await consumer._send_new_chunk("随后成功", "om_user")

    assert failed_metadata["reply_to_message_id"] == "om_user"
    assert adapter.send.await_args.kwargs["metadata"]["reply_to_message_id"] == "om_user"


def test_feishu_quote_reservation_owner_commits_success():
    lease = FeishuQuoteLease("om_user")
    lease._condition = _SignalingCondition()
    shared = {
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }

    reserved = _feishu_quote_metadata(shared)
    concurrent = {}

    def _reserve_concurrent():
        concurrent["metadata"] = _feishu_quote_metadata(shared)

    thread = threading.Thread(target=_reserve_concurrent, daemon=True)
    thread.start()
    assert lease._condition.waiting.wait(timeout=10)
    try:
        assert not lease._condition.returned.is_set()
    finally:
        lease._condition.allow_return.set()
        _consume_feishu_quote(reserved, SendResult(success=True, message_id="quoted"))
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert "reply_to_message_id" not in concurrent["metadata"]
    assert "reply_to_message_id" not in _feishu_quote_metadata(shared)


@pytest.mark.asyncio
async def test_feishu_segment_reset_does_not_restore_consumed_quote():
    adapter = _make_adapter()
    adapter.platform = "feishu"
    lease = FeishuQuoteLease("om_user")
    metadata = {
        "thread_id": "omt_topic",
        "reply_to_message_id": "om_user",
        "_feishu_quote_lease": lease,
    }
    consumer = GatewayStreamConsumer(
        adapter,
        "chat_123",
        metadata=metadata,
        initial_reply_to_id="om_user",
    )

    await consumer._send_new_chunk("首段", "om_user")
    consumer._reset_segment_state()
    await consumer._send_new_chunk("工具后的第二段", "om_user")

    first, second = adapter.send.await_args_list
    assert first.kwargs["metadata"]["reply_to_message_id"] == "om_user"
    assert "reply_to_message_id" not in second.kwargs["metadata"]
    assert second.kwargs["reply_to"] is None


class TestFeishuFallbackThreadRouting:
    """reply 锚点失效回退 create 时,话题里的回答必须**留在话题里**。

    🔴 **这个类的上一版把 bug 钉成了契约。** 它断言 ``receive_id='oc_main_chat'``
    —— 而 merge-base(上游 main)里这段代码本来就是
    ``receive_id=_thread_id`` + ``receive_id_type="thread_id"``,还带着原话注释
    「so the message lands in the topic instead of the main chat」。
    那条断言是在本分支 ``8e0a8dd2b3`` **误删**该路由之后写的,钉的是
    **当时的实现行为**,不是需求 ⇒ 它会**主动阻止修复**(H⑨ 一恢复它就红)。

    ⭐ 判据:「它钉的是需求,还是代码恰好这么跑的?」
    这里是后者 ⇒ 改测试。若是前者,正解是换驱动方式、契约一字不动。

    ⚠️ 上一版的 docstring 说的是「routes to topic on fallback」,和它自己的断言
    正好相反 —— **同一个类里两句话打架**就是钉错了的信号。
    """

    @staticmethod
    def _adapter_with_recording_client():
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = MagicMock(spec=FeishuAdapter)
        mock_client = MagicMock()
        mock_client.im.v1.message.create = MagicMock(
            return_value=SimpleNamespace(
                success=lambda: True,
                data=SimpleNamespace(message_id="new_msg_1"),
            )
        )
        adapter._client = mock_client
        adapter._build_create_message_body = FeishuAdapter._build_create_message_body
        adapter._build_create_message_request = FeishuAdapter._build_create_message_request

        # _send_raw_message 把阻塞 SDK 调用丢给 _run_blocking;spec MagicMock 会
        # 自动 mock 掉它并吞掉真实调用 ⇒ 接一个直通。
        async def _run_blocking_passthrough(func, *args):
            return func(*args)

        adapter._run_blocking = _run_blocking_passthrough
        return adapter, mock_client

    @staticmethod
    def _receive_id_of(call_args):
        body = getattr(call_args, "body", None) or getattr(call_args, "request_body", None)
        assert body is not None, "request has neither .body nor .request_body"
        receive_id = getattr(body, "receive_id", None)
        if receive_id is None and isinstance(body, str):
            import json as _json

            receive_id = _json.loads(body).get("receive_id")
        return receive_id

    @pytest.mark.asyncio
    async def test_fallback_create_lands_in_the_topic(self):
        """✅ **应该改变的行为**(H⑨ 恢复的正是这条)。"""
        import json

        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, mock_client = self._adapter_with_recording_client()
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={"thread_id": "omt_topic_abc"},
        )

        mock_client.im.v1.message.create.assert_called_once()
        call_args = mock_client.im.v1.message.create.call_args[0][0]
        assert self._receive_id_of(call_args) == "omt_topic_abc", (
            "回退 create 落到了群主时间线 ⇒ **错位回复 + 把话题内容扩散给整个群**"
        )
        assert getattr(call_args, "receive_id_type", None) == "thread_id", (
            "receive_id_type 必须是飞书的 thread_id(上游 merge-base 原样如此)"
        )

    @pytest.mark.asyncio
    async def test_non_threaded_create_still_targets_the_chat(self):
        """🔴 **必须保持不变**:没有 thread_id 的普通消息,逐字还是 chat_id。"""
        import json

        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter, mock_client = self._adapter_with_recording_client()
        await FeishuAdapter._send_raw_message(
            adapter,
            chat_id="oc_main_chat",
            msg_type="text",
            payload=json.dumps({"text": "hello"}),
            reply_to=None,
            metadata={},
        )

        call_args = mock_client.im.v1.message.create.call_args[0][0]
        assert self._receive_id_of(call_args) == "oc_main_chat"
        assert getattr(call_args, "receive_id_type", None) == "chat_id"

    @pytest.mark.asyncio
    async def test_user_prefixed_targets_are_untouched(self):
        """🔴 **必须保持不变**:``feishu_user_id:`` / ``ou_`` 两条前缀分派。"""
        import json

        from plugins.platforms.feishu.adapter import FeishuAdapter

        for chat_id, want_id, want_type in (
            ("feishu_user_id:u123", "u123", "user_id"),
            ("ou_abc", "ou_abc", "open_id"),
        ):
            adapter, mock_client = self._adapter_with_recording_client()
            await FeishuAdapter._send_raw_message(
                adapter,
                chat_id=chat_id,
                msg_type="text",
                payload=json.dumps({"text": "hello"}),
                reply_to=None,
                metadata={},
            )
            call_args = mock_client.im.v1.message.create.call_args[0][0]
            assert self._receive_id_of(call_args) == want_id
            assert getattr(call_args, "receive_id_type", None) == want_type
