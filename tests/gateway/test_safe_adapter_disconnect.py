"""覆盖有界进程退出 cleanup 与严格的未发布 adapter cleanup。"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """构造只含 cleanup helper 所需状态的 GatewayRunner 外壳。"""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_safe_disconnect_times_out_and_continues(bare_runner, monkeypatch, caplog):
    """A wedged adapter disconnect must not block gateway shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.001")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(0.2)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)

    adapter.disconnect.assert_awaited_once()
    assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text


@pytest.mark.asyncio
async def test_safe_disconnect_detaches_cancellation_swallowing_disconnect(
    bare_runner, monkeypatch, caplog
):
    """A disconnect that catches cancellation cannot block fatal recovery.

    ``asyncio.wait_for`` cancels its child at the deadline but then waits for
    it to finish.  A half-closed transport can catch that cancellation while
    unwinding, so the runner must detach the old close task and continue to the
    reconnect queue instead of waiting indefinitely.
    """
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    adapter.disconnect = AsyncMock(side_effect=swallow_cancellation)
    operation = asyncio.create_task(
        bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)
    )
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text
    finally:
        # The implementation must detach rather than abandon the old task.
        # Release it here so this test leaves no cancellation-swallowing task
        # behind when it runs against the pre-fix implementation.
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_strict_partial_cleanup_waits_for_cancellation_swallowing_worker(
    bare_runner, monkeypatch
):
    """严格 cleanup 超时后必须等待真实 worker，且期间保留 retry owner。"""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    entered = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_disconnect():
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=stubborn_disconnect)
    operation = asyncio.create_task(
        bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    )
    await entered.wait()
    await cancellation_seen.wait()
    assert not operation.done()
    assert bare_runner._partial_adapter_cleanup_retry[("", Platform.FEISHU)] is adapter

    release.set()
    await operation
    assert ("", Platform.FEISHU) not in bare_runner._partial_adapter_cleanup_retry


@pytest.mark.asyncio
async def test_partial_cleanup_failure_keeps_owner_and_retry_is_exact(bare_runner):
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(
        side_effect=[RuntimeError("old poller alive"), None]
    )

    with pytest.raises(RuntimeError, match="old poller alive"):
        await bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    assert bare_runner._partial_adapter_cleanup_retry[("", Platform.FEISHU)] is adapter

    await bare_runner._retry_unpublished_adapter_cleanup("", Platform.FEISHU)
    assert adapter.disconnect.await_count == 2
    assert ("", Platform.FEISHU) not in bare_runner._partial_adapter_cleanup_retry


@pytest.mark.asyncio
async def test_partial_cleanup_concurrent_callers_join_exact_task(
    bare_runner, monkeypatch
):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_disconnect():
        entered.set()
        await release.wait()

    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=blocked_disconnect)
    first = asyncio.create_task(
        bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    )
    await entered.wait()
    second = asyncio.create_task(
        bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    )
    second_joined = asyncio.Event()
    asyncio.get_running_loop().call_soon(second_joined.set)
    await second_joined.wait()
    assert adapter.disconnect.await_count == 1
    assert not first.done()
    assert not second.done()

    release.set()
    await asyncio.gather(first, second)
    assert adapter.disconnect.await_count == 1


@pytest.mark.asyncio
async def test_partial_cleanup_follower_cancel_does_not_cancel_owner(bare_runner):
    """follower 取消只能离开等待，不能取消第一个 cleanup owner。"""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_disconnect():
        entered.set()
        await release.wait()

    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=blocked_disconnect)
    first = asyncio.create_task(
        bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    )
    await entered.wait()
    second = asyncio.create_task(
        bare_runner._cleanup_unpublished_adapter(adapter, Platform.FEISHU)
    )
    follower_joined = asyncio.Event()
    asyncio.get_running_loop().call_soon(follower_joined.set)
    await follower_joined.wait()
    cancellation_delivered = asyncio.Event()
    second.cancel()
    asyncio.get_running_loop().call_soon(cancellation_delivered.set)
    await cancellation_delivered.wait()
    with pytest.raises(asyncio.CancelledError):
        await second

    assert not first.done()
    assert adapter.disconnect.await_count == 1
    release.set()
    await first
    assert ("", Platform.FEISHU) not in bare_runner._partial_adapter_cleanup_retry
