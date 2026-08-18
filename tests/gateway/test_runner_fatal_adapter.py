import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _FatalAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error(
            "telegram_token_lock",
            "Another local Hermes gateway is already using this Telegram bot token.",
            retryable=False,
        )
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _RuntimeRetryableAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


class _ReplacementDeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="token", typing_indicator=False),
            Platform.DISCORD,
        )
        self.sent: list[str] = []
        self.connected = True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self.connected = False

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if not self.connected:
            return SendResult(success=False, error="Not connected")
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_runner_queues_retryable_runtime_fatal_for_reconnection(monkeypatch, tmp_path):
    """Retryable runtime fatal errors queue the platform for reconnection
    AND keep the gateway alive — the background reconnect watcher recovers
    the platform when the underlying issue clears.  (Previously this
    exited-with-failure to trigger a systemd restart; that converted
    transient failures into infinite restart loops.)
    """
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error(
        "whatsapp_bridge_exited",
        "WhatsApp bridge process exited unexpectedly (code 1).",
        retryable=True,
    )

    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    runner.stop = AsyncMock()

    await runner._handle_adapter_fatal_error(adapter)

    # Gateway stays alive — watcher will retry in background
    runner.stop.assert_not_awaited()
    assert runner._exit_with_failure is False
    assert Platform.WHATSAPP in runner._failed_platforms
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
async def test_retryable_fatal_queues_reconnect_after_cancellation_swallowing_disconnect(
    monkeypatch, tmp_path
):
    """旧 adapter 真正退出前，不得删除 owner 或启动 replacement。"""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters

    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                continue
        finished.set()

    monkeypatch.setattr(adapter, "disconnect", swallow_cancellation)
    operation = asyncio.create_task(runner._handle_adapter_fatal_error(adapter))
    await started.wait()
    try:
        await cancellation_seen.wait()
        assert not operation.done()
        assert runner.adapters[Platform.WHATSAPP] is adapter
        assert Platform.WHATSAPP not in runner._failed_platforms
    finally:
        release.set()
        await asyncio.wait_for(operation, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)
    assert runner.adapters == {}
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
async def test_retryable_fatal_disconnect_failure_keeps_primary_owner(tmp_path):
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("transport_stale", "transport stale", retryable=True)
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("old listener alive"))
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters

    with pytest.raises(RuntimeError, match="old listener alive"):
        await runner._handle_adapter_fatal_error(adapter)

    assert runner.adapters[Platform.WHATSAPP] is adapter
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("startup_phase", [False, True], ids=["running", "startup"])
async def test_nonretryable_fatal_retries_cleanup_without_reconnecting(
    tmp_path, startup_phase
):
    """业务不可重连不等于 cleanup 不可重试；旧 owner 必须单独回收。"""
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    runner._running = not startup_phase
    runner._startup_restore_in_progress = startup_phase
    adapter = _RuntimeRetryableAdapter()
    adapter._set_fatal_error("auth_failed", "credentials rejected", retryable=False)
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.delivery_router.adapters = runner.adapters
    retry_entered = asyncio.Event()
    release_retry = asyncio.Event()
    calls = 0

    async def disconnect():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("old listener alive")
        retry_entered.set()
        await release_retry.wait()

    adapter.disconnect = disconnect
    try:
        with pytest.raises(RuntimeError, match="old listener alive"):
            await runner._handle_adapter_fatal_error(adapter)
        key = ("", Platform.WHATSAPP)
        assert key in runner._published_adapter_cleanup_tasks
        task = runner._published_adapter_cleanup_tasks[key]
        await retry_entered.wait()
        assert runner.adapters[Platform.WHATSAPP] is adapter
        assert Platform.WHATSAPP not in runner._failed_platforms
        assert not task.done()

        release_retry.set()
        await task
        assert runner.adapters == {}
        assert Platform.WHATSAPP not in runner._failed_platforms
        assert ("", Platform.WHATSAPP) not in runner._published_adapter_cleanup_retry
    finally:
        runner._running = False
        release_retry.set()
        tasks = list(getattr(runner, "_published_adapter_cleanup_tasks", {}).values())
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("connect_mode", ["false", "raise"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_primary_reconnect_terminal_error_only_retries_cleanup(
    monkeypatch, tmp_path, connect_mode, cleanup_fails
):
    """primary watcher 遇到 terminal 结果后只能 cleanup，不能再 connect。"""
    import gateway.run as gateway_run

    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    runner._running = True
    terminal = _RuntimeRetryableAdapter()
    terminal._set_fatal_error("auth_failed", "credentials rejected", retryable=False)
    connect_calls = 0
    cleanup_calls = 0

    def create_adapter(_platform, _config):
        return terminal

    async def connect(_adapter, _platform, *, is_reconnect=False):
        nonlocal connect_calls
        connect_calls += 1
        if connect_calls == 2:
            runner._running = False
        if connect_mode == "raise":
            raise RuntimeError("terminal connect error")
        return False

    async def disconnect():
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_fails and cleanup_calls == 1:
            raise RuntimeError("terminal cleanup failed")
        terminal._mark_disconnected()

    async def advance(_seconds):
        if not runner._failed_platforms:
            runner._running = False

    terminal.disconnect = disconnect
    runner._failed_platforms = {
        Platform.WHATSAPP: {
            "config": config.platforms[Platform.WHATSAPP],
            "attempts": 0,
            "next_retry": 0,
        }
    }
    monkeypatch.setattr(runner, "_create_adapter", create_adapter)
    monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)
    monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda _attempt: 0)
    monkeypatch.setattr(gateway_run.asyncio, "sleep", advance)

    await runner._platform_reconnect_watcher()

    assert connect_calls == 1
    assert cleanup_calls == (2 if cleanup_fails else 1)
    assert runner._failed_platforms == {}
    assert runner.adapters == {}


@pytest.mark.asyncio
async def test_primary_fatal_retry_cleans_old_owner_before_replacement(
    monkeypatch, tmp_path
):
    config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True, token="token")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    stale = _RuntimeRetryableAdapter()
    stale._set_fatal_error("transport_stale", "transport stale", retryable=True)
    stale.disconnect = AsyncMock(
        side_effect=[RuntimeError("old listener alive"), None]
    )
    runner.adapters = {Platform.WHATSAPP: stale}
    runner.delivery_router.adapters = runner.adapters

    with pytest.raises(RuntimeError, match="old listener alive"):
        await runner._handle_adapter_fatal_error(stale)
    assert runner._failed_platforms[Platform.WHATSAPP]["attempts"] == 0

    replacement = _RuntimeRetryableAdapter()
    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: replacement)

    async def connect(*_args, **_kwargs):
        runner._running = False
        return True

    monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)
    monkeypatch.setattr("gateway.run.asyncio.sleep", AsyncMock())
    runner._running = True
    await runner._platform_reconnect_watcher()

    assert stale.disconnect.await_count == 2
    assert runner.adapters[Platform.WHATSAPP] is replacement
    assert Platform.WHATSAPP not in runner._failed_platforms
