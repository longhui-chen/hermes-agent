"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
import logging
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


class _FakeAdapter:
    def __init__(self, token=None, config=None):
        self.token = token
        self.config = config


class _DirectProfileAdapter:
    platform = Platform.TELEGRAM

    def set_message_handler(self, handler):
        self.message_handler = handler

    def set_fatal_error_handler(self, handler):
        self.fatal_error_handler = handler

    def set_session_store(self, store):
        self.session_store = store

    def set_busy_session_handler(self, handler):
        self.busy_session_handler = handler

    def set_topic_recovery_fn(self, handler):
        self.topic_recovery_fn = handler

    def set_authorization_check(self, handler):
        self.authorization_check = handler


def _multiplex_profile_runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._profile_adapters = {}
    runner.session_store = object()
    runner._handle_active_session_busy_message = object()
    runner._recover_telegram_topic_thread_id = object()
    runner._busy_text_mode = "queue"
    runner._make_adapter_auth_check = lambda platform, profile_name=None: object()
    return runner


class TestCredentialFingerprint:
    def test_none_without_token(self):
        assert GatewayRunner._adapter_credential_fingerprint(_FakeAdapter()) is None


    def test_reads_photon_project_secret(self):
        class _PhotonAdapter:
            def __init__(self, secret):
                self._project_secret = secret

        fp1 = GatewayRunner._adapter_credential_fingerprint(
            _PhotonAdapter("shared-project-secret")
        )
        fp2 = GatewayRunner._adapter_credential_fingerprint(
            _PhotonAdapter("shared-project-secret")
        )

        assert fp1 == fp2
        assert fp1 is not None
        assert "shared-project-secret" not in fp1


    def test_reads_config_token(self):
        """Adapters like Discord store token on `config`, not on self.

        Without the config-token fallback, every Discord adapter in a
        multiplexed gateway returns None here and the same-token conflict
        check is silently skipped — N adapters start polling the same bot
        token and race on every inbound message.
        """
        class _Config:
            token = "discord-bot-token"
        class _ConfigBackedAdapter:
            config = _Config()
        fp = GatewayRunner._adapter_credential_fingerprint(_ConfigBackedAdapter())
        assert fp is not None
        assert "discord-bot-token" not in fp
        assert len(fp) == 16

    def test_distinct_config_tokens_distinct_fp(self):
        class _CfgA:
            token = "tok-A"
        class _CfgB:
            token = "tok-B"
        class _A:
            config = _CfgA()
        class _B:
            config = _CfgB()
        a = GatewayRunner._adapter_credential_fingerprint(_A())
        b = GatewayRunner._adapter_credential_fingerprint(_B())
        assert a is not None and b is not None
        assert a != b


@pytest.mark.asyncio
async def test_unloading_one_profile_does_not_cancel_shared_startup_caller(
    monkeypatch, tmp_path
):
    """⚠️ 夹具值从 0 改成 5.0:``0`` 是我在**加硬期限之前**写的
    「让等待变平凡」的值;期限落地后 ``0`` 表示**立刻超时**(fail-fast 语义,
    由下面 ``test_zero_timeout_means_fail_fast`` 单独钉住)。
    本用例钉的是**取消与属主**,与超时无关 ⇒ 断言逐字不变,只换夹具值。

    profile operation 必须有独立 owner，不能把整个 gateway startup 当 owner。"""
    import gateway.pairing as pairing
    import gateway.status as status
    import hermes_cli.profiles as profiles

    runner = _multiplex_profile_runner()
    runner.adapters = {}
    runner._failed_platforms = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._profile_adapter_operations = {}
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._retiring_adapter_cleanups = {}
    runner._published_adapter_cleanup_retry = {}
    runner._adapter_disconnect_timeout_secs = lambda: 5.0
    runner._adapter_credential_claim = lambda *_args: None
    runner._adapter_listener_claim = lambda *_args: None
    runner._configure_profile_adapter = lambda *_args: None
    runner.pairing_store = object()
    runner.pairing_stores = {
        "default": object(),
        "coder": object(),
        "writer": object(),
    }
    coder_entered = asyncio.Event()
    coder_cleanup_entered = asyncio.Event()
    release_coder_cleanup = asyncio.Event()
    writer_connected = asyncio.Event()

    class _Adapter:
        def __init__(self, profile_name):
            self.profile_name = profile_name

        async def disconnect(self):
            if self.profile_name == "coder":
                coder_cleanup_entered.set()
                await release_coder_cleanup.wait()

    coder = _Adapter("coder")
    writer = _Adapter("writer")
    adapters = iter((coder, writer))
    configs = iter(
        (
            GatewayConfig(
                multiplex_profiles=True,
                platforms={
                    Platform.FEISHU: PlatformConfig(enabled=True, token="coder")
                },
            ),
            GatewayConfig(
                multiplex_profiles=True,
                platforms={
                    Platform.FEISHU: PlatformConfig(enabled=True, token="writer")
                },
            ),
        )
    )

    async def connect(adapter, _platform):
        if adapter is coder:
            coder_entered.set()
            await asyncio.Event().wait()
        writer_connected.set()
        return True

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda **_kwargs: [
            ("coder", tmp_path / "coder"),
            ("writer", tmp_path / "writer"),
        ],
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: next(configs))
    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: next(adapters))
    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)
    monkeypatch.setattr(status, "write_runtime_status", lambda **_kwargs: None)
    monkeypatch.setattr(pairing, "PairingStore", lambda **_kwargs: object())

    startup = asyncio.create_task(runner._start_secondary_profile_adapters())
    await coder_entered.wait()
    runner._profile_runtime_unload_retry.add("coder")
    drain = asyncio.create_task(runner._drain_profile_adapter_operations("coder"))
    try:
        await coder_cleanup_entered.wait()
        assert not startup.done()
        release_coder_cleanup.set()
        await drain
        assert await startup == 1
        assert writer_connected.is_set()
        assert runner._profile_adapters["writer"][Platform.FEISHU] is writer
    finally:
        release_coder_cleanup.set()
        for task in (startup, drain):
            if not task.done():
                task.cancel()
        await asyncio.gather(startup, drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_secondary_startup_cleanup_failure_is_explicit_after_siblings_start(
    monkeypatch, tmp_path
):
    """一个 profile cleanup 失败要穿透 caller，但不能阻断兄弟 profile。"""
    import gateway.status as status
    import hermes_cli.profiles as profiles

    runner = _multiplex_profile_runner()
    runner.adapters = {}
    runner._failed_platforms = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._profile_adapter_operations = {}
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._retiring_adapter_cleanups = {}
    runner._published_adapter_cleanup_retry = {}
    runner._adapter_disconnect_timeout_secs = lambda: 5.0
    runner._adapter_credential_claim = lambda *_args: None
    runner._adapter_listener_claim = lambda *_args: None
    runner._configure_profile_adapter = lambda *_args: None
    runner.pairing_store = object()
    runner.pairing_stores = {
        "default": object(),
        "coder": object(),
        "writer": object(),
    }

    class _Adapter:
        def __init__(self, profile_name):
            self.profile_name = profile_name

        async def disconnect(self):
            if self.profile_name == "coder":
                raise RuntimeError("coder cleanup failed")

    coder = _Adapter("coder")
    writer = _Adapter("writer")
    adapters = iter((coder, writer))
    configs = iter(
        GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.FEISHU: PlatformConfig(enabled=True, token=name)
            },
        )
        for name in ("coder", "writer")
    )

    async def connect(adapter, _platform):
        return adapter is writer

    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(
        profiles,
        "profiles_to_serve",
        lambda **_kwargs: [
            ("coder", tmp_path / "coder"),
            ("writer", tmp_path / "writer"),
        ],
    )
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: next(configs))
    monkeypatch.setattr(runner, "_create_adapter", lambda *_args: next(adapters))
    monkeypatch.setattr(runner, "_connect_initial_adapter_with_timeout", connect)
    monkeypatch.setattr(status, "write_runtime_status", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="secondary profile cleanup failed"):
        await runner._start_secondary_profile_adapters()

    assert runner._partial_adapter_cleanup_retry[
        ("coder", Platform.FEISHU)
    ] is coder
    assert runner._profile_adapters["writer"][Platform.FEISHU] is writer


@pytest.mark.asyncio
async def test_profile_start_retries_partial_owner_when_platform_is_disabled(
    monkeypatch, tmp_path
):
    """配置删除 pending owner 后，启动仍要先完成它的 cleanup。"""
    runner = _multiplex_profile_runner()
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._adapter_disconnect_timeout_secs = lambda: 5.0
    stale = SimpleNamespace(disconnect=AsyncMock())
    sibling = object()
    runner._profile_adapters["coder"] = {Platform.SLACK: sibling}
    runner._partial_adapter_cleanup_retry[("coder", Platform.FEISHU)] = stale

    @contextmanager
    def profile_scope(_home):
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", profile_scope)
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={Platform.FEISHU: PlatformConfig(enabled=False)},
        ),
    )

    assert (
        await runner._start_one_profile_adapters_owned(
            "coder", tmp_path / "profiles" / "coder", {}
        )
        == 0
    )
    stale.disconnect.assert_awaited_once()
    assert runner._partial_adapter_cleanup_retry == {}
    assert runner._profile_adapters["coder"][Platform.SLACK] is sibling


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["load", "policy", "port"])
async def test_partial_cleanup_precedes_new_profile_validation(
    monkeypatch, tmp_path, failure_mode
):
    """旧 owner cleanup 不得被新配置的任一校验错误挡住。"""
    runner = _multiplex_profile_runner()
    runner._partial_adapter_cleanup_retry = {}
    runner._partial_adapter_cleanup_tasks = {}
    runner._profile_runtime_unloads = {}
    runner._profile_runtime_unload_retry = set()
    runner._adapter_disconnect_timeout_secs = lambda: 5.0
    stale = SimpleNamespace(disconnect=AsyncMock())
    sibling = object()
    runner._profile_adapters["coder"] = {Platform.SLACK: sibling}
    runner._partial_adapter_cleanup_retry[("coder", Platform.FEISHU)] = stale

    @contextmanager
    def profile_scope(_home):
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", profile_scope)
    if failure_mode == "load":
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: (_ for _ in ()).throw(RuntimeError("config broken")),
        )
        expected = RuntimeError
    elif failure_mode == "policy":
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: GatewayConfig(multiplex_profiles=True),
        )
        monkeypatch.setattr(
            gateway_run, "_own_policy_open_startup_violation", lambda _cfg: "dm_policy"
        )
        expected = gateway_run.MultiplexConfigError
    else:
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: GatewayConfig(
                multiplex_profiles=True,
                platforms={
                    Platform.DISCORD: PlatformConfig(enabled=True, token="token")
                },
            ),
        )
        monkeypatch.setattr(
            gateway_run, "_own_policy_open_startup_violation", lambda _cfg: None
        )
        monkeypatch.setattr(gateway_run, "_platform_binds_port", lambda *_args: True)
        expected = gateway_run.SecondaryPortBindingConfigError

    with pytest.raises(expected):
        await runner._start_one_profile_adapters_owned(
            "coder", tmp_path / "profiles" / "coder", {}
        )
    stale.disconnect.assert_awaited_once()
    assert runner._partial_adapter_cleanup_retry == {}
    assert runner._profile_adapters["coder"][Platform.SLACK] is sibling


class TestProfileMessageHandler:
    @pytest.mark.asyncio
    async def test_stamps_profile_on_unstamped_source(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        seen = {}

        async def _fake_handle(event):
            seen["profile"] = event.source.profile
            return "ok"

        runner._handle_message = _fake_handle
        handler = runner._make_profile_message_handler("coder")

        class _Src:
            profile = None

        class _Evt:
            source = _Src()

        result = await handler(_Evt())
        assert result == "ok"
        assert seen["profile"] == "coder"


class _SecondaryRecoveryAdapter:
    platform = Platform.DISCORD

    def __init__(self, *, retryable=True):
        self.fatal_error_retryable = retryable
        self.fatal_error_code = "transport_stale" if retryable else "auth_failed"
        self.fatal_error_message = "Gateway transport stale"
        self.connected = False
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True

    def set_message_handler(self, handler):
        self.message_handler = handler

    def set_fatal_error_handler(self, handler):
        self.fatal_error_handler = handler

    def set_session_store(self, store):
        self.session_store = store

    def set_busy_session_handler(self, handler):
        self.busy_session_handler = handler

    def set_topic_recovery_fn(self, handler):
        self.topic_recovery_fn = handler

    def set_authorization_check(self, handler):
        self.authorization_check = handler


def _secondary_recovery_runner(*, running=True):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = running
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._background_tasks = set()
    runner.session_store = object()
    runner._handle_active_session_busy_message = object()
    runner._recover_telegram_topic_thread_id = object()
    runner._busy_text_mode = "queue"
    runner._make_adapter_auth_check = lambda platform, profile_name=None: object()
    runner._adapter_disconnect_timeout_secs = lambda: 0
    runner._sync_voice_mode_state_to_adapter = lambda adapter: None
    return runner


def _install_secondary_reconnect_context(monkeypatch, runner, adapter, scoped_homes=None):
    @contextmanager
    def fake_scope(profile_home):
        if scoped_homes is not None:
            scoped_homes.append(Path(profile_home))
        yield

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_scope)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda name: Path("/profiles") / name
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True, token="profile-token"
                )
            },
        ),
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, config: adapter)


class TestSecondaryProfileFatalRecovery:
    @pytest.mark.asyncio
    async def test_retryable_secondary_fatal_reconnects_with_its_profile_scope(
        self, monkeypatch
    ):
        runner = _secondary_recovery_runner()
        stale = _SecondaryRecoveryAdapter()
        replacement = _SecondaryRecoveryAdapter()
        runner._profile_adapters["reviewer"] = {Platform.DISCORD: stale}
        scoped_homes: list[Path] = []
        _install_secondary_reconnect_context(
            monkeypatch, runner, replacement, scoped_homes
        )

        async def connect(adapter, platform, *, is_reconnect=False):
            assert adapter is replacement
            assert platform is Platform.DISCORD
            assert is_reconnect is True
            replacement.connected = True
            return True

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)
        await runner._handle_profile_adapter_fatal_error(
            "reviewer", Platform.DISCORD, stale
        )

        assert stale.disconnected is True
        assert Platform.DISCORD not in runner._profile_adapters["reviewer"]
        tasks = list(runner._background_tasks)
        assert len(tasks) == 1
        await tasks[0]
        assert (
            runner._profile_adapters.get("reviewer", {}).get(Platform.DISCORD)
            is replacement
        )
        assert scoped_homes
        assert all(path == Path("/profiles/reviewer") for path in scoped_homes)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("startup_phase", [False, True], ids=["running", "startup"])
    async def test_secondary_fatal_disconnect_failure_keeps_exact_owner(
        self, startup_phase
    ):
        runner = _secondary_recovery_runner(running=not startup_phase)
        runner._startup_restore_in_progress = startup_phase
        stale = _SecondaryRecoveryAdapter()
        stale.disconnect = AsyncMock(side_effect=RuntimeError("old poller alive"))
        runner._profile_adapters["reviewer"] = {Platform.DISCORD: stale}
        retry_started = asyncio.Event()
        release_retry = asyncio.Event()

        async def retry(profile_name, platform):
            assert (profile_name, platform) == ("reviewer", Platform.DISCORD)
            retry_started.set()
            await release_retry.wait()

        runner._run_secondary_profile_reconnect = retry

        with pytest.raises(RuntimeError, match="old poller alive"):
            await runner._handle_profile_adapter_fatal_error(
                "reviewer", Platform.DISCORD, stale
            )

        assert runner._profile_adapters["reviewer"][Platform.DISCORD] is stale
        assert "reviewer" in runner._profile_failed_platforms
        retry_task = runner._profile_failed_platforms["reviewer"][Platform.DISCORD]
        assert not retry_task.done()
        await retry_started.wait()
        release_retry.set()
        await retry_task

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_mode", ["false", "raise"])
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_secondary_startup_retryable_failure_retries_after_window(
        self, monkeypatch, connect_mode, cleanup_fails
    ):
        """startup 窗口的首轮 retryable 失败必须走到下一轮真实 connect。"""
        runner = _secondary_recovery_runner(running=False)
        runner._startup_restore_in_progress = True
        first = _SecondaryRecoveryAdapter()
        replacement = _SecondaryRecoveryAdapter()
        adapters = iter((first, replacement))
        runner._profile_failed_platforms["reviewer"] = {}
        _install_secondary_reconnect_context(monkeypatch, runner, replacement)
        monkeypatch.setattr(
            runner,
            "_create_adapter",
            lambda _platform, _config: next(adapters),
        )
        connect_calls = 0
        cleanup_calls = 0

        async def connect(adapter, _platform, *, is_reconnect=False):
            nonlocal connect_calls
            assert is_reconnect is True
            connect_calls += 1
            if connect_calls == 1:
                if connect_mode == "raise":
                    raise RuntimeError("retryable connect failed")
                return False
            adapter.connected = True
            return True

        async def disconnect():
            nonlocal cleanup_calls
            cleanup_calls += 1
            first.disconnected = True
            if cleanup_fails and cleanup_calls == 1:
                raise RuntimeError("retryable cleanup failed")

        first.disconnect = disconnect
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)

        async def advance_startup_window(_delay):
            # 让首轮失败先观察到 startup flag，再确定性地进入运行态。
            runner._running = True

        monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda _attempt: 0)
        monkeypatch.setattr(gateway_run.asyncio, "sleep", advance_startup_window)

        task = asyncio.create_task(
            runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)
        )
        runner._profile_failed_platforms["reviewer"][Platform.DISCORD] = task
        await task

        assert connect_calls == 2
        assert cleanup_calls == (2 if cleanup_fails else 1)
        assert first.disconnected is True
        assert (
            runner._profile_adapters.get("reviewer", {}).get(Platform.DISCORD)
            is replacement
        )
        assert runner._profile_failed_platforms == {}

    @pytest.mark.asyncio
    async def test_secondary_startup_success_publishes_replacement(self, monkeypatch):
        """startup restore 仍在进行时，首次成功的 replacement 必须登记 owner。"""
        runner = _secondary_recovery_runner(running=False)
        runner._startup_restore_in_progress = True
        replacement = _SecondaryRecoveryAdapter()
        runner._profile_failed_platforms["reviewer"] = {}
        _install_secondary_reconnect_context(monkeypatch, runner, replacement)
        monkeypatch.setattr(
            runner, "_create_adapter", lambda _platform, _config: replacement
        )

        async def connect(_adapter, _platform, *, is_reconnect=False):
            assert is_reconnect is True
            return True

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)

        task = asyncio.create_task(
            runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)
        )
        runner._profile_failed_platforms["reviewer"][Platform.DISCORD] = task
        await task

        assert (
            runner._profile_adapters.get("reviewer", {}).get(Platform.DISCORD)
            is replacement
        )
        assert replacement.disconnected is False
        assert runner._profile_failed_platforms == {}

    @pytest.mark.asyncio
    async def test_secondary_startup_shutdown_event_never_publishes(self, monkeypatch):
        """shutdown 与 startup flag 同时存在时，connect 返回也不能重新发布。"""
        runner = _secondary_recovery_runner(running=False)
        runner._startup_restore_in_progress = True
        runner._shutdown_event = asyncio.Event()
        replacement = _SecondaryRecoveryAdapter()
        runner._profile_failed_platforms["reviewer"] = {}
        _install_secondary_reconnect_context(monkeypatch, runner, replacement)
        monkeypatch.setattr(
            runner, "_create_adapter", lambda _platform, _config: replacement
        )

        async def connect(_adapter, _platform, *, is_reconnect=False):
            assert is_reconnect is True
            runner._shutdown_event.set()
            return True

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)

        task = asyncio.create_task(
            runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)
        )
        runner._profile_failed_platforms["reviewer"][Platform.DISCORD] = task
        await task

        assert (
            runner._profile_adapters.get("reviewer", {}).get(Platform.DISCORD)
            is not replacement
        )
        assert replacement.disconnected is True
        assert runner._profile_failed_platforms == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("startup_phase", [False, True], ids=["running", "startup"])
    async def test_nonretryable_secondary_fatal_retries_cleanup_only(
        self, startup_phase
    ):
        """secondary 认证 fatal 只重试 cleanup，不得偷偷进入业务重连。"""
        runner = _secondary_recovery_runner()
        runner._running = not startup_phase
        runner._startup_restore_in_progress = startup_phase
        stale = _SecondaryRecoveryAdapter(retryable=False)
        runner._profile_adapters["reviewer"] = {Platform.DISCORD: stale}
        retry_entered = asyncio.Event()
        release_retry = asyncio.Event()
        calls = 0

        async def disconnect():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("old poller alive")
            retry_entered.set()
            await release_retry.wait()

        stale.disconnect = disconnect
        try:
            with pytest.raises(RuntimeError, match="old poller alive"):
                await runner._handle_profile_adapter_fatal_error(
                    "reviewer", Platform.DISCORD, stale
                )
            key = ("reviewer", Platform.DISCORD)
            assert key in runner._published_adapter_cleanup_tasks
            retry_task = runner._published_adapter_cleanup_tasks[key]
            await retry_entered.wait()
            assert runner._profile_adapters["reviewer"][Platform.DISCORD] is stale
            assert runner._profile_failed_platforms == {}
            assert not retry_task.done()

            release_retry.set()
            await retry_task
            assert runner._profile_adapters["reviewer"] == {}
            assert runner._profile_failed_platforms == {}
        finally:
            runner._running = False
            release_retry.set()
            tasks = list(
                getattr(runner, "_published_adapter_cleanup_tasks", {}).values()
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_nonretryable_secondary_fatal_cleanup_success_never_reconnects(self):
        """首次 cleanup 成功也不能把认证 fatal 误送进 replacement 队列。"""
        runner = _secondary_recovery_runner()
        stale = _SecondaryRecoveryAdapter(retryable=False)
        runner._profile_adapters["reviewer"] = {Platform.DISCORD: stale}

        await runner._handle_profile_adapter_fatal_error(
            "reviewer", Platform.DISCORD, stale
        )

        assert stale.disconnected is True
        assert runner._profile_adapters["reviewer"] == {}
        assert runner._profile_failed_platforms == {}
        assert runner._background_tasks == set()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_mode", ["false", "raise"])
    @pytest.mark.parametrize("cleanup_fails", [False, True])
    async def test_secondary_reconnect_terminal_error_only_retries_cleanup(
        self, monkeypatch, connect_mode, cleanup_fails
    ):
        """terminal reconnect 的四种失败组合都不得创建第二个 adapter。"""
        runner = _secondary_recovery_runner()
        terminal = _SecondaryRecoveryAdapter(retryable=False)
        terminal.has_fatal_error = True
        runner._profile_failed_platforms["reviewer"] = {}
        _install_secondary_reconnect_context(monkeypatch, runner, terminal)
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
            terminal.disconnected = True

        terminal.disconnect = disconnect
        monkeypatch.setattr(runner, "_create_adapter", create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)
        monkeypatch.setattr(gateway_run, "_reconnect_backoff", lambda _attempt: 0)
        monkeypatch.setattr(gateway_run.asyncio, "sleep", AsyncMock())

        task = asyncio.create_task(
            runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)
        )
        runner._profile_failed_platforms["reviewer"][Platform.DISCORD] = task
        await task

        assert connect_calls == 1
        assert cleanup_calls == (2 if cleanup_fails else 1)
        assert terminal.disconnected is True
        assert runner._profile_failed_platforms == {}


    @pytest.mark.asyncio
    @pytest.mark.parametrize("connect_result", [True, False], ids=["success", "failure"])
    async def test_secondary_reconnect_does_not_publish_after_shutdown(
        self, monkeypatch, connect_result
    ):
        runner = _secondary_recovery_runner()
        runner._profile_failed_platforms["reviewer"] = {}
        replacement = _SecondaryRecoveryAdapter()
        _install_secondary_reconnect_context(monkeypatch, runner, replacement)
        connect_started = asyncio.Event()
        release_connect = asyncio.Event()

        async def connect(adapter, platform, *, is_reconnect=False):
            connect_started.set()
            await release_connect.wait()
            return connect_result

        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect)
        task = asyncio.create_task(
            runner._run_secondary_profile_reconnect("reviewer", Platform.DISCORD)
        )
        runner._profile_failed_platforms["reviewer"][Platform.DISCORD] = task
        await connect_started.wait()
        runner._running = False
        release_connect.set()
        await asyncio.wait_for(task, timeout=0.2)

        assert runner._profile_adapters == {}
        assert replacement.disconnected is True
        assert runner._profile_failed_platforms == {}


class TestSecondaryProfileConfigHandling:
    """Secondary config errors degrade only when the profile is safe to skip."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_uses_degradable_error(self, monkeypatch):
        from gateway.run import SecondaryPortBindingConfigError
class TestPortBindingSkip:
    """A secondary profile enabling a port-binding platform is skipped."""

    @pytest.mark.asyncio
    async def test_secondary_webhook_skips_listener(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import SecondaryPortBindingConfigError

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        # reviewer profile config enables webhook (a port-binding platform)
        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        monkeypatch.setattr(
            runner,
            "_create_adapter",
            lambda *_: pytest.fail("port-binding adapter should be skipped"),
        )

        with pytest.raises(SecondaryPortBindingConfigError) as exc_info:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert "webhook" in str(exc_info.value)
        assert "reviewer" in str(exc_info.value)
        assert "reviewer" not in runner._profile_adapters

    @pytest.mark.asyncio
    async def test_secondary_reports_all_port_binding_platforms(self, monkeypatch):
        from gateway.run import SecondaryPortBindingConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            # connection_mode=webhook: with #52563's conditional check merged,
            # default (websocket) Feishu no longer binds a port — only webhook
            # mode should be reported here.
            Platform.FEISHU: PlatformConfig(
                enabled=True, extra={"connection_mode": "webhook"}
            ),
            Platform.WEBHOOK: PlatformConfig(enabled=True, extra={"port": 8644}),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        with pytest.raises(SecondaryPortBindingConfigError) as ei:
            await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        message = str(ei.value)
        assert "feishu" in message
        assert "webhook" in message
        assert "telegram" not in message
        assert "reviewer" not in runner._profile_adapters

    @pytest.mark.asyncio
    async def test_multiplexer_skips_bad_profile_and_continues(self, monkeypatch, caplog):
        from pathlib import Path
        from gateway.config import GatewayConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {}
        runner._profile_adapters = {}

        async def fake_start_one(profile_name, profile_home, claimed):
            if profile_name == "bad":
                from gateway.run import SecondaryPortBindingConfigError
                raise SecondaryPortBindingConfigError("bad enables webhook")
            runner._profile_adapters[profile_name] = {}
            return 2

        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", Path("/tmp/default")),
                ("bad", Path("/tmp/bad")),
                ("good", Path("/tmp/good")),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name",
            lambda: "default",
        )
        monkeypatch.setattr(runner, "_start_one_profile_adapters", fake_start_one)
        monkeypatch.setattr(
            "gateway.status.write_runtime_status",
            lambda **kwargs: None,
        )

        caplog.set_level(logging.WARNING, logger="gateway.run")
        connected = await runner._start_secondary_profile_adapters()

        assert connected == 2
        assert "good" in runner._profile_adapters
        assert "bad" not in runner._profile_adapters
        assert "Skipping secondary profile 'bad'" in caplog.text

    @pytest.mark.asyncio
    async def test_multiplexer_propagates_security_config_error(self, monkeypatch):
        from pathlib import Path
        from gateway.config import GatewayConfig
        from gateway.run import MultiplexConfigError

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner.adapters = {}
        runner._profile_adapters = {}

        async def fake_start_one(profile_name, profile_home, claimed):
            raise MultiplexConfigError(
                f"Profile '{profile_name}' enables open policy without allow-all opt-in"
            )

        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", Path("/tmp/default")),
                ("unsafe", Path("/tmp/unsafe")),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name",
            lambda: "default",
        )
        monkeypatch.setattr(runner, "_start_one_profile_adapters", fake_start_one)

        with pytest.raises(MultiplexConfigError, match="open policy"):
            await runner._start_secondary_profile_adapters()


    @pytest.mark.asyncio
    async def test_secondary_distinct_photon_credentials_distinct_ports_connect(
        self, monkeypatch
    ):
        """Multiplexing remains supported when Photon sidecars cannot collide."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _PhotonAdapter:
            def __init__(self, secret, port):
                self._project_secret = secret
                self._sidecar_bind = "127.0.0.1"
                self._sidecar_port = port
                self.platform = Platform("photon")
                self.connected = False
        class _DirectAdapter:
            platform = Platform.TELEGRAM

            def set_message_handler(self, handler):
                self.message_handler = handler

            def set_fatal_error_handler(self, handler):
                self.fatal_error_handler = handler

            def set_session_store(self, store):
                self.session_store = store

            def set_busy_session_handler(self, handler):
                self.busy_session_handler = handler

            def set_topic_recovery_fn(self, handler):
                self.topic_recovery_fn = handler

            def set_authorization_check(self, handler):
                self.authorization_check = handler

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._handle_adapter_fatal_error = object()
        runner._handle_active_session_busy_message = object()
        runner._recover_telegram_topic_thread_id = object()
        runner._busy_text_mode = "queue"
        runner._make_adapter_auth_check = lambda platform, profile_name=None: object()

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.RELAY: PlatformConfig(enabled=True),
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="reviewer-token"),
        }
        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )

        direct = _DirectAdapter()
        factory_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            if platform is Platform.RELAY:
                raise AssertionError("secondary Relay factory must not be invoked")
            return direct

        connect_calls = []

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        assert connected == 1
        assert factory_calls == [Platform.TELEGRAM]
        assert connect_calls == [(direct, Platform.TELEGRAM)]
        assert runner._profile_adapters["reviewer"] == {
            Platform.TELEGRAM: direct,
        }

    @pytest.mark.asyncio
    async def test_multiplex_secondary_skips_shared_zet_listener_but_starts_direct_adapter(
        self, monkeypatch
    ):
        """Zet ingress is process-shared; direct adapters remain per-profile."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = _multiplex_profile_runner()

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.ZET_AGENT: PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": 7900},
            ),
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="reviewer-token",
            ),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: reviewer_cfg)

        direct = _DirectProfileAdapter()
        factory_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            return direct

        connect_calls = []

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        assert connected == 1
        assert factory_calls == [Platform.TELEGRAM]
        assert connect_calls == [(direct, Platform.TELEGRAM)]
        assert runner._profile_adapters["reviewer"] == {
            Platform.TELEGRAM: direct,
        }

    @pytest.mark.asyncio
    async def test_global_zet_env_keeps_secondary_served_without_second_listener(
        self, tmp_path, monkeypatch
    ):
        """Process-wide Zet config must not become a per-profile listener."""
        from gateway.config import GatewayConfig, Platform, load_gateway_config

        default_home = tmp_path / "default"
        worker_home = tmp_path / "worker"
        default_home.mkdir()
        worker_home.mkdir()
        (worker_home / ".env").write_text(
            "TELEGRAM_BOT_TOKEN=worker-token\n",
            encoding="utf-8",
        )
        (worker_home / "config.yaml").write_text(
            "multiplex_profiles: true\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("ZET_AGENT_ENABLED", "true")
        monkeypatch.setenv("ZET_AGENT_HOST", "127.0.0.1")
        monkeypatch.setenv("ZET_AGENT_PORT", "7900")
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [
                ("default", default_home),
                ("worker", worker_home),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_active_profile_name",
            lambda: "default",
        )
        resolved_configs = []

        def _load_profile_config():
            config = load_gateway_config()
            resolved_configs.append(config)
            return config

        monkeypatch.setattr("gateway.config.load_gateway_config", _load_profile_config)

        runner = _multiplex_profile_runner()
        shared_zet_adapter = object()
        runner.adapters = {Platform.ZET_AGENT: shared_zet_adapter}
        runner.pairing_stores = {"default": object(), "worker": object()}

        direct = _DirectProfileAdapter()
        factory_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            return direct

        connect_calls = []

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)
        status_updates = []
        monkeypatch.setattr(
            "gateway.status.write_runtime_status",
            lambda **kwargs: status_updates.append(kwargs),
        )

        connected = await runner._start_secondary_profile_adapters()

        assert connected == 1
        assert len(resolved_configs) == 1
        injected_zet = resolved_configs[0].platforms[Platform.ZET_AGENT]
        assert injected_zet.enabled is True
        assert injected_zet.extra["port"] == 7900
        assert runner.adapters == {Platform.ZET_AGENT: shared_zet_adapter}
        assert factory_calls == [Platform.TELEGRAM]
        assert connect_calls == [(direct, Platform.TELEGRAM)]
        assert runner._profile_adapters["worker"] == {
            Platform.TELEGRAM: direct,
        }
        assert status_updates[-1] == {
            "served_profiles": ["default", "worker"],
        }
        assert all("platform" not in update for update in status_updates)

    @pytest.mark.asyncio
    async def test_non_multiplex_profile_adapter_start_keeps_relay(self, monkeypatch):
        """The Relay skip is gated to multiplex mode."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _RelayAdapter:
            platform = Platform.RELAY

            def set_message_handler(self, handler):
                pass

            def set_fatal_error_handler(self, handler):
                pass

            def set_session_store(self, store):
                pass

            def set_busy_session_handler(self, handler):
                pass

            def set_topic_recovery_fn(self, handler):
                pass

            def set_authorization_check(self, handler):
                pass

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=False)
        runner._profile_adapters = {}
        runner.session_store = object()
        runner._handle_adapter_fatal_error = object()
        runner._handle_active_session_busy_message = object()
        runner._recover_telegram_topic_thread_id = object()
        runner._busy_text_mode = "queue"
        runner._make_adapter_auth_check = lambda platform, profile_name=None: object()

        profile_cfg = GatewayConfig(multiplex_profiles=False)
        profile_cfg.platforms = {
            Platform.RELAY: PlatformConfig(enabled=True),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)

        relay = _RelayAdapter()
        factory_calls = []
        connect_calls = []

        def _create_adapter(platform, config):
            factory_calls.append(platform)
            return relay

        async def _connect(adapter, platform):
            connect_calls.append((adapter, platform))
            return True

        monkeypatch.setattr(runner, "_create_adapter", _create_adapter)
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", {}
        )

        assert connected == 1
        assert factory_calls == [Platform.RELAY]
        assert connect_calls == [(relay, Platform.RELAY)]

    @pytest.mark.asyncio
    async def test_secondary_same_config_token_is_refused(self, monkeypatch):
        """Adapters that keep their token on config still trip the mux guard."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _ConfigTokenAdapter:
            def __init__(self, token):
                self.config = PlatformConfig(enabled=True, token=token)
                self.disconnected = False

            async def connect(self):
                raise AssertionError("duplicate adapter must not connect")

            async def disconnect(self):
                self.disconnected = True

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="same-token"),
        }
        duplicate = _ConfigTokenAdapter("same-token")
        claimed = {
            (
                Platform.TELEGRAM,
                GatewayRunner._adapter_credential_fingerprint(
                    _ConfigTokenAdapter("same-token")
                ),
            ): "default"
        }

        monkeypatch.setattr(
            "gateway.config.load_gateway_config", lambda: reviewer_cfg
        )
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: duplicate)
        monkeypatch.setattr(runner, "_adapter_disconnect_timeout_secs", lambda: 0)

        connected = await runner._start_one_profile_adapters(
            "reviewer", "/tmp/x", claimed
        )

        assert connected == 0
        # The merged registry now rejects a duplicate from its config
        # fingerprint before the adapter is started, so there is nothing to
        # disconnect.
        assert duplicate.disconnected is False
        assert runner._profile_adapters["reviewer"] == {}

    @pytest.mark.asyncio
    async def test_failed_photon_connect_releases_listener_for_later_profile(
        self, monkeypatch
    ):
        """A failed sidecar must not reserve an endpoint it never owned."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        class _PhotonAdapter:
            def __init__(self, secret, should_connect):
                self._project_secret = secret
                self._sidecar_bind = "127.0.0.1"
                self._sidecar_port = 8789
                self.platform = Platform("photon")
                self.should_connect = should_connect
                self.disconnected = False
                self.config = PlatformConfig(enabled=True)

            def __getattr__(self, name):
                if name.startswith("set_"):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

            async def disconnect(self):
                self.disconnected = True

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}
        runner.session_store = None
        runner._busy_text_mode = "queue"

        photon = Platform("photon")
        profile_cfg = GatewayConfig(multiplex_profiles=True)
        profile_cfg.platforms = {photon: PlatformConfig(enabled=True)}
        failed = _PhotonAdapter("failed-secret", False)
        later = _PhotonAdapter("later-secret", True)
        adapters = iter((failed, later))
        claimed = {}

        async def _connect(adapter, platform):
            return adapter.should_connect

        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: profile_cfg)
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: next(adapters))
        monkeypatch.setattr(runner, "_connect_adapter_with_timeout", _connect)
        monkeypatch.setattr(
            runner, "_make_adapter_auth_check", lambda p, **kwargs: None
        )

        first = await runner._start_one_profile_adapters("broken", "/tmp/x", claimed)
        second = await runner._start_one_profile_adapters("later", "/tmp/y", claimed)

        assert first == 0
        assert failed.disconnected is True
        assert second == 1
        assert runner._profile_adapters["later"][photon] is later

    def test_port_binding_set_covers_known_listeners(self):
        from gateway.run import _PORT_BINDING_PLATFORM_VALUES

        # Every adapter that binds a TCP port must be in the guard set.
        for p in (
            "webhook",
            "api_server",
            "zet_agent",
            "msgraph_webhook",
            "feishu",
            "wecom_callback",
            "bluebubbles",
            "sms",
            "whatsapp_cloud",
            "line",
        ):
            assert p in _PORT_BINDING_PLATFORM_VALUES


class TestFeishuPortBindingConditional:
    """Feishu websocket mode does NOT bind a port; only webhook mode does (#52563)."""

    @pytest.mark.asyncio
    async def test_feishu_websocket_mode_not_rejected(self, monkeypatch):
        """Feishu in websocket mode (the default) should NOT raise MultiplexConfigError."""
        from gateway.run import MultiplexConfigError
        from gateway.config import GatewayConfig, Platform, PlatformConfig

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True)
        runner._profile_adapters = {}

        reviewer_cfg = GatewayConfig(multiplex_profiles=True)
        reviewer_cfg.platforms = {
            Platform.FEISHU: PlatformConfig(
                enabled=True,
                extra={"app_id": "cli_xxx", "app_secret": "sec", "connection_mode": "websocket"},
            ),
        }
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: reviewer_cfg)
        monkeypatch.setattr(runner, "_create_adapter", lambda p, c: None)

        connected = await runner._start_one_profile_adapters("reviewer", "/tmp/x", {})
        assert connected == 0  # no error, just nothing connected
