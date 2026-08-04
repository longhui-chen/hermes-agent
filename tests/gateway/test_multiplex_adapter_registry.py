"""Phase 3: secondary-profile adapter registry + same-token conflict detection."""
import logging
import asyncio
from contextlib import contextmanager
from pathlib import Path
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
        assert runner._profile_adapters["reviewer"][Platform.DISCORD] is replacement
        assert scoped_homes
        assert all(path == Path("/profiles/reviewer") for path in scoped_homes)


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
