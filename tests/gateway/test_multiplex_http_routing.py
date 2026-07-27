"""Phase 1: HTTP-inbound /p/<profile>/ routing for the webhook adapter."""
import pytest

from gateway.config import PlatformConfig
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_key


class TestSessionSourceProfileField:
    def test_profile_roundtrips(self):
        s = SessionSource(
            platform=Platform.WEBHOOK if hasattr(Platform, "WEBHOOK") else Platform.TELEGRAM,
            chat_id="c1",
            chat_type="webhook",
            profile="coder",
        )
        restored = SessionSource.from_dict(s.to_dict())
        assert restored.profile == "coder"

    def test_profile_absent_not_serialized(self):
        s = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm")
        assert "profile" not in s.to_dict()

    def test_source_profile_drives_session_key_namespace(self):
        s = SessionSource(platform=Platform.TELEGRAM, chat_id="99", chat_type="dm")
        # build_session_key takes profile explicitly; the adapter passes
        # source.profile through. Verify the namespace follows it.
        assert build_session_key(s, profile="coder") == "agent:coder:telegram:dm:99"


class TestWebhookProfileResolution:
    """_resolve_request_profile validates the /p/<profile>/ prefix."""

    def _adapter(self, multiplex: bool, served=("default", "coder")):
        from gateway.platforms.webhook import WebhookAdapter, _PROFILE_REJECTED

        class _FakeReq:
            def __init__(self, profile):
                self.match_info = {"profile": profile} if profile is not None else {}

        cfg = GatewayConfig(multiplex_profiles=multiplex)

        class _Runner:
            config = cfg

        # Construct minimally; we only call _resolve_request_profile.
        adapter = WebhookAdapter.__new__(WebhookAdapter)
        adapter.gateway_runner = _Runner()
        return adapter, _FakeReq, _PROFILE_REJECTED, served

    def test_no_prefix_returns_none(self):
        adapter, Req, _REJ, _ = self._adapter(multiplex=True)
        assert adapter._resolve_request_profile(Req(None)) is None

    def test_prefix_ignored_when_multiplex_off(self):
        adapter, Req, _REJ, _ = self._adapter(multiplex=False)
        # Even a bogus profile is ignored (not 404'd) when multiplexing is off.
        assert adapter._resolve_request_profile(Req("anything")) is None

    def test_known_profile_accepted(self, monkeypatch):
        adapter, Req, _REJ, served = self._adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [(n, None) for n in served],
        )
        assert adapter._resolve_request_profile(Req("coder")) == "coder"

    def test_unknown_profile_rejected(self, monkeypatch):
        adapter, Req, REJ, served = self._adapter(multiplex=True)
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex: [(n, None) for n in served],
        )
        assert adapter._resolve_request_profile(Req("ghost")) is REJ


class TestZetAgentProfileUnload:
    class _FakeRequest(dict):
        def __init__(self, profile_home):
            super().__init__(
                hermes_profile="coder",
                hermes_profile_home=str(profile_home),
            )
            self.headers = {"Authorization": "Bearer test-key"}
            self.method = "POST"
            self.path_qs = "/p/coder/v1/profile/unload"
            self.remote = "127.0.0.1"
            self.transport = None

    @pytest.mark.asyncio
    async def test_active_chat_completion_blocks_profile_unload(self, tmp_path):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

        run_key = adapter._begin_profile_chat_run(profile_home)
        try:
            response = await adapter._handle_profile_unload(
                self._FakeRequest(profile_home)
            )
        finally:
            adapter._end_profile_chat_run(run_key)

        assert response.status == 409
        assert '"active_api_runs": 1' in response.text

    @pytest.mark.asyncio
    async def test_profile_unload_closes_session_db_with_resolved_home_key(self, tmp_path):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        class _DB:
            closed = False
            staging_discarded = False

            def discard_runtime_import_staging(self):
                self.staging_discarded = True
                return 1

            def close(self):
                assert self.staging_discarded is True
                self.closed = True

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        db = _DB()
        adapter._session_dbs[adapter._profile_home_key(profile_home)] = db

        response = await adapter._handle_profile_unload(
            self._FakeRequest(profile_home / ".")
        )

        assert response.status == 200
        assert db.staging_discarded is True
        assert db.closed is True
        assert adapter._session_dbs == {}


class TestZetAgentModelSwitchAuth:
    class _FakeRequest(dict):
        def __init__(self, authorization=None):
            super().__init__(
                hermes_profile="coder",
                hermes_profile_home="/tmp/hermes-test-profile",
            )
            self.headers = {}
            if authorization is not None:
                self.headers["Authorization"] = authorization
            self.method = "POST"
            self.path_qs = "/p/coder/v1/model/switch"
            self.remote = "127.0.0.1"
            self.transport = None

        async def json(self):
            raise AssertionError("unauthorized model switch must not parse JSON")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("authorization", "status"),
        [
            (None, 401),
            ("Bearer wrong-key", 401),
        ],
    )
    async def test_profile_model_switch_requires_bearer(self, authorization, status):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        response = await adapter._handle_model_switch(
            self._FakeRequest(authorization)
        )

        assert response.status == status
