"""Phase 1: HTTP-inbound /p/<profile>/ routing for the webhook adapter."""
import asyncio
import builtins
import json
import logging
import re
import threading
import time

import pytest

from gateway.config import PlatformConfig
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, build_session_key


@pytest.mark.asyncio
async def test_approval_import_failure_is_safe_and_correlated(monkeypatch, caplog):
    from gateway.platforms.zet_agent import ZetAgentAdapter

    secret_error = "/private/profile/approval.py import failed"
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tools.approval":
            raise RuntimeError(secret_error)
        return real_import(name, *args, **kwargs)

    class _Request:
        match_info = {"session_id": "session-1"}
        headers = {"Authorization": "Bearer test-key"}
        method = "POST"
        path_qs = "/v1/sessions/session-1/approval/respond"
        remote = "127.0.0.1"
        transport = None

        async def json(self):
            return {"choice": "deny"}

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with caplog.at_level(logging.ERROR):
        response = await adapter._handle_approval_respond(_Request())

    payload = json.loads(response.text)
    assert response.status == 500
    assert payload["error"]["code"] == "approval_module_unavailable"
    message = payload["error"]["message"]
    assert secret_error not in message
    reference = re.search(r"reference ([0-9a-f]{12})", message)
    assert reference is not None
    assert any(
        f"correlation_id={reference.group(1)}" in record.message
        and record.exc_info
        and secret_error in str(record.exc_info[1])
        for record in caplog.records
    )


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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_stage", ["discard", "close"])
    async def test_profile_unload_db_failure_keeps_owner_and_returns_safe_error(
        self, tmp_path, caplog, failure_stage
    ):
        """清理失败要保留精确 DB owner，且 HTTP 不泄露底层错误。"""
        from gateway.platforms.zet_agent import ZetAgentAdapter

        secret_error = "/private/profile/state.db close failed"

        class _DB:
            def __init__(self):
                self.failure_stage = failure_stage
                self.closed = False

            def discard_runtime_import_staging(self):
                if self.failure_stage == "discard":
                    raise RuntimeError(secret_error)

            def close(self):
                if self.failure_stage == "close":
                    raise RuntimeError(secret_error)
                self.closed = True

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        db = _DB()
        key = adapter._profile_home_key(profile_home)
        adapter._session_dbs[key] = db
        adapter._session_db = db

        with caplog.at_level(logging.ERROR):
            response = await adapter._handle_profile_unload(
                self._FakeRequest(profile_home)
            )

        body = json.loads(response.text)
        assert response.status == 500
        assert body["error"]["code"] == "profile_unload_unavailable"
        assert "Retry" in body["error"]["message"]
        reference = re.search(
            r"reference ([0-9a-f]{12})", body["error"]["message"]
        )
        assert reference is not None
        assert secret_error not in body["error"]["message"]
        assert adapter._session_dbs[key] is db
        assert adapter._session_db is db
        assert any(
            f"correlation_id={reference.group(1)}" in record.message
            for record in caplog.records
        )
        error_records = [record for record in caplog.records if record.exc_info]
        assert any(
            secret_error in str(record.exc_info[1])
            for record in error_records
        )

        db.failure_stage = None
        retry = await adapter._handle_profile_unload(self._FakeRequest(profile_home))
        assert retry.status == 200
        assert db.closed is True
        assert key not in adapter._session_dbs
        assert adapter._session_db is None

    @pytest.mark.asyncio
    async def test_profile_unload_db_cas_preserves_replacement_owner(
        self, tmp_path
    ):
        """close 期间 owner 改判时，不得删除刚发布的 replacement。"""
        from gateway.platforms.zet_agent import ZetAgentAdapter

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        key = adapter._profile_home_key(profile_home)
        replacement = object()

        class _DB:
            def close(self):
                adapter._session_dbs[key] = replacement
                adapter._session_db = replacement

        old_db = _DB()
        adapter._session_dbs[key] = old_db
        adapter._session_db = old_db

        response = await adapter._handle_profile_unload(
            self._FakeRequest(profile_home)
        )

        assert response.status == 500
        assert adapter._session_dbs[key] is replacement
        assert adapter._session_db is replacement

    @pytest.mark.asyncio
    async def test_profile_unload_onboarding_retiring_owner_returns_safe_error(
        self, tmp_path
    ):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        secret_error = "/private/profile/onboarding cleanup in flight"

        def fail_onboarding_cleanup(_profile):
            raise RuntimeError(secret_error)

        adapter._close_onboarding_agents_for_profile = fail_onboarding_cleanup
        response = await adapter._handle_profile_unload(
            self._FakeRequest(profile_home)
        )

        body = json.loads(response.text)
        assert response.status == 500
        assert body["error"]["code"] == "profile_unload_unavailable"
        assert "Retry" in body["error"]["message"]
        assert secret_error not in body["error"]["message"]
        assert re.search(r"reference [0-9a-f]{12}", body["error"]["message"])

    @pytest.mark.asyncio
    async def test_profile_unload_db_close_blocks_concurrent_open(
        self, tmp_path, monkeypatch
    ):
        """open 必须确定性等 close 提交删除后，不能拿到正在关闭的旧 DB。"""
        from gateway.platforms.zet_agent import ZetAgentAdapter

        close_entered = threading.Event()
        release_close = threading.Event()
        open_waiting = threading.Event()
        open_done = threading.Event()

        class _ObservedLock:
            def __init__(self):
                self._lock = threading.RLock()

            def __enter__(self):
                if threading.current_thread().name == "db-open":
                    open_waiting.set()
                self._lock.acquire()
                return self

            def __exit__(self, *_exc):
                self._lock.release()

        class _OldDB:
            def close(self):
                close_entered.set()
                assert release_close.wait(timeout=2)

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        adapter._session_db_init_lock = _ObservedLock()
        key = adapter._profile_home_key(profile_home)
        old_db = _OldDB()
        replacement = object()
        adapter._session_dbs[key] = old_db
        adapter._session_db = old_db
        monkeypatch.setattr(
            adapter,
            "_open_profile_session_db_with_repair",
            lambda _home: replacement,
        )
        opened = {}

        def open_db():
            opened["db"] = adapter._open_and_cache_session_db(profile_home)
            open_done.set()

        unload = asyncio.create_task(
            adapter._handle_profile_unload(self._FakeRequest(profile_home))
        )
        assert await asyncio.to_thread(close_entered.wait, 2)
        opener = threading.Thread(target=open_db, name="db-open")
        opener.start()
        try:
            assert open_waiting.wait(timeout=2)
            assert not open_done.is_set()
            release_close.set()
            response = await unload
            opener.join(timeout=2)
            assert not opener.is_alive()
            assert response.status == 200
            assert opened["db"] is replacement
            assert opened["db"] is not old_db
        finally:
            release_close.set()
            opener.join(timeout=2)

    @pytest.mark.asyncio
    async def test_profile_unload_strictly_closes_only_target_onboarding_cache(
        self, tmp_path
    ):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        class _Agent:
            def __init__(self, fail=False):
                self.fail = fail
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.fail:
                    raise RuntimeError("child still running")

        profile_home = tmp_path / "profiles" / "coder"
        profile_home.mkdir(parents=True)
        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        target = _Agent(fail=True)
        sibling = _Agent()
        target_key = ("coder", "session")
        sibling_key = ("main", "session")
        adapter._onboarding_agent_cache[target_key] = (target, time.monotonic())
        adapter._onboarding_agent_cache[sibling_key] = (sibling, time.monotonic())

        failed = await adapter._handle_profile_unload(
            self._FakeRequest(profile_home)
        )
        assert failed.status == 500
        assert adapter._onboarding_agent_cache[target_key][0] is target
        assert adapter._onboarding_agent_cache[sibling_key][0] is sibling
        assert sibling.close_calls == 0

        target.fail = False
        retried = await adapter._handle_profile_unload(
            self._FakeRequest(profile_home)
        )
        assert retried.status == 200
        assert target_key not in adapter._onboarding_agent_cache
        assert adapter._onboarding_agent_cache[sibling_key][0] is sibling
        assert target.close_calls == 2
        assert sibling.close_calls == 0


class TestZetAgentOnboardingCacheOwnership:
    @pytest.mark.parametrize("mode", ["replacement", "capacity"])
    def test_cleanup_failure_retains_old_owner_until_retry(
        self, mode
    ):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        adapter._onboarding_agent_cache_cap = 1

        class _Agent:
            def __init__(self):
                self.fail = True
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.fail:
                    raise RuntimeError("child still running")

        old = _Agent()
        new = _Agent()
        new.fail = False
        old_key = ("coder", "old")
        new_key = old_key if mode == "replacement" else ("coder", "new")
        adapter._onboarding_agent_cache[old_key] = (old, time.monotonic())

        with pytest.raises(RuntimeError, match="cleanup failed"):
            adapter._publish_onboarding_agent(new_key, new)
        assert adapter._onboarding_agent_cache[old_key][0] is old
        assert new.close_calls == 1
        if new_key != old_key:
            assert new_key not in adapter._onboarding_agent_cache

        old.fail = False
        replacement = _Agent()
        replacement.fail = False
        adapter._publish_onboarding_agent(new_key, replacement)
        assert adapter._onboarding_agent_cache[new_key][0] is replacement
        assert old_key == new_key or old_key not in adapter._onboarding_agent_cache

    def test_incoming_cleanup_failure_retains_retry_owner(self):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

        class _Agent:
            def __init__(self, fail):
                self.fail = fail
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.fail:
                    raise RuntimeError("child still running")

        old = _Agent(True)
        incoming = _Agent(True)
        key = ("coder", "session")
        adapter._onboarding_agent_cache[key] = (old, time.monotonic())

        with pytest.raises(RuntimeError, match="incoming-agent cleanup failed"):
            adapter._publish_onboarding_agent(key, incoming)
        assert adapter._onboarding_cleanup_retry_agents[id(incoming)][1] is incoming
        assert adapter._onboarding_agent_cache[key][0] is old

        incoming.fail = False
        old.fail = False
        assert adapter._cached_onboarding_agent(key) is None
        assert id(incoming) not in adapter._onboarding_cleanup_retry_agents
        assert id(old) not in adapter._onboarding_cleanup_retry_agents
        replacement = _Agent(False)
        adapter._publish_onboarding_agent(key, replacement)
        assert incoming.close_calls == 2
        assert adapter._onboarding_agent_cache[key][0] is replacement

    def test_ttl_cleanup_failure_retains_expired_owner_until_retry(self):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        adapter._onboarding_agent_cache_ttl_seconds = 1

        class _Agent:
            fail = True

            def close(self):
                if self.fail:
                    raise RuntimeError("child still running")

        stale = _Agent()
        key = ("coder", "expired")
        adapter._onboarding_agent_cache[key] = (stale, 0)
        sibling = _Agent()
        sibling_key = ("writer", "active")
        adapter._onboarding_agent_cache[sibling_key] = (sibling, time.monotonic())

        with pytest.raises(RuntimeError, match="cleanup failed"):
            adapter._cached_onboarding_agent(("coder", "other"))
        assert adapter._onboarding_agent_cache[key][0] is stale
        assert adapter._cached_onboarding_agent(sibling_key) is sibling
        assert adapter._onboarding_agent_cache[key][0] is stale

        stale.fail = False
        assert adapter._cached_onboarding_agent(("coder", "other")) is None
        assert key not in adapter._onboarding_agent_cache

    def test_profile_cleanup_failure_does_not_retire_unattempted_siblings(self):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

        class _Agent:
            def __init__(self, fail=False):
                self.fail = fail
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                if self.fail:
                    raise RuntimeError("child still running")

        first = _Agent(True)
        second = _Agent(False)
        adapter._onboarding_agent_cache[("coder", "first")] = (first, 0)
        adapter._onboarding_agent_cache[("coder", "second")] = (second, 0)

        with pytest.raises(RuntimeError, match="profile-unload cleanup failed"):
            adapter._close_onboarding_agents_for_profile("coder")
        assert first.close_calls == 1
        assert second.close_calls == 0
        assert id(second) not in adapter._onboarding_retiring_agents
        assert adapter._onboarding_cleanup_retry_agents[id(first)][0] == "coder"

        first.fail = False
        assert adapter._close_onboarding_agents_for_profile("coder") == 1
        assert second.close_calls == 1
        assert adapter._onboarding_agent_cache == {}

    def test_capacity_cleanup_retry_keeps_evicted_owner_profile(self):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        adapter._onboarding_agent_cache_cap = 1

        class _Agent:
            def __init__(self, fail=False):
                self.fail = fail

            def close(self):
                if self.fail:
                    raise RuntimeError("child still running")

        old = _Agent(True)
        incoming = _Agent(False)
        adapter._onboarding_agent_cache[("main", "old")] = (old, 0)
        with pytest.raises(RuntimeError, match="owner cleanup failed"):
            adapter._publish_onboarding_agent(("coder", "new"), incoming)
        assert adapter._onboarding_cleanup_retry_agents[id(old)][0] == "main"

    def test_onboarding_close_outside_lock_keeps_sibling_lookup_available(self):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
        entered = threading.Event()
        release = threading.Event()

        class _BlockingAgent:
            def close(self):
                entered.set()
                release.wait()

        main_key = ("main", "expired")
        sibling_key = ("coder", "live")
        sibling = object()
        adapter._onboarding_agent_cache[main_key] = (_BlockingAgent(), 0)
        adapter._onboarding_agent_cache[sibling_key] = (sibling, time.monotonic())
        errors = []

        def expire_main():
            try:
                adapter._cached_onboarding_agent(("main", "request"))
            except Exception as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        worker = threading.Thread(target=expire_main)
        lookup_done = threading.Event()
        lookup_result = {}

        def lookup_sibling():
            lookup_result["agent"] = adapter._cached_onboarding_agent(sibling_key)
            lookup_done.set()

        worker.start()
        assert entered.wait(2)
        lookup = threading.Thread(target=lookup_sibling)
        lookup.start()
        try:
            assert lookup_done.wait(1)
            assert lookup_result["agent"] is sibling
            with pytest.raises(RuntimeError, match="cleanup already in progress"):
                adapter._close_onboarding_agents_for_profile("main")
        finally:
            release.set()
            worker.join(timeout=2)
            lookup.join(timeout=2)
        assert not worker.is_alive()
        assert not lookup.is_alive()
        assert errors == []


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
