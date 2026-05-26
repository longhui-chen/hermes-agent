import types

import pytest
import yaml

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeRequest:
    def __init__(self, body, match_info=None):
        self._body = body
        self.headers = {"Authorization": "Bearer test-key"}
        self.match_info = match_info or {}

    async def json(self):
        return self._body


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


@pytest.mark.asyncio
async def test_model_switch_writes_api_mode_and_context_length(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "default": "old",
                    "api_mode": "anthropic_messages",
                    "context_length": 128000,
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    resp = await adapter._handle_model_switch(
        _FakeRequest(
            {
                "model": "glm-5",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                "api_key": "local-ai-proxy",
                "api_mode": "openai_chat",
                "context_length": 1000000,
            }
        )
    )
    assert resp.status == 200

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"
    assert cfg["model"]["api_mode"] == "openai_chat"
    assert cfg["model"]["context_length"] == 1000000


@pytest.mark.asyncio
async def test_model_switch_clears_stale_api_mode_and_context_length(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "default": "old",
                    "api_mode": "anthropic_messages",
                    "context_length": 128000,
                }
            }
        ),
        encoding="utf-8",
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    resp = await adapter._handle_model_switch(
        _FakeRequest(
            {
                "model": "glm-5",
                "provider": "custom",
                "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
                "api_key": "local-ai-proxy",
                "api_mode": "",
                "context_length": 0,
            }
        )
    )
    assert resp.status == 200

    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"
    assert "api_mode" not in cfg["model"]
    assert "context_length" not in cfg["model"]


@pytest.mark.asyncio
async def test_session_model_switch_queues_pending_note(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    evicted = []
    gw = types.SimpleNamespace(
        _session_model_overrides={},
        _pending_model_notes={},
        _evict_cached_agent=lambda sid: evicted.append(sid),
    )
    adapter.gateway_runner = gw

    session_id = "zettlab:user1:agent-1:42"
    resp = await adapter._handle_session_model_switch(
        _FakeRequest(
            {"model": "deepseek-v4", "provider": "custom"},
            match_info={"session_id": session_id},
        )
    )

    assert resp.status == 200
    # Runtime override stored so the next turn resolves the new model.
    assert gw._session_model_overrides[session_id]["model"] == "deepseek-v4"
    # Cached agent evicted so the next turn rebuilds with the new model.
    assert evicted == [session_id]
    # One-shot note queued for this session's next user message.
    note = gw._pending_model_notes[session_id]
    assert "deepseek-v4" in note
    assert "self-identification" in note


@pytest.mark.asyncio
async def test_session_model_switch_creates_pending_notes_when_missing(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    # gateway_runner has no pre-existing _pending_model_notes attribute.
    gw = types.SimpleNamespace(
        _session_model_overrides={},
        _evict_cached_agent=lambda sid: None,
    )
    adapter.gateway_runner = gw

    session_id = "zettlab:user1:agent-1:7"
    resp = await adapter._handle_session_model_switch(
        _FakeRequest(
            {"model": "glm-5"},
            match_info={"session_id": session_id},
        )
    )

    assert resp.status == 200
    assert "glm-5" in gw._pending_model_notes[session_id]
