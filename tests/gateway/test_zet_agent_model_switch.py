import types
from collections import OrderedDict

import pytest
import yaml
import queue

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
async def test_session_model_switch_persists_override_no_note(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter._check_auth = lambda request: None

    evicted = []
    gw = types.SimpleNamespace(
        _session_model_overrides={},
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
    # Override persisted + cached agent evicted. The identity note is no longer
    # pushed here — it's injected at the session's next turn by _run_agent's
    # open-time effective-model compare.
    assert gw._session_model_overrides[session_id]["model"] == "deepseek-v4"
    assert evicted == [session_id]
    assert not hasattr(gw, "_pending_model_notes")


@pytest.mark.asyncio
async def test_agent_model_switch_writes_config_only(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"model": {"default": "old-model"}}), encoding="utf-8"
    )

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    # Agent-level switch only writes config.yaml now; no broadcast, so it
    # doesn't need gateway_runner at all.
    adapter.gateway_runner = None

    resp = await adapter._handle_model_switch(
        _FakeRequest({"model": "glm-5", "provider": "custom"})
    )

    assert resp.status == 200
    cfg = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"]["default"] == "glm-5"


def test_status_callback_forwards_context_compaction_to_tool_progress_lane():
    stream_q = queue.Queue()
    cb = ZetAgentAdapter._make_status_cb(stream_q)

    cb("context.compaction", {
        "state": "succeeded",
        "message": "上下文压缩成功",
        "old_session_id": "old",
        "new_session_id": "new",
    })

    tag, payload = stream_q.get_nowait()
    assert tag == "__tool_progress__"
    assert payload == {
        "type": "context.compaction",
        "state": "succeeded",
        "message": "上下文压缩成功",
        "old_session_id": "old",
        "new_session_id": "new",
    }


def test_status_callback_ignores_unstructured_status():
    stream_q = queue.Queue()
    cb = ZetAgentAdapter._make_status_cb(stream_q)

    cb("lifecycle", "Compacting context")

    assert stream_q.empty()


def test_status_callback_preserves_existing_callback():
    stream_q = queue.Queue()
    seen = []
    cb = ZetAgentAdapter._make_status_cb(stream_q, lambda kind, payload=None: seen.append((kind, payload)))

    cb("lifecycle", "Compacting context")
    cb("context.compaction", {"state": "started"})

    assert seen == [("lifecycle", "Compacting context")]
    assert stream_q.get_nowait() == ("__tool_progress__", {
        "type": "context.compaction",
        "state": "started",
    })
    assert getattr(cb, "_hermes_accepts_structured_status") is True


def _seen_adapter(monkeypatch, *, config_model, seen, override=None):
    """Build a ZetAgentAdapter wired for open-time model-compare tests:
    a config default model, a pre-seeded seen-map, an optional session
    override, and a no-op persistence so tests don't touch disk."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: config_model)
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    adapter.gateway_runner = types.SimpleNamespace(
        _session_model_overrides=dict(override or {})
    )
    adapter._seen_models = OrderedDict(seen)
    adapter._seen_loaded = True
    monkeypatch.setattr(adapter, "_save_seen_models", lambda: None)
    return adapter


async def _capture_run_agent(monkeypatch, adapter, **kwargs):
    from gateway.platforms.api_server import APIServerAdapter

    captured = {}

    async def fake_super(self, **kw):
        captured.update(kw)
        return ({}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_super)
    await adapter._run_agent(conversation_history=[], **kwargs)
    return captured


@pytest.mark.asyncio
async def test_run_agent_injects_note_on_effective_model_change(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={"sess-1": "deepseek-v4"})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    assert captured["user_message"].startswith("[Note: the model has changed and is now glm-5.1")
    assert captured["user_message"].endswith("hi")
    # last-seen updated to the new effective model.
    assert adapter._seen_models["sess-1"] == "glm-5.1"


@pytest.mark.asyncio
async def test_run_agent_no_note_when_model_unchanged(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={"sess-1": "glm-5.1"})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    assert captured["user_message"] == "hi"


@pytest.mark.asyncio
async def test_run_agent_new_session_records_baseline_no_note(monkeypatch):
    adapter = _seen_adapter(monkeypatch, config_model="glm-5.1", seen={})
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="new-sess")
    # New session: no note (its system prompt is already built with the
    # current model), but baseline recorded so a later switch is detected.
    assert captured["user_message"] == "hi"
    assert adapter._seen_models["new-sess"] == "glm-5.1"


@pytest.mark.asyncio
async def test_run_agent_effective_model_prefers_override(monkeypatch):
    adapter = _seen_adapter(
        monkeypatch,
        config_model="config-default",
        seen={"sess-1": "config-default"},
        override={"sess-1": {"model": "override-model"}},
    )
    captured = await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id="sess-1")
    # Effective model = override (not config default) → note announces it.
    assert captured["user_message"].startswith("[Note: the model has changed and is now override-model")


@pytest.mark.asyncio
async def test_run_agent_evicts_oldest_seen_over_cap(monkeypatch):
    # Bound the map so it can't grow without limit (Hard Rule 第 1 条).
    monkeypatch.setattr(zet_agent, "_SEEN_MODELS_CAP", 2)
    adapter = _seen_adapter(monkeypatch, config_model="m1", seen={})
    for sid in ("s1", "s2", "s3"):
        await _capture_run_agent(monkeypatch, adapter, user_message="hi", session_id=sid)
    # cap=2 → oldest (s1) evicted LRU-style.
    assert len(adapter._seen_models) == 2
    assert set(adapter._seen_models) == {"s2", "s3"}
