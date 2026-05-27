import pytest
import yaml
import queue

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeRequest:
    def __init__(self, body):
        self._body = body
        self.headers = {"Authorization": "Bearer test-key"}

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
