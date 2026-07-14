"""POST /v1/sessions/{sid}/steer + steer_dropped push (ZET chat-queue-steer).

Covers the gateway glue only: HTTP handler -> AIAgent.steer() and the
unconsumed-steer receipt push in _run_agent. AIAgent.steer()'s own
semantics are already covered by tests/run_agent/test_steer.py.
"""

import queue

import pytest

from gateway.config import PlatformConfig
import gateway.platforms.zet_agent as zet_agent
from gateway.platforms.zet_agent import ZetAgentAdapter


class _FakeRequest:
    def __init__(self, body, match_info=None, auth="Bearer test-key"):
        self._body = body
        self.headers = {"Authorization": auth} if auth else {}
        self.match_info = match_info or {}
        # _check_auth's rejection log reads these request attributes.
        self.method = "POST"
        self.path_qs = "/v1/sessions/test/steer"
        self.remote = "127.0.0.1"
        self.transport = None

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status


class _FakeWeb:
    @staticmethod
    def json_response(payload, status=200):
        return _FakeResponse(payload, status)


class _FakeAgent:
    def __init__(self, accept=True):
        self.accept = accept
        self.steered = []

    def steer(self, text):
        self.steered.append(text)
        return self.accept


def _adapter(monkeypatch):
    monkeypatch.setattr(zet_agent, "web", _FakeWeb)
    return ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))


@pytest.mark.asyncio
async def test_steer_hits_active_agent_via_scoped_key(monkeypatch):
    # 生产路径：注册用 _active_turn_key（{hermes_home}|{sid}，goal-loop codex
    # P1 的 scoped key）——steer 查表必须命中它，否则真机上永远 not_running
    # （2026-07-13 dev rig 实测踩过：裸 sid 查 scoped 表 → steer.dropped 连发）。
    adapter = _adapter(monkeypatch)
    agent = _FakeAgent()
    adapter._register_active_session_turn("s1", [agent], None)

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "把 PDF 也算上"}, match_info={"session_id": "s1"})
    )

    assert resp.status == 200
    assert resp.payload == {"session_id": "s1", "status": "steering", "accepted": True}
    assert agent.steered == ["把 PDF 也算上"]


@pytest.mark.asyncio
async def test_steer_hits_active_agent_via_bare_sid_fallback(monkeypatch):
    adapter = _adapter(monkeypatch)
    agent = _FakeAgent()
    adapter._active_session_agents["s1"] = [agent]

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "把 PDF 也算上"}, match_info={"session_id": "s1"})
    )

    assert resp.status == 200
    assert resp.payload == {"session_id": "s1", "status": "steering", "accepted": True}
    assert agent.steered == ["把 PDF 也算上"]


@pytest.mark.asyncio
async def test_steer_not_running_is_idempotent(monkeypatch):
    adapter = _adapter(monkeypatch)

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "hello"}, match_info={"session_id": "nope"})
    )

    assert resp.status == 200
    assert resp.payload == {
        "session_id": "nope",
        "status": "not_running",
        "accepted": False,
    }


@pytest.mark.asyncio
async def test_steer_agent_ref_not_yet_filled_is_not_running(monkeypatch):
    # agent_ref is registered as [None] before the agent is constructed;
    # a steer racing that window must degrade to not_running, not crash.
    adapter = _adapter(monkeypatch)
    adapter._active_session_agents["s1"] = [None]

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "hi"}, match_info={"session_id": "s1"})
    )

    assert resp.status == 200
    assert resp.payload["status"] == "not_running"
    assert resp.payload["accepted"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": "   "},
        {"text": 42},
        "not-a-dict",
        ValueError("bad json"),
    ],
)
async def test_steer_invalid_body_is_400(monkeypatch, body):
    adapter = _adapter(monkeypatch)
    adapter._active_session_agents["s1"] = [_FakeAgent()]

    resp = await adapter._handle_session_steer(
        _FakeRequest(body, match_info={"session_id": "s1"})
    )

    assert resp.status == 400


@pytest.mark.asyncio
async def test_steer_requires_auth(monkeypatch):
    adapter = _adapter(monkeypatch)

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "hi"}, match_info={"session_id": "s1"}, auth="Bearer wrong")
    )

    assert resp.status == 401


@pytest.mark.asyncio
async def test_steer_agent_exception_reports_rejected(monkeypatch):
    adapter = _adapter(monkeypatch)

    class _Boom:
        def steer(self, text):
            raise RuntimeError("boom")

    adapter._active_session_agents["s1"] = [_Boom()]

    resp = await adapter._handle_session_steer(
        _FakeRequest({"text": "hi"}, match_info={"session_id": "s1"})
    )

    assert resp.status == 200
    assert resp.payload == {"session_id": "s1", "status": "rejected", "accepted": False}


# ---------------------------------------------------------------------------
# steer_dropped push (_push_steer_dropped_if_any)
# ---------------------------------------------------------------------------


def test_push_steer_dropped_emits_progress_event():
    q = queue.Queue()
    ZetAgentAdapter._push_steer_dropped_if_any(
        q, ({"final_response": "done", "pending_steer": "漏掉的插话"}, None)
    )

    kind, payload = q.get_nowait()
    assert kind == "__tool_progress__"
    assert payload == {"type": "steer_dropped", "text": "漏掉的插话"}
    assert q.empty()


@pytest.mark.parametrize(
    "run_result",
    [
        ({"final_response": "done"}, None),  # no pending_steer key
        ({"pending_steer": ""}, None),  # empty
        ({"pending_steer": "   "}, None),  # whitespace only
        (None, None),  # result dict missing
        "weird",  # not a tuple/dict at all
    ],
)
def test_push_steer_dropped_skips_when_nothing_pending(run_result):
    q = queue.Queue()
    ZetAgentAdapter._push_steer_dropped_if_any(q, run_result)
    assert q.empty()


def test_push_steer_dropped_tolerates_missing_queue():
    # stream_q can be None when sniffing failed; must not raise.
    ZetAgentAdapter._push_steer_dropped_if_any(
        None, ({"pending_steer": "text"}, None)
    )
