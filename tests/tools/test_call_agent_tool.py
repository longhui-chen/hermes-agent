"""call_agent tool contract (zettlab agent-to-agent calls).

The tool is a thin loopback POST — these tests pin the wire payload, the
envelope parsing, the env schema-gate, and failure shaping. Guardrails
(cycle/depth/quota/authz) are SERVER-side and covered in local-server.
"""

import json
import urllib.request

import pytest

import tools.call_agent_tool as cat

_URL = "http://127.0.0.1:9091/api/v1/internal/agent-call"


class _Resp:
    def __init__(self, body: dict, status: int = 200):
        self._body = json.dumps(body).encode("utf-8")
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def call_env(monkeypatch):
    monkeypatch.setenv("ZET_AGENT_CALL_URL", _URL)
    monkeypatch.setenv("ZET_AGENT_ID", "agentA")


def test_schema_gate_follows_env(monkeypatch):
    monkeypatch.delenv("ZET_AGENT_CALL_URL", raising=False)
    assert cat.check_call_agent_requirements() is False
    monkeypatch.setenv("ZET_AGENT_CALL_URL", _URL)
    assert cat.check_call_agent_requirements() is True


def test_requires_agent_and_message(call_env):
    assert "error" in json.loads(cat.call_agent(agent="", message="hi"))
    assert "error" in json.loads(cat.call_agent(agent="B", message=""))


def test_posts_payload_and_parses_reply(call_env, monkeypatch):
    from tools.approval import reset_current_session_key, set_current_session_key

    token = set_current_session_key("agent:main:zet_agent:dm:zettlab:u1:agentA:7")
    seen = []

    def _fake_urlopen(req, timeout=None):
        seen.append((req, timeout))
        return _Resp({
            "code": 200,
            "data": {
                "reply": "B 的回答",
                "agent_id": "agentB",
                "agent_name": "B",
                "session_id": "zettlab:u1:agentB:a2a-agentA",
                "turn_id": "t_a2a_1",
            },
        })

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    try:
        out = json.loads(cat.call_agent(agent="B", message="帮我看看这个"))
    finally:
        reset_current_session_key(token)

    assert out["reply"] == "B 的回答"
    assert out["agent_id"] == "agentB"
    assert out["session_id"] == "zettlab:u1:agentB:a2a-agentA"
    assert "error" not in out

    req, timeout = seen[0]
    assert req.full_url == _URL
    assert timeout == cat._CALL_TIMEOUT_SECONDS
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["caller_agent_id"] == "agentA"
    assert payload["caller_session_id"] == "zettlab:u1:agentA:7"
    assert payload["callee"] == "B"
    assert payload["message"] == "帮我看看这个"


def test_refusal_reason_surfaces_to_model(call_env, monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _Resp(
            {"code": 403, "data": {"error": "call cycle rejected", "reason": "cycle"}}
        ),
    )
    out = json.loads(cat.call_agent(agent="B", message="x"))
    assert out["error"] == "call cycle rejected"
    assert out["reason"] == "cycle"
    assert "reply" not in out


def test_transport_failure_returns_error_not_raise(call_env, monkeypatch):
    def _boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    out = json.loads(cat.call_agent(agent="B", message="x"))
    assert "connection refused" in out["error"]


def test_no_reply_and_no_error_normalised(call_env, monkeypatch):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _Resp({"code": 200, "data": {}}),
    )
    out = json.loads(cat.call_agent(agent="B", message="x"))
    assert out["error"] == "agent call returned no reply"


def test_own_session_id_shapes(call_env):
    """真机两种 session_key 形状（镜像 LS hermesSessionIDFromSessionKey）。"""
    from tools.approval import reset_current_session_key, set_current_session_key

    # cron 形态：agent:main:zet_agent:<chat_type>:<chat_id>（chat_id 含冒号）
    token = set_current_session_key("agent:main:zet_agent:dm:zettlab:u1:agentA:7")
    try:
        assert cat._own_session_id() == "zettlab:u1:agentA:7"
    finally:
        reset_current_session_key(token)

    # 真机聊天形态：zet_agent 原样绑定裸 LS 会话 id（无 :zet_agent: 中缀）——
    # 修复前这里返回 ""，导致 call_agent 一律 LS 400。
    token = set_current_session_key("zettlab:local-dev:main:yt-qnbPfqzR-")
    try:
        assert cat._own_session_id() == "zettlab:local-dev:main:yt-qnbPfqzR-"
    finally:
        reset_current_session_key(token)

    # CLI / 未知形状：fail-closed 空
    token = set_current_session_key("default")
    try:
        assert cat._own_session_id() == ""
    finally:
        reset_current_session_key(token)
