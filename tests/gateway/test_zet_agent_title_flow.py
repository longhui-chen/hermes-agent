"""ZetAgent 首轮回复接入 Hermes 原生 LLM 标题的流程回归测试。"""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import billing_task_id


@pytest.mark.asyncio
async def test_native_title_is_emitted_before_stream_task_finishes(monkeypatch):
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    stream_q = queue.Queue()

    def on_delta(delta):
        stream_q.put(delta)

    db = object()
    agent = SimpleNamespace(
        _session_db=db,
        model="test-model",
        provider="test-provider",
        base_url="http://model.invalid/v1",
        api_key="test-key",
        api_mode="openai_chat",
    )
    agent_ref = [agent]
    result = (
        {
            "final_response": "她的主要缺点是稳定性存疑。",
            "messages": [
                {"role": "user", "content": "她有什么缺点？"},
                {"role": "assistant", "content": "她的主要缺点是稳定性存疑。"},
            ],
            "session_id": "zettlab:u1:main:s1",
            "completed": True,
        },
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", AsyncMock(return_value=result))

    captured = {}

    def fake_maybe_auto_title(
        session_db,
        session_id,
        user_message,
        assistant_response,
        history,
        **kwargs,
    ):
        captured.update(
            {
                "session_db": session_db,
                "session_id": session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "history": history,
                "billing_task_id": billing_task_id(),
                **kwargs,
            }
        )
        kwargs["title_callback"]("询问她的缺点")

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", fake_maybe_auto_title)

    got = await adapter._run_agent(
        user_message="她有什么缺点？",
        conversation_history=[],
        session_id="zettlab:u1:main:s1",
        stream_delta_callback=on_delta,
        agent_ref=agent_ref,
    )

    assert got == result
    assert captured["session_db"] is db
    assert captured["session_id"] == "zettlab:u1:main:s1"
    assert captured["user_message"] == "她有什么缺点？"
    assert captured["assistant_response"] == "她的主要缺点是稳定性存疑。"
    assert captured["billing_task_id"] == "zettlab:u1:main:s1"
    assert captured["background"] is False
    assert stream_q.get_nowait() == (
        "__tool_progress__",
        {"type": "conversation.title", "title": "询问她的缺点"},
    )


@pytest.mark.asyncio
async def test_native_title_skips_zettlab_synthetic_turns(monkeypatch):
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    result = (
        {
            "final_response": "开始初始化",
            "messages": [],
            "session_id": "zettlab:u1:main:s2",
            "completed": True,
        },
        {},
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", AsyncMock(return_value=result))
    title_call = AsyncMock()
    monkeypatch.setattr(adapter, "_emit_native_session_title", title_call)

    got = await adapter._run_agent(
        user_message="[ZETTLAB:BOOTSTRAP_KICKOFF]",
        conversation_history=[],
        session_id="zettlab:u1:main:s2",
    )

    assert got == result
    title_call.assert_not_awaited()


def test_title_input_strips_agent_creator_routing_directive():
    routed = (
        "[Zettlab internal routing directive]\n"
        "Required flow: use the internal creation script.\n"
        "[User request]\n"
        "帮我创建一个整理照片的 Agent"
    )

    assert ZetAgentAdapter._title_user_message(routed) == "帮我创建一个整理照片的 Agent"
