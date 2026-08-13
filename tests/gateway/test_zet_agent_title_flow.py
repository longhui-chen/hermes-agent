"""ZetAgent 首轮回复接入 Hermes 原生 LLM 标题的流程回归测试。"""

import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import (
    billing_conversation_id,
    billing_task_title_encoded,
    billing_usage_id,
)


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
                # Read from inside the worker: this is what the title's own
                # auxiliary LLM call will stamp on its billing headers.
                "billing_usage_id": billing_usage_id(),
                "billing_conversation_id": billing_conversation_id(),
                "billing_task_title": unquote(billing_task_title_encoded()),
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
        turn_id="turn-1",
    )

    assert got == result
    assert captured["session_db"] is db
    assert captured["session_id"] == "zettlab:u1:main:s1"
    assert captured["user_message"] == "她有什么缺点？"
    assert captured["assistant_response"] == "她的主要缺点是稳定性存疑。"
    # F3: the title's 1 credit bills to the turn that triggered it (the first
    # exchange), not to an orphan card — the worker runs on a thread whose
    # ContextVars never saw the turn binding.
    assert captured["billing_usage_id"] == "zettlab:u1:main:s1:tturn-1"
    # 🔴 F5: routing / prompt-cache key stays at conversation granularity.
    assert captured["billing_conversation_id"] == "zettlab:u1:main:s1"
    assert captured["billing_task_title"] == "她有什么缺点？"
    assert captured["background"] is False
    assert stream_q.get_nowait() == (
        "__tool_progress__",
        {"type": "conversation.title", "title": "询问她的缺点"},
    )


@pytest.mark.asyncio
async def test_native_title_worker_binds_billing_keys_on_a_bare_thread(monkeypatch):
    """stream_q=None spawns a bare thread with a FRESH context — the only path
    where the explicit capture-and-rebind of the turn key is load-bearing
    (asyncio.to_thread copies the caller's context and would mask a regression
    here), so it gets its own coverage."""
    import threading

    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
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
            "session_id": "zettlab:u1:main:s4",
            "completed": True,
        },
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", AsyncMock(return_value=result))

    captured = {}
    worker_done = threading.Event()

    def fake_maybe_auto_title(
        session_db, session_id, user_message, assistant_response, history, **kwargs
    ):
        try:
            captured.update(
                {
                    "billing_usage_id": billing_usage_id(),
                    "billing_conversation_id": billing_conversation_id(),
                    "billing_task_title": unquote(billing_task_title_encoded()),
                    "title_callback": kwargs.get("title_callback"),
                }
            )
        finally:
            worker_done.set()

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", fake_maybe_auto_title)

    # No stream callbacks → _sniff_stream_q() → None → bare threading.Thread.
    await adapter._run_agent(
        user_message="她有什么缺点？",
        conversation_history=[],
        session_id="zettlab:u1:main:s4",
        agent_ref=agent_ref,
        turn_id="turn-1",
    )

    assert worker_done.wait(timeout=10), "bare-thread title worker never ran"
    # F3 on the load-bearing path: a fresh thread context has no ambient turn
    # binding at all — only the explicit snapshot can produce the turn key.
    assert captured["billing_usage_id"] == "zettlab:u1:main:s4:tturn-1"
    # 🔴 F5: routing / prompt-cache key stays at conversation granularity.
    assert captured["billing_conversation_id"] == "zettlab:u1:main:s4"
    assert captured["billing_task_title"] == "她有什么缺点？"
    assert captured["title_callback"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_message", "expected_title"),
    [
        ("帮我整理这周的照片", "帮我整理这周的照片"),
        # A routing directive is local-server plumbing; the card shows the ask.
        (
            "[Zettlab internal routing directive]\n"
            "Required flow: use the internal creation script.\n"
            "[User request]\n"
            "帮我创建一个整理照片的 Agent",
            "帮我创建一个整理照片的 Agent",
        ),
        # Synthetic protocol turns are not something a user typed: no title.
        ("[ZETTLAB:BOOTSTRAP_KICKOFF]", ""),
    ],
)
async def test_turn_title_is_bound_for_the_whole_turn(
    monkeypatch, user_message, expected_title
):
    """F4: the turn's model calls stamp this turn's user-message summary."""
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    captured = {}

    async def _fake_run_agent(*_args, **_kwargs):
        captured["task_title"] = unquote(billing_task_title_encoded())
        return ({"final_response": "", "messages": [], "completed": True}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", _fake_run_agent)

    await adapter._run_agent(
        user_message=user_message,
        conversation_history=[],
        session_id="zettlab:u1:main:s3",
        turn_id="turn-9",
    )

    assert captured["task_title"] == expected_title
    # The binding is turn-scoped: it must not outlive the request.
    assert billing_task_title_encoded() == ""


@pytest.mark.asyncio
async def test_turn_title_not_bound_without_a_turn_id(monkeypatch):
    """Clients that don't send metadata.turn_id keep a session-scoped ledger
    card; a per-turn title would just retitle that one card to whichever turn
    ran last, so the title binds only alongside a per-turn key."""
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    captured = {}

    async def _fake_run_agent(*_args, **_kwargs):
        captured["task_title"] = unquote(billing_task_title_encoded())
        return ({"final_response": "", "messages": [], "completed": True}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", _fake_run_agent)

    await adapter._run_agent(
        user_message="帮我整理这周的照片",
        conversation_history=[],
        session_id="zettlab:u1:main:s5",
    )

    assert captured["task_title"] == ""


@pytest.mark.asyncio
async def test_turn_title_prefers_the_user_authored_task_text(monkeypatch):
    """Skill invocations expand user_message into activation boilerplate; the
    ledger card must show what the user actually asked (trusted_user_message),
    not the same '[IMPORTANT: ...' prefix for every invocation of a skill."""
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    captured = {}

    async def _fake_run_agent(*_args, **_kwargs):
        captured["task_title"] = unquote(billing_task_title_encoded())
        return ({"final_response": "", "messages": [], "completed": True}, {})

    monkeypatch.setattr(APIServerAdapter, "_run_agent", _fake_run_agent)

    expanded = (
        '[IMPORTANT: The user has invoked the "gif-search" skill, indicating '
        "this task matches the skill's purpose.]\n\n找一张猫的 gif"
    )
    await adapter._run_agent(
        user_message=expanded,
        conversation_history=[],
        session_id="zettlab:u1:main:s6",
        turn_id="turn-9",
        trusted_user_message="找一张猫的 gif",
    )

    assert captured["task_title"] == "找一张猫的 gif"


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


@pytest.mark.asyncio
async def test_onboarding_uses_deterministic_title_without_llm(monkeypatch):
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))
    stream_q = queue.Queue()

    class _DB:
        def get_session_title(self, _session_id):
            return None

        def set_auto_title_if_empty(self, _session_id, title):
            assert title == "初始设置"
            return True

    agent = SimpleNamespace(
        _session_db=_DB(),
        _profile_name="onboarding",
        model="lite",
        provider="custom",
        base_url="http://model.invalid/v1",
        api_key="test-key",
        api_mode="openai_chat",
    )
    result = (
        {
            "final_response": "你好。",
            "messages": [
                {"role": "user", "content": "Frank"},
                {"role": "assistant", "content": "你好。"},
            ],
            "session_id": "zettlab:u1:onboarding:s1",
            "completed": True,
        },
        {},
    )
    monkeypatch.setattr(APIServerAdapter, "_run_agent", AsyncMock(return_value=result))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("onboarding must not call the LLM title worker")

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", forbidden)

    got = await adapter._run_agent(
        user_message="Frank",
        conversation_history=[],
        session_id="zettlab:u1:onboarding:s1",
        stream_delta_callback=lambda delta: stream_q.put(delta),
        agent_ref=[agent],
    )

    assert got == result
    assert stream_q.get_nowait() == (
        "__tool_progress__",
        {"type": "conversation.title", "title": "初始设置"},
    )


def test_title_input_strips_agent_creator_routing_directive():
    routed = (
        "[Zettlab internal routing directive]\n"
        "Required flow: use the internal creation script.\n"
        "[User request]\n"
        "帮我创建一个整理照片的 Agent"
    )

    assert ZetAgentAdapter._title_user_message(routed) == "帮我创建一个整理照片的 Agent"
