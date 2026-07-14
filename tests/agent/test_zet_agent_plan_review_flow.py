from types import SimpleNamespace

from agent.conversation_loop import (
    _apply_forced_present_plan_tool_choice,
    _drop_trailing_plan_protocol_messages,
    _enforce_single_plan_interaction_tool_call,
    _plan_mode_interaction_error,
    _should_force_present_plan_tool_choice,
)
from agent.tool_executor import _zet_agent_plan_mode_block_message
from run_agent import AIAgent


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def test_plan_flow_requires_clarify_before_present_plan_and_hides_side_effects():
    agent = SimpleNamespace(
        platform="zet_agent",
        api_mode="chat_completions",
        model="gpt-4o",
        provider="openai",
        base_url="https://api.openai.com/v1",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        _zet_agent_force_present_plan_disable_thinking=False,
        _zet_agent_plan_omit_thinking_disable=False,
    )

    first_request = {
        "messages": [
            {"role": "system", "content": "base"},
            {
                "role": "user",
                "content": (
                    "你帮我列个减肥的 plan\n\n"
                    "App plan mode is enabled. Use present_plan."
                ),
            },
        ],
        "tools": [_tool("clarify"), _tool("present_plan"), _tool("todo"), _tool("terminal")]
    }
    assert _apply_forced_present_plan_tool_choice(agent, first_request)
    assert [tool["function"]["name"] for tool in first_request["tools"]] == [
        "clarify",
        "present_plan",
    ]
    first_system_prompt = first_request["messages"][0]["content"]
    assert "takes precedence" in first_system_prompt
    assert "MUST call `clarify`" in first_system_prompt
    assert "Never include collecting required user information" in first_system_prompt
    assert "personalized health, diet, or fitness plans" in first_system_prompt

    # clarify 返回后仍处于同一个 Plan turn，下一次请求继续受同一门禁保护。
    after_clarify = {
        "messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "你帮我列个减肥的 plan"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "clarify"}}],
            },
            {
                "role": "tool",
                "content": "身高 175cm，体重 80kg，目标三个月减到 72kg",
            },
        ],
        "tools": [_tool("clarify"), _tool("present_plan"), _tool("write_file"), _tool("todo")]
    }
    assert _apply_forced_present_plan_tool_choice(agent, after_clarify)
    assert after_clarify["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in after_clarify["tools"]] == [
        "clarify",
        "present_plan",
    ]
    assert "After each clarify response" in after_clarify["messages"][0]["content"]

    agent._zet_agent_plan_presented = True
    settled_request = {"tools": [_tool("todo"), _tool("terminal")]}
    assert not _apply_forced_present_plan_tool_choice(agent, settled_request)
    assert [tool["function"]["name"] for tool in settled_request["tools"]] == [
        "todo",
        "terminal",
    ]


def test_plan_mode_filters_side_effects_even_when_present_plan_is_misconfigured():
    agent = SimpleNamespace(
        platform="zet_agent",
        api_mode="chat_completions",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        _zet_agent_force_present_plan_disable_thinking=False,
        _zet_agent_plan_omit_thinking_disable=False,
    )
    request = {"tools": [_tool("clarify"), _tool("todo"), _tool("terminal")]}

    assert not _apply_forced_present_plan_tool_choice(agent, request)
    assert [tool["function"]["name"] for tool in request["tools"]] == ["clarify"]
    assert request["parallel_tool_calls"] is False


def test_plan_mode_filters_side_effects_before_rejecting_unsupported_api_mode():
    agent = SimpleNamespace(
        platform="zet_agent",
        api_mode="anthropic_messages",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        _zet_agent_force_present_plan_disable_thinking=False,
        _zet_agent_plan_omit_thinking_disable=False,
    )
    request = {"tools": [_tool("clarify"), _tool("present_plan"), _tool("write_file")]}

    assert not _apply_forced_present_plan_tool_choice(agent, request)
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "clarify",
        "present_plan",
    ]


def test_valid_plan_tool_call_drops_private_protocol_retry_messages():
    messages = [
        {"role": "user", "content": "原始需求"},
        {"role": "assistant", "content": "纯文本计划", "_plan_protocol_synthetic": True},
        {"role": "user", "content": "必须调用工具", "_plan_protocol_synthetic": True},
    ]

    _drop_trailing_plan_protocol_messages(messages)

    assert messages == [{"role": "user", "content": "原始需求"}]


def test_plan_mode_execution_guard_blocks_side_effect_tool_calls():
    agent = SimpleNamespace(platform="zet_agent", _zet_agent_plan_mode_active=True)

    for tool_name in ("terminal", "write_file", "todo", "connector", "skill_manage"):
        assert _zet_agent_plan_mode_block_message(agent, tool_name, {}) is not None

    assert _zet_agent_plan_mode_block_message(agent, "clarify", {}) is None
    assert _zet_agent_plan_mode_block_message(agent, "present_plan", {}) is None


def test_cancelled_plan_returns_to_regular_chat_tools():
    agent = SimpleNamespace(
        platform="zet_agent",
        valid_tool_names={"clarify", "present_plan", "todo", "terminal"},
        _zet_agent_response_mode="",
        _zet_agent_plan_ack={
            "status": "cancelled",
            "revision_requested": False,
        },
        _zet_agent_plan_mode_active=False,
        _zet_agent_plan_presented=False,
    )

    cancel_message = "取消计划「身高170cm → 62kg 科学减脂计划」，不要执行。"
    agent._zet_agent_plan_mode_active = _should_force_present_plan_tool_choice(
        agent,
        cancel_message,
    )
    request = {
        "tools": [
            _tool("clarify"),
            _tool("present_plan"),
            _tool("todo"),
            _tool("terminal"),
        ],
    }

    assert agent._zet_agent_plan_mode_active is False
    assert not _apply_forced_present_plan_tool_choice(agent, request)
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "clarify",
        "present_plan",
        "todo",
        "terminal",
    ]


def test_text_only_plan_request_stays_regular_without_response_mode():
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_response_mode="",
        _zet_agent_plan_mode_active=False,
        _zet_agent_plan_presented=False,
    )
    user_message = (
        "计划模式；先别执行；不要执行；等我确认；确认后；present_plan"
    )

    agent._zet_agent_plan_mode_active = _should_force_present_plan_tool_choice(
        agent,
        user_message,
    )
    request = {
        "tools": [
            _tool("clarify"),
            _tool("present_plan"),
            _tool("todo"),
            _tool("terminal"),
        ],
    }

    assert agent._zet_agent_plan_mode_active is False
    assert not _apply_forced_present_plan_tool_choice(agent, request)
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "clarify",
        "present_plan",
        "todo",
        "terminal",
    ]


def test_plan_mode_suppresses_provisional_plain_text_streaming():
    streamed = []
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        stream_delta_callback=streamed.append,
        _stream_callback=None,
    )
    agent._should_suppress_plan_stream_text = lambda: AIAgent._should_suppress_plan_stream_text(agent)

    AIAgent._fire_stream_delta(agent, "这段纯文本计划不能提前进入 SSE")

    assert streamed == []


def test_plan_mode_requires_interactive_callbacks():
    missing = SimpleNamespace(
        _zet_agent_plan_mode_active=True,
        clarify_callback=None,
        plan_emit_callback=None,
    )
    ready = SimpleNamespace(
        _zet_agent_plan_mode_active=True,
        clarify_callback=lambda *_: None,
        plan_emit_callback=lambda *_: None,
    )

    assert "stream=true" in _plan_mode_interaction_error(missing)
    assert _plan_mode_interaction_error(ready) is None


def test_parallel_plan_calls_prefer_clarify_and_drop_stale_present_plan():
    assistant = SimpleNamespace(tool_calls=[
        SimpleNamespace(function=SimpleNamespace(name="present_plan")),
        SimpleNamespace(function=SimpleNamespace(name="clarify")),
    ])
    agent = SimpleNamespace(_zet_agent_plan_mode_active=True)

    assert _enforce_single_plan_interaction_tool_call(agent, assistant)
    assert [call.function.name for call in assistant.tool_calls] == ["clarify"]


def test_parallel_present_plan_calls_keep_only_the_first_call():
    first = SimpleNamespace(function=SimpleNamespace(name="present_plan"), id="first")
    second = SimpleNamespace(function=SimpleNamespace(name="present_plan"), id="second")
    assistant = SimpleNamespace(tool_calls=[first, second])
    agent = SimpleNamespace(_zet_agent_plan_mode_active=True)

    assert _enforce_single_plan_interaction_tool_call(agent, assistant)
    assert assistant.tool_calls == [first]
