from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _plan_agent(tool_names: tuple[str, ...]) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[_tool(name) for name in tool_names]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai",
            model="test-model",
            platform="zet_agent",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            clarify_callback=lambda *_: None,
        )
    agent.client = MagicMock()
    agent.plan_emit_callback = lambda *_: None
    agent._zet_agent_response_mode = "plan"
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    # 这些用例覆盖 manual 计划评审流程（present_plan 后停下等确认）；显式关掉
    # 自动执行，与新默认 auto-execute 区分开，否则 run_conversation 不会在
    # present_plan 后结束而是继续循环。
    agent._zet_agent_plan_auto_execute = False
    return agent


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
        model="test-model",
    )


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason="stop",
        )],
        usage=None,
        model="test-model",
    )


def _plan_tool_response(*, content: str, reasoning: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                tool_calls=[SimpleNamespace(
                    id="call-plan",
                    type="function",
                    function=SimpleNamespace(
                        name="present_plan",
                        arguments=(
                            '{"title":"减脂计划","groups":['
                            '{"icon":"🏃","label":"训练","items":["每周跑步三次"]}'
                            "]}"
                        ),
                    ),
                )],
                reasoning_content=reasoning,
            ),
            finish_reason="tool_calls",
        )],
        usage=None,
        model="test-model",
    )


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


def test_plan_mode_rejects_missing_clarify_instead_of_forcing_present_plan():
    agent = SimpleNamespace(
        _zet_agent_plan_mode_active=True,
        clarify_callback=lambda *_: None,
        plan_emit_callback=lambda *_: None,
        valid_tool_names={"present_plan"},
    )

    error = _plan_mode_interaction_error(agent)

    assert error is not None
    assert "clarify" in error

    request = {"tools": [_tool("present_plan"), _tool("todo")]}
    configured_agent = SimpleNamespace(
        platform="zet_agent",
        api_mode="chat_completions",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        _zet_agent_force_present_plan_disable_thinking=False,
        _zet_agent_plan_omit_thinking_disable=False,
    )
    assert not _apply_forced_present_plan_tool_choice(configured_agent, request)
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "present_plan",
    ]
    assert "tool_choice" not in request


def test_plan_mode_missing_clarify_fails_before_the_model_call():
    agent = _plan_agent(("present_plan",))

    with patch.object(agent, "_persist_session"):
        result = agent.run_conversation("帮我制定减脂计划")

    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert "clarify" in result["error"]
    agent.client.chat.completions.create.assert_not_called()


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
        _stream_writer_superseded=lambda: False,
    )
    agent._should_suppress_plan_stream_text = lambda: AIAgent._should_suppress_plan_stream_text(agent)

    AIAgent._fire_stream_delta(agent, "这段纯文本计划不能提前进入 SSE")

    assert streamed == []


def test_plan_mode_suppresses_provisional_reasoning_streaming():
    streamed = []
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        reasoning_callback=streamed.append,
        _stream_writer_superseded=lambda: False,
    )
    agent._should_suppress_plan_stream_text = lambda: AIAgent._should_suppress_plan_stream_text(agent)

    AIAgent._fire_reasoning_delta(agent, "这段 reasoning 草案不能提前进入 SSE")

    assert streamed == []


def test_plan_mode_sanitizes_tool_response_history_at_the_shared_builder():
    agent = _plan_agent(("clarify", "present_plan"))
    agent._zet_agent_plan_mode_active = True
    agent._zet_agent_plan_presented = False
    assistant = SimpleNamespace(
        content="未经审核的草案",
        tool_calls=[SimpleNamespace(
            id="call-unknown",
            type="function",
            function=SimpleNamespace(name="unknown_tool", arguments="{}"),
        )],
        reasoning_content="先构造完整计划",
    )

    message = agent._build_assistant_message(assistant, "tool_calls")

    assert message["content"] == ""
    assert message["reasoning"] is None
    assert message["reasoning_content"] == " "


def test_plan_tool_flow_drops_sibling_content_and_visible_reasoning_before_history():
    agent = _plan_agent(("clarify", "present_plan"))
    response = _plan_tool_response(
        content="这是未经确认的纯文本计划",
        reasoning="先草拟一份完整计划再调用工具",
    )
    flushed = []

    def execute_plan(_assistant_message, messages, *_args):
        agent._zet_agent_plan_presented = True
        messages.append({
            "role": "tool",
            "name": "present_plan",
            "tool_call_id": "call-plan",
            "content": '{"success":true}',
        })

    with (
        patch.object(agent, "_interruptible_api_call", return_value=response),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_plan),
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=lambda messages, _history=None: flushed.append(
                [message.copy() for message in messages]
            ),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("帮我制定减脂计划")

    assistant = next(
        message for message in result["messages"]
        if message.get("role") == "assistant" and message.get("tool_calls")
    )
    assert assistant["content"] == ""
    assert assistant["reasoning"] is None
    assert assistant["reasoning_content"] == " "
    assert flushed
    assert all(
        "未经确认" not in str(message)
        and "草拟一份完整计划" not in str(message)
        for snapshot in flushed
        for message in snapshot
    )


def test_present_plan_without_sse_callback_continues_to_text_response_flow():
    agent = _plan_agent(("present_plan",))
    agent._zet_agent_response_mode = ""
    agent.plan_emit_callback = None
    response = _plan_tool_response(content="", reasoning="")

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[
                response,
                _text_response("📋 减脂计划\n- 每周跑步三次"),
            ],
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("先给我一个减脂计划")

    assert result["failed"] is False
    assert "📋 减脂计划" in result["final_response"]
    assert "每周跑步三次" in result["final_response"]


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
        valid_tool_names={"clarify", "present_plan"},
    )

    assert "stream=true" in _plan_mode_interaction_error(missing)
    assert _plan_mode_interaction_error(ready) is None


def test_empty_plan_responses_retry_twice_then_return_protocol_error():
    agent = _plan_agent(("clarify", "present_plan"))
    empty = _empty_response()

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=[empty, empty, empty]),
        patch.object(agent, "_persist_session") as persist_session,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("帮我制定减脂计划")

    assert result["failed"] is True
    assert result["completed"] is False
    assert result["api_calls"] == 3
    assert "Plan mode protocol error" in result["error"]
    assert all(
        not message.get("_plan_protocol_synthetic")
        for message in result["messages"]
    )
    assert persist_session.call_count >= 1
    for call in persist_session.call_args_list:
        persisted_messages = call.args[0]
        assert all(
            not message.get("_plan_protocol_synthetic")
            for message in persisted_messages
        )


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


def test_manual_present_plan_outside_plan_mode_drops_parallel_side_effect_unit():
    terminal = SimpleNamespace(
        function=SimpleNamespace(name="terminal"),
        id="terminal",
    )
    present_plan = SimpleNamespace(
        function=SimpleNamespace(name="present_plan"),
        id="present-plan",
    )
    assistant = SimpleNamespace(tool_calls=[terminal, present_plan])
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_plan_mode_active=False,
        _zet_agent_plan_auto_execute=False,
    )

    assert _enforce_single_plan_interaction_tool_call(agent, assistant)
    assert assistant.tool_calls == [present_plan]


def test_manual_present_plan_outside_plan_mode_drops_parallel_side_effect_flow():
    agent = _plan_agent(("present_plan", "terminal"))
    agent._zet_agent_response_mode = ""
    response = _plan_tool_response(
        content="先展示计划",
        reasoning="展示后等待确认",
    )
    response.choices[0].message.tool_calls.append(
        SimpleNamespace(
            id="call-terminal",
            type="function",
            function=SimpleNamespace(
                name="terminal",
                arguments='{"command":"touch /tmp/x"}',
            ),
        )
    )
    executed = []

    def execute_plan(assistant_message, messages, *_args):
        executed.append([call.function.name for call in assistant_message.tool_calls])
        agent._zet_agent_plan_presented = True
        messages.append(
            {
                "role": "tool",
                "name": "present_plan",
                "tool_call_id": "call-plan",
                "content": '{"success":true}',
            }
        )

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[response, _text_response("计划已展示，继续执行。")],
        ),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_plan),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("先展示计划卡片")

    assert executed == [["present_plan"]]
    assert result["api_calls"] == 2


class _CapturingQ:
    """Minimal stream_q stand-in that records pushed SSE extension events."""

    def __init__(self):
        self.puts = []

    def put(self, item):
        self.puts.append(item)


def _last_plan_payload(q: "_CapturingQ") -> dict:
    kind, payload = q.puts[-1]
    assert kind == "__tool_progress__"
    assert payload["type"] == "hermes.plan"
    return payload


def test_plan_emit_sse_payload_carries_auto_execute_true():
    # SSE 契约：opt-in（auto flag True）→ hermes.plan.auto_execute=True，
    # App 据此渲染只读自动执行卡、不弹确认。
    from gateway.platforms.zet_agent import ZetAgentAdapter

    q = _CapturingQ()
    agent = SimpleNamespace(_zet_agent_plan_auto_execute=True)
    emit = ZetAgentAdapter._make_plan_emit_cb(q, agent)
    emit("减脂计划", [{"icon": "🏃", "label": "训练", "items": ["跑步"]}])
    assert _last_plan_payload(q)["auto_execute"] is True


def test_plan_emit_sse_payload_carries_auto_execute_false_and_defaults_manual():
    # 未 opt-in（flag False 或缺失）→ auto_execute=False，退回 legacy 确认卡语义。
    from gateway.platforms.zet_agent import ZetAgentAdapter

    for agent in (
        SimpleNamespace(_zet_agent_plan_auto_execute=False),
        SimpleNamespace(),  # flag 缺失 → getattr 默认 False
    ):
        q = _CapturingQ()
        emit = ZetAgentAdapter._make_plan_emit_cb(q, agent)
        emit("减脂计划", [{"icon": "🏃", "label": "训练", "items": ["跑步"]}])
        assert _last_plan_payload(q)["auto_execute"] is False
