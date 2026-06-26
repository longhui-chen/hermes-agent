from types import SimpleNamespace

from agent.conversation_loop import (
    _apply_forced_present_plan_tool_choice,
    _apply_plan_text_fallback_request,
    _emit_plain_text_plan_if_needed,
    _is_thinking_tool_choice_rejection,
    _is_unsupported_tools_or_tool_choice_error,
    _is_unsupported_thinking_parameter_error,
    _should_force_present_plan_tool_choice,
)
from agent.tool_executor import _zet_agent_plan_mode_block_message


def _agent(**overrides):
    base = {
        "platform": "zet_agent",
        "valid_tool_names": {"present_plan", "todo", "skill_view", "write_file"},
        "api_mode": "chat_completions",
        "model": "gpt-4o",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "_zet_agent_force_present_plan_pending": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_zet_agent_plan_mode_forces_present_plan_for_explicit_triggers():
    agent = _agent()

    assert _should_force_present_plan_tool_choice(
        agent, "用 present_plan 展示计划卡片，等我确认"
    )
    assert _should_force_present_plan_tool_choice(agent, "开启 plan 模式，我测试一下")
    assert _should_force_present_plan_tool_choice(agent, "先给我计划，先别执行")


def test_zet_agent_plan_mode_forces_from_structured_response_mode():
    agent = _agent(_zet_agent_response_mode="plan", valid_tool_names={"todo"})

    assert _should_force_present_plan_tool_choice(agent, "帮我想一个一周的减肥计划")


def test_zet_agent_plan_mode_does_not_force_plain_plan_word():
    agent = _agent()

    assert not _should_force_present_plan_tool_choice(agent, "你现在列个我每天健康饮食的plan")
    assert not _should_force_present_plan_tool_choice(
        _agent(platform="telegram"), "开启 plan 模式"
    )
    assert not _should_force_present_plan_tool_choice(
        _agent(valid_tool_names={"todo"}), "开启 plan 模式"
    )


def test_forced_present_plan_tool_choice_is_consumed_once():
    agent = _agent(_zet_agent_force_present_plan_pending=True)
    api_kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "present_plan"}},
            {"type": "function", "function": {"name": "todo"}},
        ]
    }

    assert _apply_forced_present_plan_tool_choice(agent, api_kwargs)
    assert api_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "present_plan"},
    }
    assert "extra_body" not in api_kwargs
    assert agent._zet_agent_force_present_plan_pending is False

    assert not _apply_forced_present_plan_tool_choice(agent, api_kwargs)


def test_forced_present_plan_disables_thinking_mode_and_preserves_extra_body():
    agent = _agent(
        _zet_agent_force_present_plan_pending=True,
        model="deepseek-v4-flash",
        provider="custom",
        base_url="http://127.0.0.1:9090/api/v1/custom-ai/ds/v1",
    )
    api_kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "present_plan"}},
        ],
        "extra_body": {"metadata": {"source": "test"}},
        "reasoning_effort": "high",
    }

    assert _apply_forced_present_plan_tool_choice(agent, api_kwargs)
    assert api_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "present_plan"},
    }
    assert api_kwargs["extra_body"] == {
        "metadata": {"source": "test"},
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in api_kwargs


def test_forced_present_plan_disables_thinking_for_zettlab_ai_proxy():
    agent = _agent(
        _zet_agent_force_present_plan_pending=True,
        model="lite",
        provider="custom",
        base_url="http://127.0.0.1:9090/api/v1/ai-proxy/v1",
    )
    api_kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "present_plan"}},
        ],
        "extra_body": {"metadata": {"source": "test"}},
        "reasoning_effort": "medium",
    }

    assert _apply_forced_present_plan_tool_choice(agent, api_kwargs)
    assert api_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "present_plan"},
    }
    assert api_kwargs["extra_body"] == {
        "metadata": {"source": "test"},
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in api_kwargs


def test_forced_present_plan_reactive_disable_flag_covers_custom_models():
    agent = _agent(
        _zet_agent_force_present_plan_pending=True,
        _zet_agent_force_present_plan_disable_thinking=True,
        model="custom-thinking",
        provider="custom",
        base_url="http://127.0.0.1:9090/api/v1/custom-ai/user-model/v1",
    )
    api_kwargs = {
        "tools": [
            {"type": "function", "function": {"name": "present_plan"}},
        ],
        "reasoning_effort": "medium",
    }

    assert _apply_forced_present_plan_tool_choice(agent, api_kwargs)
    assert api_kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "present_plan"},
    }
    assert api_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in api_kwargs


def test_plan_tool_choice_error_detection():
    err = RuntimeError(
        "Error code: 400 - {'error': {'message': "
        "'Thinking mode does not support this tool_choice'}}"
    )

    assert _is_thinking_tool_choice_rejection(err)
    assert not _is_unsupported_thinking_parameter_error(err)


def test_unsupported_tool_calling_parameter_detection():
    assert _is_unsupported_tools_or_tool_choice_error(
        RuntimeError("HTTP 400: Unsupported parameter: tools")
    )
    assert _is_unsupported_tools_or_tool_choice_error(
        RuntimeError("this model does not support tool calling")
    )
    assert _is_unsupported_tools_or_tool_choice_error(
        RuntimeError("unknown parameter: tool_choice")
    )


def test_unsupported_thinking_parameter_detection():
    assert _is_unsupported_thinking_parameter_error(
        RuntimeError("HTTP 400: Unsupported parameter: thinking")
    )
    assert _is_unsupported_thinking_parameter_error(
        RuntimeError("unrecognized request argument supplied: thinking")
    )


def test_plan_text_fallback_request_removes_tool_parameters_and_prompts_for_plan():
    api_kwargs = {
        "messages": [{"role": "user", "content": "帮我做计划"}],
        "tools": [{"type": "function", "function": {"name": "present_plan"}}],
        "tool_choice": {"type": "function", "function": {"name": "present_plan"}},
        "parallel_tool_calls": False,
        "extra_body": {
            "metadata": {"source": "test"},
            "thinking": {"type": "disabled"},
        },
    }

    _apply_plan_text_fallback_request(api_kwargs)

    assert "tools" not in api_kwargs
    assert "tool_choice" not in api_kwargs
    assert "parallel_tool_calls" not in api_kwargs
    assert api_kwargs["extra_body"] == {"metadata": {"source": "test"}}
    assert "structured plan only" in api_kwargs["messages"][0]["content"]


def test_plain_text_plan_response_is_emitted_as_plan_card():
    emitted = []
    agent = _agent(
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=False,
        plan_emit_callback=lambda title, groups: emitted.append((title, groups)),
    )

    _emit_plain_text_plan_if_needed(
        agent,
        "# 减肥计划\n\n准备:\n- 记录当前体重\n- 清理高糖零食\n\n执行:\n1. 每周运动三次\n2. 每天控制热量",
    )

    assert agent._zet_agent_plan_presented is True
    assert emitted
    title, groups = emitted[0]
    assert title == "减肥计划"
    assert groups[0]["label"] == "准备"
    assert groups[0]["items"] == ["记录当前体重", "清理高糖零食"]


def test_plain_text_plan_response_does_not_emit_duplicate_card():
    emitted = []
    agent = _agent(
        _zet_agent_plan_mode_active=True,
        _zet_agent_plan_presented=True,
        plan_emit_callback=lambda title, groups: emitted.append((title, groups)),
    )

    _emit_plain_text_plan_if_needed(agent, "- step")

    assert emitted == []


def test_zet_agent_blocks_legacy_markdown_plan_skill_paths():
    agent = _agent()

    skill_block = _zet_agent_plan_mode_block_message(
        agent, "skill_view", {"name": "plan"}
    )
    assert skill_block is not None
    assert "present_plan" in skill_block

    write_block = _zet_agent_plan_mode_block_message(
        agent, "write_file", {"path": "/tmp/.hermes/plans/demo.md"}
    )
    assert write_block is not None
    assert ".hermes/plans" in write_block

    relative_write_block = _zet_agent_plan_mode_block_message(
        agent, "write_file", {"path": ".hermes/plans/demo.md"}
    )
    assert relative_write_block is not None


def test_zet_agent_plan_blocks_are_scoped_to_app_plan_mode():
    assert _zet_agent_plan_mode_block_message(
        _agent(platform="telegram"), "skill_view", {"name": "plan"}
    ) is None
    assert _zet_agent_plan_mode_block_message(
        _agent(), "skill_view", {"name": "test-driven-development"}
    ) is None
    assert _zet_agent_plan_mode_block_message(
        _agent(), "write_file", {"path": "/tmp/output/plan.md"}
    ) is None
