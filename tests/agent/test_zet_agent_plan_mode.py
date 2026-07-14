from types import SimpleNamespace

from agent.conversation_loop import (
    _apply_forced_present_plan_tool_choice,
    _is_thinking_tool_choice_rejection,
    _is_unsupported_thinking_parameter_error,
    _should_end_after_present_plan,
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
        "_zet_agent_plan_mode_active": False,
        "_zet_agent_plan_presented": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_zet_agent_plan_mode_does_not_infer_mode_from_user_copy():
    agent = _agent()

    for user_message in (
        "计划模式",
        "先别执行",
        "不要执行",
        "等我确认",
        "确认后",
        "present_plan",
        "用 present_plan 展示计划卡片，等我确认",
        "开启 plan 模式，我测试一下",
        "先给我计划，先别执行",
    ):
        assert not _should_force_present_plan_tool_choice(agent, user_message)


def test_zet_agent_plan_mode_forces_from_structured_response_mode():
    agent = _agent(_zet_agent_response_mode="plan", valid_tool_names={"todo"})

    assert _should_force_present_plan_tool_choice(agent, "帮我想一个一周的减肥计划")


def test_only_structured_response_mode_activates_plan_mode():
    cancel_agent = _agent(
        _zet_agent_plan_ack={"status": "cancelled", "revision_requested": False},
    )
    confirm_agent = _agent(
        _zet_agent_plan_ack={"status": "confirmed", "revision_requested": False},
    )
    revision_agent = _agent(
        _zet_agent_plan_ack={"status": "cancelled", "revision_requested": True},
    )
    structured_plan_agent = _agent(
        _zet_agent_response_mode="plan",
        _zet_agent_plan_ack={"status": "cancelled", "revision_requested": True},
    )

    assert not _should_force_present_plan_tool_choice(
        cancel_agent,
        "取消计划「减脂计划」，不要执行。",
    )
    assert not _should_force_present_plan_tool_choice(
        confirm_agent,
        "确认执行计划，请开始执行。",
    )
    assert not _should_force_present_plan_tool_choice(
        revision_agent,
        "请重新规划，不要执行旧计划。",
    )
    assert _should_force_present_plan_tool_choice(
        structured_plan_agent,
        "任意语言和内容都不参与模式判断",
    )


def test_zet_agent_plan_mode_does_not_force_plain_plan_word():
    agent = _agent()

    assert not _should_force_present_plan_tool_choice(agent, "你现在列个我每天健康饮食的plan")
    assert not _should_force_present_plan_tool_choice(
        _agent(platform="telegram", _zet_agent_response_mode="plan"),
        "开启 plan 模式",
    )


def test_plan_mode_only_exposes_clarify_and_present_plan_on_every_call():
    agent = _agent(_zet_agent_plan_mode_active=True)
    api_kwargs = {
        "messages": [
            {"role": "system", "content": "base system prompt"},
            {"role": "user", "content": "帮我制定减肥计划"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "clarify"}},
            {"type": "function", "function": {"name": "present_plan"}},
            {"type": "function", "function": {"name": "todo"}},
            {"type": "function", "function": {"name": "write_file"}},
        ]
    }

    assert _apply_forced_present_plan_tool_choice(agent, api_kwargs)
    assert api_kwargs["tool_choice"] == "required"
    assert api_kwargs["parallel_tool_calls"] is False
    assert [tool["function"]["name"] for tool in api_kwargs["tools"]] == [
        "clarify",
        "present_plan",
    ]
    assert api_kwargs["messages"][0]["content"].startswith("base system prompt")
    assert "MUST call `clarify`" in api_kwargs["messages"][0]["content"]
    assert "MUST NOT call `present_plan` yet" in api_kwargs["messages"][0]["content"]
    assert api_kwargs["messages"][1] == {
        "role": "user",
        "content": "帮我制定减肥计划",
    }
    assert "extra_body" not in api_kwargs

    follow_up_kwargs = {
        "messages": [
            {"role": "system", "content": "base system prompt"},
            {"role": "user", "content": "帮我制定减肥计划"},
            {"role": "tool", "content": "身高 175cm，体重 80kg"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "clarify"}},
            {"type": "function", "function": {"name": "present_plan"}},
            {"type": "function", "function": {"name": "todo"}},
        ]
    }
    assert _apply_forced_present_plan_tool_choice(agent, follow_up_kwargs)
    assert follow_up_kwargs["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in follow_up_kwargs["tools"]] == [
        "clarify",
        "present_plan",
    ]
    assert "After each clarify response" in follow_up_kwargs["messages"][0]["content"]


def test_forced_present_plan_disables_thinking_mode_and_preserves_extra_body():
    agent = _agent(
        _zet_agent_plan_mode_active=True,
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
    assert api_kwargs["tool_choice"] == "required"
    assert api_kwargs["extra_body"] == {
        "metadata": {"source": "test"},
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in api_kwargs


def test_forced_present_plan_disables_thinking_for_zettlab_ai_proxy():
    agent = _agent(
        _zet_agent_plan_mode_active=True,
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
    assert api_kwargs["tool_choice"] == "required"
    assert api_kwargs["extra_body"] == {
        "metadata": {"source": "test"},
        "thinking": {"type": "disabled"},
    }
    assert "reasoning_effort" not in api_kwargs


def test_forced_present_plan_reactive_disable_flag_covers_custom_models():
    agent = _agent(
        _zet_agent_plan_mode_active=True,
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
    assert api_kwargs["tool_choice"] == "required"
    assert api_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in api_kwargs


def test_plan_tool_choice_error_detection():
    err = RuntimeError(
        "Error code: 400 - {'error': {'message': "
        "'Thinking mode does not support this tool_choice'}}"
    )

    assert _is_thinking_tool_choice_rejection(err)
    assert not _is_unsupported_thinking_parameter_error(err)


def test_unsupported_thinking_parameter_detection():
    assert _is_unsupported_thinking_parameter_error(
        RuntimeError("HTTP 400: Unsupported parameter: thinking")
    )
    assert _is_unsupported_thinking_parameter_error(
        RuntimeError("unrecognized request argument supplied: thinking")
    )


def test_present_plan_tool_result_ends_zet_agent_plan_turn():
    assert _should_end_after_present_plan(
        _agent(
            _zet_agent_plan_mode_active=True,
            _zet_agent_plan_presented=True,
        )
    )


def test_present_plan_tool_result_does_not_end_regular_tool_turns():
    assert not _should_end_after_present_plan(
        _agent(
            _zet_agent_plan_mode_active=False,
            _zet_agent_plan_presented=True,
        )
    )
    assert not _should_end_after_present_plan(
        _agent(
            platform="telegram",
            _zet_agent_plan_mode_active=True,
            _zet_agent_plan_presented=True,
        )
    )


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
