from types import SimpleNamespace

from agent.conversation_loop import (
    _apply_forced_present_plan_tool_choice,
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
    assert agent._zet_agent_force_present_plan_pending is False

    assert not _apply_forced_present_plan_tool_choice(agent, api_kwargs)


def test_forced_present_plan_disables_deepseek_thinking_mode():
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
