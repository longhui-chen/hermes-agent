from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.conversation_loop import (
    _apply_forced_video_edit_skill_view,
    _enforce_single_plan_interaction_tool_call,
)
from run_agent import AIAgent


_VIDEO_EDIT_SKILL = "video-edit-workflow-mini"


def _tool(name: str) -> dict:
    properties = {}
    required = []
    if name == "skill_view":
        properties = {
            "name": {"type": "string", "description": "Skill name"},
            "file_path": {"type": "string", "description": "Linked file"},
        }
        required = ["name"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _agent(**overrides) -> SimpleNamespace:
    base = {
        "platform": "zet_agent",
        "api_mode": "chat_completions",
        "model": "lite",
        "provider": "custom",
        "base_url": "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
        "valid_tool_names": {"skill_view", "clarify", "todo"},
        "_zet_agent_plan_mode_active": False,
        "_zet_agent_plan_presented": False,
        "_zet_agent_skill_direct_task": SimpleNamespace(
            video_edit_applicable=True,
        ),
        "_zet_agent_skill_direct_scope": None,
        "_zet_agent_plan_omit_thinking_disable": False,
        "_zet_agent_force_present_plan_disable_thinking": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tool_call(name: str, arguments: str, *, call_id: str = "call-1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _text_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="test-model",
    )


def _runtime_agent(tool_names: tuple[str, ...]) -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool(name) for name in tool_names],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:9090/api/v1/ai-proxy/v1",
            provider="custom",
            model="lite",
            platform="zet_agent",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            clarify_callback=lambda *_: None,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_video_edit_first_request_forces_exact_skill_view_without_mutating_registry(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: False,
    )
    skill_tool = _tool("skill_view")
    original_skill_parameters = skill_tool["function"]["parameters"].copy()
    api_kwargs = {
        "messages": [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "剪辑"},
        ],
        "tools": [skill_tool, _tool("clarify"), _tool("todo")],
        "reasoning_effort": "medium",
    }

    assert _apply_forced_video_edit_skill_view(_agent(), api_kwargs)

    assert api_kwargs["tool_choice"] == "required"
    assert api_kwargs["parallel_tool_calls"] is False
    assert [tool["function"]["name"] for tool in api_kwargs["tools"]] == [
        "skill_view"
    ]
    parameters = api_kwargs["tools"][0]["function"]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name",
                "enum": [_VIDEO_EDIT_SKILL],
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert api_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in api_kwargs
    assert "MUST call `skill_view`" in api_kwargs["messages"][0]["content"]
    assert skill_tool["function"]["parameters"] == original_skill_parameters


def test_video_edit_skill_force_stops_after_trusted_scope_activates(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: True,
    )
    api_kwargs = {"tools": [_tool("skill_view"), _tool("terminal")]}

    assert not _apply_forced_video_edit_skill_view(_agent(), api_kwargs)
    assert [tool["function"]["name"] for tool in api_kwargs["tools"]] == [
        "skill_view",
        "terminal",
    ]
    assert "tool_choice" not in api_kwargs


def test_video_edit_skill_force_does_not_change_non_video_or_plan_requests(monkeypatch):
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: False,
    )
    tools = [_tool("skill_view"), _tool("clarify")]

    non_video_kwargs = {"tools": list(tools)}
    assert not _apply_forced_video_edit_skill_view(
        _agent(
            _zet_agent_skill_direct_task=SimpleNamespace(
                video_edit_applicable=False,
            )
        ),
        non_video_kwargs,
    )
    assert non_video_kwargs == {"tools": tools}

    plan_kwargs = {"tools": list(tools)}
    assert not _apply_forced_video_edit_skill_view(
        _agent(_zet_agent_plan_mode_active=True),
        plan_kwargs,
    )
    assert plan_kwargs == {"tools": tools}


def test_video_edit_skill_force_drops_wrong_or_parallel_calls_before_execution(
    monkeypatch,
):
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: False,
    )
    wrong = SimpleNamespace(
        content="先问一下",
        tool_calls=[_tool_call("todo", '{"content":"unsafe"}')],
        provider_data={},
    )

    assert _enforce_single_plan_interaction_tool_call(_agent(), wrong)
    assert wrong.tool_calls == []

    exact = _tool_call("skill_view", '{"name":"video-edit-workflow-mini"}')
    parallel = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("terminal", '{"command":"touch /tmp/forbidden"}'),
            exact,
            _tool_call(
                "skill_view",
                '{"name":"video-edit-workflow-mini","file_path":"scripts/x.py"}',
                call_id="linked-file",
            ),
        ],
        provider_data={},
    )

    assert _enforce_single_plan_interaction_tool_call(_agent(), parallel)
    assert parallel.tool_calls == [exact]


def test_video_edit_skill_load_reports_missing_tool_without_calling_provider():
    agent = _runtime_agent(("clarify", "todo"))

    with (
        patch.object(agent, "_interruptible_api_call") as api_call,
        patch.object(agent, "_persist_session"),
    ):
        result = agent.run_conversation("剪辑\n[file: /data/input.mp4]")

    api_call.assert_not_called()
    assert result["failed"] is True
    assert result["api_calls"] == 0
    assert "skill_view" in result["error"]
    assert _VIDEO_EDIT_SKILL in result["error"]


def test_video_edit_plain_text_is_bounded_to_two_protocol_retries():
    agent = _runtime_agent(("skill_view", "clarify", "todo"))
    plain_text = _text_response("你想把这些视频剪成什么样的成片？")

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[plain_text, plain_text, plain_text],
        ) as api_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("剪辑\n[file: /data/input.mp4]")

    assert api_call.call_count == 3
    assert result["failed"] is True
    assert result["completed"] is False
    assert "video-edit skill protocol error" in result["error"].lower()
    assert all(
        not message.get("_video_edit_skill_protocol_synthetic")
        for message in result["messages"]
    )
