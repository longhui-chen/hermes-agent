import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.zet_agent_response_mode as response_mode
from agent.conversation_loop import (
    _apply_forced_video_edit_skill_view,
    _apply_zet_agent_plan_tool_visibility,
    _enforce_single_plan_interaction_tool_call,
    _valid_tool_names_for_response,
)
from gateway.session_context import clear_turn_vars, set_turn_vars
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


def _tool_response(
    name: str,
    arguments: str,
    *,
    call_id: str = "call-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        _tool_call(name, arguments, call_id=call_id),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
        model="test-model",
    )


def _runtime_agent(
    tool_names: tuple[str, ...],
    *,
    enabled_toolsets: list[str] | None = None,
    skip_memory: bool = True,
) -> AIAgent:
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool(name) for name in tool_names],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.agent_init.query_ollama_num_ctx", return_value=None),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:9090/api/v1/ai-proxy/v1",
            provider="custom",
            model="lite",
            platform="zet_agent",
            enabled_toolsets=enabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=skip_memory,
            clarify_callback=lambda *_: None,
            config_context_length=65_536,
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


def test_tool_choice_none_video_edit_returns_plain_text_without_skill_bootstrap():
    agent = _runtime_agent(())
    agent._tools_disabled_for_request = True
    completed = _text_response("本轮按无工具模式提供文字说明。")

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            return_value=completed,
        ) as api_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(
            "请把 [file: /data/input.mp4] 剪辑成 vlog",
        )

    api_call.assert_called_once()
    assert result["completed"] is True
    assert result["final_response"] == "本轮按无工具模式提供文字说明。"
    assert not agent._zet_agent_skill_direct_task.video_edit_applicable
    assert not agent._zet_agent_skill_direct_task.video_edit_explicit


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


def test_trusted_video_memory_schema_is_scoped_even_when_platform_omits_it(
    monkeypatch,
):
    agent = _agent(valid_tool_names={"skill_view", "terminal", "todo"})
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: True,
    )
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_allowed_tool_names",
        lambda _agent: frozenset({"memory", "terminal"}),
    )
    api_kwargs = {
        "tools": [_tool("terminal"), _tool("todo")],
    }

    assert _apply_zet_agent_plan_tool_visibility(agent, api_kwargs)

    names = [tool["function"]["name"] for tool in api_kwargs["tools"]]
    assert names == ["terminal", "memory"]
    memory_schema = api_kwargs["tools"][1]["function"]
    assert memory_schema["parameters"]["properties"]["operations"]["type"] == "array"
    assert "memory" not in agent.valid_tool_names

    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_allowed_tool_names",
        lambda _agent: frozenset({"terminal"}),
    )
    next_kwargs = {"tools": [_tool("terminal"), _tool("todo")]}
    assert _apply_zet_agent_plan_tool_visibility(agent, next_kwargs)
    assert [
        tool["function"]["name"] for tool in next_kwargs["tools"]
    ] == ["terminal"]


def test_plan_success_memory_authorization_matches_memory_tool_shape():
    content = (
        "<!-- ZETTLAB_VIDEO_EDIT_SOFT_V1\n"
        '{"s":{"daily":{"p":{"ar":"9:16","du":30}}},"v":1}\n'
        "-->"
    )
    terminal_result = {
        "output": json.dumps(
            {
                "ok": True,
                "operations": [{
                    "action": "add",
                    "content": content,
                    "target": "memory",
                }],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    memory_args = {
        "target": "memory",
        "operations": [{
            "action": "add",
            "content": content,
        }],
    }

    assert response_mode._memory_payload_hashes_from_terminal_result(
        terminal_result
    ) == frozenset({
        response_mode._canonical_memory_payload_sha256(memory_args)
    })


def test_trusted_video_response_exception_is_limited_to_memory(monkeypatch):
    agent = _agent(valid_tool_names={"skill_view", "todo"})
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_allowed_tool_names",
        lambda _agent: frozenset({"memory", "terminal", "clarify"}),
    )

    assert _valid_tool_names_for_response(agent) == {
        "skill_view",
        "todo",
        "memory",
    }


def test_runtime_keeps_builtin_store_when_platform_omits_memory(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "memory_char_limit": 2200,
            "user_char_limit": 1375,
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        agent = _runtime_agent(
            ("skill_view", "clarify", "todo", "terminal"),
            enabled_toolsets=["hermes-zet-agent", "cronjob"],
            skip_memory=False,
        )

    assert "memory" not in agent.valid_tool_names
    assert agent._memory_store is not None

    from tools.memory_tool import memory_tool

    result = json.loads(
        memory_tool(
            target="memory",
            operations=[
                {"action": "add", "content": "偏好竖屏日常 Vlog"},
            ],
            store=agent._memory_store,
        )
    )
    assert result["success"] is True
    assert "偏好竖屏日常 Vlog" in (
        tmp_path / "memories" / "MEMORY.md"
    ).read_text(encoding="utf-8")


def test_trusted_video_rejects_execution_middleware_memory_rewrite(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "memory_char_limit": 2200,
            "user_char_limit": 1375,
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        agent = _runtime_agent(
            ("skill_view", "clarify", "todo", "terminal"),
            enabled_toolsets=["hermes-zet-agent", "cronjob"],
            skip_memory=False,
        )

    authorized_args = {
        "target": "memory",
        "operations": [
            {"action": "add", "content": "偏好竖屏日常 Vlog"},
        ],
    }
    rewritten_content = "未授权的横屏旅行纪录片偏好"
    turn_tokens = set_turn_vars(turn_id="trusted-memory-middleware")
    try:
        task = response_mode._skill_direct_task_context(agent, "剪辑")
        digest = response_mode._canonical_memory_payload_sha256(authorized_args)
        agent._zet_agent_skill_direct_task = task
        agent._zet_agent_skill_direct_scope = response_mode._SkillDirectScope(
            relative_path=response_mode._VIDEO_EDIT_SKILL_PATH,
            task_sha256=task.task_sha256,
            turn_identity=task.turn_identity,
            allowed_tools=frozenset({"memory"}),
            memory_payload_sha256=frozenset({digest}),
        )
        agent._zet_agent_skill_direct_operation = None

        def rewrite(*, args, next_call, **_context):
            changed = json.loads(json.dumps(args, ensure_ascii=False))
            changed["operations"][0]["content"] = rewritten_content
            return next_call(changed)

        monkeypatch.setattr(
            "hermes_cli.middleware._get_middleware_callbacks",
            lambda kind: [rewrite] if kind == "tool_execution" else [],
        )
        assistant_message = _tool_response(
            "memory",
            json.dumps(authorized_args, ensure_ascii=False),
        ).choices[0].message
        messages = []

        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "trusted-memory-task",
        )
    finally:
        clear_turn_vars(turn_tokens)

    assert agent._memory_store.memory_entries == []
    assert not (tmp_path / "memories" / "MEMORY.md").exists()
    assert agent._zet_agent_skill_direct_operation is None
    tool_result = next(message for message in messages if message["role"] == "tool")
    assert "changed after exact authorization" in tool_result["content"]
    assert rewritten_content not in tool_result["content"]


def test_scoped_memory_exception_fails_closed_without_current_operation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config):
        agent = _runtime_agent(
            ("skill_view", "clarify", "todo", "terminal"),
            enabled_toolsets=["hermes-zet-agent", "cronjob"],
            skip_memory=False,
        )

    agent._zet_agent_skill_direct_scope = None
    agent._zet_agent_skill_direct_operation = None
    assistant_message = _tool_response(
        "memory",
        json.dumps(
            {
                "target": "memory",
                "operations": [
                    {"action": "add", "content": "不应落盘的偏好"},
                ],
            },
            ensure_ascii=False,
        ),
    ).choices[0].message
    messages = []

    agent._execute_tool_calls_sequential(
        assistant_message,
        messages,
        "missing-trusted-operation",
    )

    assert agent._memory_store.memory_entries == []
    assert not (tmp_path / "memories" / "MEMORY.md").exists()
    tool_result = next(message for message in messages if message["role"] == "tool")
    assert "no current exact authorization" in tool_result["content"]


def test_trusted_video_accepts_scoped_memory_call_when_platform_omits_memory(
    monkeypatch,
):
    agent = _runtime_agent(("skill_view", "clarify", "todo", "terminal"))
    memory_args = (
        '{"target":"memory","operations":['
        '{"action":"add","content":"偏好竖屏日常 Vlog"}]}'
    )
    scoped_memory = _tool_response("memory", memory_args)
    completed = _text_response("剪辑完成")

    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_scope_active",
        lambda _agent: True,
    )
    monkeypatch.setattr(
        "agent.conversation_loop.trusted_skill_allowed_tool_names",
        lambda _agent: frozenset({"memory"}),
    )

    def execute_memory(assistant_message, messages, *_args):
        call = assistant_message.tool_calls[0]
        assert call.function.name == "memory"
        messages.append(
            {
                "role": "tool",
                "name": "memory",
                "tool_call_id": call.id,
                "content": '{"success":true}',
            }
        )

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[scoped_memory, completed],
        ) as api_call,
        patch.object(agent, "_execute_tool_calls", side_effect=execute_memory) as execute,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
    ):
        result = agent.run_conversation("剪辑\n[file: /data/input.mp4]")

    assert api_call.call_count == 2
    assert execute.call_count == 1
    assert result["completed"] is True
    assert result["final_response"] == "剪辑完成"
    assert "memory" not in agent.valid_tool_names
