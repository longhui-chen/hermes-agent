import importlib.util
import json
from pathlib import Path


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


class _Context:
    def __init__(self) -> None:
        self.tools = []
        self.hooks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))


def test_plugin_flow_registers_zero_shot_review_and_non_creating_proposal_tool():
    spec = importlib.util.spec_from_file_location("creation_governor_flow", PLUGIN_PATH)
    plugin = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    plugin._reset_state_for_tests()
    context = _Context()
    plugin.register(context)

    assert [args[0] for args, _kwargs in context.hooks] == [
        "pre_llm_call",
        "transform_llm_output",
        "post_llm_call",
        "pre_tool_call",
    ]
    assert [tool["name"] for tool in context.tools] == ["propose_creation"]
    assert context.tools[0]["toolset"] == "creation_governor"
    assert "zero-shot" in context.tools[0]["schema"]["description"]

    pre_context = context.hooks[0][0][1](
        session_id="flow-session",
        user_message="Look into this business problem.",
        conversation_history=[],
    )
    assert "semantic rubric" in pre_context["context"]

    result = json.loads(
        context.tools[0]["handler"](
            {
                "creation_type": "agent",
                "suggested_name": "Business Research Partner",
                "reason": "Future questions benefit from retained context and judgment.",
                "evidence": "The user requested a substantive business investigation.",
                "confidence": 0.8,
                "dedup_key": "business-research-partner",
                "proposal_text": "Would you like me to prepare a Business Research Partner Agent creation plan?",
            },
            session_id="flow-session",
        )
    )

    assert result["status"] == "proposal_ready"
    assert result["creation_type"] == "agent"
    assert result["delivery"] == "deferred_to_transform_hook"
    assert "Look into this business problem" in result["next_step"]
    assert "does not create anything" in result["next_step"]
    assert "create" not in context.tools[0]["name"]


def test_real_plugin_manager_loads_all_governor_hooks():
    plugin_root = Path(__file__).resolve().parents[2] / "plugins"
    manager = PluginManager()
    manifests = manager._scan_directory(plugin_root, source="bundled")
    manifest = next(item for item in manifests if item.name == "creation-governor")

    manager._load_plugin(manifest)

    loaded = manager._plugins[manifest.key or manifest.name]
    assert loaded.enabled is True
    assert loaded.error is None
    assert set(loaded.hooks_registered) == {
        "pre_llm_call",
        "transform_llm_output",
        "post_llm_call",
        "pre_tool_call",
    }
    assert loaded.tools_registered == ["propose_creation"]


def test_real_plugin_manager_disables_governor_for_codex_app_server():
    plugin = _load_plugin()
    plugin_root = Path(__file__).resolve().parents[2] / "plugins"
    manager = PluginManager()
    manifests = manager._scan_directory(plugin_root, source="bundled")
    manifest = next(item for item in manifests if item.name == "creation-governor")
    manager._load_plugin(manifest)
    prompt = plugin._proposal_payload(
        "skill", "会议纪要流程", "固化整理步骤", "会议转录", 0.9
    )["user_prompt"]

    results = manager.invoke_hook(
        "pre_llm_call",
        session_id="codex-app-server-flow",
        api_mode="codex_app_server",
        user_message="生成方案",
        conversation_history=[{"role": "assistant", "content": prompt}],
    )

    assert results == []
    callback_globals = manager._hooks["pre_llm_call"][0].__globals__
    state_key = callback_globals["_session_key"](
        {"session_id": "codex-app-server-flow"}
    )
    state = callback_globals["_session_states"][state_key]
    assert state["proposal_stage"] is None


def test_real_plugin_manager_disables_non_followup_and_streaming_api_flows():
    plugin_root = Path(__file__).resolve().parents[2] / "plugins"
    manager = PluginManager()
    manifests = manager._scan_directory(plugin_root, source="bundled")
    manifest = next(item for item in manifests if item.name == "creation-governor")
    manager._load_plugin(manifest)

    for session_id, capabilities in (
        ("one-shot-flow", {"platform": "cli", "supports_followup_turns": False}),
        (
            "streaming-api-flow",
            {"platform": "api_server", "streaming_output": True},
        ),
    ):
        results = manager.invoke_hook(
            "pre_llm_call",
            session_id=session_id,
            user_message="帮我分析这份合同",
            conversation_history=[],
            **capabilities,
        )
        transformed = manager.invoke_hook(
            "transform_llm_output",
            session_id=session_id,
            response_text='{"result":"ok"}',
            **capabilities,
        )

        assert results == []
        assert transformed == []
        callback_globals = manager._hooks["pre_llm_call"][0].__globals__
        state_key = callback_globals["_session_key"]({"session_id": session_id})
        state = callback_globals["_session_states"][state_key]
        assert state["proposal_stage"] is None

    supported = manager.invoke_hook(
        "pre_llm_call",
        session_id="non-streaming-api-flow",
        platform="api_server",
        streaming_output=False,
        supports_followup_turns=True,
        user_message="帮我分析这份合同",
        conversation_history=[],
    )
    assert len(supported) == 1
    assert "Creation governor internal instruction" in supported[0]["context"]
