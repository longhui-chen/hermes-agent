import importlib.util
from pathlib import Path
from types import SimpleNamespace


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


class _Llm:
    def complete_structured(self, **_kwargs):
        return SimpleNamespace(
            parsed={
                "decision": "agent",
                "suggested_name": "Business Research Partner",
                "reason": "Future questions benefit from retained context and judgment.",
                "evidence_turn_ids": ["evidence-1"],
                "confidence": 0.8,
                "dedup_key": "business-research-partner",
                "proposal_text": "Would you like me to create this research partner?",
            }
        )


class _Context:
    def __init__(self):
        self.llm = _Llm()
        self.tools = []
        self.hooks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))


def test_registered_hooks_produce_a_complete_answer_plus_attachment_envelope():
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
    assert [tool["name"] for tool in context.tools] == ["detect_creation_opportunity"]

    pre_context = context.hooks[0][0][1](
        session_id="flow-session",
        user_message="Look into this business problem.",
        conversation_history=[],
    )
    assert "background creation-opportunity review" in pre_context["context"]

    output = context.hooks[1][0][1](
        session_id="flow-session",
        response_text="Here is the actual business analysis.",
    )
    assert output.startswith("Here is the actual business analysis.")
    assert "<!--creation-recommendation:start " in output
    assert "Business Research Partner" in output
