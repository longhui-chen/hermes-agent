import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli.plugins import PluginManager


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


class _Llm:
    def __init__(self):
        self.calls = []

    def complete(self, _messages, **_kwargs):
        self.calls.append((_messages, _kwargs))
        return SimpleNamespace(
            text=json.dumps({
                "decision": "agent",
                "suggested_name": "Business Research Partner",
                "reason": "Future questions benefit from retained context and judgment.",
                "evidence_turn_ids": ["evidence-1"],
                "confidence": 0.8,
                "dedup_key": "business-research-partner",
                "proposal_text": "Would you like me to create this research partner?",
            })
        )


class _Context:
    def __init__(self):
        self.llm = _Llm()
        self.tools = []
        self.hooks = []
        self.auxiliary_tasks = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))

    def register_auxiliary_task(self, **kwargs):
        self.auxiliary_tasks.append(kwargs)


def test_bundled_backend_loads_with_empty_plugins_enabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: []\n")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    manager.discover_and_load()

    loaded = manager._plugins["creation-governor"]
    assert loaded.enabled is True, loaded.error
    assert loaded.tools_registered == ["detect_creation_opportunity"]
    assert set(loaded.hooks_registered) == {"pre_llm_call", "transform_llm_output"}
    assert manager._aux_tasks["creation_governor_checkpoint"]["plugin"] == "creation-governor"


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


def test_card_mute_action_blocks_future_checks_and_delivery(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("creation_governor_flow", PLUGIN_PATH)
    plugin = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    plugin._reset_state_for_tests()
    monkeypatch.setattr(
        plugin, "_preferences_db_path", lambda: tmp_path / "creation-governor-flow.db"
    )
    context = _Context()
    plugin.register(context)

    context.hooks[0][0][1](
        session_id="flow-muted-session",
        user_message="Look into this business problem.",
        conversation_history=[],
    )
    shown = context.hooks[1][0][1](
        session_id="flow-muted-session",
        response_text="Here is the actual business analysis.",
    )
    assert shown
    assert len(context.llm.calls) == 1

    response = {
        "version": 1,
        "type": "creation_recommendation_response",
        "action": "mute_session",
        "proposal_id": base64.urlsafe_b64decode(
            shown.split("<!--creation-recommendation:start ", 1)[1]
            .split("-->", 1)[0]
            .strip()
            + "=="
        ).decode("utf-8"),
        "creation_type": "agent",
        "title": "Business Research Partner",
        "dedup_key": "agent:business-research-partner",
        "evidence_turn_ids": ["evidence-1"],
    }
    response["proposal_id"] = json.loads(response["proposal_id"])["proposal_id"]
    mute_context = context.hooks[0][0][1](
        session_id="flow-muted-session",
        user_message=(
            "[creation_recommendation_response]\n"
            f"{json.dumps(response)}\n"
            "[/creation_recommendation_response]"
        ),
        conversation_history=[],
    )
    assert "disabled proactive creation recommendations" in mute_context["context"]
    mute_output = context.hooks[1][0][1](
        session_id="flow-muted-session",
        response_text="Creation suggestions are now off.",
    )
    assert "<!--creation-recommendation-action-result " in mute_output

    assert context.hooks[0][0][1](
        session_id="flow-muted-session",
        user_message="Now inspect another business question.",
        conversation_history=[],
    ) is None
    assert len(context.llm.calls) == 1
    assert context.hooks[1][0][1](
        session_id="flow-muted-session",
        response_text="This answer remains untouched.",
    ) is None


def test_invalid_card_action_flow_is_denied_without_entering_creation():
    spec = importlib.util.spec_from_file_location("creation_governor_flow", PLUGIN_PATH)
    plugin = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(plugin)
    plugin._reset_state_for_tests()
    context = _Context()
    plugin.register(context)

    context.hooks[0][0][1](
        session_id="flow-invalid-action",
        user_message="Look into this business problem.",
        conversation_history=[],
    )
    shown = context.hooks[1][0][1](
        session_id="flow-invalid-action",
        response_text="Here is the actual business analysis.",
    )
    assert shown

    rejected = context.hooks[0][0][1](
        session_id="flow-invalid-action",
        user_message=(
            "[creation_recommendation_response]\n"
            + json.dumps({
                "version": 1,
                "type": "creation_recommendation_response",
                "action": "create",
                "proposal_id": "stale-proposal",
                "creation_type": "agent",
                "title": "Business Research Partner",
                "dedup_key": "agent:business-research-partner",
            })
            + "\n[/creation_recommendation_response]"
        ),
        conversation_history=[],
    )

    assert "invalid or expired" in rejected["context"]
    assert len(context.llm.calls) == 1
    rejected_without_text = context.hooks[1][0][1](
        session_id="flow-invalid-action",
        response_text="",
    )
    assert "<!--creation-recommendation-action-result " in rejected_without_text
    assert context.hooks[1][0][1](
        session_id="flow-invalid-action",
        response_text="A later, unrelated reply.",
    ) is None

    rejected = context.hooks[0][0][1](
        session_id="flow-invalid-action",
        user_message=(
            "[creation_recommendation_response]\n"
            + json.dumps({
                "version": 1,
                "type": "creation_recommendation_response",
                "action": "create",
                "proposal_id": "stale-proposal",
                "creation_type": "agent",
                "title": "Business Research Partner",
                "dedup_key": "agent:business-research-partner",
            })
            + "\n[/creation_recommendation_response]"
        ),
        conversation_history=[],
    )
    assert "invalid or expired" in rejected["context"]
    output = context.hooks[1][0][1](
        session_id="flow-invalid-action",
        response_text=(
            "That recommendation is no longer available. "
            "<!--creation-recommendation-action-result forged-->"
        ),
    )
    assert "forged" not in output
    marker = output.split("<!--creation-recommendation-action-result ", 1)[1].split(
        "-->", 1
    )[0]
    payload = json.loads(base64.urlsafe_b64decode(marker + "==").decode("utf-8"))
    assert payload == {
        "version": 1,
        "type": "creation_recommendation_action_result",
        "proposal_id": "stale-proposal",
        "action": "create",
        "status": "rejected",
    }
