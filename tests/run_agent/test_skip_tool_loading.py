import signal
from unittest.mock import patch

if not hasattr(signal, "SIGKILL"):
    setattr(signal, "SIGKILL", signal.SIGTERM)

from run_agent import AIAgent


def test_skip_tool_loading_bypasses_registry_discovery():
    agent = None
    try:
        with patch(
            "run_agent.get_tool_definitions",
            side_effect=AssertionError("tool registry must not be loaded"),
        ):
            agent = AIAgent(
                model="test/model",
                api_key="test-key",
                base_url="http://127.0.0.1:9/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                skip_tool_loading=True,
            )

        assert agent.tools == []
        assert agent.valid_tool_names == set()
        assert agent._skip_mcp_refresh is True
    finally:
        if agent is not None:
            agent.close()


def test_tool_snapshot_records_composite_generation_and_turn_binding():
    from tools.registry import registry

    agent = None
    turn_binding = ("turn-1", object())
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.OpenAI"),
            patch.object(registry, "cache_generation", return_value=(3, 5)),
            patch(
                "gateway.session_context.current_turn_identity",
                return_value=turn_binding,
            ),
        ):
            agent = AIAgent(
                model="test/model",
                api_key="test-key",
                base_url="http://127.0.0.1:9/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        assert agent._tool_snapshot_generation == (3, 5)
        assert agent._tool_snapshot_turn_identity == turn_binding
    finally:
        if agent is not None:
            agent.close()
