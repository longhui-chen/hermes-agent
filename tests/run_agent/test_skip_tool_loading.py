import signal
from unittest.mock import patch

if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM

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
