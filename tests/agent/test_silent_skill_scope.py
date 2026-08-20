"""Silent video turns use ordinary plugin tools, not a skill authorization hop."""

from types import SimpleNamespace

from gateway.platforms.zet_agent import _apply_execution_policy


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def test_silent_video_tools_are_available_without_skill_view_attestation():
    names = {
        "video_edit_preferences_resolve",
        "video_edit_upload_assets",
        "video_edit_create_project",
        "video_edit_wait_project",
        "video_edit_download_result",
    }
    agent = SimpleNamespace(
        tools=[*_map_tools(names), _tool("skill_view"), _tool("terminal")],
        valid_tool_names=set(names) | {"skill_view", "terminal"},
    )

    _apply_execution_policy(
        agent,
        "silent_automation",
        trusted_skill_slug="video-edit-workflow-mini",
    )

    assert agent.valid_tool_names == names
    assert {item["function"]["name"] for item in agent.tools} == names
    assert agent._zet_agent_video_edit_turn is True


def _map_tools(names: set[str]) -> list[dict]:
    return [_tool(name) for name in sorted(names)]
