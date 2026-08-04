def test_list_my_channels_in_core_tools():
    import toolsets
    assert "list_my_channels" in toolsets._HERMES_CORE_TOOLS


def test_zet_agent_toolset_includes_list_my_channels():
    import toolsets
    ts = toolsets.TOOLSETS["hermes-zet-agent"]["tools"]
    assert "list_my_channels" in ts
