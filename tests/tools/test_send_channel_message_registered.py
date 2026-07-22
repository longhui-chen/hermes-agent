def test_send_channel_message_in_core_tools():
    import toolsets
    assert "send_channel_message" in toolsets._HERMES_CORE_TOOLS
