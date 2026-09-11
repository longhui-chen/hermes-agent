"""Camera confirmation must reach the existing trusted client handoff."""

from types import SimpleNamespace

import pytest

import agent.zet_agent_response_mode as mode
from gateway.session_context import clear_turn_vars, set_turn_vars
from tools.clarify_tool import clarify_tool


@pytest.mark.parametrize("camera_id,expected", [("cam-1", True), ("other", False)])
def test_recording_confirmation_dispatch_preserves_camera_binding(camera_id, expected):
    tokens = set_turn_vars(turn_id="confirmation-turn")
    try:
        agent = SimpleNamespace(platform="zet_agent")
        task = mode._skill_direct_task_context(agent, "为摄像头准备持续录像")
        agent._zet_agent_skill_direct_task = task
        agent._zet_agent_skill_direct_scope = mode._SkillDirectScope(
            relative_path=mode._CAMERA_SKILL_PATH,
            task_sha256=task.task_sha256,
            turn_identity=task.turn_identity,
            allowed_tools=mode._CAMERA_DIRECT_TOOLS,
            camera_ids=frozenset({"cam-1"}),
        )
        agent._zet_agent_skill_direct_operation = None
        args = {"question": "请确认录像条款", "connector_setup": {
            "resource_kind": "camera", "recording": {"camera_id": camera_id},
        }}
        blocked = mode.trusted_skill_operation_block_message(
            agent, function_name="clarify", function_args=args,
        )
        assert (blocked is None) is expected
        if expected:
            delivered = []

            def callback(question, choices, *, connector_setup=None):
                delivered.append(connector_setup)
                return '{"status":"cancelled"}'

            clarify_tool(**args, callback=callback)
            assert delivered == [args["connector_setup"]]
    finally:
        clear_turn_vars(tokens)


@pytest.mark.parametrize("intent", [
    None,
    {"resource_kind": "camera"},
    {"resource_kind": "printer3d"},
    {"resource_kind": "camera", "recording": {"camera_id": "cam-1", "confirmation_token": "invented"}},
    {"resource_kind": "camera", "recording": {"camera_id": "cam-1", "retention_days": 2}},
])
def test_confirmation_rejects_plain_text_and_authority_fields(intent):
    from gateway.platforms.zet_agent_camera_chat_intent import camera_confirmation_allowed

    assert not camera_confirmation_allowed(
        {"question": "确认录像", "connector_setup": intent}, {"cam-1"},
    )
