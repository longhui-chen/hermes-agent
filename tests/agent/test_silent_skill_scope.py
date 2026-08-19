"""Silent automation may read only its transport-selected signed skill."""

import json
from types import SimpleNamespace

from agent import zet_agent_response_mode as response_mode
from gateway.session_context import clear_turn_vars, set_turn_vars


def test_silent_skill_view_blocks_unrelated_skill_before_dispatch():
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_execution_policy="silent_automation",
    )
    turn_tokens = set_turn_vars(turn_id="silent-skill-scope")
    try:
        response_mode.reset_trusted_skill_execution(
            agent,
            "执行已授权的视频任务",
            explicit_skill_slug="video-edit-workflow-mini",
        )
        wrong_args = {"name": "camsnap"}

        block = response_mode.trusted_skill_operation_block_message(
            agent,
            function_name="skill_view",
            function_args=wrong_args,
        )
        assert block is not None

        dispatched = []
        result = response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="skill_view",
            function_args=wrong_args,
            dispatch=lambda: dispatched.append(True),
        )

        assert dispatched == []
        assert json.loads(result)["trusted_skill_scope_blocked"] is True
    finally:
        clear_turn_vars(turn_tokens)


def test_silent_skill_view_allows_exact_transport_selected_skill():
    agent = SimpleNamespace(
        platform="zet_agent",
        _zet_agent_execution_policy="silent_automation",
    )
    turn_tokens = set_turn_vars(turn_id="silent-skill-scope")
    try:
        response_mode.reset_trusted_skill_execution(
            agent,
            "执行已授权的视频任务",
            explicit_skill_slug="video-edit-workflow-mini",
        )
        exact_args = {"name": "video-edit-workflow-mini"}
        dispatched = []

        assert (
            response_mode.trusted_skill_operation_block_message(
                agent,
                function_name="skill_view",
                function_args=exact_args,
            )
            is None
        )
        result = response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="skill_view",
            function_args=exact_args,
            dispatch=lambda: dispatched.append(True) or '{"success":true}',
        )

        assert dispatched == [True]
        assert json.loads(result)["success"] is True
    finally:
        clear_turn_vars(turn_tokens)
