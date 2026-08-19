from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import agent.conversation_loop as conversation_loop
import agent.zet_agent_response_mode as response_mode


class HardwareEnrollmentPreflightTest(TestCase):
    def setUp(self) -> None:
        self.agent = SimpleNamespace(platform="zet_agent", request_overrides={})

    def test_current_network_phrases_emit_camera_and_tv_card(self) -> None:
        for user_message in (
            "扫描当前网段",
            "扫描当前网段有哪些设备",
            "帮我扫描下当前网段有哪些硬件设备可以连接",
            "扫描局域网可连接设备",
        ):
            with self.subTest(user_message=user_message):
                response = response_mode.hardware_enrollment_preflight_response(
                    self.agent,
                    user_message,
                )
                self.assertIn("发现附近设备", response)
                self.assertIn('"mode": "current"', response)
                self.assertIn('"resource_kind": "camera"', response)
                self.assertIn('"resource_kind": "tv"', response)
                self.assertNotIn("printer3d", response)
                self.assertNotIn("pc_node", response)

    def test_private_cidr_is_normalized(self) -> None:
        response = response_mode.hardware_enrollment_preflight_response(
            self.agent,
            "扫描 192.168.35.18/24",
        )
        self.assertIn('"cidr": "192.168.35.0/24"', response)

    def test_explicit_discoverable_type_is_preserved(self) -> None:
        camera = response_mode.hardware_enrollment_preflight_response(
            self.agent,
            "发现当前网段的摄像头",
        )
        television = response_mode.hardware_enrollment_preflight_response(
            self.agent,
            "发现当前网段的电视",
        )
        self.assertIn('"resource_kind": "camera"', camera)
        self.assertNotIn('"resource_kind": "tv"', camera)
        self.assertIn('"resource_kind": "tv"', television)
        self.assertNotIn('"resource_kind": "camera"', television)

    def test_unsafe_or_unsupported_scans_stop_before_agent_tools(self) -> None:
        for user_message in (
            "扫描 203.0.113.0/24",
            "扫描 192.168.0.0/16",
            "扫描 192.168.1.0/24 和 192.168.2.0/24",
            "扫描当前网段的打印机",
            "扫描当前网段的电脑",
            "扫描当前网段的端口",
            "用 nmap 扫描当前网段",
        ):
            with self.subTest(user_message=user_message):
                response = response_mode.hardware_enrollment_preflight_response(
                    self.agent,
                    user_message,
                )
                self.assertIn("不会执行", response)
                self.assertNotIn("zettlab-connector-enrollment-intent", response)

    def test_explanation_and_status_questions_are_not_intercepted(self) -> None:
        for user_message in ("解释如何扫描当前网段", "当前网段是什么"):
            with self.subTest(user_message=user_message):
                self.assertEqual(
                    response_mode.hardware_enrollment_preflight_response(
                        self.agent,
                        user_message,
                    ),
                    "",
                )

    def test_subnet_discovery_returns_before_provider_or_tools(self) -> None:
        agent = SimpleNamespace(
            platform="zet_agent",
            request_overrides={},
            max_compression_attempts=3,
            _pending_steer_lock=None,
            _tools_disabled_for_request=False,
        )
        context = SimpleNamespace(
            user_message="扫描当前网段",
            original_user_message="扫描当前网段",
            messages=[{"role": "user", "content": "扫描当前网段"}],
            conversation_history=[],
            active_system_prompt="",
            effective_task_id="hardware-preflight-test",
            turn_id="hardware-preflight-turn",
            current_turn_user_idx=0,
            should_review_memory=False,
            plugin_user_context="",
            ext_prefetch_cache=None,
            preflight_compression_blocked=False,
        )
        finalized: dict[str, object] = {}

        def fake_finalize(_agent, **kwargs):
            finalized.update(kwargs)
            return {
                "final_response": kwargs["final_response"],
                "api_calls": kwargs["api_call_count"],
                "completed": True,
            }

        with (
            patch.object(conversation_loop, "build_turn_context", return_value=context),
            patch.object(
                conversation_loop,
                "_consume_trusted_skill_task_message",
                return_value="扫描当前网段",
            ),
            patch.object(conversation_loop, "_consume_trusted_skill_slug", return_value=""),
            patch.object(conversation_loop, "reset_trusted_skill_execution"),
            patch.object(
                conversation_loop,
                "_should_force_present_plan_tool_choice",
                return_value=False,
            ),
            patch.object(conversation_loop, "_plan_mode_interaction_error", return_value=""),
            patch.object(conversation_loop, "_video_edit_skill_load_error", return_value=""),
            patch("agent.turn_finalizer.finalize_turn", side_effect=fake_finalize),
        ):
            result = conversation_loop.run_conversation(agent, "扫描当前网段")

        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(finalized["api_call_count"], 0)
        self.assertEqual(
            finalized["_turn_exit_reason"],
            "text_response(hardware_enrollment_preflight)",
        )
