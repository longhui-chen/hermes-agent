"""Children must never park on present_plan's confirmation gate.

A delegated child has no user: clarify is blocked and no plan card reaches
any UI (plan_emit_callback is never wired for children). Without the forced
auto-execute flag, present_plan returns "stop and wait for the user's
confirmation" and the child burns its iteration budget waiting for a reply
that cannot come (product decision 2026-07-27: children execute directly —
no approval, no clarify, no plan gate).
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import _build_child_agent


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


class TestChildPlanAutoExecute(unittest.TestCase):
    def test_child_gets_plan_auto_execute_flag(self):
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="Research topic A",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
                role="leaf",
            )
        self.assertIs(child._zet_agent_plan_auto_execute, True)

    def test_orchestrator_child_gets_flag_too(self):
        parent = _make_mock_parent()
        with patch("run_agent.AIAgent") as MockAgent:
            MockAgent.return_value = MagicMock()
            child = _build_child_agent(
                task_index=0,
                goal="Coordinate research",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=10,
                parent_agent=parent,
                task_count=1,
                role="orchestrator",
            )
        self.assertIs(child._zet_agent_plan_auto_execute, True)


if __name__ == "__main__":
    unittest.main()


class TestPresentPlanNoCallbackAutoExecute(unittest.TestCase):
    """Flow-level guard: the forced flag must actually change present_plan's
    return path — children have callback=None, and the old code only honored
    auto_execute inside the callback branch (the child parked on "reply to
    confirm" forever)."""

    def test_auto_execute_skips_confirmation_gate(self):
        from tools.plan_tool import present_plan

        out = present_plan(
            title="T",
            groups=[{"label": "Steps", "items": ["do the thing"]}],
            callback=None,
            auto_execute=True,
        )
        self.assertNotIn("reply to confirm", out)
        self.assertIn("proceed", out.lower())
        # The plan text itself must survive as context for the child.
        self.assertIn("do the thing", out)

    def test_default_still_waits_for_confirmation(self):
        from tools.plan_tool import present_plan

        out = present_plan(
            title="T",
            groups=[{"label": "Steps", "items": ["do the thing"]}],
            callback=None,
            auto_execute=False,
        )
        # ⛔ 旧断言钉的是具体措辞 `"reply to confirm"` —— 那是**实现细节**，不是
        # 契约。它把一句「面向用户的第一人称英文话术」钉成了必须存在，于是
        # ZET-3140（中文会话蹦英文）一改文案这条就脆断，反过来逼人把英文话术
        # 加回去。契约是**语义**：这条分支必须告诉模型「要等用户确认、确认前
        # 不许执行」，至于用什么词说不归测试管。
        lowered = out.lower()
        self.assertIn("confirm", lowered)
        self.assertTrue(
            any(marker in lowered for marker in ("do not", "don't", "before")),
            f"默认分支缺少「确认前不要执行」的约束语义：{out!r}",
        )
