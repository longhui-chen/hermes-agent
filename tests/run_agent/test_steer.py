"""Tests for AIAgent.steer() — mid-run user message injection.

/steer lets the user add a note mid-turn without interrupting the current
tool call. The note is delivered as a REAL ``role:"user"`` message appended
after the tool batch (the legal "ongoing dialog" sequence), so it carries
native user authority — an earlier tool-result-marker delivery was ignored
by weak models (deepseek-v4-flash, 2026-07-13).
"""
from __future__ import annotations

import threading

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.prompt_builder import STEER_USER_PREFIX, format_steer_user_message
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__, then install the steer
    state manually — matches the existing object.__new__ stub pattern
    used elsewhere in the test suite.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    return agent


class TestSteerAcceptance:
    def test_accepts_non_empty_text(self):
        agent = _bare_agent()
        assert agent.steer("go ahead and check the logs") is True
        assert agent._pending_steer == "go ahead and check the logs"

    def test_rejects_empty_string(self):
        agent = _bare_agent()
        assert agent.steer("") is False
        assert agent._pending_steer is None

    def test_rejects_whitespace_only(self):
        agent = _bare_agent()
        assert agent.steer("   \n\t  ") is False
        assert agent._pending_steer is None

    def test_rejects_none(self):
        agent = _bare_agent()
        assert agent.steer(None) is False  # type: ignore[arg-type]
        assert agent._pending_steer is None

    def test_strips_surrounding_whitespace(self):
        agent = _bare_agent()
        assert agent.steer("  hello world  \n") is True
        assert agent._pending_steer == "hello world"

    def test_concatenates_multiple_steers_with_newlines(self):
        agent = _bare_agent()
        agent.steer("first note")
        agent.steer("second note")
        agent.steer("third note")
        assert agent._pending_steer == "first note\nsecond note\nthird note"


class TestSteerDrain:
    def test_drain_returns_and_clears(self):
        agent = _bare_agent()
        agent.steer("hello")
        assert agent._drain_pending_steer() == "hello"
        assert agent._pending_steer is None

    def test_drain_on_empty_returns_none(self):
        agent = _bare_agent()
        assert agent._drain_pending_steer() is None


class TestSteerInjection:
    def test_appends_user_message_after_tool_batch(self):
        agent = _bare_agent()
        agent.steer("please also check auth.log")
        messages = [
            {"role": "user", "content": "what's in /var/log?"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ls output A", "tool_call_id": "a"},
            {"role": "tool", "content": "ls output B", "tool_call_id": "b"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=2)
        # Tool results are untouched — the steer is a NEW user message.
        assert messages[2]["content"] == "ls output A"
        assert messages[3]["content"] == "ls output B"
        assert len(messages) == 5
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == f"{STEER_USER_PREFIX}please also check auth.log"
        # And pending_steer is consumed.
        assert agent._pending_steer is None

    def test_no_op_when_no_steer_pending(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert len(messages) == 2  # unchanged
        assert messages[-1]["content"] == "output"

    def test_no_op_when_num_tool_msgs_zero(self):
        agent = _bare_agent()
        agent.steer("steer")
        messages = [{"role": "user", "content": "hi"}]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=0)
        # Steer should remain pending (nothing to drain into)
        assert agent._pending_steer == "steer"

    def test_steer_carries_user_role_with_mid_task_prefix(self):
        """The steer must land as a real user message (native instruction
        authority — tool-channel text gets ignored by weak models) with the
        mid-task prefix so the model and turn-boundary scans can tell it
        apart from the turn-starting user message.
        """
        agent = _bare_agent()
        agent.steer("stop after next step")
        messages = [{"role": "tool", "content": "x", "tool_call_id": "1"}]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].startswith(STEER_USER_PREFIX)
        assert "stop after next step" in messages[-1]["content"]
        # Tool content untouched.
        assert messages[0]["content"] == "x"

    def test_multimodal_tool_content_untouched(self):
        """Anthropic-style list content on the tool result must stay intact —
        the steer no longer rewrites tool content at all."""
        agent = _bare_agent()
        agent.steer("extra note")
        original_blocks = [{"type": "text", "text": "existing output"}]
        messages = [
            {"role": "tool", "content": list(original_blocks), "tool_call_id": "1"}
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[0]["content"] == original_blocks
        assert messages[-1]["role"] == "user"
        assert "extra note" in messages[-1]["content"]

    def test_batch_end_injection_survives_repair_intact(self):
        """Flow: steer delivered at the END of a multi-tool batch must keep
        every tool result after repair_message_sequence. This is why
        tool_executor only drains at the batch boundary — injecting a user
        message BETWEEN results of one assistant(tool_calls) batch makes
        repair treat the later results as orphans and drop them (PR #173
        review finding)."""
        agent = _bare_agent()
        agent.steer("also check auth.log")
        messages = [
            {"role": "user", "content": "inspect logs"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "out A", "tool_call_id": "a"},
            {"role": "tool", "content": "out B", "tool_call_id": "b"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=2)
        repairs = repair_message_sequence(agent, messages)
        assert repairs == 0
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "tool", "user"]

    def test_mid_batch_user_injection_would_drop_tool_results(self):
        """Documents the failure mode the batch-boundary rule guards
        against: a user message spliced between the results of one batch
        makes repair drop the later tool result. If this behavior ever
        changes, the batch-boundary constraint can be revisited."""
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "inspect logs"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "out A", "tool_call_id": "a"},
            format_steer_user_message("mid-batch"),
            {"role": "tool", "content": "out B", "tool_call_id": "b"},
        ]
        repairs = repair_message_sequence(agent, messages)
        assert repairs > 0
        # tool b got dropped — exactly the data loss we avoid.
        assert all(m.get("tool_call_id") != "b" for m in messages)

    def test_restashed_when_no_tool_result_in_batch(self):
        """If the 'batch' contains no tool-role messages (e.g. all skipped
        after an interrupt), the steer should be put back into the pending
        slot so the caller's fallback path can deliver it."""
        agent = _bare_agent()
        agent.steer("ping")
        messages = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
        # Claim there were N tool msgs, but the tail has none — simulates
        # the interrupt-cancelled case.
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=2)
        # Messages untouched
        assert messages[-1]["content"] == "y"
        # And the steer is back in pending so the fallback can grab it
        assert agent._pending_steer == "ping"


class TestSteerThreadSafety:
    def test_concurrent_steer_calls_preserve_all_text(self):
        agent = _bare_agent()
        N = 200

        def worker(idx: int) -> None:
            agent.steer(f"note-{idx}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        text = agent._drain_pending_steer()
        assert text is not None
        # Every single note must be preserved — none dropped by the lock.
        lines = text.split("\n")
        assert len(lines) == N
        assert set(lines) == {f"note-{i}" for i in range(N)}


class TestSteerClearedOnInterrupt:
    def test_clear_interrupt_drops_pending_steer(self):
        """A hard interrupt supersedes any pending steer — the agent's
        next tool iteration won't happen, so delivering the steer later
        would be surprising."""
        agent = _bare_agent()
        # Minimal surface needed by clear_interrupt()
        agent._interrupt_requested = True
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None

        agent.steer("will be dropped")
        assert agent._pending_steer == "will be dropped"

        agent.clear_interrupt()
        assert agent._pending_steer is None


class TestPreApiCallSteerDrain:
    """Test that steers arriving during an API call are drained before the
    next API call — not deferred until the next tool batch.  This is the
    fix for the scenario where /steer sent during model thinking only lands
    after the agent is completely done."""

    def test_pre_api_drain_appends_user_message(self):
        """If a steer is pending when the main loop starts building
        api_messages, it should be appended as a mid-turn user message
        (mirrors the pre-API drain in run_conversation)."""
        agent = _bare_agent()
        # Simulate messages after a tool batch completed
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "output here", "tool_call_id": "tc1"},
        ]
        # Steer arrives during API call (set after tool execution)
        agent.steer("focus on error handling")
        # Simulate what the pre-API-call drain does:
        _pre_api_steer = agent._drain_pending_steer()
        assert _pre_api_steer == "focus on error handling"
        messages.append(format_steer_user_message(_pre_api_steer))
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == f"{STEER_USER_PREFIX}focus on error handling"
        # Tool result untouched.
        assert messages[2]["content"] == "output here"
        assert agent._pending_steer is None

    def test_pre_api_drain_first_iteration_appends_adjacent_user(self):
        """First-iteration edge: no tool batch yet, the steer still lands as
        a user message right after the turn-starting one — the repair pass
        (repair_message_sequence Pass 2) merges adjacent user messages, so
        role alternation holds."""
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "hello"},
        ]
        agent.steer("early steer")
        _pre_api_steer = agent._drain_pending_steer()
        assert _pre_api_steer == "early steer"
        messages.append(format_steer_user_message(_pre_api_steer))

        from agent.agent_runtime_helpers import repair_message_sequence

        class _RepairAgent:
            session_id = "test"

        repairs = repair_message_sequence(_RepairAgent(), messages)
        assert repairs == 1
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "hello" in messages[0]["content"]
        assert "early steer" in messages[0]["content"]


class TestSteerMessageContract:
    def test_system_prompt_note_describes_the_real_prefix(self):
        """The system-prompt note tells the model which prefix marks a
        mid-task user message; it must reference the exact prefix the
        injector emits, or the model is told to trust a shape that never
        appears (and vice-versa)."""
        from agent.prompt_builder import STEER_CHANNEL_NOTE

        emitted = format_steer_user_message("hi")
        assert emitted["role"] == "user"
        assert emitted["content"].startswith(STEER_USER_PREFIX)
        assert STEER_USER_PREFIX.strip() in STEER_CHANNEL_NOTE

    def test_steer_never_lands_in_tool_channel_labels(self):
        """Regression: tool-channel delivery ('User guidance:' inside tool
        output, and later the OUT-OF-BAND marker) got ignored or refused by
        models — steer must stay a real user message."""
        emitted = format_steer_user_message("hi")
        assert "User guidance:" not in emitted["content"]
        assert "OUT-OF-BAND" not in emitted["content"]


class TestSteerCommandRegistry:
    def test_steer_in_command_registry(self):
        """The /steer slash command must be registered so it reaches all
        platforms (CLI, gateway, TUI autocomplete, Telegram/Slack menus).
        """
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("steer")
        assert cmd is not None
        assert cmd.name == "steer"
        assert cmd.category == "Session"
        assert cmd.args_hint == "<prompt>"

    def test_steer_in_bypass_set(self):
        """When the agent is running, /steer MUST bypass the Level-1
        base-adapter queue so it reaches the gateway runner's /steer
        handler. Otherwise it would be queued as user text and only
        delivered at turn end — defeating the whole point.
        """
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, should_bypass_active_session

        assert "steer" in ACTIVE_SESSION_BYPASS_COMMANDS
        assert should_bypass_active_session("steer") is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


class TestSteerClosedWindow:
    """Finalizer 收尾后的 SSE 拆除窗口：steer 槽位已无消费者，必须拒收
    （调用方转排队），而不是 stash 后静默丢失。"""

    def test_closing_drain_refuses_later_steer(self):
        agent = _bare_agent()
        agent.steer("early")
        assert agent._drain_pending_steer(close=True) == "early"
        # 关闭后拒收 —— 端点据此回 rejected/not_running，LS 转 dropped。
        assert agent.steer("late") is False
        assert agent._pending_steer is None

    def test_plain_drain_keeps_slot_open(self):
        agent = _bare_agent()
        agent.steer("first")
        assert agent._drain_pending_steer() == "first"
        assert agent.steer("second") is True
        assert agent._pending_steer == "second"

    def test_new_turn_reopens_slot(self):
        agent = _bare_agent()
        agent._drain_pending_steer(close=True)
        assert agent.steer("x") is False
        # run_conversation 入口的复位语义。
        agent._steer_closed = False
        assert agent.steer("x") is True


class TestSteerInterruptRace:
    """/steer 后紧接 /stop：interrupt 旗标可见时绝不能把 pending steer
    注入 messages（会作为未回答 user 消息并入下一轮，复活已取消的指令）。"""

    def test_no_injection_when_interrupt_requested(self):
        agent = _bare_agent()
        agent._interrupt_requested = True
        agent.steer("change direction")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        # 不注入；槽位保留给 interrupt() 丢弃或 finalizer 转 dropped。
        assert len(messages) == 2
        assert agent._pending_steer == "change direction"

    def test_injection_proceeds_without_interrupt(self):
        agent = _bare_agent()
        agent._interrupt_requested = False
        agent.steer("go on")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._apply_pending_steer_to_tool_results(messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "user"
        assert "go on" in messages[-1]["content"]
