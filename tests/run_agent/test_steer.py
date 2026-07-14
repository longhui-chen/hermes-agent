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
    """drain_steer_for_next_api_call — the single injection point, called
    right before each API request is built."""

    def test_appends_user_message_after_tool_batch(self):
        agent = _bare_agent()
        agent.steer("please also check auth.log")
        messages = [
            {"role": "user", "content": "what's in /var/log?"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "ls output A", "tool_call_id": "a"},
            {"role": "tool", "content": "ls output B", "tool_call_id": "b"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        # Tool results are untouched — the steer is a NEW user message.
        assert messages[2]["content"] == "ls output A"
        assert messages[3]["content"] == "ls output B"
        assert len(messages) == 5
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == f"{STEER_USER_PREFIX}please also check auth.log"
        assert agent._pending_steer is None

    def test_no_op_when_no_steer_pending(self):
        agent = _bare_agent()
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "output", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        assert len(messages) == 2  # unchanged
        assert messages[-1]["content"] == "output"

    def test_empty_messages_keeps_steer_pending(self):
        agent = _bare_agent()
        agent.steer("steer")
        messages: list = []
        agent._drain_steer_for_next_api_call(messages)
        assert messages == []
        assert agent._pending_steer == "steer"

    def test_adjacent_str_user_is_merged_not_appended(self):
        """First-iteration edge: the tail is already a user turn. Strict
        providers reject adjacent user messages, so the steer is folded
        into the existing content rather than appended."""
        agent = _bare_agent()
        agent.steer("early steer")
        messages = [{"role": "user", "content": "hello"}]
        agent._drain_steer_for_next_api_call(messages)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "hello" in messages[0]["content"]
        assert f"{STEER_USER_PREFIX}early steer" in messages[0]["content"]

    def test_multimodal_user_gets_text_block_appended(self):
        """Multimodal first turn (list content): repair deliberately skips
        list merges, so the steer must be folded in as a text block here —
        adjacent user(list)+user(str) would 400 on strict providers."""
        agent = _bare_agent()
        agent.steer("also describe the colors")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        }]
        agent._drain_steer_for_next_api_call(messages)
        assert len(messages) == 1
        blocks = messages[0]["content"]
        assert blocks[-1] == {"type": "text", "text": f"{STEER_USER_PREFIX}also describe the colors"}
        # Original blocks untouched.
        assert blocks[0]["text"] == "what is in this image?"
        assert blocks[1]["type"] == "image_url"

    def test_steer_carries_user_role_with_mid_task_prefix(self):
        agent = _bare_agent()
        agent.steer("stop after next step")
        messages = [{"role": "tool", "content": "x", "tool_call_id": "1"}]
        agent._drain_steer_for_next_api_call(messages)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].startswith(STEER_USER_PREFIX)
        assert "stop after next step" in messages[-1]["content"]
        assert messages[0]["content"] == "x"

    def test_batch_end_injection_survives_repair_intact(self):
        """Flow: injecting AFTER the whole tool batch keeps every tool
        result through repair_message_sequence — injecting between results
        of one assistant(tool_calls) batch would make repair drop the later
        ones as orphans (PR #173 review finding)."""
        agent = _bare_agent()
        agent.steer("also check auth.log")
        messages = [
            {"role": "user", "content": "inspect logs"},
            {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
            {"role": "tool", "content": "out A", "tool_call_id": "a"},
            {"role": "tool", "content": "out B", "tool_call_id": "b"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        repairs = repair_message_sequence(agent, messages)
        assert repairs == 0
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "tool", "user"]

    def test_mid_batch_user_injection_would_drop_tool_results(self):
        """Documents the failure mode the single-injection-point rule guards
        against: a user message spliced between the results of one batch
        makes repair drop the later tool result."""
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
        assert all(m.get("tool_call_id") != "b" for m in messages)


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
    """Steers arriving during an API call are drained before the NEXT API
    call — and only there (the per-batch hook was removed so that turns
    which break out of the loop leave the slot pending for the finalizer's
    steer_dropped instead of persisting an unanswered user message)."""

    def test_pre_api_drain_appends_user_message(self):
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "tc1", "function": {"name": "terminal", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "output here", "tool_call_id": "tc1"},
        ]
        agent.steer("focus on error handling")
        agent._drain_steer_for_next_api_call(messages)
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == f"{STEER_USER_PREFIX}focus on error handling"
        assert messages[2]["content"] == "output here"
        assert agent._pending_steer is None

    def test_pending_steer_survives_loop_break_for_finalizer(self):
        """When no next API call happens (present-plan / guardrail / budget
        break), nothing drains the slot mid-loop — the finalizer's closing
        drain hands it back as pending_steer so the caller emits
        steer_dropped and the client re-queues the text."""
        agent = _bare_agent()
        agent.steer("redirect that never lands")
        # No drain call happens on break paths; finalizer drains with close.
        leftover = agent._drain_pending_steer(close=True)
        assert leftover == "redirect that never lands"
        # And the slot is closed against the SSE-teardown window.
        assert agent.steer("too late") is False


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
        agent._drain_steer_for_next_api_call(messages)
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
        agent._drain_steer_for_next_api_call(messages)
        assert messages[-1]["role"] == "user"
        assert "go on" in messages[-1]["content"]


class TestReclaimTailSteer:
    """pre-API drain 与成功模型响应之间的早退（rate-guard / thinking-budget
    exhaustion）：注入的 steer 必须撤回 restash，不得持久化成未回答 user
    消息并入下一轮。"""

    def test_reclaims_injected_steer_and_restashes(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        agent.steer("换个方向")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        assert messages[-1]["role"] == "user"

        reclaim_tail_steer(agent, messages)

        assert len(messages) == 2  # steer 消息已弹出
        assert messages[-1]["role"] == "tool"
        assert agent._pending_steer == "换个方向"

    def test_no_op_when_tail_is_not_a_steer(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "普通用户消息"},
            {"role": "assistant", "content": "回答"},
        ]
        reclaim_tail_steer(agent, messages)
        assert len(messages) == 2
        assert agent._pending_steer is None

    def test_plain_user_tail_untouched(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [{"role": "user", "content": "首轮问题（非 steer）"}]
        reclaim_tail_steer(agent, messages)
        assert len(messages) == 1
        assert agent._pending_steer is None

    def test_merges_in_front_of_existing_pending(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [format_steer_user_message("先到的")]
        agent.steer("后到的")
        reclaim_tail_steer(agent, messages)
        assert agent._pending_steer == "先到的\n后到的"


class TestMergedSteerPersistence:
    """steer 合并进已被 crash-resilience 持久化的首轮 user 消息时，必须补写
    独立 db 行——_persist_session 是 append-only 且跳过已标记消息，否则
    模型看到了 steer 但 SQLite 历史缺失，重启/resume/压缩后丢上下文。"""

    class _DbStub:
        def __init__(self):
            self.appended = []

        def append_message(self, **kwargs):
            self.appended.append(kwargs)

    def test_merge_into_persisted_user_appends_db_row(self):
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent.steer("换个方向")
        messages = [{"role": "user", "content": "首轮问题", "_db_persisted": True}]

        agent._drain_steer_for_next_api_call(messages)

        assert len(messages) == 1  # 内存里仍是合并形态
        assert f"{STEER_USER_PREFIX}换个方向" in messages[0]["content"]
        assert len(db.appended) == 1
        assert db.appended[0]["role"] == "user"
        assert db.appended[0]["content"] == f"{STEER_USER_PREFIX}换个方向"

    def test_merge_into_unpersisted_user_skips_db_append(self):
        # 未持久化的 user：flush 会带着合并后的内容整行写入，无需补行。
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent.steer("补充")
        messages = [{"role": "user", "content": "首轮问题"}]

        agent._drain_steer_for_next_api_call(messages)

        assert db.appended == []

    def test_multimodal_persisted_merge_appends_db_row(self):
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent.steer("看看颜色")
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "图里有什么"}],
            "_db_persisted": True,
        }]

        agent._drain_steer_for_next_api_call(messages)

        assert len(db.appended) == 1
        assert db.appended[0]["content"] == f"{STEER_USER_PREFIX}看看颜色"
