"""Tests for AIAgent.steer() — mid-run user message injection.

/steer lets the user add a note mid-turn without interrupting the current
tool call. The note is delivered as a REAL ``role:"user"`` message appended
after the tool batch (the legal "ongoing dialog" sequence), so it carries
native user authority — an earlier tool-result-marker delivery was ignored
by weak models (deepseek-v4-flash, 2026-07-13).
"""
from __future__ import annotations

import queue
import threading

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.prompt_builder import STEER_USER_PREFIX, format_steer_user_message
from gateway.platforms.zet_agent import _SteerProducer
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Build an AIAgent without running __init__, then install the steer
    state manually — matches the existing object.__new__ stub pattern
    used elsewhere in the test suite.
    """
    agent = object.__new__(AIAgent)
    agent._pending_steer = []
    agent._pending_steer_lock = threading.Lock()
    agent._pending_redirect = None
    agent._pending_redirect_lock = threading.Lock()
    agent._model_request_active = threading.Event()
    agent._executing_tools = False
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._current_streamed_assistant_text = ""
    agent._stream_needs_break = False
    agent._strip_think_blocks = lambda content: content
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


class TestSteerAcceptance:
    def test_accepts_non_empty_text(self):
        agent = _bare_agent()
        assert agent.steer("go ahead and check the logs") is True
        assert agent._pending_steer == [(None, "go ahead and check the logs")]







class TestSteerDrain:
    def test_drain_returns_and_clears(self):
        agent = _bare_agent()
        agent.steer("hello")
        assert agent._drain_pending_steer() == "hello"
        assert agent._pending_steer == []



class TestActiveTurnRedirect:
    def test_rejects_when_no_turn_is_active(self):
        agent = _bare_agent()
        assert agent.redirect("change course") is False
        assert agent._pending_redirect is None

    def test_cancels_only_an_active_model_request(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("use Postgres") is True
        assert agent._pending_redirect == "use Postgres"
        assert agent._interrupt_requested is True
        assert agent._interrupt_message is None

    def test_multiple_redirects_preserve_message_boundaries(self):
        agent = _bare_agent()
        agent._model_request_active.set()

        assert agent.redirect("first correction") is True
        assert agent.redirect("second correction") is True
        assert agent._pending_redirect == (
            "first correction\n\n"
            "[Additional user correction]\n"
            "second correction"
        )

    def test_hard_interrupt_wins_over_new_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        agent._interrupt_requested = True

        assert agent.redirect("too late") is False
        assert agent._pending_redirect is None

    def test_reasoning_deltas_are_display_only(self):
        """Streamed reasoning must never accumulate into replayable transcript
        state — an assistant checkpoint that inlines chain-of-thought trips
        Anthropic's output classifier and permanently bricks the session
        (deterministic empty-response storms on every replay)."""
        agent = _bare_agent()
        seen = []
        agent.reasoning_callback = seen.append

        agent._fire_reasoning_delta("visible provider thinking")

        # Displayed to the surface, but never checkpointed anywhere.
        assert seen == ["visible provider thinking"]
        assert not getattr(agent, "_current_streamed_reasoning_text", "")

    def test_response_completion_before_redirect_lock_rejects_correction(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        started = threading.Event()
        outcome = {}

        def redirect():
            started.set()
            outcome["accepted"] = agent.redirect("late correction")

        with agent._pending_redirect_lock:
            worker = threading.Thread(target=redirect)
            worker.start()
            assert started.wait(timeout=1)
            # Mirrors conversation_loop clearing the request-active marker
            # under this same lock before redirect can commit its slot.
            agent._model_request_active.clear()
        worker.join(timeout=1)

        assert outcome["accepted"] is False
        assert agent._pending_redirect is None

    def test_hard_stop_wins_concurrent_redirect(self):
        agent = _bare_agent()
        agent._model_request_active.set()
        start = threading.Barrier(3)
        outcome = {}

        def redirect():
            start.wait()
            outcome["redirect"] = agent.redirect("change course")

        def hard_stop():
            start.wait()
            agent.interrupt("stop requested")

        redirect_thread = threading.Thread(target=redirect)
        stop_thread = threading.Thread(target=hard_stop)
        redirect_thread.start()
        stop_thread.start()
        start.wait()
        redirect_thread.join(timeout=1)
        stop_thread.join(timeout=1)

        assert redirect_thread.is_alive() is False
        assert stop_thread.is_alive() is False
        assert agent._interrupt_requested is True
        assert agent._interrupt_message == "stop requested"
        assert agent._pending_redirect is None

    def test_codex_app_server_hard_stop_reaches_native_session(self):
        agent = _bare_agent()
        calls = []
        agent.api_mode = "codex_app_server"
        agent._codex_session = type(
            "_CodexSession",
            (),
            {"request_interrupt": lambda self: calls.append("interrupt")},
        )()

        agent.interrupt()

        assert calls == ["interrupt"]


    def test_redirect_during_tool_execution_uses_safe_steer_boundary(self):
        agent = _bare_agent()
        agent._executing_tools = True

        assert agent.redirect("also check migrations") is True
        assert agent._pending_redirect is None
        assert agent._pending_steer == [(None, "also check migrations")]
        assert agent._interrupt_requested is False


class TestActiveTurnRedirectCheckpoint:
    def test_assistant_tail_puts_correction_last(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        agent._current_streamed_assistant_text = "Visible draft."
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "committed assistant item"},
        ]

        _apply_active_turn_redirect(agent, messages, "Use Postgres instead.")

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Use Postgres instead."
        assert sum(1 for m in messages if m["role"] == "assistant") == 1
        # Scaffolding is provider-replay text, carried in the sidecar so the
        # model still sees the interrupted context — never in the transcript.
        replayed = messages[-1]["api_content"]
        assert "Visible draft." in replayed
        assert "Context from the interrupted assistant response" in replayed
        assert replayed.endswith("Use Postgres instead.")

    def test_scaffolding_never_lands_in_transcript_content(self):
        """The checkpoint machinery is for the MODEL, not the transcript.

        Persisting ``[This response was interrupted by a user correction.]``
        into ``content`` painted raw scaffolding as an assistant bubble on
        every reload. It must ride in ``api_content`` (replayed to the
        provider) while ``content`` stays clean, or be marked
        ``display_kind="hidden"`` when there is no clean form at all.
        """
        from agent.conversation_loop import _apply_active_turn_redirect

        scaffolding = (
            "[This response was interrupted by a user correction.]",
            "Visible response before the interruption:",
            "[Context from the interrupted assistant response]",
        )

        for tail_role in ("tool", "assistant"):
            for streamed in ("Partial reply on screen.", ""):
                agent = _bare_agent()
                agent._current_streamed_assistant_text = streamed
                messages = [{"role": "user", "content": "start"}]
                if tail_role == "assistant":
                    messages.append({"role": "assistant", "content": "committed"})
                else:
                    messages.append(
                        {"role": "assistant", "tool_calls": [{"id": "a"}]}
                    )
                    messages.append(
                        {"role": "tool", "content": "out", "tool_call_id": "a"}
                    )

                _apply_active_turn_redirect(agent, messages, "New direction.")

                for msg in messages:
                    if msg.get("display_kind") == "hidden":
                        continue  # dropped by every transcript surface
                    content = str(msg.get("content", ""))
                    for marker in scaffolding:
                        assert marker not in content, (
                            f"scaffolding leaked into visible content "
                            f"(tail={tail_role}, streamed={bool(streamed)}): {content!r}"
                        )

                # The user's correction is always shown verbatim.
                assert messages[-1]["content"] == "New direction."
                # ...and the model still receives the interrupted context.
                replayed = "".join(
                    str(m.get("api_content") or m.get("content", "")) for m in messages
                )
                assert "[This response was interrupted by a user correction.]" in replayed
                if streamed:
                    assert streamed in replayed

    def test_checkpoint_never_replays_chain_of_thought(self):
        """Raw CoT serialized into checkpoint content reads to Anthropic's
        output classifier as reasoning-injection; because the checkpoint is
        persisted and replayed on every later call, one redirect during a
        thinking phase permanently bricked sessions with deterministic
        empty-response storms (July 2026). Reasoning must never appear in
        replayable content — in either the assistant-checkpoint or the
        merged-user-correction shape."""
        from agent.conversation_loop import _apply_active_turn_redirect

        for tail_role in ("user", "assistant"):
            agent = _bare_agent()
            # Simulate a surface having displayed reasoning this turn.
            agent._current_streamed_reasoning_text = "SECRET chain of thought."
            agent._current_streamed_assistant_text = "Visible draft."
            messages = [{"role": "user", "content": "start"}]
            if tail_role == "assistant":
                messages.append({"role": "assistant", "content": "committed"})

            _apply_active_turn_redirect(agent, messages, "Change course.")

            # Check BOTH the transcript content and the replayed sidecar —
            # the sidecar is what actually reaches the provider.
            serialized = "".join(
                str(m.get("content", "")) + str(m.get("api_content") or "")
                for m in messages
            )
            assert "SECRET chain of thought." not in serialized
            assert "Reasoning shown before the interruption" not in serialized
            assert "Visible draft." in serialized

    def test_checkpoint_omits_reasoning_label_when_nothing_visible(self):
        from agent.conversation_loop import _apply_active_turn_redirect

        agent = _bare_agent()
        agent._current_streamed_reasoning_text = "thinking only, no text yet"
        messages = [{"role": "user", "content": "start"}]

        _apply_active_turn_redirect(agent, messages, "New direction.")

        checkpoint_row = messages[-2]
        # Nothing was on screen, so the row exists only for the model: hidden
        # from every transcript surface, scaffolding replayed via the sidecar.
        assert checkpoint_row["display_kind"] == "hidden"
        assert (
            checkpoint_row["api_content"]
            == "[This response was interrupted by a user correction.]"
        )
        assert messages[-1]["content"] == "New direction."


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
        assert agent._pending_steer == []

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
        assert agent._pending_steer == [(None, "steer")]

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
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None

        # 真实时序：steer 先被接受，用户随后 /stop（flag 置位后 steer()
        # 直接拒收，见 test_steer_refused_while_interrupt_pending）。
        agent.steer("will be dropped")
        agent._pending_redirect = "also drop this"
        assert agent._pending_steer == [(None, "will be dropped")]
        agent._interrupt_requested = True

        agent.clear_interrupt()
        assert agent._pending_steer == []
        assert agent._pending_redirect is None

    def test_clear_interrupt_keeps_named_steer_for_terminal_drop(self):
        agent = _bare_agent()
        agent._interrupt_message = None
        agent._interrupt_thread_signal_pending = False
        agent._execution_thread_id = None
        agent._tool_worker_threads = None
        agent._tool_worker_threads_lock = None
        steer_id = "01998f2d-7c00-7000-8000-000000000001"
        dropped = []
        agent._pending_steer = [(steer_id, "keep me")]
        producer = _SteerProducer(
            agent,
            turn_id="turn-1",
            stream_q=__import__("queue").Queue(),
            binding_token=object(),
            accepted_sender=lambda _payload: None,
            terminal_sender=dropped.append,
            stream_backlog_max=2000,
        )
        agent._steer_admission_hook = producer
        agent._pending_steer = [(steer_id, "keep me")]
        agent._interrupt_requested = True

        agent.clear_interrupt()

        assert agent._pending_steer == [(steer_id, "keep me")]
        assert agent._drain_pending_steer(close=True) is None
        assert dropped == [{
            "type": "steer_dropped",
            "turn_id": "turn-1",
            "steer_id": steer_id,
            "text": "keep me",
        }]


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
        assert agent._pending_steer == []

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

    def test_bound_closing_drain_keeps_empty_slot_for_interrupt_cleanup(self):
        agent = _bare_agent()
        agent._steer_admission_hook = _SteerProducer(
            agent,
            turn_id="turn-1",
            stream_q=queue.Queue(),
            binding_token=object(),
            accepted_sender=lambda _payload: None,
            terminal_sender=lambda _payload: None,
            stream_backlog_max=2000,
        )

        assert agent._drain_pending_steer(close=True) is None
        agent.clear_interrupt()
        assert agent._pending_steer == []

    def test_plain_drain_keeps_slot_open(self):
        agent = _bare_agent()
        agent.steer("first")
        assert agent._drain_pending_steer() == "first"
        assert agent.steer("second") is True
        assert agent._pending_steer == [(None, "second")]

    def test_new_turn_reopens_slot(self):
        agent = _bare_agent()
        agent._drain_pending_steer(close=True)
        assert agent.steer("x") is False
        # run_conversation 入口的复位语义。
        agent._pending_steer = []
        assert agent.steer("x") is True


class TestSteerInterruptRace:
    """/steer 后紧接 /stop：interrupt 旗标可见时绝不能把 pending steer
    注入 messages（会作为未回答 user 消息并入下一轮，复活已取消的指令）。"""

    def test_no_injection_when_interrupt_requested(self):
        # 真实时序：steer 先被接受，interrupt 后到（flag 置位后 steer()
        # 直接拒收）——drain 在锁内看到 flag，不注入、槽位留给 finalizer。
        agent = _bare_agent()
        agent.steer("change direction")
        agent._interrupt_requested = True
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        # 不注入；槽位保留给 interrupt() 丢弃或 finalizer 转 dropped。
        assert len(messages) == 2
        assert agent._pending_steer == [(None, "change direction")]

    def test_steer_refused_while_interrupt_pending(self):
        # 第十一轮 review：/interrupt 之后、agent_task 结束前的停止窗口，
        # steer() 不得再接受——finalizer 的 interrupted 分支丢弃 leftover
        # 且无 steer_dropped 回执，接受等于静默吞话。拒收让 zet 端点回
        # not_running、调用方转排队。
        agent = _bare_agent()
        agent._interrupt_requested = True

        assert agent.steer("stop 窗口的改向") is False
        assert agent._pending_steer == []

    def test_steer_accepts_again_after_interrupt_cleared(self):
        agent = _bare_agent()
        agent._interrupt_requested = True
        assert agent.steer("x") is False
        agent._interrupt_requested = False
        assert agent.steer("新一轮改向") is True
        assert agent._pending_steer == [(None, "新一轮改向")]

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
        assert agent._pending_steer == [(None, "换个方向")]

    def test_no_op_when_tail_is_not_a_steer(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "普通用户消息"},
            {"role": "assistant", "content": "回答"},
        ]
        reclaim_tail_steer(agent, messages)
        assert len(messages) == 2
        assert agent._pending_steer == []

    def test_plain_user_tail_untouched(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [{"role": "user", "content": "首轮问题（非 steer）"}]
        reclaim_tail_steer(agent, messages)
        assert len(messages) == 1
        assert agent._pending_steer == []

    def test_merges_in_front_of_existing_pending(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [format_steer_user_message("先到的")]
        agent.steer("后到的")
        reclaim_tail_steer(agent, messages)
        assert agent._pending_steer == [(None, "先到的"), (None, "后到的")]

    def test_interrupt_discards_instead_of_restash(self):
        # 第十轮 review：注入后用户 /stop（interrupt 清槽置旗标），随后同轮
        # 早退触发 reclaim——消息手术照做（撤出未回答 user 消息），但不得
        # restash 复活已取消的改向。与 drain 的锁内 interrupt 检查对称。
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        agent.steer("换个方向")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        agent._interrupt_requested = True

        reclaim_tail_steer(agent, messages)

        assert messages[-1]["role"] == "tool"  # 注入已撤出
        assert agent._pending_steer == []    # 但不复活


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

    def test_persisted_merge_respects_redaction_and_persist_guard(self):
        # 补写独立行必须走与 flush chokepoint 相同的防护：
        # _redact_message_content 脱敏（steer 里可能粘了 API key）+
        # _persist_disabled 硬停（harness 轮禁写用户会话历史）。
        from agent.prompt_builder import STEER_USER_PREFIX as _P
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent._redact_message_content = lambda c: c.replace("sk-secret", "[REDACTED]")
        agent.steer("用 sk-secret 这个 key")
        messages = [{"role": "user", "content": "首轮", "_db_persisted": True}]
        agent._drain_steer_for_next_api_call(messages)
        assert db.appended[0]["content"] == f"{_P}用 [REDACTED] 这个 key"
        # 内存里的注入内容不脱敏（模型需要原文），只有落库行脱敏。
        assert "sk-secret" in messages[0]["content"]

        agent2 = _bare_agent()
        db2 = self._DbStub()
        agent2._session_db = db2
        agent2.session_id = "sess-2"
        agent2._persist_disabled = True
        agent2.steer("harness 轮的引导")
        messages2 = [{"role": "user", "content": "首轮", "_db_persisted": True}]
        agent2._drain_steer_for_next_api_call(messages2)
        assert db2.appended == []


class TestReclaimMergedShapes:
    """第九轮 review：合并进首轮 user 的 steer 也要能拆回——zet 链路上
    没有任何机制会整条重试失败轮（App 的 steered 气泡不会自动重发），
    留在合并形态里既丢改向又把它持久化进失败轮历史。"""

    def test_reclaims_folded_str_suffix(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        agent.steer("改个方向")
        messages = [{"role": "user", "content": "首轮问题"}]
        agent._drain_steer_for_next_api_call(messages)
        assert f"{STEER_USER_PREFIX}改个方向" in messages[0]["content"]

        reclaim_tail_steer(agent, messages)

        assert messages[0]["content"] == "首轮问题"  # 原文恢复
        assert agent._pending_steer == [(None, "改个方向")]

    def test_reclaims_multimodal_text_block(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        agent.steer("看看颜色")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "图里有什么"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }]
        agent._drain_steer_for_next_api_call(messages)
        assert len(messages[0]["content"]) == 3

        reclaim_tail_steer(agent, messages)

        assert len(messages[0]["content"]) == 2  # steer block 已拆出
        assert agent._pending_steer == [(None, "看看颜色")]

    def test_plain_user_without_steer_untouched(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        messages = [{"role": "user", "content": "普通首轮（无 steer）"}]
        reclaim_tail_steer(agent, messages)
        assert messages[0]["content"] == "普通首轮（无 steer）"
        assert agent._pending_steer == []


class TestConsumedSteerMarker:
    """已消费 steer 的轮必须让 goal judge 按 user-initiated 评估——
    _turn_last_steer_text 由 drain 置位、新轮入口复位。"""

    def test_drain_records_last_steer_text(self):
        agent = _bare_agent()
        agent.steer("暂停一下")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        assert agent._turn_last_steer_text == "暂停一下"

    def test_merged_drain_also_records(self):
        agent = _bare_agent()
        agent.steer("补充要求")
        messages = [{"role": "user", "content": "首轮"}]
        agent._drain_steer_for_next_api_call(messages)
        assert agent._turn_last_steer_text == "补充要求"


class TestReclaimUndoesMergedDbRow:
    """第十轮 review：合并进已持久化 user 的 steer 会即时补写 db 行；同轮
    早退 reclaim 撤回后必须删掉该行——否则失败轮历史留下模型从未消费的
    幽灵 steer 行，客户端重排后下一轮再写一份（/resume 双份）。"""

    class _DbStub:
        def __init__(self):
            self.appended = []
            self.deleted = []

        def append_message(self, **kwargs):
            self.appended.append(kwargs)
            return len(self.appended)

        def delete_message(self, session_id, message_id):
            self.deleted.append((session_id, message_id))
            return True

    def _merged_setup(self):
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent.steer("换个方向")
        messages = [{"role": "user", "content": "首轮", "_db_persisted": True}]
        agent._drain_steer_for_next_api_call(messages)
        assert len(db.appended) == 1
        return agent, db, messages

    def test_reclaim_deletes_phantom_row(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent, db, messages = self._merged_setup()

        reclaim_tail_steer(agent, messages)

        assert db.deleted == [("sess-1", 1)]
        assert agent._steer_merged_db_rows == []
        assert agent._pending_steer == [(None, "换个方向")]

    def test_consumed_merge_row_survives(self):
        # 模型已消费（tail 变 assistant）→ reclaim 不触碰，行保留。
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent, db, messages = self._merged_setup()
        messages.append({"role": "assistant", "content": "答"})

        reclaim_tail_steer(agent, messages)

        assert db.deleted == []
        assert len(db.appended) == 1

    def test_standalone_reclaim_no_delete(self):
        # 独立注入（无补写行）→ reclaim 不做任何删除。
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent = _bare_agent()
        db = self._DbStub()
        agent._session_db = db
        agent.session_id = "sess-1"
        agent.steer("补充")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)

        reclaim_tail_steer(agent, messages)

        assert db.deleted == []

    def test_double_merge_deletes_both_rows(self):
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent, db, messages = self._merged_setup()
        agent.steer("再补一句")
        agent._drain_steer_for_next_api_call(messages)
        assert len(db.appended) == 2

        reclaim_tail_steer(agent, messages)

        assert sorted(db.deleted) == [("sess-1", 1), ("sess-1", 2)]
        assert messages[0]["content"] == "首轮"

    def test_interrupt_discard_still_deletes_phantom_row(self):
        # interrupt 丢弃 restash 时，幽灵行同样要删——文本已离开 messages，
        # DB 必须跟内存一致。
        from agent.agent_runtime_helpers import reclaim_tail_steer
        agent, db, messages = self._merged_setup()
        agent._interrupt_requested = True

        reclaim_tail_steer(agent, messages)

        assert db.deleted == [("sess-1", 1)]
        assert agent._pending_steer == []


class TestReclaimAndHandback:
    """早退 return 用的组合 helper：撤回注入 + 关槽 drain，把文本以
    pending_steer 交还调用方——classic CLI / messaging gateway 只认
    result["pending_steer"]（无槽位 salvage），不交还会滞留缓存 agent、
    下一条无关 prompt 才被 pre-API drain 乱序注入。"""

    def test_returns_reclaimed_text_and_closes_slot(self):
        from agent.agent_runtime_helpers import reclaim_and_handback_steer
        agent = _bare_agent()
        agent.steer("换个方向")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)

        handed = reclaim_and_handback_steer(agent, messages)

        assert handed == "换个方向"
        assert agent._pending_steer is None
        assert messages[-1]["role"] == "tool"
        # 槽位已关（与 finalizer 同契约）：turn 已终局，晚到 steer 拒收，
        # 由调用方转排队。
        assert agent.steer("晚到") is False

    def test_salvages_slot_only_steer(self):
        # steer 在 drain 窗口之后到达（未注入 messages）：也一并交还。
        from agent.agent_runtime_helpers import reclaim_and_handback_steer
        agent = _bare_agent()
        agent.steer("晚到的")
        messages = [{"role": "assistant", "content": "答"}]

        assert reclaim_and_handback_steer(agent, messages) == "晚到的"

    def test_returns_none_when_nothing_pending(self):
        from agent.agent_runtime_helpers import reclaim_and_handback_steer
        agent = _bare_agent()
        messages = [{"role": "assistant", "content": "答"}]
        assert reclaim_and_handback_steer(agent, messages) is None

    def test_interrupt_yields_no_handback(self):
        # interrupt 场景：reclaim 丢弃、槽位也已被 interrupt 清空 → 无交还，
        # 不给已取消的改向任何复活通道。
        from agent.agent_runtime_helpers import reclaim_and_handback_steer
        agent = _bare_agent()
        agent.steer("换个方向")
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "content": "out", "tool_call_id": "a"},
        ]
        agent._drain_steer_for_next_api_call(messages)
        agent._interrupt_requested = True

        assert reclaim_and_handback_steer(agent, messages) is None
