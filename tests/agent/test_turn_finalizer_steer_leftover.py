"""finalize_turn 的 leftover /steer 归宿（PR #173 第五轮 review）。

正常终局：drain 出的文本进 result["pending_steer"]，调用方（CLI/gateway/
TUI/ACP/zet_agent）重排为下一轮或发 steer_dropped。
中断终局：丢弃——hard interrupt 压倒 pending steer（interrupt() 自身会清
槽，但 interrupt→finalize 窗口里新落的 steer 会重新填上）；回传会让 CLI
的重排把用户刚取消的改向自动执行。
"""

import threading

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 1
    max_total = 90
    remaining = 89


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    def __init__(self, pending_steer=None):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._pending_steer = pending_steer
        self._pending_steer_lock = threading.Lock()
        self._steer_closed = False
        for attr in (
            "session_input_tokens", "session_output_tokens",
            "session_cache_read_tokens", "session_cache_write_tokens",
            "session_reasoning_tokens", "session_prompt_tokens",
            "session_completion_tokens", "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _drain_pending_steer(self, close=False):
        with self._pending_steer_lock:
            text = self._pending_steer
            self._pending_steer = None
            if close:
                self._steer_closed = True
        return text

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _finalize(agent, *, interrupted):
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "done"}],
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason="completed",
    )


def test_normal_finish_hands_back_pending_steer_and_closes_slot():
    agent = _StubAgent(pending_steer="换个方向")
    result = _finalize(agent, interrupted=False)
    assert result["pending_steer"] == "换个方向"
    assert agent._steer_closed is True


def test_interrupted_finish_drops_pending_steer():
    """/steer 后 /stop：interrupt→finalize 窗口残留的 steer 不得回传——
    CLI/gateway 会把 pending_steer 自动重排为下一轮，等于执行了用户刚
    取消的指令。丢弃 + 关槽。"""
    agent = _StubAgent(pending_steer="已被取消的改向")
    result = _finalize(agent, interrupted=True)
    assert "pending_steer" not in result
    assert agent._pending_steer is None
    assert agent._steer_closed is True
