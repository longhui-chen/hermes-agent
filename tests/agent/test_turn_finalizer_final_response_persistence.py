from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn
from hermes_state import SessionDB


class FakeAgent:
    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict[str, Any]] | None = None
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override: Any = None
        self._persist_user_message_timestamp: float | None = None
        self.streamed_deltas = []

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        # Capture the durable write before finalization restores API-local
        # guidance to the returned/live transcript.
        self.persisted_messages = [dict(message) for message in messages]
        return True

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self, close=False):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


class DurableFakeAgent(FakeAgent):
    def __init__(self, db_path):
        super().__init__()
        self._session_db = SessionDB(db_path=db_path)
        self._session_db.create_session(self.session_id, source="test")

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)
        for message in messages:
            if message.get("_db_persisted"):
                continue
            row_id = self._session_db.append_message(
                self.session_id,
                message["role"],
                message.get("content"),
            )
            message["_db_message_id"] = row_id
            message["_db_persisted"] = True
        return True


class FailingPersistenceAgent(FakeAgent):
    def _persist_session(self, messages, conversation_history):
        raise OSError("disk unavailable")


class FailedInPlaceUpdateAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self._session_db = SimpleNamespace(update_message_content=lambda *_a: False)



def test_finalizer_restores_clean_api_local_text_before_return(monkeypatch):
    """One-shot CLI notes do not replay through same-process history."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "[MODEL SWITCH NOTE]\n\nclean prompt"},
        {"role": "assistant", "content": "Done."},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean prompt"
    agent._persist_user_message_timestamp = None

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="[MODEL SWITCH NOTE]\n\nclean prompt",
        original_user_message="clean prompt",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert agent.persisted_messages is not None
    assert agent.persisted_messages[0]["content"] == "clean prompt"
    assert result["messages"][0]["content"] == "clean prompt"




def test_final_response_closes_tool_tail_before_persistence(monkeypatch):
    """A recovered/previewed final response must be durable in session history.

    Regression for turns where the caller receives a non-empty final_response,
    but the message transcript still ends at a tool result. If persisted that
    way, the next turn reloads a stale/malformed history and can appear to loop
    because the assistant's visible final answer is missing from durable state.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert result["messages"][-1] == {"role": "assistant", "content": "Done."}
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == {"role": "assistant", "content": "Done."}


def test_final_response_fills_pure_tool_call_tail(monkeypatch):
    """A tail assistant row that is a *pure tool-call turn* carries no answer.

    The role check alone ("tail is assistant ⇒ nothing to do") leaves the
    #43849/#44100 invariant unmet when the tail is ``assistant(tool_calls)``
    with no text of its own: the caller and the gateway already delivered
    ``final_response``, but it never reaches the transcript. The next turn then
    replays the user backlog and the model re-answers it — the exact symptom
    that block exists to prevent.
    """
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]

    result = finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    persisted = agent.persisted_messages
    assert any(
        m.get("role") == "assistant" and m.get("content") == result["final_response"]
        for m in persisted
    ), "delivered final_response never reached the durable transcript"
    # Filled in place — no assistant→assistant pair, tool_calls preserved.
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert sum(1 for m in persisted if m.get("role") == "assistant") == 1






def test_final_response_fill_invalidates_flush_scan_cursor():
    """The fill's marker pop must invalidate the bounded flush-scan cursor.

    The cursor (run_agent.py) skips the identity-matched prefix of its
    previous snapshot assuming no live dict loses ``_db_persisted`` in place
    — the fill is the one path that pops it. Without invalidation, the
    turn-end flush skips the filled row as 'already stamped' and the
    delivered answer never reaches state.db (the #43849 class resurfacing).
    """
    agent = FakeAgent()
    agent._db_flush_scan_prefix = ["prior-snapshot"]
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
            "_db_persisted": True,
        },
    ]

    finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert agent._db_flush_scan_prefix is None
    persisted = agent.persisted_messages
    assert persisted is not None
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert "_db_persisted" not in persisted[-1], (
        "marker must be popped so the next flush re-writes the filled content"
    )

def test_transformed_response_is_persisted_for_existing_final_delivery(monkeypatch):
    """A transform updates the final response and durable transcript."""
    proposal = "\n\n要不要为你生成创建方案？"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + proposal]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "分析一下"},
        {"role": "assistant", "content": "分析完成。"},
    ]

    result = finalize_turn(
        agent,
        final_response="分析完成。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="分析一下",
        original_user_message="分析一下",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    expected = "分析完成。" + proposal
    assert result["final_response"] == expected
    assert result["messages"][-1]["content"] == expected
    assert agent.persisted_messages[-1]["content"] == expected
    assert agent.streamed_deltas == []
    assert result["response_transformed"] is True
    assert result["response_transform_suffix"] == proposal


def test_transformed_response_survives_cold_session_db_readback(monkeypatch, tmp_path):
    proposal = "\n\n要不要为你生成创建方案？"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + proposal]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = DurableFakeAgent(tmp_path / "state.db")
    messages = [
        {"role": "user", "content": "分析一下"},
        {"role": "assistant", "content": "分析完成。"},
    ]

    finalize_turn(
        agent,
        final_response="分析完成。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="分析一下",
        original_user_message="分析一下",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    cold_db = SessionDB(db_path=tmp_path / "state.db")
    assert cold_db.get_messages(agent.session_id)[-1]["content"] == "分析完成。" + proposal


def test_output_transform_receives_turn_outcome(monkeypatch):
    transform_kwargs = {}
    post_kwargs = {}

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            transform_kwargs.update(kwargs)
        if name == "post_llm_call":
            post_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    agent._user_id = "owner-a"
    agent._user_id_alt = "canonical-owner-a"
    agent.request_overrides = {"response_format": {"type": "json_object"}}
    agent._supports_followup_turns = False
    agent.stream_delta_callback = lambda _delta: None
    messages = [
        {"role": "user", "content": "分析一下"},
        {"role": "assistant", "content": "任务失败。"},
    ]

    result = finalize_turn(
        agent,
        final_response="任务失败。",
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="分析一下",
        original_user_message="分析一下",
        _should_review_memory=False,
        _turn_exit_reason="error_near_max_iterations(provider error)",
    )

    assert transform_kwargs["completed"] is False
    assert transform_kwargs["failed"] is True
    assert transform_kwargs["interrupted"] is False
    assert transform_kwargs["turn_exit_reason"] == "error_near_max_iterations(provider error)"
    assert transform_kwargs["sender_id"] == "canonical-owner-a"
    assert transform_kwargs["structured_output"] is True
    assert transform_kwargs["supports_followup_turns"] is False
    assert transform_kwargs["streaming_output"] is True
    assert post_kwargs["assistant_response"] == "任务失败。"
    assert post_kwargs["sender_id"] == "canonical-owner-a"
    assert post_kwargs["failed"] is True
    assert post_kwargs["supports_followup_turns"] is False
    assert post_kwargs["streaming_output"] is True
    assert result["final_response"] == "任务失败。"
    assert agent.streamed_deltas == []


def test_transformed_persistence_exception_is_reported_without_losing_response(monkeypatch):
    post_kwargs = {}

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + "\n\n确认创建"]
        if name == "post_llm_call":
            post_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FailingPersistenceAgent()
    messages = [
        {"role": "user", "content": "生成方案"},
        {"role": "assistant", "content": "这是草案。"},
    ]

    result = finalize_turn(
        agent,
        final_response="这是草案。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="生成方案",
        original_user_message="生成方案",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"].endswith("确认创建")
    assert post_kwargs["assistant_response"].endswith("确认创建")
    assert any("persist_transformed_session" in item for item in result["cleanup_errors"])


def test_failed_in_place_update_is_reported_without_changing_hook_contract(monkeypatch):
    post_kwargs = {}

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + "\n\n确认创建"]
        if name == "post_llm_call":
            post_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FailedInPlaceUpdateAgent()
    messages = [
        {"role": "user", "content": "生成方案"},
        {
            "role": "assistant",
            "content": "这是草案。",
            "_db_persisted": True,
            "_db_message_id": 7,
        },
    ]

    result = finalize_turn(
        agent,
        final_response="这是草案。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="生成方案",
        original_user_message="生成方案",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert post_kwargs["assistant_response"].endswith("确认创建")
    assert any(
        "update_transformed_session_message" in item
        for item in result["cleanup_errors"]
    )


def test_output_transform_uses_last_chained_result(monkeypatch):
    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + " secret", "safe final"]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "分析一下"},
        {"role": "assistant", "content": "original"},
    ]

    result = finalize_turn(
        agent,
        final_response="original",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="分析一下",
        original_user_message="分析一下",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == "safe final"
    assert result["messages"][-1]["content"] == "safe final"


def test_empty_turn_runs_output_transform_and_persists_its_text(monkeypatch):
    transformed_text = "[plugin transformed output]"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            assert kwargs["response_text"] == ""
            assert kwargs["failed"] is True
            return [transformed_text]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [{"role": "user", "content": "创建它"}]

    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=False,
        failed=True,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="provider_error",
    )

    assert result["final_response"] == transformed_text
    assert result["messages"][-1] == {"role": "assistant", "content": transformed_text}
    assert agent.persisted_messages[-1] == result["messages"][-1]


def test_interrupted_turn_runs_output_transform_without_losing_interrupt_history(monkeypatch):
    transformed_text = "[plugin transformed output]"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            assert kwargs["response_text"] == ""
            assert kwargs["interrupted"] is True
            return [transformed_text]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "创建它"},
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "cancelled"},
    ]

    result = finalize_turn(
        agent,
        final_response="",
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="interrupted",
    )

    assert result["final_response"] == transformed_text
    assert result["messages"][-1]["content"] == "Operation interrupted.\n\n" + transformed_text
    assert agent.persisted_messages[-1] == result["messages"][-1]
