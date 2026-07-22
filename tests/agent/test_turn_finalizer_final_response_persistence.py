from types import SimpleNamespace

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
        self.platform = "api_server"
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
        self.persisted_messages = None
        self.streamed_deltas = []
        self.confirmed_streamed_deltas = []
        self.stream_delivery = True
        self.original_response_streamed = True

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
        self.persisted_messages = list(messages)
        return True

    def _has_stream_consumers(self):
        return True

    def _fire_stream_delta(self, text, *, require_confirmation=False):
        target = (
            self.confirmed_streamed_deltas
            if require_confirmation
            else self.streamed_deltas
        )
        target.append(text)
        return self.stream_delivery

    def _interim_content_was_streamed(self, _text):
        return self.original_response_streamed

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


class FalsePersistenceAgent(FakeAgent):
    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = list(messages)
        return False


class FailedInPlaceUpdateAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self._session_db = SimpleNamespace(update_message_content=lambda *_a: False)


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


def test_transformed_response_is_streamed_once_and_persisted(monkeypatch):
    """A post-stream append must become both visible and durable."""
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
    assert agent.confirmed_streamed_deltas == [proposal]
    assert result["response_transformed"] is True
    assert result["response_transform_streamed"] is True


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


def test_transform_stream_flag_requires_confirmed_callback_delivery(monkeypatch):
    proposal = "\n\n要不要为你生成创建方案？"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + proposal]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    agent.stream_delivery = False
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

    assert result["response_transform_streamed"] is False


def test_transform_suffix_requires_original_response_to_have_streamed(monkeypatch):
    proposal = "\n\n要不要为你生成创建方案？"

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + proposal]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    agent.original_response_streamed = False
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

    assert result["final_response"] == "分析完成。" + proposal
    assert result["response_transform_streamed"] is False
    assert agent.streamed_deltas == []


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
    agent.request_overrides = {"response_format": {"type": "json_object"}}
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
    assert transform_kwargs["sender_id"] == "owner-a"
    assert transform_kwargs["structured_output"] is True
    assert post_kwargs["assistant_response"] == "任务失败。"
    assert post_kwargs["persistence_succeeded"] is True
    assert post_kwargs["sender_id"] == "owner-a"
    assert post_kwargs["failed"] is True
    assert result["final_response"] == "任务失败。"
    assert agent.streamed_deltas == []


def test_post_hook_receives_false_when_transformed_response_is_not_durable(monkeypatch):
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
    assert post_kwargs["persistence_succeeded"] is False
    assert any("persist_transformed_session" in item for item in result["cleanup_errors"])


def test_post_hook_receives_false_when_persistence_reports_append_failure(monkeypatch):
    post_kwargs = {}

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            return [kwargs["response_text"] + "\n\n确认创建"]
        if name == "post_llm_call":
            post_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FalsePersistenceAgent()
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
    assert post_kwargs["persistence_succeeded"] is False
    assert any("persist_transformed_session" in item for item in result["cleanup_errors"])


def test_post_hook_preserves_failed_in_place_update_status(monkeypatch):
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

    finalize_turn(
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

    assert post_kwargs["persistence_succeeded"] is False


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
