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


def test_hardware_enrollment_intent_is_appended_and_persisted(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.platform = "zet_agent"
    agent.api_mode = "chat_completions"
    messages = [
        {"role": "user", "content": "帮我连接下摄像头"},
        {"role": "assistant", "content": "请在连接器页面添加摄像头。"},
    ]

    result = finalize_turn(
        agent,
        final_response="请在连接器页面添加摄像头。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="帮我连接下摄像头",
        original_user_message="帮我连接下摄像头",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["response_transformed"] is True
    assert result["response_transform_suffix"].startswith(
        "\n\n```zettlab-hardware-enrollment-intent\n"
    )
    assert '"requested_types": [\n    "camera"\n  ]' in result["final_response"]
    assert result["messages"][-1]["content"] == result["final_response"]
    assert agent.persisted_messages[-1]["content"] == result["final_response"]


def test_connected_pc_file_result_does_not_append_hardware_enrollment_intent(
    monkeypatch,
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.platform = "zet_agent"
    agent.api_mode = "chat_completions"
    response = "授权目录下共有 38 个条目，其中 30 个文件夹、8 个文件。"
    messages = [
        {"role": "user", "content": "查看下硬件连接中的电脑"},
        {"role": "assistant", "content": response},
    ]

    result = finalize_turn(
        agent,
        final_response=response,
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="查看下硬件连接中的电脑",
        original_user_message="查看下硬件连接中的电脑",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == response
    assert "zettlab-hardware-enrollment-intent" not in result["final_response"]
    assert result["response_transformed"] is False
    assert agent.persisted_messages[-1]["content"] == response


def test_hardware_status_result_removes_stale_enrollment_card_and_persists(
    monkeypatch,
):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    agent.platform = "zet_agent"
    agent.api_mode = "chat_completions"
    visible = "摄像头在线；打印机已连接，但当前对话未授权。"
    response = (
        visible
        + "\n\n```zettlab-hardware-enrollment-intent\n"
        + '{"schema_version":"1","kind":"hardware",'
        + '"requested_types":["camera","printer3d","pc_node"],'
        + '"discovery_requested":true}\n```'
    )
    messages = [
        {"role": "user", "content": "检查所有已连接硬件的状态"},
        {"role": "assistant", "content": response},
    ]

    result = finalize_turn(
        agent,
        final_response=response,
        api_call_count=4,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="检查所有已连接硬件的状态",
        original_user_message="检查所有已连接硬件的状态",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == visible
    assert result["messages"][-1]["content"] == visible
    assert agent.persisted_messages[-1]["content"] == visible
    assert "zettlab-hardware-enrollment-intent" not in result["final_response"]


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
    agent._zet_agent_execution_policy = "silent_automation"
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
    assert transform_kwargs["execution_policy"] == "silent_automation"
    assert transform_kwargs["structured_output"] is True
    assert transform_kwargs["supports_followup_turns"] is False
    assert transform_kwargs["streaming_output"] is True
    assert post_kwargs["assistant_response"] == "任务失败。"
    assert post_kwargs["sender_id"] == "canonical-owner-a"
    assert post_kwargs["execution_policy"] == "silent_automation"
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


def test_authoritative_identity_transform_requires_canonical_delivery(monkeypatch):
    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            kwargs["require_canonical_response"]()
            return [kwargs["response_text"]]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "创建它"},
        {"role": "assistant", "content": "trusted receipt"},
    ]

    result = finalize_turn(
        agent,
        final_response="trusted receipt",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == "trusted receipt"
    assert result["response_transformed"] is False
    assert result["canonical_response_required"] is True


def test_creation_governor_transform_hook_keeps_scope_after_session_rotation(monkeypatch):
    transform_kwargs = {}

    def invoke_hook(name, **kwargs):
        if name == "transform_llm_output":
            transform_kwargs.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    agent._creation_governor_conversation_session_id = "stable-app-conversation"
    agent.session_id = "rotated-transcript-session"
    messages = [
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": "已继续。"},
    ]

    finalize_turn(
        agent,
        final_response="已继续。",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="继续",
        original_user_message="继续",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert transform_kwargs["conversation_session_id"] == "stable-app-conversation"
    assert transform_kwargs["session_id"] == "rotated-transcript-session"


_RECEIPT_MARKER = "<!--creation-recommendation-action-result dGVzdA-->"


def test_receipt_survives_a_later_hook_that_rewrites_the_response(monkeypatch):
    """第三方 transform hook 洗掉回执后，finalizer 必须把它补回来。

    invoke_hook 会把 governor 的结果继续交给后面注册的 hook，finalizer 采用链末
    结果。后续 hook 整体重写响应时 marker 就没了，而 canonical_response_required
    仍会让这段文本作为 canonical_final_response 发出——Web 关联不上回执，已经被
    Hermes 接管的 create 会永久停在「不确定」，用户只能去别处核对。
    """

    def invoke_hook(name, **kwargs):
        if name != "transform_llm_output":
            return []
        kwargs["require_canonical_response"](_RECEIPT_MARKER)
        # 链上第一段是 governor 产出的（带回执），第二段是后续 hook 的整体重写。
        return [
            kwargs["response_text"] + "\n\n" + _RECEIPT_MARKER,
            "这段是第三方插件重写后的正文。",
        ]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "创建它"},
        {"role": "assistant", "content": "好的"},
    ]

    result = finalize_turn(
        agent,
        final_response="好的",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["canonical_response_required"] is True
    assert _RECEIPT_MARKER in result["final_response"], (
        "回执被后续 hook 洗掉且没有补回——canonical 终态会带着一段没有回执的文本发出"
    )
    assert "第三方插件重写后的正文" in result["final_response"], "补回执不该把后续 hook 的改写丢掉"


def test_receipt_is_not_synthesised_when_no_canonical_response_was_required(monkeypatch):
    """对照：没有回执要发的普通轮次，不能凭空往正文里塞 marker。"""

    def invoke_hook(name, **kwargs):
        if name != "transform_llm_output":
            return []
        return [kwargs["response_text"] + "\n\n" + _RECEIPT_MARKER, "重写后的正文。"]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "随便聊聊"},
        {"role": "assistant", "content": "好的"},
    ]

    result = finalize_turn(
        agent,
        final_response="好的",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="随便聊聊",
        original_user_message="随便聊聊",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["canonical_response_required"] is False
    assert _RECEIPT_MARKER not in result["final_response"]


_FOREIGN_RECEIPT = "<!--creation-recommendation-action-result b3RoZXI-->"


def test_a_later_hook_cannot_swap_in_a_different_receipt(monkeypatch):
    """后置 hook 换掉回执时，链末必须还原成 governor 那个原值。

    marker 只要语法合法就能骗过「链末还有没有 marker」的判断。换成指向别的
    proposal 的 marker，Web 会去结算另一张卡片。
    """

    def invoke_hook(name, **kwargs):
        if name != "transform_llm_output":
            return []
        kwargs["require_canonical_response"](_RECEIPT_MARKER)
        return [
            kwargs["response_text"] + "\n\n" + _RECEIPT_MARKER,
            "第三方重写。\n\n" + _FOREIGN_RECEIPT,
        ]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    result = finalize_turn(
        agent,
        final_response="好的",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "创建它"}, {"role": "assistant", "content": "好的"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert _RECEIPT_MARKER in result["final_response"], "governor 的原始回执没有被还原"
    assert _FOREIGN_RECEIPT not in result["final_response"], (
        "后置 hook 塞进来的回执被当成可信值发了出去——Web 会去结算别的卡片"
    )


def test_a_later_hook_cannot_append_a_conflicting_receipt(monkeypatch):
    """追加冲突 marker 同样要被清掉：两个回执并存会让 Web 永久失败关闭。"""

    def invoke_hook(name, **kwargs):
        if name != "transform_llm_output":
            return []
        kwargs["require_canonical_response"](_RECEIPT_MARKER)
        return [
            kwargs["response_text"] + "\n\n" + _RECEIPT_MARKER,
            kwargs["response_text"] + "\n\n" + _RECEIPT_MARKER + "\n\n" + _FOREIGN_RECEIPT,
        ]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    result = finalize_turn(
        agent,
        final_response="好的",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "创建它"}, {"role": "assistant", "content": "好的"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="创建它",
        original_user_message="创建它",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"].count("creation-recommendation-action-result") == 1
    assert _RECEIPT_MARKER in result["final_response"]
    assert _FOREIGN_RECEIPT not in result["final_response"]


def test_sanitisation_only_turn_does_not_borrow_another_hooks_marker(monkeypatch):
    """governor 只清洗、本轮没有回执时，链末不能从别的 hook 结果里补一个 marker。

    伪造 marker 被剥掉的轮次同样会置 canonical_response_required。若此时后置
    hook 追加了任意语法合法的 marker，把它当成权威回执发出去会让 Web 去结算
    另一张卡片。
    """

    def invoke_hook(name, **kwargs):
        if name != "transform_llm_output":
            return []
        # 只做清洗：没有回执可交，传 None。
        kwargs["require_canonical_response"](None)
        return [
            kwargs["response_text"],
            "第三方追加。\n\n" + _FOREIGN_RECEIPT,
        ]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    agent = FakeAgent()
    result = finalize_turn(
        agent,
        final_response="好的",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "随便说说"}, {"role": "assistant", "content": "好的"}],
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="随便说说",
        original_user_message="随便说说",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["canonical_response_required"] is True
    assert "creation-recommendation-action-result" not in result["final_response"], (
        "本轮没有回执，却把后置 hook 的 marker 当成权威值发了出去"
    )
