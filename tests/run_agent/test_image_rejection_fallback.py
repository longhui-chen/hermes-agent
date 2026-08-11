"""Tests for the image-rejection fallback in run_agent.

When a server rejects image content (e.g. text-only endpoints), the agent
strips image parts from the retry request and marks the visual input as
unavailable. These tests verify the trusted fallback and the role-alternation
invariants required by providers.
"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import (
    _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION,
    _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE,
    _prepare_image_fallback_attempt,
    _user_image_fallback_state,
)
from agent.transports.types import NormalizedResponse, ToolCall
from run_agent import AIAgent, _strip_images_from_messages


def _tool_def(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_agent() -> AIAgent:
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-image-fallback-test-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=[_tool_def("linear.create_issue")],
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "Base system prompt."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._model_supports_vision = MagicMock(return_value=True)
    return agent


def _response(content="done", *, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_call(name="linear.create_issue"):
    return SimpleNamespace(
        id="call-write",
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


class _ProviderError(Exception):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message}}
        self.response = None


def _image_turn(text: str | None = "The save button is disabled") -> list[dict]:
    parts = []
    if text is not None:
        parts.append({"type": "text", "text": text})
    parts.append(
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,SECRET_IMAGE_BYTES"},
        }
    )
    return parts


def _request_messages(call) -> list[dict]:
    return call.kwargs["messages"]


def _script_provider(agent: AIAgent, *outcomes):
    scripted = iter(outcomes)
    snapshots = []

    def _create(**kwargs):
        snapshots.append(deepcopy(kwargs))
        outcome = next(scripted)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    agent.client.chat.completions.create.side_effect = _create
    return snapshots


def _system_text(messages: list[dict]) -> str:
    system = next(message for message in messages if message.get("role") == "system")
    content = system.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def _has_image(messages: list[dict]) -> bool:
    return any(
        isinstance(part, dict)
        and part.get("type") in {"image", "image_url", "input_image"}
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for part in message["content"]
    )


_UNSET = object()


def _authored_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text") or "")
        for part in content
        if isinstance(part, dict) and part.get("type") in {"text", "input_text"}
    )


def _run(agent: AIAgent, user_message, *, user_authored_message=_UNSET, **kwargs):
    if user_authored_message is _UNSET and _has_image(
        [{"role": "user", "content": user_message}]
    ):
        user_authored_message = _authored_text(user_message)
    if user_authored_message is not _UNSET:
        kwargs["user_authored_message"] = user_authored_message
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(user_message, **kwargs)


class TestTrustedImageFallbackState:
    def test_detects_user_image_and_independent_text(self):
        assert _user_image_fallback_state(_image_turn()) == (True, True)
        assert _user_image_fallback_state(_image_turn(None)) == (True, False)
        assert _user_image_fallback_state(
            [{"type": "input_image", "image_url": "opaque"}]
        ) == (True, False)
        assert _user_image_fallback_state("plain text") == (False, True)

    def test_trusted_media_placeholder_is_not_independent_text(self):
        assert _user_image_fallback_state(_image_turn("[图片]")) == (True, False)
        assert _user_image_fallback_state("[Image attachment]") == (False, False)
        assert _user_image_fallback_state(
            _image_turn("[User attached image: cat.png]")
        ) == (True, False)
        assert _user_image_fallback_state(
            _image_turn("[图片] Save is disabled after entering a title.")
        ) == (True, True)

    def test_attempt_marker_is_system_owned_and_payload_free(self):
        source = [
            {"role": "system", "content": "base"},
            {"role": "user", "content": "text only retry"},
        ]
        tools = [_tool_def("linear.create_issue")]

        messages, retry_tools = _prepare_image_fallback_attempt(
            source,
            tools,
            require_text=False,
        )

        assert messages is not source
        assert source[0]["content"] == "base"
        assert _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION in _system_text(messages)
        assert retry_tools is tools
        marker = _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION.lower()
        for forbidden in (
            "secret_image_bytes",
            "data:image",
            "file://",
            "opaque_ref=",
            "provider payload",
        ):
            assert forbidden not in marker

    def test_image_only_retry_exposes_no_tools(self):
        messages, retry_tools = _prepare_image_fallback_attempt(
            [{"role": "user", "content": ""}],
            [_tool_def("linear.create_issue")],
            require_text=True,
        )

        assert retry_tools == []
        assert "must not call tools" in _system_text(messages).lower()

    def test_concurrent_attempt_preparation_does_not_cross_contaminate(self):
        source_a = [
            {"role": "system", "content": "base-a"},
            {"role": "user", "content": "enough text"},
        ]
        source_b = [
            {"role": "system", "content": "base-b"},
            {"role": "user", "content": ""},
        ]
        tools = [_tool_def("linear.create_issue")]

        with ThreadPoolExecutor(max_workers=2) as executor:
            enough_future = executor.submit(
                _prepare_image_fallback_attempt,
                source_a,
                tools,
                require_text=False,
            )
            missing_future = executor.submit(
                _prepare_image_fallback_attempt,
                source_b,
                tools,
                require_text=True,
            )
            enough_messages, enough_tools = enough_future.result()
            missing_messages, missing_tools = missing_future.result()

        assert source_a[0]["content"] == "base-a"
        assert source_b[0]["content"] == "base-b"
        assert enough_tools is tools
        assert missing_tools == []
        assert "must not call tools" not in _system_text(enough_messages).lower()
        assert "must not call tools" in _system_text(missing_messages).lower()


class TestStripImagesPreservesAlternation:
    """_strip_images_from_messages must not break message role alternation."""

    def test_noop_when_no_images(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is False
        assert msgs == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]




    def test_tool_message_with_all_images_replaced_not_deleted(self):
        """CRITICAL: tool messages must NEVER be deleted — their tool_call_id
        pairs with an assistant tool_call and providers reject unmatched IDs.
        """
        msgs = [
            {"role": "user", "content": "take a screenshot"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "computer_use", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        # Length preserved — tool message NOT deleted
        assert len(msgs) == 3
        # tool_call_id still present
        assert msgs[2]["tool_call_id"] == "call_abc"
        # Content replaced with text placeholder (now a string, not a list)
        assert isinstance(msgs[2]["content"], str)
        assert "image content removed" in msgs[2]["content"].lower()

    def test_tool_message_with_mixed_content_keeps_text_parts(self):
        msgs = [
            {"role": "user", "content": "screenshot plz"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Captured 1024x768"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        assert len(msgs) == 3
        assert msgs[2]["content"] == [{"type": "text", "text": "Captured 1024x768"}]
        assert msgs[2]["tool_call_id"] == "call_1"


    def test_multiple_tool_messages_all_preserved(self):
        """Parallel tool calls: each tool_call_id must retain a paired message."""
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "image_url", "image_url": {}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c2",
                "content": [{"type": "image_url", "image_url": {}}],
            },
        ]
        changed = _strip_images_from_messages(msgs)
        assert changed is True
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert {m["tool_call_id"] for m in tool_msgs} == {"c1", "c2"}




class TestImageRejectionPhraseIsolation:
    """The image-rejection phrase list must NOT false-match on other
    image-related error categories (size-too-large, format errors, etc.)
    so they route to the correct recovery handler (e.g. _try_shrink_image_parts).
    """

    # Reproduces the phrase list used in run_agent.py's error-handler block.
    _REJECTION_PHRASES = (
        "only 'text' content type is supported",
        "only text content type is supported",
        "image_url is not supported",
        "image content is not supported",
        "multimodal is not supported",
        "multimodal content is not supported",
        "multimodal input is not supported",
        "vision is not supported",
        "vision input is not supported",
        "does not support images",
        "does not support image input",
        "does not support multimodal",
        "does not support vision",
        "model does not support image",
        "image_url'. expected",
        "no endpoints found that support image input",
    )

    def _matches(self, body: str) -> bool:
        low = body.lower()
        return any(p in low for p in self._REJECTION_PHRASES)

    def test_anthropic_image_too_large_does_not_trip(self):
        # From agent/error_classifier.py _IMAGE_TOO_LARGE_PATTERNS —
        # these must route to image_too_large / _try_shrink_image_parts_in_messages,
        # NOT to our vision-unsupported fallback.
        bodies = [
            "messages.0.content.1.image.source.base64: image exceeds 5 MB maximum",
            "image too large: 6291456 bytes > 5242880 limit",
            "image_too_large",
            "image size exceeds per-request limit",
        ]
        for body in bodies:
            assert self._matches(body) is False, f"false positive on: {body}"


class TestImageFallbackConversationFlow:
    def test_provider_accepts_image_without_fallback_marker(self):
        agent = _make_agent()
        agent.client.chat.completions.create.return_value = _response("understood")

        result = _run(agent, _image_turn())

        assert result["final_response"] == "understood"
        assert agent.client.chat.completions.create.call_count == 1
        request = _request_messages(agent.client.chat.completions.create.call_args)
        assert _has_image(request) is True
        assert _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION not in _system_text(request)

    def test_image_rejection_retries_with_trusted_marker_and_preserves_source(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "Only 'text' content type is supported"),
            _response("The text is sufficient; continuing without image claims."),
        )

        result = _run(agent, _image_turn())

        assert result["completed"] is True
        assert agent.client.chat.completions.create.call_count == 2
        first, second = requests
        assert _has_image(first["messages"]) is True
        assert _has_image(second["messages"]) is False
        assert (
            _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION
            in _system_text(second["messages"])
        )
        assert second["tools"]
        current_user = next(
            message
            for message in reversed(result["messages"])
            if message.get("role") == "user"
        )
        assert _has_image([current_user]) is True

    def test_cached_agent_pure_image_turns_fail_closed_after_known_rejection(self):
        agent = _make_agent()
        agent._execute_tool_calls = MagicMock()
        first_requests = _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            _response("safe first fallback"),
        )

        first = _run(agent, _image_turn(None))
        assert first["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        assert first_requests[1].get("tools", []) == []
        assert agent._vision_unsupported is True

        agent.client.chat.completions.create.reset_mock()
        second_requests = _script_provider(
            agent,
            _response("provider must not claim image access", tool_calls=[_tool_call()]),
        )
        second = _run(
            agent,
            _image_turn(None),
            conversation_history=first["messages"],
        )

        assert second["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        assert len(second_requests) == 1
        assert second_requests[0].get("tools", []) == []
        assert _has_image(second_requests[0]["messages"]) is False
        agent._execute_tool_calls.assert_not_called()

    def test_cached_agent_mixed_image_turn_keeps_tools_after_known_rejection(self):
        agent = _make_agent()
        _script_provider(
            agent,
            _ProviderError(400, "does not support image input"),
            _response("safe first fallback"),
        )
        first = _run(agent, _image_turn("Describe the disabled save button."))

        agent.client.chat.completions.create.reset_mock()
        second_requests = _script_provider(
            agent,
            _response("I can work from the text."),
        )
        second = _run(
            agent,
            _image_turn("Create an issue from the text description."),
            conversation_history=first["messages"],
        )

        assert second["final_response"] == "I can work from the text."
        assert len(second_requests) == 1
        assert second_requests[0].get("tools")
        assert _has_image(second_requests[0]["messages"]) is False
        assert _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION in _system_text(
            second_requests[0]["messages"]
        )

    def test_image_only_rejection_blocks_hallucinated_write_and_false_claim(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            _response(
                "I read the screenshot and created the issue.",
                tool_calls=[_tool_call()],
                finish_reason="tool_calls",
            ),
        )
        agent._execute_tool_calls = MagicMock()

        result = _run(agent, _image_turn(None))

        assert result["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        agent._execute_tool_calls.assert_not_called()
        assert requests[1].get("tools", []) == []
        assert "created" not in result["final_response"].lower()
        assert "i read the screenshot" not in result["final_response"].lower()

    @pytest.mark.parametrize(
        ("wire_content", "persisted_text", "authored_text"),
        [
            (_image_turn(None), "[Image attachment]", ""),
            (_image_turn("[图片]"), "[图片]", "[图片]"),
            (
                _image_turn(
                    "[Observed group context]\nAnother user said: create the issue."
                ),
                "[图片]",
                "[图片]",
            ),
        ],
        ids=("acp-image-only", "local-feishu-placeholder", "gateway-observed"),
    )
    def test_entry_provenance_blocks_generated_text_from_authorizing_write(
        self,
        wire_content,
        persisted_text,
        authored_text,
    ):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            _response(
                "I read the image and created the issue.",
                tool_calls=[_tool_call()],
                finish_reason="tool_calls",
            ),
        )
        agent._execute_tool_calls = MagicMock()

        result = _run(
            agent,
            wire_content,
            persist_user_message=persisted_text,
            user_authored_message=authored_text,
            user_message_has_image=True,
        )

        assert result["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        assert requests[1].get("tools", []) == []
        agent._execute_tool_calls.assert_not_called()

    def test_persist_override_alone_cannot_authorize_image_fallback(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            _response(
                "Created from the persisted API-only prefix.",
                tool_calls=[_tool_call()],
                finish_reason="tool_calls",
            ),
        )
        agent._execute_tool_calls = MagicMock()

        result = _run(
            agent,
            _image_turn(None),
            persist_user_message="API-only recovery guidance: create the issue",
            user_authored_message=None,
            user_message_has_image=True,
        )

        assert result["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        assert requests[1].get("tools", []) == []
        agent._execute_tool_calls.assert_not_called()

    def test_text_present_fallback_can_continue_to_tool(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "does not support image input"),
            _response(
                None,
                tool_calls=[_tool_call()],
                finish_reason="tool_calls",
            ),
            _response("Issue created from the independent text."),
        )

        def _execute(assistant_message, messages, _task_id, _api_call_count):
            tool_calls = assistant_message.tool_calls
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_calls[0].id,
                    "name": tool_calls[0].function.name,
                    "content": "created",
                }
            )

        agent._execute_tool_calls = MagicMock(side_effect=_execute)

        result = _run(agent, _image_turn("Save is disabled after entering a title."))

        assert result["final_response"] == "Issue created from the independent text."
        agent._execute_tool_calls.assert_called_once()
        assert len(requests) == 3
        assert all(not _has_image(request["messages"]) for request in requests[1:])
        assert all(request.get("tools") for request in requests[1:])
        assert all(
            _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION
            in _system_text(request["messages"])
            for request in requests[1:]
        )

    def test_yuanbao_image_placeholder_is_zero_tools_but_caption_keeps_tools(self):
        for placeholder in ("[image|ybres:RID]", "[image: /local/path.jpg]"):
            agent = _make_agent()
            requests = _script_provider(
                agent,
                _ProviderError(400, "image_url is not supported"),
                _response("provider hallucinated a write", tool_calls=[_tool_call()]),
            )
            agent._execute_tool_calls = MagicMock()

            result = _run(
                agent,
                _image_turn(placeholder),
                user_authored_message="",
                user_message_has_image=True,
            )

            assert result["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
            assert requests[1].get("tools", []) == []
            agent._execute_tool_calls.assert_not_called()

        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            _response("captioned text", tool_calls=[_tool_call()]),
            _response("captioned final"),
        )
        agent._execute_tool_calls = MagicMock(
            side_effect=lambda assistant_message, messages, _task_id, _api_call_count: messages.append(
                {
                    "role": "tool",
                    "tool_call_id": assistant_message.tool_calls[0].id,
                    "name": assistant_message.tool_calls[0].function.name,
                    "content": "created",
                }
            )
        )
        _run(
            agent,
            _image_turn("[image|ybres:RID]"),
            user_authored_message="Create an issue for this screenshot",
            user_message_has_image=True,
        )
        assert requests[1].get("tools")

    def test_non_image_5xx_retry_keeps_image_and_has_no_marker(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(503, "temporary upstream failure"),
            _response("recovered"),
        )

        result = _run(agent, _image_turn())

        assert result["final_response"] == "recovered"
        second = requests[1]["messages"]
        assert _has_image(second) is True
        assert _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION not in _system_text(second)

    def test_image_phrase_without_http_status_does_not_downgrade(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(None, "image_url is not supported"),
            _response("recovered without fallback"),
        )

        result = _run(agent, _image_turn())

        assert result["final_response"] == "recovered without fallback"
        assert all(_has_image(request["messages"]) for request in requests)
        assert all(
            _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION
            not in _system_text(request["messages"])
            for request in requests
        )

    def test_unrelated_4xx_does_not_downgrade_image_turn(self):
        agent = _make_agent()
        requests = _script_provider(
            agent,
            _ProviderError(400, "invalid temperature parameter"),
            _response("recovered without fallback"),
        )

        _run(agent, _image_turn())

        assert requests
        assert all(_has_image(request["messages"]) for request in requests)
        assert all(
            _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION
            not in _system_text(request["messages"])
            for request in requests
        )

    def test_image_turn_suppresses_provider_stream_until_fallback_is_known(self):
        agent = _make_agent()
        visible_deltas = []
        visible_interim = []
        agent.stream_delta_callback = visible_deltas.append
        agent.interim_assistant_callback = (
            lambda text, **_kwargs: visible_interim.append(text)
        )
        streamed_requests = []

        def _unsafe_stream(api_kwargs, **_kwargs):
            streamed_requests.append(deepcopy(api_kwargs))
            agent._fire_stream_delta("I read the image and created the issue.")
            agent._fire_tool_gen_started("linear.create_issue")
            agent._fire_streamed_codex_commentary("I read the image.")
            if len(streamed_requests) == 1:
                raise _ProviderError(400, "image_url is not supported")
            return _response("I read the image and created the issue.")

        agent._interruptible_streaming_api_call = MagicMock(
            side_effect=_unsafe_stream
        )

        result = _run(agent, _image_turn(None))

        assert result["final_response"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
        assert agent._interruptible_streaming_api_call.call_count == 2
        assert not any(isinstance(delta, str) and delta for delta in visible_deltas)
        assert visible_interim == []
        assert len(streamed_requests) == 2

    def test_image_success_releases_buffered_stream_after_provider_accepts(self):
        agent = _make_agent()
        visible_deltas = []
        visible_interim = []
        agent.stream_delta_callback = visible_deltas.append
        agent.interim_assistant_callback = (
            lambda text, **_kwargs: visible_interim.append(text)
        )

        def _accepted_stream(_api_kwargs, **_kwargs):
            agent._fire_stream_delta("understood image")
            agent._fire_streamed_codex_commentary("checking screenshot")
            assert visible_deltas == []
            assert visible_interim == []
            return _response("understood image")

        agent._interruptible_streaming_api_call = MagicMock(
            side_effect=_accepted_stream
        )

        result = _run(agent, _image_turn())

        assert result["final_response"] == "understood image"
        assert visible_deltas == ["understood image"]
        assert visible_interim == ["checking screenshot"]

    def test_concurrent_provisional_streams_do_not_cross_contexts(self):
        agent = _make_agent()
        agent._stream_think_scrubber = None
        agent._stream_context_scrubber = None
        visible_deltas = []
        agent.stream_delta_callback = visible_deltas.append
        attempts_ready = Barrier(2)

        def _attempt(text: str, release: bool) -> int:
            token, events = agent._begin_provisional_stream()
            try:
                agent._fire_stream_delta(text)
                attempts_ready.wait()
            finally:
                agent._end_provisional_stream(token)
            if release:
                agent._release_provisional_stream(events)
            return len(events)

        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted = pool.submit(_attempt, "accepted image", True)
            rejected = pool.submit(_attempt, "rejected image claim", False)

        assert accepted.result() == 1
        assert rejected.result() == 1
        assert visible_deltas == ["accepted image"]

    def test_image_only_replacement_clears_cross_turn_provider_replay_metadata(self):
        agent = _make_agent()
        dirty_raw = _response("dirty")
        clean_raw = _response("clean")
        _script_provider(
            agent,
            _ProviderError(400, "image_url is not supported"),
            dirty_raw,
        )
        dirty = NormalizedResponse(
            content="I read the screenshot and created the issue.",
            tool_calls=[
                ToolCall(
                    id="call-write",
                    name="linear.create_issue",
                    arguments="{}",
                )
            ],
            finish_reason="tool_calls",
            reasoning="discarded reasoning",
            provider_data={
                "reasoning_content": "discarded reasoning content",
                "reasoning_details": [{"signature": "discarded"}],
                "anthropic_content_blocks": [{"type": "tool_use", "id": "bad"}],
                "codex_reasoning_items": [{"id": "rs_bad"}],
                "codex_message_items": [{"id": "msg_bad"}],
            },
        )
        clean = NormalizedResponse(
            content="clean follow-up",
            tool_calls=None,
            finish_reason="stop",
        )
        transport = agent._get_transport()
        original_normalize = transport.normalize_response

        def _normalize(response, **kwargs):
            if response is dirty_raw:
                return dirty
            if response is clean_raw:
                return clean
            return original_normalize(response, **kwargs)

        with patch.object(transport, "normalize_response", side_effect=_normalize):
            first = _run(agent, _image_turn(None))
            first_assistant = first["messages"][-1]
            assert first_assistant["content"] == _IMAGE_UNDERSTANDING_TEXT_REQUIRED_RESPONSE
            assert not first_assistant.get("tool_calls")
            assert not first_assistant.get("reasoning")
            for field in (
                "reasoning_content",
                "reasoning_details",
                "anthropic_content_blocks",
                "codex_reasoning_items",
                "codex_message_items",
            ):
                assert field not in first_assistant

            agent.client.chat.completions.create.reset_mock()
            agent.client.chat.completions.create.side_effect = None
            agent.client.chat.completions.create.return_value = clean_raw
            second = _run(
                agent,
                "normal follow-up",
                conversation_history=first["messages"],
            )

        assert second["final_response"] == "clean follow-up"
        replay = _request_messages(agent.client.chat.completions.create.call_args)
        assert "discarded" not in repr(replay).lower()
        assert "call-write" not in repr(replay)

    def test_moa_fallback_rebuilds_prepared_request_from_text_only_source(self):
        agent = _make_agent()
        agent.provider = "moa"
        requests = _script_provider(
            agent,
            _ProviderError(400, "does not support image input"),
            _response("safe fallback"),
        )
        prepared_inputs = []

        def _prepare(messages):
            prepared_inputs.append(deepcopy(messages))
            marker = (
                "SECRET_IMAGE_DERIVED_REFERENCE"
                if _has_image(messages)
                else "SAFE_TEXT_ONLY_REFERENCE"
            )
            return {
                "messages": [
                    *deepcopy(messages),
                    {"role": "system", "content": marker},
                ]
            }

        agent.client.chat.completions.prepare.side_effect = _prepare

        result = _run(agent, _image_turn("Save is disabled after entering a title."))

        assert result["final_response"] == "safe fallback"
        assert len(prepared_inputs) == 2
        assert _has_image(prepared_inputs[0]) is True
        assert _has_image(prepared_inputs[1]) is False
        second = requests[1]
        assert "SECRET_IMAGE_DERIVED_REFERENCE" not in repr(second)
        assert "SAFE_TEXT_ONLY_REFERENCE" in repr(second)
        assert second["_moa_prepared_request"] == {
            "messages": second["messages"]
        }

    def test_plain_text_turn_snapshot_is_unchanged(self):
        agent = _make_agent()
        agent.client.chat.completions.create.return_value = _response("plain reply")

        result = _run(agent, "plain request")

        request = _request_messages(agent.client.chat.completions.create.call_args)
        assert result["final_response"] == "plain reply"
        assert request == [
            {"role": "system", "content": "Base system prompt."},
            {"role": "user", "content": "plain request"},
        ]

    def test_fallback_state_does_not_leak_into_next_turn_after_exception(self):
        agent = _make_agent()
        _script_provider(
            agent,
            _ProviderError(400, "does not support image input"),
            InterruptedError("cancelled"),
        )

        first_result = _run(agent, _image_turn())
        assert first_result["turn_exit_reason"] == "interrupted_during_api_call"

        agent.client.chat.completions.create.reset_mock()
        agent.client.chat.completions.create.side_effect = None
        agent.client.chat.completions.create.return_value = _response("next turn")
        second_result = _run(agent, "normal follow-up")

        assert second_result["final_response"] == "next turn"
        request = _request_messages(agent.client.chat.completions.create.call_args)
        assert _IMAGE_UNDERSTANDING_UNAVAILABLE_INSTRUCTION not in _system_text(request)
