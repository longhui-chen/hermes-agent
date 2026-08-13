"""End-to-end tests for inline image inputs on /v1/chat/completions and /v1/responses.

Covers the multimodal normalization path added to the API server.  Unlike the
adapter-level tests that patch ``_run_agent``, these tests patch
``AIAgent.run_conversation`` instead so the adapter's full request-handling
path (including the ``run_agent`` prologue that used to crash on list content)
executes against a real aiohttp app.
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _content_has_image,
    _content_has_visible_payload,
    _extract_current_turn_reference_image,
    _normalize_multimodal_content,
    cors_middleware,
    security_headers_middleware,
)


TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


# ---------------------------------------------------------------------------
# Pure-function tests for _normalize_multimodal_content
# ---------------------------------------------------------------------------


class TestNormalizeMultimodalContent:
    def test_string_passthrough(self):
        assert _normalize_multimodal_content("hello") == "hello"

    def test_none_returns_empty_string(self):
        assert _normalize_multimodal_content(None) == ""

    def test_text_only_list_collapses_to_string(self):
        content = [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]
        assert _normalize_multimodal_content(content) == "hi\nthere"

    def test_responses_input_text_canonicalized(self):
        content = [{"type": "input_text", "text": "hello"}]
        assert _normalize_multimodal_content(content) == "hello"


    def test_input_image_converted_to_canonical_shape(self):
        content = [
            {"type": "input_text", "text": "hi"},
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
        ]
        out = _normalize_multimodal_content(content)
        assert out == [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]


class TestContentHasVisiblePayload:


    def test_list_with_image_only(self):
        assert _content_has_visible_payload([{"type": "image_url", "image_url": {"url": "x"}}])

    def test_image_detection_ignores_text_only_content(self):
        assert _content_has_image([{"type": "image_url", "image_url": {"url": "x"}}])
        assert not _content_has_image([{"type": "text", "text": "hello"}])


class TestCurrentTurnReferenceImage:
    def test_extracts_one_bounded_data_image(self):
        content = _normalize_multimodal_content(
            [{"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}}]
        )
        assert _extract_current_turn_reference_image(content) == TINY_PNG_DATA_URL

    def test_remote_image_does_not_grant_tool_context(self):
        content = _normalize_multimodal_content(
            [{"type": "image_url", "image_url": {"url": "https://example.com/pet.png"}}]
        )
        assert _extract_current_turn_reference_image(content) == ""

    def test_second_image_fails_closed(self):
        content = _normalize_multimodal_content(
            [
                {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
            ]
        )
        assert _extract_current_turn_reference_image(content) == ""

    def test_invalid_or_mismatched_data_image_fails_closed(self):
        content = _normalize_multimodal_content(
            [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,iVBORw0KGgo="}}]
        )
        assert _extract_current_turn_reference_image(content) == ""


# ---------------------------------------------------------------------------
# HTTP integration — real aiohttp client hitting the adapter handlers
# ---------------------------------------------------------------------------


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class TestChatCompletionsMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_adapter_forwards_image_caption_provenance_to_agent(self, adapter):
        image_payload = [
            {"type": "text", "text": "Describe this."},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ]
        agent = MagicMock()
        agent.run_conversation.return_value = {
            "final_response": "A cat.",
            "messages": [],
            "api_calls": 1,
        }
        agent.session_prompt_tokens = 0
        agent.session_completion_tokens = 0
        agent.session_total_tokens = 0
        agent.session_id = None

        with patch.object(adapter, "_create_agent", return_value=agent):
            await adapter._run_agent(
                user_message=image_payload,
                conversation_history=[],
                trusted_user_message=image_payload,
            )

        agent.run_conversation.assert_called_once()
        kwargs = agent.run_conversation.call_args.kwargs
        assert kwargs["user_authored_message"] == image_payload
        assert kwargs["user_message_has_image"] is True

    @pytest.mark.asyncio
    async def test_inline_image_preserved_to_run_agent(self, adapter):
        """Multimodal user content reaches _run_agent as a list of parts."""
        image_payload = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new=MagicMock(),
            ) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "A cat.", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": image_payload}],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["user_message"] == image_payload
            assert mock_run.captured["trusted_user_message"] == image_payload
            assert mock_run.captured["current_turn_reference_image"] == ""

    @pytest.mark.asyncio
    async def test_only_final_user_turn_data_image_is_bound(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "create this pet"},
                                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                                ],
                            },
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            assert mock_run.captured["current_turn_reference_image"] == TINY_PNG_DATA_URL


class TestResponsesMultimodalHTTP:
    @pytest.mark.asyncio
    async def test_input_image_canonicalized_and_forwarded(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()) as mock_run:
                async def _stub(**kwargs):
                    mock_run.captured = kwargs
                    return (
                        {"final_response": "ok", "messages": [], "api_calls": 1},
                        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    )
                mock_run.side_effect = _stub

                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": "Describe."},
                                    {
                                        "type": "input_image",
                                        "image_url": "https://example.com/cat.png",
                                    },
                                ],
                            }
                        ],
                    },
                )

            assert resp.status == 200, await resp.text()
            expected = [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]
            assert mock_run.captured["user_message"] == expected
