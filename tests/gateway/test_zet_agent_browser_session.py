from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import zettlab_browser_session_token


@pytest.mark.asyncio
async def test_chat_request_binds_browser_scope_token(monkeypatch):
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    seen = []
    response = object()

    async def base_handler(_self, _request):
        seen.append(zettlab_browser_session_token())
        return response

    monkeypatch.setattr(APIServerAdapter, "_handle_chat_completions", base_handler)
    request = SimpleNamespace(headers={
        "X-Zettlab-Browser-Session-Token": "signed-request-scope",
    })

    assert await adapter._handle_chat_completions(request) is response
    assert seen == ["signed-request-scope"]
    assert zettlab_browser_session_token() == ""


@pytest.mark.asyncio
async def test_chat_request_clears_browser_scope_token_on_error(monkeypatch):
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))

    async def base_handler(_self, _request):
        assert zettlab_browser_session_token() == "signed-request-scope"
        raise RuntimeError("boom")

    monkeypatch.setattr(APIServerAdapter, "_handle_chat_completions", base_handler)
    request = SimpleNamespace(headers={
        "X-Zettlab-Browser-Session-Token": "signed-request-scope",
    })

    with pytest.raises(RuntimeError, match="boom"):
        await adapter._handle_chat_completions(request)
    assert zettlab_browser_session_token() == ""
