from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter
from gateway.session_context import (
    get_session_env,
    pop_zettlab_browser_session_token,
    push_zettlab_browser_session_token,
    zettlab_browser_session_token,
)
from tools import approval, browser_backend_router


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


@pytest.mark.asyncio
async def test_agent_executor_keeps_browser_and_approval_session_scopes_separate(
    monkeypatch,
):
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    lineage_session_id = "api-lineage-session"
    app_session_key = "zettlab:alice:agent-1:chat-1"
    interaction_queue_key = adapter._interaction_queue_key(lineage_session_id)
    seen = {}

    class FakeAgent:
        session_id = lineage_session_id
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, **_kwargs):
            seen["session_key"] = get_session_env("HERMES_SESSION_KEY", "")
            seen["router_session_id"] = browser_backend_router._session_id()
            seen["browser_token"] = zettlab_browser_session_token()
            seen["approval_session_key"] = approval.get_current_session_key()
            return {"final_response": "ok"}

        def _drain_pending_steer(self, *, close):
            assert close is True
            return None

    def fake_create_agent(**kwargs):
        seen["agent_gateway_session_key"] = kwargs.get("gateway_session_key")
        return FakeAgent()

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)
    monkeypatch.setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(schedule_after_turn=lambda *_args, **_kwargs: None),
    )

    browser_token = push_zettlab_browser_session_token("signed-request-scope")
    try:
        result, _usage = await adapter._run_agent(
            user_message="[ZETTLAB:BROWSER_SESSION_SCOPE_TEST]",
            conversation_history=[],
            session_id=lineage_session_id,
            gateway_session_key=app_session_key,
        )
    finally:
        pop_zettlab_browser_session_token(browser_token)

    assert result["final_response"] == "ok"
    assert seen == {
        "agent_gateway_session_key": app_session_key,
        "session_key": app_session_key,
        "router_session_id": app_session_key,
        "browser_token": "signed-request-scope",
        "approval_session_key": interaction_queue_key,
    }
