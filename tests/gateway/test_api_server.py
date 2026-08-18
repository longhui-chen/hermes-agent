"""
Tests for the OpenAI-compatible API server gateway adapter.

Tests cover:
- Chat Completions endpoint (request parsing, response format)
- Responses API endpoint (request parsing, response format)
- previous_response_id chaining (store/retrieve)
- Auth (valid key, invalid key, no key configured)
- /v1/models endpoint
- /health endpoint
- System prompt extraction
- Error handling (invalid JSON, missing fields)
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.browser_state_preview import MAX_PREVIEW_BYTES
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import api_server as api_server_module
from gateway.platforms.api_server import (
    APIServerAdapter,
    ResponseStore,
    _IdempotencyCache,
    _derive_chat_session_id,
    _extract_creation_action_receipt_transport,
    _has_creation_recommendation_wrapper,
    _hermes_version,
    _redact_api_error_text,
    _request_agent_overrides,
    _extract_connector_route_capability,
    _redact_api_error_text,
    _tool_completion_payload,
    check_api_server_requirements,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# check_api_server_requirements
# ---------------------------------------------------------------------------


class TestCheckRequirements:

    @patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", False)
    def test_returns_false_without_aiohttp(self):
        assert check_api_server_requirements() is False


class TestToolCompletionPayload:
    def test_projects_bounded_browser_state_without_raw_snapshot(self):
        payload = _tool_completion_payload(
            "call_browser_state",
            "browser_navigate",
            json.dumps(
                {
                    "success": True,
                    "url": "https://user:pass@Example.com/account?token=private#fragment",
                    "title": "Account",
                    "snapshot": '- heading "Account" [e1]\n- button "Save" [e2]',
                    "element_count": 2,
                    "ui_hint": {
                        "type": "takeover_browser",
                        "agent_id": "agent-1",
                        "browser_session_id": "session-1",
                        "tab_id": "tab-1",
                    },
                },
            ),
        )

        assert payload["browserState"] == {
            "version": 1,
            "source": "navigate",
            "browserSessionId": "session-1",
            "url": {"hostname": "example.com"},
            "title": "Account",
            "elements": [
                {"role": "heading", "label": "Account"},
                {"role": "button", "label": "Save"},
            ],
            "summary": "Account",
            "elementCount": 2,
        }
        assert "snapshot" not in payload

    @patch(
        "gateway.platforms.api_server.project_browser_state_preview",
        side_effect=UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogate"),
    )
    def test_preview_failure_keeps_the_tool_completion(self, _project_preview):
        payload = _tool_completion_payload(
            "call_browser_state",
            "browser_navigate",
            {"success": True, "title": "malformed"},
        )

        assert payload == {
            "tool": "browser_navigate",
            "toolCallId": "call_browser_state",
            "status": "completed",
            "outcome": "success",
        }

    def test_native_browser_vision_flow_emits_safe_text_only(self):
        payload = _tool_completion_payload(
            "call_native_vision",
            "browser_vision",
            {
                "_multimodal": True,
                "text_summary": (
                    "Image attached natively. "
                    "Screenshot path: /volume1/agents/main/browser_screenshot.png"
                ),
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,private-image"},
                    }
                ],
                "meta": {
                    "screenshot_path": "/volume1/agents/main/browser_screenshot.png"
                },
            },
        )

        assert payload["browserState"] == {
            "version": 1,
            "source": "vision",
            "summary": "Image attached natively.",
        }
        encoded = json.dumps(payload, ensure_ascii=False)
        assert "base64" not in encoded
        assert "/volume1" not in encoded

    def test_emits_only_bounded_takeover_hint_for_live_clients(self):
        payload = _tool_completion_payload(
            "call_browser_1",
            "browser_navigate",
            json.dumps({
                "success": True,
                "snapshot": "sensitive page content",
                "ui_hint": {
                    "type": "takeover_browser",
                    "agent_id": "agent-1",
                    "browser_session_id": "session-1",
                    "tab_id": "tab-1",
                    "extra": "ignored",
                },
            }),
        )

        assert payload["ui_hint"] == {
            "type": "takeover_browser",
            "agent_id": "agent-1",
            "browser_session_id": "session-1",
            "tab_id": "tab-1",
        }
        assert "snapshot" not in payload

    def test_rejects_takeover_hint_from_a_foreign_tool(self):
        """Only the browser tool that builds a hint may project one.

        Any MCP server, connector or plugin can return arbitrary JSON; without
        the tool-name gate its output would reach the App as a genuine handoff
        entry point aimed at whatever agent/session/tab it names.
        """
        hint = {
            "type": "takeover_browser",
            "agent_id": "victim-agent",
            "browser_session_id": "victim-session",
            "tab_id": "victim-tab",
        }
        for foreign_tool in ("mcp__evil__lookup", "execute_code", "terminal", "web_search"):
            payload = _tool_completion_payload(
                "call_foreign",
                foreign_tool,
                json.dumps({"success": True, "ui_hint": hint}),
            )
            assert "ui_hint" not in payload, foreign_tool

    def test_rejects_incomplete_takeover_hint(self):
        payload = _tool_completion_payload(
            "call_browser_2",
            "browser_navigate",
            json.dumps({"success": True, "ui_hint": {"type": "takeover_browser", "agent_id": "agent-1"}}),
        )
        assert "ui_hint" not in payload

    def test_preserves_takeover_hint_when_navigation_fails(self):
        payload = _tool_completion_payload(
            "call_browser_failed",
            "browser_navigate",
            json.dumps({
                "success": False,
                "error": "browser_runtime_unavailable",
                "tabId": "tab-blank",
                "ui_hint": {
                    "type": "takeover_browser",
                    "agent_id": "agent-1",
                    "browser_session_id": "session-1",
                    "tab_id": "tab-blank",
                },
            }),
        )

        assert payload["outcome"] == "error"
        assert payload["error"] == "browser_runtime_unavailable"
        assert payload["ui_hint"] == {
            "type": "takeover_browser",
            "agent_id": "agent-1",
            "browser_session_id": "session-1",
            "tab_id": "tab-blank",
        }

    def test_media_completion_keeps_only_bounded_artifact_fields(self):
        payload = _tool_completion_payload(
            "call_image_1",
            "image_generate",
            json.dumps({
                "success": True,
                "image": "/mnt/data/agents/data/main/output/session/image.jpg",
                "prompt": "private prompt must not enter progress events",
                "assets": [{"url": "https://cdn.example/image.jpg"}],
            }),
        )

        assert payload["output"] == {
            "success": True,
            "image": "/mnt/data/agents/data/main/output/session/image.jpg",
        }
        assert "prompt" not in payload["output"]
        assert "assets" not in payload["output"]

    def test_promotes_connector_error_printed_by_execute_code_output(self):
        printed = {
            "ok": False,
            "error": {
                "code": "connector_runtime_auth_required",
                "message": "ZETTLAB_CONNECTORS_AUTH_TOKEN is not set for this skill session.",
            },
            "connector_error": {
                "code": "connector_runtime_auth_required",
                "errorCode": "connector_runtime_auth_required",
                "provider": "weread",
                "nextAction": {"type": "refresh_connector_context"},
            },
        }
        payload = _tool_completion_payload(
            "call_exec_1",
            "execute_code",
            json.dumps({
                "status": "error",
                "error": "Script exited with code 1",
                "output": json.dumps(printed),
            }),
        )

        assert payload["outcome"] == "error"
        assert payload["errorCode"] == "connector_runtime_auth_required"
        assert payload["provider"] == "weread"
        assert payload["connector_error"]["provider"] == "weread"
        assert payload["connector_error"]["nextAction"]["type"] == "refresh_connector_context"

    def test_promotes_json_rpc_connector_error_data_from_terminal_output(self):
        printed = {
            "error": {
                "code": -32000,
                "message": "Connector disabled for this chat.",
                "data": {
                    "errorCode": "denied_by_chat_override",
                    "nextAction": {
                        "type": "enable_chat_override",
                        "provider": "linear",
                        "toolName": "linear.list_issues",
                    },
                },
            },
        }
        payload = _tool_completion_payload(
            "call_terminal_1",
            "terminal",
            json.dumps({
                "status": "error",
                "error": "Command failed",
                "output": json.dumps(printed),
            }),
        )

        assert payload["outcome"] == "error"
        assert payload["errorCode"] == "denied_by_chat_override"
        assert payload["provider"] == "linear"
        connector_error = payload["connector_error"]
        assert connector_error["errorCode"] == "denied_by_chat_override"
        assert connector_error["provider"] == "linear"
        assert connector_error["nextAction"]["type"] == "enable_chat_override"
        assert connector_error["nextAction"]["toolName"] == "linear.list_issues"


# ---------------------------------------------------------------------------
# _redact_api_error_text — guards every outward error site (envelopes, SSE
# error events, cron-endpoint 500 bodies) that routes raw exception text to
# authenticated HTTP clients. #37733
# ---------------------------------------------------------------------------


class TestRedactApiErrorText:
    def test_masks_secret_value_but_preserves_structure(self):
        secret = "sk-api-server-leak-1234567890"
        out = _redact_api_error_text(Exception(f"auth failed OPENAI_API_KEY={secret}"))
        assert secret not in out
        assert "OPENAI_API_KEY=" in out

    def test_redacts_regardless_of_global_redaction_setting(self):
        # force=True must mask even when global redaction is disabled.
        secret = "sk-forced-redaction-0987654321"
        with patch("agent.redact._REDACT_ENABLED", False):
            out = _redact_api_error_text(Exception(f"boom AWS_SECRET_ACCESS_KEY={secret}"))
        assert secret not in out

    def test_limit_truncates_after_redaction(self):
        assert len(_redact_api_error_text("x" * 500, limit=50)) == 50


# ---------------------------------------------------------------------------
# ResponseStore
# ---------------------------------------------------------------------------


class TestResponseStore:
    def test_put_and_get(self):
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        assert store.get("resp_1") == {"output": "hello"}

    def test_get_missing_returns_none(self):
        store = ResponseStore(max_size=10)
        assert store.get("resp_missing") is None

    def test_lru_eviction(self):
        store = ResponseStore(max_size=3)
        store.put("resp_1", {"output": "one"})
        store.put("resp_2", {"output": "two"})
        store.put("resp_3", {"output": "three"})
        # Adding a 4th should evict resp_1
        store.put("resp_4", {"output": "four"})
        assert store.get("resp_1") is None
        assert store.get("resp_2") is not None
        assert len(store) == 3


    def test_delete_clears_conversation_mapping(self):
        """Deleting a response also removes conversation mappings that reference it."""
        store = ResponseStore(max_size=10)
        store.put("resp_1", {"output": "hello"})
        store.set_conversation("chat-a", "resp_1")
        assert store.get_conversation("chat-a") == "resp_1"
        store.delete("resp_1")
        assert store.get_conversation("chat-a") is None


# ---------------------------------------------------------------------------
# _IdempotencyCache
# ---------------------------------------------------------------------------


class TestIdempotencyCache:
    def test_business_execution_scope_digest_isolated_and_non_secret(self):
        body = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "render"}],
            "stream": False,
        }
        keys = ["model", "messages", "stream"]
        token_a = "a" * 64
        token_b = "b" * 64
        legacy_subset = {key: body.get(key) for key in keys}
        legacy_fingerprint = hashlib.sha256(
            repr(legacy_subset).encode("utf-8")
        ).hexdigest()

        assert (
            api_server_module._make_request_fingerprint(body, keys)
            == legacy_fingerprint
        )
        digest_a = api_server_module._business_execution_scope_digest(token_a)
        digest_b = api_server_module._business_execution_scope_digest(token_b)
        assert digest_a == api_server_module._business_execution_scope_digest(
            token_a
        )
        assert digest_a != digest_b
        fingerprint_a = api_server_module._make_request_fingerprint(
            body, keys, execution_scope_digest=digest_a
        )
        fingerprint_b = api_server_module._make_request_fingerprint(
            body, keys, execution_scope_digest=digest_b
        )
        assert fingerprint_a != fingerprint_b
        assert token_a not in repr((digest_a, fingerprint_a))
        assert token_b not in repr((digest_b, fingerprint_b))

    @pytest.mark.asyncio
    async def test_concurrent_same_key_and_fingerprint_runs_once(self):
        cache = _IdempotencyCache()
        gate = asyncio.Event()
        started = asyncio.Event()
        calls = 0

        async def compute():
            nonlocal calls
            calls += 1
            started.set()
            await gate.wait()
            return ("response", {"total_tokens": 1})

        first = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))
        second = asyncio.create_task(cache.get_or_set("idem-key", "fp-1", compute))

        await started.wait()
        assert calls == 1

        gate.set()
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == second_result == ("response", {"total_tokens": 1})


# ---------------------------------------------------------------------------
# Adapter initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_default_config(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 8642
        assert adapter._api_key == ""
        assert adapter.platform == Platform.API_SERVER

    def test_custom_config_from_extra(self):
        config = PlatformConfig(
            enabled=True,
            extra={
                "host": "0.0.0.0",
                "port": 9999,
                "key": "sk-test",
                "cors_origins": ["http://localhost:3000"],
            },
        )
        adapter = APIServerAdapter(config)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 9999
        assert adapter._api_key == "sk-test"
        assert adapter._cors_origins == ("http://localhost:3000",)


    def test_create_agent_forwards_runtime_config(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai-codex",
                "base_url": "https://example.test/v1",
                "api_mode": "codex_responses",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5.5")
        monkeypatch.setattr(
            "gateway.run._load_gateway_config",
            lambda: {
                "agent": {"reasoning_effort": "xhigh"},
                "checkpoints": {
                    "enabled": True,
                    "max_snapshots": 7,
                    "max_total_size_mb": 321,
                    "max_file_size_mb": 4,
                },
            },
        )
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {"enabled": True, "effort": "xhigh"}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["reasoning_config"] == {"enabled": True, "effort": "xhigh"}
        assert captured["checkpoints_enabled"] is True
        assert captured["checkpoint_max_snapshots"] == 7
        assert captured["checkpoint_max_total_size_mb"] == 321
        assert captured["checkpoint_max_file_size_mb"] == 4

    def test_create_agent_refreshes_max_iterations_from_runtime_config(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai",
                "base_url": "https://example.test/v1",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5")
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"max_turns": 200}})
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 200)
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["max_iterations"] == 200

    def test_create_agent_disables_tools_for_tool_choice_none(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = [
                    {"type": "function", "function": {"name": "present_plan"}},
                    {"type": "function", "function": {"name": "todo"}},
                ]
                self.valid_tool_names = {"present_plan", "todo"}
                self._skip_mcp_refresh = False

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openai",
                "base_url": "https://example.test/v1",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5")
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {}),
        )
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_fallback_model",
            staticmethod(lambda: None),
        )
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(
            session_id="meeting-summary",
            request_overrides={
                "tool_choice": "none",
                "response_format": {"type": "json_object"},
            },
        )

        assert agent.tools == []
        assert agent.valid_tool_names == set()
        assert agent._skip_mcp_refresh is True
        assert captured["request_overrides"] == {
            "response_format": {"type": "json_object"},
        }

    def test_create_agent_handles_fallback_model_kwarg_collision(self, monkeypatch):
        """When the primary provider auth-fails, _resolve_runtime_agent_kwargs()
        returns a runtime dict that carries its own ``model`` key. _create_agent
        must pop it and let it override the config model — otherwise the explicit
        ``model=`` collides with ``**runtime_kwargs`` and every request 500s with
        "got multiple values for keyword argument 'model'"."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
                "model": "anthropic/claude-haiku",  # from the fallback entry
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "primary/model")
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Must not raise TypeError on the duplicate 'model' kwarg.
        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        # Fallback model overrides the config model, mirroring the native path.
        assert captured["model"] == "anthropic/claude-haiku"

    def test_create_agent_keeps_config_model_when_runtime_omits_it(self, monkeypatch):
        """Happy path (no fallback active): runtime_kwargs has no 'model', so the
        resolved gateway model is used unchanged. Regression guard for the pop."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "primary/model")
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
        monkeypatch.setattr(
            "gateway.run.GatewayRunner._load_reasoning_config",
            staticmethod(lambda: {}),
        )
        monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
        monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
        monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["model"] == "primary/model"


# ---------------------------------------------------------------------------
# Auth checking
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_configured_allows_all(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {}
        assert adapter._check_auth(mock_request) is None


    def test_non_ascii_bearer_token_returns_401_not_500(self):
        """A non-ASCII byte in the bearer token must be rejected with 401, not
        crash the handler: hmac.compare_digest raises TypeError on a str with
        non-ASCII characters, and the token is raw client input."""
        config = PlatformConfig(enabled=True, extra={"key": "sk-test123"})
        adapter = APIServerAdapter(config)
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer ské-not-the-key"}
        result = adapter._check_auth(mock_request)  # must not raise
        assert result is not None
        assert result.status == 401


# ---------------------------------------------------------------------------
# Concurrency cap (gateway.api_server.max_concurrent_runs) — #7483
# ---------------------------------------------------------------------------


class TestConcurrencyCap:

    def test_resolve_reads_config_value(self):
        cfg = {"gateway": {"api_server": {"max_concurrent_runs": 3}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            assert APIServerAdapter._resolve_max_concurrent_runs() == 3


    def test_under_cap_returns_none(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 5
        adapter._inflight_agent_runs = 2
        assert adapter._concurrency_limited_response() is None

    def test_at_cap_returns_429_with_retry_after(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 3
        adapter._inflight_agent_runs = 3
        resp = adapter._concurrency_limited_response()
        assert resp is not None
        assert resp.status == 429
        assert resp.headers.get("Retry-After")


# ---------------------------------------------------------------------------
# Helpers for HTTP tests
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app from the adapter (without starting the full server)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    app.router.add_get("/v1/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_get("/api/model/options", adapter._handle_model_options)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/v1/skills", adapter._handle_skills)
    app.router.add_get("/v1/toolsets", adapter._handle_toolsets)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post(
        "/v1/chat/completions/canonical-final-v1",
        adapter._handle_canonical_final_chat_completions,
    )
    app.router.add_post("/v1/responses", adapter._handle_responses)
    app.router.add_get("/v1/responses/{response_id}", adapter._handle_get_response)
    app.router.add_delete("/v1/responses/{response_id}", adapter._handle_delete_response)
    app.router.add_post(
        "/api/platforms/{platform}/events",
        adapter._handle_platform_event_callback,
    )
    return app


class _FakeGoogleChatAdapter:
    def __init__(self, *, verify_ok: bool = True, verify_code: str = ""):
        self.verify_ok = verify_ok
        self.verify_code = verify_code
        self.dispatched = []

    def verify_http_event_request(self, auth_header: str):
        self.auth_header = auth_header
        return self.verify_ok, self.verify_code

    async def dispatch_http_event(self, payload):
        self.dispatched.append(payload)
        return {"ok": True}


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


# ---------------------------------------------------------------------------
# Adapter internals
# ---------------------------------------------------------------------------


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_run_agent_uses_session_id_as_task_id(self, adapter):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 1
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 3

        model_options = {"reasoning": {"enabled": False}, "fast": False}
        with patch.object(adapter, "_create_agent", return_value=mock_agent) as mock_create_agent:
            result, usage = await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-123",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options=model_options,
            )

        # _run_agent annotates result with the effective agent.session_id
        # when it's a real string, so the response-header writer can track
        # compression-triggered session rotations (#16938). The mock agent
        # here doesn't set an explicit session_id string so the guard skips
        # the annotation — header will fall back to the provided session_id.
        assert result["final_response"] == "ok"
        assert usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        create_kwargs = mock_create_agent.call_args.kwargs
        assert create_kwargs["requested_model"] == "MiniMax-M3"
        assert create_kwargs["requested_provider"] == "minimax"
        assert create_kwargs["model_options"] == model_options
        mock_agent.run_conversation.assert_called_once_with(
            user_message="hello",
            conversation_history=[],
            task_id="session-123",
        )

    @pytest.mark.asyncio
    async def test_run_agent_sets_and_clears_process_ownership_markers(self, adapter):
        """#76188 review: this surface runs its own agent lifecycle outside
        TurnRunner, so it needs its own baseline snapshot/clear — verify the
        markers _reap_disconnected_agent_processes() reads are actually
        populated during the turn and cleared once it finishes."""
        mock_agent = MagicMock()
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        captured = {}

        def _capture_markers(**_kwargs):
            captured["task_id"] = mock_agent._gateway_turn_process_task_id
            captured["baseline"] = mock_agent._gateway_turn_process_baseline
            return {"final_response": "ok"}

        mock_agent.run_conversation.side_effect = _capture_markers

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter._run_agent(
                user_message="hello",
                conversation_history=[],
                session_id="session-456",
                requested_model="MiniMax-M3",
                requested_provider="minimax",
                model_options={"reasoning": {"enabled": False}, "fast": False},
            )

        assert captured["task_id"] == "session-456"
        assert isinstance(captured["baseline"], frozenset)
        # Turn completed normally — markers must be cleared so a disconnect
        # arriving after this point can't reap work this turn left running.
        assert mock_agent._gateway_turn_process_task_id == ""
        assert mock_agent._gateway_turn_process_baseline == frozenset()


class TestDisconnectedAgentReap:
    """#76188 review: SSE disconnect handlers must reap only the background
    processes the disconnected turn created, and must no-op when no turn
    ownership was ever recorded on the agent."""

    def test_reaps_baseline_diff_for_owned_turn(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="session-abc",
            _gateway_turn_process_baseline=frozenset({"proc-1"}),
        )

        _reap_disconnected_agent_processes(agent)

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [
            ("session-abc", frozenset({"proc-1"}), "api_server_sse_disconnect")
        ]

    def test_noop_when_agent_has_no_ownership_markers(self, monkeypatch):
        from gateway.platforms.api_server import _reap_disconnected_agent_processes
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True),
        )
        agent = types.SimpleNamespace(
            _gateway_turn_process_task_id="",
            _gateway_turn_process_baseline=None,
        )

        _reap_disconnected_agent_processes(agent)

        time.sleep(0.1)
        assert calls == []

    def test_stale_epoch_skips_reap_when_newer_run_claimed_task_id(self, monkeypatch):
        """#76188 follow-up: concurrent API runs can share a client-provided
        session_id (same task_id). A disconnecting run whose epoch has been
        superseded must NOT kill the newer run's processes."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run_a = types.SimpleNamespace()
        run_b = types.SimpleNamespace()
        _publish_turn_process_ownership(run_a, "shared-session")
        # Run B claims the same session_id — supersedes A's epoch.
        _publish_turn_process_ownership(run_b, "shared-session")

        _reap_disconnected_agent_processes(run_a)
        time.sleep(0.2)
        assert calls == [], "stale run A must not reap run B's processes"

        # Run B disconnecting IS current — its reap proceeds.
        _reap_disconnected_agent_processes(run_b)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]
        _clear_turn_process_ownership(run_b)

    def test_reap_proceeds_when_own_clear_pruned_the_epoch_entry(self, monkeypatch):
        """A missing epoch entry (the abandoned run's own finally already
        cleared it) means no newer claimant — the reap must proceed using a
        pre-captured marker snapshot, or the leak survives."""
        from gateway.platforms.api_server import (
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
            _reap_disconnected_agent_processes,
        )
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda *a, **k: calls.append(True) or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        run = types.SimpleNamespace()
        _publish_turn_process_ownership(run, "solo-session")
        # Simulate the disconnect handler capturing the agent while the
        # worker's finally clears ownership: snapshot markers, then clear.
        stale_view = types.SimpleNamespace(
            _gateway_turn_process_task_id=run._gateway_turn_process_task_id,
            _gateway_turn_process_baseline=run._gateway_turn_process_baseline,
            _gateway_turn_process_epoch=run._gateway_turn_process_epoch,
        )
        _clear_turn_process_ownership(run)

        _reap_disconnected_agent_processes(stale_view)
        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [True]

    def test_publish_and_clear_ownership_roundtrip(self, monkeypatch):
        from gateway.platforms.api_server import (
            _TURN_PROCESS_EPOCHS,
            _clear_turn_process_ownership,
            _publish_turn_process_ownership,
        )
        from tools.process_registry import process_registry

        monkeypatch.setattr(
            process_registry,
            "snapshot_running_ids",
            lambda tid: frozenset({f"pre-{tid}"}),
        )

        agent = types.SimpleNamespace()
        _publish_turn_process_ownership(agent, "sess-rt")
        assert agent._gateway_turn_process_task_id == "sess-rt"
        assert agent._gateway_turn_process_baseline == frozenset({"pre-sess-rt"})
        assert isinstance(agent._gateway_turn_process_epoch, int)
        assert "sess-rt" in _TURN_PROCESS_EPOCHS

        _clear_turn_process_ownership(agent)
        assert agent._gateway_turn_process_task_id == ""
        assert agent._gateway_turn_process_baseline == frozenset()
        assert agent._gateway_turn_process_epoch is None
        # Entry pruned — dict stays bounded to in-flight runs.
        assert "sess-rt" not in _TURN_PROCESS_EPOCHS

    @pytest.mark.asyncio
    async def test_stop_run_reaps_owned_processes(self, adapter, monkeypatch):
        """POST /v1/runs/{id}/stop abandons the run — it must reap the
        background processes that run created (#76115 sibling surface)."""
        from gateway.platforms.api_server import _publish_turn_process_ownership
        from tools.process_registry import process_registry

        calls = []
        monkeypatch.setattr(
            process_registry,
            "kill_started_since",
            lambda task_id, baseline, *, source: calls.append(
                (task_id, baseline, source)
            )
            or 1,
        )
        monkeypatch.setattr(
            process_registry, "snapshot_running_ids", lambda _tid: frozenset()
        )

        agent = MagicMock()
        _publish_turn_process_ownership(agent, "run-stop-sess")
        adapter._active_run_agents["run_x"] = agent

        request = MagicMock()
        request.match_info = {"run_id": "run_x"}
        resp = await adapter._handle_stop_run(request)
        assert resp.status == 200

        deadline = time.time() + 1.0
        while not calls and time.time() < deadline:
            time.sleep(0.01)
        assert calls == [("run-stop-sess", frozenset(), "api_server_run_stop")]
        agent.interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_agent_preserves_tool_choice_none_as_request_boundary(
        self,
        adapter,
    ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def _run_conversation(**_kwargs):
            assert mock_agent._tools_disabled_for_request is True
            return {"final_response": "ok"}

        mock_agent.run_conversation.side_effect = _run_conversation

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            result, _usage = await adapter._run_agent(
                user_message="请把 [file: /data/input.mp4] 剪辑成 vlog",
                conversation_history=[],
                session_id="no-tools-video",
                request_overrides={"tool_choice": "none"},
            )

        assert result["final_response"] == "ok"


class TestRunEventCallback:

    @pytest.mark.asyncio
    async def test_subagent_events_redact_secrets_and_carry_child_session(self, adapter):
        """Free-text fields (goal/summary/output_tail/preview) must pass the
        forced secret redaction before hitting the public /v1/runs stream,
        and child_session_id must survive the allowlist so clients can
        correlate the child's session."""
        run_id = "run_subagent_redact"
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_statuses.pop(run_id, None)

        callback = adapter._make_run_event_callback(run_id, loop)
        secret = "sk-proj-abcdef1234567890abcdef1234567890abcdef12"
        callback(
            "subagent.complete",
            preview=f"leaked {secret}",
            goal=f"use key {secret} to fetch data",
            subagent_id="deleg_999",
            child_session_id="child-sess-42",
            status="completed",
            summary=f"exported OPENAI_API_KEY={secret} then ran",
            output_tail=f"env shows {secret}",
        )

        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["child_session_id"] == "child-sess-42"
        for field in ("preview", "goal", "summary", "output_tail"):
            assert secret not in event[field], field


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("a" * 43, "a" * 43),
        ("A-_0" * 10 + "abc", "A-_0" * 10 + "abc"),
        ("short", ""),
        ("a" * 44, ""),
        ("a" * 42 + "=", ""),
        (123, ""),
    ],
)
def test_extract_connector_route_capability_accepts_only_fixed_base64url(raw, want):
    assert _extract_connector_route_capability(
        {"metadata": {"connector_route_capability": raw}}
    ) == want


@pytest.mark.parametrize(
    ("metadata", "want"),
    [
        ({"creation_action_receipt_transport": "canonical_final_v1"}, "canonical_final_v1"),
        ({"creation_action_receipt_transport": "canonical_final_v2"}, ""),
        ({"creation_action_receipt_transport": 1}, ""),
        ({}, ""),
    ],
)
def test_extract_creation_action_receipt_transport_is_fail_closed(metadata, want):
    assert _extract_creation_action_receipt_transport({"metadata": metadata}) == want


def _body_with_action(payload_json: str) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "确认创建\n\n[creation_recommendation_response]\n"
                    + payload_json
                    + "\n[/creation_recommendation_response]"
                ),
            }
        ]
    }


@pytest.mark.parametrize(
    "payload_json",
    [
        # 严格解析认得的规范形态
        '{"version":1,"type":"creation_recommendation_response","action":"create",'
        '"creation_type":"agent","proposal_id":"p1","title":"T","dedup_key":"d1"}',
        # governor 会先规范化再接受，但严格解析不认：大写 action
        '{"version":1,"type":"creation_recommendation_response","action":"CREATE",'
        '"creation_type":"agent","proposal_id":"p1","title":"T","dedup_key":"d1"}',
        # 同上：别名 creation_type
        '{"version":1,"type":"creation_recommendation_response","action":"create",'
        '"creation_type":"scheduled-task","proposal_id":"p1","title":"T","dedup_key":"d1"}',
        # 连 JSON 都不合法——照样不能让普通端点带着 receipt transport 过去
        "{not json at all",
    ],
)
def test_plain_endpoint_never_keeps_receipt_transport_for_any_action_wrapper(payload_json):
    """降级边界必须比 governor 的接受面更宽。

    严格解析器要求 action 小写、creation_type 属于固定三项；governor 会先规范化
    （CREATE → create、scheduled-task → task）再接受。两边判据不一致时，一个
    「严格解析不认、governor 认」的 payload 打到普通 /v1/chat/completions 上，
    transport 不会被清除，普通端点就能改 proposal、拉起原生创建流程并产出可信
    回执——版本化端点这道门等于白设。
    """
    assert _has_creation_recommendation_wrapper(_body_with_action(payload_json)) is True


def test_wrapper_probe_covers_multimodal_text_parts():
    """多模态 content 是 API 正式接受的形态，降级边界必须一起覆盖。

    wrapper 藏在 parts 数组的 text part 里时，只看标量字符串会漏判，transport
    不被清除；而 governor 对整个列表做 str() 之后照样能解析出 JSON wrapper，
    普通端点于是能拉起原生创建流程并产出可信回执。
    """
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {
                        "type": "text",
                        "text": (
                            "确认创建\n\n[creation_recommendation_response]\n"
                            '{"version":1,"type":"creation_recommendation_response",'
                            '"action":"create","creation_type":"agent",'
                            '"proposal_id":"p1","title":"T","dedup_key":"d1"}\n'
                            "[/creation_recommendation_response]"
                        ),
                    },
                ],
            }
        ]
    }
    assert _has_creation_recommendation_wrapper(body) is True


def test_wrapper_probe_ignores_multimodal_without_the_envelope():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图里是什么"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
    }
    assert _has_creation_recommendation_wrapper(body) is False


def test_wrapper_probe_ignores_bodies_without_the_envelope():
    """对照：没有信封的普通聊天不受影响，不该被误清 transport。"""
    assert _has_creation_recommendation_wrapper(
        {"messages": [{"role": "user", "content": "今天天气怎么样"}]}
    ) is False
    assert _has_creation_recommendation_wrapper({"messages": "not a list"}) is False
    assert _has_creation_recommendation_wrapper({}) is False


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_security_headers_present(self, adapter):
        """Responses should include basic security headers."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Content-Security-Policy") == "default-src 'none'; frame-ancestors 'none'"
            assert resp.headers.get("Permissions-Policy") == "camera=(), microphone=(), geolocation=()"
            assert resp.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"
            assert resp.headers.get("X-XSS-Protection") == "0"
            assert resp.headers.get("Referrer-Policy") == "no-referrer"


    @pytest.mark.asyncio
    async def test_health_reports_version(self, adapter):
        """GET /health must expose a non-empty version so orchestrators (e.g.
        AgentOS) can read the gateway version without scraping. Regression
        guard for the missing-version gap."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert "version" in data
            assert isinstance(data["version"], str)
            assert data["version"] != ""


# ---------------------------------------------------------------------------
# /health/detailed endpoint
# ---------------------------------------------------------------------------


class TestHealthDetailedEndpoint:
    @pytest.mark.asyncio
    async def test_health_detailed_returns_ok(self, adapter):
        """GET /health/detailed returns status, platform, and runtime fields."""
        app = _create_app(adapter)
        with patch("gateway.status.read_runtime_status", return_value={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "active_agents": 2,
            "exit_reason": None,
            "updated_at": "2026-04-14T00:00:00Z",
        }), patch("gateway.run._resolve_gateway_model", return_value="test/model"), patch(
            "gateway.platforms.api_server.collect_runtime_readiness",
            return_value={"status": "ok"},
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health/detailed")
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["platform"] == "hermes-agent"
                assert data["gateway_state"] == "running"
                assert data["platforms"] == {"telegram": {"state": "connected"}}
                assert data["active_agents"] == 2
                # Derived busy/drainable: this endpoint is served BY the live
                # gateway, so running + 2 agents ⇒ busy and drainable.
                assert data["gateway_busy"] is True
                assert data["gateway_drainable"] is True
                assert isinstance(data["pid"], int)
                assert "updated_at" in data


    @pytest.mark.asyncio
    async def test_public_health_does_not_run_readiness_probes(self, adapter):
        app = _create_app(adapter)
        with patch("gateway.platforms.api_server.collect_runtime_readiness") as probe:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/health")
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"
        probe.assert_not_called()


    def test_readiness_work_counts_include_stopping_runs(self, adapter):
        """Regression: _handle_stop_run() sets status="stopping" and holds it
        there — cooperatively, with no hard timeout — until the agent notices
        the interrupt and the task actually exits. A run in that window is
        still doing real executor-thread work and must count as active,
        the same as "running"; excluding it undercounts active_api_runs for
        the whole (now-unbounded) cooperative-stop duration."""
        adapter._run_statuses = {
            "queued": {"status": "queued"},
            "running": {"status": "running"},
            "approval": {"status": "waiting_for_approval"},
            "stopping": {"status": "stopping"},
            "done": {"status": "completed"},
            "cancelled": {"status": "cancelled"},
        }

        with patch("tools.process_registry.process_registry.completion_queue.qsize", return_value=0), \
             patch("tools.async_delegation.active_count", return_value=0):
            assert adapter._readiness_work_counts() == (4, 0, 0)


# ---------------------------------------------------------------------------
# /v1/models endpoint
# ---------------------------------------------------------------------------


class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_models_returns_hermes_agent(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "hermes-agent"
            assert data["data"][0]["owned_by"] == "hermes"

    @pytest.mark.asyncio
    async def test_models_returns_profile_name(self):
        """When running under a named profile, /v1/models advertises the profile name."""
        with patch("gateway.platforms.api_server.APIServerAdapter._resolve_model_name", return_value="lucas"):
            adapter = _make_adapter()
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            assert resp.status == 200
            data = await resp.json()
            assert data["data"][0]["id"] == "lucas"
            assert data["data"][0]["root"] == "lucas"


    def test_resolve_model_name_default_profile(self):
        """Default profile falls back to 'hermes-agent'."""
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            assert APIServerAdapter._resolve_model_name("") == "hermes-agent"


    @pytest.mark.asyncio
    async def test_model_options_returns_shared_inventory(self, adapter, monkeypatch):
        """GET /api/model/options builds the shared picker payload off-loop."""
        from hermes_cli import inventory

        ctx = object()
        payload = {
            "providers": [{"slug": "nous", "name": "Nous Portal", "models": ["gpt-5.5"]}],
            "model": "gpt-5.5",
            "provider": "nous",
        }
        seen = {"thread_calls": 0}

        monkeypatch.setattr(inventory, "load_picker_context", lambda: ctx)

        def fake_build_model_options_payload(received_ctx, **kwargs):
            seen["ctx"] = received_ctx
            seen["kwargs"] = kwargs
            return payload

        async def fake_to_thread(func, *args, **kwargs):
            seen["thread_calls"] += 1
            return func(*args, **kwargs)

        monkeypatch.setattr(
            inventory,
            "build_model_options_payload",
            fake_build_model_options_payload,
        )
        monkeypatch.setattr(
            "gateway.platforms.api_server.asyncio.to_thread",
            fake_to_thread,
        )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/model/options?refresh=true")
            assert resp.status == 200
            data = await resp.json()

        assert data == payload
        assert seen["thread_calls"] == 1
        assert seen["ctx"] is ctx
        assert seen["kwargs"] == {
            "include_unconfigured": True,
            "refresh": True,
        }


# ---------------------------------------------------------------------------
# /v1/capabilities endpoint
# ---------------------------------------------------------------------------


class TestCapabilitiesEndpoint:
    @pytest.mark.asyncio
    async def test_capabilities_advertises_plugin_safe_contract(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "hermes.api_server.capabilities"
            assert data["platform"] == "hermes-agent"
            assert data["model"] == "hermes-agent"
            assert data["auth"]["type"] == "bearer"
            assert data["auth"]["required"] is False
            assert data["runtime"]["mode"] == "server_agent"
            assert data["runtime"]["tool_execution"] == "server"
            assert data["runtime"]["split_runtime"] is False
            assert "API-server host" in data["runtime"]["description"]
            assert data["features"]["chat_completions"] is True
            assert data["features"]["run_status"] is True
            assert data["features"]["run_events_sse"] is True
            assert data["features"]["model_options"] is True
            assert data["features"]["session_continuity_header"] == "X-Hermes-Session-Id"
            assert data["endpoints"]["run_status"]["path"] == "/v1/runs/{run_id}"
            assert data["endpoints"]["model_options"] == {"method": "GET", "path": "/api/model/options"}
            assert data["endpoints"]["skills"] == {"method": "GET", "path": "/v1/skills"}
            assert data["endpoints"]["toolsets"] == {"method": "GET", "path": "/v1/toolsets"}


# ---------------------------------------------------------------------------
# /v1/skills and /v1/toolsets endpoints
# ---------------------------------------------------------------------------


class TestSkillsEndpoint:
    @pytest.mark.asyncio
    async def test_skills_returns_list_envelope(self, adapter):
        fake_skills = [
            {"name": "github", "description": "GitHub workflow skill", "category": "github"},
            {"name": "ascii-art", "description": "ASCII art generation", "category": "creative"},
        ]
        with patch(
            "tools.skills_tool._find_all_skills",
            return_value=list(fake_skills),
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/skills")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                names = sorted(s["name"] for s in data["data"])
                assert names == ["ascii-art", "github"]
                for entry in data["data"]:
                    assert set(entry.keys()) >= {"name", "description", "category"}


class TestToolsetsEndpoint:
    @pytest.mark.asyncio
    async def test_toolsets_returns_resolved_tools(self, adapter):
        fake_toolsets = [
            ("default", "Default Tools", "Core tools"),
            ("web", "Web Tools", "Search and extract"),
        ]
        feature_snapshot = object()
        with patch(
            "hermes_cli.tools_config._get_effective_configurable_toolsets",
            return_value=fake_toolsets,
        ), patch(
            "hermes_cli.tools_config._get_platform_tools",
            return_value={"default"},
        ), patch(
            "hermes_cli.tools_config.get_nous_subscription_features",
            return_value=feature_snapshot,
        ) as resolve_features, patch(
            "hermes_cli.tools_config._toolset_has_keys",
            return_value=True,
        ) as has_keys, patch(
            "toolsets.resolve_toolset",
            side_effect=lambda name: {
                "default": ["terminal", "read_file"],
                "web": ["web_search"],
            }[name],
        ):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get("/v1/toolsets")
                assert resp.status == 200
                data = await resp.json()
                assert data["object"] == "list"
                assert data["platform"] == "api_server"
                by_name = {ts["name"]: ts for ts in data["data"]}
                assert by_name["default"]["enabled"] is True
                assert by_name["default"]["tools"] == ["read_file", "terminal"]
                assert by_name["web"]["enabled"] is False
                assert by_name["web"]["tools"] == ["web_search"]
                assert by_name["default"]["configured"] is True

        resolve_features.assert_called_once()
        assert has_keys.call_count == len(fake_toolsets)
        assert all(
            call.kwargs["features"] is feature_snapshot
            for call in has_keys.call_args_list
        )


# ---------------------------------------------------------------------------
# /v1/chat/completions endpoint
# ---------------------------------------------------------------------------


class TestChatCompletionsEndpoint:
    @staticmethod
    def _canonical_action_body(*, metadata=None, content=None):
        action = {
            "version": 1,
            "type": "creation_recommendation_response",
            "proposal_id": "proposal-1",
            "action": "create",
            "creation_type": "agent",
            "title": "Advertising analyst",
            "dedup_key": "agent:advertising-analyst",
            "evidence_turn_ids": ["turn-1"],
        }
        wrapped = (
            "[creation_recommendation_response]\n"
            f"{json.dumps(action)}\n"
            "[/creation_recommendation_response]"
        )
        return {
            "model": "hermes-agent",
            "metadata": metadata
            if metadata is not None
            else {
                "creation_action_receipt_transport": "canonical_final_v1",
                "turn_id": "turn-action-1",
            },
            "messages": [{"role": "user", "content": content or wrapped}],
            "stream": False,
        }

    @pytest.mark.asyncio
    async def test_canonical_final_endpoint_admits_exact_action_before_agent_run(
        self, adapter
    ):
        app = _create_app(adapter)
        result = (
            {"final_response": "accepted", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", return_value=result) as run_agent:
                response = await cli.post(
                    "/v1/chat/completions/canonical-final-v1",
                    json=self._canonical_action_body(),
                )

        assert response.status == 200
        assert run_agent.call_count == 1
        assert (
            run_agent.call_args.kwargs["creation_action_receipt_transport"]
            == "canonical_final_v1"
        )

    @pytest.mark.asyncio
    async def test_canonical_final_endpoint_admits_web_action_instructions_before_protocol_block(
        self, adapter
    ):
        body = self._canonical_action_body()
        body["messages"][0]["content"] = (
            "The user selected Ignore on this recommendation card. "
            "Do not create anything.\n\n"
            + body["messages"][0]["content"]
        )
        app = _create_app(adapter)
        result = (
            {"final_response": "accepted", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", return_value=result) as run_agent:
                response = await cli.post(
                    "/v1/chat/completions/canonical-final-v1", json=body
                )

        assert response.status == 200
        run_agent.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metadata", "content"),
        [
            ({}, None),
            ({"creation_action_receipt_transport": "canonical_final_v2"}, None),
            ({"creation_action_receipt_transport": "canonical_final_v1"}, None),
            (
                {"creation_action_receipt_transport": "canonical_final_v1"},
                "ordinary chat",
            ),
            (
                {"creation_action_receipt_transport": "canonical_final_v1"},
                "[creation_recommendation_response]\n{}\n"
                "[/creation_recommendation_response]",
            ),
        ],
        ids=[
            "missing-capability",
            "unknown-capability",
            "missing-turn-id",
            "ordinary",
            "malformed",
        ],
    )
    async def test_canonical_final_endpoint_rejects_non_protocol_requests_before_agent(
        self, adapter, metadata, content
    ):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent") as run_agent:
                response = await cli.post(
                    "/v1/chat/completions/canonical-final-v1",
                    json=self._canonical_action_body(metadata=metadata, content=content),
                )

        assert response.status == 400
        run_agent.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_format",
        [{"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "x"}}],
        ids=["json_object", "json_schema"],
    )
    async def test_canonical_final_endpoint_rejects_structured_output(
        self, adapter, response_format
    ):
        """结构化输出与可信回执互斥，必须在进 Agent 前拒绝。

        governor 的 _on_pre_llm_call() 遇到 structured_output 会直接进入
        suppression：既不消费动作也不生成回执，而 HTTP 请求照常以普通模型结果
        收尾。放行这类请求等于让 Web 收到一个「没接管、也没法重试」的死状态。
        """
        body = self._canonical_action_body()
        body["response_format"] = response_format
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent") as run_agent:
                response = await cli.post(
                    "/v1/chat/completions/canonical-final-v1", json=body
                )

        assert response.status == 400
        run_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_canonical_final_endpoint_allows_plain_text_response_format(self, adapter):
        """对照：非结构化的 response_format 不该被这道门误伤。"""
        body = self._canonical_action_body()
        body["response_format"] = {"type": "text"}
        app = _create_app(adapter)
        result = (
            {"final_response": "accepted", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", return_value=result) as run_agent:
                response = await cli.post(
                    "/v1/chat/completions/canonical-final-v1", json=body
                )

        assert response.status == 200
        run_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_ordinary_chat_stays_on_legacy_endpoint(self, adapter):
        app = _create_app(adapter)
        result = (
            {"final_response": "ordinary", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", return_value=result) as run_agent:
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "ordinary chat"}],
                    },
                )

        assert response.status == 200
        assert run_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_legacy_endpoint_does_not_upgrade_structured_action(self, adapter):
        app = _create_app(adapter)
        result = (
            {"final_response": "legacy", "messages": [], "api_calls": 1},
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", return_value=result) as run_agent:
                response = await cli.post(
                    "/v1/chat/completions",
                    json=self._canonical_action_body(),
                )

        assert response.status == 200
        assert run_agent.call_count == 1
        assert run_agent.call_args.kwargs["creation_action_receipt_transport"] == ""

    @pytest.mark.asyncio
    async def test_idempotency_is_scoped_by_business_execution_token(
        self, adapter, monkeypatch, caplog
    ):
        cache = _IdempotencyCache()
        monkeypatch.setattr(api_server_module, "_idem_cache", cache)
        token_a = "a" * 64
        token_b = "b" * 64
        body = {
            "model": "hermes-agent",
            "messages": [{"role": "user", "content": "render"}],
            "stream": False,
        }
        calls = []

        async def run_agent(**kwargs):
            token = kwargs["business_execution_token"]
            calls.append(token)
            return (
                {
                    "final_response": f"run-{len(calls)}",
                    "messages": [],
                    "api_calls": 1,
                },
                {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=run_agent):
                first = await cli.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={
                        "Idempotency-Key": "same-key",
                        "X-Zettlab-Business-Execution-Token": token_a,
                    },
                )
                first_body = await first.json()
                same_scope = await cli.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={
                        "Idempotency-Key": "same-key",
                        "X-Zettlab-Business-Execution-Token": token_a,
                    },
                )
                same_scope_body = await same_scope.json()
                other_scope = await cli.post(
                    "/v1/chat/completions",
                    json=body,
                    headers={
                        "Idempotency-Key": "same-key",
                        "X-Zettlab-Business-Execution-Token": token_b,
                    },
                )
                other_scope_body = await other_scope.json()

        assert [first.status, same_scope.status, other_scope.status] == [
            200,
            200,
            200,
        ]
        assert first_body["choices"][0]["message"]["content"] == "run-1"
        assert same_scope_body["choices"][0]["message"]["content"] == "run-1"
        assert other_scope_body["choices"][0]["message"]["content"] == "run-2"
        assert calls == [token_a, token_b]
        cache_state = repr((cache._store, cache._inflight))
        response_state = repr((first_body, same_scope_body, other_scope_body))
        assert token_a not in cache_state
        assert token_b not in cache_state
        assert token_a not in response_state
        assert token_b not in response_state
        assert token_a not in caplog.text
        assert token_b not in caplog.text

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_missing_messages_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/chat/completions", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "messages" in data["error"]["message"]


    @pytest.mark.asyncio
    async def test_chat_completions_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning": {"enabled": False}, "reasoning_effort": "none", "fast": False}

        async def _mock_run_agent(**kwargs):
            cb = kwargs.get("stream_delta_callback")
            if cb:
                cb("ok")
            return (
                {"final_response": "ok", "messages": [], "api_calls": 1},
                {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            )

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "data: " in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options

    @pytest.mark.asyncio
    async def test_invalid_response_format_returns_400(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for response_format in (
                "json_object",
                {"type": "bogus"},
                {"type": "json_schema"},
                {"type": "json_schema", "json_schema": {"name": "x"}},
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "response_format": response_format,
                    },
                )
                assert resp.status == 400
                data = await resp.json()
                assert "response_format" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_response_format_passed_to_agent_request_overrides(self, adapter):
        mock_result = {
            "final_response": '{"ok":true}',
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Return JSON"}],
                        "stream": False,
                        "response_format": {"type": "json_object"},
                    },
                )

            assert resp.status == 200
            assert mock_run.await_args.kwargs["request_overrides"] == {
                "response_format": {"type": "json_object"},
            }

    @pytest.mark.asyncio
    async def test_tool_choice_none_skips_skill_invocation(self, adapter):
        # tool_choice=none is an API-level "no tools this turn" boundary;
        # skill expansion injects tool-driving instructions and is not
        # side-effect-free (skills.inline_shell=true executes SKILL.md
        # preprocessing at build time), so the hook must be bypassed entirely
        # even when metadata.skill_slug is present.
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch.object(adapter, "_expand_inbound_skill_invocation", new_callable=AsyncMock) as mock_expand:
                mock_run.return_value = (mock_result, usage)
                mock_expand.side_effect = lambda msg, slug, session_id=None, on_settled=None: (on_settled() if on_settled else None) or f"<<EXPANDED:{slug}:{msg}>>"
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "/deep-research 黄金"}],
                        "stream": False,
                        "tool_choice": "none",
                        "metadata": {"skill_slug": "deep-research"},
                    },
                )
                assert resp.status == 200
                mock_expand.assert_not_awaited()
                assert mock_run.await_args.kwargs["user_message"] == "/deep-research 黄金"
                assert (
                    mock_run.await_args.kwargs["trusted_user_message"]
                    == "/deep-research 黄金"
                )
                assert mock_run.await_args.kwargs["trusted_skill_slug"] == ""

                # Without the boundary the hook runs as usual.
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "/deep-research 黄金"}],
                        "stream": False,
                        "metadata": {"skill_slug": "deep-research"},
                    },
                )
                assert resp.status == 200
                mock_expand.assert_awaited_once()
                assert mock_run.await_args.kwargs["user_message"] == "<<EXPANDED:deep-research:/deep-research 黄金>>"
                assert (
                    mock_run.await_args.kwargs["trusted_user_message"]
                    == "黄金"
                )
                assert (
                    mock_run.await_args.kwargs["trusted_skill_slug"]
                    == "deep-research"
                )
                # The hook receives the resolved session so skill templates
                # can resolve ${HERMES_SESSION_ID} (builder task_id).
                assert (
                    mock_expand.await_args.kwargs["session_id"]
                    == mock_run.await_args.kwargs["session_id"]
                )

    @pytest.mark.asyncio
    async def test_video_edit_skill_selection_preserves_trusted_scope_signal(
        self,
        adapter,
    ):
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_run_agent",
                new_callable=AsyncMock,
            ) as mock_run, patch.object(
                adapter,
                "_expand_inbound_skill_invocation",
                new_callable=AsyncMock,
            ) as mock_expand:
                mock_run.return_value = (mock_result, usage)
                mock_expand.return_value = "<<EXPANDED>>"
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{
                            "role": "user",
                            "content": (
                                "/video-edit-workflow-mini "
                                "请总结 [file: /data/input.mp4]"
                            ),
                        }],
                        "stream": False,
                        "metadata": {
                            "skill_slug": "video-edit-workflow-mini",
                        },
                    },
                )

                assert resp.status == 200
                assert (
                    mock_run.await_args.kwargs["trusted_user_message"]
                    == "请总结 [file: /data/input.mp4]"
                )
                assert (
                    mock_run.await_args.kwargs["trusted_skill_slug"]
                    == "video-edit-workflow-mini"
                )

    @pytest.mark.asyncio
    async def test_slash_text_without_skill_slug_is_never_expanded(self, adapter):
        # The explicit metadata.skill_slug field is the ONLY trigger: message
        # text is never sniffed, so a literal "/<skill> ..." (e.g. the user
        # ASKING about the command, or an old App without the field) reaches
        # the agent verbatim and the hook is never consulted.
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch.object(adapter, "_expand_inbound_skill_invocation", new_callable=AsyncMock) as mock_expand:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "/deep-research 是什么？"}],
                        "stream": False,
                    },
                )
                assert resp.status == 200
                mock_expand.assert_not_awaited()
                assert mock_run.await_args.kwargs["user_message"] == "/deep-research 是什么？"

                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{
                            "role": "user",
                            "content": (
                                "/video-edit-workflow-mini "
                                "请总结 [file: /data/input.mp4]"
                            ),
                        }],
                        "stream": False,
                    },
                )
                assert resp.status == 200
                assert (
                    mock_run.await_args.kwargs["trusted_user_message"]
                    == (
                        "/video-edit-workflow-mini "
                        "请总结 [file: /data/input.mp4]"
                    )
                )
                assert mock_run.await_args.kwargs["trusted_skill_slug"] == ""

    @pytest.mark.asyncio
    async def test_idempotency_key_dedupes_skill_invocation(self, adapter):
        # Expansion lives INSIDE the idempotency-protected compute: a retried
        # key must reuse the cached agent result WITHOUT re-running expansion —
        # with skills.inline_shell=true the build step executes SKILL.md
        # preprocessing, so re-expansion means re-running local scripts.
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run, \
                 patch.object(adapter, "_expand_inbound_skill_invocation", new_callable=AsyncMock) as mock_expand:
                mock_run.return_value = (mock_result, usage)
                mock_expand.side_effect = lambda msg, slug, session_id=None, on_settled=None: (on_settled() if on_settled else None) or f"<<EXPANDED:{slug}:{msg}>>"
                payload = {
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "/deep-research 黄金"}],
                    "stream": False,
                    "metadata": {"skill_slug": "deep-research"},
                }
                for _ in range(2):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json=payload,
                        headers={"Idempotency-Key": "idem-slash-1"},
                    )
                    assert resp.status == 200
                mock_run.assert_awaited_once()
                mock_expand.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_meeting_summary_tool_choice_none_integration_flow_disables_agent_tools(self, adapter):
        mock_result = {
            "final_response": '{"title":"产品计划会","overall":"讨论发布计划"}',
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {
                                "role": "user",
                                "content": "发布计划先给大家看，确认后再执行。",
                            }
                        ],
                        "stream": False,
                        "tool_choice": "none",
                        "response_format": {"type": "json_object"},
                    },
                )

            assert resp.status == 200
            assert mock_run.await_args.kwargs["request_overrides"] == {
                "tool_choice": "none",
                "response_format": {"type": "json_object"},
            }

    @pytest.mark.asyncio
    async def test_text_response_format_is_accepted(self, adapter):
        mock_result = {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "Return text"}],
                        "stream": False,
                        "response_format": {"type": "text"},
                    },
                )

            assert resp.status == 200
            assert mock_run.await_args.kwargs["request_overrides"] is None

    def test_structured_response_format_rejects_unsupported_transports(self, adapter):
        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"api_mode": "anthropic_messages", "provider": "anthropic", "base_url": ""},
        ), patch.object(adapter, "_create_agent") as mock_create:
            assert "Anthropic" in adapter._response_format_transport_error(
                {"response_format": {"type": "json_object"}}
            )
            mock_create.assert_not_called()

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={
                "api_mode": "chat_completions",
                "provider": "google-gemini-cli",
                "base_url": "cloudcode-pa://google",
            },
        ), patch.object(adapter, "_create_agent") as mock_create:
            assert "Gemini" in adapter._response_format_transport_error(
                {"response_format": {"type": "json_object"}}
            )
            mock_create.assert_not_called()

        with patch(
            "gateway.run._resolve_runtime_agent_kwargs",
            return_value={"api_mode": "anthropic_messages", "provider": "anthropic", "base_url": ""},
        ), patch.object(adapter, "_create_agent") as mock_create:
            assert adapter._response_format_transport_error({"response_format": {"type": "text"}}) is None
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_true_returns_sse(self, adapter):
        """stream=true returns SSE format with the full response."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                # Simulate streaming: invoke stream_delta_callback with tokens
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("Hello!")
                    cb(None)  # End signal
                return (
                    {"final_response": "Hello!", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with (
                patch.object(adapter, "_response_format_transport_error", return_value=None),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent) as mock_run,
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "response_format": {"type": "json_object"},
                    },
                )
                assert resp.status == 200
                assert mock_run.await_args.kwargs["request_overrides"] == {
                    "response_format": {"type": "json_object"},
                }
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                assert resp.headers.get("X-Accel-Buffering") == "no"
                body = await resp.text()
                assert "data: " in body
                assert "[DONE]" in body
                assert "Hello!" in body


    @pytest.mark.asyncio
    async def test_session_chat_stream_passes_request_model_provider_options(self, adapter):
        app = _create_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    adapter,
                    "_get_existing_session_or_404",
                    return_value=({"id": "s1"}, None),
                ),
                patch.object(
                    adapter, "_conversation_history_for_session", return_value=[]
                ),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/api/sessions/s1/chat/stream",
                    json={
                        "message": "hi",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: run.completed" in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options

    @pytest.mark.asyncio
    async def test_stream_response_format_error_returns_400_before_sse(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(
                    adapter,
                    "_response_format_transport_error",
                    return_value="response_format is not supported by the Gemini transport.",
                ),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "response_format": {"type": "json_object"},
                    },
                )

            assert resp.status == 400
            assert "text/event-stream" not in resp.headers.get("Content-Type", "")
            data = await resp.json()
            assert data["error"]["param"] == "response_format"
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_string_false_returns_json_completion(self, adapter):
        """Quoted false must not route chat completions into SSE mode."""
        mock_result = {
            "final_response": "Hello! How can I help you today?",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_get_existing_session_or_404", return_value=({"id": "s1"}, None)),
                patch.object(adapter, "_conversation_history_for_session", return_value=[]),
                patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run,
            ):
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/api/sessions/s1/chat/stream",
                    json={
                        "message": "hi",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "event: run.completed" in body
        kwargs = mock_run.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_chat_completions(self, adapter):
        """Regression guard for #24451: completion callback must signal SSE EOS."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_chat_completion", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 2
            stream_q = mock_write_sse.call_args.args[4]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None
            fake_task.callbacks[1](fake_task)


    @pytest.mark.asyncio
    async def test_stream_agent_failure_emits_hermes_error(self, adapter):
        mock_result = {
            "final_response": "",
            "completed": False,
            "partial": False,
            "failed": True,
            "error": "provider auth failed",
            "messages": [],
            "api_calls": 1,
            "provider_error": {
                "code": "provider_billing",
                "reason": "billing",
                "provider": "openrouter",
                "model": "gpt-5",
                "status_code": 402,
                "provider_error_code": "insufficient_credits",
                "provider_message": "insufficient credits",
                "recoverable": False,
            },
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            assert resp.status == 200
            body = await resp.text()

        assert "event: hermes.error" in body
        assert '"message": "provider auth failed"' in body
        assert '"code": "provider_billing"' in body
        assert '"reason": "billing"' in body
        assert '"provider": "openrouter"' in body
        assert '"model": "gpt-5"' in body
        assert '"status_code": 402' in body
        assert '"provider_error_code": "insufficient_credits"' in body
        assert '"provider_message": "insufficient credits"' in body
        assert '"recoverable": false' in body
        assert '"finish_reason": "error"' in body
        assert "[DONE]" in body

    @pytest.mark.asyncio
    async def test_stream_agent_failure_hermes_error_is_redacted(self, adapter):
        raw_secret = "sk-stream-error-leak-1234567890"
        mock_result = {
            "final_response": "",
            "completed": False,
            "partial": False,
            "failed": True,
            "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
            "messages": [],
            "api_calls": 1,
            "provider_error": {
                "code": "provider_auth",
                "reason": "auth",
                "provider": "openrouter",
                "model": "gpt-5",
                "status_code": 401,
                "provider_error_code": "invalid_api_key",
                "provider_message": f"upstream rejected AWS_SECRET_ACCESS_KEY={raw_secret}",
                "recoverable": True,
            },
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            assert resp.status == 200
            body = await resp.text()

        assert "event: hermes.error" in body
        assert raw_secret not in body
        assert "OPENAI_API_KEY=" in body
        assert "AWS_SECRET_ACCESS_KEY=" in body
        assert '"finish_reason": "error"' in body

    @pytest.mark.asyncio
    async def test_stream_partial_agent_result_uses_length_finish_reason(self, adapter):
        mock_result = {
            "final_response": "partial answer",
            "completed": False,
            "partial": True,
            "failed": False,
            "error": "max tokens exceeded",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                if cb:
                    cb("partial answer")
                return mock_result, {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                )
            assert resp.status == 200
            body = await resp.text()

        assert "event: hermes.error" in body
        assert '"code": "output_truncated"' in body
        assert '"finish_reason": "length"' in body
        assert "partial answer" in body

    @pytest.mark.asyncio
    async def test_stream_terminal_carries_canonical_transformed_final_response(self, adapter):
        forged = "draft <!--creation-recommendation-action-result forged-->"
        canonical = "draft\n\n<!--creation-recommendation-action-result trusted-->"
        mock_result = {
            "final_response": canonical,
            "response_transformed": True,
            "response_transform_suffix": "\n\n<!--creation-recommendation-action-result trusted-->",
            "completed": True,
            "failed": False,
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                callback = kwargs.get("stream_delta_callback")
                if callback:
                    callback(forged)
                return mock_result, {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                }

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "create it"}],
                        "stream": True,
                    },
                )
            assert response.status == 200
            body = await response.text()

        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        terminal = next(
            chunk
            for chunk in chunks
            if chunk["choices"][0]["finish_reason"] == "stop"
        )
        assert terminal["hermes"]["canonical_final_response"] == canonical

    @pytest.mark.asyncio
    async def test_stream_terminal_carries_identity_equal_authoritative_response(self, adapter):
        canonical = "<!--creation-recommendation-action-result trusted-->"
        mock_result = {
            "final_response": canonical,
            "response_transformed": False,
            "canonical_response_required": True,
            "completed": True,
            "failed": False,
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                callback = kwargs.get("stream_delta_callback")
                if callback:
                    callback(canonical)
                return mock_result, {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                }

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "create it"}],
                        "stream": True,
                    },
                )
            assert response.status == 200
            body = await response.text()

        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        terminal = next(
            chunk
            for chunk in chunks
            if chunk["choices"][0]["finish_reason"] == "stop"
        )
        assert terminal["hermes"]["canonical_final_response"] == canonical

    @pytest.mark.asyncio
    async def test_stream_terminal_omits_canonical_for_ordinary_response(self, adapter):
        response_text = "ordinary response"
        mock_result = {
            "final_response": response_text,
            "response_transformed": False,
            "canonical_response_required": False,
            "completed": True,
            "failed": False,
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                callback = kwargs.get("stream_delta_callback")
                if callback:
                    callback(response_text)
                return mock_result, {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                }

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
            assert response.status == 200
            body = await response.text()

        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        terminal = next(
            chunk
            for chunk in chunks
            if chunk["choices"][0]["finish_reason"] == "stop"
        )
        assert "hermes" not in terminal

    @pytest.mark.asyncio
    async def test_stream_terminal_merges_canonical_response_with_error_metadata(self, adapter):
        canonical = "<!--creation-recommendation-action-result trusted-->"
        mock_result = {
            "final_response": canonical,
            "response_transformed": True,
            "completed": False,
            "failed": True,
            "error": "provider failed after action dispatch",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as run_agent:
                run_agent.return_value = (
                    mock_result,
                    {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                )
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "create it"}],
                        "stream": True,
                    },
                )
            assert response.status == 200
            body = await response.text()

        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in body.splitlines()
            if line.startswith("data: {")
        ]
        terminal = next(
            chunk
            for chunk in chunks
            if chunk.get("choices")
            and chunk["choices"][0]["finish_reason"] == "error"
        )
        assert terminal["hermes"]["canonical_final_response"] == canonical
        assert terminal["hermes"]["failed"] is True
        assert terminal["hermes"]["error_code"] == "agent_error"

    @pytest.mark.asyncio
    async def test_stream_includes_tool_progress(self, adapter):
        """tool_start_callback fires → progress appears as custom SSE event, not in delta.content."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                # Simulate the structured tool start the gateway now consumes.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Here are the files.")
                return (
                    {"final_response": "Here are the files.", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list files"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()
                assert "[DONE]" in body
                # Tool progress must appear as a custom SSE event, not in
                # delta.content — prevents model from learning to imitate
                # markers instead of calling tools (#6972).
                assert "event: hermes.tool.progress" in body
                assert '"tool": "terminal"' in body
                # ``label`` is now derived by ``build_tool_preview`` from the
                # tool args rather than passed by the caller, so we assert
                # only that *some* label exists rather than a literal value.
                assert '"label":' in body
                # The progress marker must NOT appear inside any
                # chat.completion.chunk delta.content field.
                import json as _json
                for line in body.splitlines():
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[len("data: "):])
                        except _json.JSONDecodeError:
                            continue
                        if chunk.get("object") == "chat.completion.chunk":
                            for choice in chunk.get("choices", []):
                                content = choice.get("delta", {}).get("content", "")
                                # Tool emoji markers must never leak into content
                                assert "ls -la" not in content or content == "Here are the files."
                # Final content must also be present
                assert "Here are the files." in body


    @pytest.mark.asyncio
    async def test_stream_emits_tool_lifecycle_with_call_id(self, adapter):
        """Regression for #16588.

        ``/v1/chat/completions`` streaming previously emitted only a
        ``tool.started``-style ``hermes.tool.progress`` event; clients
        rendering tool lifecycle UI had no way to mark a tool as finished
        because no matching ``status: completed`` event was emitted, and
        no ``toolCallId`` was carried for correlation.

        The fix adds ``tool_start_callback`` / ``tool_complete_callback``
        to the chat completions agent invocation and writes both halves
        of the lifecycle pair on the same ``event: hermes.tool.progress``
        SSE line, with stable ``toolCallId`` and ``status``.
        """
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # The structured callbacks own the chat-completions SSE
                # channel now; ``tool_progress_callback`` is intentionally
                # not wired so each tool start emits exactly one event.
                if ts_cb:
                    ts_cb("call_terminal_1", "terminal", {"command": "ls -la"})
                if tc_cb:
                    tc_cb("call_terminal_1", "terminal", {"command": "ls -la"}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("done.")
                return (
                    {"final_response": "done.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Walk the SSE body and collect *(status, toolCallId)* pairs
            # per event so the assertions verify per-event correlation —
            # an event missing ``toolCallId`` would not pass even if a
            # different event happens to carry the right id.
            pairs: list[tuple[str | None, str | None]] = []
            lines = body.splitlines()
            for i, line in enumerate(lines):
                if line.strip() != "event: hermes.tool.progress":
                    continue
                for follow in lines[i + 1: i + 4]:
                    if follow.startswith("data: "):
                        try:
                            payload = _json.loads(follow[len("data: "):])
                        except _json.JSONDecodeError:
                            break
                        pairs.append((payload.get("status"), payload.get("toolCallId")))
                        break

            # Each tool start must emit exactly one event (no duplicate
            # legacy + new emit), and each lifecycle pair must carry the
            # same toolCallId on every event — not just somewhere in the
            # aggregate.
            assert len(pairs) == 2, f"expected 2 events (running+completed), got {pairs}"
            assert pairs[0] == ("running", "call_terminal_1"), pairs
            assert pairs[1] == ("completed", "call_terminal_1"), pairs

    @pytest.mark.asyncio
    async def test_stream_tool_complete_includes_error_outcome(self, adapter):
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_linear_1", "linear_list_issues", {"first": 3})
                if tc_cb:
                    tc_cb(
                        "call_linear_1",
                        "linear_list_issues",
                        {"first": 3},
                        _json.dumps({
                            "error": "denied_by_agent_policy",
                            "errorCode": "denied_by_agent_policy",
                            "connector_error": {
                                "code": "denied_by_agent_policy",
                                "provider": "linear",
                                "status": 403,
                            },
                        }),
                    )
                if cb:
                    await asyncio.sleep(0.05)
                    cb("The connector call failed.")
                return (
                    {"final_response": "The connector call failed.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list linear issues"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = []
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if line.strip() != "event: hermes.tool.progress":
                continue
            for follow in lines[i + 1: i + 4]:
                if follow.startswith("data: "):
                    events.append(_json.loads(follow[len("data: "):]))
                    break
        completed = [event for event in events if event.get("status") == "completed"]
        assert len(completed) == 1
        assert completed[0]["outcome"] == "error"
        assert completed[0]["error"] == "denied_by_agent_policy"
        assert completed[0]["errorCode"] == "denied_by_agent_policy"
        assert completed[0]["connector_error"]["provider"] == "linear"
        assert completed[0]["connector_error"]["status"] == 403

    @pytest.mark.asyncio
    async def test_stream_tool_complete_preserves_connector_reauth_next_action(self, adapter):
        import asyncio
        import json as _json

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                if ts_cb:
                    ts_cb("call_weread_1", "weread_get_bookshelf", {})
                if tc_cb:
                    tc_cb(
                        "call_weread_1",
                        "weread_get_bookshelf",
                        {},
                        _json.dumps({
                            "error": "expired",
                            "errorCode": "expired",
                            "connector_error": {
                                "errorCode": "expired",
                                "provider": "weread",
                                "connectionId": "conn-weread",
                                "toolName": "weread.get_bookshelf",
                                "detail": "provider_token_rejected",
                                "nextAction": {
                                    "type": "reauth",
                                    "provider": "weread",
                                    "connectionId": "conn-weread",
                                    "toolName": "weread.get_bookshelf",
                                },
                            },
                        }),
                    )
                if cb:
                    await asyncio.sleep(0.05)
                    cb("Please reconnect WeRead.")
                return (
                    {"final_response": "Please reconnect WeRead.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "list weread bookshelf"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        events = []
        lines = body.splitlines()
        for i, line in enumerate(lines):
            if line.strip() != "event: hermes.tool.progress":
                continue
            for follow in lines[i + 1: i + 4]:
                if follow.startswith("data: "):
                    events.append(_json.loads(follow[len("data: "):]))
                    break
        completed = [event for event in events if event.get("status") == "completed"]
        assert len(completed) == 1
        assert completed[0]["outcome"] == "error"
        assert completed[0]["errorCode"] == "expired"
        connector_error = completed[0]["connector_error"]
        assert connector_error["errorCode"] == "expired"
        assert connector_error["detail"] == "provider_token_rejected"
        assert connector_error["nextAction"]["type"] == "reauth"
        assert connector_error["nextAction"]["connectionId"] == "conn-weread"

    @pytest.mark.asyncio
    async def test_stream_tool_complete_treats_empty_error_fields_as_success(self, adapter):
        import asyncio
        import json as _json

        empty_error_values = [
            None,
            "",
            "   ",
            {},
            [],
            {"message": ""},
            {"message": None, "details": []},
        ]

        for index, error_value in enumerate(empty_error_values):
            app = _create_app(adapter)
            async with TestClient(TestServer(app)) as cli:
                async def _mock_run_agent(**kwargs):
                    cb = kwargs.get("stream_delta_callback")
                    ts_cb = kwargs.get("tool_start_callback")
                    tc_cb = kwargs.get("tool_complete_callback")
                    call_id = f"call_terminal_{index}"
                    if ts_cb:
                        ts_cb(call_id, "terminal", {"command": "ls -la"})
                    if tc_cb:
                        tc_cb(
                            call_id,
                            "terminal",
                            {"command": "ls -la"},
                            _json.dumps({
                                "output": "total 136\n",
                                "exit_code": 0,
                                "error": error_value,
                                "errorCode": "",
                                "connector_error": {
                                    "code": "",
                                    "provider": "",
                                    "status": None,
                                },
                            }),
                        )
                    if cb:
                        await asyncio.sleep(0.05)
                        cb("done.")
                    return (
                        {"final_response": "done.", "messages": [], "api_calls": 1},
                        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    )

                with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test",
                            "messages": [{"role": "user", "content": "list"}],
                            "stream": True,
                        },
                    )
                    assert resp.status == 200
                    body = await resp.text()

            events = []
            lines = body.splitlines()
            for line_index, line in enumerate(lines):
                if line.strip() != "event: hermes.tool.progress":
                    continue
                for follow in lines[line_index + 1: line_index + 4]:
                    if follow.startswith("data: "):
                        events.append(_json.loads(follow[len("data: "):]))
                        break
            completed = [event for event in events if event.get("status") == "completed"]
            assert len(completed) == 1
            assert completed[0]["outcome"] == "success", error_value
            assert "error" not in completed[0]
            assert "errorCode" not in completed[0]
            assert "connector_error" not in completed[0]

    @pytest.mark.asyncio
    async def test_stream_tool_lifecycle_skips_internal_and_orphan_completes(self, adapter):
        """Internal tools (``_thinking``-style) and ``completed`` events
        without a prior matching ``running`` must produce no lifecycle
        events on the wire — otherwise clients would see orphaned
        ``status: completed`` updates they cannot correlate."""
        import asyncio

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                cb = kwargs.get("stream_delta_callback")
                ts_cb = kwargs.get("tool_start_callback")
                tc_cb = kwargs.get("tool_complete_callback")
                # Internal tool — must be filtered.
                if ts_cb:
                    ts_cb("call_internal_1", "_thinking", {})
                if tc_cb:
                    tc_cb("call_internal_1", "_thinking", {}, "")
                # Completion without start — orphan, must be dropped.
                if tc_cb:
                    tc_cb("call_orphan_1", "web_search", {}, "ok")
                if cb:
                    await asyncio.sleep(0.05)
                    cb("ok.")
                return (
                    {"final_response": "ok.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with patch.object(adapter, "_run_agent", side_effect=_mock_run_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "ok"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

            # Neither the internal call_id nor the orphan call_id should
            # surface as a lifecycle payload on the wire.
            assert "call_internal_1" not in body
            assert "call_orphan_1" not in body
            assert '"status": "running"' not in body
            assert '"status": "completed"' not in body


# ---------------------------------------------------------------------------
# _derive_chat_session_id unit tests
# ---------------------------------------------------------------------------


class TestDeriveChatSessionId:
    def test_deterministic(self):
        """Same inputs always produce the same session ID."""
        a = _derive_chat_session_id("sys", "hello")
        b = _derive_chat_session_id("sys", "hello")
        assert a == b


    def test_different_system_prompt(self):
        a = _derive_chat_session_id("You are a pirate.", "Hello")
        b = _derive_chat_session_id("You are a robot.", "Hello")
        assert a != b


# ---------------------------------------------------------------------------
# /v1/responses endpoint
# ---------------------------------------------------------------------------


class TestResponsesEndpoint:


    @pytest.mark.asyncio
    async def test_successful_response_with_string_input(self, adapter):
        """String input is wrapped in a user message."""
        mock_result = {
            "final_response": "Paris is the capital of France.",
            "messages": [],
            "api_calls": 1,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "What is the capital of France?",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "response"
            assert data["id"].startswith("resp_")
            assert data["status"] == "completed"
            assert len(data["output"]) == 1
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["type"] == "output_text"
            assert data["output"][0]["content"][0]["text"] == "Paris is the capital of France."


    @pytest.mark.asyncio
    async def test_previous_response_id_stores_compressed_transcript_directly(self, adapter):
        """After compression, stored history is the compressed transcript, not prior + compressed."""
        prior_history = [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ] * 10  # 20 messages — enough to simulate a long conversation
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )

        compressed_history = [
            # Compressed transcript starts with summary, NOT with prior[0]
            {"role": "user", "content": "[Compressed summary of earlier conversation]"},
            {"role": "user", "content": "Now add 1 more"},
            {"role": "assistant", "content": "3"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "3",
                        "messages": list(compressed_history),
                        "_compressed": True,
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Now add 1 more",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        stored = adapter._response_store.get(data["id"])
        stored_history = stored["conversation_history"]
        # Must NOT contain the original prior_history messages
        for msg in prior_history:
            assert msg not in stored_history, (
                f"Prior history message leaked into stored compressed transcript: {msg}"
            )
        # Must contain the compressed transcript
        assert stored_history == compressed_history


    @pytest.mark.asyncio
    async def test_previous_response_id_outputs_only_current_turn_items(self, adapter):
        """Response output must not replay previous tool artifacts."""
        prior_history = [
            {"role": "user", "content": "Read old file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"old.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "content": '{"content":"old"}',
            },
            {"role": "assistant", "content": "old"},
        ]
        adapter._response_store.put(
            "resp_prev",
            {
                "response": {"id": "resp_prev", "status": "completed"},
                "conversation_history": list(prior_history),
                "session_id": "api-test-session",
            },
        )
        full_agent_transcript = prior_history + [
            {"role": "user", "content": "Read new file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_new",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"new.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_new",
                "content": '{"content":"new"}',
            },
            {"role": "assistant", "content": "new"},
        ]

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "new",
                        "messages": list(full_agent_transcript),
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Read new file",
                        "previous_response_id": "resp_prev",
                    },
                )
                assert resp.status == 200
                data = await resp.json()

        output_json = json.dumps(data["output"])
        assert "call_new" in output_json
        assert "call_old" not in output_json
        assert "old.txt" not in output_json


    @pytest.mark.asyncio
    async def test_invalid_previous_response_id_returns_404(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "follow up",
                    "previous_response_id": "resp_nonexistent",
                },
            )
            assert resp.status == 404


    @pytest.mark.asyncio
    async def test_store_string_false_does_not_store(self, adapter):
        """Quoted false must preserve ephemeral store=false semantics."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    mock_result,
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "store": "false",
                    },
                )

            assert resp.status == 200
            data = await resp.json()
            assert adapter._response_store.get(data["id"]) is None

    @pytest.mark.asyncio
    async def test_instructions_inherited_from_previous(self, adapter):
        """If no instructions provided, carry forward from previous response."""
        mock_result = {"final_response": "Ahoy!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # First request with instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp1 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Hello",
                        "instructions": "Be a pirate",
                    },
                )

            data1 = await resp1.json()
            resp_id = data1["id"]

            # Second request without instructions
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp2 = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "Tell me more",
                        "previous_response_id": resp_id,
                    },
                )

            assert resp2.status == 200
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["ephemeral_system_prompt"] == "Be a pirate"


    @pytest.mark.asyncio
    async def test_result_error_fallback_is_redacted(self, adapter):
        raw_secret = "sk-responses-leak-1234567890"
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {
                        "final_response": "",
                        "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
                        "messages": [],
                        "api_calls": 1,
                    },
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                )
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hello"},
                )

            assert resp.status == 200
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert "OPENAI_API_KEY=" in body
            assert data["output"][0]["content"][0]["text"] != f"provider auth failed OPENAI_API_KEY={raw_secret}"


class TestResponsesStreaming:


    @pytest.mark.asyncio
    async def test_stream_task_done_callback_enqueues_eos_for_responses(self, adapter):
        """Regression guard for #24451 on /v1/responses streaming path."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            class _FakeTask:
                def __init__(self):
                    self.callbacks = []

                def add_done_callback(self, cb):
                    self.callbacks.append(cb)

            fake_task = _FakeTask()

            def _fake_ensure_future(coro):
                # We short-circuit task scheduling in this unit test.
                coro.close()
                return fake_task

            with (
                patch.object(
                    adapter,
                    "_run_agent",
                    new=AsyncMock(
                        return_value=(
                            {"final_response": "ok", "messages": [], "api_calls": 1},
                            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        )
                    ),
                ),
                patch("gateway.platforms.api_server.asyncio.ensure_future", side_effect=_fake_ensure_future),
                patch.object(adapter, "_write_sse_responses", new_callable=AsyncMock) as mock_write_sse,
            ):
                mock_write_sse.return_value = web.Response(status=200, text="ok")
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "hi", "stream": True},
                )
                assert resp.status == 200

            assert len(fake_task.callbacks) == 1
            stream_q = mock_write_sse.call_args.kwargs["stream_q"]
            assert stream_q.empty()
            fake_task.callbacks[0](fake_task)
            assert stream_q.get_nowait() is None


    @pytest.mark.asyncio
    async def test_stream_cancelled_persists_incomplete_snapshot(self, adapter):
        """Server-side asyncio.CancelledError (shutdown, request timeout) must
        still leave an ``incomplete`` snapshot in ResponseStore so
        GET /v1/responses/{id} and previous_response_id chaining keep
        working.  Regression for PR #15171 follow-up.

        Calls _write_sse_responses directly so the test can await the
        handler to completion (TestClient disconnection races the server
        handler, which makes end-to-end assertion on the final stored
        snapshot flaky).
        """
        # Build a minimal fake request + stream queue the writer understands.
        fake_request = MagicMock()
        fake_request.headers = {}

        written_payloads: list = []

        class _FakeStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                written_payloads.append(payload)

        # Patch web.StreamResponse for the duration of the writer call.
        import gateway.platforms.api_server as api_mod
        import queue as _q

        stream_q: _q.Queue = _q.Queue()

        async def _agent_coro():
            # Feed one partial delta into the stream queue...
            stream_q.put("partial output")
            # ...then give the drain loop a moment to pick it up before
            # raising CancelledError to simulate a server-side cancel.
            await asyncio.sleep(0.01)
            raise asyncio.CancelledError()

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_FakeStreamResponse()):
            with pytest.raises(asyncio.CancelledError):
                await adapter._write_sse_responses(
                    request=fake_request,
                    response_id=response_id,
                    model="hermes-agent",
                    created_at=int(time.time()),
                    stream_q=stream_q,
                    agent_task=agent_task,
                    agent_ref=[None],
                    conversation_history=[],
                    user_message="will be cancelled",
                    instructions=None,
                    conversation=None,
                    store=True,
                    session_id=None,
                )

        # The in_progress snapshot was persisted on response.created,
        # and the CancelledError handler must have updated it to
        # ``incomplete`` with the partial text it saw.
        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must be retrievable after cancellation"
        assert stored["response"]["status"] == "incomplete"
        # Partial text captured before cancel should be preserved.
        output_text = "".join(
            part.get("text", "")
            for item in stored["response"].get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
        )
        assert "partial output" in output_text

    @pytest.mark.asyncio
    async def test_stream_client_disconnect_persists_incomplete_snapshot(self, adapter):
        """Client disconnect (ConnectionResetError) during streaming must
        persist an ``incomplete`` snapshot in ResponseStore.  Regression
        for PR #15171."""
        fake_request = MagicMock()
        fake_request.headers = {}

        write_call_count = {"n": 0}

        class _DisconnectingStreamResponse:
            async def prepare(self, req):
                pass

            async def write(self, payload):
                # First two writes succeed (prepare + response.created).
                # On the third write (a text delta), the "client"
                # disconnects — simulate with ConnectionResetError.
                write_call_count["n"] += 1
                if write_call_count["n"] >= 3:
                    raise ConnectionResetError("simulated client disconnect")

        import gateway.platforms.api_server as api_mod
        import queue as _q

        stream_q: _q.Queue = _q.Queue()
        stream_q.put("some streamed text")
        stream_q.put(None)  # EOS sentinel

        async def _agent_coro():
            await asyncio.sleep(0.01)
            return ({"final_response": "", "messages": [], "api_calls": 0},
                    {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

        agent_task = asyncio.ensure_future(_agent_coro())
        response_id = f"resp_{uuid.uuid4().hex[:28]}"

        with patch.object(api_mod.web, "StreamResponse", return_value=_DisconnectingStreamResponse()):
            await adapter._write_sse_responses(
                request=fake_request,
                response_id=response_id,
                model="hermes-agent",
                created_at=int(time.time()),
                stream_q=stream_q,
                agent_task=agent_task,
                agent_ref=[None],
                conversation_history=[],
                user_message="will disconnect",
                instructions=None,
                conversation=None,
                store=True,
                session_id=None,
            )

        stored = adapter._response_store.get(response_id)
        assert stored is not None, "snapshot must survive client disconnect"
        assert stored["response"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Auth on endpoints
# ---------------------------------------------------------------------------


class TestEndpointAuth:
    @pytest.mark.asyncio
    async def test_chat_completions_requires_auth(self, auth_adapter):
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status == 401


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_platform_enum_has_api_server(self):
        assert Platform.API_SERVER.value == "api_server"


    def test_env_override_cors_origins(self, monkeypatch):
        monkeypatch.setenv("API_SERVER_ENABLED", "true")
        monkeypatch.setenv("API_SERVER_KEY", "opensslrandhex32strongkey")
        monkeypatch.setenv(
            "API_SERVER_CORS_ORIGINS",
            "http://localhost:3000, http://127.0.0.1:3000",
        )
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        assert config.platforms[Platform.API_SERVER].extra.get("cors_origins") == [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_api_server_in_connected_platforms(self):
        config = GatewayConfig()
        config.platforms[Platform.API_SERVER] = PlatformConfig(
            enabled=True, extra={"key": "opensslrandhex32strongkey"}
        )
        connected = config.get_connected_platforms()
        assert Platform.API_SERVER in connected


# ---------------------------------------------------------------------------
# Multiple system messages
# ---------------------------------------------------------------------------


class TestMultipleSystemMessages:
    @pytest.mark.asyncio
    async def test_multiple_system_messages_concatenated(self, adapter):
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hello"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            prompt = call_kwargs["ephemeral_system_prompt"]
            assert "You are helpful." in prompt
            assert "Be concise." in prompt


# ---------------------------------------------------------------------------
# send() method (not used but required by base)
# ---------------------------------------------------------------------------


class TestSendMethod:
    @pytest.mark.asyncio
    async def test_send_returns_not_supported(self):
        config = PlatformConfig(enabled=True)
        adapter = APIServerAdapter(config)
        result = await adapter.send("chat1", "hello")
        assert result.success is False
        assert "HTTP request/response" in result.error


class TestPlatformEventCallbackEndpoint:

    @pytest.mark.asyncio
    async def test_rejects_invalid_google_chat_auth(self, adapter):
        app = _create_app(adapter)
        app["platform_event_adapters"] = {
            "google_chat": _FakeGoogleChatAdapter(
                verify_ok=False,
                verify_code="invalid_google_bearer",
            )
        }

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/platforms/google_chat/events",
                headers={"Authorization": "Bearer bad"},
                json={"type": "MESSAGE"},
            )
            body = await resp.json()

        assert resp.status == 401
        assert body["error"]["code"] == "invalid_google_bearer"


# ---------------------------------------------------------------------------
# GET /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_stored_response(self, adapter):
        """GET returns a previously stored response."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            # Create a response first
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            response_id = data["id"]

            # Now GET it
            resp2 = await cli.get(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["status"] == "completed"


# ---------------------------------------------------------------------------
# DELETE /v1/responses/{response_id}
# ---------------------------------------------------------------------------


class TestDeleteResponse:
    @pytest.mark.asyncio
    async def test_delete_stored_response(self, adapter):
        """DELETE removes a stored response and returns confirmation."""
        mock_result = {"final_response": "Hello!", "messages": [], "api_calls": 1}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            data = await resp.json()
            response_id = data["id"]

            # Delete it
            resp2 = await cli.delete(f"/v1/responses/{response_id}")
            assert resp2.status == 200
            data2 = await resp2.json()
            assert data2["id"] == response_id
            assert data2["object"] == "response"
            assert data2["deleted"] is True

            # Verify it's gone
            resp3 = await cli.get(f"/v1/responses/{response_id}")
            assert resp3.status == 404


# ---------------------------------------------------------------------------
# Tool calls in output
# ---------------------------------------------------------------------------


class TestToolCallsInOutput:
    @pytest.mark.asyncio
    async def test_tool_calls_in_output(self, adapter):
        """When agent returns tool calls, they appear as function_call items."""
        mock_result = {
            "final_response": "The result is 42.",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "6*7"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "42",
                },
                {
                    "role": "assistant",
                    "content": "The result is 42.",
                },
            ],
            "api_calls": 2,
        }

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "What is 6*7?"},
                )

            assert resp.status == 200
            data = await resp.json()
            output = data["output"]

            # Should have: function_call, function_call_output, message
            assert len(output) == 3
            assert output[0]["type"] == "function_call"
            assert output[0]["name"] == "calculator"
            assert output[0]["arguments"] == '{"expression": "6*7"}'
            assert output[0]["call_id"] == "call_abc123"
            assert output[1]["type"] == "function_call_output"
            assert output[1]["call_id"] == "call_abc123"
            assert output[1]["output"] == "42"
            assert output[2]["type"] == "message"
            assert output[2]["content"][0]["text"] == "The result is 42."


# ---------------------------------------------------------------------------
# Usage / token counting
# ---------------------------------------------------------------------------


class TestUsageCounting:
    @pytest.mark.asyncio
    async def test_responses_usage(self, adapter):
        """Responses API returns real token counts."""
        mock_result = {"final_response": "Done", "messages": [], "api_calls": 1}
        usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, usage)
                resp = await cli.post(
                    "/v1/responses",
                    json={"model": "hermes-agent", "input": "Hi"},
                )

            assert resp.status == 200
            data = await resp.json()
            assert data["usage"]["input_tokens"] == 100
            assert data["usage"]["output_tokens"] == 50
            assert data["usage"]["total_tokens"] == 150


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:


    @pytest.mark.asyncio
    async def test_truncation_auto_preserves_non_leading_compaction_summary(self, adapter):
        """A summary sitting after a retained system head must survive too.

        The gateway /compress path can force a user-leading layout that
        leaves the compaction summary after a kept system message, so the
        preservation predicate must not assume the summary is at index 0.
        """
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}

        system_head = {"role": "system", "content": "You are a helpful agent."}
        summary = {
            "role": "user",
            "content": "[CONTEXT COMPACTION — REFERENCE ONLY]\nEarlier work.",
            "_compressed_summary": True,
        }
        long_history = [system_head, summary] + [
            {"role": "user", "content": f"msg {i}"}
            for i in range(148)
        ]
        adapter._response_store.put("resp_summary_mid", {
            "response": {"id": "resp_summary_mid", "object": "response"},
            "conversation_history": long_history,
            "instructions": None,
        })

        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    json={
                        "model": "hermes-agent",
                        "input": "follow up",
                        "previous_response_id": "resp_summary_mid",
                        "truncation": "auto",
                    },
                )

        assert resp.status == 200
        history = mock_run.call_args.kwargs["conversation_history"]
        assert len(history) == 100
        assert history[0] == summary
        assert history[1]["content"] == "msg 49"
        assert history[-1]["content"] == "msg 147"


# ---------------------------------------------------------------------------
# Response-side truncation / failure handling (issue #22496)
# ---------------------------------------------------------------------------


class TestChatCompletionsAgentIncomplete:
    """When the agent run yields a partial / failed result, the API server
    must NOT pretend it succeeded. Either signal truncation via
    finish_reason='length' (with the partial text), or 502 with an OpenAI
    error envelope (no usable text). Issue #22496."""

    @pytest.mark.asyncio
    async def test_truncation_with_partial_text_uses_length_finish_reason(self, adapter):
        """Partial text + structured partial state → finish_reason='length', 200 OK,
        plus hermes extras + headers."""
        mock_result = {
            "final_response": "Here is part one of the answer",
            "completed": False,
            "partial": True,
            "error": "max tokens exceeded",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "tell me everything"}]},
                )
            assert resp.status == 200
            data = await resp.json()
            assert data["choices"][0]["finish_reason"] == "length"
            assert data["choices"][0]["message"]["content"] == "Here is part one of the answer"
            assert data["hermes"]["partial"] is True
            assert data["hermes"]["completed"] is False
            assert data["hermes"]["error_code"] == "output_truncated"
            assert resp.headers.get("X-Hermes-Completed") == "false"
            assert resp.headers.get("X-Hermes-Partial") == "true"

    @pytest.mark.asyncio
    async def test_hard_failure_redacts_secret_like_error_text(self, adapter):
        raw_secret = "sk-api-server-leak-1234567890"
        mock_result = {
            "final_response": "",
            "completed": False,
            "partial": False,
            "failed": True,
            "error": f"provider auth failed OPENAI_API_KEY={raw_secret}",
            "messages": [],
            "api_calls": 1,
        }
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hello"}]},
                )

            assert resp.status == 502
            data = await resp.json()
            body = json.dumps(data)
            assert raw_secret not in body
            assert raw_secret not in resp.headers.get("X-Hermes-Error", "")
            assert "OPENAI_API_KEY=" in body
            assert data["error"]["hermes"]["failed"] is True


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORS:
    def test_origin_allowed_for_non_browser_client(self, adapter):
        assert adapter._origin_allowed("") is True


    def test_origin_allowed_for_allowlist_match(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._origin_allowed("http://localhost:3000") is True

    def test_cors_headers_for_origin_disabled_by_default(self, adapter):
        assert adapter._cors_headers_for_origin("http://localhost:3000") is None

    def test_cors_headers_for_origin_matches_allowlist(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        headers = adapter._cors_headers_for_origin("http://localhost:3000")
        assert headers is not None
        assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        assert "POST" in headers["Access-Control-Allow-Methods"]

    def test_cors_headers_allow_business_execution_token_unit(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        headers = adapter._cors_headers_for_origin("http://localhost:3000")
        assert headers is not None
        allowed = {
            value.strip().lower()
            for value in headers["Access-Control-Allow-Headers"].split(",")
        }
        assert "x-zettlab-business-execution-token" in allowed

    def test_cors_headers_for_origin_rejects_unknown_origin(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        assert adapter._cors_headers_for_origin("http://evil.example") is None

    @pytest.mark.asyncio
    async def test_cors_headers_not_present_by_default(self, adapter):
        """CORS is disabled unless explicitly configured."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") is None

    @pytest.mark.asyncio
    async def test_browser_origin_rejected_by_default(self, adapter):
        """Browser-originated requests are rejected unless explicitly allowed."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://evil.example"})
            assert resp.status == 403
            assert resp.headers.get("Access-Control-Allow-Origin") is None


    @pytest.mark.asyncio
    async def test_cors_allows_idempotency_key_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Idempotency-Key",
                },
            )
            assert resp.status == 200
            assert "Idempotency-Key" in resp.headers.get("Access-Control-Allow-Headers", "")

    @pytest.mark.asyncio
    async def test_cors_business_execution_token_preflight_flow(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": (
                        "Content-Type, X-Zettlab-Business-Execution-Token"
                    ),
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == (
                "http://localhost:3000"
            )
            allowed = {
                value.strip().lower()
                for value in resp.headers.get(
                    "Access-Control-Allow-Headers",
                    "",
                ).split(",")
            }
            assert "x-zettlab-business-execution-token" in allowed

    @pytest.mark.asyncio
    async def test_cors_sets_vary_origin_header(self):
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health", headers={"Origin": "http://localhost:3000"})
            assert resp.status == 200
            assert resp.headers.get("Vary") == "Origin"

    @pytest.mark.asyncio
    async def test_cors_options_preflight_allowed_for_configured_origin(self):
        """Configured origins can complete browser preflight."""
        adapter = _make_adapter(cors_origins=["http://localhost:3000"])
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            assert "Authorization" in resp.headers.get("Access-Control-Allow-Headers", "")


# ---------------------------------------------------------------------------
# Conversation parameter
# ---------------------------------------------------------------------------


class TestConversationParameter:


    @pytest.mark.asyncio
    async def test_separate_conversations_are_isolated(self, adapter):
        """Different conversation names have independent histories."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "Response A", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Conversation A
                await cli.post("/v1/responses", json={"input": "conv-a msg", "conversation": "conv-a"})
                # Conversation B
                mock_run.return_value = (
                    {"final_response": "Response B", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "conv-b msg", "conversation": "conv-b"})

                # They should have different response IDs in the mapping
                assert adapter._response_store.get_conversation("conv-a") != adapter._response_store.get_conversation("conv-b")


    @pytest.mark.asyncio
    async def test_conversation_reuse_after_eviction_no_404(self, adapter):
        """After eviction clears a conversation mapping, reusing that name starts fresh (no 404)."""
        adapter._response_store = ResponseStore(max_size=1)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "First", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                # Create conversation -> resp stored
                resp1 = await cli.post("/v1/responses", json={
                    "input": "hello",
                    "conversation": "my-chat",
                })
                assert resp1.status == 200

                # Evict by adding another response
                mock_run.return_value = (
                    {"final_response": "Other", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                await cli.post("/v1/responses", json={"input": "other"})

                # Conversation mapping should have been cleaned by eviction
                assert adapter._response_store.get_conversation("my-chat") is None

                # Reuse conversation name — should start fresh, not 404
                mock_run.return_value = (
                    {"final_response": "Restarted", "messages": [], "api_calls": 1},
                    {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
                resp3 = await cli.post("/v1/responses", json={
                    "input": "hello again",
                    "conversation": "my-chat",
                })
                assert resp3.status == 200


# ---------------------------------------------------------------------------
# X-Hermes-Session-Id header (session continuity)
# ---------------------------------------------------------------------------


class TestSessionIdHeader:


    @pytest.mark.asyncio
    async def test_traversal_session_id_header_rejected(self, auth_adapter):
        """Security (#5958): a path-traversal X-Hermes-Session-Id must be
        rejected with 400 so it can't reach the filesystem artifact paths
        (session snapshot / request dump) and escape the sessions dir."""
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                for bad in ("../../../../etc/pwned", "/abs/path", "..\\win"):
                    resp = await cli.post(
                        "/v1/chat/completions",
                        headers={"X-Hermes-Session-Id": bad, "Authorization": "Bearer sk-secret"},
                        json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                    )
                    assert resp.status == 400, f"{bad!r} should be rejected"
                # The agent is never invoked for a rejected ID.
                assert mock_run.call_count == 0

    @pytest.mark.asyncio
    async def test_provided_session_id_loads_history_from_db(self, auth_adapter):
        """When X-Hermes-Session-Id is provided, history comes from SessionDB not request body."""
        mock_result = {"final_response": "OK", "messages": [], "api_calls": 1}
        db_history = [
            {"role": "user", "content": "stored message 1"},
            {"role": "assistant", "content": "stored reply 1"},
        ]
        mock_db = MagicMock()
        mock_db.get_messages_as_conversation.return_value = db_history
        # The cache gate (_profile_session_db_is_current) evicts any cached DB
        # whose recorded home/state.db file identity no longer matches disk —
        # give the mock a REAL identity from the isolated HERMES_HOME so it
        # passes the gate the way a genuinely opened DB would.
        home = Path(auth_adapter._profile_home_key())
        home.mkdir(parents=True, exist_ok=True)
        state_path = home / "state.db"
        state_path.touch()
        home_stat = os.stat(home)
        state_stat = os.stat(state_path)
        mock_db._profile_home_identity = (home_stat.st_dev, home_stat.st_ino)
        mock_db._profile_state_identity = (state_stat.st_dev, state_stat.st_ino)
        auth_adapter._session_db = mock_db
        auth_adapter._session_dbs[auth_adapter._profile_home_key()] = mock_db
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={"X-Hermes-Session-Id": "existing-session", "Authorization": "Bearer sk-secret"},
                    # Request body has different history — should be ignored
                    json={
                        "model": "hermes-agent",
                        "messages": [
                            {"role": "user", "content": "old msg from client"},
                            {"role": "assistant", "content": "old reply from client"},
                            {"role": "user", "content": "new question"},
                        ],
                    },
                )

            assert resp.status == 200
            call_kwargs = mock_run.call_args.kwargs
            # History must come from DB, not from the request body
            assert call_kwargs["conversation_history"] == db_history
            assert call_kwargs["user_message"] == "new question"


# ---------------------------------------------------------------------------
# X-Hermes-Session-Key header (long-term memory scoping)
# ---------------------------------------------------------------------------


class TestSessionKeyHeader:
    """The session key is a stable per-channel identifier that scopes
    long-term memory (e.g. Honcho) independently of the transcript-scoped
    session_id.  A third-party Web UI passes one stable key per assistant
    channel and rotates session_id on /new, matching the native
    gateway's session_key / session_id split.
    """


    @pytest.mark.asyncio
    async def test_session_key_threads_into_create_agent(self, auth_adapter):
        """End-to-end: verify AIAgent(gateway_session_key=...) receives the key via _create_agent."""
        captured_kwargs = {}

        def _fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            mock_agent = MagicMock()
            mock_agent.run_conversation.return_value = {"final_response": "ok", "messages": []}
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent", side_effect=_fake_create_agent):
                resp = await cli.post(
                    "/v1/chat/completions",
                    headers={
                        "X-Hermes-Session-Key": "agent:main:webui:dm:user-7",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                )
            assert resp.status == 200
            # _create_agent must be called with gateway_session_key threaded through
            assert captured_kwargs.get("gateway_session_key") == "agent:main:webui:dm:user-7"

    @pytest.mark.asyncio
    async def test_responses_endpoint_accepts_session_key(self, auth_adapter):
        """Responses API honors the same X-Hermes-Session-Key contract."""
        mock_result = {"final_response": "ok", "messages": [], "api_calls": 1}
        app = _create_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (mock_result, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
                resp = await cli.post(
                    "/v1/responses",
                    headers={
                        "X-Hermes-Session-Key": "webui:chan-1",
                        "Authorization": "Bearer sk-secret",
                    },
                    json={"model": "hermes-agent", "input": "hello", "store": False},
                )
            assert resp.status == 200
            assert resp.headers.get("X-Hermes-Session-Key") == "webui:chan-1"
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["gateway_session_key"] == "webui:chan-1"

    @pytest.mark.asyncio
    async def test_capabilities_advertises_session_key_header(self, adapter):
        """GET /v1/capabilities should advertise the new header so clients can feature-detect."""
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            assert resp.status == 200
            data = await resp.json()
            assert data["features"]["session_key_header"] == "X-Hermes-Session-Key"


# ---------------------------------------------------------------------------
# Per-client model routing (model_routes)
# ---------------------------------------------------------------------------


def _make_routing_adapter(routes) -> APIServerAdapter:
    """Create an adapter with model_routes configured."""
    config = PlatformConfig(enabled=True, extra={"model_routes": routes})
    return APIServerAdapter(config)


def _patch_create_agent_runtime(monkeypatch, captured: dict, fake_agent_cls):
    """Stub out every external dependency of _create_agent."""
    monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": {})
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None)
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())


class TestModelRoutesParsing:
    def test_valid_routes_are_parsed(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        assert adapter._model_routes == routes


    def test_route_without_model_is_dropped(self):
        adapter = _make_routing_adapter({"bad": {"provider": "openrouter"}})
        assert adapter._model_routes == {}


class TestModelRoutesModelsEndpoint:

    @pytest.mark.asyncio
    async def test_models_endpoint_route_alias_fields_and_no_secrets(self):
        routes = {"my-alias": {"model": "openai/gpt-5", "api_key": "sk-route-secret"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/models")
            data = await resp.json()
            alias_entry = next(m for m in data["data"] if m["id"] == "my-alias")
            assert alias_entry["root"] == "openai/gpt-5"
            assert alias_entry["parent"] == adapter._model_name
            # per-route api_key must never leak through the discovery endpoint
            assert "sk-route-secret" not in json.dumps(data)


class TestModelRoutesHandlers:
    @pytest.mark.asyncio
    async def test_chat_completions_passes_route_to_run_agent(self):
        routes = {"minimax-m2": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "minimax-m2",
                    "messages": [{"role": "user", "content": "hello"}],
                })
                assert resp.status == 200
                kwargs = mock_run.call_args.kwargs
                assert kwargs.get("route") == {
                    "model": "minimax/minimax-m1", "provider": "openrouter",
                }

    @pytest.mark.asyncio
    async def test_chat_completions_passes_connector_route_capability_to_run_agent(self):
        adapter = _make_routing_adapter({})
        app = _create_app(adapter)
        capability = "c" * 43
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hello"}],
                    "metadata": {"connector_route_capability": capability},
                })
                assert resp.status == 200
                assert (
                    mock_run.call_args.kwargs.get("connector_route_capability")
                    == capability
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("metadata", "want"),
        [
            ({"creation_action_receipt_transport": "canonical_final_v1"}, "canonical_final_v1"),
            ({"creation_action_receipt_transport": "unknown"}, ""),
            ({}, ""),
        ],
    )
    async def test_chat_completions_scopes_receipt_transport_to_each_request(
        self, metadata, want
    ):
        adapter = _make_routing_adapter({})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                response = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "hello"}],
                        "metadata": metadata,
                    },
                )

        assert response.status == 200
        assert (
            mock_run.call_args.kwargs.get("creation_action_receipt_transport")
            == want
        )

    @pytest.mark.asyncio
    async def test_chat_completions_no_route_for_unknown_model(self):
        adapter = _make_routing_adapter({"minimax-m2": {"model": "minimax/minimax-m1"}})
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/chat/completions", json={
                    "model": "unknown-model",
                    "messages": [{"role": "user", "content": "hello"}],
                })
                assert resp.status == 200
                assert mock_run.call_args.kwargs.get("route") is None

    @pytest.mark.asyncio
    async def test_responses_api_passes_route_to_run_agent(self):
        routes = {"xiaozhi": {"model": "minimax/minimax-m1", "provider": "openrouter"}}
        adapter = _make_routing_adapter(routes)
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "hi", "messages": [], "api_calls": 1},
                    {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
                )
                resp = await cli.post("/v1/responses", json={
                    "model": "xiaozhi",
                    "input": "hello",
                })
                assert resp.status == 200
                assert mock_run.call_args.kwargs.get("route") == {
                    "model": "minimax/minimax-m1", "provider": "openrouter",
                }


class TestModelRoutesAgentCreation:

    def test_route_provider_resolves_provider_credentials(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            lambda provider: {
                "provider": provider,
                "api_key": f"sk-{provider}",
                "base_url": f"https://{provider}.example/v1",
                "api_mode": "chat_completions",
            },
        )
        adapter = _make_routing_adapter(
            {"alias": {"model": "other/model", "provider": "otherprov"}}
        )
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "other/model"
        assert captured["provider"] == "otherprov"
        assert captured["api_key"] == "sk-otherprov"

    def test_route_provider_resolution_failure_drops_global_credentials(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_key": "sk-global",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
                "command": "global-provider-cli",
                "args": ["--global-provider"],
                "credential_pool": object(),
            },
        )

        def fail_provider_resolution(provider):
            raise RuntimeError(f"unknown provider: {provider}")

        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            fail_provider_resolution,
        )
        adapter = _make_routing_adapter({
            "alias": {
                "model": "other/model",
                "provider": "otherprov",
                "base_url": "https://route.example/v1",
            }
        })
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["provider"] == "otherprov"
        assert captured["base_url"] == "https://route.example/v1"
        assert "api_key" not in captured
        assert "api_mode" not in captured
        assert "command" not in captured
        assert "args" not in captured
        assert "credential_pool" not in captured

    def test_zet_route_resolution_failure_also_drops_global_credentials(self, monkeypatch):
        from gateway.platforms.zet_agent import ZetAgentAdapter

        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_key": "sk-global",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
                "command": "global-provider-cli",
                "args": ["--global-provider"],
                "credential_pool": object(),
            },
        )

        def fail_provider_resolution(provider):
            raise RuntimeError(f"unknown provider: {provider}")

        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs_for_provider",
            fail_provider_resolution,
        )
        adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={
            "model_routes": {
                "alias": {
                    "model": "other/model",
                    "provider": "otherprov",
                    "base_url": "https://route.example/v1",
                }
            }
        }))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(route=adapter._resolve_route("alias"))

        assert captured["provider"] == "otherprov"
        assert captured["base_url"] == "https://route.example/v1"
        assert "api_key" not in captured
        assert "api_mode" not in captured
        assert "command" not in captured
        assert "args" not in captured
        assert "credential_pool" not in captured

    def test_no_route_keeps_global_model(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter({"alias": {"model": "other/model"}})
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(session_id="s1", route=None)

        assert captured["model"] == "global/model"
        assert captured["api_key"] == "sk-global"

    def test_session_model_override_beats_route(self, monkeypatch):
        """A user-issued /model on the session must win over static route config."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter({"alias": {"model": "route/model", "api_key": "sk-route"}})
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(
            adapter,
            "_session_model_override_for",
            lambda key: {
                "model": "session/override-model",
                "provider": "sessionprov",
                "api_key": "sk-session",
                "base_url": "https://session.example/v1",
                "api_mode": "responses",
                "credential_pool": "pool-session",
            },
        )

        adapter._create_agent(session_id="s1", route=adapter._resolve_route("alias"))

        assert captured["model"] == "session/override-model"
        assert captured["provider"] == "sessionprov"
        assert captured["api_key"] == "sk-session"


# ---------------------------------------------------------------------------
# Event-loop offloading for synchronous SessionDB calls (P1)
# ---------------------------------------------------------------------------


class TestSessionDbOffEventLoop:
    """Regression: synchronous SessionDB calls in the OpenAI-compatible API
    server must run OFF the aiohttp event loop. A blocking SQLite read/write on
    the loop freezes every in-flight request under load (same class of bug as
    gateway build_channel_directory, #60794 / #60810), so each call is wrapped
    in asyncio.to_thread.
    """

    @pytest.mark.asyncio
    async def test_get_existing_session_or_404_offloads(self, auth_adapter):
        import threading

        captured = {}

        class FakeDB:
            def get_session(self, session_id):
                captured["thread"] = threading.current_thread()
                return {"id": session_id, "source": "api_server"}

        auth_adapter._session_db = FakeDB()
        session, err = await auth_adapter._get_existing_session_or_404("sess-x")
        assert err is None
        assert session["id"] == "sess-x"
        # The blocking DB call must NOT execute on the event-loop thread.
        assert captured["thread"] is not None
        assert captured["thread"] != threading.current_thread()


# ---------------------------------------------------------------------------
# _api_key_passes_startup_guard — fail-closed on an unverifiable key
# ---------------------------------------------------------------------------

class TestApiKeyStartupGuardFailsClosed:
    """The guard is the only thing between a guessable key and an endpoint the
    code itself describes as ``terminal-capable agent work`` where "a guessable
    key is remote code execution".

    So "the strength check could not be run" must never resolve to "start
    anyway" — the same posture ``tools/credential_files.py`` takes when its
    deny-list cannot be consulted.
    """

    class _Stub:
        name = "api_server"
        _host = "0.0.0.0"

        def __init__(self, key):
            self._api_key = key

    @staticmethod
    def _guard(key):
        return APIServerAdapter._api_key_passes_startup_guard(
            TestApiKeyStartupGuardFailsClosed._Stub(key)
        )

    @staticmethod
    def _blocking_auth_import():
        real_import = __import__

        def _blocked(name, *args, **kwargs):
            if name == "hermes_cli.auth":
                raise ImportError("simulated: hermes_cli.auth unavailable")
            return real_import(name, *args, **kwargs)

        return patch("builtins.__import__", _blocked)

    def test_weak_key_refused_when_check_is_unavailable(self):
        """The bug: an unimportable auth module silently dropped the check and
        the server started on a 4-character key."""
        with self._blocking_auth_import():
            assert self._guard("test") is False

    def test_strong_key_also_refused_when_check_is_unavailable(self):
        """Fail-closed: we cannot verify the key, so we do not expose the
        endpoint — the log tells the operator to repair the install."""
        with self._blocking_auth_import():
            assert self._guard("a" * 40) is False


class TestKeyRejectionSetsNonRetryableFatalError:
    """Each startup-guard rejection must set a non-retryable fatal error so
    the reconnect watcher drops the platform from the retry queue instead of
    looping indefinitely.

    Previously connect() returned bare ``False``, which gateway.run treated
    as retryable — re-queueing every backoff interval forever and
    re-instantiating the adapter (with its ResponseStore sqlite connection)
    each retry (#38803: ~501 leaked connections / 1002 fds over 2.5 days,
    ending in EMFILE for the whole gateway). Mirrors the port-conflict
    precedent (test_port_conflict_sets_non_retryable_fatal_error, #65665).
    """

    @staticmethod
    def _make_adapter(key, monkeypatch):
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
        return APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={"host": "127.0.0.1", "port": 0, "key": key},
            )
        )

    @staticmethod
    async def _assert_key_rejection_is_fatal(adapter):
        try:
            assert await adapter.connect() is False
            assert adapter.has_fatal_error is True
            assert adapter.fatal_error_retryable is False
            assert adapter.fatal_error_code == "api_server_key_invalid"
            assert "API_SERVER_KEY" in (adapter.fatal_error_message or "")
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_missing_key_sets_non_retryable_fatal_error(self, monkeypatch):
        adapter = self._make_adapter("", monkeypatch)
        await self._assert_key_rejection_is_fatal(adapter)


# ---------------------------------------------------------------------------
# Bare-model opt-in gate (direct_model_requests) for _request_agent_overrides
# ---------------------------------------------------------------------------


class TestDirectModelRequestsGate:
    """Bare ``model`` (no ``provider``) is opt-in on OpenAI-compatible
    endpoints so generic clients hardcoding "gpt-4o" keep falling back to
    the gateway default (idea credit: PR #22825 by @mssteuer)."""

    def test_bare_model_dropped_when_disallowed(self):
        overrides = _request_agent_overrides(
            {"model": "openai/gpt-5"}, allow_bare_model=False
        )
        assert "requested_model" not in overrides


    def test_adapter_flag_opt_in(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        assert adapter._direct_model_requests is True


    @pytest.mark.asyncio
    async def test_chat_completions_bare_model_honored_when_enabled(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"direct_model_requests": True})
        )
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                mock_run.return_value = (
                    {"final_response": "ok", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "openai/gpt-5",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
        assert resp.status == 200
        assert mock_run.call_args.kwargs.get("requested_model") == "openai/gpt-5"


class TestRouteWithoutModelKeepsDefault:
    """A model_routes alias whose route has no ``model`` key must keep the
    global default model — the alias string itself is never a model name."""

    def test_alias_never_leaks_as_model(self, monkeypatch):
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        adapter = _make_routing_adapter(
            {"alias": {"model": "", "api_key": "sk-route"}}
        )
        # _parse_model_routes drops routes without model; simulate a
        # credentials-only route surviving via direct dict (defensive path).
        adapter._model_routes = {"alias": {"api_key": "sk-route"}}
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
        monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

        adapter._create_agent(
            session_id="s1",
            route=adapter._resolve_route("alias"),
            requested_model="alias",
        )

        assert captured["model"] == "global/model"
        assert captured["api_key"] == "sk-route"


# ---------------------------------------------------------------------------
# Empty-model recovery + provider-auth error typing in _create_agent
# (salvaged from PR #57947 by @FvanW)
# ---------------------------------------------------------------------------


class TestCreateAgentModelRecovery:
    def test_create_agent_defaults_to_provider_catalog_model_when_empty(self, monkeypatch):
        """api_server.py had no equivalent of run.py's provider-catalog
        default when model resolves empty but a provider did resolve (e.g.
        `hermes auth add openai-codex` without `hermes model`) —
        AIAgent(model="") 400s every call."""
        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        _patch_create_agent_runtime(monkeypatch, captured, FakeAgent)
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": "openai-codex", "base_url": "https://example.test/v1",
                     "api_mode": "codex_responses"},
        )
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "hermes_cli.models.get_default_model_for_provider",
            lambda provider: "gpt-5.5-codex" if provider == "openai-codex" else None,
        )

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        agent = adapter._create_agent(session_id="api-session")

        assert isinstance(agent, FakeAgent)
        assert captured["model"] == "gpt-5.5-codex"

    def test_create_agent_recovers_last_known_good_model_when_empty(self, monkeypatch):
        """Last-known-good recovery (#35314): a transient config-cache miss
        producing an empty model would build AIAgent(model="") and fail every
        call until manual retry, instead of reusing the model that just
        worked."""
        captured = []

        class FakeAgent:
            def __init__(self, **kwargs):
                captured.append(dict(kwargs))

        _patch_create_agent_runtime(monkeypatch, {}, FakeAgent)
        monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

        # Turn 1: model resolves fine — populates the last-known-good cache
        # (keyed on gateway_session_key).
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "minimax/minimax-m3")
        adapter._create_agent(session_id="api-session", gateway_session_key="stable-chan-1")
        assert captured[0]["model"] == "minimax/minimax-m3"
        assert adapter._last_resolved_model["stable-chan-1"] == "minimax/minimax-m3"

        # Turn 2: transient empty resolution, no provider catalog default —
        # must recover the model from turn 1, not build model="".
        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "")
        monkeypatch.setattr(
            "gateway.run._resolve_runtime_agent_kwargs",
            lambda: {"provider": None, "base_url": None, "api_mode": None},
        )
        adapter._create_agent(session_id="another-session", gateway_session_key="stable-chan-1")
        assert captured[1]["model"] == "minimax/minimax-m3"

    @pytest.mark.asyncio
    async def test_concurrent_same_id_create_one_201_one_409(self, auth_adapter):
        """Two concurrent creates for the same ID must yield one 201 and one 409.

        The create sequence (existence check + insert + title) runs as a
        single off-loop call, so concurrent same-ID requests serialize at
        the DB level.  Before the fix the TOCTOU window between the check
        and the insert let both requests pass the existence guard and both
        return 201 via the ON CONFLICT enrichment upsert.
        """
        import asyncio

        app = _create_app(auth_adapter)
        app.router.add_post("/api/sessions", auth_adapter._handle_create_session)

        async with TestClient(TestServer(app)) as cli:
            # Fire both requests concurrently through the same server.
            resp_a, resp_b = await asyncio.gather(
                cli.post(
                    "/api/sessions",
                    json={"id": "race-same-id"},
                    headers={"Authorization": "Bearer sk-secret"},
                ),
                cli.post(
                    "/api/sessions",
                    json={"id": "race-same-id"},
                    headers={"Authorization": "Bearer sk-secret"},
                ),
            )
        assert sorted([resp_a.status, resp_b.status]) == [201, 409]

    @pytest.mark.asyncio
    async def test_ensure_session_db_first_request_path(self, auth_adapter):
        """First /api/sessions request opens the SessionDB off the event loop.

        The open goes through the sidecar-anchored ``_open_profile_session_db``,
        so we patch that (not the SessionDB class) and assert the open ran on a
        worker thread, not the aiohttp event-loop thread.
        """
        import threading

        captured = {}
        loop_thread = threading.current_thread()

        class FakeDB:
            _profile_home_identity = None
            _profile_state_identity = None

            def list_sessions_rich(self, **kwargs):
                return []

        def fake_open(profile_home, *, create=True):
            captured["open_thread"] = threading.current_thread()
            return FakeDB()

        # Simulate cold start -- no DB yet.
        auth_adapter._session_db = None
        auth_adapter._session_dbs = {}

        with patch.object(auth_adapter, "_open_profile_session_db", side_effect=fake_open):
            app = _create_app(auth_adapter)
            app.router.add_get("/api/sessions", auth_adapter._handle_list_sessions)
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.get(
                    "/api/sessions",
                    headers={"Authorization": "Bearer sk-secret"},
                )
            assert resp.status == 200
            # The open must NOT run on the event-loop thread.
            assert "open_thread" in captured
            assert captured["open_thread"] != loop_thread
        auth_adapter._session_db = None
        auth_adapter._session_dbs = {}
class TestTakeoverUIHintOverSSE:
    """End-to-end coverage for the ui_hint field on the real streaming path.

    The payload builder is unit-tested elsewhere, but that cannot catch the
    field being dropped or mangled between the tool-complete callback, the
    stream queue, and SSE serialization — nor a regression that breaks plain
    conversational streaming for older clients.
    """

    @pytest.mark.asyncio
    async def test_browser_takeover_hint_reaches_the_sse_stream(self, adapter):
        app = _create_app(adapter)
        hint_result = json.dumps({
            "success": False,
            "error": "browser_takeover_required",
            "ui_hint": {
                "type": "takeover_browser",
                "agent_id": "agent-a",
                "browser_session_id": "chat-1",
                "tab_id": "tab-agent-1",
            },
        })

        async with TestClient(TestServer(app)) as cli:
            async def _mock_run_agent(**kwargs):
                delta = kwargs.get("stream_delta_callback")
                start = kwargs.get("tool_start_callback")
                complete = kwargs.get("tool_complete_callback")
                if delta:
                    delta("Opening the browser.")
                if start:
                    start("call_1", "browser_navigate", {"url": "https://example.com"})
                if complete:
                    complete("call_1", "browser_navigate", {"url": "https://example.com"}, hint_result)
                if delta:
                    delta(None)
                return (
                    {"final_response": "Opening the browser.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with (
                patch.object(adapter, "_response_format_transport_error", return_value=None),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "open example.com"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        # Plain conversational streaming is untouched — an older client that
        # ignores the custom event still gets its content and terminator.
        assert "Opening the browser." in body
        assert "[DONE]" in body

        events = [
            json.loads(line[len("data: "):])
            for block in body.split("\n\n")
            if "event: hermes.tool.progress" in block
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        completed = [e for e in events if e.get("status") == "completed"]
        assert completed, f"no completed tool progress event in stream: {body}"
        hint = completed[-1].get("ui_hint")
        assert hint == {
            "type": "takeover_browser",
            "agent_id": "agent-a",
            "browser_session_id": "chat-1",
            "tab_id": "tab-agent-1",
        }

    @pytest.mark.asyncio
    async def test_tool_progress_surrogate_does_not_break_the_sse_stream(self, adapter):
        app = _create_app(adapter)

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                start = kwargs.get("tool_start_callback")
                if start:
                    start("call_surrogate", "terminal", {"command": "ignored"})
                return (
                    {"final_response": "Done.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with (
                patch.object(
                    adapter, "_response_format_transport_error", return_value=None
                ),
                patch(
                    "agent.display.build_tool_preview",
                    return_value="Broken \ud800 label",
                ),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "run command"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        assert "[DONE]" in body
        running_wire = next(
            line[len("data: ") :]
            for block in body.split("\n\n")
            if "event: hermes.tool.progress" in block
            for line in block.splitlines()
            if line.startswith("data: ")
            and json.loads(line[len("data: ") :]).get("status") == "running"
        )
        assert "\\ud800" in running_wire
        assert json.loads(running_wire)["label"] == "Broken \ud800 label"

    @pytest.mark.asyncio
    async def test_browser_state_preview_reaches_the_sse_stream(self, adapter):
        app = _create_app(adapter)
        result = json.dumps(
            {
                "success": True,
                "url": "https://example.com/login?return_to=private",
                "title": "Sign \ud800in 状态🦞",
                "snapshot": (
                    '- heading "Sign in 状态🦞" [e1]\n'
                    '- textbox "Email" [e2]: private@example.com\n'
                    '- link "Open https%25253A%25252F%25252FExample.com%25252Fmagic%25252Fcredential" [e3]'
                ),
                "element_count": 3,
            }
        )

        async with TestClient(TestServer(app)) as cli:

            async def _mock_run_agent(**kwargs):
                start = kwargs.get("tool_start_callback")
                complete = kwargs.get("tool_complete_callback")
                from agent.codex_runtime import make_codex_app_server_event_bridge

                bridge = make_codex_app_server_event_bridge(types.SimpleNamespace(
                    tool_progress_callback=None,
                    tool_start_callback=start,
                    tool_complete_callback=complete,
                ))
                item = {
                    "type": "mcpToolCall",
                    "id": "call_state",
                    "server": "hermes-tools",
                    "tool": "browser_navigate",
                    "arguments": {"url": "https://example.com/login"},
                }
                bridge({"method": "item/started", "params": {"item": item}})
                bridge({
                    "method": "item/completed",
                    "params": {"item": {
                        **item,
                        "result": {"content": [{"type": "text", "text": result}]},
                    }},
                })
                return (
                    {"final_response": "Opened.", "messages": [], "api_calls": 1},
                    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                )

            with (
                patch.object(
                    adapter, "_response_format_transport_error", return_value=None
                ),
                patch.object(adapter, "_run_agent", side_effect=_mock_run_agent),
            ):
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": "open login"}],
                        "stream": True,
                    },
                )
                assert resp.status == 200
                body = await resp.text()

        event_data = [
            line[len("data: ") :]
            for block in body.split("\n\n")
            if "event: hermes.tool.progress" in block
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        events = [json.loads(data) for data in event_data]
        completed = [e for e in events if e.get("status") == "completed"]
        assert len(completed) == 1
        state = completed[0]["browserState"]
        assert state["url"] == {"hostname": "example.com"}
        assert state["title"] == "Sign ?in 状态🦞"
        assert state["truncated"] is True
        assert state["elements"] == [
            {"role": "heading", "label": "Sign in 状态🦞"},
            {"role": "textbox", "label": "Email"},
            {"role": "link", "label": "Open example.com"},
        ]
        completed_wire = next(
            data for data in event_data if json.loads(data).get("status") == "completed"
        )
        state_wire = json.dumps(
            state, ensure_ascii=False, separators=(",", ":")
        )
        assert state_wire in completed_wire
        assert "状态🦞" in completed_wire
        assert len(state_wire.encode("utf-8")) <= MAX_PREVIEW_BYTES
        assert "private@example.com" not in json.dumps(completed[0])
        assert "credential" not in json.dumps(completed[0])
        assert "snapshot" not in completed[0]
