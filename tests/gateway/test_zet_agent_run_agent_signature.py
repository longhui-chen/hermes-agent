"""Regression: ZetAgentAdapter._run_agent must stay signature-compatible with
the base APIServerAdapter._run_agent it overrides.

ZET-1610 added ``turn_id`` to the base ``_run_agent`` + the
``_handle_chat_completions`` call site, but ``ZetAgentAdapter`` re-declares the
override's full signature explicitly and did NOT get ``turn_id`` — so every
chat completion through the zet_agent platform 500'd with
``TypeError: ... unexpected keyword argument 'turn_id'``.

This guards the base↔override seam: the override must accept every keyword the
base declares (or use ``**kwargs``). It is a pure-signature test so it runs
fast and would have gone red the moment the override drifted.
"""

import inspect

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import (
    ZetAgentAdapter,
    _api_request_profile,
    _onboarding_deepseek_fast_path,
)


def test_onboarding_deepseek_fast_path_sets_supported_wire_field():
    reasoning, overrides, enabled = _onboarding_deepseek_fast_path(
        profile="onboarding",
        model="deepseek-v4-flash",
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides={"extra_body": {"existing": 1}},
    )

    assert enabled is True
    assert reasoning == {"enabled": False}
    assert overrides == {
        "extra_body": {
            "existing": 1,
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        }
    }


def test_onboarding_fast_path_applies_to_catalog_alias_model():
    reasoning, overrides, enabled = _onboarding_deepseek_fast_path(
        profile="onboarding",
        model="lite",
        reasoning_config={"enabled": True},
        request_overrides={},
    )

    assert enabled is True
    assert reasoning == {"enabled": False}
    assert overrides == {
        "extra_body": {
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
        }
    }


def test_onboarding_fast_path_does_not_touch_normal_agent():
    original_reasoning = {"enabled": True, "effort": "high"}
    original_overrides = {"extra_body": {"existing": 1}}
    reasoning, overrides, enabled = _onboarding_deepseek_fast_path(
        profile="main",
        model="deepseek-v4-flash",
        reasoning_config=original_reasoning,
        request_overrides=original_overrides,
    )

    assert enabled is False
    assert reasoning is original_reasoning
    assert overrides == original_overrides


def _keyword_params(func):
    """Return (set of accepted keyword names, has **kwargs) excluding self."""
    names = set()
    has_var_kw = False
    for name, p in inspect.signature(func).parameters.items():
        if name == "self":
            continue
        if p.kind is p.VAR_KEYWORD:
            has_var_kw = True
        elif p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.add(name)
    return names, has_var_kw


def test_zet_agent_run_agent_accepts_turn_id():
    params, has_var_kw = _keyword_params(ZetAgentAdapter._run_agent)
    assert has_var_kw or "turn_id" in params, (
        "ZetAgentAdapter._run_agent must accept turn_id — _handle_chat_completions "
        "passes it; a missing param 500s every chat (ZET-1610 regression)."
    )


def test_zet_agent_run_agent_covers_base_signature():
    base_params, _ = _keyword_params(APIServerAdapter._run_agent)
    override_params, has_var_kw = _keyword_params(ZetAgentAdapter._run_agent)
    missing = base_params - override_params
    assert has_var_kw or not missing, (
        f"ZetAgentAdapter._run_agent override is missing base keyword(s): {sorted(missing)}. "
        "When you add a param to the base _run_agent, mirror it in this override "
        "(or switch the override to **kwargs) — it sits on the live chat path."
    )


def test_zet_agent_create_agent_covers_base_signature():
    base_params, _ = _keyword_params(APIServerAdapter._create_agent)
    override_params, has_var_kw = _keyword_params(ZetAgentAdapter._create_agent)
    missing = base_params - override_params
    assert has_var_kw or not missing, (
        f"ZetAgentAdapter._create_agent override is missing base keyword(s): {sorted(missing)}. "
        "When you add a param to the base _create_agent, mirror it in this override "
        "(or switch the override to **kwargs) — chat completions calls this through the subclass."
    )


def test_zet_agent_connect_covers_base_signature():
    base_params, _ = _keyword_params(APIServerAdapter.connect)
    override_params, has_var_kw = _keyword_params(ZetAgentAdapter.connect)
    missing = base_params - override_params
    assert has_var_kw or not missing, (
        f"ZetAgentAdapter.connect override is missing base keyword(s): {sorted(missing)}. "
        "Gateway startup and reconnect both call connect(is_reconnect=...), so "
        "signature drift prevents the zet_agent gateway from binding its health port."
    )


def test_zet_agent_create_agent_applies_request_runtime_options(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")
            self.provider = kwargs.get("provider")

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "global-provider",
            "model": "global/model",
            "api_key": "global-key",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        lambda: {"enabled": False},
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", lambda: None
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "gateway.platforms.zet_agent._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: {
            "provider": provider,
            "model": target_model,
            "api_key": "request-key",
        },
    )

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    public_session_id = "zettlab:userA:main:session-1"
    scoped_session_key = f"/profiles/main|{public_session_id}"
    agent = adapter._create_agent(
        session_id=public_session_id,
        gateway_session_key=scoped_session_key,
        requested_model="request/model",
        requested_provider="request-provider",
        model_options={"reasoning_effort": "high", "service_tier": "priority"},
    )

    assert isinstance(agent, FakeAgent)
    assert captured["model"] == "request/model"
    assert captured["provider"] == "request-provider"
    assert captured["api_key"] == "request-key"
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert captured["service_tier"] == "priority"
    assert captured["platform"] == "zet_agent"
    assert captured["profile_name"] == "main"


def test_onboarding_agent_is_lightweight_before_construction(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")
            self.provider = kwargs.get("provider")

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {"provider": "custom", "api_key": "test-key"},
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "lite")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("gateway.run._checkpoint_agent_kwargs", lambda _cfg: {})
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        lambda: {"enabled": True},
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model", lambda: None
    )
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools", lambda *_: {"terminal", "memory"}
    )
    monkeypatch.setattr(
        "agent.prompt_builder.load_soul_md", lambda *_: "authoritative v14 policy"
    )

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda *_: None)

    profile_token = _api_request_profile.set("onboarding")
    try:
        agent = adapter._create_agent(
            ephemeral_system_prompt="v14 onboarding policy",
            session_id="onboarding-session",
            gateway_session_key="zettlab:user:onboarding:session",
        )
    finally:
        _api_request_profile.reset(profile_token)

    assert captured["enabled_toolsets"] == []
    assert captured["skip_tool_loading"] is True
    assert captured["skip_context_files"] is True
    assert captured["skip_memory"] is True
    assert captured["ephemeral_system_prompt"] == "v14 onboarding policy"
    assert captured["reasoning_config"] == {"enabled": False}
    assert agent._tools_disabled_for_request is True
    assert agent.compression_enabled is False
    assert "authoritative v14 policy" in agent._cached_system_prompt


@pytest.mark.asyncio
async def test_zet_agent_forwards_current_turn_reference_image(monkeypatch):
    captured = {}

    async def fake_run_agent(self, **kwargs):
        del self
        captured.update(kwargs)
        return (
            {"final_response": "ok", "session_id": "session-1"},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_run_agent)
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))

    await adapter._run_agent(
        user_message="create a desktop pet",
        session_id="session-1",
        current_turn_reference_image="data:image/png;base64,cGV0",
    )

    assert captured["current_turn_reference_image"] == "data:image/png;base64,cGV0"
