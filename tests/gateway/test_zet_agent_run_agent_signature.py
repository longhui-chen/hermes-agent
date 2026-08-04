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

from agent.zet_agent_response_mode import (
    _activate_execution_policy_tools,
    trusted_skill_operation_block_message,
)
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import (
    ZetAgentAdapter,
    _apply_execution_policy,
    _zettlab_workflow_addendum,
)


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


def test_silent_automation_uses_minimal_positive_tool_allowlist():
    def tool(name):
        return {"type": "function", "function": {"name": name}}

    agent = type("Agent", (), {})()
    agent.tools = [
        tool("clarify"),
        tool("memory"),
        tool("present_plan"),
        tool("todo"),
        tool("send_message"),
        tool("list_my_channels"),
        tool("send_channel_message"),
        tool("terminal"),
        tool("skill_view"),
        tool("read_file"),
        tool("session_search"),
        tool("skill_manage"),
        tool("cronjob"),
        tool("delegate_task"),
        tool("call_agent"),
        tool("app_host"),
    ]
    agent.valid_tool_names = {
        item["function"]["name"] for item in agent.tools
    }

    _apply_execution_policy(agent, "silent_automation")

    assert agent.valid_tool_names == {"skill_view"}
    assert {
        item["function"]["name"] for item in agent.tools
    } == agent.valid_tool_names
    assert {
        item["function"]["name"]
        for item in agent._zet_agent_execution_policy_tools
    } == {"terminal", "skill_view"}
    assert trusted_skill_operation_block_message(
        agent,
        function_name="terminal",
        function_args={"command": "echo unsafe"},
    )

    _activate_execution_policy_tools(
        agent,
        frozenset({"terminal", "clarify", "todo"}),
    )

    assert agent.valid_tool_names == {"terminal"}
    assert {
        item["function"]["name"] for item in agent.tools
    } == {"terminal"}


def test_unknown_execution_policy_leaves_tool_snapshot_unchanged():
    tools = [{"type": "function", "function": {"name": "clarify"}}]
    agent = type("Agent", (), {})()
    agent.tools = tools
    agent.valid_tool_names = {"clarify"}

    _apply_execution_policy(agent, "future_policy")

    assert agent.tools is tools
    assert agent.valid_tool_names == {"clarify"}


def test_silent_automation_prompt_does_not_instruct_missing_interaction_tools():
    prompt = _zettlab_workflow_addendum(True, "silent_automation")

    assert "可信静默自动化" in prompt
    assert "不调用澄清、计划、todo 或消息发送工具" in prompt
    assert "present_plan" not in prompt
    assert "不读取或修改用户画像" in prompt
    assert "等待用户确认后再执行" not in prompt


def test_silent_automation_skips_memory_before_agent_construction(monkeypatch):
    constructed = []
    instances = []

    class FakeAgent:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            instances.append(self)
            self.tools = []
            self.valid_tool_names = set()
            self._skip_mcp_refresh = False
            self._persist_disabled = False
            self._session_db = kwargs.get("session_db")
            self._session_json_enabled = True

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "base_url": "https://example.test/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-test")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda: {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    monkeypatch.setattr(adapter, "_session_model_override_for", lambda _key: None)
    bound_session_keys = []
    monkeypatch.setattr(
        adapter,
        "_bind_turn_session_context",
        lambda session_key, session_id="": bound_session_keys.append(
            (session_key, session_id)
        ),
    )

    adapter._create_agent(
        session_id="api-lineage-tip",
        gateway_session_key="profile-scoped-interaction-key",
        request_overrides={
            "_zet_execution_policy": "silent_automation",
            "_zet_stable_gateway_session_key": (
                "proactive-pvm-aaaaaaaaaaaaaaaaaaaaaaaa"
            ),
        },
    )
    adapter._create_agent(
        session_id="ordinary-session",
        gateway_session_key="ordinary-session",
    )

    assert constructed[0]["skip_memory"] is True
    assert constructed[1]["skip_memory"] is False
    assert instances[0]._persist_disabled is True
    assert instances[0]._session_db is None
    assert instances[0]._session_json_enabled is False
    assert instances[1]._persist_disabled is False
    assert instances[1]._session_json_enabled is True
    assert bound_session_keys == [
        ("proactive-pvm-aaaaaaaaaaaaaaaaaaaaaaaa", "api-lineage-tip"),
        ("ordinary-session", "ordinary-session"),
    ]
    assert "_zet_stable_gateway_session_key" not in (
        constructed[0].get("request_overrides") or {}
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


@pytest.mark.asyncio
async def test_zet_agent_preserves_stable_session_key_across_queue_scoping(
    monkeypatch,
):
    captured = {}

    async def fake_run_agent(self, **kwargs):
        del self
        captured.update(kwargs)
        return (
            {"final_response": "ok", "session_id": "api-lineage-tip"},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    monkeypatch.setattr(APIServerAdapter, "_run_agent", fake_run_agent)
    adapter = ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    stable_key = "proactive-pvm-aaaaaaaaaaaaaaaaaaaaaaaa"

    await adapter._run_agent(
        user_message="run proactive video",
        session_id="api-lineage-tip",
        gateway_session_key=stable_key,
    )

    assert captured["gateway_session_key"] == adapter._interaction_queue_key(
        "api-lineage-tip"
    )
    assert captured["request_overrides"][
        "_zet_stable_gateway_session_key"
    ] == stable_key
