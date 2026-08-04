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
from gateway.platforms.zet_agent import ZetAgentAdapter


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
