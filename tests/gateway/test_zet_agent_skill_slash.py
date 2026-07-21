"""ZET fork: inbound skill-slash expansion + reload AsyncSessionDB await fix.

Covers two zet_agent changes:

1. ``_expand_inbound_skill_slash`` — the App's skill quick-pick inserts a
   literal ``/<skill>`` line; the OpenAI-compatible surface previously passed
   it to the LLM verbatim (skill fired only if the model volunteered a
   skill_view call). The hook expands a known leading slash into the full
   skill payload and must pass everything else through byte-identical.

2. ``_handle_skills_reload`` — ``gateway_runner._session_db`` is the
   AsyncSessionDB facade whose methods return coroutines; the handler must
   await ``clear_all_system_prompts()`` (un-awaited it was a silent no-op and
   the coroutine object 500'd the JSON response).
"""

import asyncio
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import agent.skill_commands as skill_commands
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.zet_agent import ZetAgentAdapter


def _make_adapter() -> ZetAgentAdapter:
    return ZetAgentAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))


def _expand(adapter, message):
    """Drive the async hook to completion (it offloads to a worker thread)."""
    return asyncio.run(adapter._expand_inbound_skill_slash(message))


def _patch_skill_layer(monkeypatch, *, known=("deep-research",), load_raises=False, disabled=False):
    """Stub the skill_commands entry points the hook (and the canonical
    ``build_skill_invocation_message`` it delegates to) reach lazily."""
    import tools.skill_usage as skill_usage
    import tools.skills_tool as skills_tool

    from gateway.session_context import get_session_env

    calls = {}

    def fake_is_disabled(name, platform=None):
        calls["disabled_check"] = (name, platform)
        return disabled

    monkeypatch.setattr(skills_tool, "_is_skill_disabled", fake_is_disabled)

    def fake_scan():
        calls["scan_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
        return {
            f"/{name}": {"name": name, "skill_dir": f"/fake/skills/{name}"}
            for name in known
        }

    def fake_load(identifier, task_id=None):
        if load_raises:
            raise RuntimeError("boom")
        calls["load_identifier"] = identifier
        calls["load_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
        return ({"content": "SKILL BODY"}, Path("/fake/skills/deep-research"), "deep-research")

    def fake_build(loaded_skill, skill_dir, activation_note, user_instruction="", **kwargs):
        calls["user_instruction"] = user_instruction
        calls["activation_note"] = activation_note
        return f"<<EXPANDED:{loaded_skill['content']}|task={user_instruction}>>"

    monkeypatch.setattr(skill_commands, "scan_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "get_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "_load_skill_payload", fake_load)
    monkeypatch.setattr(skill_commands, "_build_skill_message", fake_build)
    monkeypatch.setattr(skill_usage, "bump_use", lambda name: None)
    return calls


def test_known_skill_slash_expands(monkeypatch):
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    out = _expand(adapter, "/deep-research 研究黄金为什么下跌")
    assert out.startswith("<<EXPANDED:SKILL BODY")
    assert calls["user_instruction"] == "研究黄金为什么下跌"
    assert calls["load_identifier"] == "/fake/skills/deep-research"


def test_duplicate_command_lines_collapse(monkeypatch):
    # The quick-pick appends rather than replaces, so retries stack the same
    # command; the expansion must collapse them into ONE invocation.
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    text = "/deep-research 黄金\n\n/deep-research 黄金\n\n/deep-research 黄金"
    out = _expand(adapter, text)
    assert out.count("<<EXPANDED:") == 1
    assert calls["user_instruction"] == "黄金"


def test_unknown_slash_passes_through(monkeypatch):
    _patch_skill_layer(monkeypatch, known=("other-skill",))
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original


def test_plain_text_and_multimodal_pass_through(monkeypatch):
    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    assert _expand(adapter, "你好，帮我查天气") == "你好，帮我查天气"
    multimodal = [{"type": "text", "text": "/deep-research x"}]
    assert _expand(adapter, multimodal) is multimodal


def test_load_failure_falls_back_to_original(monkeypatch):
    _patch_skill_layer(monkeypatch, load_raises=True)
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original


def test_expansion_binds_zet_agent_platform_and_restores(monkeypatch):
    # The expansion runs BEFORE the session is bound, so the hook must bind
    # the platform contextvar itself: without it scan/load resolve platform
    # None and skills.platform_disabled.zet_agent is silently ignored (a
    # skill disabled only for zet_agent would still expand). The binding must
    # also be token-restored — it must not leak past the hook.
    from gateway.session_context import get_session_env

    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    before = get_session_env("HERMES_SESSION_PLATFORM")
    _expand(adapter, "/deep-research 研究黄金")
    assert calls["scan_platform"] == "zet_agent"
    assert calls["load_platform"] == "zet_agent"
    assert get_session_env("HERMES_SESSION_PLATFORM") == before


def test_expanded_payload_uses_canonical_memory_scaffolding(monkeypatch):
    # Memory compatibility: the expanded payload must round-trip through
    # extract_user_instruction_from_skill_message (what MemoryManager.
    # _strip_skill_scaffolding calls). A bespoke activation note fails the
    # canonical-prefix check and the FULL skill body would be fed to memory
    # providers as if the user typed it.
    import tools.skill_usage as skill_usage
    import tools.skills_tool as skills_tool

    def fake_scan():
        return {"/deep-research": {"name": "deep-research", "skill_dir": "/fake/skills/deep-research"}}

    def fake_load(identifier, task_id=None):
        return ({"content": "SKILL BODY"}, None, "deep-research")

    monkeypatch.setattr(skill_commands, "scan_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "get_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "_load_skill_payload", fake_load)
    monkeypatch.setattr(skill_usage, "bump_use", lambda name: None)
    monkeypatch.setattr(skills_tool, "_is_skill_disabled", lambda name, platform=None: False)

    adapter = _make_adapter()
    out = _expand(adapter, "/deep-research 研究黄金为什么下跌")
    assert out.startswith(skill_commands._SKILL_INVOCATION_PREFIX)
    assert (
        skill_commands.extract_user_instruction_from_skill_message(out)
        == "研究黄金为什么下跌"
    )
    # Bare invocation → no user content worth remembering: extract must
    # return None so memory callers skip the turn entirely.
    bare = _expand(adapter, "/deep-research")
    assert bare.startswith(skill_commands._SKILL_INVOCATION_PREFIX)
    assert skill_commands.extract_user_instruction_from_skill_message(bare) is None


def test_platform_disabled_skill_passes_through_even_with_foreign_env(monkeypatch):
    # The disabled gate must use the EXPLICIT platform argument: the
    # resolution chain reads the HERMES_PLATFORM process env before the
    # contextvar, so an externally provisioned value would shadow the
    # binding. With the skill disabled for zet_agent the message must pass
    # through as plain text — no skill payload injected.
    monkeypatch.setenv("HERMES_PLATFORM", "telegram")
    calls = _patch_skill_layer(monkeypatch, disabled=True)
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert _expand(adapter, original) == original
    assert calls["disabled_check"] == ("deep-research", "zet_agent")
    assert "user_instruction" not in calls, "disabled skill must never be built"


def test_base_api_server_hook_is_noop():
    # HR4: plain api_server behavior must be unchanged — the base hook is
    # identity for every shape.
    base = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    for value in ("/deep-research x", "hello", ["parts"], None):
        assert asyncio.run(base._expand_inbound_skill_slash(value)) is value


@pytest.mark.asyncio
async def test_skills_reload_awaits_async_session_db_flow(monkeypatch):
    """Flow: /v1/skills/reload against an AsyncSessionDB-style runner.

    Regression for the un-awaited ``clear_all_system_prompts()``: with the
    async facade the old code put a coroutine into the JSON response (500)
    and never actually cleared the rows. The fixed handler must await and
    return the real row count.
    """
    import agent.prompt_builder as prompt_builder

    monkeypatch.setattr(
        prompt_builder, "clear_skills_system_prompt_cache", lambda **kw: None
    )
    monkeypatch.setattr(skill_commands, "scan_skill_commands", dict)

    async def _clear():
        return 7

    session_db = type(
        "AsyncFacade", (), {"clear_all_system_prompts": lambda self: _clear()}
    )()
    runner = type(
        "Runner",
        (),
        {"_session_db": session_db, "invalidate_all_cached_agents": lambda self: 3},
    )()

    adapter = _make_adapter()
    adapter.gateway_runner = runner
    app = web.Application()
    app.router.add_post("/v1/skills/reload", adapter._handle_skills_reload)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/v1/skills/reload", headers={"Authorization": "Bearer test-key"}
        )
        data = await resp.json()

    assert resp.status == 200
    assert data["db_rows_cleared"] == 7
    assert data["invalidated_sessions"] == 3
    assert data["cleared"] is True
