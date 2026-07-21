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


def _patch_skill_layer(monkeypatch, *, known=("deep-research",), load_raises=False):
    """Stub the three skill_commands entry points the hook imports lazily."""
    calls = {}

    def fake_scan():
        return {
            f"/{name}": {"name": name, "skill_dir": f"/fake/skills/{name}"}
            for name in known
        }

    def fake_load(identifier, task_id=None):
        if load_raises:
            raise RuntimeError("boom")
        calls["load_identifier"] = identifier
        return ({"content": "SKILL BODY"}, Path("/fake/skills/deep-research"), "deep-research")

    def fake_build(loaded_skill, skill_dir, activation_note, user_instruction="", **kwargs):
        calls["user_instruction"] = user_instruction
        calls["activation_note"] = activation_note
        return f"<<EXPANDED:{loaded_skill['content']}|task={user_instruction}>>"

    monkeypatch.setattr(skill_commands, "scan_skill_commands", fake_scan)
    monkeypatch.setattr(skill_commands, "_load_skill_payload", fake_load)
    monkeypatch.setattr(skill_commands, "_build_skill_message", fake_build)
    return calls


def test_known_skill_slash_expands(monkeypatch):
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    out = adapter._expand_inbound_skill_slash("/deep-research 研究黄金为什么下跌")
    assert out.startswith("<<EXPANDED:SKILL BODY")
    assert calls["user_instruction"] == "研究黄金为什么下跌"
    assert calls["load_identifier"] == "/fake/skills/deep-research"


def test_duplicate_command_lines_collapse(monkeypatch):
    # The quick-pick appends rather than replaces, so retries stack the same
    # command; the expansion must collapse them into ONE invocation.
    calls = _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    text = "/deep-research 黄金\n\n/deep-research 黄金\n\n/deep-research 黄金"
    out = adapter._expand_inbound_skill_slash(text)
    assert out.count("<<EXPANDED:") == 1
    assert calls["user_instruction"] == "黄金"


def test_unknown_slash_passes_through(monkeypatch):
    _patch_skill_layer(monkeypatch, known=("other-skill",))
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert adapter._expand_inbound_skill_slash(original) == original


def test_plain_text_and_multimodal_pass_through(monkeypatch):
    _patch_skill_layer(monkeypatch)
    adapter = _make_adapter()
    assert adapter._expand_inbound_skill_slash("你好，帮我查天气") == "你好，帮我查天气"
    multimodal = [{"type": "text", "text": "/deep-research x"}]
    assert adapter._expand_inbound_skill_slash(multimodal) is multimodal


def test_load_failure_falls_back_to_original(monkeypatch):
    _patch_skill_layer(monkeypatch, load_raises=True)
    adapter = _make_adapter()
    original = "/deep-research 研究黄金"
    assert adapter._expand_inbound_skill_slash(original) == original


def test_base_api_server_hook_is_noop():
    # HR4: plain api_server behavior must be unchanged — the base hook is
    # identity for every shape.
    base = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    for value in ("/deep-research x", "hello", ["parts"], None):
        assert base._expand_inbound_skill_slash(value) is value


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
