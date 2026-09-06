"""Flow coverage for Agent policy-gated Connector Skill discovery."""

from __future__ import annotations

import json

from agent import prompt_builder, skill_commands, skill_utils
from gateway.session_context import (
    pop_chat_connector_disabled_skills,
    pop_workload_attached_skills,
    push_chat_connector_disabled_skills,
    push_workload_attached_skills,
    workload_skill_scope,
)
from tools import skills_tool


def _write_skill(skills_dir, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: Description for {name}.\n"
        "---\n\n"
        f"# {name}\n\nInstructions for {name}.\n",
        encoding="utf-8",
    )


def _visible_names() -> set[str]:
    return {item["name"] for item in skills_tool._find_all_skills()}


def test_agent_policy_hides_all_surfaces_and_workload_overlay_is_exact(
    tmp_path, monkeypatch
) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for name in ("always-visible", "policy-hidden", "user-hidden"):
        _write_skill(skills_dir, name)
    (tmp_path / "config.yaml").write_text(
        "skills:\n"
        "  disabled:\n"
        "    - user-hidden\n"
        "  connector_policy_disabled:\n"
        "    - policy-hidden\n"
        "  connector_policy_generation: 17\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skills_tool, "_DEFAULT_SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    skill_utils._raw_config_cache_clear()
    skills_tool._SKILLS_CACHE.clear()
    prompt_builder.clear_skills_system_prompt_cache(clear_snapshot=True)
    skill_commands._skill_commands_cache.clear()
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_skills_dir_key", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_visibility_key", ())

    assert skill_utils.get_disabled_skill_names() == {
        "policy-hidden",
        "user-hidden",
    }
    assert _visible_names() == {"always-visible"}
    ambient_list = json.loads(skills_tool.skills_list())
    assert ambient_list["visibility_generation"] == 17
    assert {item["name"] for item in ambient_list["skills"]} == {
        "always-visible"
    }
    ambient_prompt = prompt_builder.build_skills_system_prompt()
    assert "always-visible" in ambient_prompt
    assert "policy-hidden" not in ambient_prompt
    assert "user-hidden" not in ambient_prompt
    ambient_commands = skill_commands.scan_skill_commands()
    assert "/always-visible" in ambient_commands
    assert "/policy-hidden" not in ambient_commands
    assert "/user-hidden" not in ambient_commands
    assert json.loads(skills_tool.skill_view("policy-hidden"))["success"] is False

    token = push_workload_attached_skills(
        "application", ["policy-hidden", "policy-hidden", "ignored-after-limit"]
    )
    try:
        assert workload_skill_scope() == (
            "application",
            ("policy-hidden", "ignored-after-limit"),
        )
        assert skill_utils.get_disabled_skill_names() == {"user-hidden"}
        assert _visible_names() == {"always-visible", "policy-hidden"}
        scoped_list = json.loads(skills_tool.skills_list())
        assert {item["name"] for item in scoped_list["skills"]} == {
            "always-visible",
            "policy-hidden",
        }
        scoped_prompt = prompt_builder.build_skills_system_prompt()
        assert "policy-hidden" in scoped_prompt
        assert "user-hidden" not in scoped_prompt
        scoped_commands = skill_commands.get_skill_commands()
        assert "/policy-hidden" in scoped_commands
        assert "/user-hidden" not in scoped_commands
        assert json.loads(skills_tool.skill_view("policy-hidden"))["success"] is True
        assert json.loads(skills_tool.skill_view("user-hidden"))["success"] is False
    finally:
        pop_workload_attached_skills(token)

    assert workload_skill_scope() == ("", ())
    assert _visible_names() == {"always-visible"}
    assert "/policy-hidden" not in skill_commands.get_skill_commands()


def test_workload_scope_is_bounded_and_rejects_unknown_sources() -> None:
    values = [f"skill-{index}" for index in range(40)]
    token = push_workload_attached_skills("cron", values)
    try:
        source, attached = workload_skill_scope()
        assert source == "cron"
        assert attached == tuple(values[:32])
    finally:
        pop_workload_attached_skills(token)

    try:
        push_workload_attached_skills("chat", ["skill-1"])
    except ValueError as exc:
        assert "application or cron" in str(exc)
    else:
        raise AssertionError("unknown workload source must fail closed")


def test_chat_override_deny_set_cannot_be_bypassed_by_skill_view(
    tmp_path, monkeypatch
) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "github")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skills_tool, "_DEFAULT_SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    skill_utils._raw_config_cache_clear()
    skills_tool._SKILLS_CACHE.clear()

    token = push_chat_connector_disabled_skills(["github"])
    try:
        assert "github" in skill_utils.get_disabled_skill_names()
        assert "github" not in _visible_names()
        assert json.loads(skills_tool.skill_view("github"))["success"] is False
    finally:
        pop_chat_connector_disabled_skills(token)

    assert "github" in _visible_names()


def test_chat_visibility_command_cache_is_bounded(tmp_path, monkeypatch) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        _write_skill(skills_dir, name)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skills_tool, "_DEFAULT_SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    monkeypatch.setattr(skill_commands, "_SKILL_COMMANDS_CACHE_MAX_ENTRIES", 2)
    skill_utils._raw_config_cache_clear()
    skill_commands._skill_commands_cache.clear()
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_skills_dir_key", None)
    monkeypatch.setattr(skill_commands, "_skill_commands_visibility_key", ())

    for disabled in (["alpha"], ["beta"], ["gamma"]):
        token = push_chat_connector_disabled_skills(disabled)
        try:
            skill_commands.get_skill_commands()
        finally:
            pop_chat_connector_disabled_skills(token)

    assert len(skill_commands._skill_commands_cache) == 2
