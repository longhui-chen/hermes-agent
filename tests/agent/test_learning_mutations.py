"""Behavior contracts for journey node edit/delete (agent.learning_mutations).

Exercises the real on-disk resolution (skills dir + MEMORY.md/USER.md chunking)
against a temp HERMES_HOME, never mocks — the id→file mapping is the whole point.
"""

from __future__ import annotations

import threading

import pytest

from agent import learning_mutations as lm
from hermes_constants import get_hermes_home

_SKILL = """---
name: my-skill
description: A test skill.
---

# My Skill

Body.
"""


@pytest.fixture
def home():
    base = get_hermes_home()
    (base / "memories").mkdir(parents=True, exist_ok=True)
    (base / "memories" / "MEMORY.md").write_text("alpha note\nline two\n§\nbeta note", encoding="utf-8")
    (base / "memories" / "USER.md").write_text("user profile note", encoding="utf-8")
    skill = base / "skills" / "my-skill"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    return base


def test_parse_node_kind():
    assert lm.parse_node_kind("memory:memory:0") == "memory"
    assert lm.parse_node_kind("memory:profile:3") == "memory"
    assert lm.parse_node_kind("debugging-hermes") == "skill"


def test_memory_global_index_maps_across_files(home):
    # MEMORY.md → indices 0,1; USER.md → index 2 (global, memory cards first).
    assert lm.node_detail("memory:memory:0")["content"].startswith("alpha note")
    assert lm.node_detail("memory:memory:1")["content"] == "beta note"
    assert lm.node_detail("memory:profile:2")["content"] == "user profile note"


def test_memory_label_is_first_line(home):
    assert lm.node_detail("memory:memory:0")["label"] == "alpha note"


def test_delete_memory_rewrites_file(home):
    assert lm.delete_node("memory:memory:0")["ok"]
    remaining = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "alpha note" not in remaining
    assert "beta note" in remaining


def test_edit_memory_replaces_chunk(home):
    assert lm.edit_node("memory:profile:2", "rewritten profile")["ok"]
    assert (home / "memories" / "USER.md").read_text(encoding="utf-8").strip() == "rewritten profile"


def test_edit_memory_empty_is_rejected(home):
    res = lm.edit_node("memory:memory:1", "   ")
    assert not res["ok"]
    assert "delete" in res["message"]


def test_stale_memory_index_errors(home):
    res = lm.node_detail("memory:memory:9")
    assert not res["ok"]


def test_bad_memory_id_returns_error(home):
    res = lm.delete_node("memory:bogus:0")
    assert not res["ok"]


def test_skill_detail_returns_skill_md(home):
    d = lm.node_detail("my-skill")
    assert d["ok"] and d["kind"] == "skill"
    assert "name: my-skill" in d["content"]


def test_delete_skill_archives_recoverably(home):
    res = lm.delete_node("my-skill")
    assert res["ok"]
    assert not (home / "skills" / "my-skill").exists()
    assert (home / "skills" / ".archive" / "my-skill" / "SKILL.md").exists()


def test_delete_pinned_skill_refused(home):
    from tools import skill_usage

    skill_usage.set_pinned("my-skill", True)
    res = lm.delete_node("my-skill")
    assert not res["ok"]
    assert "pinned" in res["message"]
    assert (home / "skills" / "my-skill").exists()


def test_edit_skill_rewrites_and_validates(home):
    bad = lm.edit_node("my-skill", "no frontmatter here")
    assert not bad["ok"]
    good = lm.edit_node("my-skill", _SKILL.replace("A test skill.", "Updated desc."))
    assert good["ok"]
    assert "Updated desc." in (home / "skills" / "my-skill" / "SKILL.md").read_text(encoding="utf-8")


def test_missing_skill_detail(home):
    assert not lm.node_detail("nonexistent-skill")["ok"]


def test_memory_writes_match_memory_tool_format(home):
    """A journey mutation must leave the file byte-identical to what the memory
    tool itself writes — same §-join, no trailing-newline drift — so the two
    surfaces never fight over format and indices stay aligned."""
    from tools.memory_tool import ENTRY_DELIMITER, MemoryStore

    assert lm.edit_node("memory:memory:0", "alpha rewritten")["ok"]
    path = home / "memories" / "MEMORY.md"
    entries = MemoryStore._read_file(path)

    assert entries == ["alpha rewritten", "beta note"]
    assert path.read_text(encoding="utf-8") == ENTRY_DELIMITER.join(entries)


@pytest.mark.parametrize("operation", ["edit", "delete"])
def test_memory_mutation_cannot_resurrect_data_after_privacy_reset(
    home, monkeypatch, operation
):
    """A journey RMW that starts during reset must locate only after reset.

    The reset is paused after atomically staging MEMORY.md while it still owns
    the transaction and target locks.  Edit/delete must remain outside
    ``_locate_memory`` until reset releases those locks, then observe that the
    canonical file is gone instead of publishing a stale pre-reset snapshot.
    """
    from tools import memory_tool

    reset_staged = threading.Event()
    release_reset = threading.Event()
    mutation_started = threading.Event()
    locate_started = threading.Event()
    mutation_done = threading.Event()
    failures: list[BaseException] = []
    result: dict[str, object] = {}

    original_move = memory_tool._reset_move_no_replace
    original_locate = lm._locate_memory

    def paused_move(handles, source_scope, source_name, stage_name, stage_scope=None):
        original_move(handles, source_scope, source_name, stage_name, stage_scope)
        if source_scope == "memory" and source_name == "MEMORY.md":
            reset_staged.set()
            if not release_reset.wait(5):
                raise AssertionError("test did not release paused reset")

    def observed_locate(source, global_index):
        locate_started.set()
        return original_locate(source, global_index)

    monkeypatch.setattr(memory_tool, "_reset_move_no_replace", paused_move)
    monkeypatch.setattr(lm, "_locate_memory", observed_locate)

    def run_reset():
        try:
            memory_tool.reset_curated_memory("memory")
        except BaseException as exc:
            failures.append(exc)

    def run_mutation():
        mutation_started.set()
        try:
            if operation == "edit":
                result.update(lm.edit_node("memory:memory:0", "stale rewrite"))
            else:
                result.update(lm.delete_node("memory:memory:0"))
        except BaseException as exc:
            failures.append(exc)
        finally:
            mutation_done.set()

    reset_thread = threading.Thread(target=run_reset)
    mutation_thread = threading.Thread(target=run_mutation)
    reset_thread.start()
    assert reset_staged.wait(5)

    mutation_thread.start()
    assert mutation_started.wait(5)
    assert not locate_started.wait(0.2)
    assert not mutation_done.is_set()

    release_reset.set()
    reset_thread.join(5)
    mutation_thread.join(5)

    assert not reset_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert failures == []
    assert result["ok"] is False
    assert "not found" in str(result["message"])
    assert not (home / "memories" / "MEMORY.md").exists()
