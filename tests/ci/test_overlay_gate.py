"""Unit + flow tests for scripts/test-harness/overlay_gate.py (HR8)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "test-harness" / "overlay_gate.py"
CONFIG = ROOT / "scripts" / "test-harness" / "overlay_gate.json"

spec = importlib.util.spec_from_file_location("overlay_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = gate  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(gate)

MARKER = "# zettlab-overlay(TEST): keep pending steers as tuples; upstream: none"


@pytest.fixture(scope="module")
def config() -> dict:
    return gate.load_config(CONFIG)


# ---------------------------------------------------------------- unit


def test_protected_and_exempt_globs(config):
    assert gate.is_protected("run_agent.py", config)
    assert gate.is_protected("agent/conversation_loop.py", config)
    assert gate.is_protected("tools/todo_tool.py", config)
    assert gate.is_protected("gateway/platforms/api_server.py", config)
    assert not gate.is_protected("gateway/platforms/zet_agent.py", config)
    assert not gate.is_protected("gateway/platforms/ui_map/projector.py", config)
    assert not gate.is_protected("tests/agent/test_x.py", config)
    assert not gate.is_protected("README.md", config)
    # no wildcard exemptions inside the kernel directories
    assert gate.is_protected("tools/zet_overlay_state.py", config)
    assert gate.is_protected("agent/zet_agent_bridge.py", config)


def test_marker_regex_requires_upstream_field(config):
    marker = re.compile(config["marker_regex"])
    assert marker.search("# zettlab-overlay(B1): keep codes in lockstep; upstream: none")
    assert marker.search("# zettlab-overlay(U2d): tuple pending; upstream: https://github.com/NousResearch/hermes-agent/pull/1")
    assert marker.search("# zettlab-overlay(BT): hook; upstream: #42")
    assert not marker.search("# zettlab-overlay(B1): missing upstream field")
    assert not marker.search("# zettlab-overlay(B1) no colon; upstream: none")
    # must be a comment, not a string literal or docstring fragment
    assert not marker.search('marker = "zettlab-overlay(TEST): not a comment; upstream: none"')
    assert marker.search("    # zettlab-overlay(TEST): indented comment is fine; upstream: none")


def test_parse_hunks_separates_added_and_deleted():
    diff = (
        "diff --git a/run_agent.py b/run_agent.py\n"
        "--- a/run_agent.py\n+++ b/run_agent.py\n"
        "@@ -10,0 +11,2 @@\n+x = 1\n+\n"
        "@@ -20,2 +22,0 @@\n-old\n-older\n"
    )
    hunks = gate.parse_hunks(diff)
    assert [h.new_start for h in hunks] == [11, 22]
    assert hunks[0].added_nonblank == 1
    assert hunks[1].added_nonblank == 0 and hunks[1].removed == 2


def test_hunk_marker_lookback(config):
    marker = re.compile(config["marker_regex"])
    file_lines = ["a", MARKER, "b", "c", "d"]
    hunk = gate.Hunk(path="run_agent.py", new_start=4, new_count=1, added=["c"])
    assert gate.hunk_has_marker(hunk, file_lines, marker, lookback=5)
    assert not gate.hunk_has_marker(hunk, file_lines, marker, lookback=1)


def test_pr_body_parsers(config):
    assert gate.pr_body_has_upstream("## HR\nupstream-pr: none - generic hook, PR later", config)
    assert not gate.pr_body_has_upstream("no field here", config)
    assert gate.pr_body_has_upstream("upstream-pr: https://github.com/NousResearch/hermes-agent/pull/12", config)
    assert gate.pr_body_has_upstream("upstream-pr: #12", config)
    assert not gate.pr_body_has_upstream("upstream-pr: x", config)
    assert not gate.pr_body_has_upstream("upstream-pr: none", config)
    assert not gate.pr_body_has_upstream("upstream-pr: none - short", config)
    assert gate.pr_body_budget_exception("overlay-budget-exception: 上游 rebase 一次性同步三处 hook", config)
    assert gate.pr_body_budget_exception("overlay-budget-exception: short", config) is None


def test_skip_upstream_sync_branches(config):
    assert gate.should_skip("sync/upstream-v2026.9.1", config)
    assert not gate.should_skip("feat/chat-ui-u2d", config)
    assert not gate.should_skip("upstream-bypass", config)
    assert not gate.should_skip("upstream", config)
    assert not gate.should_skip("", config)


# ---------------------------------------------------------------- flow


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "agent").mkdir()
    (repo / "run_agent.py").write_text("def steer(self, text):\n    self._pending_steer = text\n", encoding="utf-8")
    (repo / "agent" / "loop.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "gateway" / "platforms").mkdir(parents=True)
    (repo / "gateway" / "platforms" / "zet_agent.py").write_text("adapter = True\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "branch", "-q", "base")
    return repo


def _commit(repo: Path, msg: str = "change") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)


def _run(repo: Path, config: dict, pr_body: str = "upstream-pr: none - unit test fixture", head_ref: str = "feat/x"):
    return gate.run_gate(repo, "base", "HEAD", pr_body, head_ref, config)


def test_flow_adapter_only_change_passes_without_marker(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "gateway" / "platforms" / "zet_agent.py").write_text("adapter = True\nmore = 2\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config, pr_body="")
    assert result.ok and result.protected_files == []


def test_flow_marked_small_core_change_passes(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "run_agent.py").write_text(
        f"def steer(self, text):\n    {MARKER}\n    self._pending_steer = [(None, text)]\n", encoding="utf-8"
    )
    _commit(repo)
    result = _run(repo, config)
    assert result.ok, [str(v) for v in result.violations]
    assert result.protected_files == ["run_agent.py"]


def test_flow_unmarked_core_hunk_fails(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config)
    assert not result.ok
    assert {v.kind for v in result.violations} == {"marker"}


def test_flow_deletion_only_hunk_needs_no_marker(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "run_agent.py").write_text("def steer(self, text):\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config)
    assert result.ok, [str(v) for v in result.violations]


def test_flow_missing_upstream_pr_line_fails(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "run_agent.py").write_text(f"{MARKER}\ndef steer(self, text):\n    self._pending_steer = text\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config, pr_body="no upstream line")
    assert {v.kind for v in result.violations} == {"upstream-pr"}


def test_flow_budget_exceeded_requires_exception(tmp_path, config):
    repo = _repo(tmp_path)
    body = MARKER + "\n" + "\n".join(f"v{i} = {i}" for i in range(config["added_lines_budget"] + 5)) + "\n"
    (repo / "agent" / "loop.py").write_text("x = 1\n" + body, encoding="utf-8")
    _commit(repo)
    result = _run(repo, config)
    assert {v.kind for v in result.violations} == {"budget"}
    result = _run(repo, config, pr_body="upstream-pr: none - unit test fixture\noverlay-budget-exception: one-off upstream rebase alignment")
    assert result.ok, [str(v) for v in result.violations]


def test_flow_business_state_in_core_fails(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text(
        f"x = 1\n{MARKER}\nagent._steer_binding_token = None\nagent._steer_accepted_sender = None\n", encoding="utf-8"
    )
    _commit(repo)
    result = _run(repo, config)
    kinds = [v.kind for v in result.violations]
    assert kinds.count("business-state") == 2 and "marker" not in kinds


def test_flow_upstream_lookalike_branch_not_skipped(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config, pr_body="", head_ref="upstream-bypass")
    assert not result.skipped and not result.ok
    assert {v.kind for v in result.violations} == {"marker", "upstream-pr"}


def test_flow_new_zet_file_in_kernel_dir_is_gated(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "tools").mkdir()
    (repo / "tools" / "zet_overlay_state.py").write_text("agent._steer_binding_token = None\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config)
    kinds = {v.kind for v in result.violations}
    assert result.protected_files == ["tools/zet_overlay_state.py"]
    assert {"marker", "business-state"} <= kinds


def test_flow_marker_inside_string_literal_is_not_a_marker(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text(
        'x = 1\nmarker = "zettlab-overlay(TEST): not a comment; upstream: none"\ny = 2\n', encoding="utf-8"
    )
    _commit(repo)
    result = _run(repo, config)
    assert {v.kind for v in result.violations} == {"marker"}


def test_flow_gitattributes_nodiff_cannot_hide_kernel_hunks(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / ".gitattributes").write_text("agent/** -diff\n", encoding="utf-8")
    (repo / "agent" / "loop.py").write_text("x = 1\nagent._binding_token = None\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config)
    kinds = {v.kind for v in result.violations}
    assert {"marker", "business-state"} <= kinds


def test_flow_upstream_sync_branch_skipped(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _commit(repo)
    result = _run(repo, config, pr_body="", head_ref="sync/upstream-v2026.9.1")
    assert result.skipped and result.ok


def test_cli_exit_codes(tmp_path, config):
    repo = _repo(tmp_path)
    (repo / "agent" / "loop.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _commit(repo)
    body = tmp_path / "body.txt"
    body.write_text("upstream-pr: none - unit test fixture", encoding="utf-8")
    proc = subprocess.run(
        ["python3", str(SCRIPT), "--repo", str(repo), "--base", "base", "--head", "HEAD", "--pr-body-file", str(body), "--config", str(CONFIG)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "[marker]" in proc.stdout
