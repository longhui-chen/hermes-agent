"""Regression guards for the fork's upstream-release PR workflow."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "upstream-release-pr.yml"


def _existing_pr_branch(script: str) -> str:
    marker = 'if [ -n "$EXISTING" ]; then'
    assert marker in script
    return script.split(marker, 1)[1].split("\n          else\n", 1)[0]


def test_upstream_release_workflow_is_valid_yaml() -> None:
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert parsed["jobs"]["open-pr"]["steps"]


def test_existing_sync_pr_body_is_preserved() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    existing_branch = _existing_pr_branch(workflow)

    assert "preserving its body" in existing_branch
    assert "gh api -X PATCH" not in existing_branch
    assert '-f body="$BODY"' not in existing_branch

    # A newly created PR must still receive the generated bootstrap body.
    create_branch = workflow.split("\n          else\n", 1)[1]
    assert "gh api -X POST" in create_branch
    assert '-f body="$BODY"' in create_branch
