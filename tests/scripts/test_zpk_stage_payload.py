from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_zpk_stage.py"
PROTECTED_TREES = (
    "config",
    "locales",
    "optional-mcps",
    "optional-skills",
    "plugins",
    "skills",
    "venv",
)


def _zpk_excludes() -> list[str]:
    result = subprocess.run(
        [
            "make",
            "--silent",
            "--no-print-directory",
            "-f",
            str(MAKEFILE),
            "-f",
            "-",
            "print-zpk-excludes",
        ],
        input="print-zpk-excludes:\n\t@for arg in $(ZPK_EXCLUDES); do printf '%s\\n' \"$$arg\"; done\n",
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return result.stdout.splitlines()


def _write_file(root: Path, relative_path: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(relative_path, encoding="utf-8")


def _create_protected_trees(root: Path) -> None:
    for tree in PROTECTED_TREES:
        (root / tree).mkdir(parents=True, exist_ok=True)


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_zpk_stage", CHECK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_gnu_tar() -> bool:
    if shutil.which("tar") is None or shutil.which("make") is None:
        return False
    result = subprocess.run(
        ["tar", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return "GNU tar" in result.stdout


def test_zpk_stage_flow_preserves_bundled_plugins_and_excludes_root_build_inputs(
    tmp_path: Path,
) -> None:
    if not _is_gnu_tar():
        pytest.skip("ZPK staging flow requires make and GNU tar")

    source = tmp_path / "source"
    staged = tmp_path / "staged"
    archive = tmp_path / "payload.tar"

    for relative_path in (
        "web/root-only.txt",
        "data/root-only.txt",
        "dist/root-only.txt",
        "docs/root-only.txt",
        "tests/root-only.txt",
        "config/skill_seed_policy.json",
        "locales/en.yaml",
        "optional-mcps/linear/manifest.yaml",
        "optional-skills/research/demo/SKILL.md",
        "plugins/web/exa/provider.py",
        "plugins/kanban/dashboard/dist/index.js",
        "plugins/hermes-achievements/README.md",
        "plugins/hermes-achievements/dashboard/dist/index.js",
        "plugins/hermes-achievements/docs/runtime-note.md",
        "plugins/hermes-achievements/tests/test_runtime_contract.py",
        "skills/software-development/plan/SKILL.md",
        "venv/lib/python3.11/site-packages/botocore/data/endpoints.json",
        "venv/lib/python3.11/site-packages/slack_sdk/web/client.py",
    ):
        _write_file(source, relative_path)

    staged.mkdir()
    subprocess.run(
        ["tar", *_zpk_excludes(), "-cf", str(archive), "-C", str(source), "."],
        check=True,
    )
    subprocess.run(["tar", "-xf", str(archive), "-C", str(staged)], check=True)

    for root_only_path in (
        "web/root-only.txt",
        "data/root-only.txt",
        "dist/root-only.txt",
        "docs/root-only.txt",
        "tests/root-only.txt",
    ):
        assert not (staged / root_only_path).exists()

    for runtime_path in (
        "config/skill_seed_policy.json",
        "locales/en.yaml",
        "optional-mcps/linear/manifest.yaml",
        "optional-skills/research/demo/SKILL.md",
        "plugins/web/exa/provider.py",
        "plugins/kanban/dashboard/dist/index.js",
        "plugins/hermes-achievements/README.md",
        "plugins/hermes-achievements/dashboard/dist/index.js",
        "plugins/hermes-achievements/docs/runtime-note.md",
        "plugins/hermes-achievements/tests/test_runtime_contract.py",
        "skills/software-development/plan/SKILL.md",
        "venv/lib/python3.11/site-packages/botocore/data/endpoints.json",
        "venv/lib/python3.11/site-packages/slack_sdk/web/client.py",
    ):
        assert (staged / runtime_path).is_file(), f"runtime path was excluded: {runtime_path}"


def test_find_missing_runtime_paths_reports_files_removed_from_protected_trees(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    kept_path = "plugins/web/exa/provider.py"
    missing_path = "venv/lib/python3.11/site-packages/botocore/data/endpoints.json"
    ignored_path = "venv/lib/python3.11/site-packages/demo/__pycache__/module.pyc"
    missing_plugin_path = "plugins/hermes-achievements/dashboard/dist/index.js"

    for relative_path in (kept_path, missing_path, ignored_path, missing_plugin_path):
        _write_file(source, relative_path)
    _write_file(staged, kept_path)
    _create_protected_trees(source)
    _create_protected_trees(staged)

    module = _load_check_module()

    assert module.find_missing_runtime_paths(source, staged) == [
        Path(missing_plugin_path),
        Path(missing_path),
    ]


def test_zpk_stage_payload_check_flow_fails_when_runtime_content_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    missing_path = "plugins/kanban/dashboard/dist/index.js"
    _create_protected_trees(source)
    _create_protected_trees(staged)
    _write_file(source, missing_path)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--source-root",
            str(source),
            str(staged),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 1
    assert "ZPK stage check failed" in result.stdout
    assert missing_path in result.stdout


def test_zpk_stage_payload_check_flow_accepts_complete_runtime_trees(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    runtime_paths = (
        "plugins/web/exa/provider.py",
        "plugins/kanban/dashboard/dist/index.js",
        "venv/lib/python3.11/site-packages/botocore/data/endpoints.json",
    )
    for relative_path in runtime_paths:
        _write_file(source, relative_path)
    _create_protected_trees(source)
    shutil.copytree(source, staged)

    result = subprocess.run(
        [
            sys.executable,
            str(CHECK_SCRIPT),
            "--source-root",
            str(source),
            str(staged),
        ],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "ZPK stage check ok" in result.stdout
