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


def test_zpk_stage_flow_excludes_root_build_inputs_but_keeps_nested_runtime_content(
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
        "plugins/web/exa/provider.py",
        "plugins/kanban/dashboard/dist/index.js",
        "plugins/hermes-achievements/docs/runtime-note.md",
        "plugins/hermes-achievements/tests/test_runtime_contract.py",
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
        "plugins/web/exa/provider.py",
        "plugins/kanban/dashboard/dist/index.js",
        "plugins/hermes-achievements/docs/runtime-note.md",
        "plugins/hermes-achievements/tests/test_runtime_contract.py",
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

    for relative_path in (kept_path, missing_path, ignored_path):
        _write_file(source, relative_path)
    _write_file(staged, kept_path)

    module = _load_check_module()

    assert module.find_missing_runtime_paths(source, staged) == [Path(missing_path)]


def test_zpk_stage_payload_check_flow_fails_when_runtime_content_is_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    missing_path = "plugins/kanban/dashboard/dist/index.js"
    _write_file(source, missing_path)
    (source / "venv").mkdir()
    (staged / "plugins").mkdir(parents=True)
    (staged / "venv").mkdir()

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
    shutil.copytree(source / "plugins", staged / "plugins")
    shutil.copytree(source / "venv", staged / "venv")

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
