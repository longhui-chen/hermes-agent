#!/usr/bin/env python3
"""Verify that ZPK staging preserved bundled runtime content."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TREES = ("plugins", "venv")
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".gk",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".worktrees",
    "__pycache__",
    "node_modules",
}
IGNORED_RELATIVE_PATHS = {
    Path("venv/.zpk-install-spec"),
    Path("venv/.zpk-venv.stamp"),
}
MAX_REPORTED_PATHS = 50


def _is_intentionally_excluded(relative_path: Path) -> bool:
    if relative_path in IGNORED_RELATIVE_PATHS:
        return True
    if relative_path.suffix == ".pyc":
        return True
    return any(
        part in IGNORED_DIRECTORY_NAMES or part.endswith(".egg-info")
        for part in relative_path.parts
    )


def find_missing_runtime_paths(source_root: Path, stage_root: Path) -> list[Path]:
    """Return source runtime files that disappeared while staging."""
    missing: list[Path] = []
    for tree_name in PROTECTED_TREES:
        source_tree = source_root / tree_name
        if not source_tree.is_dir():
            missing.append(Path(tree_name))
            continue

        for source_path in source_tree.rglob("*"):
            relative_path = source_path.relative_to(source_root)
            if (
                source_path.is_symlink()
                or not source_path.is_file()
                or _is_intentionally_excluded(relative_path)
            ):
                continue
            if not (stage_root / relative_path).is_file():
                missing.append(relative_path)

    return sorted(missing, key=lambda path: path.as_posix())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_root", type=Path, help="staged hermes-agent payload root")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT,
        help="source hermes-agent root (default: repository root)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    stage_root = args.stage_root.resolve()

    missing = find_missing_runtime_paths(source_root, stage_root)
    if missing:
        print("ZPK stage check failed: runtime content was removed during staging")
        for relative_path in missing[:MAX_REPORTED_PATHS]:
            print(f"  - {relative_path.as_posix()}")
        if len(missing) > MAX_REPORTED_PATHS:
            print(f"  - ... and {len(missing) - MAX_REPORTED_PATHS} more")
        return 1

    print("ZPK stage check ok: plugins/ and venv/ runtime files preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
