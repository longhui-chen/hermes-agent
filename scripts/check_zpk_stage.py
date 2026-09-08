#!/usr/bin/env python3
"""Verify that ZPK staging preserved bundled runtime content."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import struct
import subprocess
from itertools import chain
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TREES = (
    "config",
    "locales",
    "optional-mcps",
    "optional-skills",
    "plugins",
    "skills",
    "tools",
    "venv",
)
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
PYTHON_RELATIVE_PATH = Path("venv/bin/python")
PYTHON_CONFIG_RELATIVE_PATH = Path("venv/pyvenv.cfg")
ELF_MACHINE_AARCH64 = 183


def find_invalid_venv_permissions(stage_root: Path) -> list[tuple[Path, int, int]]:
    """Return staged venv paths whose modes violate the package contract."""
    venv_root = stage_root / "venv"
    invalid: list[tuple[Path, int, int]] = []
    for path in chain((venv_root,), venv_root.rglob("*")):
        if path.is_symlink():
            continue
        actual = stat.S_IMODE(path.stat().st_mode)
        if path == venv_root:
            expected = 0o700
        elif path.is_dir():
            expected = 0o755
        elif path == venv_root / ".lock":
            expected = 0o600
        elif path.is_file():
            expected = 0o755 if actual & 0o111 else 0o644
        else:
            continue
        if actual != expected:
            invalid.append((path.relative_to(stage_root), expected, actual))
    return invalid


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


def _elf_machine(path: Path) -> int | None:
    header = path.read_bytes()[:20]
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    if header[5] == 1:
        return struct.unpack_from("<H", header, 18)[0]
    if header[5] == 2:
        return struct.unpack_from(">H", header, 18)[0]
    raise RuntimeError(f"unsupported ELF endianness in {path}")


def _needed_libraries(path: Path) -> tuple[str, ...]:
    readelf = "readelf"
    if not shutil.which(readelf):
        raise RuntimeError(
            "readelf is required to inspect the staged Python interpreter"
        )
    result = subprocess.run(
        [readelf, "-d", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"readelf failed for staged Python: {result.stdout.strip()}")
    return tuple(re.findall(r"Shared library: \[([^\]]+)\]", result.stdout))


def _missing_dynamic_libraries(path: Path) -> tuple[str, ...]:
    ldd = shutil.which("ldd")
    if not ldd:
        raise RuntimeError("ldd is required to inspect the staged Python interpreter")
    result = subprocess.run(
        [ldd, str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ldd failed for staged Python: {result.stdout.strip()}")
    return tuple(
        match.group(1)
        for line in result.stdout.splitlines()
        if (match := re.search(r"^\s*([^\s]+)\s*=>\s*not found\s*$", line))
    )


def _venv_config_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"staged Python venv config is missing: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def check_staged_python(
    stage_root: Path,
    *,
    target_arch: str | None = None,
    python_version: str | None = None,
    python_home: str | None = None,
) -> None:
    """Validate the interpreter that will actually enter the ZPK payload."""

    interpreter = stage_root / PYTHON_RELATIVE_PATH
    if (
        interpreter.is_symlink()
        or not interpreter.is_file()
        or not os.access(interpreter, os.X_OK)
    ):
        raise RuntimeError(
            f"staged Python is missing or not a regular executable: {interpreter}"
        )

    config = _venv_config_values(stage_root / PYTHON_CONFIG_RELATIVE_PATH)
    if config.get("include-system-site-packages", "").lower() != "false":
        raise RuntimeError("staged Python venv must disable system site packages")
    if python_home:
        actual_home = Path(config.get("home", "")).resolve()
        expected_home = Path(python_home).resolve()
        if actual_home != expected_home:
            raise RuntimeError(
                f"staged Python home mismatch: expected {expected_home}, got {actual_home}"
            )

    result = subprocess.run(
        [str(interpreter), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    version_output = result.stdout.strip()
    if result.returncode != 0:
        raise RuntimeError(f"staged Python cannot start: {version_output}")
    if python_version and not version_output.startswith(f"Python {python_version}"):
        raise RuntimeError(
            f"staged Python version mismatch: expected {python_version}, got {version_output}"
        )

    machine = _elf_machine(interpreter)
    if target_arch == "arm64":
        if machine != ELF_MACHINE_AARCH64:
            actual = "non-ELF" if machine is None else f"machine {machine}"
            raise RuntimeError(f"staged Python is not Linux arm64 ELF ({actual})")
        needed = _needed_libraries(interpreter)
        unbundled_python = tuple(
            name for name in needed if re.fullmatch(r"libpython[^/]*", name)
        )
        if unbundled_python:
            raise RuntimeError(
                "staged Python depends on an unbundled libpython: "
                + ", ".join(unbundled_python)
            )
        missing = _missing_dynamic_libraries(interpreter)
        if missing:
            raise RuntimeError(
                "staged Python has unresolved dynamic libraries: " + ", ".join(missing)
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage_root", type=Path, help="staged hermes-agent payload root"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT,
        help="source hermes-agent root (default: repository root)",
    )
    parser.add_argument(
        "--target-arch",
        choices=("arm64",),
        default=None,
        help="require a Linux arm64 staged interpreter (used by device packaging)",
    )
    parser.add_argument(
        "--python-version",
        default=None,
        help="require the staged interpreter to report this Python version prefix",
    )
    parser.add_argument(
        "--python-home",
        default=None,
        help="require pyvenv.cfg home to resolve to this target Python directory",
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

    invalid_permissions = find_invalid_venv_permissions(stage_root)
    if invalid_permissions:
        print("ZPK stage check failed: invalid venv permissions")
        for relative_path, expected, actual in invalid_permissions[:MAX_REPORTED_PATHS]:
            print(f"  - {relative_path.as_posix()}: {actual:04o}, expected {expected:04o}")
        if len(invalid_permissions) > MAX_REPORTED_PATHS:
            print(
                f"  - ... and {len(invalid_permissions) - MAX_REPORTED_PATHS} more"
            )
        return 1

    try:
        check_staged_python(
            stage_root,
            target_arch=args.target_arch,
            python_version=args.python_version,
            python_home=args.python_home,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ZPK stage check failed: {exc}")
        return 1

    protected = ", ".join(f"{tree}/" for tree in PROTECTED_TREES)
    print(f"ZPK stage check ok: runtime trees and interpreter preserved ({protected})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
