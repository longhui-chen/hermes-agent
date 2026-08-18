#!/usr/bin/env python3
"""Fail a ZPK build when its staging tree or final archive contains credentials."""

from __future__ import annotations

import argparse
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO


CHUNK_BYTES = 1024 * 1024
OVERLAP_BYTES = 4096
MAX_ARCHIVE_ENTRY_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
CONFIG_SUFFIXES = {
    ".cfg", ".conf", ".env", ".ini", ".json", ".properties",
    ".service", ".sh", ".toml", ".yaml", ".yml",
}
LEGACY_FINGERPRINT = (
    b"e95f2107774881bc3bbb5af665eabf71"
    b"5c1624ba186197829b866d5f426df25b"
)
BINARY_RULES = (
    ("legacy-telemetry-key", re.compile(re.escape(LEGACY_FINGERPRINT))),
    ("langfuse-project-key", re.compile(rb"(?<![A-Za-z0-9_-])(?:pk|sk)-lf-[A-Za-z0-9_-]{20,}")),
    (
        "static-basic-auth",
        re.compile(
            rb"(?i)(?:Authorization\s*[:=]|LANGFUSE_BASIC_AUTH\s*[:=])\s*"
            rb"[\"']?Basic(?:\s+|%20)[A-Za-z0-9+/]{4,}={0,2}"
        ),
    ),
)
SAFE_BINARY_MATCHES = {
    "static-basic-auth": {
        b"Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW",
    },
}
CONFIG_ASSIGNMENT = re.compile(
    rb"(?im)(?:^|[{,])\s*(?:export\s+)?[\"']?"
    rb"(?:HERMES_LANGFUSE_(?:PUBLIC|SECRET)_KEY|LANGFUSE_(?:PUBLIC|SECRET)_KEY|"
    rb"LANGFUSE_BASIC_AUTH|public_key|secret_key|ingestion_key)[\"']?\s*[:=]\s*"
    rb"(?![\"']{2}\s*(?:[,}#]|$)|null\b|[{\[])\S+"
)
MAX_FINDINGS = 50


def _scan_stream(handle: BinaryIO, *, config_like: bool) -> set[str]:
    findings: set[str] = set()
    previous = b""
    while chunk := handle.read(CHUNK_BYTES):
        data = previous + chunk
        for name, pattern in BINARY_RULES:
            safe_matches = SAFE_BINARY_MATCHES.get(name, set())
            if any(match.group(0) not in safe_matches for match in pattern.finditer(data)):
                findings.add(name)
        if config_like and CONFIG_ASSIGNMENT.search(data):
            findings.add("nonempty-monitoring-key-assignment")
        previous = data[-OVERLAP_BYTES:]
    return findings


def scan_tree(root: Path) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        try:
            with path.open("rb") as handle:
                rules = _scan_stream(handle, config_like=path.suffix.lower() in CONFIG_SUFFIXES)
        except OSError:
            rules = {"unreadable-file"}
        findings.extend((relative, rule) for rule in sorted(rules))
        if len(findings) >= MAX_FINDINGS:
            break
    return findings


def scan_archive(archive_path: Path) -> list[tuple[Path, str]]:
    findings: list[tuple[Path, str]] = []
    total_bytes = 0
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            display = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                findings.append((display, "unsafe-archive-path"))
                continue
            if not member.isfile():
                continue
            total_bytes += member.size
            if member.size > MAX_ARCHIVE_ENTRY_BYTES or total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                findings.append((display, "oversized-archive-content"))
                break
            handle = archive.extractfile(member)
            if handle is None:
                findings.append((display, "unreadable-archive-entry"))
                continue
            rules = _scan_stream(handle, config_like=name.suffix.lower() in CONFIG_SUFFIXES)
            findings.extend((display, rule) for rule in sorted(rules))
            if len(findings) >= MAX_FINDINGS:
                break
    return findings


def scan_path(target: Path) -> list[tuple[Path, str]]:
    if target.is_dir():
        return scan_tree(target)
    if target.is_file():
        return scan_archive(target)
    raise ValueError("target is neither a directory nor a regular file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="staged ZPK root or final .zpk archive")
    args = parser.parse_args()
    target = args.target.resolve()
    try:
        findings = scan_path(target)
    except (OSError, tarfile.TarError, ValueError) as exc:
        print(f"ZPK secret scan failed: unable to inspect artifact ({type(exc).__name__})")
        return 2
    if findings:
        print("ZPK secret scan failed: credential material detected")
        for path, rule in findings[:MAX_FINDINGS]:
            print(f"  - {path.as_posix()}: {rule}")
        return 1
    print("ZPK secret scan ok: no monitoring credentials detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
