#!/usr/bin/env python3
"""HR8 upstream-core overlay gate.

Hermes is a fork of NousResearch/hermes-agent. zettlab-product-dev 总方案 §1.3
treats the upstream core (``run_agent.py``, ``agent/**``, ``tools/**``,
``gateway/run.py``, ``gateway/platforms/base.py``, ``gateway/platforms/api_server.py``)
as a *stable kernel*: changes go upstream first; when an overlay is unavoidable it
must be minimal, rebase-friendly, marked, and never carry Zettlab business state
machines. This script turns that rule into a fail-closed CI check on a PR diff.

Checks (all must pass):

1. **Marker per hunk** — every diff hunk that *adds* lines to a protected file must
   carry, in the added lines or in the ``marker_lookback_lines`` lines above the
   hunk in the new file, a *comment* marker matching ``marker_regex``::

       # zettlab-overlay(U2d): keep pending steers as (id, text); upstream: none

   Deletion-only hunks (converging back to upstream) never need a marker.
2. **Budget** — the PR may add at most ``added_lines_budget`` non-blank lines to
   protected files in total. Exceeding it requires an
   ``overlay-budget-exception: <reason>`` line (>= 20 chars) in the PR body.
3. **Upstream field** — when any protected file is touched the PR body must carry
   an ``upstream-pr: <https://github.com/... | #n | none - <reason>>`` line, and every marker must carry its
   own ``upstream:`` field (enforced by the marker regex).
4. **No business state in core** — added lines matching any
   ``forbidden_added_patterns`` entry fail with the configured reason (heuristic
   defence in depth; the marker and the budget are the primary controls).
5. **LF only** — an added kernel line containing a bare CR fails: Python treats CR as a
   line terminator while git diff counts it as one line.

Upstream sync branches (``skip_head_ref_prefixes``) are skipped: they exist to move
the kernel *towards* upstream.

CI runs this script from the base branch via ``pull_request_target`` (workflow, script and
config all come from base; the PR head is only fetched as diff input and never executed),
so a PR cannot disable the gate it is subject to. The ``overlay-gate`` job is the one to
mark as a required check in branch protection.

Usage::

    python scripts/test-harness/overlay_gate.py --base origin/main --head HEAD \
        --pr-body-file "$RUNNER_TEMP/pr_body.txt" --head-ref "$GITHUB_HEAD_REF"

Exit codes: 0 pass / skip, 1 violations, 2 usage or git error.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

DEFAULT_CONFIG = Path(__file__).with_name("overlay_gate.json")
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    path: str
    new_start: int
    new_count: int
    added: List[str] = field(default_factory=list)
    removed: int = 0

    @property
    def added_nonblank(self) -> int:
        return sum(1 for line in self.added if line.strip())


@dataclass
class Violation:
    kind: str
    path: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.kind}] {self.path}: {self.detail}"


@dataclass
class GateResult:
    skipped: bool = False
    skip_reason: str = ""
    protected_files: List[str] = field(default_factory=list)
    added_total: int = 0
    violations: List[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.skipped or not self.violations


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = (
        "protected",
        "exempt",
        "added_lines_budget",
        "marker_regex",
        "marker_lookback_lines",
        "pr_body_upstream_regex",
        "pr_body_budget_exception_regex",
        "skip_head_ref_prefixes",
        "forbidden_added_patterns",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SystemExit(f"overlay_gate.json missing keys: {', '.join(missing)}")
    return config


def _glob_match(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatchcase(path, pattern):
            return True
        # ``dir/**`` must also match ``dir/file`` and nested paths.
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
    return False


def is_protected(path: str, config: dict) -> bool:
    return _glob_match(path, config["protected"]) and not _glob_match(path, config["exempt"])


def _git(repo: Path, *args: str) -> str:
    # Bytes in, manual decode: text mode would translate a bare CR into a newline and
    # let a CR-only kernel file hide many logical lines inside one diff line.
    proc = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return proc.stdout.decode("utf-8", "replace")


def changed_files(repo: Path, base: str, head: str) -> List[str]:
    merge_base = _git(repo, "merge-base", base, head).strip()
    out = _git(repo, "diff", "--name-only", merge_base, head)
    return [line.strip() for line in out.splitlines() if line.strip()]


def parse_hunks(diff_text: str) -> List[Hunk]:
    """Parse ``git diff -U0`` output into per-file hunks (added lines only)."""
    hunks: List[Hunk] = []
    current_path: Optional[str] = None
    current: Optional[Hunk] = None
    for raw in diff_text.split("\n"):  # LF only: a bare CR is content, not a record separator
        if raw.startswith("diff --git "):
            current = None
            current_path = None
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            current_path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
            continue
        if raw.startswith("--- "):
            continue
        match = _HUNK_RE.match(raw)
        if match:
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            current = Hunk(path=current_path or "", new_start=new_start, new_count=new_count)
            hunks.append(current)
            continue
        if current is None:
            continue
        if raw.startswith("+"):
            current.added.append(raw[1:])
        elif raw.startswith("-"):
            current.removed += 1
    return [hunk for hunk in hunks if hunk.path]


def new_file_lines(repo: Path, head: str, path: str) -> List[str]:
    try:
        return _git(repo, "show", f"{head}:{path}").split("\n")
    except SystemExit:
        return []


def hunk_has_marker(hunk: Hunk, file_lines: Sequence[str], marker: re.Pattern, lookback: int) -> bool:
    if any(marker.search(line) for line in hunk.added):
        return True
    if not file_lines:
        return False
    start = max(0, hunk.new_start - 1 - lookback)
    end = min(len(file_lines), hunk.new_start - 1 + max(hunk.new_count, 1))
    return any(marker.search(line) for line in file_lines[start:end])


def pr_body_has_upstream(pr_body: str, config: dict) -> bool:
    return re.search(config["pr_body_upstream_regex"], pr_body or "") is not None


def pr_body_budget_exception(pr_body: str, config: dict) -> Optional[str]:
    match = re.search(config["pr_body_budget_exception_regex"], pr_body or "")
    return match.group(1).strip() if match else None


def should_skip(head_ref: str, config: dict) -> bool:
    ref = (head_ref or "").strip()
    return any(ref.startswith(prefix) for prefix in config["skip_head_ref_prefixes"]) if ref else False


def run_gate(repo: Path, base: str, head: str, pr_body: str, head_ref: str, config: dict) -> GateResult:
    result = GateResult()
    if should_skip(head_ref, config):
        result.skipped = True
        result.skip_reason = f"head ref {head_ref!r} is an upstream sync branch"
        return result

    files = [path for path in changed_files(repo, base, head) if is_protected(path, config)]
    result.protected_files = files
    if not files:
        return result

    marker = re.compile(config["marker_regex"])
    forbidden = [(re.compile(item["pattern"]), item["reason"]) for item in config["forbidden_added_patterns"]]
    lookback = int(config["marker_lookback_lines"])
    merge_base = _git(repo, "merge-base", base, head).strip()
    # Force a textual diff: a PR-supplied .gitattributes (`agent/** -diff`, textconv, external
    # diff) must not be able to turn kernel hunks into "Binary files differ".
    diff_text = _git(
        repo,
        "-c",
        "core.attributesFile=/dev/null",
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--text",
        "-U0",
        merge_base,
        head,
        "--",
        *files,
    )
    hunks = parse_hunks(diff_text)

    file_cache: dict = {}
    for hunk in hunks:
        if hunk.added_nonblank == 0:
            continue  # deletion-only: converging to upstream needs no marker
        result.added_total += hunk.added_nonblank
        lines = file_cache.setdefault(hunk.path, new_file_lines(repo, head, hunk.path))
        if not hunk_has_marker(hunk, lines, marker, lookback):
            result.violations.append(
                Violation(
                    "marker",
                    hunk.path,
                    f"hunk at new line {hunk.new_start} adds {hunk.added_nonblank} lines without a "
                    "`zettlab-overlay(<batch>): <one line>; upstream: <PR|none>` marker",
                )
            )
        for line in hunk.added:
            if "\r" in line:
                result.violations.append(
                    Violation("bare-cr", hunk.path, "added line contains a bare CR; kernel files must use LF so every logical line is a diff line")
                )
            for logical in line.split("\r"):  # a bare CR still separates logical lines for Python
                for pattern, reason in forbidden:
                    if pattern.search(logical):
                        result.violations.append(Violation("business-state", hunk.path, f"{logical.strip()!r}: {reason}"))

    budget = int(config["added_lines_budget"])
    if result.added_total > budget:
        exception = pr_body_budget_exception(pr_body, config)
        if exception:
            print(f"budget exceeded ({result.added_total} > {budget}) but PR body declares exception: {exception}")
        else:
            result.violations.append(
                Violation(
                    "budget",
                    ",".join(files),
                    f"{result.added_total} added lines in protected files exceed budget {budget}; "
                    "shrink the overlay (move logic to gateway/platforms/zet_agent.py) or add "
                    "`overlay-budget-exception: <reason>` to the PR body",
                )
            )

    if not pr_body_has_upstream(pr_body, config):
        result.violations.append(
            Violation(
                "upstream-pr",
                ",".join(files),
                "PR body must carry `upstream-pr: <url>` or `upstream-pr: none - <reason>` when touching upstream core",
            )
        )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--base", required=True, help="base ref (e.g. origin/main)")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--head-ref", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file and args.pr_body_file.exists() else ""
    result = run_gate(args.repo.resolve(), args.base, args.head, pr_body, args.head_ref, config)

    if result.skipped:
        print(f"overlay gate: SKIP ({result.skip_reason})")
        return 0
    if not result.protected_files:
        print("overlay gate: PASS (no upstream-core files touched)")
        return 0
    print(f"overlay gate: protected files touched: {', '.join(result.protected_files)}")
    print(f"overlay gate: added non-blank lines in protected files: {result.added_total} (budget {config['added_lines_budget']})")
    if result.violations:
        print(f"overlay gate: FAIL ({len(result.violations)} violation(s)) — HR8 / 总方案 §1.3")
        for violation in result.violations:
            print(f"  {violation}")
        return 1
    print("overlay gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
