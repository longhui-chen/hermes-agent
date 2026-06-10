#!/usr/bin/env python3
"""Guard: keep the de-branded sibling ``zettlab-memo-setup`` honest vs upstream
``hermes-agent``.

``zettlab-memo-setup`` is a de-branded copy of the upstream ``hermes-agent``
runtime skill (it is seeded in place of it so the assistant never surfaces the
Hermes / Nous Research identity). Because it is a copy, it can silently drift
when upstream updates ``hermes-agent``. This guard enforces three invariants
WITHOUT mutating anything:

  1. Brand-clean: the sibling tree carries zero Hermes / Nous Research / Teknium /
     nousresearch tokens — a future edit must not re-introduce the identity leak
     the sibling exists to remove. (Functional refs the de-brand deliberately
     keeps — the ``nous`` provider option, ``Nous Portal``, the ``hermes`` CLI —
     are not identity claims and are intentionally NOT matched here.)

  2. Structural parity: the sibling mirrors ``hermes-agent``'s structure — the
     same set of files and the same Markdown section headers (after the product
     name is neutralised). Catches a sibling that drops/gains a section or file.

  3. Upstream provenance: the CURRENT ``hermes-agent`` content hash must equal the
     baseline recorded in ``config/sibling_upstream_baseline.json`` — the upstream
     state the sibling was last de-branded from. This catches IN-SECTION upstream
     changes (a new CLI flag, command, MCP/webhook/security note) that add no
     header and no file — exactly the drift a structural check alone would miss.

Maintenance workflow: when an upstream sync changes ``hermes-agent``, the
provenance check fails. Re-derive the sibling (mirror the change, keep it
brand-clean), then record the new upstream hash::

    python tools/check_sibling_debrand.py --update-baseline

``--update-baseline`` refuses to write unless the brand-clean + structural checks
already pass, so the hash can't be rubber-stamped green over a brand-dirty or
structurally divergent sibling. It still cannot prove the prose was *faithfully*
re-synced (part of the de-brand — the opening identity paragraph — is a hand
rewrite); the contract is "a human re-examined the sibling whenever upstream
changed", enforced by the failing provenance check.

Exit codes: 0 ok | 1 brand leak | 2 structural drift | 3 upstream content changed.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "skills" / "autonomous-ai-agents" / "hermes-agent"
SIBLING = ROOT / "skills" / "autonomous-ai-agents" / "zettlab-memo-setup"
BASELINE = ROOT / "config" / "sibling_upstream_baseline.json"

# Identity tokens the sibling must never carry. Deliberately NOT a broad "nous"
# match: the kept functional provider refs (`nous`, `Nous Portal`) are config
# options, not self-identity claims.
BRAND_RE = re.compile(r"Hermes|Nous Research|Teknium|nousresearch|NousResearch")

# Neutralise the product name so a header that differs only by branding compares
# equal across the two trees. Multi-word forms first.
_NEUTRAL = [("Hermes Agent", "<RT>"), ("Zettlab Memo", "<RT>"), ("Hermes", "<RT>")]
_HEADER_RE = re.compile(r"#{1,6}\s+")

# Build/editor junk that must not affect the file set, headers, or content hash
# (a stray .DS_Store / __pycache__ in a working tree would otherwise read as
# drift). Mirrors tools/skills_sync.py's ignore set.
_IGNORE_DIRS = {"__pycache__", ".DS_Store"}
_IGNORE_SUFFIXES = {".pyc", ".pyo"}


def _neutral(s: str) -> str:
    for a, b in _NEUTRAL:
        s = s.replace(a, b)
    return s


def _headers(text: str) -> list:
    """Brand-neutralised Markdown headers, skipping fenced code blocks (so shell
    comments like ``# Install`` inside ```bash``` are not mistaken for headers)."""
    out = []
    in_fence = False
    fence = ""
    for line in text.splitlines():
        s = line.lstrip()
        if not in_fence and (s.startswith("```") or s.startswith("~~~")):
            in_fence, fence = True, s[:3]
            continue
        if in_fence:
            if s.startswith(fence):
                in_fence, fence = False, ""
            continue
        if _HEADER_RE.match(line):
            out.append(_neutral(line.strip()))
    return out


def _rel_files(base: Path) -> dict:
    out = {}
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(base)
        if any(part in _IGNORE_DIRS for part in rel.parts) or p.suffix in _IGNORE_SUFFIXES:
            continue
        out[rel.as_posix()] = p
    return out


def _content_hash(base: Path) -> str:
    """Order-independent sha256 over every (relative path, bytes) in the tree,
    excluding build/editor junk (see _rel_files)."""
    h = hashlib.sha256()
    for rel, p in sorted(_rel_files(base).items()):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _check_brand_and_structure() -> int:
    """Checks 1 + 2 (brand-clean, structural parity). rc: 0 | 1 (leak) | 2 (drift)
    — max of both. Prints failures."""
    sib_files = _rel_files(SIBLING)
    up_files = _rel_files(UPSTREAM)
    rc = 0

    # 1. brand-clean
    leaks = []
    for rel, p in sorted(sib_files.items()):
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if BRAND_RE.search(line):
                    leaks.append(f"  {rel}:{i}: {line.strip()[:100]}")
        except OSError:
            continue
    if leaks:
        print("FAIL [brand-leak] sibling zettlab-memo-setup still contains brand tokens:")
        print("\n".join(leaks))
        rc = max(rc, 1)

    # 2. structural parity — file set
    only_up = sorted(set(up_files) - set(sib_files))
    only_sib = sorted(set(sib_files) - set(up_files))
    if only_up:
        print(f"FAIL [drift] hermes-agent has files the sibling lacks (mirror them): {only_up}")
        rc = max(rc, 2)
    if only_sib:
        print(f"FAIL [drift] sibling has files not in hermes-agent (remove or justify): {only_sib}")
        rc = max(rc, 2)

    # 2b. structural parity — section headers per shared .md file
    for rel in sorted(set(up_files) & set(sib_files)):
        if not rel.endswith(".md"):
            continue
        up_h = _headers(up_files[rel].read_text(encoding="utf-8", errors="replace"))
        sib_h = _headers(sib_files[rel].read_text(encoding="utf-8", errors="replace"))
        missing = [h for h in up_h if h not in sib_h]
        extra = [h for h in sib_h if h not in up_h]
        if missing:
            print(f"FAIL [drift] {rel}: sections in hermes-agent but NOT in sibling (sync them):")
            for h in missing:
                print(f"    {h}")
            rc = max(rc, 2)
        if extra:
            print(f"FAIL [drift] {rel}: sections in sibling but NOT in hermes-agent (remove or justify):")
            for h in extra:
                print(f"    {h}")
            rc = max(rc, 2)
    return rc


def _check_provenance() -> int:
    """Check 3 (upstream content unchanged since the sibling was synced). rc: 0 | 3."""
    current = _content_hash(UPSTREAM)
    recorded = None
    if BASELINE.exists():
        try:
            recorded = json.loads(BASELINE.read_text(encoding="utf-8")).get("upstream_sha256")
        except (OSError, ValueError):
            recorded = None
    if recorded is None:
        print(f"FAIL [provenance] baseline missing/unreadable at {BASELINE} — run --update-baseline")
        return 3
    if recorded != current:
        print("FAIL [upstream-drift] hermes-agent content changed since the sibling was last synced —")
        print("  its CLI / MCP / webhook / security / config text may have changed inside existing")
        print("  sections without adding a header, so the de-branded sibling must be re-derived.")
        print(f"    baseline: {recorded}")
        print(f"    current:  {current}")
        print("  Fix: mirror the upstream change into zettlab-memo-setup (keep it brand-clean), then")
        print("       run `python tools/check_sibling_debrand.py --update-baseline`.")
        return 3
    return 0


def main() -> int:
    if not SIBLING.exists():
        print(f"FAIL: sibling skill not found at {SIBLING}")
        return 2
    if not UPSTREAM.exists():
        print(f"FAIL: upstream hermes-agent skill not found at {UPSTREAM}")
        return 2

    rc = _check_brand_and_structure()
    rc = max(rc, _check_provenance())

    if rc == 0:
        n = len(_rel_files(SIBLING))
        print(
            f"ok: sibling zettlab-memo-setup is brand-clean, structurally in sync, and matches the "
            f"recorded hermes-agent baseline ({n} files)."
        )
    return rc


def _update_baseline() -> int:
    if not SIBLING.exists():
        print(f"FAIL: sibling skill not found at {SIBLING}")
        return 2
    if not UPSTREAM.exists():
        print(f"FAIL: upstream hermes-agent skill not found at {UPSTREAM}")
        return 2
    # Refuse to rubber-stamp the hash green over a brand-dirty or structurally
    # divergent sibling: the brand-clean + structural checks must pass first.
    rc = _check_brand_and_structure()
    if rc != 0:
        print("refusing to update baseline: fix the brand/structure issues above first.")
        return rc
    data = {
        "_note": (
            "sha256 of the upstream hermes-agent skill that zettlab-memo-setup was "
            "de-branded from. When an upstream sync changes hermes-agent, re-derive "
            "the sibling (mirror the change, keep it brand-clean) then run: "
            "python tools/check_sibling_debrand.py --update-baseline"
        ),
        "upstream": "skills/autonomous-ai-agents/hermes-agent",
        "sibling": "skills/autonomous-ai-agents/zettlab-memo-setup",
        "upstream_sha256": _content_hash(UPSTREAM),
    }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        shown = BASELINE.relative_to(ROOT)
    except ValueError:
        shown = BASELINE
    print(f"updated {shown} -> {data['upstream_sha256']}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Guard the de-branded zettlab-memo-setup sibling vs upstream hermes-agent.")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="Record the current hermes-agent content hash as the synced-from baseline "
        "(run after re-deriving the sibling; refuses if brand/structure checks fail).",
    )
    args = ap.parse_args()
    sys.exit(_update_baseline() if args.update_baseline else main())
