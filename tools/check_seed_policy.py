#!/usr/bin/env python3
"""Validate config/skill_seed_policy.json against the bundled skills tree.

Fail-closed CI guard for the policy-driven seed allowlist. This PR seeds a
curated subset of the PRISTINE upstream skills/ — it does NOT modify skills/ and
does NOT de-brand, so this guard only checks that the policy is internally
consistent and resolves against the upstream tree. Brand/cleanliness of the
seeded skills' names + files (e.g. a seeded name carrying a vendor brand, or a
dev note like PORT_NOTES.md shipping inside a seeded dir) is intentionally OUT
OF SCOPE here and enforced by a separate follow-up PR. Verifies, without
mutating anything:

  * every 'seed' id resolves to a real bundled skill (no stale allowlist entry);
  * runtime-required skills (force-loaded by hermes) stay seeded under their id;
  * no two bundled skills share a frontmatter name used by the seed set
    (ambiguous seed source → error; any other duplicate → warn);
  * the pre-baked installer fallback manifest stays in sync with the policy.

References from a seeded skill to a non-seeded or non-existent skill are reported
as WARNINGS only: skills/ is upstream-verbatim here, so a dangling reference is
pre-existing upstream and not fixable without editing skills/ (deferred to the
de-brand follow-up).

Exit codes: 0 ok | 1 schema/other | 2 stale seed / duplicate seeded name |
6 runtime-required violation | 7 fallback-manifest drift.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"
POLICY = ROOT / "config" / "skill_seed_policy.json"
# Pre-baked seed list (derived from POLICY) used by the installers' python-free
# fallback: each line is a source dir under skills/ to copy into the profile.
FALLBACK_MANIFEST = ROOT / "config" / "seed_fallback_manifest.txt"

# Narrow load idioms — how hermes actually pulls a skill in by id. Deliberately
# NOT a broad regex: homepage: URLs, markdown attribution links and CLI tool
# names (e.g. `nano-pdf edit`) must NOT count as skill references.
_RELATED_RE = re.compile(r"related_skills:\s*\[([^\]]*)\]")
# YAML block form of related_skills (no inline [...]):
#   related_skills:
#     - excalidraw
#     - foo
# The header line, then one id per `- ` item at a deeper indent. Without this the
# guard would silently miss a stale/dangling ref written in block style and fail
# its "stale id fails CI" contract.
_RELATED_BLOCK_HDR_RE = re.compile(r"^(\s*)related_skills:\s*$")
_RELATED_BLOCK_ITEM_RE = re.compile(r"^(\s*)-\s*['\"]?([a-z0-9-]+)")
_IDIOMS = [
    re.compile(r"skill_view\(\s*name\s*=\s*['\"]([a-z0-9-]+)['\"]"),
    re.compile(r"--skills?\s+([a-z0-9][a-z0-9,-]*)"),
    re.compile(r"`([a-z0-9-]+)`\s+skill\b"),
]


def _extract_block_related(text: str) -> set:
    """Parse a YAML block-list ``related_skills:`` (id per ``- `` line)."""
    refs = set()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        hdr = _RELATED_BLOCK_HDR_RE.match(line)
        if not hdr:
            continue
        base = len(hdr.group(1))
        for nxt in lines[i + 1:]:
            if not nxt.strip():
                continue
            item = _RELATED_BLOCK_ITEM_RE.match(nxt)
            if item and len(item.group(1)) > base:
                refs.add(item.group(2))
            else:
                break  # dedent / non-item line ends the block
        break
    return refs


def read_name(skill_md: Path, fallback: str) -> str:
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_fm = False
    for line in content.split("\n"):
        s = line.strip()
        if s == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm and s.startswith("name:"):
            v = s.split(":", 1)[1].strip().strip("\"'")
            if v:
                return v
    return fallback


def discover(root: Path) -> dict:
    out = {}
    if not root.exists():
        return out
    for md in root.rglob("SKILL.md"):
        out[read_name(md, md.parent.name)] = md.parent
    return out


def find_duplicate_names(root: Path) -> dict:
    """Map frontmatter name -> [dirs] for any name shared by >1 bundled skill.

    discover() keys by frontmatter name and silently overwrites on collision, so
    a duplicate could shadow a seeded skill's source."""
    seen: dict = {}
    if root.exists():
        for md in root.rglob("SKILL.md"):
            seen.setdefault(read_name(md, md.parent.name), []).append(md.parent)
    return {nm: dirs for nm, dirs in seen.items() if len(dirs) > 1}


def extract_refs(text: str) -> set:
    refs = set()
    m = _RELATED_RE.search(text)
    if m:
        for x in m.group(1).split(","):
            x = x.strip().strip("'\"")
            if x:
                refs.add(x)
    else:
        # No inline [...]; try the YAML block-list form.
        refs |= _extract_block_related(text)
    for rx in _IDIOMS:
        for mm in rx.finditer(text):
            for tok in mm.group(1).split(","):
                tok = tok.strip()
                if tok:
                    refs.add(tok)
    return refs


def build_fallback_manifest(seed: set, bundled: dict) -> list:
    """Derive the python-free installer fallback list from the policy.

    Each entry is a source dir under skills/ (relative to the install root) the
    fallback copies into the profile. Sorted for a stable, diffable file."""
    lines = []
    for sid in seed:
        if sid in bundled:
            lines.append("skills/" + bundled[sid].relative_to(SKILLS).as_posix())
    return sorted(lines)


def main() -> int:
    if not POLICY.exists():
        print(f"FAIL: policy not found at {POLICY}")
        return 1
    try:
        pol = json.loads(POLICY.read_text(encoding="utf-8"))
    except ValueError as e:
        # Truncated/corrupt JSON: fail with a clean message (rc 1) instead of an
        # uncaught JSONDecodeError traceback. Mirrors _read_seed_policy()'s catch.
        print(f"FAIL: policy is not valid JSON: {e}")
        return 1

    # Structural validation: a parseable-but-malformed policy ({} or a string
    # 'seed') would silently seed 0 / garbage at runtime instead of failing
    # closed. Reject it here so a bad policy can't be committed.
    if pol.get("mode", "allowlist") != "allowlist":
        print("FAIL: policy 'mode' must be 'allowlist'")
        return 1
    seed_raw = pol.get("seed")
    if not isinstance(seed_raw, list) or not all(isinstance(x, str) for x in seed_raw):
        print("FAIL: policy 'seed' must be a list of strings")
        return 1
    runtime_required = pol.get("runtime_required", [])
    if not isinstance(runtime_required, list) or not all(isinstance(x, str) for x in runtime_required):
        print("FAIL: policy 'runtime_required' must be a list of strings")
        return 1

    seed = set(seed_raw)
    bundled = discover(SKILLS)

    rc = 0
    errs = []
    warns = []

    # 1. stale seed ids. Matching is by SKILL.md frontmatter name; if a dir by
    #    this name exists, the id is almost certainly a frontmatter rename
    #    upstream rather than a true removal — say so.
    dir_names = {d.name for d in bundled.values()}
    for sid in sorted(seed):
        if sid not in bundled:
            if sid in dir_names:
                hint = (
                    " (a skill directory by this name exists — its SKILL.md "
                    "frontmatter `name` likely changed upstream; update the seed id)"
                )
            else:
                hint = " (no bundled skill or directory by this name — removed/renamed upstream)"
            errs.append(f"[stale-seed] '{sid}' in seed but not a bundled skill{hint}")
            rc = max(rc, 2)

    # 2. runtime-required: must stay seeded under their own id
    for r in runtime_required:
        if r not in seed:
            errs.append(f"[runtime-required] '{r}' is force-loaded by hermes but not seeded")
            rc = max(rc, 6)

    # 3. duplicate frontmatter names among bundled skills. discover() silently
    #    overwrites on collision, so a duplicate that shares a seeded name makes
    #    the seed source ambiguous (error); any other duplicate is a latent
    #    hazard (warn).
    for nm, dirs in sorted(find_duplicate_names(SKILLS).items()):
        locs = ", ".join(sorted(d.relative_to(SKILLS).as_posix() for d in dirs))
        if nm in seed:
            errs.append(
                f"[dup-name] seeded id '{nm}' is shared by {len(dirs)} bundled "
                f"skills ({locs}) — seed source is ambiguous"
            )
            rc = max(rc, 2)
        else:
            warns.append(f"[dup-name] '{nm}' shared by {len(dirs)} bundled skills: {locs}")

    # 4. reference guard (WARN-only here). A seeded skill's related_skills /
    #    load idioms may point at a skill that is not seeded, or at an id that
    #    exists nowhere in skills/. Since this PR keeps skills/ upstream-verbatim
    #    (no renames), any such reference is PRE-EXISTING upstream and not fixable
    #    without editing skills/ — so both are warnings, not errors. (The de-brand
    #    follow-up that renames skills re-introduces the dangling-ref ERROR to
    #    catch rename-stranded references.)
    for sid in sorted(seed):
        md = (bundled[sid] / "SKILL.md") if sid in bundled else None
        if not (md and md.exists()):
            continue
        for r in sorted(extract_refs(md.read_text(encoding="utf-8", errors="replace"))):
            if r == sid or r in seed:
                continue
            if r in bundled or r in dir_names:
                warns.append(f"[ref-unseeded] '{sid}' -> '{r}' (not seeded)")
            else:
                warns.append(
                    f"[ref-dangling-upstream] '{sid}' -> '{r}' (not a bundled skill; "
                    f"pre-existing upstream — left as-is, de-brand follow-up may fix)"
                )

    # 5. fallback manifest must stay in sync with the policy (single source of
    #    truth = the policy; the manifest is a derived projection the installers
    #    consume without python).
    derived_manifest = build_fallback_manifest(seed, bundled)
    if not FALLBACK_MANIFEST.exists():
        errs.append("[fallback-manifest] config/seed_fallback_manifest.txt missing; run --emit-fallback-manifest")
        rc = max(rc, 7)
    else:
        committed = [l for l in FALLBACK_MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
        if committed != derived_manifest:
            errs.append("[fallback-manifest] config/seed_fallback_manifest.txt out of sync with policy; regenerate via --emit-fallback-manifest")
            rc = max(rc, 7)

    not_seeded = sorted(set(bundled) - seed)

    print("=== seed policy check ===")
    print(
        f"  bundled={len(bundled)} seed={len(seed)} "
        f"not_seeded={len(not_seeded)} runtime_required={len(runtime_required)}"
    )
    print(f"  errors={len(errs)} warnings={len(warns)}")
    for e in errs:
        print(f"  ERROR {e}")
    for w in warns:
        print(f"  warn  {w}")
    print("ok" if rc == 0 else f"FAIL rc={rc}")
    return rc


def _emit_fallback_manifest() -> int:
    pol = json.loads(POLICY.read_text(encoding="utf-8"))
    seed = set(pol.get("seed", []))
    bundled = discover(SKILLS)
    print("\n".join(build_fallback_manifest(seed, bundled)))
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Validate the seed policy (allowlist).")
    ap.add_argument(
        "--emit-fallback-manifest",
        action="store_true",
        help="Print the python-free installer fallback list derived from the policy.",
    )
    args = ap.parse_args()
    if args.emit_fallback_manifest:
        sys.exit(_emit_fallback_manifest())
    sys.exit(main())
