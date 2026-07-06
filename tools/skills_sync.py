#!/usr/bin/env python3
"""
Skills Sync -- Manifest-based seeding and updating of bundled skills.

Copies bundled skills from the repo's skills/ directory into ~/.hermes/skills/
and uses a manifest to track which skills have been synced and their origin hash.

Manifest format (v2): each line is "skill_name:origin_hash" where origin_hash
is the MD5 of the bundled skill at the time it was last synced to the user dir.
Old v1 manifests (plain names without hashes) are auto-migrated.

Update logic:
  - NEW skills (not in manifest): copied to user dir, origin hash recorded.
  - EXISTING skills (in manifest, present in user dir):
      * If user copy matches origin hash: user hasn't modified it → safe to
        update from bundled if bundled changed. New origin hash recorded.
      * If user copy differs from origin hash: user customized it → SKIP.
  - DELETED by user (in manifest, absent from user dir): respected, not re-added.
  - REMOVED from bundled (in manifest, gone from repo): cleaned from manifest.

The manifest lives at ~/.hermes/skills/.bundled_manifest.
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from hermes_constants import get_bundled_skills_dir, get_hermes_home, get_optional_skills_dir
from agent.skill_utils import is_excluded_skill_path
from typing import Dict, List, Optional, Set, Tuple
from utils import atomic_replace

logger = logging.getLogger(__name__)


HERMES_HOME = get_hermes_home()
SKILLS_DIR = HERMES_HOME / "skills"
MANIFEST_FILE = SKILLS_DIR / ".bundled_manifest"

# Marker file written by `hermes profile create --no-skills` (named profiles)
# and by the installer's `--no-skills` flag (the default ~/.hermes profile).
# When present in HERMES_HOME, sync_skills() is a no-op so neither the
# installer, `hermes update`, nor a direct sync re-injects bundled skills.
# Delete the file to opt back in. Mirrors
# hermes_cli.profiles.NO_BUNDLED_SKILLS_MARKER (kept as a literal here to
# avoid importing the CLI layer into this low-level sync module).
NO_BUNDLED_SKILLS_MARKER = ".no-bundled-skills"

# Written into a profile's skills/ once it has been seeded under the seed
# policy. Lets sync tell a policy-managed profile (keep filtering) from one
# created before the policy existed (left untouched — no filtering, no cleanup).
SEED_POLICY_MARKER = ".seed_policy"

# Never copy build/editor junk into a user profile (defense in depth: the seed
# guard in tools/check_seed_policy.py keeps tracked junk out of the bundle, this
# stops untracked local junk — __pycache__, *.pyc, .DS_Store — from a working
# tree being copied in). _dir_hash() MUST skip the SAME patterns, or change
# detection compares a hash-of-source-incl-junk against a hash-of-dest-excl-junk
# and wrongly flags the skill "user-modified" (then never updates it).
_IGNORE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo", ".DS_Store")
_COPY_IGNORE = shutil.ignore_patterns(*_IGNORE_PATTERNS)


def _is_copy_ignored(rel: Path) -> bool:
    """True if a relative path matches a copytree-ignored pattern (any path
    component), so _dir_hash() and copytree() agree on the file set."""
    import fnmatch

    return any(
        fnmatch.fnmatch(part, pat) for part in rel.parts for pat in _IGNORE_PATTERNS
    )


def _get_bundled_dir() -> Path:
    """Locate the bundled skills/ directory.

    Checks HERMES_BUNDLED_SKILLS env var first (set by Nix wrapper),
    then a wheel-installed data dir, then falls back to the relative
    path from this source file.
    """
    return get_bundled_skills_dir(Path(__file__).parent.parent / "skills")


def _get_optional_dir() -> Path:
    """Locate the official optional-skills/ directory."""
    return get_optional_skills_dir(Path(__file__).parent.parent / "optional-skills")


def _build_external_skill_index() -> Set[str]:
    """Index every skill available in external_dirs by name and frontmatter name.

    Returns a set of skill names that are already provided by external dirs.
    Used to prevent sync_skills from shadowing externally-delegated skills.
    """
    try:
        from agent.skill_utils import get_external_skills_dirs, _external_dirs_cache_clear
    except ImportError:
        return set()

    # Clear the external dirs cache so a config edit (or a test patch) is seen.
    _external_dirs_cache_clear()

    external_names: Set[str] = set()
    for ext_dir in get_external_skills_dirs():
        for skill_md in ext_dir.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            skill_dir = skill_md.parent
            # Index by directory name (how _find_skill resolves skills)
            external_names.add(skill_dir.name)
            # Also index by frontmatter name (alternate identifier)
            frontmatter_name = _read_skill_name(skill_md, "")
            if frontmatter_name:
                external_names.add(frontmatter_name)
    return external_names


def _read_manifest() -> Dict[str, str]:
    """
    Read the manifest as a dict of {skill_name: origin_hash}.

    Handles both v1 (plain names) and v2 (name:hash) formats.
    v1 entries get an empty hash string which triggers migration on next sync.
    """
    if not MANIFEST_FILE.exists():
        return {}
    try:
        result = {}
        for line in MANIFEST_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                # v2 format: name:hash
                name, _, hash_val = line.partition(":")
                result[name.strip()] = hash_val.strip()
            else:
                # v1 format: plain name — empty hash triggers migration
                result[line] = ""
        return result
    except (OSError, IOError):
        return {}


def _read_suppressed_names() -> set:
    """Built-in skills the curator pruned — must NOT be re-seeded on sync.

    Delegates to ``tools.skill_usage`` (single source of truth) and falls back
    to reading ``~/.hermes/skills/.curator_suppressed`` directly if that import
    is unavailable in a packaged/update context.
    """
    try:
        from tools.skill_usage import read_suppressed_names

        return read_suppressed_names()
    except Exception:
        path = SKILLS_DIR / ".curator_suppressed"
        if not path.exists():
            return set()
        names = set()
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
        except OSError:
            pass
        return names


def _write_manifest(entries: Dict[str, str]):
    """Write the manifest file atomically in v2 format (name:hash).

    Uses a temp file + os.replace() to avoid corruption if the process
    crashes or is interrupted mid-write.
    """
    import tempfile

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(f"{name}:{hash_val}" for name, hash_val in sorted(entries.items())) + "\n"

    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(MANIFEST_FILE.parent),
            prefix=".bundled_manifest_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, MANIFEST_FILE)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.debug("Failed to write skills manifest %s: %s", MANIFEST_FILE, e, exc_info=True)


def _read_skill_name(skill_md: Path, fallback: str) -> str:
    """Read the name field from SKILL.md YAML frontmatter, falling back to *fallback*."""
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return fallback
    in_frontmatter = False
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            if in_frontmatter:
                break
            in_frontmatter = True
            continue
        if in_frontmatter and stripped.startswith("name:"):
            value = stripped.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return fallback


def _discover_bundled_skills(bundled_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all SKILL.md files in the bundled directory.
    Returns list of (skill_name, skill_directory_path) tuples.
    """
    skills = []
    if not bundled_dir.exists():
        return skills

    for skill_md in bundled_dir.rglob("SKILL.md"):
        # Check exclusion RELATIVE to bundled_dir. The install prefix itself may
        # legitimately be a virtualenv: `pip install` into ./venv lands the wheel's
        # data files at venv/skills/..., and is_excluded_skill_path matches ANY
        # path component (venv/.venv/site-packages/…). An absolute check would
        # then treat the install path's own venv component as a dependency dir and
        # drop EVERY bundled skill (seed 0) — the two most common venv names
        # (`venv`, `.venv`) trigger it. Only components BELOW bundled_dir (a stray
        # __pycache__/node_modules inside a skill) should exclude.
        if is_excluded_skill_path(skill_md.relative_to(bundled_dir)):
            continue
        skill_dir = skill_md.parent
        skill_name = _read_skill_name(skill_md, skill_dir.name)
        skills.append((skill_name, skill_dir))

    return skills


def _compute_relative_dest(skill_dir: Path, bundled_dir: Path) -> Path:
    """
    Compute the destination path in SKILLS_DIR preserving the category structure.
    e.g., bundled/skills/mlops/axolotl -> ~/.hermes/skills/mlops/axolotl
    """
    rel = skill_dir.relative_to(bundled_dir)
    return SKILLS_DIR / rel


# ── Policy-driven seed allowlist ──────────────────────────────────────────
# skills/ is kept upstream-verbatim (this PR does not rename/de-brand); the
# policy below is a pure allowlist deciding which bundled skills are copied into
# a NEW profile, matched against each skill's frontmatter `name`. The filter is
# applied BEFORE hashing, so non-seeded skills cost only a cheap set lookup.

def _read_seed_policy() -> Optional[dict]:
    """Load config/skill_seed_policy.json.

    Returns the policy dict augmented with a 'seed_set' (set of allowlisted skill
    names), or None when no policy is present or it cannot be parsed — in which
    case sync seeds every bundled skill (upstream behaviour, fully backward
    compatible).
    """
    # Look for the policy via the explicit env override first, then co-located
    # with the bundled skills (config/ next to skills/). It is intentionally NOT
    # resolved relative to this source file: the policy is part of the *bundle*,
    # so a test (or any caller) that points _get_bundled_dir() elsewhere gets
    # the upstream "seed everything" behaviour instead of silently inheriting
    # the repo's policy.
    candidates = []
    env = os.environ.get("HERMES_SEED_POLICY")
    if env:
        candidates.append(Path(env))
    candidates.append(_get_bundled_dir().parent / "config" / "skill_seed_policy.json")
    for p in candidates:
        try:
            if p and p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if not _policy_is_well_formed(data):
                    continue  # parseable but structurally invalid -> treat as corrupt
                data["seed_set"] = set(data["seed"])
                return data
        except (OSError, ValueError):
            continue  # malformed/unreadable candidate -> try the next one
    return None


def _policy_is_well_formed(data) -> bool:
    """Structural validation beyond JSON parse: a policy whose 'seed' isn't a
    list of strings (e.g. {} or {"seed": "keep"}) must be rejected so a fresh
    profile fails closed (policy_error) instead of silently seeding 0/garbage."""
    if not isinstance(data, dict):
        return False
    if data.get("mode", "allowlist") != "allowlist":
        return False
    seed = data.get("seed")
    if not isinstance(seed, list) or not all(isinstance(x, str) for x in seed):
        return False
    rr = data.get("runtime_required", [])
    if not isinstance(rr, list) or not all(isinstance(x, str) for x in rr):
        return False
    return True


def _seed_policy_file_present() -> bool:
    """True if a seed-policy file EXISTS at a known location (parseable or not).

    Lets sync_skills distinguish "no policy at all" (seed everything — upstream,
    backward-compatible) from "a policy is shipped but corrupt/unreadable" (fail
    closed — never seed the full un-curated set). Mirrors _read_seed_policy()'s
    candidate order."""
    candidates = []
    env = os.environ.get("HERMES_SEED_POLICY")
    if env:
        candidates.append(Path(env))
    candidates.append(_get_bundled_dir().parent / "config" / "skill_seed_policy.json")
    for p in candidates:
        try:
            if p and p.exists():
                return True
        except OSError:
            continue
    return False


def _resolve_seed_target(
    skill_name: str, skill_src: Path, bundled_dir: Path, policy: Optional[dict]
):
    """Map a discovered bundled skill to its seed target.

    Returns ``(skill_name, skill_src, dest_dir)`` or ``None`` to skip the skill
    (not allowlisted → never hashed or copied). skills/ is upstream-verbatim, so a
    seeded skill is always copied from the bundled dir as-is. With ``policy=None``
    every skill seeds verbatim.
    """
    if policy is None or skill_name in policy.get("seed_set", set()):
        return (skill_name, skill_src, _compute_relative_dest(skill_src, bundled_dir))
    return None


def _dir_hash(directory: Path) -> str:
    """Compute a hash of all file contents in a directory for change detection.

    Skips the same junk patterns copytree ignores (_IGNORE_PATTERNS) so the
    baseline hash matches the file set actually copied — otherwise a skill whose
    source carries __pycache__/*.pyc/.DS_Store hashes differently than its seeded
    copy and is wrongly flagged "user-modified" (and frozen from updates).
    """
    hasher = hashlib.md5()
    try:
        for fpath in sorted(directory.rglob("*")):
            if fpath.is_file():
                rel = fpath.relative_to(directory)
                if _is_copy_ignored(rel):
                    continue
                hasher.update(str(rel).encode("utf-8"))
                hasher.update(fpath.read_bytes())
    except (OSError, IOError):
        pass
    return hasher.hexdigest()


def _safe_rel_install_path(path: Path, base: Path) -> str:
    """Return a normalized relative POSIX path, rejecting traversal/absolute paths."""
    rel = path.relative_to(base)
    posix = rel.as_posix()
    pure = PurePosixPath(posix)
    parts = [part for part in pure.parts if part not in {"", "."}]
    if pure.is_absolute() or not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe optional skill path: {posix}")
    return "/".join(parts)


def _skill_file_list(skill_dir: Path) -> List[str]:
    """List files inside a skill directory in lock-file format."""
    files: List[str] = []
    for fpath in sorted(skill_dir.rglob("*")):
        if fpath.is_file():
            files.append(fpath.relative_to(skill_dir).as_posix())
    return files


def _content_hash(directory: Path) -> str:
    """Return the same hash style the skills hub lock uses, falling back locally."""
    try:
        from tools.skills_guard import content_hash

        return content_hash(directory)
    except Exception:
        # Hashing is provenance metadata only; keep sync resilient if guard
        # dependencies are unavailable in a packaged/update context.
        return _dir_hash(directory)


def _optional_skill_index() -> Dict[str, Tuple[str, str, Path]]:
    """Return official optional skills keyed by folder name and frontmatter name.

    Values are ``(folder_name, install_path, source_dir)``. Multiple keys may
    point to the same skill so callers can accept either the folder slug used
    by the hub lock or the user-facing frontmatter name.
    """
    optional_dir = _get_optional_dir()
    index: Dict[str, Tuple[str, str, Path]] = {}
    if not optional_dir.exists():
        return index
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        # Relative to optional_dir: the install prefix may be a venv (see
        # _discover_bundled_skills) — an absolute check would drop every optional
        # skill when installed under venv/.venv/site-packages.
        if is_excluded_skill_path(skill_md.relative_to(optional_dir)):
            continue
        src = skill_md.parent
        try:
            install_path = _safe_rel_install_path(src, optional_dir)
        except ValueError:
            continue
        folder_name = src.name
        frontmatter_name = _read_skill_name(skill_md, folder_name)
        value = (folder_name, install_path, src)
        index[folder_name] = value
        index[frontmatter_name] = value
    return index


def _move_to_restore_backup(path: Path, backup_root: Path) -> str:
    """Move an existing skill directory into a restore backup, preserving rel path."""
    rel = path.relative_to(SKILLS_DIR)
    target = backup_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        suffix = 1
        while target.with_name(f"{target.name}-{suffix}").exists():
            suffix += 1
        target = target.with_name(f"{target.name}-{suffix}")
    shutil.move(str(path), str(target))
    return rel.as_posix()


def restore_official_optional_skill(name: str, *, restore: bool = False) -> dict:
    """Restore one or all official optional skills from repo source.

    ``restore=False`` only performs exact-match provenance backfill. ``restore=True``
    repairs already-mutated/reorganized skills by backing up matching active
    copies and copying the official optional source into its canonical path.
    """
    index = _optional_skill_index()
    if not index:
        return {"ok": False, "message": "No official optional skills directory found.", "restored": [], "backfilled": [], "backed_up": []}

    targets = sorted(set(index.values()), key=lambda item: item[1]) if name in {"all", "*"} else []
    if not targets:
        target = index.get(name)
        if target is None:
            return {"ok": False, "message": f"Official optional skill not found: {name}", "restored": [], "backfilled": [], "backed_up": []}
        targets = [target]

    restored: List[str] = []
    backed_up: List[str] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = SKILLS_DIR / ".restore-backups" / f"official-optional-{timestamp}"

    for folder_name, install_path, src in targets:
        dest = SKILLS_DIR / Path(*install_path.split("/"))
        src_hash = _dir_hash(src)
        canonical_ok = dest.exists() and _dir_hash(dest) == src_hash

        # Find already-active copies of this official skill by frontmatter name
        # or folder slug, even if curator moved it into another category.
        src_frontmatter = _read_skill_name(src / "SKILL.md", folder_name)
        matches: List[Path] = []
        if SKILLS_DIR.exists():
            for skill_md in sorted(SKILLS_DIR.rglob("SKILL.md")):
                if is_excluded_skill_path(skill_md):
                    continue
                candidate = skill_md.parent
                try:
                    candidate.relative_to(SKILLS_DIR)
                except ValueError:
                    continue
                candidate_name = _read_skill_name(skill_md, candidate.name)
                if candidate == dest:
                    continue
                if candidate.name == folder_name or candidate_name in {folder_name, src_frontmatter}:
                    matches.append(candidate)

        if restore:
            for match in matches:
                if match.exists():
                    backed_up.append(_move_to_restore_backup(match, backup_root))
            if dest.exists() and not canonical_ok:
                backed_up.append(_move_to_restore_backup(dest, backup_root))
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, ignore=_COPY_IGNORE)
                restored.append(folder_name)
        elif not canonical_ok:
            continue

    backfilled = _backfill_optional_provenance(quiet=True)
    return {
        "ok": True,
        "message": "Official optional skill repair complete.",
        "restored": restored,
        "backfilled": backfilled,
        "backed_up": backed_up,
        "backup_dir": str(backup_root) if backed_up else "",
    }


def _backfill_optional_provenance(quiet: bool = False) -> List[str]:
    """Mark already-present official optional skills as hub-installed.

    This covers the migration case where a skill used to be bundled (or was
    manually copied into the active skills tree) and later lives under
    optional-skills/. If the active copy is byte-identical to the official
    optional source, record official hub provenance without copying or
    reinstalling anything. Modified/local skills are left alone.
    """
    optional_dir = _get_optional_dir()
    if not optional_dir.exists():
        return []

    lock_path = SKILLS_DIR / ".hub" / "lock.json"
    try:
        data = json.loads(lock_path.read_text()) if lock_path.exists() else {"version": 1, "installed": {}}
    except (json.JSONDecodeError, OSError):
        data = {"version": 1, "installed": {}}
    installed = data.setdefault("installed", {})
    existing_paths = {
        entry.get("install_path")
        for entry in installed.values()
        if isinstance(entry, dict)
    }

    backfilled: List[str] = []
    changed = False
    for skill_md in sorted(optional_dir.rglob("SKILL.md")):
        # Relative to optional_dir (install prefix may be a venv; see
        # _discover_bundled_skills).
        if is_excluded_skill_path(skill_md.relative_to(optional_dir)):
            continue
        src = skill_md.parent
        try:
            install_path = _safe_rel_install_path(src, optional_dir)
        except ValueError as e:
            logger.debug("Skipping optional skill with unsafe path %s: %s", src, e)
            continue
        dest = SKILLS_DIR / Path(*install_path.split("/"))
        if not dest.exists() or not dest.is_dir():
            continue
        if _dir_hash(dest) != _dir_hash(src):
            continue

        lock_name = src.name
        if lock_name in installed or install_path in existing_paths:
            continue

        timestamp = datetime.now(timezone.utc).isoformat()
        installed[lock_name] = {
            "source": "official",
            "identifier": f"official/{install_path}",
            "trust_level": "builtin",
            "scan_verdict": "backfilled",
            "content_hash": _content_hash(dest),
            "install_path": install_path,
            "files": _skill_file_list(dest),
            "metadata": {"backfilled_from": "optional-skills"},
            "installed_at": timestamp,
            "updated_at": timestamp,
        }
        existing_paths.add(install_path)
        backfilled.append(lock_name)
        changed = True
        if not quiet:
            print(f"  = {lock_name} (official optional provenance backfilled)")

    if changed:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write can't silently wipe all provenance
        # via the JSONDecodeError fallback above (which resets `installed` to
        # an empty dict).
        import tempfile

        payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(lock_path.parent),
            prefix=".lock_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, lock_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    return backfilled


def sync_skills(quiet: bool = False) -> dict:
    """
    Sync bundled skills into ~/.hermes/skills/ using the manifest.

    Returns:
        dict with keys: copied (list), updated (list), skipped (int),
                        user_modified (list), cleaned (list), total_bundled (int)
    """
    # Opt-out: a profile (named or the default ~/.hermes) that wrote the
    # .no-bundled-skills marker gets zero bundled-skill seeding. Returning the
    # empty-result shape with skipped_opt_out lets callers report "opted out"
    # instead of "synced 0 / failed". This is the default-profile counterpart
    # to seed_profile_skills()'s marker check for named profiles.
    if (HERMES_HOME / NO_BUNDLED_SKILLS_MARKER).exists():
        if not quiet:
            print("  (skipped — profile opted out of bundled skills via .no-bundled-skills)")
        return {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "total_bundled": 0,
            "optional_provenance_backfilled": [], "skipped_opt_out": True,
        }

    bundled_dir = _get_bundled_dir()
    if not bundled_dir.exists():
        return {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "suppressed": [], "total_bundled": 0,
            "optional_provenance_backfilled": [],
        }

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    bundled_names = {name for name, _ in bundled_skills}
    suppressed = _read_suppressed_names()
    # Index of skills already provided by external_dirs (skip writing them)
    external_index = _build_external_skill_index()
    shadowed_by_external: List[str] = []

    copied = []
    updated = []
    user_modified = []
    suppressed_skipped: List[str] = []
    skipped = 0
    not_seeded = 0

    # Seed-time curation applies to NEW profiles only. An existing
    # profile is left exactly as-is (the boundary: don't migrate/cleanup existing
    # profiles). A profile is policy-managed once it has been seeded under the
    # policy (SEED_POLICY_MARKER); a profile that predates the policy keeps the
    # upstream "seed everything" behaviour.
    #
    # "Existing" can't be judged by the manifest alone: an old installer fallback
    # (or any pre-manifest path) may have copied skills WITHOUT writing one, and
    # such a profile must NOT be misread as new and pulled onto the policy path.
    # Signal: a profile carries a bundled skill that is NOT in the seed set. A
    # policy-seeded profile only ever has seeded skills, so this precisely tells
    # an old/pre-policy profile (has extras) apart from a fresh or
    # just-seeded-but-marker-write-failed one (seeded skills only) — without
    # breaking the marker-failure re-seed path.
    raw_policy = _read_seed_policy()
    policy_marker = SKILLS_DIR / SEED_POLICY_MARKER
    # .is_file() (not .exists()): a stray directory/symlink named .seed_policy must
    # NOT count as policy-managed. The genuine marker is always a regular file.
    policy_managed = policy_marker.is_file()
    _seed_set = (raw_policy or {}).get("seed_set", set())
    # Canonical seeded path (relative to SKILLS_DIR) for each bundled skill name —
    # used to confirm an on-disk skill is genuinely that bundled skill, not a
    # name-colliding user/hub skill placed elsewhere.
    _bundled_rel_by_name = {nm: src.relative_to(bundled_dir) for nm, src in bundled_skills}
    # "Existing pre-policy profile?" The reliable signal differs by whether the
    # policy is readable:
    #   * readable: a manifest, OR an on-disk skill that is a BUNDLED skill which
    #     is NOT seeded AND sits at its canonical seeded path — an old fallback
    #     that copied bundled skills without a manifest. BOTH qualifiers are
    #     load-bearing: a user's OWN local/hub skill is legitimately absent from
    #     the seed set, and a name match alone is not enough — a user/hub skill
    #     could collide on frontmatter `name` with a bundled-but-unseeded skill.
    #     Without the bundled-name + canonical-path check, `hermes skills opt-in
    #     --sync` (and any sync on a markerless profile holding such a skill) would
    #     fall to seed_policy=None and seed the full un-curated set. Anchoring on a
    #     real bundled skill AT ITS CANONICAL PATH distinguishes a true pre-policy /
    #     old-fallback profile from a fresh/just-seeded one (seeded skills only)
    #     without breaking the marker-failure re-seed path.
    #   * unreadable/corrupt: _seed_set is EMPTY, so "not in _seed_set" holds for
    #     EVERY on-disk skill. Using it would misclassify a fresh or just-seeded
    #     (marker-write-failed / partial-fallback) profile as an existing
    #     pre-policy one, skip the fail-closed branch below, and seed the full
    #     un-curated branded set. So when the policy can't be read, fall back to
    #     the only trustworthy "established profile" signal: a manifest. A
    #     genuine pre-policy profile (manifest, no marker) still seeds everything;
    #     a fresh/just-seeded profile (no manifest) fails closed.
    if raw_policy is not None:
        def _is_pre_policy_evidence(md: Path) -> bool:
            # Genuine pre-policy/old-fallback evidence = a bundled skill that is
            # NOT seeded AND sits at its canonical seeded path. A user/hub skill
            # whose frontmatter name collides with a bundled-but-unseeded skill
            # lives at a different path (or isn't a bundled name at all) and must
            # NOT count, else it flips the profile onto the full un-curated seed.
            nm = _read_skill_name(md, md.parent.name)
            canonical = _bundled_rel_by_name.get(nm)
            if canonical is None or nm in _seed_set:
                return False
            try:
                return md.parent.relative_to(SKILLS_DIR) == canonical
            except ValueError:
                return False
        profile_existed = MANIFEST_FILE.exists() or any(
            _is_pre_policy_evidence(md)
            for md in SKILLS_DIR.rglob("SKILL.md")
            if not is_excluded_skill_path(md)
        )
    else:
        profile_existed = MANIFEST_FILE.exists()
    # The policy applies to a NEW profile (no manifest) or one already managed
    # under it (has the marker); a profile that predates the policy keeps the
    # upstream "seed everything" behaviour.
    would_manage = (not profile_existed) or policy_managed

    # Fail CLOSED rather than seed the full un-curated (branded) set whenever a
    # profile that WOULD be policy-managed cannot get a readable policy. Two
    # triggers:
    #   * a policy file is shipped but corrupt/unreadable (present-but-broken), or
    #   * the profile is already policy-managed (.seed_policy marker) but no
    #     policy is available at all — a downgrade, or a packaging wrapper that
    #     shipped skills/ without the co-located config/. A curated profile must
    #     never silently expand to every bundled skill just because its policy
    #     vanished.
    # A genuinely pre-policy profile (no marker) with no policy present is NOT
    # affected: would_manage is False for it, so it stays on the upstream
    # "seed everything" path (backward compatible). A truly fresh install with no
    # policy at all is likewise upstream behaviour (would_manage True, but neither
    # a policy file nor a marker is present, so this branch does not fire).
    policy_file_present = _seed_policy_file_present()
    if raw_policy is None and would_manage and (policy_file_present or policy_managed):
        if policy_file_present:
            reason = "the seed policy is present but unreadable/corrupt"
        else:
            reason = (
                "this profile is policy-managed but no seed policy is available "
                "(was config/skill_seed_policy.json dropped from the bundle?)"
            )
        msg = (
            f"{reason} — refusing to seed the full un-curated skill set. Fix the "
            "seed policy (config/skill_seed_policy.json or the HERMES_SEED_POLICY "
            "path) and re-run; no skills were seeded."
        )
        logger.error("skills_sync: %s", msg)
        if not quiet:
            print(f"  ✗ {msg}")
        return {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "suppressed": [],
            "total_bundled": len(bundled_skills), "not_seeded": 0,
            "optional_provenance_backfilled": [], "policy_error": True,
        }

    seed_policy = raw_policy if raw_policy is not None and would_manage else None
    seeded_names: set = set()

    for skill_name, skill_src in bundled_skills:
        # Curator-pruned built-ins: do not re-seed. The suppression list
        # (~/.hermes/skills/.curator_suppressed) is written when the curator
        # archives a bundled skill with curator.prune_builtins enabled. Without
        # this skip, every `hermes update` would resurrect a skill the user
        # deliberately pruned. Restoring the skill clears its suppression entry.
        if skill_name in suppressed:
            suppressed_skipped.append(skill_name)
            continue

        # Resolve the seed target (allowlist filter) BEFORE hashing so
        # non-seeded skills cost only a set lookup.
        resolved = _resolve_seed_target(skill_name, skill_src, bundled_dir, seed_policy)
        if resolved is None:
            not_seeded += 1
            continue
        skill_name, skill_src, dest = resolved
        seeded_names.add(skill_name)
        bundled_hash = _dir_hash(skill_src)

        # Recover an orphaned backup before classifying. If a previous
        # update was interrupted between moving dest aside and copying the
        # new version in, the user's only copy sits in ``dest.bak`` while
        # dest is gone — without this, the "in manifest but not on disk"
        # branch below misreads the skill as user-deleted and it silently
        # vanishes from discovery.
        _orphan = dest.with_suffix(".bak")
        if _orphan.exists() and not dest.exists():
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(_orphan), str(dest))
                logger.info("Recovered orphaned skill backup: %s", _orphan)
            except (OSError, IOError):
                logger.warning(
                    "Could not recover orphaned skill backup %s", _orphan,
                    exc_info=True,
                )

        if skill_name in external_index:
            # An external_dirs source already provides this skill. Writing it
            # into the profile-local tree would create a name collision the
            # loader refuses to resolve (#28126). Defer to the external copy
            # for ALL manifest states (new, previously-synced, user-deleted).
            shadowed_by_external.append(skill_name)
            skipped += 1
            if not quiet:
                print(
                    f"  ⇢ {skill_name} (deferred to external_dirs, "
                    "not written to local tree)"
                )
            # Self-healing: a prior sync (before external_dirs was configured,
            # or an older buggy sync) may have left a local shadow that now
            # collides. We own that shadow only when it is byte-identical to
            # the bundled source — a user's own customized skill by the same
            # name differs, so never delete or re-baseline it. Drop the stale
            # manifest entry so the skill isn't later misread as user-deleted.
            if dest.exists() and _dir_hash(dest) == bundled_hash:
                _rmtree_writable(dest)
                if not quiet:
                    print(f"  ✓ removed stale shadow of {skill_name}")
                manifest.pop(skill_name, None)
            continue

        if skill_name not in manifest:
            # ── New skill — never offered before ──
            try:
                if dest.exists():
                    # User already has a skill with the same name — don't overwrite.
                    # Only baseline in the manifest when the on-disk copy is
                    # byte-identical to bundled (e.g. a reset that re-syncs, or
                    # a coincidentally identical install); that case is harmless
                    # to track. If the copy differs (custom skill, hub-installed,
                    # or user-edited) skip the manifest write: recording
                    # bundled_hash there would poison update detection by making
                    # user_hash != origin_hash read as "user-modified" on every
                    # subsequent sync, permanently blocking bundled updates.
                    skipped += 1
                    if _dir_hash(dest) == bundled_hash:
                        manifest[skill_name] = bundled_hash
                    elif not quiet:
                        print(
                            f"  ⚠ {skill_name}: bundled version shipped but you "
                            f"already have a local skill by this name — yours "
                            f"was kept. Run `hermes skills reset {skill_name}` "
                            f"to replace it with the bundled version."
                        )
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(skill_src, dest, ignore=_COPY_IGNORE)
                    copied.append(skill_name)
                    manifest[skill_name] = bundled_hash
                    if not quiet:
                        print(f"  + {skill_name}")
            except (OSError, IOError) as e:
                if not quiet:
                    print(f"  ! Failed to copy {skill_name}: {e}")
                # Do NOT add to manifest — next sync should retry

        elif dest.exists():
            # ── Existing skill — in manifest AND on disk ──
            origin_hash = manifest.get(skill_name, "")
            user_hash = _dir_hash(dest)

            if not origin_hash:
                # v1 migration: no origin hash recorded. Set baseline from
                # user's current copy so future syncs can detect modifications.
                manifest[skill_name] = user_hash
                if user_hash == bundled_hash:
                    skipped += 1  # already in sync
                else:
                    # Can't tell if user modified or bundled changed — be safe
                    skipped += 1
                continue

            if _is_tracked_user_modification(origin_hash, user_hash):
                # User modified this skill — don't overwrite their changes
                user_modified.append(skill_name)
                if not quiet:
                    print(f"  ~ {skill_name} (user-modified, skipping)")
                continue

            # User copy matches origin — check if bundled has a newer version
            if bundled_hash != origin_hash:
                try:
                    # Move old copy to a backup so we can restore on failure
                    backup = dest.with_suffix(".bak")
                    # A stale backup left by an earlier failure would make
                    # shutil.move() nest dest *inside* it (or fail outright)
                    # and would poison the restore path below. The current
                    # dest is the authoritative copy — clear the leftover.
                    if backup.exists():
                        _rmtree_writable(backup)
                    shutil.move(str(dest), str(backup))
                    try:
                        shutil.copytree(skill_src, dest, ignore=_COPY_IGNORE)
                        manifest[skill_name] = bundled_hash
                        updated.append(skill_name)
                        if not quiet:
                            print(f"  ↑ {skill_name} (updated)")
                        # Remove backup after successful copy
                        try:
                            _rmtree_writable(backup)
                        except (OSError, IOError):
                            logger.debug("Could not remove backup %s", backup, exc_info=True)
                    except (OSError, IOError):
                        # Restore from backup. A partially-written dest must
                        # not shadow the user's copy or block the restore —
                        # clear it first, then move the backup home.
                        if backup.exists():
                            if dest.exists():
                                try:
                                    _rmtree_writable(dest)
                                except (OSError, IOError):
                                    logger.warning(
                                        "Could not clear partial copy %s during restore",
                                        dest, exc_info=True,
                                    )
                            if not dest.exists():
                                shutil.move(str(backup), str(dest))
                        raise
                except (OSError, IOError) as e:
                    if not quiet:
                        print(f"  ! Failed to update {skill_name}: {e}")
            else:
                skipped += 1  # bundled unchanged, user unchanged

        else:
            # ── In manifest but not on disk — user deleted it ──
            skipped += 1

    # Clean stale manifest entries (skills no longer seeded). Under a seed
    # policy the universe is the set of effective seeded names; otherwise it is
    # every discovered bundled skill (upstream behaviour).
    universe = seeded_names if seed_policy is not None else bundled_names
    cleaned = sorted(set(manifest.keys()) - universe)
    for name in cleaned:
        del manifest[name]

    # Also copy DESCRIPTION.md files — but only for categories that actually
    # have a seeded skill in this profile. Otherwise a seed policy that filtered
    # a whole category out would still leave its (now-empty) category
    # description behind. (No policy / existing profile -> every category has
    # skills -> identical to before.)
    for desc_md in bundled_dir.rglob("DESCRIPTION.md"):
        rel = desc_md.relative_to(bundled_dir)
        dest_desc = SKILLS_DIR / rel
        if dest_desc.exists():
            continue
        cat_dir = dest_desc.parent
        if not (cat_dir.exists() and any(cat_dir.rglob("SKILL.md"))):
            continue
        try:
            dest_desc.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(desc_md, dest_desc)
        except (OSError, IOError) as e:
            logger.debug("Could not copy %s: %s", desc_md, e)

    # Mark the profile as policy-managed BEFORE persisting the manifest. A crash
    # between the two then leaves "marker, no manifest", which the next sync
    # reads as a fresh profile and re-applies the policy (safe). The reverse
    # order could leave "manifest, no marker" and permanently flip the profile
    # onto the unfiltered "seed everything" path.
    marker_persisted = True
    if seed_policy is not None:
        try:
            policy_marker.write_text(
                str(seed_policy.get("version", 1)) + "\n", encoding="utf-8"
            )
        except OSError as e:
            # The marker is the ONLY signal that tells the next sync this
            # profile is policy-managed. If we cannot write it but DO write the
            # manifest, the next sync sees "manifest, no marker", classifies the
            # profile as pre-policy, and seeds the full un-curated (branded) set
            # — permanently. So when the marker write fails, skip the manifest
            # write too: the profile stays manifest-less and the next sync
            # re-seeds under the policy (cheap — the seeded skills are already on
            # disk and baseline silently) and retries the marker. A repeated
            # seed is strictly safer than a silent, permanent flip to the full
            # branded set.
            marker_persisted = False
            logger.warning(
                "Could not write seed-policy marker %s: %s; skipping manifest "
                "write so the next sync re-seeds under policy.",
                policy_marker,
                e,
            )
            if not quiet:
                print(
                    f"  ⚠ could not persist seed-policy marker ({e}); "
                    f"skills will re-seed under policy on the next run"
                )
    if marker_persisted:
        _write_manifest(manifest)
    optional_provenance_backfilled = _backfill_optional_provenance(quiet=quiet)

    return {
        "copied": copied,
        "updated": updated,
        "skipped": skipped,
        "user_modified": user_modified,
        "cleaned": cleaned,
        "suppressed": suppressed_skipped,
        "total_bundled": len(bundled_skills),
        "not_seeded": not_seeded,
        "optional_provenance_backfilled": optional_provenance_backfilled,
        "shadowed_by_external": shadowed_by_external,
    }


def _rmtree_writable(path: Path) -> None:
    """Remove a directory tree, making read-only entries writable first.

    Handles immutable package sources (Nix store, deb/rpm installs) that
    preserve read-only permissions on copied files *and* directories
    (``r-xr-xr-x``).  Removing a child requires write permission on its
    parent directory, so the retry handler makes the failing path **and its
    parent** writable before re-attempting.  See #34860, #34972.
    """
    # Defense in depth (#48200): refuse to rmtree anything outside
    # ``HERMES_HOME/skills/`` to prevent the catastrophic wipe of
    # ``~/.hermes/`` (``.env``, ``MEMORY.md``, ``kanban.db``, custom
    # skills, scripts, …) that an earlier incident observed. Five call
    # sites in this file invoke this helper; if any one of them ever
    # computes a destination outside the skills root — through a bad
    # path join, a missing ``HERMES_HOME`` default, a malicious
    # bundled-manifest entry, or a mid-flight exception that leaves a
    # stale path in scope — this guard turns the resulting
    # ``shutil.rmtree(~/.hermes)`` into a loud, recoverable ``ValueError``
    # instead of silently destroying the user's install.
    target = Path(path).resolve()
    skills_root = SKILLS_DIR.resolve()
    # Every legitimate caller passes a skill directory or its ``.bak``
    # sibling — always a strict child of the skills root. The skills root
    # itself must never be removed: a ``dest`` that collapses to
    # ``SKILLS_DIR`` (e.g. a relative path resolving to ``.``) would wipe
    # every installed skill, and its ``.bak`` sibling lands one level up in
    # ``HERMES_HOME``. Require a strict-child relationship so both escape
    # into the skills root and out of it are refused.
    if skills_root not in target.parents:
        raise ValueError(
            f"refusing to rmtree {target!r}: not strictly under {skills_root!r} "
            f"(scope guard — see #48200)"
        )
    import stat

    def _on_error(func, fpath, exc_info):
        # Unlinking a child requires the parent dir to be writable, so chmod
        # the parent as well as the failing path, then retry.
        for target in (os.path.dirname(fpath), fpath):
            try:
                os.chmod(target, stat.S_IRWXU)
            except OSError:
                pass
        func(fpath)

    shutil.rmtree(path, onerror=_on_error)


def reset_bundled_skill(name: str, restore: bool = False) -> dict:
    """
    Reset a bundled skill's manifest tracking so future syncs work normally.

    When a user edits a bundled skill, subsequent syncs mark it as
    ``user_modified`` and skip it forever — even if the user later copies
    the bundled version back into place, because the manifest still holds
    the *old* origin hash. This function breaks that loop.

    Args:
        name: The skill name (matches the manifest key / skill frontmatter name).
        restore: If True, also delete the user's copy in SKILLS_DIR and let
                 the next sync re-copy the current bundled version. If False
                 (default), only clear the manifest entry — the user's
                 current copy is preserved but future updates work again.

    Returns:
        dict with keys:
          - ok: bool, whether the reset succeeded
          - action: one of "manifest_cleared", "restored", "not_in_manifest",
                    "bundled_missing", "opted_out"
          - message: human-readable description
          - synced: dict from sync_skills() if a sync was triggered, else None
    """
    # Opt-out profiles take NO bundled-skill state changes. sync_skills() below
    # is a no-op under the .no-bundled-skills marker, so clearing the manifest
    # entry (or deleting the on-disk copy) here would mutate tracking that nothing
    # will rebuild — emptying the manifest while still reporting success. Refuse
    # up front so opt-out's "sync/update don't touch bundled state" invariant holds.
    if (HERMES_HOME / NO_BUNDLED_SKILLS_MARKER).exists():
        return {
            "ok": False,
            "action": "opted_out",
            "message": (
                f"This profile opted out of bundled skills (.no-bundled-skills), "
                f"so '{name}' was not reset. Run `hermes skills opt-in` first."
            ),
            "synced": None,
        }

    manifest = _read_manifest()
    bundled_dir = _get_bundled_dir()
    bundled_skills = _discover_bundled_skills(bundled_dir)
    bundled_by_name = dict(bundled_skills)

    in_manifest = name in manifest
    is_bundled = name in bundled_by_name

    if not in_manifest and not is_bundled:
        return {
            "ok": False,
            "action": "not_in_manifest",
            "message": (
                f"'{name}' is not a tracked bundled skill. Nothing to reset. "
                f"(Hub-installed skills use `hermes skills uninstall`.)"
            ),
            "synced": None,
        }

    # Step 1 (optional): delete the user's copy so next sync re-copies bundled.
    # Must happen BEFORE manifest deletion so that a failed rmtree does not
    # leave the skill in a manifest-less limbo state (see #34972).
    deleted_user_copy = False
    if restore:
        if not is_bundled:
            return {
                "ok": False,
                "action": "bundled_missing",
                "message": (
                    f"'{name}' has no bundled source — manifest entry preserved "
                    f"but cannot restore from bundled (skill was removed upstream)."
                ),
                "synced": None,
            }
        dest = _compute_relative_dest(bundled_by_name[name], bundled_dir)
        if dest.exists():
            try:
                _rmtree_writable(dest)
                deleted_user_copy = True
            except (OSError, IOError) as e:
                return {
                    "ok": False,
                    "action": "not_reset",
                    "message": (
                        f"Could not delete user copy at {dest}: {e}. "
                        f"Manifest entry preserved — nothing was changed."
                    ),
                    "synced": None,
                }
        # Restoring is an explicit un-prune: drop any curator suppression so the
        # following sync actually re-seeds + tracks the skill (sync_skills skips
        # suppressed names — `if skill_name in suppressed: continue` — leaving it
        # restored on disk but never updated). Do this ONLY AFTER the user copy was
        # removed (or there was none): clearing it before the rmtree above would
        # change the prune state even when restore bails out with "nothing was
        # changed". (sync_skills' own comment promises restore clears suppression.)
        try:
            from tools.skill_usage import remove_suppressed_name

            remove_suppressed_name(name)
        except Exception:
            logger.debug("Could not clear suppression for %s", name, exc_info=True)

    # Step 2: drop the manifest entry so next sync treats it as new
    if in_manifest:
        del manifest[name]
        _write_manifest(manifest)

    # Step 3: run sync to re-baseline (or re-copy if we deleted)
    synced = sync_skills(quiet=True)

    restored_unmanaged = False
    if restore:
        # sync_skills filters to the seed allowlist, so a bundled skill that is
        # NOT in the policy won't be re-copied — `reset --restore` would then
        # delete the copy and report success without restoring anything. An
        # explicit reset is a direct request to restore THIS skill, so copy it
        # from bundled regardless of the allowlist when sync didn't.
        dest = _compute_relative_dest(bundled_by_name[name], bundled_dir)
        if not dest.exists():
            try:
                src = bundled_by_name[name]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, ignore=_COPY_IGNORE)
                # Deliberately NO manifest entry: this skill is outside the seed
                # set, so the next policy-managed sync would drop the entry anyway
                # (cleanup keeps the manifest to the seeded set). Writing one would
                # only churn and falsely imply it's tracked for auto-updates.
                restored_unmanaged = True
            except (OSError, IOError) as e:
                return {
                    "ok": False,
                    "action": "not_reset",
                    "message": f"Could not restore '{name}' from bundled source: {e}.",
                    "synced": synced,
                }

    if restore and restored_unmanaged:
        action = "restored"
        message = (
            f"Restored '{name}' from bundled source. It is not in this profile's "
            f"seed set, so it stays on disk but is NOT auto-updated on future syncs."
        )
    elif restore and deleted_user_copy:
        action = "restored"
        message = f"Restored '{name}' from bundled source."
    elif restore:
        action = "restored"
        message = f"Restored '{name}' from bundled source (no prior user copy)."
    else:
        action = "manifest_cleared"
        message = (
            f"Cleared manifest entry for '{name}'. Future `hermes update` runs "
            f"will re-baseline against your current copy and accept upstream changes."
        )

    if synced and synced.get("policy_error"):
        # The internal re-baseline sync fail-closed (seed policy corrupt/missing),
        # so bundled-skill tracking was NOT rebuilt and the manifest entry cleared
        # above was not re-seeded. Don't report a plain green success — the user
        # must fix the policy and re-run. (Any explicit on-disk restore above still
        # happened; this flags the seeding/tracking failure so the UI shows it.)
        return {
            "ok": False,
            "action": "policy_error",
            "message": (
                f"{message} ⚠ Bundled-skill seeding is fail-closed: the seed policy "
                f"is present but unreadable/corrupt, so tracking was not rebuilt. "
                f"Fix config/skill_seed_policy.json and run `hermes update`."
            ),
            "synced": synced,
        }
    return {"ok": True, "action": action, "message": message, "synced": synced}


def _is_tracked_user_modification(origin_hash: str, user_hash: str) -> bool:
    """Whether an on-disk skill counts as a user modification ``hermes update`` keeps.

    Shared by the sync loop (which decides what to skip) and
    ``list_user_modified_bundled_skills`` (which surfaces the names) so the two
    can never drift. A skill is a tracked modification only when it has a
    recorded origin hash (an un-baselined / v1 entry with an empty hash is not)
    and its current content hash differs from that origin.
    """
    return bool(origin_hash) and user_hash != origin_hash


def list_user_modified_bundled_skills() -> List[dict]:
    """Return the bundled skills that ``hermes update`` keeps because the user
    edited them locally.

    A skill counts as user-modified when its on-disk copy no longer matches the
    origin hash recorded in the manifest the last time it was synced — the exact
    same test the sync loop uses to decide what to skip. This is the discovery
    half of that behavior, so a user can find the names the ``~ N user-modified
    (kept)`` notice only counts.

    Returns a list (sorted by name) of dicts:
        ``{"name": str, "dest": Path, "bundled_src": Path}``
    where ``dest`` is the user's copy and ``bundled_src`` is the current stock
    copy (so callers can diff or restore).
    """
    manifest = _read_manifest()
    if not manifest:
        return []
    bundled_dir = _get_bundled_dir()
    modified: List[dict] = []
    for skill_name, skill_dir in _discover_bundled_skills(bundled_dir):
        origin_hash = manifest.get(skill_name, "")
        # No entry, or a v1 entry not yet baselined (empty hash): not a tracked
        # modification — the next sync handles it.
        if not origin_hash:
            continue
        dest = _compute_relative_dest(skill_dir, bundled_dir)
        if not dest.exists():
            continue
        if _is_tracked_user_modification(origin_hash, _dir_hash(dest)):
            modified.append(
                {"name": skill_name, "dest": dest, "bundled_src": skill_dir}
            )
    modified.sort(key=lambda e: e["name"])
    return modified


def _read_for_diff(path: Path) -> Tuple[Optional[bytes], Optional[str]]:
    """Read a file once for diffing.

    Returns ``(raw_bytes, text)`` where ``text`` is ``None`` if the file is
    binary; ``(None, None)`` if it could not be read. Returning the raw bytes
    lets the caller compare binary files without re-reading them.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None, None
    if b"\x00" in data:
        return data, None
    try:
        return data, data.decode("utf-8")
    except UnicodeDecodeError:
        return data, None


def diff_bundled_skill(name: str) -> dict:
    """Diff a user's copy of a bundled skill against the current stock version.

    Lets a user see exactly what diverged before deciding whether to keep their
    edits or ``hermes skills reset`` back to upstream.

    Returns a dict:
        ``ok`` (bool), ``name`` (str), ``found`` (bool — bundled source exists),
        ``modified`` (bool), ``message`` (str),
        ``diffs``: list of ``{"path": str, "status": str, "diff": str}`` where
        status is one of ``modified`` / ``added`` (only in user copy) /
        ``removed`` (only in bundled) / ``binary``.
    """
    import difflib

    bundled_dir = _get_bundled_dir()
    bundled_by_name = dict(_discover_bundled_skills(bundled_dir))
    bundled_src = bundled_by_name.get(name)
    if bundled_src is None:
        return {
            "ok": False,
            "name": name,
            "found": False,
            "modified": False,
            "diffs": [],
            "message": (
                f"'{name}' is not a tracked bundled skill (no stock version to "
                f"diff against). Hub-installed skills use `hermes skills inspect`."
            ),
        }
    dest = _compute_relative_dest(bundled_src, bundled_dir)
    if not dest.exists():
        return {
            "ok": False,
            "name": name,
            "found": True,
            "modified": False,
            "diffs": [],
            "message": f"No local copy of '{name}' found at {dest}.",
        }

    user_files = set(_skill_file_list(dest))
    stock_files = set(_skill_file_list(bundled_src))

    diffs: List[dict] = []
    for rel in sorted(user_files | stock_files):
        in_user = rel in user_files
        in_stock = rel in stock_files
        user_bytes, user_text = (
            _read_for_diff(dest / rel) if in_user else (None, None)
        )
        stock_bytes, stock_text = (
            _read_for_diff(bundled_src / rel) if in_stock else (None, None)
        )

        if in_user and in_stock:
            if user_text is None or stock_text is None:
                # At least one side is binary — report only if bytes differ
                # (reuse the bytes already read above, no second read).
                if user_bytes != stock_bytes:
                    diffs.append(
                        {"path": rel, "status": "binary", "diff": "<binary file differs>"}
                    )
                continue
            if user_text == stock_text:
                continue
            text = "".join(
                difflib.unified_diff(
                    stock_text.splitlines(keepends=True),
                    user_text.splitlines(keepends=True),
                    fromfile=f"stock/{rel}",
                    tofile=f"yours/{rel}",
                )
            )
            diffs.append({"path": rel, "status": "modified", "diff": text})
        elif in_user:
            diffs.append(
                {"path": rel, "status": "added", "diff": f"+ only in your copy: {rel}"}
            )
        else:
            diffs.append(
                {"path": rel, "status": "removed", "diff": f"- only in stock: {rel}"}
            )

    modified = bool(diffs)
    return {
        "ok": True,
        "name": name,
        "found": True,
        "modified": modified,
        "diffs": diffs,
        "message": (
            f"'{name}' matches the stock version."
            if not modified
            else f"'{name}' differs from the stock version in {len(diffs)} file(s)."
        ),
    }


def set_bundled_skills_opt_out(enabled: bool) -> dict:
    """Toggle the .no-bundled-skills opt-out marker for the active profile.

    When ``enabled`` is True, writes HERMES_HOME/.no-bundled-skills so the
    installer, ``hermes update``, and any direct sync stop seeding bundled
    skills. When False, removes the marker so seeding resumes on the next
    sync. This is the on-disk-state half of ``hermes skills opt-out`` /
    ``opt-in``; removal of already-present skills is a separate, explicit
    step (see ``remove_pristine_bundled_skills``).

    Returns:
        dict with keys: ok (bool), changed (bool), marker (str path),
                        message (str).
    """
    marker = HERMES_HOME / NO_BUNDLED_SKILLS_MARKER
    existed = marker.exists()
    try:
        if enabled:
            HERMES_HOME.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                "This profile opted out of bundled-skill seeding "
                "(`hermes skills opt-out`).\n"
                "Delete this file to re-enable sync on the next `hermes update`.\n",
                encoding="utf-8",
            )
            changed = not existed
            message = (
                "Opted out of bundled skills. Future install / update / sync "
                "runs will not seed bundled skills into this profile."
                if changed
                else "Already opted out — marker was already present."
            )
        else:
            if existed:
                marker.unlink()
            changed = existed
            message = (
                "Opted back in. The next `hermes update` (or `hermes skills "
                "opt-in --sync`) will re-seed bundled skills."
                if changed
                else "Not opted out — no marker to remove."
            )
    except OSError as e:
        return {
            "ok": False, "changed": False, "marker": str(marker),
            "message": f"Could not update opt-out marker at {marker}: {e}",
        }
    return {"ok": True, "changed": changed, "marker": str(marker), "message": message}


def is_bundled_skills_opt_out() -> bool:
    """Return True if the active profile carries the opt-out marker."""
    return (HERMES_HOME / NO_BUNDLED_SKILLS_MARKER).exists()


def remove_pristine_bundled_skills(dry_run: bool = False) -> dict:
    """Delete bundled skills that are present, manifest-tracked, AND unmodified.

    Safety is the whole point of this function. A skill on disk is removed
    ONLY when all of these hold:
      - it is recorded in the sync manifest (so it is genuinely a bundled
        skill, not a hub-installed or hand-written one), AND
      - it still exists in the bundled source (so we can hash-compare), AND
      - its on-disk copy is byte-identical to the manifest origin hash
        (so the user has not edited it).

    Anything user-modified, hub-installed, or locally authored is left
    untouched and reported under ``skipped``. The manifest entry for each
    removed skill is dropped so a later opt-in re-seed treats it as new.

    Args:
        dry_run: When True, compute what would be removed without deleting.

    Returns:
        dict with keys: ok (bool), removed (list[str]),
                        skipped (list[dict]) where each dict is
                        {name, reason}, dry_run (bool), message (str).
    """
    manifest = _read_manifest()
    bundled_dir = _get_bundled_dir()
    bundled_by_name = dict(_discover_bundled_skills(bundled_dir))

    removed: List[str] = []
    skipped: List[dict] = []

    for name, origin_hash in sorted(manifest.items()):
        src = bundled_by_name.get(name)
        if src is None:
            # Tracked but no longer bundled upstream — leave it; not ours to judge.
            skipped.append({"name": name, "reason": "no bundled source (removed upstream)"})
            continue
        dest = _compute_relative_dest(src, bundled_dir)
        if not dest.exists():
            # Already gone from disk; just forget the stale manifest entry.
            if not dry_run and name in manifest:
                del manifest[name]
            continue
        on_disk = _dir_hash(dest)
        if on_disk != origin_hash:
            skipped.append({"name": name, "reason": "user-modified (kept)"})
            continue
        # Pristine bundled copy — safe to remove.
        if dry_run:
            removed.append(name)
            continue
        try:
            _rmtree_writable(dest)
        except (OSError, IOError) as e:
            skipped.append({"name": name, "reason": f"delete failed: {e}"})
            continue
        if name in manifest:
            del manifest[name]
        removed.append(name)

    if not dry_run and removed:
        _write_manifest(manifest)

    verb = "Would remove" if dry_run else "Removed"
    message = f"{verb} {len(removed)} pristine bundled skill(s); kept {len(skipped)}."
    return {
        "ok": True, "removed": removed, "skipped": skipped,
        "dry_run": dry_run, "message": message,
    }


if __name__ == "__main__":
    import sys

    print("Syncing bundled skills into ~/.hermes/skills/ ...")
    result = sync_skills(quiet=False)
    # Fail-closed corrupt-policy path: sync_skills already printed a loud error
    # and seeded nothing. Exit non-zero so callers (install.sh / setup-hermes.sh
    # `if python …; then … else <fallback>`, and any CI/profile-create wrapper)
    # treat it as a failure and fall back to the pre-baked manifest instead of
    # reporting a bogus "0 synced" success.
    if result.get("policy_error"):
        print("Skill seeding FAILED: seed policy present but unreadable/corrupt.")
        sys.exit(2)
    parts = [
        f"{len(result['copied'])} new",
        f"{len(result['updated'])} updated",
        f"{result['skipped']} unchanged",
    ]
    if result["user_modified"]:
        names = result["user_modified"]
        MAX_SHOW = 5
        shown = ", ".join(names[:MAX_SHOW])
        if len(names) > MAX_SHOW:
            shown += f", +{len(names) - MAX_SHOW} more"
        parts.append(f"{len(names)} user-modified (kept): {shown}")
    if result["cleaned"]:
        parts.append(f"{len(result['cleaned'])} cleaned from manifest")
    if result.get("optional_provenance_backfilled"):
        parts.append(f"{len(result['optional_provenance_backfilled'])} official optional backfilled")
    print(f"\nDone: {', '.join(parts)}. {result['total_bundled']} total bundled.")
