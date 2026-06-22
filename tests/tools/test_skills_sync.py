"""Tests for tools/skills_sync.py — manifest-based skill seeding and updating."""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from tools.skills_sync import (
    SEED_POLICY_MARKER,
    _get_bundled_dir,
    _read_manifest,
    _read_seed_policy,
    _read_skill_name,
    _resolve_seed_target,
    _write_manifest,
    _discover_bundled_skills,
    _compute_relative_dest,
    _dir_hash,
    sync_skills,
    reset_bundled_skill,
    restore_official_optional_skill,
)


class TestReadWriteManifest:
    def test_read_missing_manifest(self, tmp_path):
        with patch(
            "tools.skills_sync.MANIFEST_FILE",
            tmp_path / "nonexistent",
        ):
            result = _read_manifest()
        assert result == {}

    def test_write_and_read_roundtrip_v2(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        entries = {"skill-a": "abc123", "skill-b": "def456", "skill-c": "789012"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(entries)
            result = _read_manifest()

        assert result == entries

    def test_write_manifest_sorted(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        entries = {"zebra": "hash1", "alpha": "hash2", "middle": "hash3"}

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest(entries)

        lines = manifest_file.read_text().strip().splitlines()
        names = [line.split(":")[0] for line in lines]
        assert names == ["alpha", "middle", "zebra"]

    def test_read_v1_manifest_migration(self, tmp_path):
        """v1 format (plain names, no hashes) should be read with empty hashes."""
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("skill-a\nskill-b\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"skill-a": "", "skill-b": ""}

    def test_read_manifest_ignores_blank_lines(self, tmp_path):
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("skill-a:hash1\n\n  \nskill-b:hash2\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"skill-a": "hash1", "skill-b": "hash2"}

    def test_read_manifest_mixed_v1_v2(self, tmp_path):
        """Manifest with both v1 and v2 lines (shouldn't happen but handle gracefully)."""
        manifest_file = tmp_path / ".bundled_manifest"
        manifest_file.write_text("old-skill\nnew-skill:abc123\n")

        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            result = _read_manifest()

        assert result == {"old-skill": "", "new-skill": "abc123"}


class TestDirHash:
    def test_same_content_same_hash(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        for d in (dir_a, dir_b):
            d.mkdir()
            (d / "SKILL.md").write_text("# Test")
            (d / "main.py").write_text("print(1)")
        assert _dir_hash(dir_a) == _dir_hash(dir_b)

    def test_different_content_different_hash(self, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "SKILL.md").write_text("# Version 1")
        (dir_b / "SKILL.md").write_text("# Version 2")
        assert _dir_hash(dir_a) != _dir_hash(dir_b)

    def test_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        h = _dir_hash(d)
        assert isinstance(h, str) and len(h) == 32

    def test_nonexistent_dir(self, tmp_path):
        h = _dir_hash(tmp_path / "nope")
        assert isinstance(h, str)  # returns hash of empty content

    def test_dir_hash_ignores_copy_junk(self, tmp_path):
        """_dir_hash must skip the same patterns copytree ignores, so the seed
        baseline matches the copied file set (else skills with __pycache__/.pyc/
        .DS_Store in source are wrongly flagged user-modified and never update)."""
        clean = tmp_path / "clean"
        clean.mkdir()
        (clean / "SKILL.md").write_text("# s")
        (clean / "scripts").mkdir()
        (clean / "scripts" / "run.py").write_text("print(1)")

        dirty = tmp_path / "dirty"
        dirty.mkdir()
        (dirty / "SKILL.md").write_text("# s")
        (dirty / "scripts").mkdir()
        (dirty / "scripts" / "run.py").write_text("print(1)")
        (dirty / "scripts" / "__pycache__").mkdir()
        (dirty / "scripts" / "__pycache__" / "run.cpython-311.pyc").write_bytes(b"\x00j")
        (dirty / ".DS_Store").write_bytes(b"\x00")

        assert _dir_hash(clean) == _dir_hash(dirty)


class TestDiscoverBundledSkills:
    def test_finds_skills_with_skill_md(self, tmp_path):
        (tmp_path / "category" / "skill-a").mkdir(parents=True)
        (tmp_path / "category" / "skill-a" / "SKILL.md").write_text("# Skill A")
        (tmp_path / "skill-b").mkdir()
        (tmp_path / "skill-b" / "SKILL.md").write_text("# Skill B")
        (tmp_path / "not-a-skill").mkdir()
        (tmp_path / "not-a-skill" / "README.md").write_text("Not a skill")

        skills = _discover_bundled_skills(tmp_path)
        skill_names = {name for name, _ in skills}
        assert "skill-a" in skill_names
        assert "skill-b" in skill_names
        assert "not-a-skill" not in skill_names

    def test_ignores_git_directories(self, tmp_path):
        (tmp_path / ".git" / "hooks").mkdir(parents=True)
        (tmp_path / ".git" / "hooks" / "SKILL.md").write_text("# Fake")
        skills = _discover_bundled_skills(tmp_path)
        assert len(skills) == 0

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        skills = _discover_bundled_skills(tmp_path / "nonexistent")
        assert skills == []

    def test_venv_named_install_prefix_does_not_exclude_bundled(self, tmp_path):
        """Regression: a wheel installed into a venv named `venv`/`.venv` lands the
        bundled skills at <venv>/skills/... is_excluded_skill_path matches ANY path
        component, so an ABSOLUTE check would treat the install prefix's own
        `venv`/`site-packages` component as a dependency dir and drop every bundled
        skill (seed 0) — the two most common venv names trigger it. Discovery must
        check exclusion RELATIVE to the bundled dir."""
        for prefix in ("venv", ".venv", "site-packages"):
            bundled = tmp_path / prefix / "skills"
            (bundled / "cat" / "keep-me").mkdir(parents=True)
            (bundled / "cat" / "keep-me" / "SKILL.md").write_text("---\nname: keep-me\n---\n# k")
            names = {n for n, _ in _discover_bundled_skills(bundled)}
            assert "keep-me" in names, f"bundled skill dropped under install prefix '{prefix}'"
        # ...but junk BELOW the bundled root (inside a skill) is still excluded.
        b2 = tmp_path / "ok" / "skills"
        (b2 / "cat" / "s" / "__pycache__").mkdir(parents=True)
        (b2 / "cat" / "s" / "__pycache__" / "SKILL.md").write_text("---\nname: junk\n---\n# j")
        assert "junk" not in {n for n, _ in _discover_bundled_skills(b2)}


class TestReadSkillName:
    def test_reads_name_from_frontmatter(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname: audiocraft-audio-generation\n---\n# Skill")
        assert _read_skill_name(skill_md, "audiocraft") == "audiocraft-audio-generation"

    def test_falls_back_to_dir_name_without_frontmatter(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# Just a heading\nNo frontmatter here")
        assert _read_skill_name(skill_md, "my-skill") == "my-skill"

    def test_falls_back_when_name_field_empty(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("---\nname:\n---\n")
        assert _read_skill_name(skill_md, "fallback") == "fallback"

    def test_handles_quoted_name(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text('---\nname: "serving-llms-vllm"\n---\n')
        assert _read_skill_name(skill_md, "vllm") == "serving-llms-vllm"

    def test_discover_uses_frontmatter_name(self, tmp_path):
        skill_dir = tmp_path / "category" / "audiocraft"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: audiocraft-audio-generation\n---\n# Skill"
        )
        skills = _discover_bundled_skills(tmp_path)
        assert skills[0][0] == "audiocraft-audio-generation"


class TestComputeRelativeDest:
    def test_preserves_category_structure(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/mlops/axolotl")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert str(dest).endswith("mlops/axolotl")

    def test_flat_skill(self):
        bundled = Path("/repo/skills")
        skill_dir = Path("/repo/skills/simple")
        dest = _compute_relative_dest(skill_dir, bundled)
        assert dest.name == "simple"


class TestSyncSkills:
    def _setup_bundled(self, tmp_path):
        """Create a fake bundled skills directory."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "category" / "new-skill").mkdir(parents=True)
        (bundled / "category" / "new-skill" / "SKILL.md").write_text("# New")
        (bundled / "category" / "new-skill" / "main.py").write_text("print(1)")
        (bundled / "category" / "DESCRIPTION.md").write_text("Category desc")
        (bundled / "old-skill").mkdir()
        (bundled / "old-skill" / "SKILL.md").write_text("# Old")
        return bundled

    def _patches(self, bundled, skills_dir, manifest_file):
        """Return context manager stack for patching sync globals."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def test_suppressed_builtin_not_reseeded(self, tmp_path):
        """A curator-pruned built-in in the suppression list must NOT be
        re-copied on sync — that's what makes the prune durable across updates.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync._read_suppressed_names", return_value={"old-skill"}):
            result = sync_skills(quiet=True)

        # old-skill is suppressed → skipped, not copied.
        assert "old-skill" in result["suppressed"]
        assert "old-skill" not in result["copied"]
        assert not (skills_dir / "old-skill").exists()
        # The non-suppressed bundled skill is still copied normally.
        assert "new-skill" in result["copied"]
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

    def test_fresh_install_copies_all(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert len(result["copied"]) == 2
        assert result["total_bundled"] == 2
        assert result["updated"] == []
        assert result["user_modified"] == []
        assert result["cleaned"] == []
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()
        assert (skills_dir / "old-skill" / "SKILL.md").exists()
        assert (skills_dir / "category" / "DESCRIPTION.md").exists()

    def test_fresh_install_records_origin_hashes(self, tmp_path):
        """After fresh install, manifest should have v2 format with hashes."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)
            manifest = _read_manifest()

        assert "new-skill" in manifest
        assert "old-skill" in manifest
        # Hashes should be non-empty MD5 strings
        assert len(manifest["new-skill"]) == 32
        assert len(manifest["old-skill"]) == 32

    def test_user_deleted_skill_not_re_added(self, tmp_path):
        """Skill in manifest but not on disk = user deleted it. Don't re-add."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        # old-skill is in manifest (v2 format) but NOT on disk
        old_hash = _dir_hash(bundled / "old-skill")
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "new-skill" in result["copied"]
        assert "old-skill" not in result["copied"]
        assert "old-skill" not in result.get("updated", [])
        assert not (skills_dir / "old-skill").exists()

    def test_unmodified_skill_gets_updated(self, tmp_path):
        """Skill in manifest + on disk + user hasn't modified = update from bundled."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate: user has old version that was synced from an older bundled
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_origin_hash = _dir_hash(user_skill)

        # Record origin hash = hash of what was synced (the old version)
        manifest_file.write_text(f"old-skill:{old_origin_hash}\n")

        # Now bundled has a newer version ("# Old" != "# Old v1")
        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        # Should be updated because user copy matches origin (unmodified)
        assert "old-skill" in result["updated"]
        assert (user_skill / "SKILL.md").read_text() == "# Old"

    def test_user_modified_skill_not_overwritten(self, tmp_path):
        """Skill modified by user should NOT be overwritten even if bundled changed."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate: user had the old version synced, then modified it
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_origin_hash = _dir_hash(user_skill)

        # Record origin hash from what was originally synced
        manifest_file.write_text(f"old-skill:{old_origin_hash}\n")

        # User modifies their copy
        (user_skill / "SKILL.md").write_text("# My custom version")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        # Should NOT update — user modified it
        assert "old-skill" in result["user_modified"]
        assert "old-skill" not in result.get("updated", [])
        assert (user_skill / "SKILL.md").read_text() == "# My custom version"

    def test_unchanged_skill_not_updated(self, tmp_path):
        """Skill in sync (user == bundled == origin) = no action needed."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Copy bundled to user dir (simulating perfect sync state)
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old")
        origin_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{origin_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "old-skill" not in result.get("updated", [])
        assert "old-skill" not in result.get("user_modified", [])
        assert result["skipped"] >= 1

    def test_v1_manifest_migration_sets_baseline(self, tmp_path):
        """v1 manifest entries (no hash) should set baseline from user's current copy."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-create skill on disk
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old modified by user")

        # v1 manifest (no hashes)
        manifest_file.write_text("old-skill\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
            # Should skip (migration baseline set), NOT update
            assert "old-skill" not in result.get("updated", [])
            assert "old-skill" not in result.get("user_modified", [])

            # Now check manifest was upgraded to v2 with user's hash as baseline
            manifest = _read_manifest()
            assert len(manifest["old-skill"]) == 32  # MD5 hash

    def test_v1_migration_then_bundled_update_detected(self, tmp_path):
        """After v1 migration, a subsequent sync should detect bundled updates."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # User has the SAME content as bundled (in sync)
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old")

        # v1 manifest
        manifest_file.write_text("old-skill\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # First sync: migration — sets baseline
            sync_skills(quiet=True)

            # Now change bundled content
            (bundled / "old-skill" / "SKILL.md").write_text("# Old v2 — improved")

            # Second sync: should detect bundled changed + user unmodified → update
            result = sync_skills(quiet=True)

        assert "old-skill" in result["updated"]
        assert (user_skill / "SKILL.md").read_text() == "# Old v2 — improved"

    def test_stale_manifest_entries_cleaned(self, tmp_path):
        """Skills in manifest that no longer exist in bundled dir get cleaned."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("old-skill:abc123\nremoved-skill:def456\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert "removed-skill" in result["cleaned"]
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert "removed-skill" not in manifest

    def test_does_not_overwrite_existing_unmanifested_skill(self, tmp_path):
        """New skill whose name collides with user-created skill = skipped."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# User modified")

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert (user_skill / "SKILL.md").read_text() == "# User modified"

    def test_collision_does_not_poison_manifest(self, tmp_path):
        """Collision with an unmanifested user skill must NOT record bundled_hash.

        Otherwise the next sync compares user_hash against the recorded
        bundled_hash, finds a mismatch, and permanently flags the skill as
        'user-modified' — even though the user never touched a bundled copy.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Pre-existing user skill (e.g. from hub, custom, or leftover) that
        # happens to share a name with a newly bundled skill.
        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)

        # User file must survive (existing invariant).
        assert (user_skill / "SKILL.md").read_text() == (
            "# From hub — unrelated to bundled"
        )

        # Manifest must NOT contain the skill — it was never synced from bundled.
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            manifest = _read_manifest()
        assert "new-skill" not in manifest, (
            "Collision path wrote bundled_hash to the manifest even though "
            "the on-disk copy is unrelated to bundled. This poisons update "
            "detection: the next sync will mark the skill as 'user-modified'."
        )

    def test_collision_does_not_trigger_false_user_modified_on_resync(self, tmp_path):
        """End-to-end: after a collision, a second sync must not flag user_modified.

        Pre-fix bug: first sync wrote bundled_hash to the manifest; second
        sync then diffed user_hash vs bundled_hash, mismatched, and shoved
        the skill into the user_modified bucket forever.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)  # first sync: collision path
            result2 = sync_skills(quiet=True)  # second sync: must not flag

        assert "new-skill" not in result2["user_modified"], (
            "Second sync after a collision falsely flagged the user's skill "
            "as 'user-modified' — the manifest was poisoned on the first sync."
        )

    def test_collision_prints_reset_hint(self, tmp_path, capsys):
        """Non-quiet sync must print a reset hint when a collision is skipped.

        Silent skip hides the fact that a bundled skill shipped but was
        shadowed by the user's local copy. The hint tells the user the
        exact command to take the bundled version instead.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        user_skill = skills_dir / "category" / "new-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# From hub — unrelated to bundled")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=False)

        captured = capsys.readouterr().out
        assert "new-skill" in captured
        assert "hermes skills reset new-skill" in captured

    def test_backfills_official_optional_provenance_for_existing_identical_skill(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# TRL\n"
        )
        (optional_skill / "references").mkdir()
        (optional_skill / "references" / "api.md").write_text("api\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        active = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        active.mkdir(parents=True)
        (active / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# TRL\n"
        )
        (active / "references").mkdir()
        (active / "references" / "api.md").write_text("api\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = sync_skills(quiet=True)

        assert result["optional_provenance_backfilled"] == ["trl-fine-tuning"]
        lock_path = skills_dir / ".hub" / "lock.json"
        data = json.loads(lock_path.read_text())
        entry = data["installed"]["trl-fine-tuning"]
        assert entry["source"] == "official"
        assert entry["identifier"] == "official/mlops/training/trl-fine-tuning"
        assert entry["trust_level"] == "builtin"
        assert entry["install_path"] == "mlops/training/trl-fine-tuning"

    def test_does_not_backfill_optional_provenance_for_modified_skill(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text("# upstream optional\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        active = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        active.mkdir(parents=True)
        (active / "SKILL.md").write_text("# user modified\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = sync_skills(quiet=True)

        assert result["optional_provenance_backfilled"] == []
        assert not (skills_dir / ".hub" / "lock.json").exists()

    def test_repair_official_optional_restores_reorganized_skill_with_backup(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# Official TRL\n"
        )

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        wrong = skills_dir / "mlops" / "trl-fine-tuning"
        wrong.mkdir(parents=True)
        (wrong / "SKILL.md").write_text(
            "---\nname: fine-tuning-with-trl\n---\n# Curator mangled\n"
        )

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = restore_official_optional_skill("fine-tuning-with-trl", restore=True)

        canonical = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        assert result["ok"] is True
        assert result["restored"] == ["trl-fine-tuning"]
        assert result["backed_up"] == ["mlops/trl-fine-tuning"]
        assert "Official TRL" in (canonical / "SKILL.md").read_text()
        assert not wrong.exists()
        assert (Path(result["backup_dir"]) / "mlops" / "trl-fine-tuning" / "SKILL.md").exists()

        data = json.loads((skills_dir / ".hub" / "lock.json").read_text())
        assert data["installed"]["trl-fine-tuning"]["source"] == "official"
        assert data["installed"]["trl-fine-tuning"]["install_path"] == "mlops/training/trl-fine-tuning"

    def test_repair_official_optional_without_restore_does_not_replace_modified_copy(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        optional = tmp_path / "optional-skills"
        optional_skill = optional / "mlops" / "training" / "trl-fine-tuning"
        optional_skill.mkdir(parents=True)
        (optional_skill / "SKILL.md").write_text("# official\n")

        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        canonical = skills_dir / "mlops" / "training" / "trl-fine-tuning"
        canonical.mkdir(parents=True)
        (canonical / "SKILL.md").write_text("# modified\n")

        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._get_optional_dir", return_value=optional):
                result = restore_official_optional_skill("trl-fine-tuning", restore=False)

        assert result["ok"] is True
        assert result["restored"] == []
        assert result["backfilled"] == []
        assert (canonical / "SKILL.md").read_text() == "# modified\n"
        assert not (skills_dir / ".hub" / "lock.json").exists()

    def test_nonexistent_bundled_dir(self, tmp_path):
        with patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "nope"):
            result = sync_skills(quiet=True)
        assert result == {
            "copied": [], "updated": [], "skipped": 0,
            "user_modified": [], "cleaned": [], "suppressed": [], "total_bundled": 0,
            "optional_provenance_backfilled": [],
        }

    def test_failed_copy_does_not_poison_manifest(self, tmp_path):
        """If copytree fails, the skill must NOT be added to the manifest.

        Otherwise the next sync treats it as 'user deleted' and never retries.
        """
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        with self._patches(bundled, skills_dir, manifest_file):
            # Patch copytree to fail for new-skill
            original_copytree = __import__("shutil").copytree

            def failing_copytree(src, dst, *a, **kw):
                if "new-skill" in str(src):
                    raise OSError("Simulated disk full")
                return original_copytree(src, dst, *a, **kw)

            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)

            # new-skill should NOT be in copied (it failed)
            assert "new-skill" not in result["copied"]

            # Critical: new-skill must NOT be in the manifest
            manifest = _read_manifest()
            assert "new-skill" not in manifest, (
                "Failed copy was recorded in manifest — next sync will "
                "treat it as 'user deleted' and never retry"
            )

            # Now run sync again (copytree works this time) — it should retry
            result2 = sync_skills(quiet=True)
            assert "new-skill" in result2["copied"]
            assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

    def test_failed_update_does_not_destroy_user_copy(self, tmp_path):
        """If copytree fails during update, the user's existing copy must survive."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Start with old synced version
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # Patch copytree to fail (rmtree succeeds, copytree fails)
            original_copytree = __import__("shutil").copytree

            def failing_copytree(src, dst, *a, **kw):
                if "old-skill" in str(src):
                    raise OSError("Simulated write failure")
                return original_copytree(src, dst, *a, **kw)

            with patch("shutil.copytree", side_effect=failing_copytree):
                result = sync_skills(quiet=True)

            # old-skill should NOT be in updated (it failed)
            assert "old-skill" not in result.get("updated", [])

            # The skill directory should still exist (rmtree destroyed it
            # but copytree failed to replace it — this is data loss)
            assert user_skill.exists(), (
                "Update failure destroyed user's skill copy without replacing it"
            )

    def test_update_records_new_origin_hash(self, tmp_path):
        """After updating a skill, the manifest should record the new bundled hash."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Start with old synced version
        user_skill = skills_dir / "old-skill"
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("# Old v1")
        old_hash = _dir_hash(user_skill)
        manifest_file.write_text(f"old-skill:{old_hash}\n")

        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)  # updates to "# Old"
            manifest = _read_manifest()

        # New origin hash should match the bundled version
        new_bundled_hash = _dir_hash(bundled / "old-skill")
        assert manifest["old-skill"] == new_bundled_hash
        assert manifest["old-skill"] != old_hash


class TestGetBundledDir:
    def test_env_var_override(self, tmp_path, monkeypatch):
        """HERMES_BUNDLED_SKILLS env var overrides the default path resolution."""
        custom_dir = tmp_path / "custom_skills"
        custom_dir.mkdir()
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(custom_dir))
        assert _get_bundled_dir() == custom_dir

    def test_default_without_env_var(self, monkeypatch):
        """Without the env var, falls back to relative path from __file__."""
        monkeypatch.delenv("HERMES_BUNDLED_SKILLS", raising=False)
        result = _get_bundled_dir()
        assert result.name == "skills"

    def test_env_var_empty_string_ignored(self, monkeypatch):
        """Empty HERMES_BUNDLED_SKILLS should fall back to default."""
        monkeypatch.setenv("HERMES_BUNDLED_SKILLS", "")
        result = _get_bundled_dir()
        assert result.name == "skills"


class TestResetBundledSkill:
    """Covers reset_bundled_skill() — the escape hatch for the 'user-modified' trap."""

    def _setup_bundled(self, tmp_path):
        """Create a minimal bundled skills tree with a single 'google-workspace' skill."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "productivity" / "google-workspace").mkdir(parents=True)
        (bundled / "productivity" / "google-workspace" / "SKILL.md").write_text(
            "---\nname: google-workspace\n---\n# GW v2 (upstream)\n"
        )
        return bundled

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        # Hermetic opt-out check: point HERMES_HOME at the temp tree so
        # reset_bundled_skill's .no-bundled-skills guard never reads the real
        # ~/.hermes. Individual tests override this to exercise opt-out.
        stack.enter_context(patch("tools.skills_sync.HERMES_HOME", skills_dir.parent))
        return stack

    def test_reset_clears_stuck_user_modified_flag(self, tmp_path):
        """The core bug repro: copy-pasted bundled restore doesn't un-stick the flag; reset does."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate the stuck state: user edited the skill on an older bundled version,
        # so manifest has an old origin hash that no longer matches anything on disk.
        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("---\nname: google-workspace\n---\n# GW v2 (upstream)\n")
        # Stale origin_hash — from some prior bundled version. User "restored" by pasting
        # the current bundled contents, so user_hash == current bundled_hash, but manifest
        # still points at the stale hash → treated as user_modified forever.
        manifest_file.write_text("google-workspace:STALEHASH000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            # Sanity check: without reset, sync would flag it user_modified
            pre = sync_skills(quiet=True)
            assert "google-workspace" in pre["user_modified"]

            # Reset (no --restore) should clear the manifest entry and re-baseline
            result = reset_bundled_skill("google-workspace", restore=False)

            assert result["ok"] is True
            assert result["action"] == "manifest_cleared"

            # After reset, the manifest should hold the *current* bundled hash
            manifest_after = _read_manifest()
            expected = _dir_hash(bundled / "productivity" / "google-workspace")
            assert manifest_after["google-workspace"] == expected
        # User's copy was preserved (we didn't delete)
        assert dest.exists()
        assert "GW v2" in (dest / "SKILL.md").read_text()

    def test_reset_restore_replaces_user_copy(self, tmp_path):
        """--restore nukes the user's copy and re-copies the bundled version."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# heavily edited by user\n")
        (dest / "my_custom_file.py").write_text("print('user-added')\n")
        manifest_file.write_text("google-workspace:STALEHASH000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("google-workspace", restore=True)

        assert result["ok"] is True
        assert result["action"] == "restored"
        # User's custom file should be gone
        assert not (dest / "my_custom_file.py").exists()
        # SKILL.md should be the bundled content
        assert "GW v2 (upstream)" in (dest / "SKILL.md").read_text()

    def test_reset_on_opt_out_profile_refuses_and_keeps_manifest(self, tmp_path):
        """An opt-out profile (.no-bundled-skills) takes NO bundled-skill state
        change from reset: sync_skills() is a no-op there, so clearing the
        manifest entry would empty tracking that nothing rebuilds. reset must
        refuse up front and leave the manifest byte-identical."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("google-workspace:abc\n")
        home = tmp_path / "home"
        home.mkdir()
        (home / ".no-bundled-skills").write_text("opted out\n")
        before = manifest_file.read_text()
        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync.HERMES_HOME", home):
                result = reset_bundled_skill("google-workspace", restore=False)
        assert result["ok"] is False
        assert result["action"] == "opted_out"
        assert result["synced"] is None
        assert manifest_file.read_text() == before   # untouched

    def test_reset_surfaces_policy_error_instead_of_false_success(self, tmp_path):
        """When reset's internal re-baseline sync fail-closes (corrupt/missing seed
        policy on a policy-managed profile), reset must NOT report green success:
        the manifest entry was already cleared and tracking was not rebuilt, so the
        user must fix the policy. (The reset UI keys off ok=False.)"""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("---\nname: google-workspace\n---\n# GW\n")
        manifest_file.write_text("google-workspace:abc\n")
        (skills_dir / SEED_POLICY_MARKER).write_text("3\n")   # policy-managed
        with self._patches(bundled, skills_dir, manifest_file):
            with patch("tools.skills_sync._read_seed_policy", return_value=None), \
                    patch("tools.skills_sync._seed_policy_file_present", return_value=True):
                result = reset_bundled_skill("google-workspace", restore=False)
        assert result["ok"] is False
        assert result["action"] == "policy_error"
        assert result["synced"]["policy_error"] is True

    def test_reset_restore_failure_preserves_curator_suppression(self, tmp_path):
        """A failed restore (rmtree of the user copy errors) must NOT have already
        cleared the skill's curator suppression: prune state stays put when the op
        reports 'nothing was changed'. (Regression: the clear was moved to AFTER
        the rmtree succeeds.)"""
        import tools.skill_usage as su
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# user\n")
        manifest_file.write_text("google-workspace:abc\n")
        (skills_dir / ".curator_suppressed").write_text("google-workspace\n")
        with self._patches(bundled, skills_dir, manifest_file):
            with patch.object(su, "_skills_dir", return_value=skills_dir), \
                    patch("tools.skills_sync._rmtree_writable", side_effect=OSError("boom")):
                result = reset_bundled_skill("google-workspace", restore=True)
        assert result["ok"] is False
        assert result["action"] == "not_reset"
        remaining = (skills_dir / ".curator_suppressed").read_text().split()
        assert "google-workspace" in remaining   # suppression untouched on failure

    def test_reset_restore_clears_curator_suppression(self, tmp_path):
        """reset --restore is an explicit un-prune: it must clear the skill's
        curator suppression. Otherwise sync_skills() re-skips it (suppressed names
        are skipped) and the skill is restored on disk but never tracked/updated
        again — exactly what sync_skills' own comment promises restore prevents."""
        import tools.skill_usage as su
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("google-workspace:abc\n")
        (skills_dir / ".curator_suppressed").write_text("google-workspace\n")
        with self._patches(bundled, skills_dir, manifest_file):
            with patch.object(su, "_skills_dir", return_value=skills_dir):
                result = reset_bundled_skill("google-workspace", restore=True)
        assert result["ok"] is True
        supp = skills_dir / ".curator_suppressed"
        remaining = supp.read_text().split() if supp.exists() else []
        assert "google-workspace" not in remaining
        assert "google-workspace" not in (result["synced"].get("suppressed") or [])

    def test_reset_restore_installs_skill_not_in_seed_policy(self, tmp_path):
        """`reset --restore` is an explicit per-skill request: it must install the
        bundled skill even when the seed policy would NOT auto-seed it. Otherwise
        the internal sync (which filters to the allowlist) leaves it uninstalled
        while the function reports 'restored'."""
        bundled = self._setup_bundled(tmp_path)  # bundles google-workspace
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = skills_dir / "productivity" / "google-workspace"

        with self._patches(bundled, skills_dir, manifest_file), patch(
            # active policy that does NOT include google-workspace
            "tools.skills_sync._read_seed_policy",
            return_value={"seed_set": {"something-else"}},
        ):
            result = reset_bundled_skill("google-workspace", restore=True)

            assert result["ok"] is True
            assert result["action"] == "restored"
            assert (dest / "SKILL.md").exists()  # installed despite not seeded
            assert "GW v2 (upstream)" in (dest / "SKILL.md").read_text()
            # honest about not being managed, and no manifest entry (which the
            # next policy-managed sync would only clean — the churn flagged)
            assert "auto-updated" in result["message"]
            assert "google-workspace" not in _read_manifest()
            # the restored copy survives a subsequent normal sync
            sync_skills(quiet=True)
            assert (dest / "SKILL.md").exists()

    def test_reset_nonexistent_skill_errors_gracefully(self, tmp_path):
        """Resetting a skill that's neither bundled nor in the manifest returns a clear error."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("some-hub-skill", restore=False)

        assert result["ok"] is False
        assert result["action"] == "not_in_manifest"
        assert "not a tracked bundled skill" in result["message"]

    def test_reset_restore_when_bundled_removed_upstream(self, tmp_path):
        """If a skill was removed upstream, --restore should fail with a clear message."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        dest = skills_dir / "productivity" / "ghost-skill"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("---\nname: ghost-skill\n---\n# Ghost\n")
        manifest_file.write_text("ghost-skill:OLDHASH00000000000000000000000000\n")

        with self._patches(bundled, skills_dir, manifest_file):
            result = reset_bundled_skill("ghost-skill", restore=True)

        assert result["ok"] is False
        assert result["action"] == "bundled_missing"

    def test_reset_no_op_when_already_clean(self, tmp_path):
        """If manifest has skill but user copy is in-sync, reset still safely clears + re-baselines."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        # Simulate a clean state — do a fresh sync first
        with self._patches(bundled, skills_dir, manifest_file):
            sync_skills(quiet=True)
            pre_manifest = _read_manifest()
            assert "google-workspace" in pre_manifest

            result = reset_bundled_skill("google-workspace", restore=False)

            assert result["ok"] is True
            assert result["action"] == "manifest_cleared"
            # Manifest entry still present (re-baselined), user copy still present
            post_manifest = _read_manifest()
            assert "google-workspace" in post_manifest
        assert (skills_dir / "productivity" / "google-workspace" / "SKILL.md").exists()

    def test_reset_restore_succeeds_on_readonly_nix_tree(self, tmp_path):
        """#34972: --restore must succeed even when the user copy is a fully
        read-only tree (r-xr-xr-x dirs + files), as produced by copying a
        Nix-store source. The manifest is re-baselined and bundled re-copied."""
        import os
        import stat

        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        sub = dest / "references"
        sub.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# user version\n")
        (sub / "ref.md").write_text("# nested ref\n")
        manifest_file.write_text(
            "google-workspace:STALEHASH000000000000000000000000\n"
        )

        # Read-only files AND directories — the real Nix-store case.
        ro_dir = (
            stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
            | stat.S_IROTH | stat.S_IXOTH
        )
        os.chmod(sub / "ref.md", stat.S_IREAD)
        os.chmod(dest / "SKILL.md", stat.S_IREAD)
        os.chmod(sub, ro_dir)
        os.chmod(dest, ro_dir)

        try:
            with self._patches(bundled, skills_dir, manifest_file):
                result = reset_bundled_skill("google-workspace", restore=True)

            assert result["ok"] is True
            assert result["action"] == "restored"
            # Bundled version was re-copied over the (deleted) user copy.
            assert "upstream" in (dest / "SKILL.md").read_text()
            # The read-only nested user dir/file was fully removed, not left behind.
            assert not (sub / "ref.md").exists()
            # sync ran and re-copied the skill (not stuck in limbo).
            assert "google-workspace" in result["synced"]["copied"]
        finally:
            # Restore perms so tmp_path teardown can remove anything left.
            for p in (sub, dest):
                if p.exists():
                    os.chmod(p, stat.S_IRWXU)

    def test_reset_restore_preserves_manifest_on_rmtree_failure(self, tmp_path):
        """#34972: when the user copy genuinely cannot be removed, the manifest
        entry must NOT be deleted — otherwise the skill enters a limbo state
        where future syncs silently skip it forever."""
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"

        dest = skills_dir / "productivity" / "google-workspace"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text("# user version\n")
        manifest_file.write_text(
            "google-workspace:STALEHASH000000000000000000000000\n"
        )

        # Simulate an unremovable tree (e.g. a busy mountpoint or a path even
        # chmod can't rescue) by making the removal helper raise.
        def _boom(_path):
            raise PermissionError(13, "Permission denied")

        with self._patches(bundled, skills_dir, manifest_file), patch(
            "tools.skills_sync._rmtree_writable", side_effect=_boom
        ):
            result = reset_bundled_skill("google-workspace", restore=True)

        # Restore failed, and the manifest must be left untouched.
        assert result["ok"] is False
        assert result["action"] == "not_reset"
        assert "Manifest entry preserved" in result["message"]
        manifest_after = manifest_file.read_text()
        assert "google-workspace" in manifest_after
        # User copy is still on disk (we changed nothing).
        assert (dest / "SKILL.md").exists()


class TestNoBundledSkillsOptOut:
    """The .no-bundled-skills marker makes sync_skills() a no-op.

    This is what `hermes profile create --no-skills` (named profiles) and the
    installer's `--no-skills` flag (default ~/.hermes) rely on so bundled
    skills are never seeded at install time NOR re-injected by `hermes update`.
    """

    def _setup_bundled(self, tmp_path):
        bundled = tmp_path / "bundled"
        skill = bundled / "category" / "new-skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: new-skill\n---\nbody\n")
        return bundled

    def test_marker_skips_sync(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        (hermes_home / ".no-bundled-skills").write_text("opted out\n")

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", hermes_home):
            result = sync_skills(quiet=True)

        # Opt-out signalled, nothing copied, nothing written to disk.
        assert result["skipped_opt_out"] is True
        assert result["copied"] == []
        assert result["total_bundled"] == 0
        assert not (skills_dir / "category" / "new-skill" / "SKILL.md").exists()

    def test_no_marker_seeds_normally(self, tmp_path):
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        # No marker written.

        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", hermes_home):
            result = sync_skills(quiet=True)

        assert result.get("skipped_opt_out") is not True
        assert "new-skill" in result["copied"]
        assert (skills_dir / "category" / "new-skill" / "SKILL.md").exists()


class TestOptOutToggleAndRemove:
    """`hermes skills opt-out/opt-in` core: marker toggle + safe removal."""

    def _setup_bundled(self, tmp_path):
        bundled = tmp_path / "bundled"
        for n in ("alpha", "beta"):
            d = bundled / n
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {n}\n---\nbody {n}\n")
        return bundled

    def test_marker_toggle(self, tmp_path):
        from tools.skills_sync import (
            set_bundled_skills_opt_out, is_bundled_skills_opt_out,
        )
        home = tmp_path / "home"
        home.mkdir()
        with patch("tools.skills_sync.HERMES_HOME", home):
            assert is_bundled_skills_opt_out() is False
            r = set_bundled_skills_opt_out(True)
            assert r["ok"] and r["changed"]
            assert is_bundled_skills_opt_out() is True
            # idempotent
            r2 = set_bundled_skills_opt_out(True)
            assert r2["ok"] and r2["changed"] is False
            # opt back in
            r3 = set_bundled_skills_opt_out(False)
            assert r3["ok"] and r3["changed"]
            assert is_bundled_skills_opt_out() is False

    def test_remove_keeps_user_modified(self, tmp_path):
        from tools.skills_sync import (
            sync_skills, remove_pristine_bundled_skills,
        )
        bundled = self._setup_bundled(tmp_path)
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        home = tmp_path / "home"
        home.mkdir()
        with patch("tools.skills_sync._get_bundled_dir", return_value=bundled), \
             patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"), \
             patch("tools.skills_sync.SKILLS_DIR", skills_dir), \
             patch("tools.skills_sync.MANIFEST_FILE", manifest_file), \
             patch("tools.skills_sync.HERMES_HOME", home):
            sync_skills(quiet=True)
            # User edits 'beta'
            (skills_dir / "beta" / "SKILL.md").write_text("---\nname: beta\n---\nEDITED\n")
            # A hand-written, non-bundled skill must also survive.
            (skills_dir / "mine").mkdir()
            (skills_dir / "mine" / "SKILL.md").write_text("---\nname: mine\n---\nlocal\n")

            preview = remove_pristine_bundled_skills(dry_run=True)
            assert "alpha" in preview["removed"]
            assert "beta" not in preview["removed"]

            result = remove_pristine_bundled_skills(dry_run=False)
            assert "alpha" in result["removed"]
            assert not (skills_dir / "alpha").exists()
            # user-modified bundled skill kept
            assert (skills_dir / "beta" / "SKILL.md").exists()
            assert "EDITED" in (skills_dir / "beta" / "SKILL.md").read_text()
            # non-bundled local skill never considered
            assert (skills_dir / "mine" / "SKILL.md").exists()


class TestUpdateBackupRecovery:
    """Regression tests for backup handling in the bundled-update path."""

    def _setup(self, tmp_path, bundled_text="# Old v2 (updated)"):
        """Bundled dir with one flat skill, plus user dirs."""
        bundled = tmp_path / "bundled_skills"
        (bundled / "old-skill").mkdir(parents=True)
        (bundled / "old-skill" / "SKILL.md").write_text(bundled_text)
        skills_dir = tmp_path / "user_skills"
        skills_dir.mkdir()
        manifest_file = skills_dir / ".bundled_manifest"
        return bundled, skills_dir, manifest_file

    def _patches(self, bundled, skills_dir, manifest_file):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        return stack

    def _seed_synced_copy(self, skills_dir, manifest_file, text="# Old v1"):
        """User copy of old-skill whose hash matches the manifest origin."""
        dest = skills_dir / "old-skill"
        dest.mkdir(parents=True)
        (dest / "SKILL.md").write_text(text)
        with patch("tools.skills_sync.MANIFEST_FILE", manifest_file):
            _write_manifest({"old-skill": _dir_hash(dest)})
        return dest

    def test_stale_backup_does_not_poison_failed_update(self, tmp_path):
        """A leftover .bak must not nest the live copy or corrupt restore."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)

        stale = skills_dir / "old-skill.bak"
        stale.mkdir()
        (stale / "SKILL.md").write_text("# stale junk from an earlier failure")

        def _boom(src, dst, **kwargs):
            raise OSError("simulated copy failure")

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync.shutil.copytree", side_effect=_boom):
            sync_skills(quiet=True)

        assert (dest / "SKILL.md").read_text() == "# Old v1"
        assert not (dest / "old-skill").exists()
        assert not (skills_dir / "old-skill.bak").exists()

    def test_orphaned_backup_is_recovered_not_treated_as_deleted(self, tmp_path):
        """Crash between move and copytree must not lose the skill."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)
        shutil.move(str(dest), str(skills_dir / "old-skill.bak"))

        with self._patches(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)

        assert (dest / "SKILL.md").exists()
        assert (dest / "SKILL.md").read_text() == "# Old v2 (updated)"
        assert "old-skill" in result["updated"]
        assert not (skills_dir / "old-skill.bak").exists()

    def test_partial_copy_failure_restores_original(self, tmp_path):
        """A half-written dest must not block restore-on-failure."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        dest = self._seed_synced_copy(skills_dir, manifest_file)

        def _partial_then_fail(src, dst, **kwargs):
            Path(dst).mkdir(parents=True, exist_ok=True)
            (Path(dst) / "PARTIAL").write_text("half-written")
            raise OSError("simulated failure mid-copy")

        with self._patches(bundled, skills_dir, manifest_file), \
                patch("tools.skills_sync.shutil.copytree", side_effect=_partial_then_fail):
            sync_skills(quiet=True)

        assert (dest / "SKILL.md").read_text() == "# Old v1"
        assert not (dest / "PARTIAL").exists()
        assert not (skills_dir / "old-skill.bak").exists()

        with self._patches(bundled, skills_dir, manifest_file):
            result2 = sync_skills(quiet=True)
        assert "old-skill" in result2["updated"]
        assert result2["user_modified"] == []


class TestSeedTimeCuration:
    """Policy-driven seed allowlist over upstream-verbatim skills/ (this PR does
    not rename/de-brand). The policy is a pure allowlist deciding which ids a NEW
    profile seeds (existing profiles are left untouched)."""

    def _setup(self, tmp_path):
        bundled = tmp_path / "bundled_skills"
        # seeded (in allowlist)
        (bundled / "cat" / "keep-me").mkdir(parents=True)
        (bundled / "cat" / "keep-me" / "SKILL.md").write_text("---\nname: keep-me\n---\n# Keep")
        # NOT in allowlist → filtered out
        (bundled / "cat" / "drop-me").mkdir(parents=True)
        (bundled / "cat" / "drop-me" / "SKILL.md").write_text("---\nname: drop-me\n---\n# Drop")
        skills_dir = tmp_path / "user_skills"
        manifest_file = skills_dir / ".bundled_manifest"
        return bundled, skills_dir, manifest_file

    def _policy(self):
        return {"seed_set": {"keep-me"}}

    def _patches(self, bundled, skills_dir, manifest_file, policy):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        stack.enter_context(patch("tools.skills_sync._read_seed_policy", return_value=policy))
        return stack

    def test_allowlist_filters_non_seeded(self, tmp_path):
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        assert "keep-me" in result["copied"]
        assert "drop-me" not in result["copied"]
        assert not (skills_dir / "cat" / "drop-me").exists()
        assert (skills_dir / "cat" / "keep-me" / "SKILL.md").exists()
        assert result["not_seeded"] == 1            # drop-me
        assert result["total_bundled"] == 2

    def test_idempotent_second_sync_copies_nothing(self, tmp_path):
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            sync_skills(quiet=True)
            result2 = sync_skills(quiet=True)
        assert result2["copied"] == []
        assert result2["updated"] == []

    def test_source_junk_not_flagged_user_modified(self, tmp_path):
        """A seeded skill whose SOURCE carries __pycache__/.pyc (copytree ignores
        them, so the dest lacks them) must NOT be flagged user-modified on the
        second sync — _dir_hash skips the same patterns."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        pc = bundled / "cat" / "keep-me" / "scripts" / "__pycache__"
        pc.mkdir(parents=True)
        (pc / "run.cpython-311.pyc").write_bytes(b"\x00junk")
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            sync_skills(quiet=True)
            result2 = sync_skills(quiet=True)
        assert result2["user_modified"] == []   # not wrongly frozen
        # and the junk never reached the profile
        assert not (skills_dir / "cat" / "keep-me" / "scripts" / "__pycache__").exists()

    def test_no_policy_seeds_everything(self, tmp_path):
        """policy=None → upstream behaviour: every bundled skill seeded."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches(bundled, skills_dir, manifest_file, None):
            result = sync_skills(quiet=True)
        assert set(result["copied"]) == {"keep-me", "drop-me"}
        assert result["not_seeded"] == 0

    def test_new_profile_writes_policy_marker(self, tmp_path):
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            sync_skills(quiet=True)
        assert (skills_dir / SEED_POLICY_MARKER).exists()

    def test_marked_profile_keeps_filtering_on_resync(self, tmp_path):
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            sync_skills(quiet=True)            # new profile -> filters + writes marker
            result2 = sync_skills(quiet=True)  # marker present -> still filters
        assert result2["not_seeded"] == 1      # drop-me stays filtered
        assert not (skills_dir / "cat" / "drop-me").exists()

    def test_crash_between_marker_and_manifest_stays_filtered(self, tmp_path):
        """Crash-safety: marker is written before the manifest, so a crash that
        leaves 'marker present, manifest absent' is read as a fresh profile on
        the next sync and the policy is re-applied (never silently flips to the
        unfiltered path)."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        (skills_dir / SEED_POLICY_MARKER).write_text("1\n")  # marker, but NO manifest
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        assert result["not_seeded"] == 1                 # drop-me still filtered
        assert not (skills_dir / "cat" / "drop-me").exists()

    def test_marker_write_failure_skips_manifest_so_next_sync_reseeds(self, tmp_path):
        """If the policy marker can't be persisted, the manifest must NOT be
        written either. Otherwise the next sync sees 'manifest, no marker',
        classifies the profile as pre-policy, and permanently seeds the full
        un-curated set. Instead the profile stays manifest-less and re-seeds
        under policy on the next run."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        # Block the marker write: a directory at .seed_policy makes write_text()
        # raise IsADirectoryError (an OSError subclass), simulating a failed write.
        (skills_dir / SEED_POLICY_MARKER).mkdir()
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        # Filtering still happened (curated set copied), but the manifest was NOT
        # persisted — so the profile cannot be misread as pre-policy next time.
        assert "keep-me" in result["copied"]
        assert result.get("not_seeded") == 1
        assert not manifest_file.exists()

        # Recovery: clear the blocker; the next sync writes marker + manifest and
        # the profile stays curated — drop-me is never seeded.
        import shutil
        shutil.rmtree(skills_dir / SEED_POLICY_MARKER)
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            sync_skills(quiet=True)
        assert manifest_file.exists()
        assert (skills_dir / SEED_POLICY_MARKER).is_file()
        assert not (skills_dir / "cat" / "drop-me").exists()

    def _patches_no_policy_but_present(self, bundled, skills_dir, manifest_file):
        """Patches simulating a policy FILE that exists but cannot be parsed:
        _read_seed_policy() -> None, _seed_policy_file_present() -> True."""
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch("tools.skills_sync._get_bundled_dir", return_value=bundled))
        stack.enter_context(patch("tools.skills_sync._get_optional_dir", return_value=bundled.parent / "optional-skills"))
        stack.enter_context(patch("tools.skills_sync.SKILLS_DIR", skills_dir))
        stack.enter_context(patch("tools.skills_sync.MANIFEST_FILE", manifest_file))
        stack.enter_context(patch("tools.skills_sync._read_seed_policy", return_value=None))
        stack.enter_context(patch("tools.skills_sync._seed_policy_file_present", return_value=True))
        return stack

    def test_corrupt_policy_file_fails_closed_no_seed(self, tmp_path):
        """A policy file present but unparseable must NOT fall through to seeding
        the full un-curated set. New profile: seed nothing, write no
        marker/manifest, return policy_error."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        with self._patches_no_policy_but_present(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
        assert result.get("policy_error") is True
        assert result["copied"] == []
        assert not (skills_dir / "cat" / "keep-me").exists()
        assert not (skills_dir / "cat" / "drop-me").exists()        # NOT seeded
        assert not (skills_dir / SEED_POLICY_MARKER).exists()
        assert not manifest_file.exists()

    def test_corrupt_policy_does_not_affect_pre_policy_profile(self, tmp_path):
        """A pre-policy profile (manifest, no marker) was going to seed everything
        anyway, so a corrupt policy changes nothing for it — no fail-closed, no
        regression."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("preexisting:abc\n")   # existing, NO marker
        with self._patches_no_policy_but_present(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
        assert result.get("policy_error") is not True
        assert "drop-me" in result["copied"]            # seeds everything (upstream)
        assert (skills_dir / "cat" / "drop-me").exists()

    def test_corrupt_policy_with_skill_on_disk_no_manifest_fails_closed(self, tmp_path):
        """Regression: a corrupt policy makes _seed_set EMPTY, so every on-disk
        skill reads as 'not in seed set'. A just-seeded / partial-fallback
        profile (a seeded skill on disk, NO manifest, NO marker) must NOT be
        misread as an existing pre-policy one and seed the full un-curated set —
        it must fail closed. profile_existed falls back to the manifest signal
        when the policy can't be read."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        # A seeded skill already on disk (just-seeded-marker-failed, or a partial
        # python-free fallback copy), but NO manifest and NO marker.
        d = skills_dir / "cat" / "keep-me"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: keep-me\n---\n# Keep")
        with self._patches_no_policy_but_present(bundled, skills_dir, manifest_file):
            result = sync_skills(quiet=True)
        assert result.get("policy_error") is True
        assert result["copied"] == []
        assert not (skills_dir / "cat" / "drop-me").exists()   # NOT seeded
        assert not manifest_file.exists()
        assert not (skills_dir / SEED_POLICY_MARKER).exists()

    def test_managed_profile_with_vanished_policy_fails_closed(self, tmp_path):
        """A policy-managed profile (.seed_policy marker) whose policy file has
        VANISHED (downgrade, or a packaging wrapper that shipped skills/ without
        the co-located config/) must fail closed — it must NOT silently expand
        the curated set to every bundled skill. Symmetric with the corrupt case;
        only a genuinely pre-policy profile (no marker) stays on seed-everything.
        (_patches sets _read_seed_policy→None; _seed_policy_file_present runs for
        real and is False because no config/ exists under the temp bundle.)"""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("keep-me:abc\n")
        (skills_dir / SEED_POLICY_MARKER).write_text("3\n")   # marker => managed
        with self._patches(bundled, skills_dir, manifest_file, None):
            result = sync_skills(quiet=True)
        assert result.get("policy_error") is True
        assert result["copied"] == []
        assert not (skills_dir / "cat" / "drop-me").exists()  # did NOT expand to full set

    def test_old_fallback_profile_without_manifest_not_filtered(self, tmp_path):
        """An old installer fallback can copy skills WITHOUT writing a manifest.
        Such a profile (it carries a non-seeded skill on disk) must be recognised
        as existing — not misread as new and pulled onto the policy path."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        # simulate old fallback: both skills on disk, NO manifest, NO marker
        for n in ("keep-me", "drop-me"):
            d = skills_dir / "cat" / n
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"---\nname: {n}\n---\n# {n}")
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        assert result["not_seeded"] == 0                       # not filtered
        assert not (skills_dir / SEED_POLICY_MARKER).exists()  # not made policy-managed

    def test_user_local_skill_does_not_trigger_full_seed(self, tmp_path):
        """Regression for the `hermes skills opt-in --sync` policy bypass: a
        profile that carries only the user's OWN local/hub skill (NOT a bundled
        skill), with no manifest and no .seed_policy marker, must seed the CURATED
        set — it must NOT be misread as a pre-policy profile and seed the full
        un-curated set. Only a BUNDLED-but-unseeded on-disk skill is pre-policy
        evidence (contrast test_old_fallback_profile_without_manifest_not_filtered);
        a user skill is legitimately absent from the seed set."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        # User's own local skill — absent from the bundled tree, no manifest, no
        # marker (e.g. an opt-out profile the user added a skill to, then ran
        # `hermes skills opt-in --sync`).
        mine = skills_dir / "mine"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("---\nname: mine\n---\n# Local")
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        # Curated, NOT the full un-curated set.
        assert "keep-me" in result["copied"]
        assert "drop-me" not in result["copied"]
        assert not (skills_dir / "cat" / "drop-me").exists()
        assert result["not_seeded"] == 1
        # The user's own skill is left untouched, and the profile is now managed.
        assert (mine / "SKILL.md").exists()
        assert (skills_dir / SEED_POLICY_MARKER).exists()

    def test_name_collision_with_bundled_does_not_trigger_full_seed(self, tmp_path):
        """A user/hub skill whose frontmatter name COLLIDES with a bundled-but-
        unseeded skill, but lives at a non-canonical path, must NOT count as
        pre-policy evidence — the profile stays curated. Pre-policy detection
        requires the canonical seeded path, not just a name match (review on the
        seed-policy bypass: name match alone let a colliding user skill flip the
        profile onto the full un-curated seed)."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        # User skill whose frontmatter name collides with bundled `drop-me`, at a
        # DIFFERENT (non-canonical) path.
        d = skills_dir / "my-own" / "drop-me-clone"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: drop-me\n---\n# coincidental name collision")
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        # Curated, NOT the full un-curated set — the real bundled drop-me is not seeded.
        assert "keep-me" in result["copied"]
        assert "drop-me" not in result["copied"]
        assert not (skills_dir / "cat" / "drop-me").exists()
        assert result["not_seeded"] == 1
        assert (d / "SKILL.md").exists()                   # user skill left untouched
        assert (skills_dir / SEED_POLICY_MARKER).exists()  # curated + managed

    def test_existing_profile_not_filtered(self, tmp_path):
        """A profile that predates the policy (manifest present, no marker) is
        left on the upstream path: every bundled skill seeds, no filtering, no
        cleanup — the 'don't migrate existing' boundary."""
        bundled, skills_dir, manifest_file = self._setup(tmp_path)
        skills_dir.mkdir(parents=True)
        manifest_file.write_text("preexisting:abc\n")   # existing profile, NO .seed_policy marker
        with self._patches(bundled, skills_dir, manifest_file, self._policy()):
            result = sync_skills(quiet=True)
        assert "drop-me" in result["copied"]            # not filtered
        assert result["not_seeded"] == 0
        assert (skills_dir / "cat" / "drop-me").exists()
        assert not (skills_dir / SEED_POLICY_MARKER).exists()  # unmanaged profile stays unmarked


class TestSeedPolicyHelpers:
    def test_read_seed_policy_builds_set(self, tmp_path):
        pol = tmp_path / "skill_seed_policy.json"
        pol.write_text(json.dumps({"seed": ["a", "b"]}))
        with patch.dict("os.environ", {"HERMES_SEED_POLICY": str(pol)}):
            data = _read_seed_policy()
        assert data is not None
        assert data["seed_set"] == {"a", "b"}

    def test_read_seed_policy_invalid_returns_none(self, tmp_path):
        """A malformed candidate is skipped; with no other valid candidate -> None."""
        pol = tmp_path / "bad.json"
        pol.write_text("{not valid json")
        with patch.dict("os.environ", {"HERMES_SEED_POLICY": str(pol)}), \
                patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "skills"):
            assert _read_seed_policy() is None

    def test_read_seed_policy_skips_malformed_then_uses_colocated(self, tmp_path):
        """A malformed env candidate falls through to the policy co-located with
        the bundled skills (not a hard return None on the first failure)."""
        bad = tmp_path / "bad.json"
        bad.write_text("{nope")
        bundle = tmp_path / "bundle"
        (bundle / "config").mkdir(parents=True)
        (bundle / "skills").mkdir(parents=True)
        (bundle / "config" / "skill_seed_policy.json").write_text(
            json.dumps({"seed": ["x"]})
        )
        with patch.dict("os.environ", {"HERMES_SEED_POLICY": str(bad)}), \
                patch("tools.skills_sync._get_bundled_dir", return_value=bundle / "skills"):
            data = _read_seed_policy()
        assert data is not None and data["seed_set"] == {"x"}

    def test_resolve_seed_target_allowlist(self, tmp_path):
        bundled = tmp_path / "b"
        (bundled / "cat" / "s").mkdir(parents=True)
        policy = {"seed_set": {"s"}}
        r = _resolve_seed_target("s", bundled / "cat" / "s", bundled, policy)
        assert r is not None and r[0] == "s"
        assert _resolve_seed_target("other", bundled / "cat" / "s", bundled, policy) is None

    def test_resolve_seed_target_none_policy_passes_through(self, tmp_path):
        bundled = tmp_path / "b"
        (bundled / "cat" / "s").mkdir(parents=True)
        r = _resolve_seed_target("anything", bundled / "cat" / "s", bundled, None)
        assert r is not None and r[0] == "anything"

    def test_read_seed_policy_rejects_malformed_schema(self, tmp_path):
        """A parseable-but-malformed policy (no/non-list 'seed', bad mode) is
        rejected like a corrupt one, so the caller fails closed instead of
        silently seeding 0 / garbage."""
        for bad in ('{}', '{"seed": "keep"}', '{"seed": [1, 2]}',
                    '{"mode": "deny", "seed": ["a"]}',
                    '{"seed": ["a"], "runtime_required": "a"}'):
            pol = tmp_path / "p.json"
            pol.write_text(bad)
            with patch.dict("os.environ", {"HERMES_SEED_POLICY": str(pol)}), \
                    patch("tools.skills_sync._get_bundled_dir", return_value=tmp_path / "skills"):
                assert _read_seed_policy() is None, bad
