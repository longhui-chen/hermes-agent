"""Tests for the `hermes memory reset` CLI command.

Covers:
- Reset both stores (MEMORY.md + USER.md)
- Reset individual stores (--target memory / --target user)
- Skip confirmation with --yes
- Graceful handling when no memory files exist
- Profile-scoped reset (uses HERMES_HOME)
"""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    """Set up a fake HERMES_HOME with memory files."""
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Create sample memory files
    (memories / "MEMORY.md").write_text(
        "§\nHermes repo is at ~/.hermes/hermes-agent\n§\nUser prefers dark themes",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text(
        "§\nUser is Teknium\n§\nTimezone: US Pacific",
        encoding="utf-8",
    )
    return hermes_home, memories


def _run_memory_reset(target="all", yes=False, monkeypatch=None, confirm_input="no"):
    """Invoke the memory reset logic from cmd_memory in main.py.

    Simulates what happens when `hermes memory reset` is run.
    """
    from hermes_constants import get_hermes_home
    from tools.memory_tool import curated_memory_has_state, reset_curated_memory

    mem_dir = get_hermes_home() / "memories"
    files_to_reset = []
    if target in {"all", "memory"}:
        files_to_reset.append(("MEMORY.md", "agent notes"))
    if target in {"all", "user"}:
        files_to_reset.append(("USER.md", "user profile"))

    existing = []
    if target == "all" and curated_memory_has_state("all"):
        existing = files_to_reset
    else:
        for f, desc in files_to_reset:
            item = "memory" if f == "MEMORY.md" else "user"
            if curated_memory_has_state(item):
                existing.append((f, desc))
    if not existing:
        return "nothing"

    if not yes:
        if confirm_input != "yes":
            return "cancelled"

    reset_curated_memory(target)

    return "deleted"


class TestMemoryReset:
    """Tests for `hermes memory reset` subcommand."""

    def test_reset_all_with_yes_flag(self, memory_env):
        """--yes flag should skip confirmation and delete both files."""
        hermes_home, memories = memory_env
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_memory_only(self, memory_env):
        """--target memory should only delete MEMORY.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="memory", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_user_only(self, memory_env):
        """--target user should only delete USER.md."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="user", yes=True)
        assert result == "deleted"
        assert (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_cleanup_pending_exits_nonzero_without_success_message(
        self, memory_env, monkeypatch, capsys
    ):
        import tools.memory_tool as memory_tool
        from hermes_cli.main import cmd_memory

        monkeypatch.setattr(
            memory_tool,
            "reset_curated_memory",
            lambda _target: {
                "deleted": ["MEMORY.md"],
                "targets": ["memory"],
                "status": "cleanup_pending",
            },
        )

        with pytest.raises(SystemExit) as raised:
            cmd_memory(
                SimpleNamespace(memory_command="reset", target="memory", yes=True)
            )

        assert raised.value.code == 1
        output = capsys.readouterr().out
        assert "cleanup is still pending" in output
        assert "Memory reset complete" not in output

    def test_unsupported_reset_exits_nonzero_before_any_mutation(
        self, memory_env, monkeypatch, capsys
    ):
        import tools.memory_tool as memory_tool
        from hermes_cli.main import cmd_memory

        reset_called = False

        def unexpected_reset(_target):
            nonlocal reset_called
            reset_called = True

        monkeypatch.setattr(
            memory_tool, "portable_memory_reset_supported", lambda: False
        )
        monkeypatch.setattr(memory_tool, "reset_curated_memory", unexpected_reset)

        with pytest.raises(SystemExit) as raised:
            cmd_memory(
                SimpleNamespace(memory_command="reset", target="memory", yes=True)
            )

        assert raised.value.code == 1
        assert reset_called is False
        output = capsys.readouterr().out
        assert "unsupported" in output
        assert "No memory files were changed" in output
        assert "Memory reset complete" not in output

    def test_reset_no_files_exist(self, tmp_path, monkeypatch):
        """Should return 'nothing' when no memory files exist."""
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "memories").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"

    def test_reset_confirmation_denied(self, memory_env):
        """Without --yes and without typing 'yes', should be cancelled."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="no")
        assert result == "cancelled"
        # Files should still exist
        assert (memories / "MEMORY.md").exists()
        assert (memories / "USER.md").exists()

    def test_reset_confirmation_accepted(self, memory_env):
        """Typing 'yes' should proceed with deletion."""
        hermes_home, memories = memory_env

        result = _run_memory_reset(target="all", yes=False, confirm_input="yes")
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_profile_scoped(self, tmp_path, monkeypatch):
        """Reset should work on the active profile's HERMES_HOME."""
        profile_home = tmp_path / "profiles" / "myprofile"
        memories = profile_home / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("profile memory", encoding="utf-8")
        (memories / "USER.md").write_text("profile user", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()
        assert not (memories / "USER.md").exists()

    def test_reset_partial_files(self, memory_env):
        """Reset should work when only one memory file exists."""
        hermes_home, memories = memory_env
        (memories / "USER.md").unlink()

        result = _run_memory_reset(target="all", yes=True)
        assert result == "deleted"
        assert not (memories / "MEMORY.md").exists()

    def test_reset_cleans_import_recovery_when_canonical_is_missing(
        self, tmp_path, monkeypatch
    ):
        import hashlib

        from tools.memory_tool import MemoryStore

        home = tmp_path / ".hermes"
        memories = home / "memories"
        memories.mkdir(parents=True)
        (memories / "MEMORY.md").write_text("private old memory", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        result = MemoryStore(memory_char_limit=100, user_char_limit=100).import_replace(
            target="memory",
            entries=["imported memory"],
            import_id="cli-reset-import",
            payload_sha256=hashlib.sha256(b"cli-reset-import").hexdigest(),
        )
        (memories / "MEMORY.md").unlink()

        reset = _run_memory_reset(target="memory", yes=True)

        assert reset == "deleted"
        assert not Path(result["backup_path"]).exists()
        assert not Path(result["recovery_path"]).exists()
        assert not list((memories / ".imports").glob("*.json"))

    def test_reset_empty_memories_dir(self, tmp_path, monkeypatch):
        """No memories dir at all should report nothing."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        # No memories dir
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # The memories dir won't exist; get_hermes_home() / "memories" won't have files
        result = _run_memory_reset(target="all", yes=True)
        assert result == "nothing"

    def test_reset_all_cleans_unclassified_receipt(self, tmp_path, monkeypatch):
        home = tmp_path / ".hermes"
        imports = home / "memories" / ".imports"
        imports.mkdir(parents=True)
        receipt = imports / "corrupt.json"
        receipt.write_text("{", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        assert _run_memory_reset(target="all", yes=True) == "deleted"
        assert not receipt.exists()

    def test_cli_reset_all_labels_only_unclassified_recovery_state(
        self, tmp_path, monkeypatch, capsys
    ):
        from hermes_cli.main import cmd_memory

        home = tmp_path / ".hermes"
        imports = home / "memories" / ".imports"
        imports.mkdir(parents=True)
        (imports / "corrupt.json").write_text("{", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        cmd_memory(SimpleNamespace(
            memory_command="reset", target="all", yes=True
        ))

        output = capsys.readouterr().out
        assert "managed import recovery state" in output
        assert "Deleted MEMORY.md" not in output
        assert "Deleted USER.md" not in output

    def test_cli_reset_all_does_not_read_sparse_oversize_receipt(
        self, tmp_path, monkeypatch
    ):
        import tools.memory_tool as memory_tool
        from hermes_cli.main import cmd_memory

        home = tmp_path / ".hermes"
        imports = home / "memories" / ".imports"
        imports.mkdir(parents=True)
        receipt = imports / "oversize.json"
        with receipt.open("wb") as handle:
            handle.truncate(65 << 20)
        monkeypatch.setenv("HERMES_HOME", str(home))
        original_read = memory_tool.os.read
        reads = []

        def record_read(fd, size):
            reads.append(size)
            return original_read(fd, size)

        monkeypatch.setattr(memory_tool.os, "read", record_read)

        cmd_memory(SimpleNamespace(
            memory_command="reset", target="all", yes=True
        ))

        assert not receipt.exists()
        assert reads == []
