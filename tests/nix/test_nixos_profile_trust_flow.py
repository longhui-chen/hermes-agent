import importlib.util
import os
import stat
from pathlib import Path

import pytest


_SOURCE = Path(__file__).parents[2] / "nix" / "safeProfileDirs.py"
_SPEC = importlib.util.spec_from_file_location("safe_profile_dirs", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
safe_profile_dirs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(safe_profile_dirs)


def test_safe_profile_directory_flow_creates_exact_managed_chain(tmp_path):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    current = home.stat()

    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )

    for path in (memories / ".imports", memories / ".imports" / "backups"):
        opened = path.stat(follow_symlinks=False)
        assert opened.st_uid == current.st_uid
        assert opened.st_gid == current.st_gid
        assert opened.st_mode & 0o7777 == 0o2770


@pytest.mark.parametrize("leaf", ["home", "memories", ".imports", "backups"])
def test_safe_profile_directory_flow_never_follows_transaction_symlink(
    tmp_path, leaf
):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    imports = memories / ".imports"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    if leaf == "home":
        home.symlink_to(outside, target_is_directory=True)
    elif leaf == "memories":
        home.mkdir()
        memories.symlink_to(outside, target_is_directory=True)
    elif leaf == ".imports":
        home.mkdir()
        memories.mkdir()
        imports.symlink_to(outside, target_is_directory=True)
    else:
        home.mkdir()
        memories.mkdir()
        imports.mkdir()
        (imports / "backups").symlink_to(outside, target_is_directory=True)
    before = outside.stat(follow_symlinks=False)
    current = tmp_path.stat()

    with pytest.raises(OSError):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    after = outside.stat(follow_symlinks=False)
    assert (after.st_uid, after.st_gid, after.st_mode) == (
        before.st_uid,
        before.st_gid,
        before.st_mode,
    )


@pytest.mark.parametrize("kind", ["file", "fifo"])
def test_safe_profile_directory_flow_rejects_non_directory_leaf(tmp_path, kind):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    invalid = memories / ".imports"
    if kind == "file":
        invalid.write_text("not a directory")
    else:
        os.mkfifo(invalid)
    current = home.stat()

    with pytest.raises(OSError):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    assert invalid.is_file() if kind == "file" else stat.S_ISFIFO(invalid.lstat().st_mode)


def test_safe_profile_directory_flow_detects_rename_before_open(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    imports = memories / ".imports"
    imports.mkdir(parents=True)
    imports.chmod(0o755)
    current = home.stat()
    original_open = safe_profile_dirs.os.open
    swapped = False

    def swap_imports_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == ".imports" and dir_fd is not None and not swapped:
            swapped = True
            os.rename(".imports", ".imports-old", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.mkdir(".imports", 0o711, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_profile_dirs.os, "open", swap_imports_before_open)

    with pytest.raises(RuntimeError, match="changed during open"):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    assert swapped is True
    assert stat.S_IMODE(imports.stat().st_mode) == 0o711


def test_safe_profile_directory_flow_normalizes_only_opened_inodes(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    current = home.stat()
    normalized = []
    original_fchown = safe_profile_dirs.os.fchown

    def record_fchown(fd, uid, gid):
        opened = os.fstat(fd)
        normalized.append((opened.st_dev, opened.st_ino))
        return original_fchown(fd, uid, gid)

    monkeypatch.setattr(safe_profile_dirs.os, "fchown", record_fchown)
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )

    for path in (home, memories, memories / ".imports", memories / ".imports/backups"):
        opened = path.stat(follow_symlinks=False)
        assert (opened.st_dev, opened.st_ino) in normalized


def test_nixos_module_uses_nofollow_helper_for_transaction_directories():
    source = (Path(__file__).parents[2] / "nix" / "nixosModules.nix").read_text()

    assert "python3 ${safeProfileDirs}" in source
    assert "mkdir -p ${profileBackupsShell}" not in source
    assert "chown ${profileOwnerShell}" not in source
    assert 'memories/.imports 2770' not in source
    assert 'mkdir -p ${cfg.stateDir}/.hermes' not in source
    assert '"d ${cfg.stateDir}/.hermes' not in source
    assert "--recursive-ownership" in source
    assert "--shared-file-modes" in source
    assert 'find "$HERMES_HOME"' not in source
