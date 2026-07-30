"""POSIX-lock preservation in the profile session DB open/validate path.

POSIX record locks are per (process, inode): closing ANY fd of a file drops
every lock the process holds on it. The old ``_open_profile_session_db`` /
``_profile_session_db_is_current`` opened and closed throwaway fds on
``state.db`` and its sidecars, silently releasing the locks the live SQLite
connections depended on. Any other process's closing RW connection then passed
SQLite's last-closer probe and checkpoint-deleted the active WAL/SHM
(sqlite.org/howtocorrupt.html §2.3), splitting connections across WAL
generations — surviving connections kept writing into an unlinked WAL while
path openers saw a shorter history (the "lost chat context" incident).

These tests hold a real SessionDB open, run the open/validate helpers, and
assert from a separate process that (a) the fcntl locks are still present and
(b) an external RW connection close cannot unlink the WAL.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from gateway.platforms.api_server import APIServerAdapter

_PROBE = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    sys.exit(1)
sys.exit(0)
"""

_RW_CLOSE = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("SELECT count(*) FROM _lock_probe").fetchall()
conn.close()
"""


def _open_with_wal(tmp_path: Path):
    profile_home = tmp_path / "profiles" / "main"
    profile_home.mkdir(parents=True)
    db = APIServerAdapter._open_profile_session_db(profile_home)
    # Materialise the WAL and make the connection hold its persistent locks.
    db._conn.execute("CREATE TABLE IF NOT EXISTS _lock_probe (x INTEGER)")
    db._conn.execute("INSERT INTO _lock_probe VALUES (1)")
    return profile_home, db


def _exclusive_lock_obtainable(db_path: Path) -> bool:
    """True if a separate process can take an exclusive fcntl lock."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(db_path)],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


def test_validate_helpers_preserve_posix_locks(tmp_path):
    profile_home, db = _open_with_wal(tmp_path)
    db_path = profile_home / "state.db"
    try:
        assert not _exclusive_lock_obtainable(db_path), (
            "SessionDB must hold fcntl locks right after opening"
        )

        for _ in range(3):
            assert APIServerAdapter._profile_session_db_is_current(
                profile_home, db
            )
        reopened = APIServerAdapter._open_profile_session_db(profile_home)
        reopened.close()

        assert not _exclusive_lock_obtainable(db_path), (
            "identity checks must not drop the live connection's POSIX locks"
        )
    finally:
        db.close()


def test_external_rw_close_cannot_unlink_wal(tmp_path):
    profile_home, db = _open_with_wal(tmp_path)
    db_path = profile_home / "state.db"
    wal_path = profile_home / "state.db-wal"
    assert wal_path.exists(), "WAL must exist while the writer is open"
    wal_ino = os.stat(wal_path).st_ino
    try:
        APIServerAdapter._profile_session_db_is_current(profile_home, db)

        result = subprocess.run(
            [sys.executable, "-c", _RW_CLOSE, str(db_path)],
            capture_output=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr.decode()

        assert wal_path.exists(), (
            "external RW close checkpoint-deleted the WAL out from under the "
            "live connection (locks were dropped)"
        )
        assert os.stat(wal_path).st_ino == wal_ino, (
            "WAL generation split: path now points at a different inode"
        )
        # The surviving connection must still see and extend its history.
        db._conn.execute("INSERT INTO _lock_probe VALUES (2)")
        with sqlite3.connect(db_path) as fresh:
            rows = fresh.execute("SELECT count(*) FROM _lock_probe").fetchone()[0]
        assert rows == 2, "path readers must see the survivor's writes"
    finally:
        db.close()


def test_anchor_registry_refcounts_and_reclaims_on_rotation(tmp_path):
    """Codex P1 on PR #243: anchor fds must not leak, yet must never be
    closed while their inode is still the live generation (a close would
    drop sibling connections' POSIX locks). Refcount + park + rotate-GC."""
    home = tmp_path / "profiles" / "main"
    home.mkdir(parents=True)
    (home / "state.db").write_bytes(b"gen-1")
    directory_fd = os.open(home, os.O_RDONLY)
    try:
        expected = os.stat(home / "state.db", follow_symlinks=False)
        fd1, key = APIServerAdapter._acquire_profile_db_anchor(
            directory_fd, expected, home
        )
        fd2, key2 = APIServerAdapter._acquire_profile_db_anchor(
            directory_fd, expected, home
        )
        assert (fd1, key) == (fd2, key2), "same inode must share one anchor fd"
        assert APIServerAdapter._profile_db_anchors[key]["refs"] == 2

        APIServerAdapter._release_profile_db_anchor(key)
        os.fstat(fd1)  # one holder left: fd must stay open
        APIServerAdapter._release_profile_db_anchor(key)
        # refs == 0 but the inode is still the live generation: parked open.
        os.fstat(fd1)
        assert APIServerAdapter._profile_db_anchors[key]["refs"] == 0

        # Rotate the generation; the next registry touch reclaims the fd.
        # Assert the close BEFORE opening anything else: a fresh open would
        # reuse the just-closed fd number and make os.fstat(fd1) ambiguous.
        replacement = tmp_path / "gen-2"
        replacement.write_bytes(b"gen-2")
        os.replace(replacement, home / "state.db")
        with APIServerAdapter._profile_db_anchor_lock:
            APIServerAdapter._gc_retired_profile_db_anchors_locked()
        assert key not in APIServerAdapter._profile_db_anchors
        with pytest.raises(OSError):
            os.fstat(fd1)

        expected2 = os.stat(home / "state.db", follow_symlinks=False)
        fd3, key3 = APIServerAdapter._acquire_profile_db_anchor(
            directory_fd, expected2, home
        )
        assert key3 != key
        APIServerAdapter._release_profile_db_anchor(key3)
        os.remove(home / "state.db")
        with APIServerAdapter._profile_db_anchor_lock:
            APIServerAdapter._gc_retired_profile_db_anchors_locked()
        assert key3 not in APIServerAdapter._profile_db_anchors
    finally:
        os.close(directory_fd)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="/proc anchor is Linux-only"
)
def test_session_db_close_releases_anchor_and_rotation_reclaims_fd(tmp_path):
    profile_home, db = _open_with_wal(tmp_path)
    st = os.stat(profile_home / "state.db")
    key = (st.st_dev, st.st_ino)
    entry = APIServerAdapter._profile_db_anchors.get(key)
    assert entry is not None and entry["refs"] == 1
    anchor_fd = entry["fd"]

    db.close()
    db.close()  # double close must not double-release
    entry = APIServerAdapter._profile_db_anchors.get(key)
    assert entry is not None and entry["refs"] == 0
    os.fstat(anchor_fd)  # parked while the inode is still current

    # Rotate the generation: the retired anchor must be closed. Assert
    # before any new open so the fd number cannot have been reused.
    copy = tmp_path / "rotated.db"
    shutil.copy(profile_home / "state.db", copy)
    os.replace(copy, profile_home / "state.db")
    with APIServerAdapter._profile_db_anchor_lock:
        APIServerAdapter._gc_retired_profile_db_anchors_locked()
    assert key not in APIServerAdapter._profile_db_anchors
    with pytest.raises(OSError):
        os.fstat(anchor_fd)

    # And a normal reopen on the rotated generation still works.
    db2 = APIServerAdapter._open_profile_session_db(profile_home)
    db2.close()
