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
import sqlite3
import subprocess
import sys
from pathlib import Path

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
