"""Sidecar validation in ``APIServerAdapter._open_profile_session_db``.

Production boards mount each profile home as a btrfs subvolume: the profile
directory inode reports the parent filesystem's ``st_dev`` while every file
inside reports the subvolume's own, so the old sidecar check (``st_dev`` must
match the *directory*) rejected every legitimate ``state.db-wal``/``-shm`` and
the whole REST session surface answered 503 ``session_db_unavailable``.

The check now compares sidecars against ``state.db`` itself. These tests
simulate the subvolume dev split by patching ``os.stat``/``os.fstat`` for the
specific inode, and keep the original attack coverage: symlink, hardlink, and
cross-filesystem sidecars must still be rejected.
"""

import logging
import os
from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


class _ShiftedDevStat:
    """Proxy a real stat result with ``st_dev`` moved to another filesystem."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        if name == "st_dev":
            return self._inner.st_dev + 0x10000
        return getattr(self._inner, name)


def _shift_dev_for_inode(monkeypatch, target_ino: int):
    """Make os.stat/os.fstat report a different st_dev for one inode only."""
    real_stat = os.stat
    real_fstat = os.fstat

    def fake_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if result.st_ino == target_ino:
            return _ShiftedDevStat(result)
        return result

    def fake_fstat(fd):
        result = real_fstat(fd)
        if result.st_ino == target_ino:
            return _ShiftedDevStat(result)
        return result

    monkeypatch.setattr(os, "stat", fake_stat)
    monkeypatch.setattr(os, "fstat", fake_fstat)


def _make_profile(tmp_path: Path) -> Path:
    profile_home = tmp_path / "profiles" / "main"
    db = SessionDB(profile_home / "state.db")
    db.close()
    return profile_home


def test_sidecar_accepted_when_directory_is_a_subvolume_boundary(
    tmp_path, monkeypatch
):
    profile_home = _make_profile(tmp_path)
    (profile_home / "state.db-wal").write_bytes(b"")
    _shift_dev_for_inode(monkeypatch, os.stat(profile_home).st_ino)

    db = APIServerAdapter._open_profile_session_db(profile_home)
    try:
        assert db is not None
    finally:
        db.close()


def test_sidecar_on_different_filesystem_than_state_db_is_rejected(
    tmp_path, monkeypatch
):
    profile_home = _make_profile(tmp_path)
    wal = profile_home / "state.db-wal"
    wal.write_bytes(b"")
    _shift_dev_for_inode(monkeypatch, os.stat(wal).st_ino)

    with pytest.raises(
        RuntimeError, match="state.db-wal must be a private regular file"
    ):
        APIServerAdapter._open_profile_session_db(profile_home)


def test_symlink_sidecar_is_rejected(tmp_path):
    profile_home = _make_profile(tmp_path)
    victim = tmp_path / "victim-wal"
    victim.write_bytes(b"")
    (profile_home / "state.db-wal").symlink_to(victim)

    with pytest.raises(
        RuntimeError, match="state.db-wal must be a private regular file"
    ):
        APIServerAdapter._open_profile_session_db(profile_home)


def test_hardlink_sidecar_is_rejected(tmp_path):
    profile_home = _make_profile(tmp_path)
    victim = tmp_path / "victim-wal"
    victim.write_bytes(b"")
    os.link(victim, profile_home / "state.db-wal")

    with pytest.raises(
        RuntimeError, match="state.db-wal must be a private regular file"
    ):
        APIServerAdapter._open_profile_session_db(profile_home)


def test_non_regular_sidecar_is_rejected(tmp_path):
    profile_home = _make_profile(tmp_path)
    (profile_home / "state.db-wal").mkdir()

    with pytest.raises(
        RuntimeError, match="state.db-wal must be a private regular file"
    ):
        APIServerAdapter._open_profile_session_db(profile_home)


def test_open_failure_is_logged_as_warning(tmp_path, caplog):
    profile_home = tmp_path / "profiles" / "main"
    (profile_home / "state.db").mkdir(parents=True)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.api_server"):
        assert adapter._open_and_cache_session_db(profile_home) is None

    assert any(
        "SessionDB unavailable" in record.getMessage()
        and record.levelno == logging.WARNING
        for record in caplog.records
    )
