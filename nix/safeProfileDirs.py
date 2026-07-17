#!/usr/bin/env python3
"""Safely normalize Nix-managed memory transaction directories."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"unsafe managed profile path: {path}")
    fd = os.open(path.anchor or "/", _directory_flags())
    try:
        for part in path.parts[1:]:
            child = os.open(part, _directory_flags(), dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _require_visible_identity(
    parent_fd: int,
    name: str,
    child_fd: int,
    expected: os.stat_result | None = None,
) -> os.stat_result:
    visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    opened = os.fstat(child_fd)
    if (
        not stat.S_ISDIR(visible.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or expected is not None
        and (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino)
        or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino)
    ):
        raise RuntimeError(f"managed profile directory changed during open: {name}")
    return opened


def _open_or_create_child(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    try:
        _require_visible_identity(parent_fd, name, child_fd, before)
        return child_fd
    except BaseException:
        os.close(child_fd)
        raise


def _normalize_child(parent_fd: int, name: str, uid: int, gid: int, mode: int) -> int:
    child_fd = _open_or_create_child(parent_fd, name)
    try:
        os.fchown(child_fd, uid, gid)
        os.fchmod(child_fd, mode)
        _require_visible_identity(parent_fd, name, child_fd)
        return child_fd
    except BaseException:
        os.close(child_fd)
        raise


def _normalize_regular_file(
    parent_fd: int,
    name: str,
    uid: int,
    gid: int,
    *,
    chown: bool,
    group_write: bool,
    expected: os.stat_result,
) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(fd)
        if (
            not stat.S_ISREG(visible.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino)
            or (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RuntimeError(f"managed profile file changed during open: {name}")
        if chown:
            os.fchown(fd, uid, gid)
        if group_write:
            os.fchmod(fd, stat.S_IMODE(opened.st_mode) | stat.S_IRGRP | stat.S_IWGRP)
    finally:
        os.close(fd)


def _normalize_tree(
    directory_fd: int,
    uid: int,
    gid: int,
    root_device: int,
    *,
    chown: bool,
    shared_file_modes: bool,
    group_write_tree: bool = False,
    top_level: bool = False,
) -> None:
    for name in os.listdir(directory_fd):
        visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if visible.st_dev != root_device:
            continue
        if stat.S_ISDIR(visible.st_mode):
            child_fd = os.open(name, _directory_flags(), dir_fd=directory_fd)
            try:
                opened = _require_visible_identity(
                    directory_fd, name, child_fd, visible
                )
                if chown:
                    os.fchown(child_fd, uid, gid)
                _normalize_tree(
                    child_fd,
                    uid,
                    gid,
                    root_device,
                    chown=chown,
                    shared_file_modes=shared_file_modes,
                    group_write_tree=(
                        group_write_tree
                        or top_level
                        and name in {"cron", "sessions", "logs", "memories", "plugins"}
                    ),
                )
            finally:
                os.close(child_fd)
        elif stat.S_ISREG(visible.st_mode):
            root_shared = top_level and (
                name == "SOUL.md"
                or name.endswith((".db", ".db-wal", ".db-shm"))
            )
            _normalize_regular_file(
                directory_fd,
                name,
                uid,
                gid,
                chown=chown,
                group_write=shared_file_modes and (group_write_tree or root_shared),
                expected=visible,
            )


def ensure_transaction_directories(
    home: Path,
    uid: int,
    gid: int,
    *,
    recursive_ownership: bool = False,
    shared_file_modes: bool = False,
) -> None:
    """Normalize the managed profile chain without following any symlink."""
    parent_fd = _open_absolute_directory(home.parent)
    home_fd = memories_fd = imports_fd = backups_fd = -1
    state_fds: list[int] = []
    try:
        home_fd = _normalize_child(parent_fd, home.name, uid, gid, 0o2770)
        for name in ("cron", "sessions", "logs", "plugins"):
            state_fds.append(_normalize_child(home_fd, name, uid, gid, 0o2770))
        memories_fd = _normalize_child(home_fd, "memories", uid, gid, 0o2770)
        imports_fd = _normalize_child(memories_fd, ".imports", uid, gid, 0o2770)
        backups_fd = _normalize_child(imports_fd, "backups", uid, gid, 0o2770)
        if recursive_ownership or shared_file_modes:
            _normalize_tree(
                home_fd,
                uid,
                gid,
                os.fstat(home_fd).st_dev,
                chown=recursive_ownership,
                shared_file_modes=shared_file_modes,
                top_level=True,
            )
    finally:
        for fd in (backups_fd, imports_fd, memories_fd, *state_fds, home_fd, parent_fd):
            if fd >= 0:
                os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("home", type=Path)
    parser.add_argument("uid", type=int)
    parser.add_argument("gid", type=int)
    parser.add_argument("--recursive-ownership", action="store_true")
    parser.add_argument("--shared-file-modes", action="store_true")
    args = parser.parse_args()
    ensure_transaction_directories(
        args.home,
        args.uid,
        args.gid,
        recursive_ownership=args.recursive_ownership,
        shared_file_modes=args.shared_file_modes,
    )


if __name__ == "__main__":
    main()
