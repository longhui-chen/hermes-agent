#!/usr/bin/env python3
"""Safely normalize Nix-managed memory transaction directories."""

from __future__ import annotations

import argparse
import errno
import json
import os
import secrets
import stat
import sys
from pathlib import Path

_RECURSIVE_IDENTITY_RETRIES = 3
_MAX_RECURSIVE_DEPTH = 64
_MANAGED_ROOT_LEAVES = {"config.yaml", ".managed", ".container-mode", "auth.json", ".env"}
_MAX_MANAGED_LEAF_BYTES = 16 << 20
_MAX_FDINFO_BYTES = 64 << 10


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _mount_id(fd: int) -> int | None:
    """Return the Linux mount identity for fd; other kernels use st_dev."""
    if not sys.platform.startswith("linux"):
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        info_fd = os.open(f"/proc/self/fdinfo/{fd}", flags)
    except OSError as exc:
        raise OSError(
            errno.ENOTSUP, f"cannot determine managed profile mount identity: {exc}"
        ) from exc
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(info_fd, min(4096, _MAX_FDINFO_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_FDINFO_BYTES:
                raise OSError(errno.EFBIG, "managed profile fdinfo is too large")
    finally:
        os.close(info_fd)
    for line in b"".join(chunks).splitlines():
        key, separator, value = line.partition(b":")
        if separator and key == b"mnt_id":
            try:
                return int(value.strip())
            except ValueError as exc:
                raise OSError(
                    errno.ENOTSUP, "invalid managed profile mount identity"
                ) from exc
    raise OSError(errno.ENOTSUP, "managed profile mount identity is unavailable")


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


def _normalize_child(
    parent_fd: int,
    name: str,
    uid: int,
    gid: int,
    mode: int,
    *,
    expected_device: int | None = None,
    expected_mount_id: int | None = None,
) -> int:
    child_fd = _open_or_create_child(parent_fd, name)
    try:
        opened = os.fstat(child_fd)
        if expected_device is not None and opened.st_dev != expected_device:
            raise OSError(errno.EXDEV, f"managed profile directory crosses device: {name}")
        if expected_mount_id is not None and _mount_id(child_fd) != expected_mount_id:
            raise OSError(errno.EXDEV, f"managed profile directory crosses mount: {name}")
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
    root_device: int,
    root_mount_id: int | None,
) -> bool:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    for attempt in range(_RECURSIVE_IDENTITY_RETRIES):
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        try:
            try:
                visible = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            opened = os.fstat(fd)
            stable = (
                stat.S_ISREG(visible.st_mode)
                and stat.S_ISREG(opened.st_mode)
                and (expected.st_dev, expected.st_ino)
                == (opened.st_dev, opened.st_ino)
                and (visible.st_dev, visible.st_ino)
                == (opened.st_dev, opened.st_ino)
            )
            if stable:
                if opened.st_dev != root_device:
                    return False
                if root_mount_id is not None and _mount_id(fd) != root_mount_id:
                    return False
                if chown:
                    os.fchown(fd, uid, gid)
                if group_write:
                    os.fchmod(
                        fd,
                        stat.S_IMODE(opened.st_mode) | stat.S_IRGRP | stat.S_IWGRP,
                    )
                return True
        finally:
            os.close(fd)
        if attempt + 1 == _RECURSIVE_IDENTITY_RETRIES:
            raise RuntimeError(f"managed profile file kept changing: {name}")
        try:
            expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(expected.st_mode):
            return False
    return False


def _open_recursive_directory(
    parent_fd: int, name: str, expected: os.stat_result
) -> tuple[int, os.stat_result] | None:
    retry_errnos = {errno.ENOENT, errno.ENOTDIR, getattr(errno, "ELOOP", errno.ENOTDIR)}
    for attempt in range(_RECURSIVE_IDENTITY_RETRIES):
        try:
            child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno not in retry_errnos:
                raise
            child_fd = -1
        if child_fd >= 0:
            try:
                opened = _require_visible_identity(parent_fd, name, child_fd, expected)
                return child_fd, opened
            except FileNotFoundError:
                os.close(child_fd)
                return None
            except RuntimeError:
                os.close(child_fd)
        if attempt + 1 == _RECURSIVE_IDENTITY_RETRIES:
            raise RuntimeError(f"managed profile directory kept changing: {name}")
        try:
            expected = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(expected.st_mode):
            return None
    return None


def _reject_unsafe_managed_leaves(home_fd: int) -> None:
    for name in _MANAGED_ROOT_LEAVES:
        try:
            visible = os.stat(name, dir_fd=home_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(visible.st_mode):
            raise OSError(errno.EINVAL, f"unsafe managed profile leaf: {name}")


def _identity_token(opened: os.stat_result) -> str:
    return f"{opened.st_dev}:{opened.st_ino}"


def _open_verified_home(home: Path, expected_identity: str) -> int:
    home_fd = _open_absolute_directory(home)
    if _identity_token(os.fstat(home_fd)) != expected_identity:
        os.close(home_fd)
        raise RuntimeError("managed profile changed after secure setup")
    return home_fd


def _validate_leaf_name(name: str) -> None:
    if name not in _MANAGED_ROOT_LEAVES:
        raise ValueError(f"unsupported managed profile leaf: {name}")


def _read_managed_leaf(home_fd: int, name: str) -> bytes:
    _validate_leaf_name(name)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(name, flags, dir_fd=home_fd)
    except FileNotFoundError:
        return b""
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError(errno.EINVAL, f"unsafe managed profile leaf: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(64 << 10, _MAX_MANAGED_LEAF_BYTES + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_MANAGED_LEAF_BYTES:
                raise OSError(errno.EFBIG, f"managed profile leaf is too large: {name}")
    finally:
        os.close(fd)


def _write_managed_leaf(
    home_fd: int,
    name: str,
    content: bytes,
    uid: int,
    gid: int,
    mode: int,
    *,
    if_missing: bool = False,
) -> None:
    _validate_leaf_name(name)
    if len(content) > _MAX_MANAGED_LEAF_BYTES:
        raise OSError(errno.EFBIG, f"managed profile leaf is too large: {name}")
    if if_missing:
        try:
            existing = os.stat(name, dir_fd=home_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise OSError(errno.EINVAL, f"unsafe managed profile leaf: {name}")
            return
    temp_name = f".managed-leaf-{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp_name, flags, 0o600, dir_fd=home_fd)
    published = False
    try:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, f"failed to write managed profile leaf: {name}")
            view = view[written:]
        os.fsync(fd)
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
        os.close(fd)
        fd = -1
        os.replace(temp_name, name, src_dir_fd=home_fd, dst_dir_fd=home_fd)
        published = True
        os.fsync(home_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if not published:
            try:
                os.unlink(temp_name, dir_fd=home_fd)
            except FileNotFoundError:
                pass


def _remove_managed_leaf(home_fd: int, name: str) -> None:
    _validate_leaf_name(name)
    try:
        current = os.stat(name, dir_fd=home_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(current.st_mode):
        raise OSError(errno.EINVAL, f"unsafe managed profile leaf: {name}")
    os.unlink(name, dir_fd=home_fd)
    os.fsync(home_fd)


def _sync_plugin_links(
    home_fd: int, manifest_path: Path, uid: int, gid: int
) -> None:
    raw = manifest_path.read_bytes()
    if len(raw) > _MAX_MANAGED_LEAF_BYTES:
        raise OSError(errno.EFBIG, "managed plugin manifest is too large")
    manifest = json.loads(raw)
    if not isinstance(manifest, list):
        raise ValueError("managed plugin manifest must be an array")
    before = os.stat("plugins", dir_fd=home_fd, follow_symlinks=False)
    plugins_fd = os.open("plugins", _directory_flags(), dir_fd=home_fd)
    try:
        opened = _require_visible_identity(home_fd, "plugins", plugins_fd, before)
        home_stat = os.fstat(home_fd)
        home_mount_id = _mount_id(home_fd)
        if opened.st_dev != home_stat.st_dev or (
            home_mount_id is not None and _mount_id(plugins_fd) != home_mount_id
        ):
            raise OSError(errno.EXDEV, "managed plugins directory crosses device or mount")
        desired: list[tuple[str, str]] = []
        for entry in manifest:
            if not isinstance(entry, dict) or set(entry) != {"name", "target"}:
                raise ValueError("invalid managed plugin manifest entry")
            name = entry["name"]
            target = entry["target"]
            if (
                not isinstance(name, str)
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\0" in name
                or not isinstance(target, str)
                or not target.startswith("/")
                or "\0" in target
            ):
                raise ValueError("unsafe managed plugin manifest entry")
            desired.append((f"nix-managed-{name}", target))
        for name in os.listdir(plugins_fd):
            if not name.startswith("nix-managed-"):
                continue
            current = os.stat(name, dir_fd=plugins_fd, follow_symlinks=False)
            if stat.S_ISLNK(current.st_mode):
                os.unlink(name, dir_fd=plugins_fd)
        for name, target in desired:
            temp_name = f".managed-plugin-{secrets.token_hex(16)}.tmp"
            os.symlink(target, temp_name, dir_fd=plugins_fd)
            published = False
            try:
                os.chown(
                    temp_name,
                    uid,
                    gid,
                    dir_fd=plugins_fd,
                    follow_symlinks=False,
                )
                os.replace(temp_name, name, src_dir_fd=plugins_fd, dst_dir_fd=plugins_fd)
                published = True
            finally:
                if not published:
                    try:
                        os.unlink(temp_name, dir_fd=plugins_fd)
                    except FileNotFoundError:
                        pass
        os.fsync(plugins_fd)
        _require_visible_identity(home_fd, "plugins", plugins_fd, opened)
    finally:
        os.close(plugins_fd)


def _write_trust_anchor(
    path: Path, home: Path, uid: int, gid: int, opened: os.stat_result
) -> None:
    content = (
        f"version=2\nhome={home}\nuid={uid}\ngid={gid}\n"
        f"dev={opened.st_dev}\nino={opened.st_ino}\n"
    ).encode()
    temp = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    published = False
    try:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "failed to write managed profile trust")
            view = view[written:]
        os.fsync(fd)
        os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o444)
        os.close(fd)
        fd = -1
        os.replace(temp, path)
        published = True
        parent_fd = _open_absolute_directory(path.parent)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if not published:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass


def _normalize_tree(
    directory_fd: int,
    uid: int,
    gid: int,
    root_device: int,
    root_mount_id: int | None,
    *,
    chown: bool,
    shared_file_modes: bool,
    group_write_tree: bool = False,
    top_level: bool = False,
    depth: int = 0,
) -> None:
    for name in os.listdir(directory_fd):
        try:
            visible = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if visible.st_dev != root_device:
            continue
        if stat.S_ISDIR(visible.st_mode):
            if depth >= _MAX_RECURSIVE_DEPTH:
                print(
                    f"warning: skipping managed profile subtree deeper than "
                    f"{_MAX_RECURSIVE_DEPTH}: {name}",
                    file=sys.stderr,
                )
                continue
            opened_child = _open_recursive_directory(directory_fd, name, visible)
            if opened_child is None:
                continue
            child_fd, opened = opened_child
            try:
                if opened.st_dev != root_device:
                    continue
                if root_mount_id is not None and _mount_id(child_fd) != root_mount_id:
                    continue
                if chown:
                    os.fchown(child_fd, uid, gid)
                _normalize_tree(
                    child_fd,
                    uid,
                    gid,
                    root_device,
                    root_mount_id,
                    chown=chown,
                    shared_file_modes=shared_file_modes,
                    group_write_tree=(
                        group_write_tree
                        or top_level
                        and name in {"cron", "sessions", "logs", "memories", "plugins"}
                    ),
                    depth=depth + 1,
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
                root_device=root_device,
                root_mount_id=root_mount_id,
            )


def ensure_transaction_directories(
    home: Path,
    uid: int,
    gid: int,
    *,
    recursive_ownership: bool = False,
    shared_file_modes: bool = False,
    trust_anchor: Path | None = None,
) -> str:
    """Normalize the managed profile chain without following any symlink."""
    parent_fd = _open_absolute_directory(home.parent)
    home_fd = memories_fd = imports_fd = backups_fd = -1
    state_fds: list[int] = []
    try:
        home_fd = _normalize_child(parent_fd, home.name, uid, gid, 0o2770)
        root_device = os.fstat(home_fd).st_dev
        root_mount_id = _mount_id(home_fd)
        for name in ("cron", "sessions", "logs", "plugins"):
            state_fds.append(
                _normalize_child(
                    home_fd,
                    name,
                    uid,
                    gid,
                    0o2770,
                    expected_device=root_device,
                    expected_mount_id=root_mount_id,
                )
            )
        memories_fd = _normalize_child(
            home_fd,
            "memories",
            uid,
            gid,
            0o2770,
            expected_device=root_device,
            expected_mount_id=root_mount_id,
        )
        imports_fd = _normalize_child(
            memories_fd,
            ".imports",
            uid,
            gid,
            0o2770,
            expected_device=root_device,
            expected_mount_id=root_mount_id,
        )
        backups_fd = _normalize_child(
            imports_fd,
            "backups",
            uid,
            gid,
            0o2770,
            expected_device=root_device,
            expected_mount_id=root_mount_id,
        )
        _reject_unsafe_managed_leaves(home_fd)
        opened_home = os.fstat(home_fd)
        if trust_anchor is not None:
            _write_trust_anchor(trust_anchor, home, uid, gid, opened_home)
        if recursive_ownership or shared_file_modes:
            _normalize_tree(
                home_fd,
                uid,
                gid,
                root_device,
                root_mount_id,
                chown=recursive_ownership,
                shared_file_modes=shared_file_modes,
                top_level=True,
            )
        return _identity_token(opened_home)
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
    parser.add_argument("--trust-anchor", type=Path)
    parser.add_argument("--expected-identity")
    parser.add_argument("--read-leaf", choices=sorted(_MANAGED_ROOT_LEAVES))
    parser.add_argument("--write-leaf", choices=sorted(_MANAGED_ROOT_LEAVES))
    parser.add_argument("--remove-leaf", choices=sorted(_MANAGED_ROOT_LEAVES))
    parser.add_argument("--sync-plugins-manifest", type=Path)
    parser.add_argument("--content-file", type=Path)
    parser.add_argument("--leaf-mode", type=lambda value: int(value, 8))
    parser.add_argument("--if-missing", action="store_true")
    args = parser.parse_args()
    leaf_actions = [
        args.read_leaf,
        args.write_leaf,
        args.remove_leaf,
        args.sync_plugins_manifest,
    ]
    if sum(action is not None for action in leaf_actions) > 1:
        parser.error("choose only one managed leaf action")
    if any(action is not None for action in leaf_actions):
        if not args.expected_identity:
            parser.error("managed leaf actions require --expected-identity")
        home_fd = _open_verified_home(args.home, args.expected_identity)
        action_succeeded = False
        try:
            if args.read_leaf:
                sys.stdout.buffer.write(_read_managed_leaf(home_fd, args.read_leaf))
            elif args.write_leaf:
                if args.content_file is None or args.leaf_mode is None:
                    parser.error("--write-leaf requires --content-file and --leaf-mode")
                _write_managed_leaf(
                    home_fd,
                    args.write_leaf,
                    args.content_file.read_bytes(),
                    args.uid,
                    args.gid,
                    args.leaf_mode,
                    if_missing=args.if_missing,
                )
            elif args.remove_leaf:
                _remove_managed_leaf(home_fd, args.remove_leaf)
            else:
                _sync_plugin_links(
                    home_fd,
                    args.sync_plugins_manifest,
                    args.uid,
                    args.gid,
                )
            action_succeeded = True
        finally:
            os.close(home_fd)
        if action_succeeded:
            verification_fd = _open_verified_home(args.home, args.expected_identity)
            os.close(verification_fd)
        return
    identity = ensure_transaction_directories(
        args.home,
        args.uid,
        args.gid,
        recursive_ownership=args.recursive_ownership,
        shared_file_modes=args.shared_file_modes,
        trust_anchor=args.trust_anchor,
    )
    print(identity)


if __name__ == "__main__":
    main()
