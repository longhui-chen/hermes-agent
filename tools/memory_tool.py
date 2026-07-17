#!/usr/bin/env python3
"""
Memory Tool Module - Persistent Curated Memory

Provides bounded, file-backed memory that persists across sessions. Two stores:
  - MEMORY.md: agent's personal notes and observations (environment facts, project
    conventions, tool quirks, things learned)
  - USER.md: what the agent knows about the user (preferences, communication style,
    expectations, workflow habits)

Both are injected into the system prompt as a frozen snapshot at session start.
Mid-session writes update files on disk immediately (durable) but do NOT change
the system prompt -- this preserves the prefix cache for the entire session.
The snapshot refreshes on the next session start.

Entry delimiter: § (section sign). Entries can be multiline.
Character limits (not tokens) because char counts are model-independent.

Design:
- Single `memory` tool with action parameter: add, replace, remove
- replace/remove use short unique substring matching (not full text or IDs)
- Behavioral guidance lives in the tool schema description
- Frozen snapshot pattern: system prompt is stable, tool responses show live state
"""

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import tempfile
import time
from contextvars import ContextVar
from contextlib import ExitStack, contextmanager
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, Any, List, Optional

from utils import atomic_replace

# fcntl is Unix-only; on Windows use msvcrt for file locking
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS = {
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}
_LINK_COPY_FALLBACK_ERRNOS = {
    errno.EPERM,
    errno.EXDEV,
    getattr(errno, "ENOTSUP", errno.EPERM),
    getattr(errno, "EOPNOTSUPP", errno.EPERM),
}


def _path_identity(path: Path) -> tuple[int, int]:
    current = os.stat(path, follow_symlinks=False)
    return current.st_dev, current.st_ino


def _open_resolved_directory_chain(path: Path, flags: int) -> int:
    """Open a realpath from `/` one no-follow component at a time."""
    resolved = Path(os.path.realpath(path))
    current_fd = os.open(resolved.anchor or "/", flags)
    try:
        for part in resolved.parts[1:]:
            next_fd = os.open(part, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _fsync_directory(path: Path) -> None:
    """Persist directory metadata after an atomic rename on POSIX."""
    if os.name == "nt":
        return
    try:
        dir_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno in unsupported:
            logger.debug("directory fsync unsupported for %s: %s", path, exc)
            return
        raise


def _fsync_directory_fd(directory_fd: int, path: Path) -> None:
    """Strictly persist an open directory used by import/reset transactions."""
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        if exc.errno in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
            raise MemoryImportUnsupported(
                f"durable directory fsync is unsupported for {path}"
            ) from exc
        raise

# Where memory files live — resolved dynamically so profile overrides
# (HERMES_HOME env var changes) are always respected.  The old module-level
# constant was cached at import time and could go stale if a profile switch
# happened after the first import.
def get_memory_dir() -> Path:
    """Return the profile-scoped memories directory."""
    return get_hermes_home() / "memories"


def portable_memory_import_supported() -> bool:
    """Report API availability without mutating the selected profile."""
    try:
        home_identity, mem_identity = _require_durable_profile_filesystem()
    except (MemoryImportConflict, MemoryImportUnsupported, OSError):
        return False
    if home_identity is not None and mem_identity is not None:
        try:
            key = _import_hardlink_cache_key(home_identity, mem_identity)
        except OSError:
            return False
        if key in _IMPORT_HARDLINK_NEGATIVE_CACHE:
            return False
    return True


def portable_memory_reset_supported() -> bool:
    """Whether reset can enforce durable directory transaction semantics."""
    try:
        _require_durable_profile_filesystem()
    except (MemoryImportConflict, MemoryImportUnsupported, OSError):
        return False
    return True

ENTRY_DELIMITER = "\n§\n"
MEMORY_IMPORT_BACKUP_LIMIT = 5
MAX_CURATED_MEMORY_FILE_BYTES = 1 << 20
_MISSING_MEMORY_FILE_SHA256 = "missing"
_MEMORY_TARGET_FILES = {"memory": "MEMORY.md", "user": "USER.md"}
_MEMORY_TRANSACTION_LOCK = ".curated-memory-transaction"
_RESET_RECEIPT_PREFIX = ".reset_tx_"
_RESET_STAGE_PREFIX = ".reset_stage_"
_IMPORT_LINK_PROBE_PREFIX = ".import_link_probe_"
_IMPORT_HARDLINK_NEGATIVE_CACHE: set[
    tuple[str, tuple[int, int], tuple[int, int], int]
] = set()


class MemoryImportConflict(ValueError):
    """An import id or receipt conflicts with the live curated memory."""


class MemoryImportUnsupported(MemoryImportConflict):
    """The platform/filesystem cannot provide durable portable memory writes."""


class _ImportDirectoryHandles:
    """Stable POSIX directory handles for one curated-memory import."""

    def __init__(
        self,
        mem_dir: Path,
        *,
        create_managed: bool = True,
        expected_home_identity: Optional[tuple[int, int]] = None,
        expected_mem_identity: Optional[tuple[int, int]] = None,
    ):
        self.home_dir = mem_dir.parent
        self.mem_dir = mem_dir
        self.imports_dir = mem_dir / ".imports"
        self.backup_dir = self.imports_dir / "backups"
        self.create_managed = create_managed
        self.expected_home_identity = (
            expected_home_identity or _path_identity(self.home_dir)
        )
        self.expected_mem_identity = (
            expected_mem_identity or _path_identity(self.mem_dir)
        )
        self._unopened_identities: Dict[Path, Optional[tuple[int, int, int]]] = {}
        self.home_fd = -1
        self.mem_fd = -1
        self.imports_fd = -1
        self.backup_fd = -1

    @staticmethod
    def _directory_flags() -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return flags

    @staticmethod
    def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> int:
        if create:
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
        try:
            fd = os.open(
                name,
                _ImportDirectoryHandles._directory_flags(),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if not create:
                return -1
            raise
        except OSError as exc:
            if not create:
                try:
                    current = os.stat(
                        name, dir_fd=parent_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    return -1
                except OSError as inspect_error:
                    raise MemoryImportConflict(
                        f"cannot inspect managed memory directory {name}: "
                        f"{inspect_error}"
                    ) from exc
                if not stat.S_ISDIR(current.st_mode):
                    raise MemoryImportConflict(
                        f"managed memory directory {name} must be a real directory"
                    ) from exc
            raise MemoryImportConflict(
                f"managed memory directory {name} is unsafe: {exc}"
            ) from exc
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            os.close(fd)
            raise MemoryImportConflict(
                f"managed memory directory {name} must be a real directory"
            )
        if (
            hasattr(os, "geteuid") and opened.st_uid != os.geteuid()
        ) or stat.S_IMODE(opened.st_mode) & 0o022:
            os.close(fd)
            raise MemoryImportConflict(
                f"managed memory directory {name} must be owned by this user "
                "and not group/world writable"
            )
        return fd

    def __enter__(self):
        if os.name == "nt" or not _OPEN_SUPPORTS_DIR_FD:
            raise MemoryImportConflict(
                "secure memory import requires directory-relative file operations"
            )
        try:
            self.home_fd = _open_resolved_directory_chain(
                self.home_dir, self._directory_flags()
            )
            if (
                os.fstat(self.home_fd).st_dev,
                os.fstat(self.home_fd).st_ino,
            ) != self.expected_home_identity:
                raise MemoryImportConflict("HERMES_HOME changed before secure open")
            self.mem_fd = os.open(
                self.mem_dir.name,
                self._directory_flags(),
                dir_fd=self.home_fd,
            )
            if (
                os.fstat(self.mem_fd).st_dev,
                os.fstat(self.mem_fd).st_ino,
            ) != self.expected_mem_identity:
                raise MemoryImportConflict(
                    "profile memories directory changed before secure open"
                )
            self.imports_fd = self._open_child_directory(
                self.mem_fd, ".imports", create=self.create_managed
            )
            if self.imports_fd < 0:
                self._remember_unopened(self.imports_dir)
            if self.imports_fd >= 0:
                self.backup_fd = self._open_child_directory(
                    self.imports_fd, "backups", create=self.create_managed
                )
                if self.backup_fd < 0:
                    self._remember_unopened(self.backup_dir)
            self.verify_attached()
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for attribute in ("backup_fd", "imports_fd", "mem_fd", "home_fd"):
            fd = getattr(self, attribute)
            if fd >= 0:
                os.close(fd)
                setattr(self, attribute, -1)

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()

    @staticmethod
    def _same_open_directory(path: Path, fd: int) -> bool:
        try:
            current = os.lstat(path)
            opened = os.fstat(fd)
        except OSError:
            return False
        return (
            stat.S_ISDIR(current.st_mode)
            and stat.S_ISDIR(opened.st_mode)
            and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
        )

    def _remember_unopened(self, path: Path) -> None:
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            self._unopened_identities[path] = None
            return
        self._unopened_identities[path] = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
        )

    def verify_attached(self) -> None:
        for path, fd in (
            (self.home_dir, self.home_fd),
            (self.mem_dir, self.mem_fd),
            (self.imports_dir, self.imports_fd),
            (self.backup_dir, self.backup_fd),
        ):
            if fd < 0:
                try:
                    current = os.lstat(path)
                    identity = (current.st_dev, current.st_ino, current.st_mode)
                except FileNotFoundError:
                    identity = None
                if identity != self._unopened_identities.get(path):
                    raise MemoryImportConflict(
                        f"managed memory directory changed during import: {path.name}"
                    )
                continue
            if not self._same_open_directory(path, fd):
                raise MemoryImportConflict(
                    f"managed memory directory changed during import: {path.name}"
                )

    def ensure_managed(self) -> None:
        """Create/open managed children only after the transaction lock."""
        if self.imports_fd < 0:
            self.imports_fd = self._open_child_directory(
                self.mem_fd, ".imports", create=True
            )
            self._unopened_identities.pop(self.imports_dir, None)
        if self.backup_fd < 0:
            self.backup_fd = self._open_child_directory(
                self.imports_fd, "backups", create=True
            )
            self._unopened_identities.pop(self.backup_dir, None)
        self.verify_attached()

    def require_durable_scopes(self) -> None:
        """Strictly probe every currently opened transaction directory."""
        self.verify_attached()
        for directory_fd, path in (
            (self.home_fd, self.home_dir),
            (self.mem_fd, self.mem_dir),
            (self.imports_fd, self.imports_dir),
            (self.backup_fd, self.backup_dir),
        ):
            if directory_fd >= 0:
                _fsync_directory_fd(directory_fd, path)
        self.verify_attached()

    def _remove_import_link_probe_residue(self, names: List[str]) -> None:
        """Remove only zero-byte regular files reserved for the link probe."""
        changed = False
        try:
            for name in names:
                try:
                    current = os.stat(
                        name, dir_fd=self.mem_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(current.st_mode) or current.st_size != 0:
                    raise MemoryImportConflict(
                        f"memory import link probe residue {name} is unsafe"
                    )
                os.unlink(name, dir_fd=self.mem_fd)
                changed = True
        finally:
            if changed:
                _fsync_directory_fd(self.mem_fd, self.mem_dir)
        self.verify_attached()

    def require_import_hardlink_support(self) -> None:
        """Prove hard-link publish support without writing imported content."""
        self.verify_attached()
        fd, source_name = self._create_temp(
            self.mem_fd, _IMPORT_LINK_PROBE_PREFIX
        )
        link_name = f"{source_name}.link"
        try:
            os.fsync(fd)
            os.close(fd)
            fd = -1
            try:
                os.link(
                    source_name,
                    link_name,
                    src_dir_fd=self.mem_fd,
                    dst_dir_fd=self.mem_fd,
                    follow_symlinks=False,
                )
            except OSError as exc:
                if exc.errno in _LINK_COPY_FALLBACK_ERRNOS:
                    _IMPORT_HARDLINK_NEGATIVE_CACHE.add(
                        _import_hardlink_cache_key(
                            self.expected_home_identity,
                            self.expected_mem_identity,
                        )
                    )
                    raise MemoryImportUnsupported(
                        "curated memory import requires hard-link support on "
                        f"{self.mem_dir}"
                    ) from exc
                raise
            _fsync_directory_fd(self.mem_fd, self.mem_dir)
        finally:
            if fd >= 0:
                os.close(fd)
            self._remove_import_link_probe_residue(
                [link_name, source_name]
            )

    @staticmethod
    def read_bytes(directory_fd: int, name: str) -> Optional[bytes]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            fd = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise MemoryImportConflict(f"cannot safely open {name}: {exc}") from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode):
                raise MemoryImportConflict(f"{name} must be a regular file")
            if opened.st_size > MAX_CURATED_MEMORY_FILE_BYTES:
                raise MemoryImportConflict(f"{name} exceeds the memory file size limit")
            chunks = []
            total = 0
            while True:
                chunk = os.read(
                    fd, min(64 << 10, MAX_CURATED_MEMORY_FILE_BYTES + 1 - total)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_CURATED_MEMORY_FILE_BYTES:
                    raise MemoryImportConflict(
                        f"{name} exceeds the memory file size limit"
                    )
            return b"".join(chunks)
        finally:
            os.close(fd)

    @staticmethod
    def read_text(directory_fd: int, name: str) -> Optional[str]:
        raw = _ImportDirectoryHandles.read_bytes(directory_fd, name)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryImportConflict(f"{name} is not valid UTF-8") from exc

    def unlink_non_directory(self, directory_fd: int, name: str) -> bool:
        self.verify_attached()
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISDIR(current.st_mode):
            return False
        os.unlink(name, dir_fd=directory_fd)
        self.verify_attached()
        return True

    @staticmethod
    def read_receipt(directory_fd: int, name: str) -> Optional[Dict[str, Any]]:
        try:
            raw = _ImportDirectoryHandles.read_bytes(directory_fd, name)
        except MemoryImportConflict:
            try:
                current = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                return None
            if not stat.S_ISREG(current.st_mode):
                return None
            raise
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            return None
        return value if isinstance(value, dict) else None

    def preflight_reset(
        self, *, parse_import_receipts: bool = True
    ) -> tuple[List[str], List[str], List[str], Dict[str, Optional[Dict[str, Any]]]]:
        """Snapshot all reset inputs before allowing the first unlink."""
        try:
            memory_names = os.listdir(self.mem_fd)
            imports_names = (
                os.listdir(self.imports_fd) if self.imports_fd >= 0 else []
            )
            backup_names = (
                os.listdir(self.backup_fd) if self.backup_fd >= 0 else []
            )
            for directory_fd, names in (
                (self.mem_fd, memory_names),
                (self.imports_fd, imports_names),
                (self.backup_fd, backup_names),
            ):
                if directory_fd < 0:
                    continue
                for name in names:
                    os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            receipt_targets = (
                {
                    name: self.read_receipt(self.imports_fd, name)
                    for name in imports_names
                    if name.endswith(".json")
                }
                if parse_import_receipts
                else {}
            )
        except OSError as exc:
            raise MemoryImportConflict(
                f"cannot preflight curated memory reset: {exc}"
            ) from exc
        self.verify_attached()
        return memory_names, imports_names, backup_names, receipt_targets

    @staticmethod
    def _create_temp(directory_fd: int, prefix: str) -> tuple[int, str]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        for _ in range(128):
            name = f"{prefix}{secrets.token_hex(16)}.tmp"
            try:
                return os.open(name, flags, 0o600, dir_fd=directory_fd), name
            except FileExistsError:
                continue
        raise MemoryImportConflict("cannot allocate a unique memory import temp file")

    def atomic_write(
        self, directory_fd: int, name: str, content: bytes, *, prefix: str
    ) -> None:
        fd, temp_name = self._create_temp(directory_fd, prefix)
        identity = None
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                opened = os.fstat(handle.fileno())
                identity = (opened.st_dev, opened.st_ino)
            current = os.stat(temp_name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(current.st_mode) or (
                current.st_dev,
                current.st_ino,
            ) != identity:
                raise MemoryImportConflict("memory import temp file changed before publish")
            self.verify_attached()
            os.replace(
                temp_name,
                name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temp_name = ""
            directory_path = (
                self.mem_dir
                if directory_fd == self.mem_fd
                else self.imports_dir
                if directory_fd == self.imports_fd
                else self.backup_dir
            )
            _fsync_directory_fd(directory_fd, directory_path)
            self.verify_attached()
        finally:
            if fd >= 0:
                os.close(fd)
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=directory_fd)
                except OSError:
                    pass

    def _restore_displaced_no_replace(
        self, displaced_name: str, canonical_name: str
    ) -> None:
        if self.read_bytes(self.mem_fd, displaced_name) is None:
            raise MemoryImportConflict("displaced memory recovery file is missing")
        self.verify_attached()
        try:
            os.link(
                displaced_name,
                canonical_name,
                src_dir_fd=self.mem_fd,
                dst_dir_fd=self.mem_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return
        _fsync_directory_fd(self.mem_fd, self.mem_dir)

    def write_canonical_cas(
        self,
        canonical_name: str,
        displaced_name: str,
        content: bytes,
        expected_sha256: str,
    ) -> None:
        fd, temp_name = self._create_temp(self.mem_fd, ".mem_")
        identity = None
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                opened = os.fstat(handle.fileno())
                identity = (opened.st_dev, opened.st_ino)
            current_temp = os.stat(
                temp_name, dir_fd=self.mem_fd, follow_symlinks=False
            )
            if not stat.S_ISREG(current_temp.st_mode) or (
                current_temp.st_dev,
                current_temp.st_ino,
            ) != identity:
                raise MemoryImportConflict("memory import temp file changed before publish")

            displaced = self.read_bytes(self.mem_fd, displaced_name)
            canonical = self.read_bytes(self.mem_fd, canonical_name)
            if displaced is not None:
                if hashlib.sha256(displaced).hexdigest() != expected_sha256:
                    if canonical is None:
                        self._restore_displaced_no_replace(
                            displaced_name, canonical_name
                        )
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    )
                if canonical is not None:
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    )
            elif expected_sha256 == _MISSING_MEMORY_FILE_SHA256:
                if canonical is not None:
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    )
            else:
                self.verify_attached()
                try:
                    os.replace(
                        canonical_name,
                        displaced_name,
                        src_dir_fd=self.mem_fd,
                        dst_dir_fd=self.mem_fd,
                    )
                except FileNotFoundError as exc:
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    ) from exc
                _fsync_directory_fd(self.mem_fd, self.mem_dir)
                displaced = self.read_bytes(self.mem_fd, displaced_name)
                if (
                    displaced is None
                    or hashlib.sha256(displaced).hexdigest() != expected_sha256
                ):
                    if self.read_bytes(self.mem_fd, canonical_name) is None:
                        self._restore_displaced_no_replace(
                            displaced_name, canonical_name
                        )
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    )

            self.verify_attached()
            try:
                os.link(
                    temp_name,
                    canonical_name,
                    src_dir_fd=self.mem_fd,
                    dst_dir_fd=self.mem_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise MemoryImportConflict(
                    "memory changed after import prepare; refusing to overwrite user edits"
                ) from exc
            except OSError as publish_error:
                if self.read_bytes(self.mem_fd, displaced_name) is not None:
                    try:
                        self._restore_displaced_no_replace(
                            displaced_name, canonical_name
                        )
                    except BaseException as restore_error:
                        raise RuntimeError(
                            "memory publish failed and displaced recovery also failed: "
                            f"{restore_error}"
                        ) from publish_error
                raise
            # Persist the new canonical name before removing the already
            # durable temp source.
            _fsync_directory_fd(self.mem_fd, self.mem_dir)
            os.unlink(temp_name, dir_fd=self.mem_fd)
            temp_name = ""
            _fsync_directory_fd(self.mem_fd, self.mem_dir)
        finally:
            if fd >= 0:
                os.close(fd)
            if temp_name:
                try:
                    os.unlink(temp_name, dir_fd=self.mem_fd)
                except OSError:
                    pass

    def confirm_completed(
        self, canonical_name: str, receipt_name: str, content_sha256: str
    ) -> None:
        """Confirm the visible anchored state immediately before success."""
        self.verify_attached()
        canonical = self.read_bytes(self.mem_fd, canonical_name)
        receipt = self.read_receipt(self.imports_fd, receipt_name)
        if (
            canonical is None
            or hashlib.sha256(canonical).hexdigest() != content_sha256
            or receipt is None
            or receipt.get("state") != "completed"
            or receipt.get("content_sha256") != content_sha256
        ):
            raise MemoryImportConflict(
                "memory import durable state changed before completion"
            )
        _fsync_directory_fd(self.mem_fd, self.mem_dir)
        _fsync_directory_fd(self.imports_fd, self.imports_dir)
        self.verify_attached()


_ACTIVE_MEMORY_IMPORT_DIRS: ContextVar[Optional[_ImportDirectoryHandles]] = (
    ContextVar("active_memory_import_dirs", default=None)
)


@contextmanager
def _anchored_import_directories(
    mem_dir: Path,
    *,
    create_managed: bool = True,
    expected_home_identity: Optional[tuple[int, int]] = None,
    expected_mem_identity: Optional[tuple[int, int]] = None,
):
    with _ImportDirectoryHandles(
        mem_dir,
        create_managed=create_managed,
        expected_home_identity=expected_home_identity,
        expected_mem_identity=expected_mem_identity,
    ) as handles:
        handles.require_durable_scopes()
        token = _ACTIVE_MEMORY_IMPORT_DIRS.set(handles)
        try:
            yield handles
        except BaseException:
            raise
        else:
            handles.verify_attached()
        finally:
            _ACTIVE_MEMORY_IMPORT_DIRS.reset(token)


def _require_real_directory(
    path: Path, *, label: str, create: bool, secure_owner: bool = False
) -> None:
    if create:
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
    try:
        current = os.lstat(path)
    except FileNotFoundError as exc:
        raise MemoryImportConflict(f"{label} does not exist") from exc
    if not stat.S_ISDIR(current.st_mode):
        raise MemoryImportConflict(f"{label} must be a real directory, not a symlink")
    if secure_owner and (
        (
            hasattr(os, "geteuid")
            and current.st_uid != os.geteuid()
        )
        or stat.S_IMODE(current.st_mode) & 0o022
    ):
        raise MemoryImportConflict(
            f"{label} must be owned by this user and not group/world writable"
        )


def _existing_profile_directory_for_fsync_probe() -> Path:
    """Return a non-mutating probe directory on the profile's filesystem."""
    home = get_hermes_home()
    mem_dir = home / "memories"
    for path, label in (
        (home, "HERMES_HOME"),
        (mem_dir, "profile memories directory"),
    ):
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            continue
        except NotADirectoryError as exc:
            raise MemoryImportConflict(f"{label} is not a directory") from exc
        if not stat.S_ISDIR(mode):
            raise MemoryImportConflict(
                f"{label} must be a real directory, not a symlink"
            )
        _require_real_directory(
            path, label=label, create=False, secure_owner=True
        )
    if os.path.lexists(mem_dir):
        return mem_dir
    if os.path.lexists(home):
        return home
    candidate = home.parent
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    if not candidate.is_dir():
        raise MemoryImportConflict(
            "cannot find an existing profile parent directory for fsync probe"
        )
    return candidate


def _require_durable_profile_filesystem(
) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """Probe directory fsync support before import/reset can mutate the profile."""
    if os.name == "nt" or not _OPEN_SUPPORTS_DIR_FD:
        raise MemoryImportUnsupported(
            "durable curated-memory import/reset is unsupported on this platform"
        )
    home = get_hermes_home()
    mem_dir = home / "memories"
    home_identity = _path_identity(home) if os.path.lexists(home) else None
    mem_identity = _path_identity(mem_dir) if os.path.lexists(mem_dir) else None
    probe_path = _existing_profile_directory_for_fsync_probe()
    fd = -1
    try:
        fd = _open_resolved_directory_chain(
            probe_path, _ImportDirectoryHandles._directory_flags()
        )
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
            raise MemoryImportUnsupported(
                f"durable directory fsync is unsupported for {probe_path}"
            ) from exc
        raise
    finally:
        if fd >= 0:
            os.close(fd)
    try:
        if (
            home_identity is not None
            and _path_identity(home) != home_identity
        ) or (
            mem_identity is not None
            and _path_identity(mem_dir) != mem_identity
        ):
            raise MemoryImportConflict(
                "profile memory root changed during durability probe"
            )
    except OSError as exc:
        raise MemoryImportConflict(
            f"profile memory root changed during durability probe: {exc}"
        ) from exc
    if mem_identity is not None:
        with _ImportDirectoryHandles(
            mem_dir,
            create_managed=False,
            expected_home_identity=home_identity,
            expected_mem_identity=mem_identity,
        ) as handles:
            handles.require_durable_scopes()
    return home_identity, mem_identity


def _import_hardlink_cache_key(
    home_identity: tuple[int, int],
    mem_identity: tuple[int, int],
) -> tuple[str, tuple[int, int], tuple[int, int], int]:
    mem_dir = get_memory_dir()
    return (
        str(mem_dir),
        home_identity,
        mem_identity,
        os.stat(mem_dir, follow_symlinks=False).st_dev,
    )


def _durably_create_directory_chain(path: Path) -> None:
    """Create a missing absolute directory chain with durable parent entries."""
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise MemoryImportConflict(
            "profile directory creation requires an absolute path without dot components"
        )
    missing = []
    ancestor = path
    while not os.path.lexists(ancestor):
        missing.append(ancestor.name)
        parent = ancestor.parent
        if parent == ancestor:
            raise MemoryImportConflict(
                f"cannot find an existing parent for {path}"
            )
        ancestor = parent
    try:
        resolved_ancestor = ancestor.resolve(strict=True)
    except OSError as exc:
        raise MemoryImportConflict(
            f"cannot resolve profile parent directory: {exc}"
        ) from exc
    if not stat.S_ISDIR(os.lstat(resolved_ancestor).st_mode):
        raise MemoryImportConflict(
            f"profile parent {ancestor} must be a real directory"
        )

    parent_fd = _open_resolved_directory_chain(
        resolved_ancestor, _ImportDirectoryHandles._directory_flags()
    )
    current_path = resolved_ancestor
    try:
        ancestor_parent = resolved_ancestor.parent
        ancestor_parent_fd = _open_resolved_directory_chain(
            ancestor_parent, _ImportDirectoryHandles._directory_flags()
        )
        try:
            _fsync_directory_fd(ancestor_parent_fd, ancestor_parent)
            _fsync_directory_fd(parent_fd, resolved_ancestor)
        finally:
            os.close(ancestor_parent_fd)
        for name in reversed(missing):
            created = False
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            try:
                before_open = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
            except OSError as exc:
                raise MemoryImportConflict(
                    f"cannot inspect profile directory component {name}: {exc}"
                ) from exc
            try:
                child_fd = os.open(
                    name,
                    _ImportDirectoryHandles._directory_flags(),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                raise MemoryImportConflict(
                    f"profile directory component {name} is unsafe: {exc}"
                ) from exc
            child_path = current_path / name
            try:
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or (before_open.st_dev, before_open.st_ino)
                    != (opened.st_dev, opened.st_ino)
                ):
                    raise MemoryImportConflict(
                        f"profile directory component {name} changed before open"
                    )
                if (
                    hasattr(os, "geteuid")
                    and opened.st_uid != os.geteuid()
                ) or stat.S_IMODE(opened.st_mode) & 0o077:
                    origin = "created" if created else "pre-existing"
                    raise MemoryImportConflict(
                        f"{origin} profile directory component {name} is not "
                        "owned privately by this user"
                    )
                visible = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (visible.st_dev, visible.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    raise MemoryImportConflict(
                        f"profile directory component {name} changed after open"
                    )
                # A FileExists race may be another creator or residue from a
                # previous failed fsync. Re-run both barriers unconditionally.
                _fsync_directory_fd(parent_fd, current_path)
                _fsync_directory_fd(child_fd, child_path)
            except BaseException:
                os.close(child_fd)
                raise
            os.close(parent_fd)
            parent_fd = child_fd
            current_path = child_path
    finally:
        os.close(parent_fd)


def _require_profile_memory_snapshot(
    *,
    create: bool,
    expected_home_identity: Optional[tuple[int, int]] = None,
    expected_mem_identity: Optional[tuple[int, int]] = None,
) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    """Validate the profile root and capture identities for its secure open."""
    home = get_hermes_home()
    if create:
        _durably_create_directory_chain(home)
    _require_real_directory(
        home, label="HERMES_HOME", create=False, secure_owner=True
    )
    try:
        resolved_home = home.resolve(strict=True)
    except OSError as exc:
        raise MemoryImportConflict(f"cannot resolve HERMES_HOME: {exc}") from exc
    mem_dir = home / "memories"
    if create:
        _durably_create_directory_chain(mem_dir)
    _require_real_directory(
        mem_dir,
        label="profile memories directory",
        create=False,
        secure_owner=True,
    )
    try:
        resolved_memory = mem_dir.resolve(strict=True)
    except OSError as exc:
        raise MemoryImportConflict(
            f"cannot resolve profile memories directory: {exc}"
        ) from exc
    if resolved_memory != resolved_home / "memories":
        raise MemoryImportConflict("profile memories directory escapes HERMES_HOME")
    try:
        home_identity = _path_identity(home)
        mem_identity = _path_identity(mem_dir)
        # Resolve and identity checks must describe one snapshot.  A leaf swap
        # during validation is a conflict, not a new profile root to trust.
        changed = (
            home.resolve(strict=True) != resolved_home
            or mem_dir.resolve(strict=True) != resolved_memory
            or _path_identity(home) != home_identity
            or _path_identity(mem_dir) != mem_identity
        )
    except OSError as exc:
        raise MemoryImportConflict(
            f"profile memory root changed during validation: {exc}"
        ) from exc
    if changed:
        raise MemoryImportConflict("profile memory root changed during validation")
    if (
        expected_home_identity is not None
        and home_identity != expected_home_identity
    ) or (
        expected_mem_identity is not None
        and mem_identity != expected_mem_identity
    ):
        raise MemoryImportConflict("profile memory root changed before secure open")
    return mem_dir, home_identity, mem_identity


def _require_profile_memory_directory(*, create: bool) -> Path:
    """Return the profile memory root without traversing a symlinked child."""
    return _require_profile_memory_snapshot(create=create)[0]


def _require_managed_memory_directory(path: Path, *, create: bool) -> Path:
    """Validate every managed directory component below the memory root."""
    mem_dir = _require_profile_memory_directory(create=create)
    try:
        relative = path.relative_to(mem_dir)
    except ValueError as exc:
        raise MemoryImportConflict("managed memory directory escapes HERMES_HOME") from exc
    current = mem_dir
    for part in relative.parts:
        current /= part
        _require_real_directory(
            current,
            label=f"managed memory directory {current.name}",
            create=create,
            secure_owner=True,
        )
    expected = mem_dir.resolve(strict=True) / relative
    if path.resolve(strict=True) != expected:
        raise MemoryImportConflict("managed memory directory escapes HERMES_HOME")
    return path


def _read_bounded_regular_file_bytes(path: Path) -> Optional[bytes]:
    """Read one bounded regular file without following its leaf entry."""
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode):
        raise MemoryImportConflict(f"{path.name} must be a regular file")
    if before.st_size > MAX_CURATED_MEMORY_FILE_BYTES:
        raise MemoryImportConflict(f"{path.name} exceeds the memory file size limit")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MemoryImportConflict(f"cannot safely open {path.name}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise MemoryImportConflict(f"{path.name} must be a regular file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise MemoryImportConflict(f"{path.name} changed while it was opened")
        if opened.st_size > MAX_CURATED_MEMORY_FILE_BYTES:
            raise MemoryImportConflict(f"{path.name} exceeds the memory file size limit")
        chunks = []
        total = 0
        while True:
            chunk = os.read(
                fd, min(64 << 10, MAX_CURATED_MEMORY_FILE_BYTES + 1 - total)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_CURATED_MEMORY_FILE_BYTES:
                raise MemoryImportConflict(
                    f"{path.name} exceeds the memory file size limit"
                )
        after = os.fstat(fd)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise MemoryImportConflict(f"{path.name} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _read_bounded_regular_file_text(path: Path) -> Optional[str]:
    raw = _read_bounded_regular_file_bytes(path)
    if raw is None:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemoryImportConflict(f"{path.name} is not valid UTF-8") from exc


def _is_real_directory(path: Path) -> bool:
    """Return True only for a directory entry that is not a symlink."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _require_optional_real_directory(path: Path, *, label: str) -> bool:
    """Return False only for ENOENT; reject every unsafe existing leaf."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return False
    except NotADirectoryError as exc:
        raise MemoryImportConflict(f"{label} is not a directory") from exc
    if not stat.S_ISDIR(mode):
        raise MemoryImportConflict(
            f"{label} must be a real directory, not a symlink"
        )
    return True


def _validate_optional_managed_memory_directories(mem_dir: Path) -> None:
    imports_dir = mem_dir / ".imports"
    if not _require_optional_real_directory(
        imports_dir, label="managed memory directory .imports"
    ):
        return
    _require_optional_real_directory(
        imports_dir / "backups", label="managed memory directory backups"
    )


def _read_import_receipt_no_follow(path: Path) -> Optional[Dict[str, Any]]:
    """Read one regular receipt without following a symlink receipt."""
    try:
        raw = _read_bounded_regular_file_bytes(path)
    except (OSError, MemoryImportConflict):
        return None
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def _target_has_import_state(mem_dir: Path, target: str) -> bool:
    filename = _MEMORY_TARGET_FILES[target]
    backup_prefix = f"{target}-"
    displaced_prefix = f".{filename}."
    imports_dir = mem_dir / ".imports"
    backup_dir = imports_dir / "backups"

    if _is_real_directory(mem_dir):
        with os.scandir(mem_dir) as entries:
            if any(
                (
                    entry.name.startswith(displaced_prefix)
                    and entry.name.endswith(".displaced")
                )
                or entry.name.startswith(f"{filename}.bak.")
                for entry in entries
            ):
                return True
    if _is_real_directory(backup_dir):
        with os.scandir(backup_dir) as entries:
            if any(
                entry.name.startswith(backup_prefix) and entry.name.endswith(".bak")
                for entry in entries
            ):
                return True
    if _is_real_directory(imports_dir):
        with os.scandir(imports_dir) as entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                if not stat.S_ISREG(
                    entry.stat(follow_symlinks=False).st_mode
                ):
                    raise MemoryImportConflict(
                        f"managed memory receipt {entry.name} must be a regular file"
                    )
                receipt = _read_import_receipt_no_follow(Path(entry.path))
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("target") not in {"memory", "user"}
                ):
                    raise MemoryImportConflict(
                        f"managed memory receipt {entry.name} has no valid target"
                    )
                if receipt.get("target") == target:
                    return True
    return False


def _has_any_managed_import_leaf_state(mem_dir: Path) -> bool:
    """Treat every regular leaf in import-owned directories as reset-all state."""
    imports_dir = mem_dir / ".imports"
    if not _is_real_directory(imports_dir):
        return False
    found = False
    with os.scandir(imports_dir) as entries:
        for entry in entries:
            if entry.name == "backups" and stat.S_ISDIR(
                entry.stat(follow_symlinks=False).st_mode
            ):
                continue
            if not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
                raise MemoryImportConflict(
                    f"managed memory import leaf {entry.name} must be a regular file"
                )
            found = True
    backup_dir = imports_dir / "backups"
    if _is_real_directory(backup_dir):
        with os.scandir(backup_dir) as entries:
            for entry in entries:
                if not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
                    raise MemoryImportConflict(
                        f"managed memory backup leaf {entry.name} must be a regular file"
                    )
                found = True
    return found


def _has_import_temp_state(mem_dir: Path) -> bool:
    """Return whether a crash may have left profile-local import plaintext."""
    imports_dir = mem_dir / ".imports"
    backup_dir = imports_dir / "backups"
    locations = (
        (mem_dir, _IMPORT_LINK_PROBE_PREFIX, ""),
        (mem_dir, ".mem_", ".tmp"),
        (mem_dir, ".drift_", ".tmp"),
        (mem_dir, ".reset_receipt_", ".tmp"),
        (mem_dir, _RESET_RECEIPT_PREFIX, ""),
        (mem_dir, _RESET_STAGE_PREFIX, ""),
        (imports_dir, ".receipt_", ".tmp"),
        (imports_dir, _RESET_STAGE_PREFIX, ""),
        (backup_dir, ".backup_", ".tmp"),
        (backup_dir, _RESET_STAGE_PREFIX, ""),
    )
    for directory, prefix, suffix in locations:
        if not _is_real_directory(directory):
            continue
        with os.scandir(directory) as entries:
            if any(
                entry.name.startswith(prefix) and entry.name.endswith(suffix)
                for entry in entries
            ):
                return True
    return False


def _validate_reset_lock(path: Path) -> None:
    lock_path = path.with_suffix(path.suffix + ".lock")
    if os.path.lexists(lock_path) and not stat.S_ISREG(os.lstat(lock_path).st_mode):
        raise MemoryImportConflict(f"refusing to follow unsafe reset lock {lock_path}")


def _reset_scope_fd(handles: _ImportDirectoryHandles, scope: str) -> int:
    mapping = {
        "memory": handles.mem_fd,
        "imports": handles.imports_fd,
        "backups": handles.backup_fd,
    }
    fd = mapping.get(scope, -1)
    if fd < 0:
        raise MemoryImportConflict(f"reset source directory is unavailable: {scope}")
    return fd


def _reset_entry_stat(directory_fd: int, name: str):
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _reset_fsync_scope(handles: _ImportDirectoryHandles, scope: str) -> None:
    fd = _reset_scope_fd(handles, scope)
    path = (
        handles.mem_dir
        if scope == "memory"
        else handles.imports_dir
        if scope == "imports"
        else handles.backup_dir
    )
    _fsync_directory_fd(fd, path)


def _reset_stage_scope(entry: Dict[str, str]) -> str:
    return entry.get("stage_scope", "memory")


def _reset_copy_regular_no_follow(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    destination_path: Path,
) -> None:
    """Create one bounded, durable copy without following either leaf."""
    try:
        expected = os.stat(
            source_name, dir_fd=source_fd, follow_symlinks=False
        )
    except OSError as exc:
        raise MemoryImportConflict(
            f"cannot inspect reset copy source {source_name}: {exc}"
        ) from exc
    if not stat.S_ISREG(expected.st_mode):
        raise MemoryImportConflict(
            f"reset copy source {source_name} must be a regular file"
        )
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        read_flags |= os.O_NONBLOCK
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        write_flags |= os.O_NOFOLLOW
    source_handle = -1
    destination_handle = -1
    destination_created = False
    try:
        source_handle = os.open(source_name, read_flags, dir_fd=source_fd)
        opened = os.fstat(source_handle)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise MemoryImportConflict(
                f"reset copy source {source_name} changed before open"
            )
        destination_handle = os.open(
            destination_name,
            write_flags,
            0o600,
            dir_fd=destination_fd,
        )
        destination_created = True
        source_mode = stat.S_IMODE(expected.st_mode) & 0o777
        os.fchmod(destination_handle, source_mode or 0o600)
        total = 0
        while True:
            chunk = os.read(source_handle, min(64 << 10, expected.st_size + 1 - total))
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_handle, view)
                if written <= 0:
                    raise OSError("short write during reset copy")
                view = view[written:]
            total += len(chunk)
        after = os.fstat(source_handle)
        if (
            total != expected.st_size
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
                expected.st_mtime_ns,
            )
        ):
            raise MemoryImportConflict(
                f"reset copy source {source_name} changed during copy"
            )
        os.fsync(destination_handle)
    except BaseException:
        if destination_handle >= 0:
            os.close(destination_handle)
            destination_handle = -1
        if source_handle >= 0:
            os.close(source_handle)
            source_handle = -1
        if destination_created:
            try:
                os.unlink(destination_name, dir_fd=destination_fd)
                _fsync_directory_fd(destination_fd, destination_path)
            except OSError:
                pass
        raise
    finally:
        if destination_handle >= 0:
            os.close(destination_handle)
        if source_handle >= 0:
            os.close(source_handle)


def _reset_link_or_copy(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    destination_path: Path,
) -> None:
    try:
        os.link(
            source_name,
            destination_name,
            src_dir_fd=source_fd,
            dst_dir_fd=destination_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        if exc.errno not in _LINK_COPY_FALLBACK_ERRNOS:
            raise
        _reset_copy_regular_no_follow(
            source_fd,
            source_name,
            destination_fd,
            destination_name,
            destination_path,
        )


def _reset_compare_regular_files(
    first_fd: int, first_name: str, second_fd: int, second_name: str
) -> str:
    """Compare bounded regular files without loading reset-sized data in memory."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    handles = []
    stats = []
    try:
        for directory_fd, name in (
            (first_fd, first_name),
            (second_fd, second_name),
        ):
            expected = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(expected.st_mode):
                raise MemoryImportConflict(
                    f"reset copy residue {name} is not a regular file"
                )
            handle = os.open(name, flags, dir_fd=directory_fd)
            opened = os.fstat(handle)
            if (opened.st_dev, opened.st_ino) != (
                expected.st_dev,
                expected.st_ino,
            ):
                os.close(handle)
                raise MemoryImportConflict(
                    f"reset copy residue {name} changed before open"
                )
            handles.append(handle)
            stats.append(expected)
        common_size = min(stats[0].st_size, stats[1].st_size)
        offset = 0
        while offset < common_size:
            size = min(64 << 10, common_size - offset)
            if os.pread(handles[0], size, offset) != os.pread(
                handles[1], size, offset
            ):
                return "different"
            offset += size
        for handle, expected in zip(handles, stats):
            after = os.fstat(handle)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ) != (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
                expected.st_mtime_ns,
            ):
                raise MemoryImportConflict("reset copy residue changed during compare")
        if stats[0].st_size == stats[1].st_size:
            return "equal"
        return (
            "first_prefix"
            if stats[0].st_size < stats[1].st_size
            else "second_prefix"
        )
    finally:
        for handle in handles:
            os.close(handle)


def _reset_move_no_replace(
    handles: _ImportDirectoryHandles,
    source_scope: str,
    source_name: str,
    stage_name: str,
    stage_scope: Optional[str] = None,
) -> None:
    atomic_stage = stage_scope is not None
    if stage_scope is None:
        # Legacy callers and receipts staged into memories by link/copy.
        stage_scope = "memory"
    source_fd = _reset_scope_fd(handles, source_scope)
    stage_fd = _reset_scope_fd(handles, stage_scope)
    if _reset_entry_stat(stage_fd, stage_name) is not None:
        raise MemoryImportConflict(f"reset stage already exists: {stage_name}")
    handles.verify_attached()
    if atomic_stage and stage_scope == source_scope:
        expected = _reset_entry_stat(source_fd, source_name)
        if expected is None or not stat.S_ISREG(expected.st_mode):
            raise MemoryImportConflict(
                f"reset source {source_name} must be a regular file"
            )
        os.rename(
            source_name,
            stage_name,
            src_dir_fd=source_fd,
            dst_dir_fd=stage_fd,
        )
        staged = _reset_entry_stat(stage_fd, stage_name)
        if staged is None or (staged.st_dev, staged.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise MemoryImportConflict(
                f"reset stage changed during atomic move: {stage_name}"
            )
        _reset_fsync_scope(handles, stage_scope)
        handles.verify_attached()
        return
    _reset_link_or_copy(
        source_fd,
        source_name,
        stage_fd,
        stage_name,
        (
            handles.mem_dir
            if stage_scope == "memory"
            else handles.imports_dir
            if stage_scope == "imports"
            else handles.backup_dir
        ),
    )
    # The staged dentry must be durable before the source dentry can be
    # durably removed.  This ordering matters when source and stage live in
    # different directories: fsyncing the source first can lose both names on
    # power failure even though link(2) succeeded in memory.
    _reset_fsync_scope(handles, stage_scope)
    os.unlink(source_name, dir_fd=source_fd)
    _reset_fsync_scope(handles, source_scope)
    handles.verify_attached()


def _reset_restore_plan(
    handles: _ImportDirectoryHandles, plan: List[Dict[str, str]]
) -> None:
    for entry in reversed(plan):
        source_fd = _reset_scope_fd(handles, entry["scope"])
        stage_scope = _reset_stage_scope(entry)
        stage_fd = _reset_scope_fd(handles, stage_scope)
        source = _reset_entry_stat(source_fd, entry["name"])
        stage = _reset_entry_stat(stage_fd, entry["stage"])
        if stage is None:
            if source is None:
                raise MemoryImportConflict(
                    f"reset recovery lost both source and stage: {entry['name']}"
                )
            continue
        if "stage_scope" in entry and stage_scope == entry["scope"]:
            if source is not None:
                raise MemoryImportConflict(
                    "cannot atomically restore reset source occupied by another "
                    f"entry: {entry['name']}"
                )
            os.rename(
                entry["stage"],
                entry["name"],
                src_dir_fd=stage_fd,
                dst_dir_fd=source_fd,
            )
            restored = _reset_entry_stat(source_fd, entry["name"])
            if restored is None or (restored.st_dev, restored.st_ino) != (
                stage.st_dev,
                stage.st_ino,
            ):
                raise MemoryImportConflict(
                    f"reset restore changed during atomic move: {entry['name']}"
                )
            _reset_fsync_scope(handles, entry["scope"])
            continue
        if source is not None:
            if (source.st_dev, source.st_ino) != (stage.st_dev, stage.st_ino):
                relationship = _reset_compare_regular_files(
                    source_fd,
                    entry["name"],
                    stage_fd,
                    entry["stage"],
                )
                if relationship == "equal":
                    pass
                elif relationship == "second_prefix":
                    # A move-side copy was interrupted. The original source is
                    # authoritative; discard only the incomplete stage.
                    _reset_fsync_scope(handles, entry["scope"])
                    os.unlink(entry["stage"], dir_fd=stage_fd)
                    _reset_fsync_scope(handles, stage_scope)
                    continue
                elif relationship == "first_prefix":
                    # Prefix shape alone cannot prove this is our interrupted
                    # restore: a non-cooperating writer may have created a
                    # legitimate new source after the crash. Preserve both.
                    raise MemoryImportConflict(
                        "cannot restore over a different source that is a prefix "
                        f"of staged data: {entry['name']}"
                    )
                else:
                    raise MemoryImportConflict(
                        "cannot restore reset source occupied by different data: "
                        f"{entry['name']}"
                    )
        if source is None:
            _reset_link_or_copy(
                stage_fd,
                entry["stage"],
                source_fd,
                entry["name"],
                (
                    handles.mem_dir
                    if entry["scope"] == "memory"
                    else handles.imports_dir
                    if entry["scope"] == "imports"
                    else handles.backup_dir
                ),
            )
        # Confirm the restored source name before deleting the only durable
        # staged name.  If either fsync fails, the staging receipt and at least
        # one linked name remain for the next recovery attempt.
        _reset_fsync_scope(handles, entry["scope"])
        os.unlink(entry["stage"], dir_fd=stage_fd)
        _reset_fsync_scope(handles, stage_scope)
    handles.verify_attached()


def _write_reset_receipt(
    handles: _ImportDirectoryHandles, receipt_name: str, receipt: Dict[str, Any]
) -> None:
    handles.atomic_write(
        handles.mem_fd,
        receipt_name,
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        prefix=".reset_receipt_",
    )


def _cleanup_isolated_reset(
    handles: _ImportDirectoryHandles,
    receipt_name: str,
    plan: List[Dict[str, str]],
) -> bool:
    cleanup_pending = False
    for entry in plan:
        try:
            stage_scope = _reset_stage_scope(entry)
            stage_fd = _reset_scope_fd(handles, stage_scope)
            if _reset_entry_stat(stage_fd, entry["stage"]) is not None:
                os.unlink(entry["stage"], dir_fd=stage_fd)
                _reset_fsync_scope(handles, stage_scope)
        except OSError:
            cleanup_pending = True
    try:
        _fsync_directory_fd(handles.mem_fd, handles.mem_dir)
    except OSError:
        # Keep the isolated receipt so a later reset retries the purge.
        return True
    if cleanup_pending:
        return True
    try:
        os.unlink(receipt_name, dir_fd=handles.mem_fd)
    except FileNotFoundError:
        pass
    except OSError:
        return True
    try:
        _fsync_directory_fd(handles.mem_fd, handles.mem_dir)
    except OSError:
        # The receipt removal was not durably confirmed. Recreate a marker so
        # callers can observe cleanup_pending and a retry has work to recover.
        try:
            _write_reset_receipt(
                handles,
                receipt_name,
                {"version": 1, "state": "isolated", "plan": plan},
            )
        except (OSError, MemoryImportConflict):
            pass
        return True
    handles.verify_attached()
    return False


def _validate_reset_receipt(
    value: Any,
    *,
    allow_legacy: bool = False,
    allow_cross_scope: bool = False,
) -> tuple[str, List[Dict[str, str]]]:
    if not isinstance(value, dict) or value.get("state") not in {"staging", "isolated"}:
        raise MemoryImportConflict("memory reset receipt is invalid")
    raw_plan = value.get("plan")
    if not isinstance(raw_plan, list):
        raise MemoryImportConflict("memory reset receipt plan is invalid")
    plan = []
    for raw in raw_plan:
        if not isinstance(raw, dict):
            raise MemoryImportConflict("memory reset receipt entry is invalid")
        scope, stage_scope, name, stage, label = (
            raw.get("scope"),
            raw.get("stage_scope", "memory"),
            raw.get("name"),
            raw.get("stage"),
            raw.get("label"),
        )
        has_stage_scope = "stage_scope" in raw
        if (
            scope not in {"memory", "imports", "backups"}
            or stage_scope not in {"memory", "imports", "backups"}
            or (not has_stage_scope and not allow_legacy)
            or (
                has_stage_scope
                and stage_scope != scope
                and not allow_cross_scope
            )
            or not all(isinstance(item, str) and item not in {"", ".", ".."}
                       and "/" not in item and "\\" not in item
                       for item in (name, stage))
            or not isinstance(label, str)
            or not stage.startswith(_RESET_STAGE_PREFIX)
            or (
                scope == "memory"
                and name in {
                    f"{_MEMORY_TRANSACTION_LOCK}.lock",
                    "MEMORY.md.lock",
                    "USER.md.lock",
                }
            )
        ):
            raise MemoryImportConflict("memory reset receipt entry is unsafe")
        entry = {
            "scope": scope,
            "name": name,
            "stage": stage,
            "label": label,
        }
        if has_stage_scope:
            entry["stage_scope"] = stage_scope
        plan.append(entry)
    return value["state"], plan


def _reset_all_receipt_entries(
    handles: _ImportDirectoryHandles, memory_names: List[str]
) -> List[Dict[str, str]]:
    """Safely enumerate receipt-owned leaves without replaying recovery."""
    entries: List[Dict[str, str]] = []
    for receipt_name in memory_names:
        if not (
            receipt_name.startswith(_RESET_RECEIPT_PREFIX)
            and receipt_name.endswith(".json")
        ):
            continue
        try:
            receipt = handles.read_receipt(handles.mem_fd, receipt_name)
        except MemoryImportConflict as exc:
            raise MemoryImportConflict(
                f"cannot safely enumerate reset-all receipt {receipt_name}"
            ) from exc
        if not isinstance(receipt, dict) or not isinstance(
            receipt.get("plan"), list
        ):
            raise MemoryImportConflict(
                f"cannot safely enumerate reset-all receipt {receipt_name}"
            )
        try:
            _state, plan = _validate_reset_receipt(
                receipt,
                allow_legacy=True,
                allow_cross_scope=True,
            )
        except MemoryImportConflict as exc:
            # Once a receipt exposes a plan, silently discarding only the
            # receipt could orphan a private opaque source and falsely report
            # an empty profile. Preserve everything and fail explicitly.
            raise MemoryImportConflict(
                f"cannot safely enumerate reset-all receipt {receipt_name}"
            ) from exc
        entries.extend(plan)
    return entries


def _reset_receipt_targets(receipt: Any) -> tuple[str, ...]:
    if not isinstance(receipt, dict):
        raise MemoryImportConflict("memory reset receipt is invalid")
    raw_targets = receipt.get("targets")
    if (
        not isinstance(raw_targets, list)
        or not raw_targets
        or any(
            not isinstance(item, str) or item not in {"memory", "user"}
            for item in raw_targets
        )
        or len(raw_targets) != len(set(raw_targets))
    ):
        raise MemoryImportConflict("memory reset receipt targets are invalid")
    return tuple(raw_targets)


def _reset_receipts_require_forward_all(
    handles: _ImportDirectoryHandles, memory_names: List[str]
) -> bool:
    """Return the durable direction of any unfinished reset-all transaction."""
    forward_all = False
    for receipt_name in memory_names:
        if not (
            receipt_name.startswith(_RESET_RECEIPT_PREFIX)
            and receipt_name.endswith(".json")
        ):
            continue
        receipt = handles.read_receipt(handles.mem_fd, receipt_name)
        _validate_reset_receipt(receipt)
        original_targets = _reset_receipt_targets(receipt)
        if set(original_targets) == {"memory", "user"}:
            forward_all = True
    return forward_all


def _recover_reset_transactions(
    handles: _ImportDirectoryHandles,
    memory_names: List[str],
    *,
    purge_all: bool = False,
) -> bool:
    # Reset-all is a forward-only privacy purge. Receipt-owned leaves are
    # enumerated separately for the fresh deletion plan; never roll back here.
    if purge_all:
        return False
    cleanup_pending = False
    receipt_names = [
        name
        for name in memory_names
        if name.startswith(_RESET_RECEIPT_PREFIX) and name.endswith(".json")
    ]
    for receipt_name in receipt_names:
        receipt = handles.read_receipt(handles.mem_fd, receipt_name)
        state, plan = _validate_reset_receipt(receipt)
        _reset_receipt_targets(receipt)
        if state == "staging":
            _reset_restore_plan(handles, plan)
            os.unlink(receipt_name, dir_fd=handles.mem_fd)
            _fsync_directory_fd(handles.mem_fd, handles.mem_dir)
        else:
            cleanup_pending |= _cleanup_isolated_reset(
                handles, receipt_name, plan
            )
    return cleanup_pending


def _validate_reset_candidate_types(
    handles: _ImportDirectoryHandles,
    target: str,
    memory_names: List[str],
    imports_names: List[str],
    backup_names: List[str],
) -> None:
    """Reject unsafe managed leaves before reset creates lock/receipt files."""
    targets = ("memory", "user") if target == "all" else (target,)
    candidates = set()
    for item in targets:
        filename = _MEMORY_TARGET_FILES[item]
        candidates.add(("memory", filename))
        for name in memory_names:
            if (
                name.startswith(f".{filename}.")
                and name.endswith(".displaced")
            ) or name.startswith(f"{filename}.bak."):
                candidates.add(("memory", name))
        for name in backup_names:
            if name.startswith(f"{item}-") and name.endswith(".bak"):
                candidates.add(("backups", name))
    for name in memory_names:
        if (
            (name.startswith((".mem_", ".drift_", ".reset_receipt_"))
             and name.endswith(".tmp"))
            or name.startswith(_IMPORT_LINK_PROBE_PREFIX)
            or name.startswith((_RESET_STAGE_PREFIX, _RESET_RECEIPT_PREFIX))
        ):
            candidates.add(("memory", name))
    for name in imports_names:
        if (
            (target == "all" and name != "backups")
            or name.startswith(_RESET_STAGE_PREFIX)
            or name.endswith(".json")
            or (name.startswith(".receipt_") and name.endswith(".tmp"))
        ):
            candidates.add(("imports", name))
    for name in backup_names:
        if (
            target == "all"
            or name.startswith(_RESET_STAGE_PREFIX)
            or (name.startswith(".backup_") and name.endswith(".tmp"))
        ):
            candidates.add(("backups", name))
    for scope, name in candidates:
        current = _reset_entry_stat(_reset_scope_fd(handles, scope), name)
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise MemoryImportConflict(
                f"reset state {name} must be a regular file"
            )


def _validate_reset_receipts(
    target: str,
    imports_names: List[str],
    receipt_targets: Dict[str, Optional[Dict[str, Any]]],
) -> None:
    """Fail closed when a single-target reset cannot classify a receipt."""
    if target == "all":
        return
    for name in imports_names:
        if not name.endswith(".json"):
            continue
        receipt = receipt_targets.get(name)
        if (
            not isinstance(receipt, dict)
            or receipt.get("target") not in {"memory", "user"}
        ):
            raise MemoryImportConflict(
                f"managed memory receipt {name} has no valid target"
            )


def curated_memory_has_state(target: str) -> bool:
    """Return whether a reset target has canonical or managed import state."""
    if target not in {"all", "memory", "user"}:
        raise ValueError("target must be all, memory, or user")
    home = get_hermes_home()
    mem_dir = home / "memories"
    try:
        home_mode = os.lstat(home).st_mode
    except FileNotFoundError:
        return False
    except NotADirectoryError as exc:
        raise MemoryImportConflict("HERMES_HOME is not a directory") from exc
    if not stat.S_ISDIR(home_mode):
        raise MemoryImportConflict(
            "HERMES_HOME must be a real directory, not a symlink"
        )
    try:
        memory_mode = os.lstat(mem_dir).st_mode
    except FileNotFoundError:
        return False
    except NotADirectoryError as exc:
        raise MemoryImportConflict(
            "profile memories directory is not a directory"
        ) from exc
    if not stat.S_ISDIR(memory_mode):
        raise MemoryImportConflict(
            "profile memories directory must be a real directory, not a symlink"
        )
    _validate_optional_managed_memory_directories(mem_dir)
    if _has_import_temp_state(mem_dir):
        return True
    if target == "all" and _has_any_managed_import_leaf_state(mem_dir):
        return True
    targets = ("memory", "user") if target == "all" else (target,)
    for item in targets:
        if os.path.lexists(mem_dir / _MEMORY_TARGET_FILES[item]):
            return True
        if _target_has_import_state(mem_dir, item):
            return True
    return False


def reset_curated_memory(target: str) -> Dict[str, Any]:
    """Permanently unlink canonical and profile-local import state for a target.

    Receipt-provided paths are deliberately ignored. Managed backup and
    displaced names are discovered only in fixed profile-local directories,
    and directory symlinks are never traversed. Managed leaves must be regular
    files; directories, symlinks, and special files fail closed before receipt
    creation so reset can never report success while unsafe state remains.
    """
    if target not in {"all", "memory", "user"}:
        raise ValueError("target must be all, memory, or user")
    expected_home_identity, expected_mem_identity = (
        _require_durable_profile_filesystem()
    )
    mem_dir = get_memory_dir()
    memory_exists = _require_optional_real_directory(
        mem_dir, label="profile memories directory"
    )
    if not memory_exists:
        if expected_mem_identity is not None:
            raise MemoryImportConflict(
                "profile memories directory changed after durability probe"
            )
        if expected_home_identity is not None:
            try:
                home_unchanged = (
                    _path_identity(get_hermes_home()) == expected_home_identity
                )
            except OSError as exc:
                raise MemoryImportConflict(
                    f"HERMES_HOME changed after durability probe: {exc}"
                ) from exc
            if not home_unchanged:
                raise MemoryImportConflict(
                    "HERMES_HOME changed after durability probe"
                )
        return {"deleted": [], "targets": [], "status": "completed"}
    mem_dir, home_identity, mem_identity = _require_profile_memory_snapshot(
        create=False,
        expected_home_identity=expected_home_identity,
        expected_mem_identity=expected_mem_identity,
    )

    deleted: List[str] = []

    transaction_path = mem_dir / _MEMORY_TRANSACTION_LOCK
    target_paths = [mem_dir / _MEMORY_TARGET_FILES[item] for item in ("memory", "user")]
    for lock_target in [transaction_path, *target_paths]:
        _validate_reset_lock(lock_target)

    # Import/reset use the transaction lock. All enumeration, reads, unlinks,
    # and fsyncs remain relative to the same no-follow directory descriptors.
    # Swapping any visible directory can make reset fail, but cannot redirect a
    # deletion into the replacement tree.
    with _anchored_import_directories(
        mem_dir,
        create_managed=False,
        expected_home_identity=home_identity,
        expected_mem_identity=mem_identity,
    ) as reset_dirs, ExitStack() as stack:
        (
            initial_memory_names,
            initial_imports_names,
            initial_backup_names,
            initial_receipt_targets,
        ) = reset_dirs.preflight_reset(
            parse_import_receipts=False
        )
        effective_target = target
        if target != "all" and _reset_receipts_require_forward_all(
            reset_dirs, initial_memory_names
        ):
            effective_target = "all"
        targets = (
            ("memory", "user")
            if effective_target == "all"
            else (effective_target,)
        )
        if effective_target != "all":
            (
                initial_memory_names,
                initial_imports_names,
                initial_backup_names,
                initial_receipt_targets,
            ) = reset_dirs.preflight_reset(parse_import_receipts=True)
        _validate_reset_candidate_types(
            reset_dirs,
            effective_target,
            initial_memory_names,
            initial_imports_names,
            initial_backup_names,
        )
        _validate_reset_receipts(
            effective_target, initial_imports_names, initial_receipt_targets
        )
        if effective_target == "all":
            # Reject an unenumerable reset receipt before lock files or a new
            # transaction can be created. The same bounded parse is repeated
            # under the transaction locks below to bind the deletion plan.
            _reset_all_receipt_entries(reset_dirs, initial_memory_names)
        stack.enter_context(
            MemoryStore._file_lock(transaction_path, create_parent=False)
        )
        for path in target_paths:
            stack.enter_context(MemoryStore._file_lock(path, create_parent=False))

        reset_dirs.verify_attached()
        # Preflight every directory and receipt needed to decide this reset
        # before the first unlink. A real managed directory that exists but is
        # unreadable must fail the whole reset, never look like an empty one.
        (
            memory_names,
            imports_names,
            backup_names,
            receipt_targets,
        ) = reset_dirs.preflight_reset(
            parse_import_receipts=effective_target != "all"
        )
        if effective_target != "all" and _reset_receipts_require_forward_all(
            reset_dirs, memory_names
        ):
            effective_target = "all"
            targets = ("memory", "user")
            receipt_targets = {}
        _validate_reset_candidate_types(
            reset_dirs,
            effective_target,
            memory_names,
            imports_names,
            backup_names,
        )
        _validate_reset_receipts(
            effective_target, imports_names, receipt_targets
        )
        if _recover_reset_transactions(
            reset_dirs,
            memory_names,
            purge_all=effective_target == "all",
        ):
            return {
                "deleted": [],
                "targets": list(targets),
                "status": "cleanup_pending",
            }
        (
            memory_names,
            imports_names,
            backup_names,
            receipt_targets,
        ) = reset_dirs.preflight_reset(
            parse_import_receipts=effective_target != "all"
        )
        reset_all_entries = (
            _reset_all_receipt_entries(reset_dirs, memory_names)
            if effective_target == "all"
            else []
        )

        transaction_id = secrets.token_hex(16)
        plan: List[Dict[str, str]] = []
        planned = set()

        def add_plan(scope: str, name: str, label: str) -> None:
            key = (scope, name)
            if key in planned:
                return
            directory_fd = _reset_scope_fd(reset_dirs, scope)
            current = _reset_entry_stat(directory_fd, name)
            if current is None:
                return
            if not stat.S_ISREG(current.st_mode):
                raise MemoryImportConflict(
                    f"reset state {label} must be a regular file"
                )
            planned.add(key)
            plan.append({
                "scope": scope,
                "stage_scope": scope,
                "name": name,
                "stage": f"{_RESET_STAGE_PREFIX}{transaction_id}_{len(plan)}",
                "label": label,
            })

        for entry in reset_all_entries:
            add_plan(entry["scope"], entry["name"], entry["label"])
            stage_scope = _reset_stage_scope(entry)
            add_plan(stage_scope, entry["stage"], entry["stage"])

        for item in targets:
            filename = _MEMORY_TARGET_FILES[item]
            add_plan("memory", filename, filename)
            displaced_prefix = f".{filename}."
            for name in memory_names:
                if (
                    name.startswith(displaced_prefix)
                    and name.endswith(".displaced")
                ) or name.startswith(f"{filename}.bak."):
                    add_plan("memory", name, name)
            for name in backup_names:
                if name.startswith(f"{item}-") and name.endswith(".bak"):
                    add_plan(
                        "backups", name, str(Path(".imports") / "backups" / name)
                    )
            for name, receipt in receipt_targets.items():
                if receipt is not None and receipt.get("target") == item:
                    add_plan("imports", name, str(Path(".imports") / name))

        if effective_target == "all":
            for name in imports_names:
                if name != "backups":
                    add_plan("imports", name, str(Path(".imports") / name))
            for name in backup_names:
                add_plan(
                    "backups",
                    name,
                    str(Path(".imports") / "backups" / name),
                )

        for name in memory_names:
            if name.startswith(_IMPORT_LINK_PROBE_PREFIX):
                add_plan("memory", name, name)

        for scope, names, prefix, relative_dir in (
            ("memory", memory_names, ".mem_", Path(".")),
            ("memory", memory_names, ".drift_", Path(".")),
            ("memory", memory_names, ".reset_receipt_", Path(".")),
            ("imports", imports_names, ".receipt_", Path(".imports")),
            (
                "backups", backup_names, ".backup_",
                Path(".imports") / "backups",
            ),
        ):
            for name in names:
                if name.startswith(prefix) and name.endswith(".tmp"):
                    add_plan(scope, name, str(relative_dir / name))
        for name in memory_names:
            if name.startswith(_RESET_STAGE_PREFIX):
                add_plan("memory", name, name)
            elif name.startswith(_RESET_RECEIPT_PREFIX) and (
                effective_target == "all" or not name.endswith(".json")
            ):
                add_plan("memory", name, name)
        for scope, names in (
            ("imports", imports_names),
            ("backups", backup_names),
        ):
            for name in names:
                if name.startswith(_RESET_STAGE_PREFIX):
                    add_plan(scope, name, name)

        receipt_name = f"{_RESET_RECEIPT_PREFIX}{transaction_id}.json"
        receipt = {
            "version": 1,
            "state": "staging",
            "targets": list(targets),
            "plan": plan,
            "created_at": time.time(),
        }
        _write_reset_receipt(reset_dirs, receipt_name, receipt)
        try:
            for entry in plan:
                _reset_move_no_replace(
                    reset_dirs,
                    entry["scope"],
                    entry["name"],
                    entry["stage"],
                    _reset_stage_scope(entry),
                )
            reset_dirs.verify_attached()
            for scope in {entry["scope"] for entry in plan} | {"memory"}:
                _reset_fsync_scope(reset_dirs, scope)
            receipt["state"] = "isolated"
            receipt["isolated_at"] = time.time()
            _write_reset_receipt(reset_dirs, receipt_name, receipt)
            reset_dirs.verify_attached()
        except BaseException as reset_error:
            # atomic_write may have published the isolated receipt before a
            # directory fsync reported failure.  Once that state is visible,
            # never start a rollback under an `isolated` receipt: a crash in
            # that rollback would recover as a commit and leave partial state.
            current_receipt = reset_dirs.read_receipt(
                reset_dirs.mem_fd, receipt_name
            )
            if (
                isinstance(current_receipt, dict)
                and current_receipt.get("state") == "isolated"
            ):
                reset_dirs.verify_attached()
                try:
                    _fsync_directory_fd(reset_dirs.mem_fd, reset_dirs.mem_dir)
                except OSError:
                    return {
                        "deleted": [entry["label"] for entry in plan],
                        "targets": list(targets),
                        "status": "cleanup_pending",
                    }
                cleanup_pending = _cleanup_isolated_reset(
                    reset_dirs, receipt_name, plan
                )
                return {
                    "deleted": [entry["label"] for entry in plan],
                    "targets": list(targets),
                    "status": (
                        "cleanup_pending" if cleanup_pending else "completed"
                    ),
                }
            if not any(
                _reset_entry_stat(
                    _reset_scope_fd(reset_dirs, _reset_stage_scope(entry)),
                    entry["stage"],
                ) is not None
                for entry in plan
            ):
                try:
                    reset_dirs.verify_attached()
                    os.unlink(receipt_name, dir_fd=reset_dirs.mem_fd)
                    _fsync_directory_fd(reset_dirs.mem_fd, reset_dirs.mem_dir)
                except (OSError, MemoryImportConflict):
                    pass
                raise
            try:
                _reset_restore_plan(reset_dirs, plan)
                try:
                    os.unlink(receipt_name, dir_fd=reset_dirs.mem_fd)
                    _fsync_directory_fd(reset_dirs.mem_fd, reset_dirs.mem_dir)
                except FileNotFoundError:
                    pass
            except BaseException as restore_error:
                raise RuntimeError(
                    "memory reset failed and recovery remains pending: "
                    f"{restore_error}"
                ) from reset_error
            raise

        deleted = [entry["label"] for entry in plan]
        cleanup_pending = _cleanup_isolated_reset(reset_dirs, receipt_name, plan)
        reset_dirs.verify_attached()
        return {
            "deleted": deleted,
            "targets": list(targets),
            "status": "cleanup_pending" if cleanup_pending else "completed",
        }


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

from tools.threat_patterns import first_threat_message as _first_threat_message

_HAN_RE = re.compile(r'[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


def _zettlab_user_profile_language_enabled() -> bool:
    """True in the Zettlab per-agent gateway runtime."""
    enabled = os.getenv("ZET_AGENT_ENABLED", "").strip().lower()
    if enabled in {"1", "true", "yes", "on"}:
        return True
    return bool(os.getenv("ZET_AGENT_ID") or os.getenv("ZETTLAB_AGENT_ACTION_TOKEN"))


def _validate_user_profile_language(target: str, content: str) -> Optional[str]:
    """Keep Zettlab-generated USER.md entries in Simplified Chinese."""
    if target != "user" or not _zettlab_user_profile_language_enabled():
        return None
    text = content.strip()
    if not text or _HAN_RE.search(text):
        return None
    latin_words = _LATIN_WORD_RE.findall(text)
    # Short names, codes, model IDs, and timezones are not a language choice.
    if len(latin_words) < 3 and len(text) < 24:
        return None
    return (
        "Blocked: Zettlab user profile entries must be written in Simplified Chinese. "
        "Keep names, product names, commands, and code identifiers as-is, but rewrite "
        "the surrounding user-profile statement in Chinese before calling memory again."
    )


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation, point the operator at
    the .bak.<ts> snapshot we took, and tell them what to do next.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # After this many failed consolidation attempts (overflow / zero-match) in
    # ONE turn, stop instructing the model to "retry in this turn" and return a
    # terminal "save skipped" result so a fragile replace/add can't loop the
    # turn to budget exhaustion and suppress the user's reply (issue #42405).
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed at-capacity consolidation attempts; reset
        # at each turn boundary by reset_consolidation_failures() (#42405).
        self._consolidation_failures = 0

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count an at-capacity consolidation failure and degrade gracefully.

        Under the per-turn cap, return ``response`` unchanged (it already tells
        the model how to self-correct + retry in this turn). Once the cap is
        exceeded, drop the retry instruction and return a TERMINAL result so the
        model stops looping memory calls and proceeds to answer the user — a
        failed memory side effect must never block the turn's reply (#42405).
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    def load_from_disk(self, *, bounded: bool = False):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so the user can still SEE poisoned entries via
        see poisoned entries by inspecting the source files directly, and remove them — silently dropping them would hide the attack from the user.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        if bounded:
            home = get_hermes_home()
            if not _require_optional_real_directory(
                home, label="HERMES_HOME"
            ):
                self.memory_entries = []
                self.user_entries = []
            else:
                candidate = home / "memories"
                if not _require_optional_real_directory(
                    candidate, label="profile memories directory"
                ):
                    self.memory_entries = []
                    self.user_entries = []
                else:
                    mem_dir = _require_profile_memory_directory(create=False)
                    self.memory_entries = self._read_import_file(
                        mem_dir / "MEMORY.md"
                    )
                    self.user_entries = self._read_import_file(
                        mem_dir / "USER.md"
                    )
        else:
            mem_dir = get_memory_dir()
            mem_dir.mkdir(parents=True, exist_ok=True)
            self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
            self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text so the user
        # can see + remove poisoned entries via the memory tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty or already-block-marker entries pass through unchanged.
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path, *, create_parent: bool = True):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        if create_parent:
            lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        raw_fd = None
        parent_fd = None
        try:
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
            if import_dirs is not None and lock_path.parent == import_dirs.mem_dir:
                raw_fd = os.open(
                    lock_path.name, flags, 0o600, dir_fd=import_dirs.mem_fd
                )
                current = os.stat(
                    lock_path.name,
                    dir_fd=import_dirs.mem_fd,
                    follow_symlinks=False,
                )
            elif os.name != "nt" and _OPEN_SUPPORTS_DIR_FD:
                parent_flags = os.O_RDONLY
                if hasattr(os, "O_DIRECTORY"):
                    parent_flags |= os.O_DIRECTORY
                if hasattr(os, "O_NOFOLLOW"):
                    parent_flags |= os.O_NOFOLLOW
                parent_fd = os.open(lock_path.parent, parent_flags)
                raw_fd = os.open(lock_path.name, flags, 0o600, dir_fd=parent_fd)
                current = os.lstat(lock_path)
            else:
                raw_fd = os.open(lock_path, flags, 0o600)
                current = os.lstat(lock_path)
            opened = os.fstat(raw_fd)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise MemoryImportConflict(f"refusing to follow unsafe lock {lock_path}")
            fd = os.fdopen(raw_fd, "a+", encoding="utf-8")
            raw_fd = None
        except OSError as exc:
            raise MemoryImportConflict(f"cannot safely open lock {lock_path}: {exc}") from exc
        finally:
            if raw_fd is not None:
                os.close(raw_fd)
            if parent_fd is not None:
                os.close(parent_fd)
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(
        self, target: str, *, skip_drift: bool = False, bounded: bool = False
    ) -> Optional[str]:
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns the backup path if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns None on clean reload.

        When *skip_drift* is True the round-trip / entry-size check is
        bypassed.  Used by the ``add`` action which appends without
        rewriting, so existing content is never clobbered.
        """
        path = self._path_for(target)
        bak = None if skip_drift else self._detect_external_drift(target, bounded=bounded)
        fresh = self._read_import_file(path) if bounded else self._read_file(path)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return bak

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        language_error = _validate_user_profile_language(target, content)
        if language_error:
            return {"success": False, "error": language_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # For add (append-only), we skip the drift guard — appending never
            # clobbers existing content, so round-trip mismatches from prior
            # tool-written entries in the same session are harmless.  The drift
            # guard remains active for replace/remove where full-file rewrite
            # would discard un-roundtrippable content (issue #26045).
            self._reload_target(target, skip_drift=True)

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Consolidate now: use 'replace' to merge overlapping entries into "
                        f"shorter ones or 'remove' stale or less important entries (see "
                        f"current_entries below), then retry this add — all in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}
        language_error = _validate_user_profile_language(target, new_content)
        if language_error:
            return {"success": False, "error": language_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to replace.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content, or 'remove' other stale or less important "
                        f"entries to make room (see current_entries below), then retry — all "
                        f"in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to remove.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        All operations are validated and applied against the FINAL budget --
        intermediate overflow is irrelevant. This lets the model free space
        (remove/replace) and add new entries in a SINGLE tool call instead of
        the multi-turn consolidate-then-retry dance that re-sends the whole
        conversation context several times.

        Semantics: all-or-nothing. If any op is malformed, doesn't match, or
        the net result would exceed the char limit, NOTHING is written and an
        error is returned describing the first failure plus the live state.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content for injection/exfil BEFORE touching
        # disk -- a single poisoned op rejects the whole batch.
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak:
                return _drift_error(self._path_for(target), bak)

            # Work on a copy; only commit if the whole batch validates.
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue  # idempotent -- skip duplicate, don't fail the batch
                    working.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working[matches[0]] = content

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"After applying all {len(operations)} operations, memory would be at "
                        f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                        f"entries in the same batch (see current_entries below), then retry."
                    ),
                    "current_entries": self._entries_for(target),
                    "usage": f"{current:,}/{limit:,}",
                })

            # Commit.
            self._set_entries(target, working)
            self.save_to_disk(target)

        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def import_replace(
        self, *, target: str, entries: List[str], import_id: str, payload_sha256: str
    ) -> Dict[str, Any]:
        """Atomically replace one curated-memory file with an idempotent receipt."""
        from portable_import_security import reject_portable_credentials

        if target not in {"memory", "user"}:
            raise ValueError("target must be memory or user")
        if not isinstance(import_id, str) or not import_id or len(import_id) > 128:
            raise ValueError("import_id must be 1..128 characters")
        reject_portable_credentials(import_id, field="import_id")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", payload_sha256 or ""):
            raise ValueError("payload_sha256 must be 64 hexadecimal characters")
        if not isinstance(entries, list) or len(entries) > 128:
            raise ValueError("entries must be a list with at most 128 items")
        normalized: List[str] = []
        for index, raw in enumerate(entries):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"entries[{index}] must be non-empty text")
            entry = raw.strip()
            reject_portable_credentials(entry, field=f"entries[{index}]")
            error = _scan_memory_content(entry) or _validate_user_profile_language(target, entry)
            if error:
                raise ValueError(f"entries[{index}]: {error}")
            if entry not in normalized:
                normalized.append(entry)
        char_count = len(ENTRY_DELIMITER.join(normalized)) if normalized else 0
        if char_count > self._char_limit(target):
            raise ValueError(f"imported {target} exceeds {self._char_limit(target)} character limit")

        content = ENTRY_DELIMITER.join(normalized)
        content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        expected_home_identity, expected_mem_identity = (
            _require_durable_profile_filesystem()
        )
        mem_dir, home_identity, mem_identity = _require_profile_memory_snapshot(
            create=True,
            expected_home_identity=expected_home_identity,
            expected_mem_identity=expected_mem_identity,
        )
        path = mem_dir / _MEMORY_TARGET_FILES[target]
        # Reject an unsafe canonical leaf before creating managed import state.
        if path.is_symlink():
            raise MemoryImportConflict(
                f"refusing to import through symlinked {path.name}"
            )
        _read_bounded_regular_file_bytes(path)
        transaction_path = mem_dir / _MEMORY_TRANSACTION_LOCK
        for lock_target in (transaction_path, path):
            _validate_reset_lock(lock_target)
        imports_dir = mem_dir / ".imports"
        backup_dir = imports_dir / "backups"
        receipt_path = imports_dir / (
            hashlib.sha256(import_id.encode("utf-8")).hexdigest() + ".json"
        )
        import_hash = hashlib.sha256(import_id.encode("utf-8")).hexdigest()
        backup_candidate = backup_dir / f"{target}-{import_hash}.bak"
        displaced_candidate = self._import_displaced_path(path, receipt_path)
        with _anchored_import_directories(
            mem_dir,
            create_managed=False,
            expected_home_identity=home_identity,
            expected_mem_identity=mem_identity,
        ) as import_dirs, self._file_lock(
            transaction_path, create_parent=False
        ), self._file_lock(path, create_parent=False):
            # The CAS publish and recovery protocol is hard-link based. Probe
            # the exact anchored directory under the shared transaction lock.
            # Lock leaves may exist, but no receipt, backup, displacement, or
            # imported content can be created before this succeeds.
            import_dirs.require_import_hardlink_support()
            # Every read and write below is relative to directory descriptors
            # opened before lock acquisition. Path swaps can abort the import,
            # but cannot redirect plaintext into an attacker-controlled tree.
            import_dirs.ensure_managed()
            canonical_raw = import_dirs.read_bytes(import_dirs.mem_fd, path.name)
            receipt_text = import_dirs.read_text(
                import_dirs.imports_fd, receipt_path.name
            )
            import_dirs.read_bytes(import_dirs.backup_fd, backup_candidate.name)
            import_dirs.read_bytes(import_dirs.mem_fd, displaced_candidate.name)
            if receipt_text is not None:
                try:
                    receipt = json.loads(receipt_text)
                except json.JSONDecodeError as exc:
                    raise MemoryImportConflict(f"memory import receipt is unreadable: {exc}") from exc
                expected = (import_id, target, payload_sha256.lower(), content_sha)
                actual = (receipt.get("import_id"), receipt.get("target"),
                          receipt.get("payload_sha256"), receipt.get("content_sha256"))
                if actual != expected:
                    raise MemoryImportConflict("import_id is already bound to different memory data")
                if canonical_raw is None:
                    live_entries = []
                    live_file_sha = _MISSING_MEMORY_FILE_SHA256
                else:
                    try:
                        live_text = canonical_raw.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise MemoryImportConflict(
                            f"{path.name} is not valid UTF-8"
                        ) from exc
                    live_entries = [
                        item
                        for item in (
                            part.strip() for part in live_text.split(ENTRY_DELIMITER)
                        )
                        if item
                    ]
                    live_file_sha = hashlib.sha256(canonical_raw).hexdigest()
                live_sha = hashlib.sha256(
                    ENTRY_DELIMITER.join(live_entries).encode("utf-8")
                ).hexdigest()
                state = receipt.get("state")
                if state == "prepared":
                    previous_sha = receipt.get("previous_content_sha256")
                    displaced_path = receipt.get("displaced_path")
                    expected_displaced = str(
                        self._import_displaced_path(path, receipt_path)
                    )
                    if displaced_path is not None and displaced_path != expected_displaced:
                        raise MemoryImportConflict(
                            "memory import receipt has an invalid displaced path"
                        )
                    if live_sha == content_sha and live_file_sha == content_sha:
                        pass
                    elif displaced_path is not None:
                        previous_file_sha = receipt.get("previous_file_sha256")
                        if previous_file_sha is None:
                            raise MemoryImportConflict(
                                "memory import receipt is missing the previous file hash"
                            )
                        self._write_file(
                            path,
                            normalized,
                            previous_file_sha,
                            Path(displaced_path),
                        )
                    elif live_sha == previous_sha:
                        previous_file_sha = receipt.get("previous_file_sha256")
                        if previous_file_sha is not None and live_file_sha != previous_file_sha:
                            raise MemoryImportConflict(
                                "memory changed after import prepare; refusing to overwrite user edits"
                            )
                        displaced_path = expected_displaced
                        receipt["displaced_path"] = displaced_path
                        self._write_import_receipt(receipt_path, receipt)
                        self._write_file(
                            path,
                            normalized,
                            live_file_sha,
                            Path(displaced_path),
                        )
                    else:
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        )
                    # Also covers recovery of a legacy prepared receipt where
                    # the target rename landed before process death.
                    import_dirs.verify_attached()
                    _fsync_directory_fd(import_dirs.mem_fd, import_dirs.mem_dir)
                    receipt["state"] = "completed"
                    receipt["completed_at"] = time.time()
                    if (
                        displaced_path is not None
                        and receipt.get("previous_file_sha256")
                        != _MISSING_MEMORY_FILE_SHA256
                    ):
                        receipt["displaced_retention"] = "manual"
                    self._write_import_receipt(receipt_path, receipt)
                elif (
                    state != "completed"
                    or live_sha != receipt.get("content_sha256")
                    or live_file_sha != receipt.get("content_sha256")
                ):
                    raise MemoryImportConflict(
                        "memory changed after import; refusing to overwrite user edits"
                    )
                else:
                    displaced_path = receipt.get("displaced_path")
                    if displaced_path is not None:
                        expected_displaced = str(
                            self._import_displaced_path(path, receipt_path)
                        )
                        if displaced_path != expected_displaced:
                            raise MemoryImportConflict(
                                "memory import receipt has an invalid displaced path"
                            )
                        if (
                            import_dirs.read_bytes(
                                import_dirs.mem_fd, Path(displaced_path).name
                            ) is not None
                            and receipt.get("displaced_retention") != "manual"
                        ):
                            receipt["displaced_retention"] = "manual"
                            self._write_import_receipt(receipt_path, receipt)
                self._set_entries(target, normalized)
                result = {"import_id": import_id, "status": "completed", "target": target,
                          "char_count": char_count, "replayed": True,
                          "effective_from": "next_session"}
                if receipt.get("backup_path"):
                    result["backup_path"] = receipt["backup_path"]
                displaced_path = receipt.get("displaced_path")
                if (
                    displaced_path is not None
                    and import_dirs.read_bytes(
                        import_dirs.mem_fd, Path(displaced_path).name
                    ) is not None
                ):
                    result["recovery_path"] = displaced_path
                    result["recovery_retention"] = "manual"
                import_dirs.confirm_completed(
                    path.name, receipt_path.name, content_sha
                )
                return result

            # Capture the same parsed on-disk representation used by crash
            # recovery before _reload_target deduplicates the live entries.
            # Otherwise a legacy file containing duplicate entries produces a
            # prepared receipt whose previous SHA can never match that file.
            if canonical_raw is None:
                previous_file_sha = _MISSING_MEMORY_FILE_SHA256
                previous_text = ""
            else:
                previous_file_sha = hashlib.sha256(canonical_raw).hexdigest()
                try:
                    previous_text = canonical_raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise MemoryImportConflict(
                        f"{path.name} is not valid UTF-8"
                    ) from exc
            previous_entries = [
                item
                for item in (
                    part.strip() for part in previous_text.split(ENTRY_DELIMITER)
                )
                if item
            ]
            backup = self._detect_external_drift(
                target, bounded=True, raw_text=previous_text
            )
            if backup:
                raise MemoryImportConflict(_drift_error(path, backup)["error"])
            self._set_entries(target, list(dict.fromkeys(previous_entries)))
            backup_path = self._write_import_backup(
                path=path, target=target, import_id=import_id
            )
            previous_sha = hashlib.sha256(
                ENTRY_DELIMITER.join(previous_entries).encode("utf-8")
            ).hexdigest()
            displaced_path = self._import_displaced_path(path, receipt_path)
            receipt = {
                "state": "prepared", "import_id": import_id, "target": target,
                "payload_sha256": payload_sha256.lower(),
                "content_sha256": content_sha,
                "previous_content_sha256": previous_sha,
                "previous_file_sha256": previous_file_sha,
                "displaced_path": str(displaced_path),
                "prepared_at": time.time(),
            }
            if previous_file_sha != _MISSING_MEMORY_FILE_SHA256:
                receipt["displaced_retention"] = "manual"
            if backup_path:
                receipt["backup_path"] = backup_path
            # Durable prepare MUST precede the target rename. A crash can then
            # be recovered without guessing whether a later edit is user data.
            self._write_import_receipt(receipt_path, receipt)
            self._write_file(path, normalized, previous_file_sha, displaced_path)
            receipt["state"] = "completed"
            receipt["completed_at"] = time.time()
            self._write_import_receipt(receipt_path, receipt)
            import_dirs.confirm_completed(
                path.name, receipt_path.name, content_sha
            )
            self._set_entries(target, normalized)
            result = {
                "import_id": import_id,
                "status": "completed",
                "target": target,
                "char_count": char_count,
                "replayed": False,
                "effective_from": "next_session",
            }
            if backup_path:
                result["backup_path"] = backup_path
            if import_dirs.read_bytes(
                import_dirs.mem_fd, displaced_path.name
            ) is not None:
                result["recovery_path"] = str(displaced_path)
                result["recovery_retention"] = "manual"
            import_dirs.verify_attached()
            return result

    @staticmethod
    def _write_import_backup(*, path: Path, target: str, import_id: str) -> Optional[str]:
        """Atomically retain the previous curated-memory bytes before replace."""
        import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
        if import_dirs is not None and path.parent == import_dirs.mem_dir:
            previous = import_dirs.read_bytes(import_dirs.mem_fd, path.name)
        else:
            previous = _read_bounded_regular_file_bytes(path)
        if previous is None:
            return None

        backup_dir = path.parent / ".imports" / "backups"
        _require_managed_memory_directory(backup_dir, create=False)
        import_hash = hashlib.sha256(import_id.encode("utf-8")).hexdigest()
        backup_path = backup_dir / f"{target}-{import_hash}.bak"
        if import_dirs is not None and backup_dir == import_dirs.backup_dir:
            import_dirs.atomic_write(
                import_dirs.backup_fd,
                backup_path.name,
                previous,
                prefix=".backup_",
            )
            retained = []
            for name in os.listdir(import_dirs.backup_fd):
                if not name.startswith(f"{target}-") or not name.endswith(".bak"):
                    continue
                candidate_stat = os.stat(
                    name, dir_fd=import_dirs.backup_fd, follow_symlinks=False
                )
                if not stat.S_ISREG(candidate_stat.st_mode):
                    raise MemoryImportConflict(
                        f"managed memory backup {name} must be a regular file"
                    )
                retained.append((name, candidate_stat.st_mtime_ns))
            retained.sort(key=lambda item: (item[1], item[0]), reverse=True)
            for name, _mtime_ns in retained[MEMORY_IMPORT_BACKUP_LIMIT:]:
                os.unlink(name, dir_fd=import_dirs.backup_fd)
            import_dirs.verify_attached()
            _fsync_directory_fd(import_dirs.backup_fd, import_dirs.backup_dir)
            return str(backup_path)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(backup_dir), suffix=".tmp", prefix=".backup_"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(previous)
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_path, backup_path)
            _fsync_directory(backup_dir)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        retained = []
        for candidate in backup_dir.iterdir():
            candidate_stat = candidate.stat(follow_symlinks=False)
            if not (
                candidate.name.startswith(f"{target}-")
                and candidate.name.endswith(".bak")
            ):
                continue
            if not stat.S_ISREG(candidate_stat.st_mode):
                raise MemoryImportConflict(
                    f"managed memory backup {candidate.name} must be a regular file"
                )
            retained.append((candidate, candidate_stat.st_mtime_ns))
        retained.sort(
            key=lambda item: (item[1], item[0].name),
            reverse=True,
        )
        for candidate, _mtime_ns in retained[MEMORY_IMPORT_BACKUP_LIMIT:]:
            candidate.unlink()
        _fsync_directory(backup_dir)
        return str(backup_path)

    @staticmethod
    def _write_import_receipt(path: Path, receipt: Dict[str, Any]) -> None:
        """Atomically persist and fsync a memory-import prepare/receipt."""
        import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
        if import_dirs is not None and path.parent == import_dirs.imports_dir:
            encoded = json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            import_dirs.atomic_write(
                import_dirs.imports_fd,
                path.name,
                encoded,
                prefix=".receipt_",
            )
            return
        _require_managed_memory_directory(path.parent, create=False)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".receipt_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(receipt, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            atomic_replace(tmp_path, path)
            _fsync_directory(path.parent)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        })

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures,
        # not lifetime ones within a turn) (#42405).
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"USER PROFILE (who the user is) [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"MEMORY (your personal notes) [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries.

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, IOError):
            return []

        if not raw.strip():
            return []

        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _read_import_file(path: Path) -> List[str]:
        """Read an import target through the bounded no-follow reader."""
        raw = _read_bounded_regular_file_text(path)
        if raw is None or not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    def _detect_external_drift(
        self,
        target: str,
        *,
        bounded: bool = False,
        raw_text: Optional[str] = None,
    ) -> Optional[str]:
        """Return a backup-path string if on-disk content shows external drift.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Returns the absolute path of the .bak file when drift was found and
        backed up; returns None when the file looks tool-shaped.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        path = self._path_for(target)
        if raw_text is not None:
            raw = raw_text
        elif bounded:
            raw = _read_bounded_regular_file_text(path)
            if raw is None:
                return None
        else:
            if not path.exists():
                return None
            try:
                raw = path.read_text(encoding="utf-8")
            except (OSError, IOError):
                return None
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # Drift confirmed — snapshot the file so the operator can recover
        # whatever the external writer added, then return the .bak path so
        # the caller can refuse the mutation.
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        if bounded:
            import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
            if import_dirs is not None and path.parent == import_dirs.mem_dir:
                try:
                    import_dirs.atomic_write(
                        import_dirs.mem_fd,
                        bak_path.name,
                        raw.encode("utf-8"),
                        prefix=".drift_",
                    )
                except OSError:
                    return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
                return str(bak_path)
            fd = None
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=".tmp", prefix=".drift_"
                )
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    fd = None
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_replace(tmp_path, bak_path)
                tmp_path = None
                _fsync_directory(path.parent)
            except OSError:
                return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
            finally:
                if fd is not None:
                    os.close(fd)
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            return str(bak_path)
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _live_file_sha256(path: Path) -> str:
        content = _read_bounded_regular_file_bytes(path)
        if content is None:
            return _MISSING_MEMORY_FILE_SHA256
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _import_displaced_path(path: Path, receipt_path: Path) -> Path:
        import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
        if import_dirs is not None and path.parent == import_dirs.mem_dir:
            return import_dirs.mem_dir / f".{path.name}.{receipt_path.stem}.displaced"
        if path.is_symlink():
            raise MemoryImportConflict(
                f"refusing to import through symlinked {path.name}"
            )
        return path.parent / (
            f".{path.name}.{receipt_path.stem}.displaced"
        )

    @staticmethod
    def _restore_displaced_no_replace(displaced_path: Path, path: Path) -> None:
        source_content = _read_bounded_regular_file_bytes(displaced_path)
        if source_content is None:
            raise MemoryImportConflict("displaced memory recovery file is missing")
        source_mode = stat.S_IMODE(os.lstat(displaced_path).st_mode) or 0o600
        try:
            os.link(displaced_path, path, follow_symlinks=False)
        except FileExistsError:
            return
        except OSError:
            # Hard links are unavailable on some supported profile filesystems.
            # Fall back to an O_EXCL copy so a concurrent external winner is
            # never overwritten. The displaced recovery inode remains retained.
            target_fd = None
            target_identity = None
            try:
                target_fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, source_mode
                )
                target_stat = os.fstat(target_fd)
                target_identity = (target_stat.st_dev, target_stat.st_ino)
                for offset in range(0, len(source_content), 1 << 20):
                    chunk = source_content[offset:offset + (1 << 20)]
                    view = memoryview(chunk)
                    while view:
                        written = os.write(target_fd, view)
                        if written <= 0:
                            raise OSError("short write while restoring displaced memory")
                        view = view[written:]
                os.fsync(target_fd)
            except FileExistsError:
                return
            except BaseException:
                if target_fd is not None and target_identity is not None:
                    try:
                        current = os.lstat(path)
                        if (current.st_dev, current.st_ino) == target_identity:
                            os.unlink(path)
                    except OSError:
                        pass
                raise
            finally:
                if target_fd is not None:
                    os.close(target_fd)
        _fsync_directory(path.parent)

    @staticmethod
    def _write_file(
        path: Path,
        entries: List[str],
        expected_live_sha256: Optional[str] = None,
        displaced_path: Optional[Path] = None,
    ):
        """Write entries atomically, with no-clobber CAS for imports.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.

        Import writes with an expected SHA use a stricter state machine: move
        the old target to ``displaced_path``, verify those exact bytes, then
        hard-link the prepared file into the unoccupied canonical name. This
        deliberately creates a short missing-name window so an external writer
        can win with create-if-absent instead of ever being overwritten.

        The displaced inode is retained at its receipt-recorded path after a
        successful import. A different process may still hold an open file
        descriptor for that inode and write after completion; POSIX provides no
        safe way to prove all such descriptors are closed. Automatic cleanup
        would therefore risk losing those late writes. Operators may remove the
        recovery path manually only after quiescing external writers.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        import_dirs = _ACTIVE_MEMORY_IMPORT_DIRS.get()
        if (
            import_dirs is not None
            and path.parent == import_dirs.mem_dir
            and expected_live_sha256 is not None
        ):
            if displaced_path is None or displaced_path.parent != import_dirs.mem_dir:
                raise MemoryImportConflict(
                    "memory import is missing a valid displaced path"
                )
            try:
                import_dirs.write_canonical_cas(
                    path.name,
                    displaced_path.name,
                    content.encode("utf-8"),
                    expected_live_sha256,
                )
            except MemoryImportConflict:
                raise
            except (OSError, IOError) as exc:
                raise RuntimeError(f"Failed to write memory file {path}: {exc}") from exc
            return
        if path.is_symlink():
            raise MemoryImportConflict(
                f"refusing to import through symlinked {path.name}"
            )
        effective_path = path
        try:
            # Write to temp file in same directory (same filesystem for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(effective_path.parent), suffix=".tmp", prefix=".mem_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                if expected_live_sha256 is None:
                    atomic_replace(tmp_path, path)
                    _fsync_directory(effective_path.parent)
                    return
                if displaced_path is None or displaced_path.parent != effective_path.parent:
                    raise MemoryImportConflict(
                        "memory import is missing a valid displaced path"
                    )

                if os.path.lexists(displaced_path):
                    displaced_sha = MemoryStore._live_file_sha256(displaced_path)
                    if displaced_sha != expected_live_sha256:
                        if MemoryStore._live_file_sha256(effective_path) == _MISSING_MEMORY_FILE_SHA256:
                            MemoryStore._restore_displaced_no_replace(
                                displaced_path, effective_path
                            )
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        )
                    if MemoryStore._live_file_sha256(effective_path) != _MISSING_MEMORY_FILE_SHA256:
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        )
                elif expected_live_sha256 == _MISSING_MEMORY_FILE_SHA256:
                    if MemoryStore._live_file_sha256(effective_path) != _MISSING_MEMORY_FILE_SHA256:
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        )
                else:
                    try:
                        os.replace(effective_path, displaced_path)
                    except FileNotFoundError as exc:
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        ) from exc
                    _fsync_directory(effective_path.parent)
                    if MemoryStore._live_file_sha256(displaced_path) != expected_live_sha256:
                        if MemoryStore._live_file_sha256(effective_path) == _MISSING_MEMORY_FILE_SHA256:
                            MemoryStore._restore_displaced_no_replace(
                                displaced_path, effective_path
                            )
                        raise MemoryImportConflict(
                            "memory changed after import prepare; refusing to overwrite user edits"
                        )

                try:
                    os.link(tmp_path, effective_path)
                except FileExistsError as exc:
                    raise MemoryImportConflict(
                        "memory changed after import prepare; refusing to overwrite user edits"
                    ) from exc
                except OSError as publish_error:
                    try:
                        MemoryStore._restore_displaced_no_replace(
                            displaced_path, effective_path
                        )
                    except BaseException as restore_error:
                        raise RuntimeError(
                            "memory publish failed and displaced recovery also failed: "
                            f"{restore_error}"
                        ) from publish_error
                    raise
                os.unlink(tmp_path)
                _fsync_directory(effective_path.parent)
            except BaseException:
                # Clean up temp file on any failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def load_on_disk_store(*, bounded: bool = False) -> "MemoryStore":
    """Build a fresh on-disk :class:`MemoryStore`, honoring configured char limits.

    Use this from any context that has no live agent (the messaging gateway, the
    Desktop GUI, the bare CLI ``/memory`` handler) but still needs to read or
    apply approved memory writes. Mirrors how the live agent constructs its store
    in ``agent/agent_init.py`` — including the user's ``memory.memory_char_limit``
    / ``memory.user_char_limit`` overrides — so an approval applied without a live
    agent enforces the SAME caps as one applied with one.

    Falls back to the built-in defaults if config can't be loaded, so this can
    never raise on a missing/unreadable config.
    """
    memory_char_limit = 2200
    user_char_limit = 1375
    try:
        from hermes_cli.config import load_config

        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        memory_char_limit = int(mem_cfg.get("memory_char_limit", memory_char_limit))
        user_char_limit = int(mem_cfg.get("user_char_limit", user_char_limit))
    except Exception:
        pass  # config optional — fall back to defaults rather than break /memory

    store = MemoryStore(
        memory_char_limit=memory_char_limit,
        user_char_limit=user_char_limit,
    )
    store.load_from_disk(bounded=bounded)
    return store


def _apply_write_gate(action: str, target: str, content: Optional[str],
                      old_text: Optional[str]) -> Optional[str]:
    """Evaluate the memory write gate. Returns a JSON tool-result string when
    the write should NOT proceed normally (blocked or staged), or None when the
    caller should perform the real write.

    Only the mutating actions (add/replace/remove) are gated.
    """
    if action not in {"add", "replace", "remove"}:
        return None

    try:
        from tools import write_approval as wa
    except Exception:
        # If the gate module can't load, fail open (current behaviour) rather
        # than blocking all memory writes.
        return None

    # Build a small inline summary/detail for the foreground approval prompt.
    label = "user profile" if target == "user" else "memory"
    if action == "add":
        summary = f"add to {label}"
        detail = content or ""
    elif action == "replace":
        summary = f"replace in {label}"
        detail = f"old: {old_text}\nnew: {content}"
    else:  # remove
        summary = f"remove from {label}"
        detail = old_text or ""

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    # stage
    payload = {
        "action": action,
        "target": target,
        "content": content,
        "old_text": old_text,
    }
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _apply_batch_write_gate(target: str, operations: List[Dict[str, Any]]) -> Optional[str]:
    """Evaluate the write gate for a batch of memory operations.

    Returns a JSON tool-result string when the batch should NOT proceed
    (blocked or staged), or None when the caller should perform the real
    batch write. The whole batch is gated as a single unit.
    """
    try:
        from tools import write_approval as wa
    except Exception:
        return None

    label = "user profile" if target == "user" else "memory"
    summary = f"apply {len(operations)} op(s) to {label}"
    detail_lines = []
    for op in operations:
        op = op or {}
        act = op.get("action", "?")
        if act == "remove":
            detail_lines.append(f"- remove: {op.get('old_text', '')}")
        elif act == "replace":
            detail_lines.append(f"- replace: {op.get('old_text', '')} -> {op.get('content', '')}")
        else:
            detail_lines.append(f"- {act}: {op.get('content', '')}")
    detail = "\n".join(detail_lines)

    decision = wa.evaluate_gate(wa.MEMORY, inline_summary=summary, inline_detail=detail)

    if decision.allow:
        return None

    if decision.blocked:
        return tool_error(decision.message, success=False)

    payload = {"action": "batch", "target": target, "operations": operations}
    record = wa.stage_write(
        wa.MEMORY, payload,
        summary=f"{summary}: {detail[:120]}",
        origin=wa.current_origin(),
    )
    return json.dumps(
        {"success": True, "staged": True, "pending_id": record["id"],
         "message": decision.message},
        ensure_ascii=False,
    )


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def memory_tool(
    action: str = None,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Two shapes:
      - Single op: action + (content / old_text).
      - Batch:     operations=[{action, content?, old_text?}, ...] applied
                   atomically against the final char budget in ONE call.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        gate_result = _apply_batch_write_gate(target, operations)
        if gate_result is not None:
            return gate_result
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # The client/model omitted old_text. Replace is inherently targeted
            # -- we can't guess which entry. Return the current inventory plus a
            # retry instruction so the model can reissue with old_text set,
            # instead of hitting a dead-end error. (issues #43412, #49466)
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    # Approval gate: when on, stages the write (background/gateway) or prompts
    # inline (interactive CLI); when off (default) passes straight through.
    gate_result = _apply_write_gate(action, target, content, old_text)
    if gate_result is not None:
        return gate_result

    if action == "add":
        result = store.add(target, content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """Memory tool has no external requirements -- always available."""
    return True


def apply_memory_pending(payload: Dict[str, Any], store: "MemoryStore") -> Dict[str, Any]:
    """Replay a staged memory write directly against the store, bypassing the
    write gate. Called by the /memory approve handler.

    Returns the store's result dict.
    """
    action = payload.get("action")
    target = payload.get("target", "memory")
    content = payload.get("content") or ""
    old_text = payload.get("old_text") or ""
    if action == "batch":
        return store.apply_batch(target, payload.get("operations") or [])
    if action == "add":
        return store.add(target, content)
    if action == "replace":
        return store.replace(target, old_text, content)
    if action == "remove":
        return store.remove(target, old_text)
    return {"success": False, "error": f"Unknown staged action '{action}'."}
# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes or shortens enough stale entries and adds the new one together.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). "
        "In the Zettlab App runtime, write these user-profile entries in Simplified Chinese. "
        "'memory' = your notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform (single-op shape). Omit when using 'operations'."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=lambda args, **kw: memory_tool(
        action=args.get("action", ""),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        operations=args.get("operations"),
        store=kw.get("store")),
    check_fn=check_memory_requirements,
    emoji="🧠",
)
