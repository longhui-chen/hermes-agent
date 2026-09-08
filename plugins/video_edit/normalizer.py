"""Bounded adapter for the packaged hardware media normalizer.

The skill never invokes this script directly.  Hermes owns the subprocess
boundary and only passes validated local files; the normalizer itself is a
media utility, not an authorization or workflow checkpoint implementation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from plugins.video_edit.paths import (
    VideoPathError,
    inspect_video_descriptor,
    safe_id,
    state_root,
)

logger = logging.getLogger(__name__)

NORMALIZER_TIMEOUT_SECONDS = 1800
MEDIA_INSPECTION_TIMEOUT_SECONDS = 360
MAX_INSPECT_FILES = 10
MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 128 * 1024
DEFAULT_PRESETS_ROOT = "/zettos/main/apps/com.zettlab.presets/current"
_NORMALIZER_ENV_ALLOWLIST = frozenset({
    "PATH",
    "HOME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "LD_LIBRARY_PATH",
    "ZETTLAB_UPLOAD_FFMPEG",
    "ZETTLAB_UPLOAD_FFPROBE",
})


@dataclass(frozen=True)
class _PresetsAnchor:
    configured_root: Path
    resolved_root: Path
    root_identity: tuple[int, int, int]
    script: Path
    script_identity: tuple[int, int, int, int, int]
    generation: str


class _UnavailablePresets:
    """Process-lifetime marker for a release unavailable at registration."""


_PRESETS_UNAVAILABLE = _UnavailablePresets()
_PRESETS_ANCHOR: _PresetsAnchor | _UnavailablePresets | None = None
_PRESETS_ANCHOR_LOCK = threading.Lock()


FileIdentity = tuple[int, int, int, int, int]


class NormalizeError(RuntimeError):
    """Raised when the bounded hardware normalization step cannot complete."""


class NormalizerUnavailableError(NormalizeError):
    """The packaged helper or a requested helper capability is unavailable.

    ``identities`` is set when the capability gap was only discovered after
    the helper ran: the pins have already been linked and removed, so the
    caller must re-anchor its source identities on these post-pin values
    (each verified untouched through the whole pin lifecycle) instead of the
    stat it captured before the probe.
    """

    def __init__(
        self,
        message: str,
        *,
        identities: list[FileIdentity] | None = None,
    ) -> None:
        super().__init__(message)
        self.identities = identities


@dataclass(frozen=True)
class NormalizedOutput:
    """Process-local normalized artifact bound to the inode the helper produced."""

    path: Path
    identity: FileIdentity


def _run_bounded_subprocess(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    pass_fds: tuple[int, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run a helper while draining stdout/stderr into fixed-size buffers."""
    popen_options: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "posix":
        popen_options["start_new_session"] = True
        if pass_fds:
            popen_options["pass_fds"] = pass_fds
    elif pass_fds:
        raise NormalizeError("video normalizer is unavailable")
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, **popen_options)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": MAX_STDOUT_BYTES, "stderr": MAX_STDERR_BYTES}
    overflowed = threading.Event()

    def drain(name: str, stream: Any) -> None:
        buffer = buffers[name]
        limit = limits[name]
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            remaining = limit - len(buffer)
            if remaining > 0:
                buffer.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                overflowed.set()

    threads = [
        threading.Thread(
            target=drain,
            args=(name, stream),
            name=f"video-normalizer-{name}",
            daemon=True,
        )
        for name, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        )
        if stream is not None
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if overflowed.is_set() or time.monotonic() >= deadline:
            timed_out = time.monotonic() >= deadline
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    process.kill()
            else:
                process.kill()
            break
        time.sleep(0.01)
    returncode = process.wait()
    for thread in threads:
        thread.join(timeout=2)
    completed = subprocess.CompletedProcess(
        command,
        returncode,
        stdout=bytes(buffers["stdout"]).decode("utf-8", "replace"),
        stderr=bytes(buffers["stderr"]).decode("utf-8", "replace"),
    )
    return completed, overflowed.is_set() or timed_out


def _configured_presets_root() -> Path:
    raw = str(os.environ.get("ZETTLAB_PRESETS_DIR", DEFAULT_PRESETS_ROOT) or "").strip()
    if not raw or not os.path.isabs(raw):
        raise NormalizeError("video normalizer is unavailable")
    return Path(raw)


def _root_identity(path: Path) -> tuple[int, int, int]:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise NormalizeError("video normalizer is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise NormalizeError("video normalizer is unavailable")
    return info.st_dev, info.st_ino, info.st_mode


def _script_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _resolve_normalizer_script(
    root: Path,
) -> tuple[Path, tuple[int, int, int, int, int]]:
    candidates = (
        root / "skills" / "video-edit-workflow-mini" / "scripts" / "normalize.py",
        root / "skills" / "common" / "video-edit-workflow-mini" / "scripts" / "normalize.py",
    )
    selected: Path | None = None
    for candidate in candidates:
        try:
            os.stat(candidate, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise NormalizeError("video normalizer is unavailable") from exc
        selected = candidate
        break
    if selected is None or selected.name != "normalize.py":
        raise NormalizeError("video normalizer is unavailable")

    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise NormalizeError("video normalizer is unavailable") from exc
    current = root
    selected_info: os.stat_result | None = None
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            info = os.stat(current, follow_symlinks=False)
        except OSError as exc:
            raise NormalizeError("video normalizer is unavailable") from exc
        expected = stat.S_ISREG(info.st_mode) if index == len(relative.parts) - 1 else stat.S_ISDIR(info.st_mode)
        if not expected:
            raise NormalizeError("video normalizer is unavailable")
        selected_info = info
    if selected_info is None or selected_info.st_size <= 0:
        raise NormalizeError("video normalizer is unavailable")
    return selected, _script_identity(selected_info)


def _capture_presets_anchor() -> _PresetsAnchor:
    configured_root = _configured_presets_root()
    try:
        resolved_root = configured_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NormalizeError("video normalizer is unavailable") from exc
    root_identity = _root_identity(resolved_root)
    script, script_identity = _resolve_normalizer_script(resolved_root)
    generation = hashlib.sha256(
        "\x00".join(
            (
                str(resolved_root),
                *(str(value) for value in root_identity),
                str(script.relative_to(resolved_root)),
                *(str(value) for value in script_identity),
            )
        ).encode("utf-8", "replace")
    ).hexdigest()
    return _PresetsAnchor(
        configured_root=configured_root,
        resolved_root=resolved_root,
        root_identity=root_identity,
        script=script,
        script_identity=script_identity,
        generation=generation,
    )


def initialize_runtime() -> bool:
    """Fix the current presets release, or its absence, for this process."""

    global _PRESETS_ANCHOR
    with _PRESETS_ANCHOR_LOCK:
        if _PRESETS_ANCHOR is None:
            try:
                _PRESETS_ANCHOR = _capture_presets_anchor()
            except NormalizeError:
                _PRESETS_ANCHOR = _PRESETS_UNAVAILABLE
                logger.warning(
                    "video media helper unavailable at plugin registration"
                )
        return isinstance(_PRESETS_ANCHOR, _PresetsAnchor)


def _presets_anchor() -> _PresetsAnchor:
    initialize_runtime()
    configured_root = _configured_presets_root()
    with _PRESETS_ANCHOR_LOCK:
        anchor = _PRESETS_ANCHOR
    if not isinstance(anchor, _PresetsAnchor):
        raise NormalizeError("video normalizer is unavailable")
    if configured_root != anchor.configured_root:
        raise NormalizeError("video normalizer is unavailable")
    if _root_identity(anchor.resolved_root) != anchor.root_identity:
        raise NormalizeError("video normalizer is unavailable")
    script, script_identity = _resolve_normalizer_script(anchor.resolved_root)
    if script != anchor.script or script_identity != anchor.script_identity:
        raise NormalizeError("video normalizer is unavailable")
    return anchor


def _presets_root() -> Path:
    return _presets_anchor().resolved_root


def normalizer_script() -> Path:
    return _presets_anchor().script


def generation() -> str:
    """Return a non-authorizing identity for retry consistency."""
    return _presets_anchor().generation


def _inherited_script_path(descriptor: int) -> str:
    for root in (Path("/proc/self/fd"), Path("/dev/fd")):
        if root.is_dir():
            return str(root / str(descriptor))
    raise NormalizeError("video normalizer is unavailable")


@contextlib.contextmanager
def _opened_normalizer_script() -> Iterable[tuple[str, tuple[int, ...]]]:
    """Open the anchored script once and execute that same read-only inode."""

    anchor = _presets_anchor()
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if os.name != "posix" or no_follow is None:
        raise NormalizeError("video normalizer is unavailable")
    try:
        descriptor = os.open(
            anchor.script,
            os.O_RDONLY | os.O_CLOEXEC | no_follow,
        )
    except OSError as exc:
        raise NormalizeError("video normalizer is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or _script_identity(info) != anchor.script_identity
        ):
            raise NormalizeError("video normalizer is unavailable")
        yield _inherited_script_path(descriptor), (descriptor,)
    finally:
        os.close(descriptor)


def _under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def _temporary_root(workflow_id: str) -> Path:
    root = state_root() / "tmp" / safe_id(workflow_id, fallback="workflow")
    if root.exists() and root.is_symlink():
        raise NormalizeError("video normalization workspace is invalid")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def _create_snapshot_link(
    source: Path,
    workflow_id: str,
    index: int,
    *,
    prefix: str,
    directory: Path | None = None,
) -> Path:
    """Hard-link ``source`` under a private name (default: its own directory).

    The helper and its ffprobe grandchild reopen the pinned inode through this
    ordinary path, so the handoff never depends on ``/proc/<pid>/fd`` access.
    Raises ``OSError`` when the directory cannot host the link.
    """
    fd, name = tempfile.mkstemp(
        dir=str(directory if directory is not None else source.parent),
        prefix=f".{prefix}-{safe_id(workflow_id)}-{index}-",
    )
    os.close(fd)
    os.unlink(name)
    snapshot = Path(name)
    try:
        os.link(source, snapshot, follow_symlinks=False)
    except OSError:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot


def _pin_input(source: Path, workflow_id: str, index: int) -> Path:
    """Create a same-filesystem inode snapshot for the helper process."""
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise NormalizeError("video normalization input is unavailable") from exc
    snapshot: Path | None = None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise NormalizeError("video normalization input is invalid")
        snapshot = _create_snapshot_link(
            source, workflow_id, index, prefix="hermes-video-input"
        )
        snapshot_info = os.stat(snapshot, follow_symlinks=False)
        if (
            snapshot_info.st_dev != info.st_dev
            or snapshot_info.st_ino != info.st_ino
            or snapshot_info.st_size != info.st_size
        ):
            raise NormalizeError("video normalization input changed")
        return snapshot
    except NormalizeError:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as exc:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)
        raise NormalizeError("video normalization input cannot be pinned") from exc
    finally:
        os.close(descriptor)


def _inspection_fd_path(descriptor: int) -> str:
    """Return a parent-descriptor path for the fallback inspection handoff.

    This is only used when the source directory cannot host a hard-link pin.
    The parent PID is intentional: ``normalize.py`` starts ffprobe without
    inheriting the media descriptor, so a ``/proc/self/fd`` path would point
    at an unrelated fd in that grandchild.  Reading ``/proc/<parent>/fd`` from
    a child requires ptrace access to the parent; a non-dumpable gateway that
    dropped ``CAP_SYS_PTRACE`` cannot be read that way, which is why the pin
    is the primary handoff.
    """
    proc_root = Path(f"/proc/{os.getpid()}/fd")
    if not proc_root.is_dir():
        raise NormalizeError("video normalizer is unavailable")
    path = proc_root / str(descriptor)
    try:
        os.stat(path)
    except OSError as exc:
        raise NormalizeError("video normalizer is unavailable") from exc
    return str(path)


def _open_inspection_input(source: Path) -> tuple[int, os.stat_result]:
    """Open an inspection input once and return its stable descriptor identity."""
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise NormalizeError("video media inspection input is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        path_info = os.stat(source, follow_symlinks=False)
        identity = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
            or identity
            != (
                path_info.st_dev,
                path_info.st_ino,
                path_info.st_size,
                path_info.st_mtime_ns,
                path_info.st_ctime_ns,
            )
        ):
            raise NormalizeError("video media inspection input changed")
        return descriptor, info
    except Exception:
        os.close(descriptor)
        raise


def _stable_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    """Identity fields a hard-link pin must leave untouched (ctime excluded)."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


# Pins live under the local-server task-cache subtree of the source directory.
# The server's file watcher skips ``<dir>/.cache/tasks`` and everything below
# it, so the indexer never re-stamps the pinned inode (which would move its
# ctime mid-probe and trip the strict identity check).  Any other placement
# beside the media is observed and touched.
_INSPECTION_PIN_SUBDIRS = (".cache", "tasks", "hermes-video-inspect")


@dataclass(frozen=True)
class _InspectionInput:
    """One opened inspection source and the handoff path given to the helper."""

    source: Path
    descriptor: int
    opened: os.stat_result
    pin: Path | None
    created_dirs: tuple[Path, ...]


def _inspection_pin_directory(source: Path) -> tuple[Path, tuple[Path, ...]]:
    """Ensure the pin directory exists and report which levels were created.

    Raises ``OSError`` when the directory cannot be created or an existing
    level is not a real directory (a symlink is refused rather than followed).
    """
    created: list[Path] = []
    current = source.parent
    for part in _INSPECTION_PIN_SUBDIRS:
        current = current / part
        try:
            os.mkdir(current, 0o700)
        except FileExistsError:
            info = os.stat(current, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(str(current))
        else:
            created.append(current)
    return current, tuple(created)


def _pin_inspection_input(
    source: Path,
    opened: os.stat_result,
    workflow_id: str,
    index: int,
) -> tuple[Path, tuple[Path, ...]] | None:
    """Pin the opened inode under the source's task cache for the helper.

    Returns ``None`` when the source directory cannot host the pin (read-only
    or permission-restricted shares) so the caller can fall back to the
    parent-descriptor handoff instead of failing the upload outright.
    """
    created: tuple[Path, ...] = ()
    try:
        directory, created = _inspection_pin_directory(source)
        snapshot = _create_snapshot_link(
            source,
            workflow_id,
            index,
            prefix="hermes-video-inspect",
            directory=directory,
        )
    except OSError as exc:
        _remove_created_dirs(created)
        logger.info(
            "video media inspection pin unavailable errno=%s; using descriptor handoff",
            exc.errno,
        )
        return None
    try:
        snapshot_info = os.stat(snapshot, follow_symlinks=False)
    except OSError as exc:
        snapshot.unlink(missing_ok=True)
        _remove_created_dirs(created)
        raise NormalizeError("video media inspection input changed") from exc
    if (
        not stat.S_ISREG(snapshot_info.st_mode)
        or _stable_identity(snapshot_info) != _stable_identity(opened)
    ):
        snapshot.unlink(missing_ok=True)
        _remove_created_dirs(created)
        raise NormalizeError("video media inspection input changed")
    return snapshot, created


def _remove_created_dirs(created: Iterable[Path]) -> None:
    """Remove pin directories this process created, innermost first.

    Only-empty semantics: a level that meanwhile gained other entries stays.
    """
    for directory in reversed(tuple(created)):
        with contextlib.suppress(OSError):
            os.rmdir(directory)


def _prepare_inspection_input(
    source: Path,
    workflow_id: str,
    index: int,
) -> _InspectionInput:
    descriptor, opened = _open_inspection_input(source)
    pinned: tuple[Path, tuple[Path, ...]] | None = None
    try:
        pinned = _pin_inspection_input(source, opened, workflow_id, index)
        pin, created = pinned if pinned is not None else (None, ())
        return _InspectionInput(
            source=source,
            descriptor=descriptor,
            opened=opened,
            pin=pin,
            created_dirs=created,
        )
    except BaseException:
        if pinned is not None:
            pinned[0].unlink(missing_ok=True)
            _remove_created_dirs(pinned[1])
        os.close(descriptor)
        raise


def _probe_identity(item: _InspectionInput) -> FileIdentity:
    """Capture the full identity the inode must keep while the helper runs.

    Called after every pin is linked, so the links' own ctime transitions are
    already included; any later transition means something else touched the
    inode during the probe.
    """
    try:
        info = os.fstat(item.descriptor)
    except OSError as exc:
        raise NormalizeError("video media inspection input changed") from exc
    if _stable_identity(info) != _stable_identity(item.opened):
        raise NormalizeError("video media inspection input changed")
    return _file_identity(info)


def _verify_probe_untouched(item: _InspectionInput, expected: FileIdentity) -> None:
    try:
        info = os.fstat(item.descriptor)
    except OSError as exc:
        raise NormalizeError("video media inspection input changed") from exc
    if _file_identity(info) != expected:
        raise NormalizeError("video media inspection input changed")


def _remove_inspection_pin(item: _InspectionInput, *, strict: bool) -> None:
    """Unlink the pin.

    ``strict`` treats a pin that already vanished as a handoff tampered with
    by someone else and refuses instead of guessing.  Directory pruning is a
    separate pass (``_prune_pin_directories``) because several sources from
    one directory share the pin directory until the last pin is gone.
    """
    if item.pin is None:
        return
    try:
        item.pin.unlink()
    except OSError as exc:
        if strict:
            raise NormalizeError("video media inspection input changed") from exc


def _prune_pin_directories(inputs: Iterable[_InspectionInput]) -> None:
    for item in inputs:
        _remove_created_dirs(item.created_dirs)


_PR_GET_DUMPABLE = 3
_CAP_SYS_PTRACE = 19


def _parent_descriptor_handoff_usable() -> bool:
    """Report whether a child can read this process's ``/proc/<pid>/fd``.

    Linux lets a child open its parent's descriptor links only when the
    parent is dumpable or the child holds ``CAP_SYS_PTRACE``.  The managed
    gateway is non-dumpable and drops that capability from its bounding set
    (which is exactly what an exec'd child inherits), so the fallback handoff
    is known to fail there before the helper is even started.
    """
    if not sys.platform.startswith("linux"):
        return True
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        dumpable = int(libc.prctl(_PR_GET_DUMPABLE, 0, 0, 0, 0))
    except (OSError, AttributeError, ValueError):
        return True
    if dumpable != 0:
        return True
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("CapBnd:"):
                    bounding = int(line.split()[1], 16)
                    return bool(bounding & (1 << _CAP_SYS_PTRACE))
    except (OSError, ValueError, IndexError):
        return True
    return True


def _release_all(
    inputs: list[_InspectionInput],
    expected: list[FileIdentity],
) -> list[FileIdentity]:
    """Verify, unlink and re-anchor every input in the strict order."""
    for item, identity in zip(inputs, expected):
        _verify_probe_untouched(item, identity)
    for item in inputs:
        _remove_inspection_pin(item, strict=True)
    _prune_pin_directories(inputs)
    return [_final_identity(item) for item in inputs]


def _final_identity(item: _InspectionInput) -> FileIdentity:
    """Return the post-unlink identity the upload boundary must match.

    Captured after every pin is gone so the caller's own ``fstat`` compares
    equal, and only after confirming the original path still names the
    opened inode.
    """
    try:
        current = os.fstat(item.descriptor)
        path_info = os.stat(item.source, follow_symlinks=False)
    except OSError as exc:
        raise NormalizeError("video media inspection input changed") from exc
    identity = _file_identity(current)
    if (
        not stat.S_ISREG(path_info.st_mode)
        or _file_identity(path_info) != identity
        or _stable_identity(current) != _stable_identity(item.opened)
    ):
        raise NormalizeError("video media inspection input changed")
    return identity


def _bounded(value: str, limit: int) -> str:
    encoded = str(value or "").encode("utf-8", "replace")
    if len(encoded) <= limit:
        return encoded.decode("utf-8", "replace")
    return encoded[-limit:].decode("utf-8", "replace")


def _normalizer_env() -> dict[str, str]:
    """Build the media helper's data-only environment.

    The normalizer needs paths/locale only.  Constructing an allowlist rather
    than copying the Hermes process environment prevents every current or
    retired bearer channel from reaching the packaged Python/ffmpeg process.
    """
    return {
        key: value
        for key in _NORMALIZER_ENV_ALLOWLIST
        if (value := os.environ.get(key)) is not None
    }


def _require_inspection_helper() -> None:
    """Check helper availability before creating source-directory snapshots."""
    try:
        _presets_anchor()
    except NormalizeError as exc:
        if str(exc).strip() == "video normalizer is unavailable":
            raise NormalizerUnavailableError(
                "video normalizer is unavailable"
            ) from exc
        raise


def _inspection_capability_missing(
    completed: subprocess.CompletedProcess[str],
) -> bool:
    """Recognize an older helper that rejects the inspection flag itself."""
    detail = f"{completed.stdout}\n{completed.stderr}".lower()
    if "--inspect-input" not in detail:
        return False
    return any(
        marker in detail
        for marker in (
            "unrecognized argument",
            "unrecognized arguments",
            "unknown option",
            "no such option",
            "invalid option",
            "invalid argument",
        )
    )


def _file_identity(info: os.stat_result) -> FileIdentity:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _capture_normalized_output(
    target: Path,
    workspace: Path,
) -> NormalizedOutput:
    """Bind a helper-produced file to its validated path and inode identity."""
    try:
        if target.parent.resolve() != workspace.resolve():
            raise NormalizeError("video normalization output is invalid")
        descriptor = os.open(
            target,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, RuntimeError) as exc:
        raise NormalizeError("video normalization output is invalid") from exc
    try:
        initial = os.fstat(descriptor)
        path_info = os.stat(target, follow_symlinks=False)
        identity = _file_identity(initial)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_size <= 0
            or not stat.S_ISREG(path_info.st_mode)
            or _file_identity(path_info) != identity
        ):
            raise NormalizeError("video normalization output is invalid")
        try:
            _, proven = inspect_video_descriptor(target, descriptor)
        except VideoPathError as exc:
            raise NormalizeError("video normalization output is invalid") from exc
        current = os.fstat(descriptor)
        current_path = os.stat(target, follow_symlinks=False)
        if (
            not proven
            or _file_identity(current) != identity
            or _file_identity(current_path) != identity
        ):
            raise NormalizeError("video normalization output changed")
        return NormalizedOutput(path=target, identity=identity)
    except OSError as exc:
        raise NormalizeError("video normalization output changed") from exc
    finally:
        os.close(descriptor)


def normalize_file(
    source: Path,
    workflow_id: str,
    index: int,
) -> NormalizedOutput:
    if not source.is_file() or source.is_symlink():
        raise NormalizeError("video input is unavailable")
    pinned_source = _pin_input(source, workflow_id, index)
    try:
        workspace = _temporary_root(workflow_id)
        target = workspace / f"vewm_{index}.mp4"
        if target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)
    except Exception:
        pinned_source.unlink(missing_ok=True)
        raise
    try:
        with _opened_normalizer_script() as (script_path, pass_fds):
            command = [
                sys.executable,
                script_path,
                "--input",
                str(pinned_source),
                "--output",
                str(target),
            ]
            completed, bounded_failure = _run_bounded_subprocess(
                command,
                timeout=NORMALIZER_TIMEOUT_SECONDS,
                env=_normalizer_env(),
                pass_fds=pass_fds,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        target.unlink(missing_ok=True)
        raise NormalizeError("video normalization failed") from exc
    finally:
        pinned_source.unlink(missing_ok=True)
    if bounded_failure:
        target.unlink(missing_ok=True)
        raise NormalizeError("video normalizer output exceeded its bounded capture")
    if completed.returncode != 0 or not target.is_file() or target.is_symlink() or target.stat().st_size <= 0:
        target.unlink(missing_ok=True)
        detail = _bounded(completed.stderr, MAX_STDERR_BYTES)
        logger.warning(
            "video media helper failed returncode=%s stderr_bytes=%s stderr_sha256=%s",
            completed.returncode,
            len(detail.encode("utf-8", "replace")),
            hashlib.sha256(detail.encode("utf-8", "replace")).hexdigest(),
        )
        raise NormalizeError("video normalization failed")
    # Parse the helper's bounded JSON result so a script that exits 0 without
    # producing its declared artifact cannot be treated as a successful upload.
    output = _bounded(completed.stdout, MAX_STDOUT_BYTES)
    try:
        payload: Any = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        target.unlink(missing_ok=True)
        raise NormalizeError("video normalizer returned invalid metadata") from exc
    if not isinstance(payload, dict) or str(payload.get("output") or "") != str(target):
        target.unlink(missing_ok=True)
        raise NormalizeError("video normalizer returned invalid metadata")
    try:
        return _capture_normalized_output(target, workspace)
    except Exception:
        target.unlink(missing_ok=True)
        raise


def normalize_files(
    sources: Iterable[Path],
    workflow_id: str,
) -> list[NormalizedOutput]:
    normalized: list[NormalizedOutput] = []
    try:
        for index, source in enumerate(sources):
            normalized.append(normalize_file(source, workflow_id, index))
        return normalized
    except Exception:
        cleanup(normalized, workflow_id)
        raise


def inspect_files(
    sources: Iterable[Path],
    workflow_id: str,
) -> list[tuple[int, int, int, int, int]]:
    """Confirm video streams through the packaged probe without creating output."""
    source_list = list(sources)
    if not 1 <= len(source_list) <= MAX_INSPECT_FILES:
        raise NormalizeError("video media inspection input is invalid")
    # Resolve helper availability before opening source descriptors so an
    # unavailable optional helper can be reported cleanly to the raw-direct
    # fallback without touching media files.
    _require_inspection_helper()
    inputs: list[_InspectionInput] = []
    identities: list[FileIdentity] = []
    try:
        # Open the trusted script before touching any source: an unavailable
        # helper must surface without a single pin having moved a ctime.
        with _opened_normalizer_script() as (script_path, pass_fds):
            for index, source in enumerate(source_list):
                inputs.append(_prepare_inspection_input(source, workflow_id, index))
            # Baseline only once every pin is linked: duplicate hard links of
            # one inode would otherwise invalidate each other's ctime.
            expected = [_probe_identity(item) for item in inputs]
            # Sources that could not be pinned (read-only shares) fall back
            # to the parent-descriptor handoff.  When this process cannot
            # offer that either, report the capability gap now — with the
            # post-pin identities — so the caller's bounded local proof can
            # admit the upload instead of the helper reading zero bytes.
            unpinned = [item for item in inputs if item.pin is None]
            if unpinned and not _parent_descriptor_handoff_usable():
                raise NormalizerUnavailableError(
                    "video media inspection capability is unavailable",
                    identities=_release_all(inputs, expected),
                )
            command = [sys.executable, script_path]
            for item in inputs:
                if item.pin is not None:
                    handoff = str(item.pin)
                else:
                    try:
                        handoff = _inspection_fd_path(item.descriptor)
                    except NormalizeError:
                        raise NormalizerUnavailableError(
                            "video media inspection capability is unavailable",
                            identities=_release_all(inputs, expected),
                        ) from None
                command.extend(["--inspect-input", handoff])
            # Pinned inputs are reopened by path; only descriptor-handoff
            # fallbacks still need the parent descriptor inherited.
            child_fds = tuple(
                dict.fromkeys((*pass_fds, *(item.descriptor for item in unpinned)))
            )
            completed, bounded_failure = _run_bounded_subprocess(
                command,
                timeout=MEDIA_INSPECTION_TIMEOUT_SECONDS,
                env=_normalizer_env(),
                pass_fds=child_fds,
            )
        # Nothing may have touched the inode while the helper read it; then
        # drop every pin before capturing the identities the upload boundary
        # compares against its own fstat.
        identities = _release_all(inputs, expected)
    except NormalizerUnavailableError:
        raise
    except NormalizeError as exc:
        if str(exc).strip() == "video normalizer is unavailable":
            raise NormalizerUnavailableError(
                "video media inspection capability is unavailable"
            ) from exc
        raise
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NormalizeError("video media inspection failed") from exc
    finally:
        for item in inputs:
            _remove_inspection_pin(item, strict=False)
        _prune_pin_directories(inputs)
        for item in inputs:
            with contextlib.suppress(OSError):
                os.close(item.descriptor)

    if bounded_failure or completed.returncode != 0:
        if not bounded_failure and _inspection_capability_missing(completed):
            raise NormalizerUnavailableError(
                "video media inspection capability is unavailable",
                identities=identities,
            )
        logger.warning(
            "video media inspection helper failed returncode=%s bounded_failure=%s",
            completed.returncode,
            bounded_failure,
        )
        raise NormalizeError("video media inspection failed")
    output = _bounded(completed.stdout, MAX_STDOUT_BYTES)
    try:
        payload: Any = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise NormalizeError("video media inspection returned invalid metadata") from exc
    items = payload.get("items") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("ok") is not True
        or not isinstance(items, list)
    ):
        raise NormalizeError("video media inspection returned invalid metadata")
    if len(items) != len(source_list) or not all(
        isinstance(item, dict) for item in items
    ):
        raise NormalizeError("video media inspection returned invalid metadata")
    reasons = [str(item.get("reason") or "").strip() for item in items]
    if any(reason == "PROBE_FAILED" for reason in reasons):
        raise VideoPathError("input is not a supported video file")
    if any(reasons) or any(
        item.get("category") not in {"direct_only", "compress_only", "both"}
        for item in items
    ):
        # Helper reason codes are fixed enum strings, never media content.
        logger.warning(
            "video media inspection rejected input reasons=%s categories=%s",
            [reason or "-" for reason in reasons],
            [str(item.get("category") or "-") for item in items],
        )
        raise NormalizeError("video media inspection failed")
    return identities


def cleanup(
    outputs: Iterable[NormalizedOutput],
    workflow_id: str,
) -> None:
    workspace = _temporary_root(workflow_id)
    for output in outputs:
        if type(output) is not NormalizedOutput:
            continue
        path = output.path
        if not _under(path.resolve(), workspace.resolve()) or path.is_symlink():
            continue
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError:
            continue
        if (
            not stat.S_ISREG(current.st_mode)
            or _file_identity(current) != output.identity
        ):
            continue
        with contextlib.suppress(OSError):
            path.unlink()
    with contextlib.suppress(OSError):
        if workspace.exists() and not any(workspace.iterdir()):
            workspace.rmdir()
