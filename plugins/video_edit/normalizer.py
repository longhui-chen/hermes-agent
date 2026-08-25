"""Bounded adapter for the packaged hardware media normalizer.

The skill never invokes this script directly.  Hermes owns the subprocess
boundary and only passes validated local files; the normalizer itself is a
media utility, not an authorization or workflow checkpoint implementation.
"""

from __future__ import annotations

import contextlib
import json
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

from plugins.video_edit.paths import safe_id, state_root

NORMALIZER_TIMEOUT_SECONDS = 1800
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


class NormalizeError(RuntimeError):
    """Raised when the bounded hardware normalization step cannot complete."""


def _run_bounded_subprocess(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
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


def _presets_root() -> Path:
    raw = str(os.environ.get("ZETTLAB_PRESETS_DIR", DEFAULT_PRESETS_ROOT) or "").strip()
    if not raw or not os.path.isabs(raw):
        raise NormalizeError("video normalizer is unavailable")
    root = Path(raw).resolve()
    if root.is_symlink() or not root.is_dir():
        raise NormalizeError("video normalizer is unavailable")
    return root


def normalizer_script() -> Path:
    override = str(os.environ.get("ZETTLAB_VIDEO_NORMALIZER", "") or "").strip()
    if override:
        candidates = [Path(override).resolve()]
        root = None
    else:
        root = _presets_root()
        candidates = [
            root / "skills" / "video-edit-workflow-mini" / "scripts" / "normalize.py",
            root / "skills" / "common" / "video-edit-workflow-mini" / "scripts" / "normalize.py",
        ]
    candidate = next((path.resolve() for path in candidates if path.is_file()), candidates[0])
    if candidate.name != "normalize.py" or not candidate.is_file() or candidate.is_symlink():
        raise NormalizeError("video normalizer is unavailable")
    if override:
        # Test/development overrides remain confined to an explicit absolute
        # script; production uses the signed presets bundle above.
        return candidate
    if root is None or not _under(candidate, root):
        raise NormalizeError("video normalizer is unavailable")
    return candidate


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
        fd, name = tempfile.mkstemp(
            dir=str(source.parent),
            prefix=f".hermes-video-input-{safe_id(workflow_id)}-{index}-",
        )
        os.close(fd)
        os.unlink(name)
        snapshot = Path(name)
        os.link(source, snapshot, follow_symlinks=False)
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


def normalize_file(source: Path, workflow_id: str, index: int) -> Path:
    if not source.is_file() or source.is_symlink():
        raise NormalizeError("video input is unavailable")
    pinned_source = _pin_input(source, workflow_id, index)
    try:
        workspace = _temporary_root(workflow_id)
        target = workspace / f"vewm_{index}.mp4"
        if target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(normalizer_script()),
            "--input",
            str(pinned_source),
            "--output",
            str(target),
        ]
    except Exception:
        pinned_source.unlink(missing_ok=True)
        raise
    try:
        completed, bounded_failure = _run_bounded_subprocess(
            command,
            timeout=NORMALIZER_TIMEOUT_SECONDS,
            env=_normalizer_env(),
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
        raise NormalizeError(detail or "video normalization failed")
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
    return target


def normalize_files(sources: Iterable[Path], workflow_id: str) -> list[Path]:
    normalized: list[Path] = []
    try:
        for index, source in enumerate(sources):
            normalized.append(normalize_file(source, workflow_id, index))
        return normalized
    except Exception:
        cleanup(normalized, workflow_id)
        raise


def cleanup(paths: Iterable[Path], workflow_id: str) -> None:
    workspace = _temporary_root(workflow_id)
    for raw in paths:
        path = Path(raw)
        if not _under(path.resolve(), workspace.resolve()) or path.is_symlink():
            continue
        with contextlib.suppress(OSError):
            path.unlink()
    with contextlib.suppress(OSError):
        if workspace.exists() and not any(workspace.iterdir()):
            workspace.rmdir()
