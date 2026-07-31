"""Bounded subprocess runner for already-verified packaged Python scripts.

Callers own the trust decision for the script path and argv.  This module owns
the execution boundary shared by dedicated terminal runners:

* no model-authored shell is involved;
* new scoped-secret callers use inherited one-shot pipes, never argv/Popen env;
* output is redacted while streaming and retained with a fixed memory bound;
* timeout termination covers the wrapper and its subprocess tree.
"""

from __future__ import annotations

import base64
import codecs
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence


_CONTROL_WRAPPER = r"""
import base64
import ctypes
import io
import json
import os
import re
import runpy
import stat
import sys

def harden_linux_process():
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

def write_cgroup_file(path, payload):
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("managed cgroup control is not a regular file")
        encoded = payload.encode("ascii")
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short managed cgroup control write")
    finally:
        os.close(descriptor)

def current_unified_cgroup():
    with open("/proc/self/cgroup", "r", encoding="ascii") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise OSError("oversized process cgroup metadata")
    matches = [
        line[3:]
        for line in raw.splitlines()
        if line.startswith("0::")
    ]
    if len(matches) != 1 or not matches[0].startswith("/"):
        raise OSError("unified cgroup membership unavailable")
    return matches[0]

def enter_managed_cgroup():
    if len(sys.argv) == 1:
        harden_linux_process()
        return
    if (
        len(sys.argv) != 6
        or sys.argv[1] != "--managed-cgroup"
        or not sys.platform.startswith("linux")
    ):
        raise OSError("invalid managed cgroup bootstrap")

    cgroup_path = sys.argv[2]
    expected_relative = sys.argv[3]
    target_uid = int(sys.argv[4])
    target_gid = int(sys.argv[5])
    if (
        not cgroup_path.startswith("/sys/fs/cgroup/")
        or not expected_relative.startswith("/")
        or target_uid != 65534
        or target_gid != 65534
        or os.geteuid() != 0
    ):
        raise OSError("invalid managed cgroup boundary")

    write_cgroup_file(
        os.path.join(cgroup_path, "cgroup.procs"),
        str(os.getpid()),
    )
    if current_unified_cgroup() != expected_relative:
        raise OSError("managed cgroup attach verification failed")

    os.setgroups([])
    if hasattr(os, "setresgid"):
        os.setresgid(target_gid, target_gid, target_gid)
    else:
        os.setgid(target_gid)
    if hasattr(os, "setresuid"):
        os.setresuid(target_uid, target_uid, target_uid)
    else:
        os.setuid(target_uid)
    if (
        os.getuid() != target_uid
        or os.geteuid() != target_uid
        or os.getgid() != target_gid
        or os.getegid() != target_gid
        or os.getgroups()
    ):
        raise OSError("managed identity drop verification failed")
    harden_linux_process()

enter_managed_cgroup()
control = json.loads(sys.stdin.read() or "{}")
env = control.get("env") or {}
secret_fds = control.get("secret_fds") or {}
script = control["script"]
argv = control.get("argv") or [script]
pythonpath = control.get("pythonpath")
stdin_text = control.get("stdin")
script_b64 = control.get("script_b64")
if isinstance(pythonpath, list):
    sys.path = [str(item) for item in pythonpath if item]
for key, value in env.items():
    if value is not None:
        os.environ[str(key)] = str(value)
for key, descriptor in secret_fds.items():
    key = str(key)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
        raise ValueError("invalid secret environment key")
    descriptor = int(descriptor)
    if descriptor < 3:
        raise ValueError("invalid secret descriptor")
    os.fstat(descriptor)
    os.set_inheritable(descriptor, False)
    os.environ.pop(key, None)
    os.environ[key + "_FD"] = str(descriptor)
os.environ.pop("PYTHONPATH", None)
sys.argv = [script, *[str(arg) for arg in argv[1:]]]
if isinstance(stdin_text, str):
    sys.stdin = io.StringIO(stdin_text)
if isinstance(script_b64, str):
    source = base64.b64decode(script_b64, validate=True)
    namespace = {
        "__name__": "__main__",
        "__file__": script,
        "__cached__": None,
        "__loader__": None,
        "__package__": None,
        "__spec__": None,
    }
    exec(compile(source, script, "exec"), namespace, namespace)
else:
    runpy.run_path(script, run_name="__main__")
"""

_PROCESS_TERM_GRACE_SECONDS = 0.25
_PROCESS_KILL_GRACE_SECONDS = 1.0
_OUTPUT_DRAIN_GRACE_SECONDS = 1.0
_DEFAULT_MAX_OUTPUT_CHARS = 20_000
_SECRET_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_INJECTED_SECRETS = 8
_MAX_INJECTED_SECRET_BYTES = 4 * 1024
_MAX_TOTAL_INJECTED_SECRET_BYTES = 16 * 1024
_MANAGED_GATEWAY_ENV = "HERMES_MANAGED_GATEWAY"
_CGROUP2_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_CGROUP_METADATA_MAX_BYTES = 4096
_CGROUP_CLEANUP_TIMEOUT_SECONDS = 2.0
_CGROUP_POLL_INTERVAL_SECONDS = 0.02
_CGROUP_KILL_RETRY_ATTEMPTS = 3
_CGROUP_KILL_RETRY_SECONDS = 0.05
_MANAGED_INVOCATION_MEMORY_MAX_BYTES = 128 * 1024 * 1024
_MANAGED_INVOCATION_MEMORY_SWAP_MAX_BYTES = 0
_MANAGED_INVOCATION_PIDS_MAX = 64
_MANAGED_RUNNER_UID = 65534
_MANAGED_RUNNER_GID = 65534
_MANAGED_CGROUP_PREFIX = "agentcomputer"
_MANAGED_SUPERVISOR_CGROUP = "agentcomputer-supervisor"
_MANAGED_CGROUP_ROOT_ENV = "HERMES_MANAGED_CGROUP_ROOT"
_MANAGED_CGROUP_UNIT_ENV = "HERMES_MANAGED_CGROUP_UNIT"
_SYSTEMCTL_PATHS = ("/bin/systemctl", "/usr/bin/systemctl")
_SYSTEMD_KILL_RETRY_ATTEMPTS = 2
_SYSTEMD_KILL_TIMEOUT_SECONDS = 0.5
_SYSTEMD_KILL_POLL_SECONDS = 0.02


@dataclass(frozen=True)
class TrustedPythonResult:
    output: str
    returncode: int
    timed_out: bool = False
    interrupted: bool = False


@dataclass(frozen=True)
class _ManagedInvocationCgroup:
    path: Path
    relative_path: str
    delegation_root_path: Path | None = None
    delegation_root_relative_path: str | None = None
    delegation_root_identity: tuple[int, int] | None = None


class _BoundedText:
    """Keep enough head/tail text to render a fixed-size truncation result."""

    def __init__(self, limit: int) -> None:
        self.limit = max(256, int(limit))
        self.total = 0
        self._head = ""
        self._tail = ""

    def append(self, text: str) -> None:
        if not text:
            return
        self.total += len(text)
        if len(self._head) < self.limit:
            missing = self.limit - len(self._head)
            self._head += text[:missing]
        self._tail = (self._tail + text)[-self.limit :]

    def render(self) -> str:
        if self.total <= self.limit:
            return self._head
        marker = (
            f"\n\n... [OUTPUT TRUNCATED - {self.total - self.limit} chars omitted "
            f"out of {self.total} total] ...\n\n"
        )
        if len(marker) >= self.limit:
            return marker[: self.limit]
        available = self.limit - len(marker)
        head_chars = int(available * 0.4)
        tail_chars = available - head_chars
        return self._head[:head_chars] + marker + self._tail[-tail_chars:]


class _StreamingSecretRedactor:
    """Redact exact secret values without retaining unbounded raw output."""

    def __init__(self, secrets: Iterable[str], sink: _BoundedText) -> None:
        normalized_secrets: set[str] = {str(value) for value in secrets if value}
        self._secrets: tuple[str, ...] = tuple(
            sorted(
                normalized_secrets,
                key=lambda value: len(value),
                reverse=True,
            )
        )
        self._pattern = (
            re.compile("|".join(re.escape(value) for value in self._secrets))
            if self._secrets
            else None
        )
        self._overlap = max((len(value) - 1 for value in self._secrets), default=0)
        self._pending = ""
        self._sink = sink

    def feed(self, text: str, *, final: bool = False) -> None:
        data = self._pending + text
        if final:
            if self._pattern is not None:
                data = self._pattern.sub("[REDACTED]", data)
            self._pending = ""
            self._sink.append(data)
            return
        if self._pattern is None:
            self._sink.append(data)
            self._pending = ""
            return

        safe_end = max(0, len(data) - self._overlap)
        cursor = 0
        pending_start = safe_end
        ready_parts: list[str] = []
        for match in self._pattern.finditer(data):
            if match.start() >= safe_end:
                break
            ready_parts.append(data[cursor : match.start()])
            ready_parts.append("[REDACTED]")
            cursor = match.end()
            pending_start = max(pending_start, cursor)
        if cursor < safe_end:
            ready_parts.append(data[cursor:safe_end])

        self._pending = data[pending_start:]
        self._sink.append("".join(ready_parts))


def _max_output_chars() -> int:
    try:
        from tools.tool_output_limits import get_max_bytes

        return max(256, int(get_max_bytes()))
    except Exception:
        return _DEFAULT_MAX_OUTPUT_CHARS


def isolated_python_path(*, cwd: Path, stdlib_only: bool = False) -> list[str]:
    """Build an import path without model-writable cwd/PYTHONPATH entries."""

    import sysconfig

    blocked_exact: set[Path] = set()
    blocked_roots: set[Path] = set()
    for raw in ("", ".", str(cwd), os.getcwd()):
        try:
            blocked_exact.add(Path(raw or ".").resolve())
        except OSError:
            pass
    for raw in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not raw:
            continue
        try:
            blocked_roots.add(Path(raw).resolve())
        except OSError:
            pass

    candidates = list(sys.path)
    library_keys = (
        ("stdlib", "platstdlib")
        if stdlib_only
        else ("stdlib", "platstdlib", "purelib", "platlib")
    )
    for key in library_keys:
        value = sysconfig.get_paths().get(key)
        if value:
            candidates.append(value)

    base_roots: tuple[Path, ...] = ()
    if stdlib_only:
        roots = []
        for raw in (sys.base_prefix, sys.base_exec_prefix):
            try:
                resolved = Path(raw).resolve()
            except OSError:
                continue
            if resolved not in roots:
                roots.append(resolved)
        base_roots = tuple(roots)

    allowed: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not raw:
            continue
        try:
            resolved = Path(raw).resolve()
        except OSError:
            continue
        if resolved in blocked_exact:
            continue
        if any(resolved == root or root in resolved.parents for root in blocked_roots):
            continue
        if stdlib_only:
            if not any(
                resolved == root or root in resolved.parents for root in base_roots
            ):
                continue
            if any(
                part in {"site-packages", "dist-packages"} for part in resolved.parts
            ):
                continue
        value = str(resolved)
        if value not in seen:
            seen.add(value)
            allowed.append(value)
    return allowed


def _read_bounded_ascii(path: Path, *, limit: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"not a regular control file: {path.name}")
        raw = os.read(descriptor, limit + 1)
        if len(raw) > limit:
            raise OSError(f"oversized control file: {path.name}")
        return raw.decode("ascii")
    finally:
        os.close(descriptor)


def _write_control_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"not a regular control file: {path.name}")
        if os.write(descriptor, payload) != len(payload):
            raise OSError(f"short control write: {path.name}")
    finally:
        os.close(descriptor)


def _current_unified_cgroup() -> str:
    raw = _read_bounded_ascii(
        _PROC_SELF_CGROUP,
        limit=_CGROUP_METADATA_MAX_BYTES,
    )
    matches = [
        line[3:]
        for line in raw.splitlines()
        if line.startswith("0::")
    ]
    if len(matches) != 1:
        raise OSError("unified cgroup membership unavailable")
    relative = matches[0]
    parsed = PurePosixPath(relative)
    if (
        not relative.startswith("/")
        or ".." in parsed.parts
        or "\x00" in relative
    ):
        raise OSError("invalid unified cgroup membership")
    return relative


def _resolve_managed_delegation_root() -> tuple[Path, str, tuple[int, int]]:
    expected_unit = os.environ.get(_MANAGED_CGROUP_UNIT_ENV, "")
    expected_root_relative = os.environ.get(_MANAGED_CGROUP_ROOT_ENV, "")
    parsed_root = PurePosixPath(expected_root_relative)
    if (
        not expected_unit
        or "/" in expected_unit
        or expected_unit in {".", ".."}
        or not expected_unit.endswith(".service")
        or not expected_root_relative.startswith("/")
        or expected_root_relative == "/"
        or ".." in parsed_root.parts
        or "\x00" in expected_root_relative
        or parsed_root.name != expected_unit
    ):
        raise OSError("managed Agent Creator delegation identity is unavailable")

    expected_supervisor_relative = (
        f"{expected_root_relative.rstrip('/')}/{_MANAGED_SUPERVISOR_CGROUP}"
    )
    if _current_unified_cgroup() != expected_supervisor_relative:
        raise OSError("managed Agent Creator is outside its supervisor cgroup")

    root = _CGROUP2_ROOT.resolve(strict=True)
    delegation_root = (
        root / expected_root_relative.lstrip("/")
    ).resolve(strict=True)
    supervisor = (delegation_root / _MANAGED_SUPERVISOR_CGROUP).resolve(
        strict=True
    )
    try:
        delegation_root.relative_to(root)
    except ValueError as exc:
        raise OSError("managed delegation root escapes cgroup v2") from exc
    if (
        delegation_root.name != expected_unit
        or supervisor.parent != delegation_root
        or supervisor.name != _MANAGED_SUPERVISOR_CGROUP
        or not (delegation_root / "cgroup.procs").is_file()
        or not (delegation_root / "cgroup.kill").is_file()
        or not (supervisor / "cgroup.procs").is_file()
    ):
        raise OSError("managed Agent Creator delegation root is invalid")

    enabled = set(
        _read_bounded_ascii(
            delegation_root / "cgroup.subtree_control",
            limit=_CGROUP_METADATA_MAX_BYTES,
        ).split()
    )
    if not {"memory", "pids"}.issubset(enabled):
        raise OSError("managed Agent Creator controllers are not enabled")
    if _read_bounded_ascii(
        delegation_root / "cgroup.procs",
        limit=_CGROUP_METADATA_MAX_BYTES,
    ).strip():
        raise OSError("managed Agent Creator delegation root is not process-free")

    identity = delegation_root.stat()
    return (
        delegation_root,
        expected_root_relative.rstrip("/"),
        (identity.st_dev, identity.st_ino),
    )


def _create_managed_invocation_cgroup() -> _ManagedInvocationCgroup:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise OSError("managed Agent Creator requires Linux root delegation")
    if not (_CGROUP2_ROOT / "cgroup.controllers").is_file():
        raise OSError("unified cgroup v2 is unavailable")

    (
        delegation_root,
        delegation_root_relative,
        delegation_root_identity,
    ) = _resolve_managed_delegation_root()

    cgroup_name = (
        f"{_MANAGED_CGROUP_PREFIX}-{os.getpid()}-"
        f"{threading.get_native_id()}-{secrets.token_hex(8)}"
    )
    cgroup_path = delegation_root / cgroup_name
    os.mkdir(cgroup_path, 0o755)
    try:
        if stat.S_ISLNK(cgroup_path.lstat().st_mode):
            raise OSError("managed invocation cgroup is a symlink")
        for control in (
            "cgroup.procs",
            "cgroup.kill",
            "cgroup.events",
            "memory.max",
            "memory.swap.max",
            "memory.oom.group",
            "pids.max",
        ):
            if not (cgroup_path / control).is_file():
                raise OSError(f"managed invocation cgroup lacks {control}")
        _write_control_file(
            cgroup_path / "memory.max",
            str(_MANAGED_INVOCATION_MEMORY_MAX_BYTES).encode("ascii"),
        )
        _write_control_file(
            cgroup_path / "memory.swap.max",
            str(_MANAGED_INVOCATION_MEMORY_SWAP_MAX_BYTES).encode("ascii"),
        )
        _write_control_file(cgroup_path / "memory.oom.group", b"1")
        _write_control_file(
            cgroup_path / "pids.max",
            str(_MANAGED_INVOCATION_PIDS_MAX).encode("ascii"),
        )
        expected_limits = {
            "memory.max": str(_MANAGED_INVOCATION_MEMORY_MAX_BYTES),
            "memory.swap.max": str(
                _MANAGED_INVOCATION_MEMORY_SWAP_MAX_BYTES
            ),
            "memory.oom.group": "1",
            "pids.max": str(_MANAGED_INVOCATION_PIDS_MAX),
        }
        for control, expected in expected_limits.items():
            actual = _read_bounded_ascii(
                cgroup_path / control,
                limit=_CGROUP_METADATA_MAX_BYTES,
            ).strip()
            if actual != expected:
                raise OSError(f"managed invocation cgroup rejected {control}")
    except Exception:
        try:
            os.rmdir(cgroup_path)
        except OSError:
            pass
        raise

    relative = f"{delegation_root_relative}/{cgroup_name}"
    return _ManagedInvocationCgroup(
        path=cgroup_path,
        relative_path=relative,
        delegation_root_path=delegation_root,
        delegation_root_relative_path=delegation_root_relative,
        delegation_root_identity=delegation_root_identity,
    )


def _cgroup_is_populated(cgroup: _ManagedInvocationCgroup) -> bool:
    events = _read_bounded_ascii(
        cgroup.path / "cgroup.events",
        limit=_CGROUP_METADATA_MAX_BYTES,
    )
    for line in events.splitlines():
        key, separator, value = line.partition(" ")
        if key == "populated" and separator:
            if value == "0":
                return False
            if value == "1":
                return True
            break
    raise OSError("managed invocation cgroup lacks valid populated state")


def _request_systemd_unit_sigkill(unit_name: str) -> None:
    """Ask PID 1 to SIGKILL one already-verified service control group."""

    if (
        unit_name != os.environ.get(_MANAGED_CGROUP_UNIT_ENV, "")
        or "/" in unit_name
        or unit_name in {"", ".", ".."}
        or not unit_name.endswith(".service")
    ):
        raise OSError("managed systemd unit identity is invalid")

    clean_env = {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "SYSTEMD_COLORS": "0",
    }
    last_error: BaseException = OSError("systemctl is unavailable")
    for attempt in range(_SYSTEMD_KILL_RETRY_ATTEMPTS):
        for systemctl_path in _SYSTEMCTL_PATHS:
            try:
                process = subprocess.Popen(
                    [
                        systemctl_path,
                        "--system",
                        "kill",
                        "--kill-whom=all",
                        "--signal=SIGKILL",
                        unit_name,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=clean_env,
                    close_fds=True,
                )
            except FileNotFoundError as exc:
                last_error = exc
                continue
            except (OSError, ValueError) as exc:
                last_error = exc
                break
            deadline = time.monotonic() + _SYSTEMD_KILL_TIMEOUT_SECONDS
            while True:
                try:
                    returncode = process.poll()
                except OSError as exc:
                    last_error = exc
                    break
                if returncode == 0:
                    return
                if returncode is not None:
                    last_error = OSError(
                        f"systemctl kill exited with status {returncode}"
                    )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    last_error = TimeoutError("systemctl kill timed out")
                    break
                time.sleep(min(_SYSTEMD_KILL_POLL_SECONDS, remaining))
            break
        if attempt + 1 < _SYSTEMD_KILL_RETRY_ATTEMPTS:
            time.sleep(_CGROUP_KILL_RETRY_SECONDS)
    raise OSError("systemd unit SIGKILL request failed") from last_error


def _escalate_managed_service_cleanup(
    cgroup: _ManagedInvocationCgroup,
    cause: BaseException,
) -> None:
    """Kill the service scope when the delegated child cannot be proven empty."""

    try:
        root = _CGROUP2_ROOT.resolve(strict=True)
        service_path = cgroup.delegation_root_path
        service_relative = cgroup.delegation_root_relative_path
        expected_identity = cgroup.delegation_root_identity
        if (
            service_path is None
            or service_relative is None
            or expected_identity is None
        ):
            raise OSError("managed service cgroup identity is unavailable")
        resolved_service = service_path.resolve(strict=True)
        expected_service = (
            root / service_relative.lstrip("/")
        ).resolve(strict=True)
        identity = resolved_service.stat()
        if (
            resolved_service != expected_service
            or resolved_service.name
            != os.environ.get(_MANAGED_CGROUP_UNIT_ENV, "")
            or (identity.st_dev, identity.st_ino) != expected_identity
            or cgroup.path.parent.resolve(strict=True) != resolved_service
            or PurePosixPath(cgroup.relative_path).parent
            != PurePosixPath(service_relative)
            or not cgroup.path.name.startswith(f"{_MANAGED_CGROUP_PREFIX}-")
            or _current_unified_cgroup()
            != f"{service_relative}/{_MANAGED_SUPERVISOR_CGROUP}"
        ):
            raise OSError("managed invocation is outside its delegated service")
        service_path = resolved_service
    except BaseException as exc:
        # If the path identity cannot be proven, never guess another cgroup.
        # Killing the tracked systemd MainPID still enters KillMode=control-group.
        os.kill(os.getpid(), signal.SIGKILL)
        raise OSError("managed service cgroup identity verification failed") from exc

    service_kill_error: BaseException | None = None
    for attempt in range(_CGROUP_KILL_RETRY_ATTEMPTS):
        try:
            _write_control_file(service_path / "cgroup.kill", b"1")
            service_kill_error = None
            break
        except BaseException as exc:
            service_kill_error = exc
            if attempt + 1 < _CGROUP_KILL_RETRY_ATTEMPTS:
                time.sleep(_CGROUP_KILL_RETRY_SECONDS)

    systemd_kill_error: BaseException | None = None
    if service_kill_error is not None:
        try:
            _request_systemd_unit_sigkill(service_path.name)
        except BaseException as exc:
            systemd_kill_error = exc

    # Both successful group-kill paths include this process. Self-SIGKILL is
    # an independent final trigger if either request returns unexpectedly.
    os.kill(os.getpid(), signal.SIGKILL)
    if systemd_kill_error is not None:
        raise OSError("managed service SIGKILL escalation failed") from (
            systemd_kill_error
        )
    if service_kill_error is not None:
        raise OSError("managed service cgroup cleanup escalation failed") from (
            service_kill_error
        )
    raise OSError("managed service cgroup cleanup escalation returned") from cause


def _kill_and_remove_managed_cgroup(
    cgroup: _ManagedInvocationCgroup,
    process: subprocess.Popen[bytes] | None,
) -> None:
    kill_error: BaseException | None = None
    kill_succeeded = False
    for attempt in range(_CGROUP_KILL_RETRY_ATTEMPTS):
        try:
            _write_control_file(cgroup.path / "cgroup.kill", b"1")
            kill_succeeded = True
            kill_error = None
            break
        except BaseException as exc:
            kill_error = exc
            if attempt + 1 < _CGROUP_KILL_RETRY_ATTEMPTS:
                time.sleep(_CGROUP_KILL_RETRY_SECONDS)

    if not kill_succeeded:
        _escalate_managed_service_cleanup(
            cgroup,
            kill_error or OSError("managed invocation cgroup kill failed"),
        )

    cleanup_error: BaseException | None = None
    deadline = time.monotonic() + _CGROUP_CLEANUP_TIMEOUT_SECONDS
    populated = True
    while time.monotonic() < deadline:
        try:
            populated = _cgroup_is_populated(cgroup)
        except BaseException as exc:
            cleanup_error = exc
            break
        if not populated:
            break
        time.sleep(_CGROUP_POLL_INTERVAL_SECONDS)
    if populated and cleanup_error is None:
        cleanup_error = kill_error or OSError(
            "managed invocation cgroup remained populated"
        )

    if process is not None:
        try:
            process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
            except (subprocess.TimeoutExpired, OSError) as exc:
                cleanup_error = cleanup_error or exc
        except OSError as exc:
            cleanup_error = cleanup_error or exc

    if cleanup_error is None:
        try:
            os.rmdir(cgroup.path)
        except BaseException as exc:
            cleanup_error = exc
    if cleanup_error is not None:
        _escalate_managed_service_cleanup(cgroup, cleanup_error)
        # The real escalation kills this service. Keep a defensive exception
        # for tests or a non-conforming kernel where SIGKILL unexpectedly returns.
        raise OSError("managed invocation cgroup cleanup failed") from cleanup_error


def _terminate_process_tree(process: subprocess.Popen) -> None:
    """Terminate a dedicated wrapper and all descendants, cross-platform."""

    if os.name != "nt":
        # Unmanaged/dev invocations retain the existing dedicated PGID boundary.
        process_group = process.pid
        try:
            os.killpg(process_group, signal.SIGTERM)  # windows-footgun: ok
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                pass

    try:
        process.wait(timeout=_PROCESS_TERM_GRACE_SECONDS)
    except (subprocess.TimeoutExpired, OSError):
        pass

    if os.name != "nt":
        try:
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
            os.killpg(process_group, kill_signal)  # windows-footgun: ok
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=_PROCESS_KILL_GRACE_SECONDS)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _drain_output(
    stream,
    *,
    secrets: Sequence[str],
    sink: _BoundedText,
    completed: threading.Event,
) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    redactor = _StreamingSecretRedactor(secrets, sink)
    try:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                redactor.feed(decoder.decode(chunk))
            redactor.feed(decoder.decode(b"", final=True), final=True)
        except (OSError, ValueError):
            redactor.feed(decoder.decode(b"", final=True), final=True)
    finally:
        completed.set()


def run_trusted_python_script(
    *,
    script: Path,
    argv: Sequence[str],
    cwd: Path,
    base_env: Mapping[str, str],
    injected_env: Mapping[str, str],
    injected_secrets: Mapping[str, str] | None = None,
    timeout: float,
    stdin_text: Optional[str] = None,
    secret_values: Sequence[str] = (),
    script_bytes: Optional[bytes] = None,
    stdlib_only: bool = False,
) -> TrustedPythonResult:
    """Execute one verified Python script through the isolated wrapper.

    New callers must keep secrets in ``injected_secrets``; ``injected_env``
    remains for compatibility and non-secret metadata. Each injected secret
    travels over a dedicated inherited pipe and is exposed to the verified
    script only as ``<KEY>_FD``.
    """

    run_env = {str(key): str(value) for key, value in base_env.items()}
    run_env.pop("PYTHONPATH", None)
    for key in injected_env:
        run_env.pop(str(key), None)
    normalized_secrets = {
        str(key): str(value)
        for key, value in (injected_secrets or {}).items()
    }
    if normalized_secrets and os.name == "nt":
        raise OSError("injected secret descriptors are unsupported on Windows")
    if len(normalized_secrets) > _MAX_INJECTED_SECRETS:
        raise ValueError("too many injected secrets")

    encoded_secrets: dict[str, bytes] = {}
    total_secret_bytes = 0
    injected_env_keys = {str(key) for key in injected_env}
    secret_keys = set(normalized_secrets)
    descriptor_keys = {f"{key}_FD" for key in secret_keys}
    if secret_keys & descriptor_keys:
        raise ValueError("injected secret descriptor keys overlap")
    for key, value in normalized_secrets.items():
        descriptor_key = f"{key}_FD"
        if _SECRET_ENV_KEY_RE.fullmatch(key) is None:
            raise ValueError("invalid injected secret key")
        if key in injected_env_keys or descriptor_key in injected_env_keys:
            raise ValueError("secret and metadata environment keys overlap")
        encoded = value.encode("utf-8")
        if len(encoded) > _MAX_INJECTED_SECRET_BYTES:
            raise ValueError("injected secret is too large")
        total_secret_bytes += len(encoded)
        if total_secret_bytes > _MAX_TOTAL_INJECTED_SECRET_BYTES:
            raise ValueError("injected secrets are too large")
        encoded_secrets[key] = encoded
        run_env.pop(key, None)
        run_env.pop(descriptor_key, None)

    secret_pipes: dict[str, tuple[int, int]] = {}
    try:
        for key in encoded_secrets:
            secret_pipes[key] = os.pipe()
    except Exception:
        for read_fd, write_fd in secret_pipes.values():
            os.close(read_fd)
            os.close(write_fd)
        raise

    try:
        control = json.dumps(
            {
                "script": str(script),
                "argv": [str(value) for value in argv],
                "env": {
                    str(key): str(value)
                    for key, value in injected_env.items()
                },
                "secret_fds": {
                    key: read_fd
                    for key, (read_fd, _write_fd) in secret_pipes.items()
                },
                "pythonpath": isolated_python_path(
                    cwd=cwd,
                    stdlib_only=stdlib_only,
                ),
                "stdin": stdin_text,
                "script_b64": (
                    base64.b64encode(script_bytes).decode("ascii")
                    if script_bytes is not None
                    else None
                ),
            },
            ensure_ascii=False,
        ).encode("utf-8")
    except Exception:
        for read_fd, write_fd in secret_pipes.values():
            os.close(read_fd)
            os.close(write_fd)
        raise

    managed_cgroup: _ManagedInvocationCgroup | None = None
    try:
        if os.environ.get(_MANAGED_GATEWAY_ENV) == "1":
            managed_cgroup = _create_managed_invocation_cgroup()
        worker_argv = [sys.executable, "-I", "-S", "-c", _CONTROL_WRAPPER]
        if managed_cgroup is not None:
            worker_argv.extend([
                "--managed-cgroup",
                str(managed_cgroup.path),
                managed_cgroup.relative_path,
                str(_MANAGED_RUNNER_UID),
                str(_MANAGED_RUNNER_GID),
            ])
        if os.name == "nt":
            from hermes_cli._subprocess_compat import windows_hide_flags

            process: subprocess.Popen[bytes] = subprocess.Popen(
                worker_argv,
                cwd=str(cwd),
                env=run_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=(
                    windows_hide_flags()
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
            )
        else:
            process = subprocess.Popen(
                worker_argv,
                cwd=str(cwd),
                env=run_env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=tuple(
                    read_fd for read_fd, _write_fd in secret_pipes.values()
                ),
            )
    except Exception:
        for read_fd, write_fd in secret_pipes.values():
            os.close(read_fd)
            os.close(write_fd)
        if managed_cgroup is not None:
            _kill_and_remove_managed_cgroup(managed_cgroup, None)
        raise

    open_read_fds = {
        read_fd for read_fd, _write_fd in secret_pipes.values()
    }
    open_write_fds = {
        write_fd for _read_fd, write_fd in secret_pipes.values()
    }
    sink: _BoundedText | None = None
    output_completed: threading.Event | None = None
    drain_thread: threading.Thread | None = None
    drain_started = False
    redaction_values = tuple(secret_values) + tuple(normalized_secrets.values())
    timed_out = False
    interrupted = False
    execution_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        for read_fd in tuple(open_read_fds):
            try:
                os.close(read_fd)
            finally:
                open_read_fds.discard(read_fd)
        if process.stdout is None:
            raise OSError("trusted runner output pipe unavailable")
        sink = _BoundedText(_max_output_chars())
        output_completed = threading.Event()
        drain_thread = threading.Thread(
            target=_drain_output,
            kwargs={
                "stream": process.stdout,
                "secrets": redaction_values,
                "sink": sink,
                "completed": output_completed,
            },
            name="trusted-python-output",
            daemon=True,
        )
        drain_thread.start()
        drain_started = True

        if process.stdin is not None:
            try:
                process.stdin.write(control)
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        for key, (_read_fd, write_fd) in secret_pipes.items():
            try:
                encoded = encoded_secrets[key]
                offset = 0
                while offset < len(encoded):
                    offset += os.write(write_fd, encoded[offset:])
            except (BrokenPipeError, OSError):
                pass
            finally:
                if write_fd in open_write_fds:
                    try:
                        os.close(write_fd)
                    finally:
                        open_write_fds.discard(write_fd)

        deadline = time.monotonic() + max(0.001, float(timeout))
        assert output_completed is not None
        while process.poll() is None or not output_completed.is_set():
            try:
                from tools.interrupt import is_interrupted

                interrupted = bool(is_interrupted())
            except Exception:
                interrupted = False
            if interrupted:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            wait_time = min(0.1, remaining)
            if process.poll() is None:
                try:
                    process.wait(timeout=wait_time)
                except subprocess.TimeoutExpired:
                    continue
            else:
                output_completed.wait(timeout=wait_time)
        if (timed_out or interrupted) and managed_cgroup is None:
            _terminate_process_tree(process)
    except BaseException as exc:
        execution_error = exc
    finally:
        for descriptor in tuple(open_read_fds | open_write_fds):
            try:
                os.close(descriptor)
            except OSError:
                pass
            finally:
                open_read_fds.discard(descriptor)
                open_write_fds.discard(descriptor)
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        if managed_cgroup is not None:
            try:
                _kill_and_remove_managed_cgroup(managed_cgroup, process)
            except BaseException as exc:
                cleanup_error = exc
        elif execution_error is not None and process.poll() is None:
            _terminate_process_tree(process)
        if not drain_started and process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass

    if drain_started and drain_thread is not None:
        drain_thread.join(timeout=_OUTPUT_DRAIN_GRACE_SECONDS)
    if (
        drain_started
        and drain_thread is not None
        and drain_thread.is_alive()
    ):
        try:
            process.stdout.close()
        except OSError:
            pass
        drain_thread.join(timeout=_OUTPUT_DRAIN_GRACE_SECONDS)
    if cleanup_error is not None:
        raise OSError("managed Agent Creator cleanup failed") from cleanup_error
    if execution_error is not None:
        raise execution_error.with_traceback(execution_error.__traceback__)

    assert sink is not None
    output = sink.render()
    try:
        from tools.ansi_strip import strip_ansi

        output = strip_ansi(output)
    except Exception:
        pass
    # ANSI removal can join a secret that was deliberately split by escape
    # sequences and therefore invisible to the streaming exact-match pass.
    for secret in redaction_values:
        if secret:
            output = output.replace(str(secret), "[REDACTED]")
    try:
        from agent.redact import redact_sensitive_text

        output = (
            redact_sensitive_text(output.strip(), force=True, code_file=False)
            if output
            else ""
        )
    except Exception:
        output = output.strip()

    return TrustedPythonResult(
        output=output,
        returncode=124
        if timed_out
        else (130 if interrupted else int(process.returncode or 0)),
        timed_out=timed_out,
        interrupted=interrupted,
    )
