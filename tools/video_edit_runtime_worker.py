#!/usr/bin/env python3
"""Preloaded worker for trusted video-edit helper execution.

The gateway freezes this source and its standard-library closure before any
model-controlled terminal command. The process tree starts lazily for the first
trusted video command and is recycled after bounded idle time. Each one-shot
helper can therefore use only the frozen runtime modules plus the source bundle
verified by its owning profile process.
"""

from __future__ import annotations

import argparse
import array
import bz2
import contextlib
import dataclasses
import datetime
import errno
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import ipaddress
import json
import lzma
import math
import mimetypes
import os
import pathlib
import re
import secrets
import select
import shutil
import signal
import socket
import ssl
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import types
import typing
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from process_security import (
    apply_worker_memory_limit,
    bind_process_to_parent,
    enable_child_subreaper,
    harden_sensitive_process,
)


_MAX_FRAME_BYTES = 8 * 1024 * 1024
# Python and native codec libraries reserve substantially more virtual address
# space than their resident footprint. Keep RSS bounded separately while
# allowing enough address-space headroom for trusted helpers to finish.
_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_SUBTREE_RSS_LIMIT_BYTES = 256 * 1024 * 1024
_SUBTREE_PROCESS_LIMIT = 16
_OWNED_PROCESS_CLEANUP_LIMIT = 256
_PROC_TABLE_SCAN_LIMIT = 8192
_PROCESS_TREE_CLEANUP_SECONDS = 2.0
_RUNTIME_CONTEXT_MODULE = "_zettlab_video_runtime_context"


def _preload_optional_runtime_modules() -> None:
    """Load lazy stdlib implementation modules before filesystem lockdown."""
    for name in (
        "_blake2",
        "_hashlib",
        "_md5",
        "_sha1",
        "_sha2",
        "_sha3",
        "encodings.ascii",
        "encodings.idna",
        "encodings.latin_1",
        "encodings.utf_8",
        "http.client",
        "stringprep",
        "unicodedata",
    ):
        try:
            importlib.import_module(name)
        except ImportError:
            # CPython versions expose slightly different accelerator names.
            # Missing optional accelerators are safe because later filesystem
            # imports are disabled and the helper will fail closed if needed.
            continue


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("video runtime channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(channel: socket.socket) -> dict[str, typing.Any]:
    size = struct.unpack("!I", _recv_exact(channel, 4))[0]
    if size <= 0 or size > _MAX_FRAME_BYTES:
        raise ValueError("invalid video runtime frame size")
    payload = json.loads(_recv_exact(channel, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid video runtime frame")
    return payload


def _send_frame(channel: socket.socket, payload: dict[str, typing.Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_FRAME_BYTES:
        raise ValueError("video runtime response too large")
    channel.sendall(struct.pack("!I", len(encoded)) + encoded)


class _PinnedSourceFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, source_bundle: dict[str, dict[str, str]]) -> None:
        self._source_bundle = source_bundle

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname in self._source_bundle:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):  # noqa: ANN001
        return None

    def exec_module(self, module) -> None:  # noqa: ANN001
        source = self._source_bundle[module.__name__]
        module.__file__ = source["path"]
        exec(compile(source["source"], source["path"], "exec"), module.__dict__)


def _exit_code(value: typing.Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return 1


def _execute(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
    source_bundle = payload.get("source_bundle") or {}
    env = payload.get("env") or {}
    context = payload.get("context") or {}
    secrets = payload.get("secrets") or {}
    argv = payload.get("argv") or []
    script = str(payload.get("script") or "")
    cwd = str(payload.get("cwd") or "")
    if (
        not isinstance(source_bundle, dict)
        or "__main__" not in source_bundle
        or not isinstance(env, dict)
        or not isinstance(context, dict)
        or not isinstance(secrets, dict)
        or not isinstance(argv, list)
        or not script
    ):
        raise ValueError("invalid video runtime payload")

    stdout = io.StringIO()
    stderr = io.StringIO()
    original_argv = sys.argv
    original_cwd = os.getcwd()
    original_env = dict(os.environ)
    original_main_module = sys.modules.get("__main__")
    finder = _PinnedSourceFinder(source_bundle)
    context_module = types.ModuleType(_RUNTIME_CONTEXT_MODULE)
    context_values = {
        str(key): str(value)
        for key, value in context.items()
        if value is not None
    }
    secret_values = {
        str(key): str(value)
        for key, value in secrets.items()
        if value is not None
    }
    context_values.update(secret_values)

    def runtime_value(name: str, default: str = "") -> str:
        return context_values.get(str(name), default)

    context_module.get = runtime_value  # type: ignore[attr-defined]
    module_names = [name for name in source_bundle if name != "__main__"]
    returncode = 0
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        sys.modules[_RUNTIME_CONTEXT_MODULE] = context_module
        sys.meta_path.insert(0, finder)
        os.environ.clear()
        os.environ.update({str(key): str(value) for key, value in env.items()})
        os.environ.pop("PYTHONPATH", None)
        if cwd and os.path.isdir(cwd):
            os.chdir(cwd)
        sys.argv = [script, *[str(item) for item in argv[1:]]]
        entry = source_bundle["__main__"]
        entry_module = types.ModuleType("__main__")
        entry_module.__file__ = entry["path"]
        entry_module.__package__ = None
        entry_module.__cached__ = None
        sys.modules["__main__"] = entry_module
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(
                    compile(entry["source"], entry["path"], "exec"),
                    entry_module.__dict__,
                )
            except SystemExit as exc:
                returncode = _exit_code(exc.code)
            except BaseException:  # noqa: BLE001 - serialize trusted worker failure
                traceback.print_exc()
                returncode = 1
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        sys.modules.pop(_RUNTIME_CONTEXT_MODULE, None)
        for name in module_names:
            sys.modules.pop(name, None)
        if original_main_module is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = original_main_module
        sys.argv = original_argv
        try:
            os.chdir(original_cwd)
        except OSError:
            pass
        os.environ.clear()
        os.environ.update(original_env)
        context_values.clear()
        secret_values.clear()

    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "returncode": returncode,
    }


def _max_rss_bytes() -> int:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, ValueError):
        return 0


def _scrub_request(payload: dict[str, typing.Any]) -> None:
    context = payload.get("context")
    if isinstance(context, dict):
        context.clear()
    secrets = payload.get("secrets")
    if isinstance(secrets, dict):
        secrets.clear()
    payload.clear()


@dataclasses.dataclass(frozen=True)
class _LinuxProcessRecord:
    pid: int
    ppid: int
    start_time: int
    rss_bytes: int


def _read_linux_process_record(
    pid: int,
    *,
    page_size: int,
) -> _LinuxProcessRecord:
    raw = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        raise ValueError("invalid /proc stat comm field")
    fields = raw[closing_paren + 2 :].split()
    if len(fields) < 22:
        raise ValueError("incomplete /proc stat record")
    return _LinuxProcessRecord(
        pid=pid,
        ppid=int(fields[1]),
        start_time=int(fields[19]),
        rss_bytes=max(int(fields[21]), 0) * page_size,
    )


def _read_linux_process_table() -> tuple[dict[int, _LinuxProcessRecord], bool]:
    """Read a bounded process table; ambiguity must prevent a reap ack."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        entries = os.scandir("/proc")
    except (OSError, ValueError):
        return {}, False
    table: dict[int, _LinuxProcessRecord] = {}
    complete = True
    scanned = 0
    with entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            scanned += 1
            if scanned > _PROC_TABLE_SCAN_LIMIT:
                complete = False
                break
            pid = int(entry.name)
            try:
                table[pid] = _read_linux_process_record(
                    pid,
                    page_size=page_size,
                )
            except OSError as exc:
                if exc.errno not in {errno.ENOENT, errno.ESRCH}:
                    complete = False
            except (IndexError, ValueError):
                complete = False
    return table, complete


def _owned_process_snapshot(
    owner_pid: int,
) -> tuple[dict[int, _LinuxProcessRecord], bool]:
    table, complete = _read_linux_process_table()
    if owner_pid not in table:
        complete = False
    owned_ids = {
        pid
        for pid, record in table.items()
        if pid != owner_pid and record.ppid == owner_pid
    }
    changed = True
    while changed:
        changed = False
        for pid, record in table.items():
            if pid not in owned_ids and record.ppid in owned_ids:
                owned_ids.add(pid)
                changed = True
    if len(owned_ids) > _OWNED_PROCESS_CLEANUP_LIMIT:
        complete = False
    return {pid: table[pid] for pid in owned_ids}, complete


def _signal_linux_process(record: _LinuxProcessRecord, signum: int) -> bool:
    """Signal an identity-stable PID, preferring pidfds over reusable PIDs."""
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is not None and pidfd_send_signal is not None:
        try:
            descriptor = pidfd_open(record.pid, 0)
        except ProcessLookupError:
            return True
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOSYS}:
                return False
        else:
            try:
                page_size = int(os.sysconf("SC_PAGE_SIZE"))
                current = _read_linux_process_record(
                    record.pid,
                    page_size=page_size,
                )
                if current.start_time != record.start_time:
                    return False
                pidfd_send_signal(descriptor, signum, None, 0)
                return True
            except ProcessLookupError:
                return True
            except (OSError, ValueError):
                return False
            finally:
                os.close(descriptor)

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        current = _read_linux_process_record(record.pid, page_size=page_size)
        if current.start_time != record.start_time:
            return False
        os.kill(record.pid, signum)
        return True
    except ProcessLookupError:
        return True
    except (OSError, ValueError):
        return False


def _reap_direct_children_nonblocking() -> tuple[bool, bool]:
    """Return (children_remain, unambiguous) for this subreaper."""
    while True:
        try:
            waited_pid, _status = os.waitpid(-1, os.WNOHANG)
        except InterruptedError:
            continue
        except ChildProcessError:
            return False, True
        except OSError:
            return True, False
        if waited_pid == 0:
            return True, True


def _signal_linux_processes(
    records: typing.Iterable[_LinuxProcessRecord],
    signum: int,
) -> bool:
    complete = True
    for record in records:
        if not _signal_linux_process(record, signum):
            complete = False
    return complete


def _stop_owned_processes_at_fixed_point(
    owner_pid: int,
    leader_pid: int | None,
    *,
    deadline: float,
) -> tuple[dict[int, _LinuxProcessRecord], bool]:
    previous: set[tuple[int, int]] | None = None
    unambiguous = True
    latest: dict[int, _LinuxProcessRecord] = {}
    while time.monotonic() < deadline:
        latest, complete = _owned_process_snapshot(owner_pid)
        unambiguous = unambiguous and complete
        signaled = _signal_linux_processes(
            latest.values(),
            signal.SIGSTOP,
        )
        unambiguous = unambiguous and signaled
        identities = {
            (record.pid, record.start_time)
            for record in latest.values()
        }
        if previous == identities and complete and signaled:
            return latest, unambiguous
        previous = identities
        time.sleep(0.01)
    return latest, False


def _cleanup_linux_owned_processes(
    owner_pid: int,
    leader_pid: int | None,
) -> bool:
    """Stop, kill, adopt, and reap one serialized call's entire descendant tree."""
    deadline = time.monotonic() + _PROCESS_TREE_CLEANUP_SECONDS
    snapshot, unambiguous = _stop_owned_processes_at_fixed_point(
        owner_pid,
        leader_pid,
        deadline=deadline,
    )
    ordered = sorted(
        snapshot.values(),
        key=lambda record: (record.pid == leader_pid, record.pid),
    )
    unambiguous = _signal_linux_processes(
        ordered,
        signal.SIGKILL,
    ) and unambiguous

    empty_confirmations = 0
    while time.monotonic() < deadline:
        children_remain, reap_complete = _reap_direct_children_nonblocking()
        current, scan_complete = _owned_process_snapshot(owner_pid)
        unambiguous = unambiguous and reap_complete and scan_complete
        if current:
            empty_confirmations = 0
            if not _signal_linux_processes(current.values(), signal.SIGKILL):
                unambiguous = False
        elif not children_remain:
            empty_confirmations += 1
            if empty_confirmations >= 2:
                return unambiguous
        else:
            empty_confirmations = 0
        time.sleep(0.01)
    return False


def _legacy_kill_and_reap_executor(pid: int) -> bool:
    """Best available non-Linux cleanup for the trusted one-shot leader."""
    sigkill = getattr(signal, "SIGKILL", 9)
    try:
        os.killpg(pid, sigkill)
    except (PermissionError, ProcessLookupError):
        # The child may not have completed setsid() yet. Killing the PID closes
        # that race; descendants are still caught by killpg once the session
        # exists. Darwin can also return EPERM for a group whose only remaining
        # member is our already-dead zombie leader; waitpid below still reaps it.
        pass
    try:
        os.kill(pid, sigkill)
    except (PermissionError, ProcessLookupError):
        pass
    while True:
        try:
            _, _status = os.waitpid(pid, 0)
            return True
        except InterruptedError:
            continue
        except ChildProcessError:
            return True


def _kill_and_reap_executor(pid: int) -> bool:
    if sys.platform.startswith("linux"):
        return _cleanup_linux_owned_processes(os.getpid(), pid)
    return _legacy_kill_and_reap_executor(pid)


def _reap_exited_broker_leader(pid: int) -> bool:
    """Detect and reap a dead leader even when descendants retain its socket."""
    while True:
        try:
            waited_pid, _status = os.waitpid(pid, os.WNOHANG)
            return waited_pid == pid
        except InterruptedError:
            continue
        except ChildProcessError:
            return True


def _owned_process_usage() -> tuple[int, int]:
    """Return process-count and aggregate RSS for this seed-owned call tree."""
    if not sys.platform.startswith("linux"):
        return 0, 0
    owned, complete = _owned_process_snapshot(os.getpid())
    if not complete:
        return _SUBTREE_PROCESS_LIMIT + 1, _SUBTREE_RSS_LIMIT_BYTES + 1
    return len(owned), sum(record.rss_bytes for record in owned.values())


def _owned_processes_within_limits() -> bool:
    process_count, rss_bytes = _owned_process_usage()
    return (
        process_count <= _SUBTREE_PROCESS_LIMIT
        and rss_bytes <= _SUBTREE_RSS_LIMIT_BYTES
    )


def _execute_one_shot(
    channel: socket.socket,
    request: dict[str, typing.Any],
    *,
    memory_limit: dict[str, int | bool],
) -> None:
    """Execute exactly one secret-bearing request in this disposable process."""
    request.pop("executor_timeout_seconds", None)
    try:
        response = _execute(request)
        response["worker"] = {
            "one_shot": True,
            "pid": os.getpid(),
            "call_index": 1,
            "max_rss_bytes": _max_rss_bytes(),
            **memory_limit,
        }
    except BaseException as exc:  # noqa: BLE001 - serialize executor failure
        response = {
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "returncode": 1,
            "worker": {
                "one_shot": True,
                "pid": os.getpid(),
                "call_index": 1,
                "max_rss_bytes": _max_rss_bytes(),
                **memory_limit,
            },
        }
    finally:
        _scrub_request(request)
    _send_frame(channel, response)


def _send_seed_broker_response(
    channel: socket.socket,
    payload: dict[str, typing.Any],
    *,
    broker_fd: int | None,
) -> None:
    marker = b"F" if broker_fd is not None else b"E"
    ancillary = []
    if broker_fd is not None:
        descriptors = array.array("i", [broker_fd])
        ancillary = [
            (socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors.tobytes())
        ]
    if channel.sendmsg([marker], ancillary) != 1:
        raise OSError("trusted worker seed fd transfer failed")
    _send_frame(channel, payload)


def _one_shot_parent_exit_handler(_signum, _frame) -> None:  # noqa: ANN001
    """On seed loss, synchronously reap this broker's adopted descendant tree."""
    sigkill = getattr(signal, "SIGKILL", 9)
    if sys.platform.startswith("linux"):
        cleaned = _cleanup_linux_owned_processes(os.getpid(), None)
        if not cleaned and os.getpgrp() == os.getpid():
            try:
                os.killpg(os.getpid(), sigkill)
            except OSError:
                pass
        os._exit(1)
    try:
        if os.getpgrp() == os.getpid():
            os.killpg(os.getpid(), sigkill)
        else:
            os.kill(os.getpid(), sigkill)
    except OSError:
        os._exit(1)


def _broker_loop(
    channel: socket.socket,
    *,
    memory_limit: dict[str, int | bool],
) -> int:
    _send_frame(channel, {"broker_ready": True, "broker_pid": os.getpid()})
    try:
        request = _recv_frame(channel)
    except EOFError:
        return 0
    if request.get("operation") != "run":
        _scrub_request(request)
        _send_frame(channel, {"error": "invalid operation"})
        return 2
    _execute_one_shot(
        channel,
        request,
        memory_limit=memory_limit,
    )
    return 0


def _spawn_broker_from_seed(
    control_channel: socket.socket,
    *,
    memory_limit: dict[str, int | bool],
) -> int:
    gateway_channel, broker_channel = socket.socketpair()
    seed_pid = os.getpid()
    try:
        broker_pid = os.fork()
    except OSError as exc:
        gateway_channel.close()
        broker_channel.close()
        _send_seed_broker_response(
            control_channel,
            {"broker_ready": False, "error": f"{type(exc).__name__}: {exc}"},
            broker_fd=None,
        )
        return 0
    if broker_pid == 0:
        gateway_channel.close()
        try:
            signal.signal(signal.SIGTERM, _one_shot_parent_exit_handler)
            if not bind_process_to_parent(seed_pid, death_signal=signal.SIGTERM):
                raise PermissionError("broker parent-death boundary is unavailable")
            control_channel.close()
            os.setsid()
            if not harden_sensitive_process(no_new_privs=True, drop_ptrace=True):
                raise PermissionError("broker process memory boundary is unavailable")
            if not enable_child_subreaper():
                raise PermissionError("broker child-subreaper boundary is unavailable")
            returncode = _broker_loop(
                broker_channel,
                memory_limit=memory_limit,
            )
        except BaseException:  # noqa: BLE001 - child must fail closed
            returncode = 1
        finally:
            broker_channel.close()
        os._exit(returncode)

    broker_channel.close()
    try:
        gateway_channel.settimeout(5)
        ready = _recv_frame(gateway_channel)
        if (
            ready.get("broker_ready") is not True
            or ready.get("broker_pid") != broker_pid
        ):
            raise RuntimeError("trusted video-edit broker failed to initialize")
        gateway_channel.settimeout(None)
        _send_seed_broker_response(
            control_channel,
            {"broker_ready": True, "broker_pid": broker_pid},
            broker_fd=gateway_channel.fileno(),
        )
        return broker_pid
    except Exception as exc:
        _kill_and_reap_executor(broker_pid)
        _send_seed_broker_response(
            control_channel,
            {"broker_ready": False, "error": f"{type(exc).__name__}: {exc}"},
            broker_fd=None,
        )
        return 0
    finally:
        gateway_channel.close()


def main() -> int:
    if len(sys.argv) != 2:
        return 2
    try:
        fd = int(sys.argv[1])
    except ValueError:
        return 2

    if not harden_sensitive_process(no_new_privs=True, drop_ptrace=True):
        return 1
    if not enable_child_subreaper():
        return 1
    memory_limit = apply_worker_memory_limit(_MEMORY_LIMIT_BYTES)
    if sys.platform.startswith("linux") and memory_limit.get("applied") is not True:
        return 1
    _preload_optional_runtime_modules()
    # The seed is the post-terminal root of trust. It never receives execution
    # payloads or secrets; it only forks brokers from this already-loaded memory.
    # After readiness, neither seed nor broker can import from the filesystem.
    sys.path[:] = []
    sys.path_importer_cache.clear()
    sys.meta_path[:] = [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
    ]

    channel = socket.socket(fileno=fd)
    broker_pid = 0
    last_reaped_pid = 0
    try:
        _send_frame(channel, {
            "ready": True,
            "dumpable": 0 if sys.platform.startswith("linux") else None,
            "memory_limit": memory_limit,
            "fork_seed": True,
            "accepts_secrets": False,
            "child_subreaper": True,
        })
        while True:
            readable, _, _ = select.select(
                [channel],
                [],
                [],
                0.05 if broker_pid else None,
            )
            if not readable:
                if broker_pid and _reap_exited_broker_leader(broker_pid):
                    if (
                        not sys.platform.startswith("linux")
                        or _cleanup_linux_owned_processes(os.getpid(), None)
                    ):
                        last_reaped_pid = broker_pid
                        broker_pid = 0
                elif broker_pid and not _owned_processes_within_limits():
                    if _kill_and_reap_executor(broker_pid):
                        last_reaped_pid = broker_pid
                        broker_pid = 0
                continue
            operation = channel.recv(1)
            if not operation:
                return 0
            if operation == b"S":
                if broker_pid:
                    _send_seed_broker_response(
                        channel,
                        {
                            "broker_ready": False,
                            "error": "trusted worker seed already owns an active call",
                        },
                        broker_fd=None,
                    )
                    continue
                if sys.platform.startswith("linux") and not (
                    _cleanup_linux_owned_processes(os.getpid(), None)
                ):
                    _send_seed_broker_response(
                        channel,
                        {
                            "broker_ready": False,
                            "error": "trusted worker seed child invariant failed",
                        },
                        broker_fd=None,
                    )
                    continue
                broker_pid = _spawn_broker_from_seed(
                    channel,
                    memory_limit=memory_limit,
                )
                continue
            if operation == b"T":
                requested_pid = struct.unpack("!Q", _recv_exact(channel, 8))[0]
                cleanup = "unknown"
                if broker_pid and requested_pid == broker_pid:
                    if _kill_and_reap_executor(broker_pid):
                        last_reaped_pid = broker_pid
                        broker_pid = 0
                        cleanup = "stopped"
                elif requested_pid == last_reaped_pid:
                    cleanup = "already_clean"
                _send_frame(channel, {
                    "cleanup": cleanup,
                    "pid": requested_pid,
                    "reaped": cleanup in {"stopped", "already_clean"},
                })
                continue
            if operation == b"Q":
                if broker_pid:
                    if not _kill_and_reap_executor(broker_pid):
                        _send_frame(channel, {"shutdown": False})
                        continue
                    broker_pid = 0
                _send_frame(channel, {"shutdown": True})
                return 0
            return 2
    finally:
        if broker_pid:
            _kill_and_reap_executor(broker_pid)
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
