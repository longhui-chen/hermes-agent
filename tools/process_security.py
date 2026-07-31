"""Linux process boundaries for model-controlled and secret-bearing children."""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import sys
from typing import Any


_IS_LINUX = sys.platform.startswith("linux")

_PR_SET_DUMPABLE = 4
_PR_GET_DUMPABLE = 3
_PR_SET_PDEATHSIG = 1
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_PR_SET_NO_NEW_PRIVS = 38
_PR_GET_NO_NEW_PRIVS = 39
_PR_CAPBSET_READ = 23
_PR_CAPBSET_DROP = 24
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4
_PR_SET_PTRACER = 0x59616D61
_CAP_SYS_PTRACE = 19
_CAP_SYS_RESOURCE = 24
_LINUX_CAPABILITY_VERSION_3 = 0x20080522


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _prctl(option: int, arg2: int = 0) -> int:
    libc = _libc()
    result = libc.prctl(
        ctypes.c_int(option),
        ctypes.c_ulong(arg2),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _capability_sets() -> tuple[_CapHeader, Any]:
    libc = _libc()
    header = _CapHeader(version=_LINUX_CAPABILITY_VERSION_3, pid=0)
    data = (_CapData * 2)()
    if libc.capget(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return header, data


def _clear_sensitive_capabilities() -> None:
    header, data = _capability_sets()
    blocked_capabilities = (_CAP_SYS_PTRACE, _CAP_SYS_RESOURCE)
    for capability in blocked_capabilities:
        word = capability // 32
        mask = ~(1 << (capability % 32)) & 0xFFFFFFFF
        data[word].effective &= mask
        data[word].permitted &= mask
        data[word].inheritable &= mask
    libc = _libc()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    # Ambient capabilities survive exec, so clear the whole ambient set.
    try:
        _prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOSYS}:
            raise

    # Root/CAP_SETPCAP processes can remove these from the bounding set. A
    # regular user may not be allowed to mutate CapBnd. Retaining those bits is
    # safe only when no_new_privs provides an irreversible exec boundary; the
    # caller separately verifies an empty bounding set when elevation remains
    # allowed.
    for capability in blocked_capabilities:
        if _prctl(_PR_CAPBSET_READ, capability):
            try:
                _prctl(_PR_CAPBSET_DROP, capability)
            except OSError as exc:
                if exc.errno != errno.EPERM:
                    raise

    _, verified = _capability_sets()
    for capability in blocked_capabilities:
        word = capability // 32
        cap_mask = 1 << (capability % 32)
        if any(
            getattr(verified[word], field) & cap_mask
            for field in ("effective", "permitted", "inheritable")
        ):
            raise PermissionError(
                f"capability {capability} remained in an active capability set"
            )
        if os.geteuid() == 0 and _prctl(_PR_CAPBSET_READ, capability):
            raise PermissionError(
                f"capability {capability} remained in the root bounding set"
            )


def _sensitive_capability_remains_bounded() -> bool:
    return any(
        _prctl(_PR_CAPBSET_READ, capability)
        for capability in (_CAP_SYS_PTRACE, _CAP_SYS_RESOURCE)
    )


def harden_sensitive_process(
    *,
    no_new_privs: bool = False,
    drop_ptrace: bool = False,
) -> bool:
    """Make a secret-bearing process unreadable by same-UID children.

    Non-Linux platforms keep their native process inspection policy.  The
    production device boundary is Linux and fails closed when prctl cannot be
    applied.
    """
    if not _IS_LINUX:
        return True
    try:
        _prctl(_PR_SET_DUMPABLE, 0)
        try:
            _prctl(_PR_SET_PTRACER, 0)
        except OSError as exc:
            # PR_SET_PTRACER is a Yama extension, not a universal prctl. Kernels
            # without Yama may reject it even though the mandatory dumpable,
            # capability, and no_new_privs boundaries remain enforceable.
            if exc.errno not in {errno.EINVAL, errno.ENOSYS}:
                raise
        if no_new_privs:
            _prctl(_PR_SET_NO_NEW_PRIVS, 1)
        if drop_ptrace:
            _clear_sensitive_capabilities()
            if not no_new_privs and _sensitive_capability_remains_bounded():
                raise PermissionError(
                    "sensitive capability remained in the bounding set"
                )
        return (
            _prctl(_PR_GET_DUMPABLE) == 0
            and (
                not no_new_privs
                or _prctl(_PR_GET_NO_NEW_PRIVS) == 1
            )
        )
    except OSError:
        return False


def bind_process_to_parent(
    expected_parent_pid: int,
    *,
    death_signal: int = signal.SIGKILL,
) -> bool:
    """Kill a Linux child if its trusted parent exits, closing the fork race."""
    if not _IS_LINUX:
        return os.getppid() == expected_parent_pid
    try:
        _prctl(_PR_SET_PDEATHSIG, int(death_signal))
        return os.getppid() == expected_parent_pid
    except OSError:
        return False


def _get_child_subreaper() -> int:
    value = ctypes.c_int(0)
    libc = _libc()
    result = libc.prctl(
        ctypes.c_int(_PR_GET_CHILD_SUBREAPER),
        ctypes.byref(value),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
        ctypes.c_ulong(0),
    )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(value.value)


def enable_child_subreaper() -> bool:
    """Keep daemonized descendants attached to this trusted process tree."""
    if not _IS_LINUX:
        return True
    try:
        _prctl(_PR_SET_CHILD_SUBREAPER, 1)
        return _get_child_subreaper() == 1
    except OSError:
        return False


def apply_worker_memory_limit(limit_bytes: int) -> dict[str, int | bool]:
    """Apply a hard address-space cap to the Linux trusted worker."""
    if not _IS_LINUX:
        return {"applied": False, "limit_bytes": 0}
    try:
        import resource

        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        infinity = resource.RLIM_INFINITY
        target = int(limit_bytes)
        if current_hard != infinity:
            target = min(target, int(current_hard))
        resource.setrlimit(resource.RLIMIT_AS, (target, target))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        applied = soft == target and hard == target
        return {"applied": applied, "limit_bytes": target if applied else 0}
    except (ImportError, OSError, ValueError):
        return {"applied": False, "limit_bytes": 0}
