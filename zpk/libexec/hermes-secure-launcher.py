#!/usr/bin/env python3
"""Enter the packaged Hermes CLI with a non-dumpable Linux process."""

from __future__ import annotations

import ctypes
import os
import runpy
import stat
import sys
from pathlib import Path, PurePosixPath


_PR_SET_DUMPABLE = 4
_PR_SET_NO_NEW_PRIVS = 38
_MANAGED_GATEWAY_ENV = "HERMES_MANAGED_GATEWAY"
_MANAGED_CGROUP_ROOT_ENV = "HERMES_MANAGED_CGROUP_ROOT"
_MANAGED_CGROUP_UNIT_ENV = "HERMES_MANAGED_CGROUP_UNIT"
_MANAGED_SUPERVISOR_CGROUP = "agentcomputer-supervisor"
_CGROUP2_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
_CGROUP_METADATA_MAX_BYTES = 4096
_MANAGED_SERVICE_LIMITS = {
    "memory.high": "805306368",
    "memory.max": "1073741824",
    "memory.swap.max": "0",
    "pids.max": "512",
}


def _read_bounded_ascii(path: Path, *, limit: int) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"not a regular cgroup control: {path.name}")
        raw = os.read(descriptor, limit + 1)
        if len(raw) > limit:
            raise OSError(f"oversized cgroup control: {path.name}")
        return raw.decode("ascii")
    finally:
        os.close(descriptor)


def _write_control_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"not a regular cgroup control: {path.name}")
        if os.write(descriptor, payload) != len(payload):
            raise OSError(f"short cgroup control write: {path.name}")
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
        or relative == "/"
        or ".." in parsed.parts
        or "\x00" in relative
    ):
        raise OSError("invalid unified cgroup membership")
    return relative.rstrip("/")


def _verify_managed_service_limits(service: Path) -> None:
    for control, expected in _MANAGED_SERVICE_LIMITS.items():
        actual = _read_bounded_ascii(
            service / control,
            limit=_CGROUP_METADATA_MAX_BYTES,
        ).strip()
        if actual != expected:
            raise OSError(
                f"managed gateway service limit mismatch: {control}"
            )


def _prepare_managed_service_cgroup() -> None:
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise OSError("managed gateway requires Linux root cgroup delegation")
    if not (_CGROUP2_ROOT / "cgroup.controllers").is_file():
        raise OSError("managed gateway requires unified cgroup v2")

    expected_unit = os.environ.get(_MANAGED_CGROUP_UNIT_ENV, "")
    if (
        not expected_unit
        or "/" in expected_unit
        or expected_unit in {".", ".."}
        or not expected_unit.endswith(".service")
    ):
        raise OSError("managed gateway service cgroup identity is unavailable")

    current_relative = _current_unified_cgroup()
    current = PurePosixPath(current_relative)
    if current.name == expected_unit:
        service_relative = current_relative
        initial_entry = True
    elif (
        current.name == _MANAGED_SUPERVISOR_CGROUP
        and current.parent.name == expected_unit
        and os.environ.get(_MANAGED_CGROUP_ROOT_ENV) == str(current.parent)
    ):
        service_relative = str(current.parent)
        initial_entry = False
    else:
        raise OSError("managed gateway service cgroup identity is unavailable")

    root = _CGROUP2_ROOT.resolve(strict=True)
    service = (root / service_relative.lstrip("/")).resolve(strict=True)
    try:
        service.relative_to(root)
    except ValueError as exc:
        raise OSError("managed gateway service cgroup escapes cgroup v2") from exc
    if (
        service.name != expected_unit
        or not (service / "cgroup.procs").is_file()
        or not (service / "cgroup.kill").is_file()
        or not (service / "cgroup.subtree_control").is_file()
    ):
        raise OSError("managed gateway service cgroup is invalid")
    _verify_managed_service_limits(service)

    controllers = set(
        _read_bounded_ascii(
            service / "cgroup.controllers",
            limit=_CGROUP_METADATA_MAX_BYTES,
        ).split()
    )
    if not {"memory", "pids"}.issubset(controllers):
        raise OSError("managed gateway requires memory and pids delegation")
    if not (service / "memory.swap.max").is_file():
        raise OSError("managed gateway requires memory.swap.max accounting")

    supervisor = service / _MANAGED_SUPERVISOR_CGROUP
    if not initial_entry:
        if (
            stat.S_ISLNK(supervisor.lstat().st_mode)
            or not supervisor.is_dir()
            or not (supervisor / "cgroup.procs").is_file()
            or not (supervisor / "cgroup.events").is_file()
            or _read_bounded_ascii(
                service / "cgroup.procs",
                limit=_CGROUP_METADATA_MAX_BYTES,
            ).strip()
        ):
            raise OSError("managed gateway supervisor cgroup is invalid")
        enabled = set(
            _read_bounded_ascii(
                service / "cgroup.subtree_control",
                limit=_CGROUP_METADATA_MAX_BYTES,
            ).split()
        )
        if not {"memory", "pids"}.issubset(enabled):
            raise OSError("managed gateway controllers are not enabled")
        for control in ("memory.max", "memory.swap.max", "pids.max"):
            if not (supervisor / control).is_file():
                raise OSError(f"managed gateway supervisor lacks {control}")
        os.environ[_MANAGED_CGROUP_ROOT_ENV] = service_relative
        return

    created = False
    try:
        os.mkdir(supervisor, 0o755)
        created = True
    except FileExistsError:
        pass
    try:
        if (
            stat.S_ISLNK(supervisor.lstat().st_mode)
            or not supervisor.is_dir()
            or not (supervisor / "cgroup.procs").is_file()
            or not (supervisor / "cgroup.events").is_file()
        ):
            raise OSError("managed gateway supervisor cgroup is invalid")
        events = dict(
            line.split(maxsplit=1)
            for line in _read_bounded_ascii(
                supervisor / "cgroup.events",
                limit=_CGROUP_METADATA_MAX_BYTES,
            ).splitlines()
            if len(line.split(maxsplit=1)) == 2
        )
        if events.get("populated") != "0":
            raise OSError("managed gateway supervisor cgroup is already populated")

        _write_control_file(
            supervisor / "cgroup.procs",
            str(os.getpid()).encode("ascii"),
        )
        supervisor_relative = (
            f"{service_relative}/{_MANAGED_SUPERVISOR_CGROUP}"
        )
        if _current_unified_cgroup() != supervisor_relative:
            raise OSError("managed gateway supervisor attach verification failed")
        if _read_bounded_ascii(
            service / "cgroup.procs",
            limit=_CGROUP_METADATA_MAX_BYTES,
        ).strip():
            raise OSError("managed gateway delegation root is not process-free")

        _write_control_file(
            service / "cgroup.subtree_control",
            b"+memory +pids",
        )
        enabled = set(
            _read_bounded_ascii(
                service / "cgroup.subtree_control",
                limit=_CGROUP_METADATA_MAX_BYTES,
            ).split()
        )
        if not {"memory", "pids"}.issubset(enabled):
            raise OSError("managed gateway controller enablement was rejected")
        for control in ("memory.max", "memory.swap.max", "pids.max"):
            if not (supervisor / control).is_file():
                raise OSError(f"managed gateway supervisor lacks {control}")
    except Exception:
        if created:
            try:
                os.rmdir(supervisor)
            except OSError:
                pass
        raise

    os.environ[_MANAGED_CGROUP_ROOT_ENV] = service_relative


def _harden_linux_process(*, managed_gateway: bool | None = None) -> None:
    if not sys.platform.startswith("linux"):
        return

    libc = ctypes.CDLL(None, use_errno=True)
    hardening = [(_PR_SET_DUMPABLE, 0)]
    if managed_gateway is None:
        managed_gateway = os.environ.get(_MANAGED_GATEWAY_ENV) == "1"
    if managed_gateway:
        hardening.append((_PR_SET_NO_NEW_PRIVS, 1))
    for option, value in hardening:
        if libc.prctl(option, value, 0, 0, 0) == 0:
            continue
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main() -> int:
    if len(sys.argv) < 2:
        print("missing packaged Hermes entry point", file=sys.stderr)
        return 127

    try:
        entry_point = Path(sys.argv[1]).resolve(strict=True)
    except OSError:
        print("packaged Hermes entry point is unavailable", file=sys.stderr)
        return 127
    if not entry_point.is_file():
        print("packaged Hermes entry point is not a file", file=sys.stderr)
        return 127

    managed_gateway = os.environ.get(_MANAGED_GATEWAY_ENV) == "1"
    if managed_gateway:
        try:
            _prepare_managed_service_cgroup()
        except OSError as exc:
            print(f"managed gateway cgroup setup failed: {exc}", file=sys.stderr)
            return 125
    _harden_linux_process(managed_gateway=managed_gateway)
    sys.argv = [str(entry_point), *sys.argv[2:]]
    runpy.run_path(str(entry_point), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
