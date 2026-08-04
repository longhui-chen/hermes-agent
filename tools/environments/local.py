"""Local execution environment — spawn-per-call with session snapshot."""

import hashlib
import hmac
import logging
import ntpath
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from tools.environments.base import BaseEnvironment, _pipe_stdin
from hermes_cli._subprocess_compat import windows_hide_flags

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger(__name__)

_MANAGED_GATEWAY_ENV = "HERMES_MANAGED_GATEWAY"
_MANAGED_BOOTSTRAP_ENV_KEYS = frozenset({
    _MANAGED_GATEWAY_ENV,
    "HERMES_MANAGED_CGROUP_UNIT",
    "HERMES_MANAGED_CGROUP_ROOT",
})
_MANAGED_SETPRIV_PATH = "/usr/bin/setpriv"
_MANAGED_UNSHARE_PATH = "/usr/bin/unshare"
_MANAGED_TERMINAL_UID_MIN = 100_000
_MANAGED_TERMINAL_UID_MAX = 2_000_000_000
_MANAGED_TERMINAL_IDENTITY_ATTEMPTS = 64
_MANAGED_TERMINAL_IDENTITY_CACHE_MAX = 4096
_MANAGED_TERMINAL_IDENTITY_LOCK = threading.Lock()
_MANAGED_TERMINAL_SCOPE_BY_UID: dict[int, str] = {}
_MANAGED_SKILL_TREE_MAX_ENTRIES = 20_000
_MANAGED_OUTPUT_TREE_MAX_ENTRIES = 100_000
_MANAGED_TERMINAL_RETIRED_UIDS: set[int] = set()
_MANAGED_TERMINAL_RETIRED_SCOPES: set[str] = set()
_MANAGED_TERMINAL_RETIRED_MAX = 4096
_MANAGED_TERMINAL_HOME_ROOT = Path("/run/zettlab-claw/terminal-homes")
_MANAGED_TERMINAL_CGROUP_PREFIX = "terminal-profile"
_MANAGED_TERMINAL_CGROUP_MEMORY_MAX_BYTES = 256 * 1024 * 1024
_MANAGED_TERMINAL_CGROUP_MEMORY_SWAP_MAX_BYTES = 0
_MANAGED_TERMINAL_CGROUP_PIDS_MAX = 64
_MANAGED_TERMINAL_CGROUP_LOCK = threading.Lock()
_MANAGED_TERMINAL_CGROUP_CLEANUP_TIMEOUT_SECONDS = 2.0
_MANAGED_TERMINAL_CGROUP_POLL_SECONDS = 0.05
_MANAGED_EXECUTE_CODE_CGROUP_LOCK = threading.Lock()
_MANAGED_EXECUTE_CODE_CGROUP_BY_UID: dict[int, object] = {}
_MANAGED_TERMINAL_CGROUP_ENTER = (
    "import os,sys\n"
    "path=os.path.join(sys.argv[1],'cgroup.procs')\n"
    "flags=os.O_WRONLY|os.O_CLOEXEC|getattr(os,'O_NOFOLLOW',0)\n"
    "fd=os.open(path,flags)\n"
    "try:\n os.write(fd,(str(os.getpid())+'\\n').encode('ascii'))\n"
    "finally:\n os.close(fd)\n"
    "os.execv(sys.argv[2],sys.argv[2:])\n"
)
_MANAGED_TERMINAL_PRIVATE_TMP_ENTER = (
    "import ctypes,os,stat,sys\n"
    "if len(sys.argv)<4:\n raise OSError('managed private tmp argv is invalid')\n"
    "sources=sys.argv[1:3]\n"
    "for source in sources:\n"
    " info=os.lstat(source)\n"
    " if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or info.st_mode&0o077:\n"
    "  raise OSError('managed private tmp source is not trusted')\n"
    "for target in ('/tmp','/var/tmp'):\n"
    " info=os.lstat(target)\n"
    " if not stat.S_ISDIR(info.st_mode):\n  raise OSError('managed private tmp target is unavailable')\n"
    "libc=ctypes.CDLL(None,use_errno=True)\n"
    "libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_void_p]\n"
    "libc.mount.restype=ctypes.c_int\n"
    "def mount(source,target,flags):\n"
    " result=libc.mount(source,target,None,flags,None)\n"
    " if result!=0:\n  error=ctypes.get_errno();raise OSError(error,os.strerror(error),os.fsdecode(target))\n"
    "mount(None,b'/',16384|262144)\n"
    "mount(os.fsencode(sources[0]),b'/tmp',4096|16384)\n"
    "mount(os.fsencode(sources[1]),b'/var/tmp',4096|16384)\n"
    "skill_root=os.path.join(os.environ.get('HERMES_HOME',''),'skills')\n"
    "if skill_root and os.path.isdir(skill_root):\n"
    " info=os.lstat(skill_root)\n"
    " if not stat.S_ISDIR(info.st_mode) or info.st_mode&0o022:\n"
    "  raise OSError('managed skill source is not trusted')\n"
    " encoded=os.fsencode(skill_root)\n"
    " mount(encoded,encoded,4096|16384)\n"
    " mount(None,encoded,32|4096|1|2|4)\n"
    "os.umask(0o077)\n"
    "os.execv(sys.argv[3],sys.argv[3:])\n"
)
_MANAGED_EXECUTE_CODE_PRIVATE_TMP_ENTER = (
    "import ctypes,os,stat,sys\n"
    "if len(sys.argv)<4:\n raise OSError('managed execute_code private tmp argv is invalid')\n"
    "workspace,var_tmp=sys.argv[1:3]\n"
    "for source in (workspace,var_tmp):\n"
    " info=os.lstat(source)\n"
    " if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_gid!=0 or info.st_mode&0o077:\n"
    "  raise OSError('managed execute_code private tmp source is not trusted')\n"
    "for target in ('/tmp','/var/tmp'):\n"
    " info=os.lstat(target)\n"
    " if not stat.S_ISDIR(info.st_mode):\n  raise OSError('managed execute_code private tmp target is unavailable')\n"
    "libc=ctypes.CDLL(None,use_errno=True)\n"
    "libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_void_p]\n"
    "libc.mount.restype=ctypes.c_int\n"
    "def mount(source,target,flags):\n"
    " result=libc.mount(source,target,None,flags,None)\n"
    " if result!=0:\n  error=ctypes.get_errno();raise OSError(error,os.strerror(error),os.fsdecode(target))\n"
    "mount(None,b'/',16384|262144)\n"
    "mount(os.fsencode(var_tmp),b'/var/tmp',4096|16384)\n"
    "mount(os.fsencode(workspace),b'/tmp',4096|16384)\n"
    "os.chdir('/tmp')\n"
    "os.umask(0o077)\n"
    "os.execv(sys.argv[3],sys.argv[3:])\n"
)


def _managed_terminal_profile_scope(
    env: Mapping[str, str] | None = None,
) -> str:
    raw_scope = str((env or {}).get("HERMES_HOME") or "").strip()
    if (
        not raw_scope
        or "\x00" in raw_scope
        or len(raw_scope.encode("utf-8")) > 4096
    ):
        raise OSError("managed terminal profile identity is unavailable")
    return str(Path(raw_scope).expanduser().resolve())


def _validate_managed_root_directory_chain(directory: Path) -> Path:
    """Resolve and validate a root-owned, non-writable directory chain."""

    resolved = directory.resolve(strict=True)
    for component in (resolved, *resolved.parents):
        info = os.lstat(component)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_mode & 0o022
        ):
            raise OSError("managed profile directory chain is not trusted")
    return resolved


def _managed_terminal_identity(
    env: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Derive one device-independent non-root identity per multiplex profile."""

    if _IS_WINDOWS or os.geteuid() != 0:
        raise OSError("managed terminal requires a root identity broker")
    secret = os.environ.get("ZET_AGENT_KEY", "")
    scope = _managed_terminal_profile_scope(env)
    profile_id = Path(scope).name
    if (
        not secret
        or "\x00" in secret
        or len(secret.encode("utf-8")) > 4096
        or not profile_id
        or profile_id in (".", "..")
    ):
        raise OSError("managed terminal profile identity is unavailable")

    import pwd

    registered = {entry.pw_uid for entry in pwd.getpwall()}
    population = _MANAGED_TERMINAL_UID_MAX - _MANAGED_TERMINAL_UID_MIN + 1
    with _MANAGED_TERMINAL_IDENTITY_LOCK:
        existing = [
            uid
            for uid, owner_scope in _MANAGED_TERMINAL_SCOPE_BY_UID.items()
            if owner_scope == scope
        ]
        if len(existing) == 1:
            return existing[0], existing[0]
        if len(existing) > 1:
            raise OSError("managed terminal profile has ambiguous identities")
        for counter in range(_MANAGED_TERMINAL_IDENTITY_ATTEMPTS):
            digest = hashlib.sha256(
                f"zettlab-managed-profile-v1\0{profile_id}\0{counter}".encode(
                    "utf-8"
                )
            ).digest()
            uid = _MANAGED_TERMINAL_UID_MIN + (
                int.from_bytes(digest[:8], "big") % population
            )
            owner = _MANAGED_TERMINAL_SCOPE_BY_UID.get(uid)
            if (
                uid in registered
                or uid in _MANAGED_TERMINAL_RETIRED_UIDS
                or (owner is not None and owner != scope)
            ):
                continue
            if owner is None:
                if (
                    len(_MANAGED_TERMINAL_SCOPE_BY_UID)
                    >= _MANAGED_TERMINAL_IDENTITY_CACHE_MAX
                ):
                    raise OSError("managed terminal identity cache is full")
                _MANAGED_TERMINAL_SCOPE_BY_UID[uid] = scope
                _MANAGED_TERMINAL_RETIRED_SCOPES.discard(scope)
            return uid, uid
    raise OSError("managed terminal profile identity collision")


def _managed_terminal_privilege_drop_prefix(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return the fixed fail-closed capability drop for model shell commands."""

    try:
        info = os.lstat(_MANAGED_SETPRIV_PATH)
    except OSError as exc:
        raise OSError("managed terminal privilege drop is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
    ):
        raise OSError("managed terminal privilege drop is not trusted")
    uid, gid = _managed_terminal_identity(env)
    return [
        _MANAGED_SETPRIV_PATH,
        f"--reuid={uid}",
        f"--regid={gid}",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
    ]


def _managed_execute_code_identity(
    env: Mapping[str, str],
    execution_scope: str,
) -> tuple[int, int]:
    """Reserve a per-execution UID distinct from every persistent terminal."""

    if _IS_WINDOWS or os.geteuid() != 0:
        raise OSError("managed execute_code requires a root identity broker")
    secret = os.environ.get("ZET_AGENT_KEY", "")
    profile_scope = _managed_terminal_profile_scope(env)
    if (
        not secret
        or not execution_scope
        or "\x00" in secret
        or "\x00" in execution_scope
        or len(secret.encode("utf-8")) > 4096
        or len(execution_scope.encode("utf-8")) > 128
    ):
        raise OSError("managed execute_code identity is unavailable")

    import pwd

    owner_scope = f"execute-code\0{profile_scope}\0{execution_scope}"
    registered = {entry.pw_uid for entry in pwd.getpwall()}
    population = _MANAGED_TERMINAL_UID_MAX - _MANAGED_TERMINAL_UID_MIN + 1
    with _MANAGED_TERMINAL_IDENTITY_LOCK:
        for counter in range(_MANAGED_TERMINAL_IDENTITY_ATTEMPTS):
            digest = hmac.new(
                secret.encode("utf-8"),
                f"{owner_scope}\0{counter}".encode("utf-8"),
                hashlib.sha256,
            ).digest()
            uid = _MANAGED_TERMINAL_UID_MIN + (
                int.from_bytes(digest[:8], "big") % population
            )
            owner = _MANAGED_TERMINAL_SCOPE_BY_UID.get(uid)
            if (
                uid in registered
                or uid in _MANAGED_TERMINAL_RETIRED_UIDS
                or (owner is not None and owner != owner_scope)
            ):
                continue
            if owner is None:
                if (
                    len(_MANAGED_TERMINAL_SCOPE_BY_UID)
                    >= _MANAGED_TERMINAL_IDENTITY_CACHE_MAX
                ):
                    raise OSError("managed identity cache is full")
                _MANAGED_TERMINAL_SCOPE_BY_UID[uid] = owner_scope
            return uid, uid
    raise OSError("managed execute_code identity collision")


def _release_managed_execute_code_identity(
    uid: int,
    env: Mapping[str, str],
    execution_scope: str,
) -> None:
    """Release an invocation UID after its process tree and RPC socket are gone."""

    owner_scope = (
        f"execute-code\0{_managed_terminal_profile_scope(env)}\0"
        f"{execution_scope}"
    )
    with _MANAGED_TERMINAL_IDENTITY_LOCK:
        if _MANAGED_TERMINAL_SCOPE_BY_UID.get(uid) == owner_scope:
            _MANAGED_TERMINAL_SCOPE_BY_UID.pop(uid, None)


def _managed_execute_code_sandbox_argv(
    argv: list[str],
    *,
    env: Mapping[str, str],
    execution_scope: str | None,
    workspace: str | None = None,
) -> list[str]:
    """Drop one execute_code invocation into its non-shared identity domain."""

    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return list(argv)
    if execution_scope is None:
        raise OSError("managed execute_code scope is unavailable")
    if workspace is None:
        raise OSError("managed execute_code workspace is unavailable")
    try:
        info = os.lstat(_MANAGED_SETPRIV_PATH)
    except OSError as exc:
        raise OSError("managed execute_code privilege drop is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
    ):
        raise OSError("managed execute_code privilege drop is not trusted")
    uid, gid = _managed_execute_code_identity(env, execution_scope)
    launcher = _trusted_managed_python()
    namespace_launcher = _trusted_managed_unshare()
    private_tmp, private_var_tmp = _managed_execute_code_private_tmp_paths(
        workspace, uid
    )
    from tools.trusted_direct_runner import (
        _create_managed_invocation_cgroup,
        _kill_and_remove_managed_cgroup,
    )

    cgroup = _create_managed_invocation_cgroup()
    try:
        with _MANAGED_EXECUTE_CODE_CGROUP_LOCK:
            if uid in _MANAGED_EXECUTE_CODE_CGROUP_BY_UID:
                raise OSError("managed execute_code cgroup identity collision")
            _MANAGED_EXECUTE_CODE_CGROUP_BY_UID[uid] = cgroup
    except Exception:
        _kill_and_remove_managed_cgroup(cgroup, None)
        raise
    return [
        launcher,
        "-I",
        "-c",
        _MANAGED_TERMINAL_CGROUP_ENTER,
        str(cgroup.path),
        _MANAGED_SETPRIV_PATH,
        f"--reuid={uid}",
        f"--regid={gid}",
        "--clear-groups",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
        namespace_launcher,
        "--user",
        "--map-root-user",
        "--mount",
        "--",
        launcher,
        "-I",
        "-c",
        _MANAGED_EXECUTE_CODE_PRIVATE_TMP_ENTER,
        str(private_tmp),
        str(private_var_tmp),
        *argv,
    ]


def _managed_terminal_argv(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Apply the managed capability boundary to every local terminal path."""

    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return list(argv)
    cgroup = _ensure_managed_terminal_cgroup(env)
    _home, private_tmp, private_var_tmp = _managed_terminal_home_paths(env)
    launcher = _trusted_managed_python()
    return [
        launcher,
        "-I",
        "-c",
        _MANAGED_TERMINAL_CGROUP_ENTER,
        str(cgroup),
        *_managed_terminal_privilege_drop_prefix(env),
        _trusted_managed_unshare(),
        "--user",
        "--map-root-user",
        "--mount",
        "--fork",
        "--kill-child=KILL",
        "--",
        launcher,
        "-I",
        "-c",
        _MANAGED_TERMINAL_PRIVATE_TMP_ENTER,
        str(private_tmp),
        str(private_var_tmp),
        *list(argv),
    ]


def _trusted_managed_python() -> str:
    """Return the immutable interpreter used by the root cgroup trampoline."""

    try:
        interpreter = Path(sys.executable).resolve(strict=True)
        info = interpreter.stat()
    except OSError as exc:
        raise OSError("managed terminal cgroup launcher is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
        or not info.st_mode & 0o111
    ):
        raise OSError("managed terminal cgroup launcher is not trusted")
    return str(interpreter)


def _trusted_managed_unshare() -> str:
    """Return the fixed root-owned user/mount namespace launcher."""

    try:
        info = os.lstat(_MANAGED_UNSHARE_PATH)
    except OSError as exc:
        raise OSError("managed terminal namespace launcher is unavailable") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
        or not info.st_mode & 0o111
    ):
        raise OSError("managed terminal namespace launcher is not trusted")
    return _MANAGED_UNSHARE_PATH


def _managed_terminal_cgroup_for_uid(
    uid: int,
    *,
    create: bool,
):
    """Resolve one root-owned delegated cgroup and validate all controls."""

    if _IS_WINDOWS or os.geteuid() != 0:
        raise OSError("managed terminal requires Linux root delegation")
    if not (_MANAGED_TERMINAL_UID_MIN <= uid <= _MANAGED_TERMINAL_UID_MAX):
        raise OSError("managed terminal cgroup identity is invalid")
    from tools import trusted_direct_runner as runner

    delegation_root, _relative, _identity = (
        runner._resolve_managed_delegation_root()
    )
    cgroup = delegation_root / f"{_MANAGED_TERMINAL_CGROUP_PREFIX}-{uid}"
    created = False
    if create:
        try:
            os.mkdir(cgroup, 0o755)
            created = True
        except FileExistsError:
            pass
    else:
        try:
            cgroup.lstat()
        except FileNotFoundError:
            return None, runner
    try:
        info = cgroup.lstat()
        resolved = cgroup.resolve(strict=True)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or resolved.parent != delegation_root
            or resolved.name != f"{_MANAGED_TERMINAL_CGROUP_PREFIX}-{uid}"
        ):
            raise OSError("managed terminal cgroup is not trusted")
        controls = (
            "cgroup.procs",
            "cgroup.kill",
            "cgroup.events",
            "memory.max",
            "memory.swap.max",
            "memory.oom.group",
            "pids.max",
        )
        for control in controls:
            if not (resolved / control).is_file():
                raise OSError(f"managed terminal cgroup lacks {control}")
        return resolved, runner
    except Exception:
        if created:
            try:
                os.rmdir(cgroup)
            except OSError:
                pass
        raise


def _ensure_managed_terminal_cgroup(
    env: Mapping[str, str] | None = None,
) -> Path:
    """Create/configure the bounded cgroup shared by one profile terminal."""

    uid, _gid = _managed_terminal_identity(env)
    expected = {
        "memory.max": str(_MANAGED_TERMINAL_CGROUP_MEMORY_MAX_BYTES),
        "memory.swap.max": str(_MANAGED_TERMINAL_CGROUP_MEMORY_SWAP_MAX_BYTES),
        "memory.oom.group": "1",
        "pids.max": str(_MANAGED_TERMINAL_CGROUP_PIDS_MAX),
    }
    with _MANAGED_TERMINAL_CGROUP_LOCK:
        cgroup, runner = _managed_terminal_cgroup_for_uid(uid, create=True)
        assert cgroup is not None
        for control, value in expected.items():
            runner._write_control_file(
                cgroup / control,
                value.encode("ascii"),
            )
        for control, value in expected.items():
            actual = runner._read_bounded_ascii(
                cgroup / control,
                limit=4096,
            ).strip()
            if actual != value:
                raise OSError(f"managed terminal cgroup rejected {control}")
        return cgroup


def _remove_managed_terminal_cgroup(uid: int) -> bool:
    """Kill every descendant and remove one profile's delegated cgroup."""

    with _MANAGED_TERMINAL_CGROUP_LOCK:
        cgroup, runner = _managed_terminal_cgroup_for_uid(uid, create=False)
        if cgroup is None:
            return False
        runner._write_control_file(cgroup / "cgroup.kill", b"1")
        deadline = (
            time.monotonic()
            + _MANAGED_TERMINAL_CGROUP_CLEANUP_TIMEOUT_SECONDS
        )
        while time.monotonic() < deadline:
            events = runner._read_bounded_ascii(
                cgroup / "cgroup.events",
                limit=4096,
            ).splitlines()
            if "populated 0" in events:
                os.rmdir(cgroup)
                return True
            time.sleep(_MANAGED_TERMINAL_CGROUP_POLL_SECONDS)
        raise OSError("managed terminal cgroup remained populated")


def _prepare_managed_terminal_workspace(
    directory: str,
    paths: list[str],
    *,
    env: Mapping[str, str],
) -> None:
    """Transfer one generated scratch workspace to its profile identity."""

    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return
    uid, gid = _managed_terminal_identity(env)
    root = os.lstat(directory)
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_uid != 0
        or root.st_mode & 0o022
    ):
        raise OSError("managed terminal workspace is not trusted")
    for path in paths:
        info = os.lstat(path)
        if (
            not (stat.S_ISREG(info.st_mode) or stat.S_ISSOCK(info.st_mode))
            or info.st_uid != 0
            or info.st_mode & 0o022
        ):
            raise OSError("managed terminal workspace entry is not trusted")
        os.chown(path, uid, gid)
        os.chmod(path, 0o600)
    os.chown(directory, uid, gid)
    os.chmod(directory, 0o700)


def _prepare_managed_execute_code_workspace(
    directory: str,
    paths: list[str],
    *,
    env: Mapping[str, str],
    execution_scope: str,
) -> int:
    """Transfer one scratch workspace to a per-invocation execute_code UID."""

    uid, gid = _managed_execute_code_identity(env, execution_scope)
    root = os.lstat(directory)
    if (
        not stat.S_ISDIR(root.st_mode)
        or root.st_uid != 0
        or root.st_mode & 0o022
    ):
        raise OSError("managed execute_code workspace is not trusted")
    for path in paths:
        info = os.lstat(path)
        if (
            not (stat.S_ISREG(info.st_mode) or stat.S_ISSOCK(info.st_mode))
            or info.st_uid != 0
            or info.st_mode & 0o022
        ):
            raise OSError("managed execute_code workspace entry is not trusted")
        os.chown(path, uid, gid)
        os.chmod(path, 0o600)
    for child_name in ("var-tmp",):
        child = Path(directory) / child_name
        os.mkdir(child, 0o700)
        child_info = os.lstat(child)
        if (
            not stat.S_ISDIR(child_info.st_mode)
            or child_info.st_uid != 0
            or child_info.st_gid != 0
            or child_info.st_mode & 0o077
        ):
            raise OSError("managed execute_code private tmp is not trusted")
        os.chown(child, uid, gid)
        os.chmod(child, 0o700)
    os.chown(directory, uid, gid)
    os.chmod(directory, 0o700)
    return uid


def _managed_execute_code_private_tmp_paths(
    workspace: str, uid: int
) -> tuple[Path, Path]:
    """Validate invocation-owned mount sources created in its scratch workspace."""

    raw_home = str(workspace or "")
    if not raw_home or not os.path.isabs(raw_home) or "\x00" in raw_home:
        raise OSError("managed execute_code private tmp is unavailable")
    home = Path(raw_home)
    home_info = os.lstat(home)
    if (
        not stat.S_ISDIR(home_info.st_mode)
        or home_info.st_uid != uid
        or home_info.st_gid != uid
        or home_info.st_mode & 0o077
    ):
        raise OSError("managed execute_code HOME is not trusted")

    private_var_tmp = home / "var-tmp"
    child_info = os.lstat(private_var_tmp)
    if (
        not stat.S_ISDIR(child_info.st_mode)
        or child_info.st_uid != uid
        or child_info.st_gid != uid
        or child_info.st_mode & 0o077
    ):
        raise OSError("managed execute_code private tmp is not trusted")
    return home, private_var_tmp


def _managed_terminal_home_paths(
    env: Mapping[str, str] | None,
) -> tuple[Path, Path, Path]:
    """Create the profile HOME and private tmp mount sources."""

    uid, gid = _managed_terminal_identity(env)
    parent = _MANAGED_TERMINAL_HOME_ROOT.parent
    parent_info = os.lstat(parent)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != 0
        or parent_info.st_mode & 0o022
    ):
        raise OSError("managed terminal runtime directory is not trusted")

    try:
        os.mkdir(_MANAGED_TERMINAL_HOME_ROOT, 0o711)
    except FileExistsError:
        pass
    root_info = os.lstat(_MANAGED_TERMINAL_HOME_ROOT)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or root_info.st_uid != 0
        or root_info.st_gid != 0
        or root_info.st_mode & 0o022
    ):
        raise OSError("managed terminal home root is not trusted")
    os.chmod(_MANAGED_TERMINAL_HOME_ROOT, 0o711)

    home = _MANAGED_TERMINAL_HOME_ROOT / str(uid)
    created = False
    try:
        os.mkdir(home, 0o700)
        created = True
    except FileExistsError:
        pass
    if created:
        os.chown(home, uid, gid)
        os.chmod(home, 0o700)
    home_info = os.lstat(home)
    if (
        not stat.S_ISDIR(home_info.st_mode)
        or home_info.st_uid != uid
        or home_info.st_gid != gid
        or home_info.st_mode & 0o077
    ):
        raise OSError("managed terminal profile home is not trusted")

    private_paths = []
    for child_name in ("tmp", "var-tmp"):
        child = home / child_name
        created = False
        try:
            os.mkdir(child, 0o700)
            created = True
        except FileExistsError:
            pass
        if created:
            os.chown(child, uid, gid)
        child_info = os.lstat(child)
        if (
            not stat.S_ISDIR(child_info.st_mode)
            or child_info.st_uid != uid
            or child_info.st_gid != gid
        ):
            raise OSError("managed terminal private tmp is not trusted")
        os.chmod(child, 0o700, follow_symlinks=False)
        child_info = os.lstat(child)
        if child_info.st_mode & 0o077:
            raise OSError("managed terminal private tmp is not owner-only")
        private_paths.append(child)

    return home, private_paths[0], private_paths[1]


def _prepare_managed_terminal_home(env: dict[str, str]) -> str:
    """Set a profile-scoped HOME and namespace-local tmp environment."""

    home, _private_tmp, _private_var_tmp = _managed_terminal_home_paths(env)
    home_text = str(home)
    env["HOME"] = home_text
    env["TMPDIR"] = "/tmp"
    env["TMP"] = "/tmp"
    env["TEMP"] = "/tmp"
    return home_text


def _managed_uid_processes(
    uid: int,
    proc_root: Path = Path("/proc"),
) -> set[int]:
    """Return Linux processes whose effective UID is the managed identity."""
    if _IS_WINDOWS or not proc_root.is_dir():
        return set()
    processes: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(
                encoding="utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        effective_uid: int | None = None
        is_zombie = False
        for line in status.splitlines():
            if line.startswith("State:"):
                state_fields = line.split()
                is_zombie = len(state_fields) >= 2 and state_fields[1] == "Z"
            elif line.startswith("Uid:"):
                uid_fields = line.split()
                if len(uid_fields) >= 3:
                    effective_uid = int(uid_fields[2])
        # Zombies have no address space, open descriptors, or executable
        # thread left to cross a profile boundary. Their parent may reap them
        # after this bounded cleanup, so treating them as killable would make
        # every SIGKILL path time out forever.
        if effective_uid == uid and not is_zombie:
            processes.add(int(entry.name))
    return processes


def _terminate_managed_uid(uid: int, timeout: float = 2.0) -> int:
    """Terminate every process in a managed identity and verify it is empty."""
    killed: set[int] = set()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        live = _managed_uid_processes(uid)
        if not live:
            return len(killed)
        for pid in live:
            try:
                os.kill(pid, sig)
                killed.add(pid)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _managed_uid_processes(uid):
                return len(killed)
            time.sleep(0.05)
    live = _managed_uid_processes(uid)
    if live:
        raise OSError(
            "managed terminal processes survived identity retirement: "
            + ",".join(str(pid) for pid in sorted(live))
        )
    return len(killed)


def retire_managed_execute_code_identity(
    uid: int,
    env: Mapping[str, str],
    execution_scope: str,
) -> int:
    """Empty one invocation UID before making it available for reuse.

    A model script can detach descendants from the process group that owns the
    top-level ``execute_code`` child.  The per-invocation UID is the durable
    containment boundary, so it must be verified empty before its reservation
    is released.  If termination fails, the reservation deliberately remains
    live and the caller fails closed.
    """

    with _MANAGED_EXECUTE_CODE_CGROUP_LOCK:
        cgroup = _MANAGED_EXECUTE_CODE_CGROUP_BY_UID.get(uid)
    if cgroup is not None:
        from tools.trusted_direct_runner import _kill_and_remove_managed_cgroup

        _kill_and_remove_managed_cgroup(cgroup, None)
        with _MANAGED_EXECUTE_CODE_CGROUP_LOCK:
            if _MANAGED_EXECUTE_CODE_CGROUP_BY_UID.get(uid) is cgroup:
                _MANAGED_EXECUTE_CODE_CGROUP_BY_UID.pop(uid, None)

    killed = _terminate_managed_uid(uid)
    _release_managed_execute_code_identity(uid, env, execution_scope)
    return killed


def retire_managed_terminal_profile(profile_home: str) -> dict[str, object]:
    """Destroy a profile UID domain before that profile can be recreated."""
    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return {
            "killed_uid_processes": 0,
            "terminal_home_removed": False,
            "terminal_cgroup_removed": False,
            "identity_retired": False,
        }
    scope = _managed_terminal_profile_scope({"HERMES_HOME": profile_home})
    with _MANAGED_TERMINAL_IDENTITY_LOCK:
        matches = [
            uid
            for uid, owner in _MANAGED_TERMINAL_SCOPE_BY_UID.items()
            if owner == scope
        ]
        if not matches:
            return {
                "killed_uid_processes": 0,
                "terminal_home_removed": False,
                "terminal_cgroup_removed": False,
                "identity_retired": scope in _MANAGED_TERMINAL_RETIRED_SCOPES,
            }
        if len(matches) != 1:
            raise OSError("managed terminal profile has ambiguous identities")
        uid = matches[0]
        if (
            uid not in _MANAGED_TERMINAL_RETIRED_UIDS
            and len(_MANAGED_TERMINAL_RETIRED_UIDS)
            >= _MANAGED_TERMINAL_RETIRED_MAX
        ):
            raise OSError("managed terminal retired identity cache is full")

        killed = _terminate_managed_uid(uid)
        cgroup_removed = _remove_managed_terminal_cgroup(uid)
        home = _MANAGED_TERMINAL_HOME_ROOT / str(uid)
        removed = False
        try:
            info = os.lstat(home)
        except FileNotFoundError:
            pass
        else:
            if (
                not stat.S_ISDIR(info.st_mode)
                or info.st_uid != uid
                or info.st_gid != uid
                or info.st_mode & 0o077
            ):
                raise OSError("managed terminal profile home is not trusted")
            shutil.rmtree(home)
            removed = True
        if _managed_uid_processes(uid):
            raise OSError("managed terminal identity is still active")
        _MANAGED_TERMINAL_SCOPE_BY_UID.pop(uid, None)
        _MANAGED_TERMINAL_RETIRED_UIDS.add(uid)
        _MANAGED_TERMINAL_RETIRED_SCOPES.add(scope)
        return {
            "killed_uid_processes": killed,
            "terminal_home_removed": removed,
            "terminal_cgroup_removed": cgroup_removed,
            "identity_retired": True,
        }


def _managed_identity_can_traverse(
    directory: str,
    *,
    uid: int,
    gid: int,
) -> bool:
    """Check directory traversal using the runner's cleared-group identity."""

    try:
        resolved = Path(directory).resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    components = [resolved, *resolved.parents]
    for component in reversed(components):
        try:
            info = component.stat()
        except OSError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            return False
        if info.st_uid == uid:
            permission = (info.st_mode >> 6) & 0o7
        elif info.st_gid == gid:
            permission = (info.st_mode >> 3) & 0o7
        else:
            permission = info.st_mode & 0o7
        if permission & 0o1 == 0:
            return False
    return True


def _managed_skill_entry_mode(info: os.stat_result) -> int | None:
    if info.st_uid != 0:
        raise OSError("managed profile skill entry is not trusted")
    if stat.S_ISLNK(info.st_mode):
        return None
    if stat.S_ISDIR(info.st_mode):
        return 0o750
    if stat.S_ISREG(info.st_mode):
        if info.st_nlink != 1:
            raise OSError("managed profile skill hard link is not trusted")
        return 0o640 | (0o110 if info.st_mode & stat.S_IXUSR else 0)
    raise OSError("managed profile skill entry type is not trusted")


def _normalize_managed_skill_package(package: Path, gid: int) -> None:
    """FD-walk and harden one root-owned package without following symlinks."""

    root_info = os.lstat(package)
    root_mode = _managed_skill_entry_mode(root_info)
    if root_mode is None:
        return
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    root_flags = directory_flags if stat.S_ISDIR(root_info.st_mode) else flags
    root_fd = os.open(package, root_flags)
    opened_root = os.fstat(root_fd)
    if (
        opened_root.st_dev != root_info.st_dev
        or opened_root.st_ino != root_info.st_ino
        or _managed_skill_entry_mode(opened_root) != root_mode
    ):
        os.close(root_fd)
        raise OSError("managed profile skill package changed during preparation")
    if opened_root.st_gid != gid:
        os.fchown(root_fd, 0, gid)
    if stat.S_IMODE(opened_root.st_mode) != root_mode:
        os.fchmod(root_fd, root_mode)
    if not stat.S_ISDIR(opened_root.st_mode):
        os.close(root_fd)
        return

    stack: list[tuple[int, object]] = [(root_fd, None)]
    count = 0
    try:
        while stack:
            directory_fd, iterator = stack[-1]
            if iterator is None:
                iterator = os.scandir(directory_fd)
                stack[-1] = (directory_fd, iterator)
            try:
                entry = next(iterator)
            except StopIteration:
                iterator.close()
                os.close(directory_fd)
                stack.pop()
                continue
            count += 1
            if count > _MANAGED_SKILL_TREE_MAX_ENTRIES:
                raise OSError("managed profile skill tree exceeds the safety limit")
            info = entry.stat(follow_symlinks=False)
            mode = _managed_skill_entry_mode(info)
            if mode is None:
                continue
            entry_flags = directory_flags if stat.S_ISDIR(info.st_mode) else flags
            entry_fd = os.open(entry.name, entry_flags, dir_fd=directory_fd)
            opened = os.fstat(entry_fd)
            if (
                opened.st_dev != info.st_dev
                or opened.st_ino != info.st_ino
                or _managed_skill_entry_mode(opened) != mode
            ):
                os.close(entry_fd)
                raise OSError("managed profile skill entry changed during preparation")
            if opened.st_gid != gid:
                os.fchown(entry_fd, 0, gid)
            if stat.S_IMODE(opened.st_mode) != mode:
                os.fchmod(entry_fd, mode)
            if stat.S_ISDIR(opened.st_mode):
                stack.append((entry_fd, None))
            else:
                os.close(entry_fd)
    finally:
        for directory_fd, iterator in stack:
            if iterator is not None:
                iterator.close()
            os.close(directory_fd)


def _managed_python_skill_sources(
    command: str,
    profile_home: Path,
) -> set[tuple[Path, Path]]:
    """Identify Python's first script operand inside the active skill root."""

    lexical_skills_root = profile_home.expanduser() / "skills"
    try:
        resolved_skills_root = lexical_skills_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return set()
    sources: set[tuple[Path, Path]] = set()
    for segment in re.split(r"&&|\|\||;|\||\n|&", command or ""):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        while tokens and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[0]):
            tokens.pop(0)
        if not tokens or not re.fullmatch(
            r"python(?:\d+(?:\.\d+)*)?",
            os.path.basename(tokens[0]),
        ):
            continue
        script = ""
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in ("-c", "-m"):
                break
            if token in ("-W", "-X"):
                index += 2
                continue
            if token == "--":
                index += 1
                if index < len(tokens):
                    script = tokens[index]
                break
            if token.startswith("-"):
                index += 1
                continue
            script = token
            break
        if not script or not os.path.isabs(script):
            continue
        try:
            lexical_source = Path(os.path.normpath(script))
            lexical_source.relative_to(lexical_skills_root)
            resolved_source = lexical_source.resolve(strict=True)
            resolved_source.relative_to(resolved_skills_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved_source.is_file():
            sources.add((lexical_source, resolved_source))
    return sources


def _prepare_managed_command_skill_sources(
    command: str,
    env: Mapping[str, str],
) -> None:
    """Make each invoked skill package profile-readable before namespace mount."""

    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return
    raw_profile_home = str(env.get("HERMES_HOME") or "").strip()
    if not raw_profile_home:
        return
    lexical_profile_home = Path(raw_profile_home).expanduser()
    profile_home = Path(_managed_terminal_profile_scope(env))
    lexical_skills_root = lexical_profile_home / "skills"
    skills_root = profile_home / "skills"
    _uid, gid = _managed_terminal_identity(env)
    packages: set[Path] = set()
    for lexical_source, resolved_source in _managed_python_skill_sources(
        command, lexical_profile_home
    ):
        lexical_relative = lexical_source.relative_to(lexical_skills_root)
        packages.add(lexical_skills_root / lexical_relative.parts[0])
        resolved_relative = resolved_source.relative_to(skills_root)
        packages.add(skills_root / resolved_relative.parts[0])
    prepared_inodes: set[tuple[int, int]] = set()
    for package in packages:
        root_info = os.lstat(package)
        inode_key = (root_info.st_dev, root_info.st_ino)
        if inode_key in prepared_inodes:
            continue
        prepared_inodes.add(inode_key)
        _normalize_managed_skill_package(package, gid)


def _migrate_managed_output_tree(
    output_dir: Path,
    *,
    uid: int,
    gid: int,
    expected: os.stat_result,
) -> None:
    """FD-walk, validate, then transfer an older identity's profile output."""

    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(output_dir, directory_flags)

    def _trusted(info: os.stat_result) -> bool:
        allowed_type = (
            stat.S_ISDIR(info.st_mode)
            or stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
        )
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            return False
        # Historical output can come from trusted helpers/containers running
        # under a different numeric UID. The output root is anchored beneath a
        # validated root-owned parent, and transfer opens every non-symlink via
        # dir_fd + O_NOFOLLOW before changing ownership. Restrict entry shape
        # and hard links here; normalize the previous owner during transfer.
        return allowed_type

    def _walk(*, transfer: bool) -> None:
        stack: list[tuple[int, object]] = [(os.dup(root_fd), None)]
        count = 0
        try:
            while stack:
                directory_fd, iterator = stack[-1]
                if iterator is None:
                    iterator = os.scandir(directory_fd)
                    stack[-1] = (directory_fd, iterator)
                try:
                    entry = next(iterator)
                except StopIteration:
                    iterator.close()
                    os.close(directory_fd)
                    stack.pop()
                    continue
                count += 1
                if count > _MANAGED_OUTPUT_TREE_MAX_ENTRIES:
                    raise OSError("managed profile tree exceeds the safety limit")
                info = entry.stat(follow_symlinks=False)
                if not _trusted(info):
                    raise OSError("managed profile output entry is not trusted")
                if stat.S_ISLNK(info.st_mode):
                    # Symlink ownership has no bearing on traversal and changing
                    # it by pathname would reintroduce a lstat/chown race.
                    continue
                entry_flags = directory_flags if stat.S_ISDIR(info.st_mode) else flags
                entry_fd = os.open(entry.name, entry_flags, dir_fd=directory_fd)
                opened_info = os.fstat(entry_fd)
                if (
                    opened_info.st_dev != info.st_dev
                    or opened_info.st_ino != info.st_ino
                    or not _trusted(opened_info)
                ):
                    os.close(entry_fd)
                    raise OSError("managed profile output changed during transfer")
                if transfer:
                    os.fchown(entry_fd, uid, gid)
                    if stat.S_ISDIR(opened_info.st_mode):
                        os.fchmod(entry_fd, 0o700)
                    else:
                        mode = (stat.S_IMODE(opened_info.st_mode) & 0o700) | 0o600
                        os.fchmod(entry_fd, mode)
                if stat.S_ISDIR(opened_info.st_mode):
                    stack.append((entry_fd, None))
                else:
                    os.close(entry_fd)
        finally:
            for directory_fd, iterator in stack:
                if iterator is not None:
                    iterator.close()
                os.close(directory_fd)

    try:
        root_info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_dev != expected.st_dev
            or root_info.st_ino != expected.st_ino
        ):
            raise OSError("managed profile output changed before transfer")
        _walk(transfer=False)
        _walk(transfer=True)
        current = os.lstat(output_dir)
        if current.st_dev != root_info.st_dev or current.st_ino != root_info.st_ino:
            raise OSError("managed profile output changed during transfer")
        os.fchown(root_fd, uid, gid)
        os.fchmod(root_fd, 0o700)
    finally:
        os.close(root_fd)


def _prepare_managed_profile_runtime(env: Mapping[str, str]) -> None:
    """Expose only this profile's skills and output to its terminal identity.

    Platform profiles are root-owned. The model shell receives a stable,
    profile-stable UID/GID, so it needs execute permission on the common parents,
    profile-group access to its own source, and ownership of its output. The
    skill tree is additionally bind-mounted read-only in the child namespace.
    """

    output_text = str(env.get("ZET_AGENT_OUTPUT_DIR") or "").strip()
    if not output_text:
        return
    uid, gid = _managed_terminal_identity(env)
    try:
        profile_home = _validate_managed_root_directory_chain(
            Path(_managed_terminal_profile_scope(env))
        )
        profiles_root = profile_home.parent
        hermes_root = profiles_root.parent
        skills_root = profile_home / "skills"
        output_raw = Path(output_text)
        output_dir = output_raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OSError("managed profile runtime paths are unavailable") from exc

    if profiles_root.name != "profiles":
        raise OSError("managed profile runtime scope is invalid")
    if output_raw != output_dir or output_dir.name != "output":
        raise OSError("managed profile output is invalid")
    if output_dir.parent.name != profile_home.name:
        raise OSError("managed profile output does not match the active profile")
    _validate_managed_root_directory_chain(output_dir.parent)
    output_parent_info = os.lstat(output_dir.parent)
    if (
        not stat.S_ISDIR(output_parent_info.st_mode)
        or output_parent_info.st_uid != 0
        or output_parent_info.st_mode & 0o022
    ):
        raise OSError("managed profile output parent is not trusted")

    for common in (hermes_root, profiles_root):
        info = os.lstat(common)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != 0
            or info.st_mode & 0o022
        ):
            raise OSError("managed profile parent is not trusted")
        os.chmod(common, 0o711, follow_symlinks=False)

    profile_info = os.lstat(profile_home)
    if (
        not stat.S_ISDIR(profile_info.st_mode)
        or profile_info.st_uid != 0
        or profile_info.st_mode & 0o022
    ):
        raise OSError("managed profile home is not trusted")
    os.chown(profile_home, 0, gid, follow_symlinks=False)
    os.chmod(profile_home, 0o710, follow_symlinks=False)

    try:
        skills_info = os.lstat(skills_root)
    except FileNotFoundError:
        skills_info = None
    if skills_info is not None:
        if (
            not stat.S_ISDIR(skills_info.st_mode)
            or skills_info.st_uid != 0
            or skills_info.st_mode & 0o022
        ):
            raise OSError("managed profile skills are not trusted")
        os.chown(skills_root, 0, gid, follow_symlinks=False)
        os.chmod(skills_root, 0o750, follow_symlinks=False)

    output_info = os.lstat(output_dir)
    if (
        not stat.S_ISDIR(output_info.st_mode)
        or output_info.st_mode & 0o022
        or output_info.st_uid not in (0, uid)
    ):
        raise OSError("managed profile output is not trusted")
    if (
        output_info.st_uid != uid
        or output_info.st_gid != gid
        or stat.S_IMODE(output_info.st_mode) != 0o700
    ):
        _migrate_managed_output_tree(
            output_dir,
            uid=uid,
            gid=gid,
            expected=output_info,
        )


def _managed_terminal_cwd(
    cwd: str,
    *,
    env: dict[str, str],
) -> str:
    """Return an accessible cwd and set the matching per-profile HOME."""

    if _IS_WINDOWS or os.environ.get(_MANAGED_GATEWAY_ENV) != "1":
        return cwd
    home = _prepare_managed_terminal_home(env)
    _prepare_managed_profile_runtime(env)
    uid, gid = _managed_terminal_identity(env)
    if cwd and _managed_identity_can_traverse(cwd, uid=uid, gid=gid):
        return cwd
    return home


def _msys_to_windows_path(cwd: str) -> str:
    """Translate a Git Bash / MSYS-style POSIX path (``/c/Users/x``) to the
    native Windows form (``C:\\Users\\x``) so ``os.path.isdir`` and
    ``subprocess.Popen(..., cwd=...)`` can find it.

    Also accepts the Cygwin (``/cygdrive/c/...``) and WSL-mount
    (``/mnt/c/...``) spellings of a drive root. Multi-segment POSIX paths
    like ``/home/x`` or ``/tmp/foo`` are left untouched.

    No-ops on non-Windows hosts or for paths that aren't in MSYS form.
    Returns the input unchanged when no translation applies. This is
    idempotent — calling it on an already-Windows path returns it as-is.
    """
    if not _IS_WINDOWS or not cwd:
        return cwd
    # Match leading "/<single letter>/" or exactly "/<letter>" (bare drive root),
    # plus /cygdrive/<letter>/... and /mnt/<letter>/... variants.
    m = re.match(r'^/(?:(?:cygdrive|mnt)/)?([a-zA-Z])(/.*)?$', cwd)
    if not m:
        return cwd
    # Reject /cygdrive or /mnt with no drive letter — the optional group above
    # already requires the letter. Multi-char first segments (/home, /tmp)
    # fail the single-letter capture and fall through as no-ops.
    drive = m.group(1).upper()
    tail = (m.group(2) or "").replace('/', '\\')
    return f"{drive}:{tail or chr(92)}"  # chr(92) = backslash, avoid raw-string escape


def _resolve_local_initial_cwd(cwd: str) -> str:
    """Resolve the local backend's initial cwd to an absolute host path.

    ``TERMINAL_CWD`` can be populated from config.yaml before the terminal
    backend is created.  If that value is relative and happens to match the
    directory Hermes was already launched from (for example ``hermes-agent``
    while the process cwd is ``~/.hermes/hermes-agent``), passing it through
    unchanged makes the wrapper run ``cd hermes-agent`` *inside* the project
    and fail with a confusing nested-path error.  Anchor relative local cwd
    values once, up front, so both ``subprocess.Popen(cwd=...)`` and the
    in-shell ``cd`` use the same absolute directory.
    """
    expanded = os.path.expanduser(cwd) if cwd else os.getcwd()
    if _IS_WINDOWS:
        expanded = _msys_to_windows_path(expanded)
        # Use the Windows-aware check explicitly: when _IS_WINDOWS is
        # patched in tests on a POSIX host, os.path.isabs would reject
        # ``C:\Users\x`` and mangle it through the relative branch.
        import ntpath
        if ntpath.isabs(expanded):
            return expanded
    if os.path.isabs(expanded):
        return expanded

    candidate = os.path.abspath(expanded)
    current = os.getcwd()

    # Common recovery for config values like ``hermes-agent`` when Hermes was
    # launched from that directory already.  ``os.path.abspath`` would point at
    # a nonexistent nested ``./hermes-agent``; use the current directory instead.
    if not os.path.isdir(candidate):
        wanted_parts = Path(expanded).parts
        current_parts = Path(current).parts
        if wanted_parts and len(wanted_parts) <= len(current_parts):
            if current_parts[-len(wanted_parts):] == wanted_parts:
                return current

    return candidate


def _windows_to_msys_path(cwd: str) -> str:
    """Translate a native Windows path (``C:\\Users\\x``) to Git Bash /
    MSYS form (``/c/Users/x``) so ``builtin cd`` resolves it reliably.

    No-ops on non-Windows hosts or for paths that aren't drive-qualified
    native Windows paths. Returns the input unchanged when no translation
    applies.
    """
    if not _IS_WINDOWS or not cwd:
        return cwd
    m = re.match(r'^([a-zA-Z]):[\\/]*(.*)$', cwd)
    if not m:
        return cwd
    drive = m.group(1).lower()
    tail = (m.group(2) or "").replace('\\', '/').lstrip('/')
    return f"/{drive}/{tail}" if tail else f"/{drive}/"


def _bash_safe_path(path: str) -> str:
    """Return *path* in a form safe to embed in a Git Bash script.

    Native ``C:\\Users\\x`` / ``C:/Users/x`` → ``/c/Users/x`` via
    :func:`_windows_to_msys_path`. Mixed MSYS leftovers
    (``/c/Users\\Alexander\\Documents``) get backslashes normalized so
    bash does not eat ``\\U`` and trip the ``Directory \\drivers\\etc``
    failure class. No-op off Windows and for empty input.

    ``get_temp_dir`` already emits forward-slash ``C:/...`` forms for
    Python compatibility; those still need the ``/c/...`` rewrite —
    MSYS argument conversion treats ``C:/...`` as a Windows path and
    can corrupt the login-shell ``drivers\\etc`` lookup.
    """
    if not _IS_WINDOWS or not path:
        return path
    path = _windows_to_msys_path(path)
    if "\\" in path:
        path = path.replace("\\", "/")
    return path


def _quote_bash_path(path: str) -> str:
    """Quote *path* for safe interpolation into a Git Bash script on Windows."""
    import shlex

    return shlex.quote(_bash_safe_path(path))


def _cwd_usable(path: str) -> bool:
    """True when *path* is a directory this process can actually chdir into.

    ``os.path.isdir`` alone is not enough: stat() on ``/root`` succeeds for a
    non-root user (only ``/`` needs search permission), but
    ``subprocess.Popen(cwd='/root')`` then dies with ``PermissionError:
    [Errno 13] Permission denied: '/root'``. Seen in the wild when a
    root-launched CLI session leaks ``/root`` into shared state that a
    non-root gateway/cron process later reads (#65583) — every cron job's
    terminal/file tool then fails on every command, forever. Checking
    X_OK up front lets the caller fall back instead.
    """
    return os.path.isdir(path) and os.access(path, os.X_OK)


def _resolve_safe_cwd(cwd: str) -> str:
    """Return ``cwd`` if it exists as a directory this process can enter,
    else the nearest existing accessible ancestor.  Falls back to
    ``tempfile.gettempdir()`` only if walking up the path can't find any
    usable directory (effectively never on a healthy filesystem, but cheap
    belt-and-braces).

    On Windows, also normalizes Git Bash / MSYS-style POSIX paths
    (``/c/Users/x``) to native Windows form before the isdir check so a
    perfectly valid ``pwd -P`` result from bash doesn't get rejected as
    "missing" (see ``_msys_to_windows_path``).

    Used by ``_run_bash`` to recover when the configured cwd is gone — most
    commonly because a previous tool call deleted its own working directory
    (issue #17558) — or inaccessible to this user, e.g. ``/root`` leaking
    from a root-launched CLI session into a non-root gateway's cron jobs
    (issue #65583).  Without this guard, ``subprocess.Popen(..., cwd=...)``
    raises ``FileNotFoundError``/``PermissionError`` before bash starts,
    wedging every subsequent terminal call until the gateway restarts.
    """
    cwd = _msys_to_windows_path(cwd) if _IS_WINDOWS else cwd
    if cwd and _cwd_usable(cwd):
        return cwd
    if cwd and os.path.isdir(cwd):
        logger.warning(
            "Configured terminal cwd %r exists but is not accessible to "
            "this user (uid=%s) — falling back to the nearest usable "
            "directory. If this is a gateway/cron process, check for "
            "root-owned paths leaking into terminal.cwd / TERMINAL_CWD "
            "(#65583).",
            cwd, getattr(os, "getuid", lambda: "?")(),
        )
    parent = os.path.dirname(cwd) if cwd else ""
    while parent:
        if _cwd_usable(parent):
            return parent
        next_parent = os.path.dirname(parent)
        if next_parent == parent:
            # Reached the filesystem root and it doesn't exist either —
            # genuinely nothing to fall back to except the temp dir.
            break
        parent = next_parent
    return tempfile.gettempdir()


# Hermes-internal env vars that should NOT leak into terminal subprocesses.
_HERMES_PROVIDER_ENV_FORCE_PREFIX = "_HERMES_FORCE_"

# Hermes-managed AWS *inference* credentials for ``auth_type="aws_sdk"``
# providers (Bedrock).  Scoped DELIBERATELY NARROW: this lists only the
# Bedrock-specific bearer token, which is a Hermes inference secret exactly
# analogous to ``OPENAI_API_KEY`` — nobody drives the ``aws``/``terraform``/
# ``boto3`` toolchain off it, so stripping it from terminal/execute_code
# subprocesses costs no user capability.
#
# The GENERAL AWS credential chain (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
# AWS_SESSION_TOKEN, AWS_PROFILE, and the config/role pointers) is INTENTIONALLY
# left inheritable.  Per SECURITY.md §3.2 the local terminal is the user's
# trusted operator shell; the agent having the same general AWS access the
# user's own shell has is the intended posture, not a leak.  Hard-blocklisting
# those vars would (a) regress every user who runs aws/terraform/cdk/boto3 in
# the agent terminal — not just Bedrock users, since the registry is iterated
# unconditionally — and (b) be unrecoverable, because env_passthrough.py
# refuses to re-allow anything in this blocklist (GHSA-rhgp-j443-p4rf).  See
# issue #32314 discussion.
_AWS_SDK_CREDENTIAL_ENV_VARS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
})


def _build_provider_env_blocklist() -> frozenset:
    """Derive the blocklist from provider, tool, and gateway config."""
    blocked: set[str] = set()

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        for pconfig in PROVIDER_REGISTRY.values():
            blocked.update(pconfig.api_key_env_vars)
            if pconfig.auth_type == "aws_sdk":
                blocked.update(_AWS_SDK_CREDENTIAL_ENV_VARS)
            if pconfig.base_url_env_var:
                blocked.add(pconfig.base_url_env_var)
    except ImportError:
        pass

    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS
        for name, metadata in OPTIONAL_ENV_VARS.items():
            category = metadata.get("category")
            if category in {"tool", "messaging"}:
                blocked.add(name)
            elif category == "setting" and metadata.get("password"):
                blocked.add(name)
    except ImportError:
        pass

    blocked.update({
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "LLM_MODEL",
        "GOOGLE_API_KEY",
        # Path to a GCP service-account JSON, not a bare key, so
        # OPTIONAL_ENV_VARS marks it password=False and the loop above skips it.
        "VERTEX_CREDENTIALS_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "PERPLEXITY_API_KEY",
        "COHERE_API_KEY",
        "FIREWORKS_API_KEY",
        "XAI_API_KEY",
        "HELICONE_API_KEY",
        "PARALLEL_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_NAME",
        "DISCORD_HOME_CHANNEL",
        "DISCORD_HOME_CHANNEL_NAME",
        "DISCORD_REQUIRE_MENTION",
        "DISCORD_FREE_RESPONSE_CHANNELS",
        "DISCORD_AUTO_THREAD",
        "SLACK_HOME_CHANNEL",
        "SLACK_HOME_CHANNEL_NAME",
        "SLACK_ALLOWED_USERS",
        "WHATSAPP_ENABLED",
        "WHATSAPP_MODE",
        "WHATSAPP_ALLOWED_USERS",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
        "SIGNAL_ALLOWED_USERS",
        "SIGNAL_GROUP_ALLOWED_USERS",
        "SIGNAL_HOME_CHANNEL",
        "SIGNAL_HOME_CHANNEL_NAME",
        "SIGNAL_IGNORE_STORIES",
        "HASS_TOKEN",
        "HASS_URL",
        "EMAIL_ADDRESS",
        "EMAIL_PASSWORD",
        "EMAIL_IMAP_HOST",
        "EMAIL_SMTP_HOST",
        "EMAIL_HOME_ADDRESS",
        "EMAIL_HOME_ADDRESS_NAME",
        "HERMES_DASHBOARD_SESSION_TOKEN",
        "GATEWAY_ALLOWED_USERS",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "DAYTONA_API_KEY",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_DELIVERY_KEY",
    })
    # CLAUDE_CODE_OAUTH_TOKEN is deliberately NOT stripped.  It is set and
    # owned by the user's Claude Code install (subscription OAuth), not a
    # Hermes-managed inference credential — Claude subscription auth is not a
    # working Hermes provider path.  Stripping it broke agent-spawned
    # ``claude`` CLIs: the child fell through to the shared macOS Keychain /
    # ``~/.claude/.credentials.json`` store and, on auth failure, cleared it,
    # logging the user out of their interactive Claude sessions (#55878).
    # It arrives via the registry loop above (anthropic api_key_env_vars),
    # so remove it explicitly.
    blocked.discard("CLAUDE_CODE_OAUTH_TOKEN")
    return frozenset(blocked)


_HERMES_PROVIDER_ENV_BLOCKLIST = _build_provider_env_blocklist()

# Active-virtualenv markers that must NOT leak into terminal subprocesses.
# The gateway runs inside its own venv, so its process environment carries
# VIRTUAL_ENV (and possibly CONDA_PREFIX). If those leak into commands the
# agent runs against OTHER Python projects, tools like ``uv``/``poetry`` treat
# the inherited value as the active environment and build/sync that other
# project's dependencies into the Hermes venv path instead of the project's own
# ``.venv`` — silently clobbering the Hermes environment (e.g. a project pinned
# to a different Python version overwrites it and breaks the gateway). The
# Hermes venv stays reachable via PATH (its bin dir is first), so stripping
# these markers is safe and only prevents the cross-project clobber (#23473).
_ACTIVE_VENV_MARKER_VARS = ("VIRTUAL_ENV", "CONDA_PREFIX")


def _is_hermes_internal_secret(key: str) -> bool:
    """Return True for Hermes-internal secrets injected under *dynamic* names.

    ``_HERMES_PROVIDER_ENV_BLOCKLIST`` is name-based and derived from the
    provider/tool registries, but the gateway and CLI also inject secrets into
    ``os.environ`` at runtime under names no static registry knows about:

    - ``AUXILIARY_<TASK>_API_KEY`` / ``AUXILIARY_<TASK>_BASE_URL`` — per-task
      side-LLM credentials bridged from ``config.yaml[auxiliary]`` by
      ``gateway/run.py`` and ``cli.py`` (vision, web_extract, approval,
      compression, and any plugin-registered auxiliary task). These are
      separate, often higher-spend API keys plus base URLs that may point at
      private endpoints; a model-authored shell command must never see them.
    - ``GATEWAY_RELAY_*_SECRET`` / ``GATEWAY_RELAY_*_KEY`` /
      ``GATEWAY_RELAY_*_TOKEN`` — relay-auth material provisioned by the
      gateway (``GATEWAY_RELAY_SECRET``, ``GATEWAY_RELAY_DELIVERY_KEY``).
      These are Tier-1 gateway secrets, like the messaging bot tokens in
      ``_ALWAYS_STRIP_KEYS``. Non-secret ``GATEWAY_RELAY_*`` routing hints
      (``GATEWAY_RELAY_URL``, ``GATEWAY_RELAY_PLATFORMS``, …) are NOT matched
      and remain visible.

    ``code_execution_tool.py`` already catches these via substring matching on
    ``KEY`` / ``SECRET`` / ``TOKEN``; the terminal backend's narrower name-based
    blocklist did not, which is the leak this predicate closes.

    This is the single source of truth for "Hermes-internal dynamic secret"
    across every spawn path — the terminal ``_make_run_env`` /
    ``_sanitize_subprocess_env`` filters, the Docker passthrough filter, and the
    non-terminal :func:`hermes_subprocess_env` helper all call it, so the
    dynamic patterns are stripped **unconditionally** regardless of
    ``env_passthrough`` skill registration or ``inherit_credentials``. Nothing
    a model-driving CLI legitimately needs matches these patterns.
    """
    upper = key.upper()
    if upper.startswith("AUXILIARY_") and (
        upper.endswith("_API_KEY") or upper.endswith("_BASE_URL")
    ):
        return True
    if upper.startswith("GATEWAY_RELAY_") and (
        upper.endswith("_SECRET") or upper.endswith("_KEY") or upper.endswith("_TOKEN")
    ):
        return True
    return False


def _inject_context_hermes_home(env: dict) -> None:
    """Bridge the context-local Hermes home override into subprocess env."""
    try:
        from hermes_constants import get_hermes_home_override

        value = get_hermes_home_override()
        if value:
            env["HERMES_HOME"] = value
    except Exception:
        pass


def _inject_session_context_env(env: dict) -> None:
    """Bridge gateway session ContextVars into a subprocess environment dict.

    ContextVars don't propagate to child processes, so the live session vars
    (HERMES_SESSION_*) are bridged onto the child env here.

    🔴 Cross-session leak guard. The session vars also have a process-global
    os.environ mirror (written last-writer-wins as a CLI/cron fallback, never
    cleared). Under a concurrent multi-session host (the messaging gateway, ACP
    adapter, API server, TUI) that global belongs to *whichever turn wrote it
    last* — NOT necessarily this task. A subprocess spawned from a task whose
    ContextVar is _UNSET (e.g. a sibling message task that never bound, or one
    that inherited another session's context) would otherwise inherit the
    FOREIGN global and act on another session's identity.

    So once the session-context machinery is engaged in this process (any host
    has called set_session_vars), the session vars are ContextVar-authoritative:
    - ContextVar set (incl. explicitly-empty "") → that value wins, overriding
      any stale snapshot/global value.
    - ContextVar _UNSET → STRIP the var from the child env rather than inherit
      the possibly-foreign process-global.
    In a pure single-process CLI/one-shot that never engaged the session-context
    system there is no concurrency to leak across, so the inherited fallback is
    kept. See gateway/session_context.session_context_engaged and
    tests/tools/test_local_env_session_leak.py.
    """
    try:
        from gateway.session_context import (
            _UNSET,
            _VAR_MAP,
            session_context_engaged,
        )
    except Exception:
        return

    _engaged = session_context_engaged()
    for var_name, var in _VAR_MAP.items():
        value = var.get()
        if value is not _UNSET:
            # Explicitly bound (including "") — authoritative for this task.
            env[var_name] = "" if value is None else str(value)
        elif _engaged:
            # Unset for THIS task while a concurrent host is engaged: drop any
            # inherited global so a sibling session's value can't leak in.
            env.pop(var_name, None)


def _with_zettlab_turn_id(command: str) -> str:
    """Prefix a terminal command with this request's correlation token."""
    try:
        from gateway.session_context import zettlab_turn_id

        turn_id = zettlab_turn_id()
    except Exception:
        turn_id = ""
    if not turn_id:
        return command
    return f"export ZETTLAB_TURN_ID={shlex.quote(turn_id)}\n{command}"


CONNECTOR_RUNTIME_ENV_KEYS: frozenset[str] = frozenset({
    # Connector skill runtime routing. These are generated per Zettlab agent
    # profile by local-server and live in <profile>/.env under the multiplex
    # gateway, so subprocesses must receive the current profile's scope instead
    # of whatever os.environ/shell snapshot happened to contain.
    "ZETTLAB_CONNECTORS_URL",
    "ZETTLAB_CONNECTORS_AUTH_TOKEN",
    "ZET_AGENT_ID",
})

AGENT_CREATOR_RUNTIME_ENV_KEYS: frozenset[str] = frozenset({
    "ZETTLAB_AGENT_ACTION_TOKEN",
})
VIDEO_EDIT_RUNTIME_ENV_KEYS: frozenset[str] = frozenset({
    # Turn-scoped side-effect capability. Generic subprocesses must not
    # inherit either a live ContextVar or a stale process-global fallback.
    "ZETTLAB_BUSINESS_EXECUTION_TOKEN",
})
MANAGED_SERVICE_SECRET_ENV_KEYS: frozenset[str] = frozenset({
    "ZET_AGENT_KEY",
})
PROFILE_PUBLIC_RUNTIME_ENV_KEYS: frozenset[str] = frozenset({
    # Platform-owned, profile-scoped filesystem capability. Unlike connector
    # and action tokens this value is safe for model-authored shell commands,
    # and skills use it to keep mutable state outside their read-only source.
    "ZET_AGENT_OUTPUT_DIR",
})
_AGENT_CREATOR_ACTION_TOKEN_MAX_BYTES = 4 * 1024
_AGENT_CREATOR_TURN_ID_MAX_BYTES = 256

PROFILE_SCOPED_SUBPROCESS_ENV_KEYS: frozenset[str] = frozenset(
    CONNECTOR_RUNTIME_ENV_KEYS
    | AGENT_CREATOR_RUNTIME_ENV_KEYS
    | VIDEO_EDIT_RUNTIME_ENV_KEYS
    | MANAGED_SERVICE_SECRET_ENV_KEYS
    | PROFILE_PUBLIC_RUNTIME_ENV_KEYS
)


def _apply_profile_secret_scope_env(env: dict, *, inject: bool) -> None:
    """Scrub profile values and optionally inject safe terminal runtime data.

    The multiplex gateway intentionally avoids merging every profile's .env into
    process-global os.environ. Generic terminal/background/helper subprocesses
    are not a trusted runner, so they must never inherit connector or Agent
    action bearer material from globals, extra env, or a shell snapshot. Skills
    that need secret values receive them through a dedicated allowlisted path.
    The one public terminal value is re-read from the active profile scope only;
    a stale process-global or shell-snapshot value is never trusted in multiplex
    mode.
    """
    for key in PROFILE_SCOPED_SUBPROCESS_ENV_KEYS:
        env.pop(key, None)
    if not inject:
        return

    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        scope = current_secret_scope()
        multiplex_active = is_multiplex_active()
    except Exception:
        scope = None
        multiplex_active = True

    for key in PROFILE_PUBLIC_RUNTIME_ENV_KEYS:
        if scope is not None:
            raw_value = scope.get(key)
        elif not multiplex_active:
            raw_value = os.environ.get(key)
        else:
            raw_value = None
        value = str(raw_value or "").strip()
        if (
            not value
            or "\x00" in value
            or len(value.encode("utf-8")) > 4096
            or not os.path.isabs(value)
            or not os.path.isdir(value)
        ):
            continue
        env[key] = os.path.normpath(value)


def build_connector_runtime_env(base_env: dict | None = None) -> dict[str, str]:
    """Build env for the dedicated connector_runtime.py runner.

    This is intentionally separate from the generic terminal env. Connector
    runtime bearer may be supplied to the allowlisted runner subprocess, but it
    must not be inherited by arbitrary model-authored shell commands.
    """
    env = _sanitize_subprocess_env(os.environ, base_env)

    scope = None
    multiplex_active = False
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active

        multiplex_active = is_multiplex_active()
        scope = current_secret_scope()
    except Exception:
        scope = None

    for key in CONNECTOR_RUNTIME_ENV_KEYS:
        value = None
        if scope is not None:
            value = scope.get(key)
        elif not multiplex_active:
            value = os.environ.get(key)
        if value is not None:
            env[key] = str(value)
        else:
            env.pop(key, None)
    try:
        from gateway.session_context import zettlab_connector_route_capability

        route_capability = zettlab_connector_route_capability()
    except Exception:
        route_capability = ""
    if route_capability:
        # Reuse the legacy runner header transport without exposing the real
        # session key as selection authority. Generic terminal subprocesses
        # never receive this private ContextVar.
        env["HERMES_SESSION_KEY"] = route_capability
    return env


def build_agent_creator_runtime_env() -> dict[str, str]:
    """Build the minimal env for the trusted agent-creator preset runner.

    The action token is never read from process env or a profile ``.env``.
    After command validation and mutation approval, the trusted gateway process
    obtains a short-lived AgentComputer-only token from local-server's Unix
    broker. The direct runner then gives it to the CLI over a one-shot FD.
    """

    from agent.credential_broker import request_agentcomputer_token
    from agent.secret_scope import current_secret_scope, is_multiplex_active

    scope = current_secret_scope()
    if scope is None and is_multiplex_active():
        raise RuntimeError("agent creator secret scope unavailable")
    agent_id = str(
        (scope or {}).get("ZET_AGENT_ID")
        or ("" if is_multiplex_active() else os.environ.get("ZET_AGENT_ID", ""))
    ).strip()
    if not agent_id:
        raise RuntimeError("agent creator profile identity unavailable")
    token = request_agentcomputer_token(agent_id)
    if (
        "\x00" in token
        or len(token.encode("utf-8")) > _AGENT_CREATOR_ACTION_TOKEN_MAX_BYTES
    ):
        raise RuntimeError("agent creator action token invalid")

    env = {"ZETTLAB_AGENT_ACTION_TOKEN": token}
    try:
        from gateway.session_context import zettlab_turn_id

        turn_id = zettlab_turn_id()
    except Exception:
        turn_id = ""
    if turn_id:
        turn_id = str(turn_id)
        if (
            "\x00" in turn_id
            or len(turn_id.encode("utf-8")) > _AGENT_CREATOR_TURN_ID_MAX_BYTES
        ):
            raise RuntimeError("agent creator turn id invalid")
        env["ZETTLAB_TURN_ID"] = turn_id
    return env


def build_video_edit_runtime_env(base_env: dict | None = None) -> dict[str, str]:
    """Build the minimal env for the trusted video-edit script runner."""
    env = _sanitize_subprocess_env(os.environ, base_env)
    for key in PROFILE_SCOPED_SUBPROCESS_ENV_KEYS:
        env.pop(key, None)
    _inject_session_context_env(env)

    try:
        from agent.zet_agent_response_mode import trusted_video_edit_runtime_receipt

        frozen_receipt = trusted_video_edit_runtime_receipt()
    except Exception:
        frozen_receipt = {}
    if not frozen_receipt:
        raise PermissionError("trusted video-edit execution receipt unavailable")
    env.update(frozen_receipt)
    return env


def _sanitize_subprocess_env(
    base_env: Mapping[str, str] | None,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Filter Hermes-managed secrets from a subprocess environment."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    sanitized: dict[str, str] = {}

    for key, value in (base_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            continue
        if _is_hermes_internal_secret(key):
            continue
        if key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    for key, value in (extra_env or {}).items():
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = key[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if _is_hermes_internal_secret(real_key):
                continue
            sanitized[real_key] = value
        elif _is_hermes_internal_secret(key):
            continue
        elif key not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(key):
            sanitized[key] = value

    _inject_context_hermes_home(sanitized)

    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(sanitized)

    # Same cross-session leak guard as _make_run_env, for the background/PTY
    # spawn path (process_registry.spawn_local builds env via this function).
    _inject_session_context_env(sanitized)
    _apply_profile_secret_scope_env(sanitized, inject=False)

    for _marker in _ACTIVE_VENV_MARKER_VARS:
        sanitized.pop(_marker, None)
    # These values authorize only the root gateway's ExecStart bootstrap.
    # Model-controlled terminal/background/PTY children are already placed in
    # their profile UID+cgroup boundary and must never re-enter that bootstrap.
    for _marker in _MANAGED_BOOTSTRAP_ENV_KEYS:
        sanitized.pop(_marker, None)

    _apply_windows_msys_bash_env_defaults(sanitized)

    return sanitized


# Tier-1 secrets: stripped from EVERY spawned subprocess unconditionally —
# even when the caller opts into credential inheritance for a model-driving
# CLI (claude / codex / gemini).  These are not LLM provider credentials; no
# legitimate child Hermes spawns needs them, and they are the highest-value
# secrets to keep out of a compromised dependency's reach (gateway bot tokens,
# GitHub auth, remote-compute tokens, dashboard session secret).  The set is a
# narrow subset of _HERMES_PROVIDER_ENV_BLOCKLIST; provider keys are handled by
# the conditional Tier-2 strip in hermes_subprocess_env().
_ALWAYS_STRIP_KEYS: frozenset[str] = frozenset({
    # GitHub auth
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY_PATH",
    "GITHUB_APP_INSTALLATION_ID",
    # Gateway / messaging bot tokens and access control
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_SIGNING_SECRET",
    "ZET_AGENT_KEY",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
    # Gateway relay auth — the ID/secret/delivery-key triplet the gateway
    # provisions and persists to the 0600 .env. Stripped unconditionally on
    # EVERY spawn surface (terminal + model-driving CLIs) so it can't drift
    # between paths: _SECRET / _DELIVERY_KEY are also matched by
    # _is_hermes_internal_secret, but _ID has no secret suffix, so it must be
    # enumerated here to stay stripped on the inherit_credentials=True path
    # (codex / copilot), which skips the Tier-2 blocklist.
    "GATEWAY_RELAY_ID",
    "GATEWAY_RELAY_SECRET",
    "GATEWAY_RELAY_DELIVERY_KEY",
    "HASS_TOKEN",
    "EMAIL_PASSWORD",
    "HERMES_DASHBOARD_SESSION_TOKEN",
    # Remote-compute / infrastructure secrets
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "DAYTONA_API_KEY",
})


def hermes_subprocess_env(*, inherit_credentials: bool = False) -> dict[str, str]:
    """Build a sanitized environment dict for a spawned subprocess.

    Centralized helper for the **non-terminal** spawn surface (browser,
    ACP/CLI executors, computer-use driver, dep-ensure, TUI Node host,
    detached gateway).  Use this instead of copying ``os.environ`` directly
    so strip-by-default is the uniform policy across every spawn site, with a
    single source of truth (``_HERMES_PROVIDER_ENV_BLOCKLIST``).  The terminal
    / execute_code path keeps using :func:`_sanitize_subprocess_env`, which is
    skill-aware (``env_passthrough``); this helper is for spawns that have no
    skill-passthrough concept.

    Two-tier stripping:

    * **Tier 1 (always):** ``_ALWAYS_STRIP_KEYS`` — gateway bot tokens, GitHub
      auth, and remote-compute secrets are removed regardless of
      ``inherit_credentials``.  No child Hermes spawns legitimately needs them.
    * **Tier 2 (conditional):** the rest of ``_HERMES_PROVIDER_ENV_BLOCKLIST``
      (LLM provider API keys, tool secrets) is removed unless the caller passes
      ``inherit_credentials=True``.

    Pass ``inherit_credentials=True`` **only** when the child legitimately
    needs LLM provider credentials — a user-blessed ``claude`` / ``codex`` /
    ``gemini`` CLI executor, or the TUI Node host that makes model calls.  The
    flag is grep-able for audit: ``grep -rn 'inherit_credentials=True'`` lists
    every spawn site that still receives provider credentials.

    Callers that need a *specific* non-provider secret (e.g. the browser worker
    needs ``BROWSERBASE_API_KEY`` / ``FIRECRAWL_API_KEY``) should call with
    ``inherit_credentials=False`` and copy just those keys back from
    ``os.environ`` into the returned dict.
    """
    env = os.environ.copy()

    # Tier 1 — always strip.
    for key in _ALWAYS_STRIP_KEYS:
        env.pop(key, None)
    # Internal routing hints and Hermes-internal dynamic secrets
    # (``AUXILIARY_<TASK>_API_KEY`` / ``_BASE_URL`` side-LLM credentials,
    # ``GATEWAY_RELAY_*`` relay-auth material) must never reach a child,
    # regardless of ``inherit_credentials`` — a model-driving CLI has no
    # legitimate use for them. See :func:`_is_hermes_internal_secret`.
    for key in list(env):
        if key.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            env.pop(key, None)
        elif _is_hermes_internal_secret(key):
            env.pop(key, None)

    if not inherit_credentials:
        # Tier 2 — strip provider/tool credentials unless explicitly inherited.
        for key in _HERMES_PROVIDER_ENV_BLOCKLIST:
            env.pop(key, None)

    # Windows UTF-8 safety for spawned processes (#31420).
    env.setdefault("PYTHONUTF8", "1")

    _inject_context_hermes_home(env)
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)

    # Active-venv markers must not clobber another project's environment.
    for _marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(_marker, None)

    _apply_windows_msys_bash_env_defaults(env)

    # Cross-session leak guard, same as the terminal spawn paths: this helper
    # copies os.environ, whose HERMES_SESSION_* mirror is a last-writer-wins
    # global under a concurrent multi-session host. A caller that re-binds the
    # session identity explicitly (slash_worker/ACP via --session-key argv) is
    # unaffected — bound ContextVars win here — but a caller that spawns without
    # re-binding (e.g. tui_gateway cli.exec) would otherwise inherit a FOREIGN
    # session's identity. Strip _UNSET session vars when engaged so that can't
    # happen; single uniform policy across every spawn surface.
    _inject_session_context_env(env)
    _apply_profile_secret_scope_env(env, inject=False)

    return env


def _find_bash() -> str:
    """Find bash for command execution."""
    if not _IS_WINDOWS:
        return (
            shutil.which("bash")
            or ("/usr/bin/bash" if os.path.isfile("/usr/bin/bash") else None)
            or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
            or os.environ.get("SHELL")
            or "/bin/sh"
        )

    candidates: list[str] = []

    custom = os.environ.get("HERMES_GIT_BASH_PATH")
    if custom and os.path.isfile(custom):
        candidates.append(custom)

    # Prefer our own portable Git install — a broken or partially-uninstalled
    # system Git (or a stale HERMES_GIT_BASH_PATH pointing at one) must not
    # brick the terminal.  install.ps1 drops PortableGit here when needed.
    #
    # Layouts (both checked so upgrades between MinGit and PortableGit
    # installs work transparently):
    #   PortableGit: %LOCALAPPDATA%\hermes\git\bin\bash.exe   (primary)
    #   MinGit:      %LOCALAPPDATA%\hermes\git\usr\bin\bash.exe (legacy/32-bit fallback)
    _local_appdata = os.environ.get("LOCALAPPDATA", "")
    _hermes_portable_git = os.path.join(_local_appdata, "hermes", "git") if _local_appdata else ""
    if _hermes_portable_git:
        for candidate in (
            os.path.join(_hermes_portable_git, "bin", "bash.exe"),        # PortableGit (primary)
            os.path.join(_hermes_portable_git, "usr", "bin", "bash.exe"), # MinGit fallback
        ):
            if os.path.isfile(candidate) and candidate not in candidates:
                candidates.append(candidate)

    # Check known Git for Windows install locations before PATH lookup.
    # On machines with both WSL and Git for Windows, shutil.which("bash")
    # may return WSL's bash (which doesn't understand Windows paths and
    # will fail silently).  Explicit Git-for-Windows paths avoid that.
    for candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Git", "bin", "bash.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Git", "bin", "bash.exe"),
        os.path.join(_local_appdata, "Programs", "Git", "bin", "bash.exe") if _local_appdata else "",
    ):
        if candidate and os.path.isfile(candidate) and candidate not in candidates:
            candidates.append(candidate)

    found = shutil.which("bash")
    if found and found not in candidates:
        candidates.append(found)

    # Prefer the first candidate that can actually start.  A stale
    # HERMES_GIT_BASH_PATH pointing at a broken Git-for-Windows install
    # (``Directory \\drivers\\etc does not exist``) must not win over a
    # healthy portable Git under %LOCALAPPDATA%\\hermes\\git.
    for candidate in candidates:
        if _bash_starts(candidate):
            if candidate != custom and custom and os.path.isfile(custom):
                logger.warning(
                    "HERMES_GIT_BASH_PATH=%s fails to start; using %s instead",
                    custom,
                    candidate,
                )
            return candidate

    if candidates:
        probe_details = "\n".join(
            detail
            for candidate in candidates
            if (detail := _bash_probe_details_cache.get(candidate))
        )
        if _mandatory_aslr_enabled() is True or _looks_like_msys_spawn_failure(
            probe_details
        ):
            raise RuntimeError(_git_bash_aslr_help(candidates[0], probe_details))

        # Last resort for failures unrelated to the known MSYS/ASLR class:
        # return the first path so the caller still sees the real bash error
        # instead of the less useful "not found" message.
        return candidates[0]

    raise RuntimeError(
        "Git Bash not found. Hermes Agent requires Git for Windows on Windows.\n"
        "Install it from: https://git-scm.com/download/win\n"
        "Or set HERMES_GIT_BASH_PATH to your bash.exe location."
    )


_bash_starts_cache: dict[str, bool] = {}
_bash_probe_details_cache: dict[str, str] = {}
_mandatory_aslr_enabled_cache: "bool | None" = None

_BASH_EXTERNAL_PROGRAM_PROBE = "/usr/bin/true; /usr/bin/cat --version >/dev/null"


def _looks_like_msys_spawn_failure(details: str) -> bool:
    """Match Git-for-Windows child-launch failures associated with ASLR."""
    lowered = details.lower()
    return any(
        marker in lowered
        for marker in (
            "dofork:",
            "child_copy:",
            "0xc0000142",
            "0xc0000005",
        )
    )


def _mandatory_aslr_enabled() -> "bool | None":
    """Return Windows' system-wide ForceRelocateImages state when available."""
    global _mandatory_aslr_enabled_cache
    if _mandatory_aslr_enabled_cache is not None:
        return _mandatory_aslr_enabled_cache

    try:
        powershell = shutil.which("powershell.exe") or "powershell.exe"
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-ProcessMitigation -System).Aslr.ForceRelocateImages.ToString()",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip().upper()
        if value == "ON":
            _mandatory_aslr_enabled_cache = True
            return True
        if value in {"OFF", "NOTSET"}:
            _mandatory_aslr_enabled_cache = False
            return False
    except Exception as exc:
        logger.debug("Could not query Windows Mandatory ASLR state: %s", exc)
    return None


def _git_root_from_bash(bash: str) -> str:
    """Resolve Git's root from either <root>/bin or <root>/usr/bin bash."""
    bin_dir = ntpath.dirname(ntpath.normpath(bash))
    if ntpath.basename(bin_dir).lower() != "bin":
        return ntpath.dirname(bin_dir)
    parent = ntpath.dirname(bin_dir)
    if ntpath.basename(parent).lower() == "usr":
        return ntpath.dirname(parent)
    return parent


def _git_bash_aslr_help(bash: str, details: str = "") -> str:
    """Build the targeted per-program Mandatory-ASLR remediation."""
    git_root = _git_root_from_bash(bash)
    escaped_root = git_root.replace("'", "''")
    detail_line = f"\nGit Bash probe output: {details[:500]}" if details else ""
    return (
        f"Git Bash at {bash} cannot launch required MSYS child processes while "
        "Windows Mandatory ASLR (ForceRelocateImages) is enabled, or its output "
        f"matches that Git-for-Windows failure class.{detail_line}\n"
        "Reinstalling Git will not change the Windows mitigation policy. Open "
        "PowerShell as Administrator and run:\n"
        f"$gitRoot = '{escaped_root}'\n"
        'Get-Item "$gitRoot\\bin\\bash.exe", "$gitRoot\\usr\\bin\\*.exe" '
        "-ErrorAction SilentlyContinue | ForEach-Object { "
        "Set-ProcessMitigation -Name $_.FullName -Disable ForceRelocateImages }\n"
        "Then restart Hermes. If the override is blocked or later re-applied, "
        "ask your Windows administrator to allow this per-program exception."
    )


def _bash_starts(bash: str) -> bool:
    """True if *bash* can launch external MSYS programs.

    Uses ``--noprofile --norc`` so a broken login post-install
    (``Directory \\drivers\\etc``) does not falsely condemn an otherwise
    usable bash. The external ``true`` and ``cat`` calls are intentional:
    a builtin-only ``exit 0`` probe misses Git-for-Windows fork/spawn failures
    under system-wide Mandatory ASLR. Cached per path for the process lifetime.
    """
    cached = _bash_starts_cache.get(bash)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [bash, "--noprofile", "--norc", "-c", _BASH_EXTERNAL_PROGRAM_PROBE],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=windows_hide_flags() if _IS_WINDOWS else 0,
        )
        ok = result.returncode == 0
        if not ok:
            combined = f"{result.stdout or ''}{result.stderr or ''}"
            _bash_probe_details_cache[bash] = combined.strip()[:2000]
            logger.debug("bash probe failed for %s: %s", bash, combined.strip()[:200])
    except Exception as exc:
        _bash_probe_details_cache[bash] = str(exc)[:2000]
        logger.debug("bash probe error for %s: %s", bash, exc)
        ok = False

    _bash_starts_cache[bash] = ok
    return ok


_git_bash_bin_dirs_cache: "list[str] | None" = None


def _git_bash_bin_dirs() -> list[str]:
    """Git Bash's coreutils/binary dirs, in ``/etc/profile`` precedence order.

    A non-login ``bash -c`` (the fallback used when ``bash -l`` is broken —
    the classic Windows ``Directory \\drivers\\etc does not exist`` failure)
    never sources ``/etc/profile``, so it never gets ``…\\usr\\bin`` on PATH.
    That directory holds every coreutil the file/terminal tools shell out to
    (``cat``, ``mktemp``, ``mv``, ``wc``, ``head``, ``stat``, ``chmod``,
    ``mkdir``, ``find`` …).  Without it, ``write_file`` fails with an empty
    error (the failure text went to a missing binary's stderr) and terminal
    commands exit 127.  We derive these dirs from the resolved ``bash.exe`` so
    the fallback shell can find coreutils regardless of the login shell.

    Returns ``[]`` off Windows or when bash can't be located.  Dirs are
    returned in the order Git Bash's own ``/etc/profile`` prepends them
    (mingw first, then usr/bin, then bin) and only if they exist on disk.
    """
    global _git_bash_bin_dirs_cache
    if _git_bash_bin_dirs_cache is not None:
        return _git_bash_bin_dirs_cache

    if not _IS_WINDOWS:
        _git_bash_bin_dirs_cache = []
        return _git_bash_bin_dirs_cache

    dirs: list[str] = []
    try:
        bash = _find_bash()
    except Exception:
        _git_bash_bin_dirs_cache = []
        return _git_bash_bin_dirs_cache

    bin_dir = os.path.dirname(bash)          # <root>\bin  or  <root>\usr\bin
    parent = os.path.dirname(bin_dir)
    # MinGit ships bash under usr\bin; PortableGit/system Git under bin.
    root = os.path.dirname(parent) if os.path.basename(parent).lower() == "usr" else parent

    # Order mirrors Git-for-Windows /etc/profile so coreutils win over the
    # same-named Windows System32 tools (find.exe, sort.exe) inside the shell.
    for candidate in (
        os.path.join(root, "mingw64", "bin"),
        os.path.join(root, "mingw32", "bin"),
        os.path.join(root, "usr", "local", "bin"),
        os.path.join(root, "usr", "bin"),
        os.path.join(root, "bin"),
    ):
        if os.path.isdir(candidate) and candidate not in dirs:
            dirs.append(candidate)

    _git_bash_bin_dirs_cache = dirs
    return dirs


def _prepend_git_bash_dirs(existing_path: str) -> str:
    """Prepend Git Bash's binary dirs to ``existing_path`` if missing.

    No-op off Windows or when the dirs can't be resolved.  First-occurrence
    wins, so a PATH that already lists a dir keeps its position.  This is what
    lets the non-login ``bash -c`` fallback find coreutils; in the healthy
    case the session snapshot re-exports the full login PATH inside the shell,
    so this only matters when that snapshot is absent.
    """
    git_dirs = _git_bash_bin_dirs()
    if not git_dirs:
        return existing_path
    sep = os.pathsep
    entries = [e for e in existing_path.split(sep) if e] if existing_path else []
    missing = [d for d in git_dirs if d not in entries]
    if not missing:
        return existing_path
    return sep.join([*missing, *entries])


# POSIX-sh-family shells that understand the ``[shell, "-lic", "set +m; …"]``
# invocation spawn_local uses. $SHELL values outside this set (fish, csh/tcsh,
# nushell, elvish, xonsh, …) would error on that syntax, so _find_shell falls
# back to bash for them rather than honouring $SHELL. (#42203)
_SPAWN_COMPATIBLE_SHELLS = frozenset({"bash", "zsh", "sh", "dash", "ksh", "mksh"})


def _find_shell() -> str:
    """Find the user's login shell for background process spawning.

    Unlike ``_find_bash`` (which always returns a bash binary for callers
    that explicitly need bash), this function prefers the user's configured
    ``$SHELL`` on POSIX so that ``spawn_local`` uses the shell the user
    actually logs in with.

    On macOS Catalina+ the default login shell is zsh, but
    ``shutil.which("bash")`` still finds the system ``/bin/bash`` (GNU bash
    3.2).  When bash 3.2 is invoked with ``-l`` (login) and stdin is
    ``/dev/null``, it sources ``~/.bash_profile`` which on many macOS setups
    contains ``exec /bin/zsh -l``.  That ``exec`` replaces bash with zsh but
    drops the ``-c`` argument, so the background command never runs — the
    subprocess exits 0 with no output and no side effects.

    Preferring ``$SHELL`` (when it is a POSIX-``sh``-family shell) avoids this
    because zsh/bash/sh/dash/ksh handle ``-lic`` correctly even with
    redirected stdin.

    Only POSIX-sh-family shells are honoured: ``spawn_local`` invokes the
    shell as ``[shell, "-lic", "set +m; <cmd>"]``, and that ``-lic`` bundle +
    ``set +m`` job-control syntax is NOT understood by fish, csh/tcsh,
    nushell, elvish, xonsh, etc.  Returning such a ``$SHELL`` would trade the
    bash-3.2 swallow for a parse error on every background command, so for any
    non-allowlisted shell we fall back to ``_find_bash`` (the prior behaviour).

    On Windows, ``$SHELL`` is typically bash (Git Bash), so behaviour is
    unchanged — we fall through to ``_find_bash``.
    """
    if not _IS_WINDOWS:
        user_shell = os.environ.get("SHELL")
        if (
            user_shell
            and os.path.isfile(user_shell)
            and os.access(user_shell, os.X_OK)
            and Path(user_shell).name in _SPAWN_COMPATIBLE_SHELLS
        ):
            return user_shell
    return _find_bash()


# Standard PATH entries for environments with minimal PATH.
_SANE_PATH = (
    "/opt/homebrew/bin:/opt/homebrew/sbin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

# Cached directory containing the ``hermes`` console-script.
# ``_SENTINEL`` distinguishes "not resolved yet" from a resolved ``None``.
_SENTINEL = object()
_HERMES_BIN_DIR: "str | None | object" = _SENTINEL


def _resolve_hermes_bin_dir() -> str | None:
    """Return the directory holding the ``hermes`` console-script, or None.

    The terminal tool runs in a freshly-spawned subshell whose PATH is the
    agent process's PATH plus a static set of system dirs (``_SANE_PATH``).
    When the gateway is launched by something that does NOT source the user's
    shell rc — systemd, a service manager, a desktop launcher, cron — the
    hermes install dir (``~/.local/bin``, the venv ``bin``/``Scripts``, pipx,
    nix) is absent from that PATH, so plugins shelling out to bare ``hermes``
    via the terminal tool hit ``command not found`` (exit 127) even though
    ``hermes`` works fine in the user's own interactive terminal.

    We resolve the install dir once (it never changes within a process) and
    prepend-if-missing it to the subshell PATH so bare ``hermes`` resolves
    regardless of how the gateway was started.

    Resolution order (cheap, no heavy imports):
      1. ``shutil.which("hermes")`` — normal PATH-installed shim.
      2. The directory of ``sys.argv[0]`` when it's an absolute path to a
         real ``hermes`` executable (covers nix-store / venv wrappers).
      3. The directory of ``sys.executable`` — the running interpreter's
         venv ``bin``/``Scripts`` is where its console-scripts live.
    """
    global _HERMES_BIN_DIR
    if _HERMES_BIN_DIR is not _SENTINEL:
        return _HERMES_BIN_DIR  # type: ignore[return-value]

    candidate: str | None = None

    which = shutil.which("hermes")
    if which:
        candidate = os.path.dirname(which)

    if candidate is None:
        argv0 = sys.argv[0] if sys.argv else ""
        base = os.path.basename(argv0).lower()
        if (
            os.path.isabs(argv0)
            and (base == "hermes" or base.startswith("hermes."))
            and os.path.isfile(argv0)
        ):
            candidate = os.path.dirname(argv0)

    if candidate is None:
        exe_dir = os.path.dirname(sys.executable) if sys.executable else ""
        if exe_dir:
            shim = "hermes.exe" if _IS_WINDOWS else "hermes"
            if os.path.isfile(os.path.join(exe_dir, shim)):
                candidate = exe_dir

    if candidate and not os.path.isdir(candidate):
        candidate = None

    _HERMES_BIN_DIR = candidate
    return candidate


def _prepend_hermes_bin_dir(existing_path: str) -> str:
    """Prepend the hermes install dir to ``existing_path`` if it's missing.

    Cross-platform (uses ``os.pathsep``). First-occurrence wins, so a PATH
    that already contains the dir is returned unchanged. Returns the input
    unchanged when the install dir can't be resolved.
    """
    bin_dir = _resolve_hermes_bin_dir()
    if not bin_dir:
        return existing_path
    sep = os.pathsep
    entries = [e for e in existing_path.split(sep) if e] if existing_path else []
    if bin_dir in entries:
        return existing_path
    return sep.join([bin_dir, *entries])


def _append_missing_sane_path_entries(existing_path: str) -> str:
    """Return a normalised POSIX PATH with missing sane entries appended.

    On POSIX the caller-supplied PATH is rewritten (not merely appended to):
    empty entries and duplicate entries are dropped, preserving
    first-occurrence order, then each missing ``_SANE_PATH`` entry is appended
    once at the end so existing entries keep their precedence.

    Two intentional normalisations beyond the bare "add Homebrew dirs" fix:

    - **Empty entries are stripped.** A leading/trailing/double ``:`` encodes
      an empty PATH element, which POSIX shells interpret as the current
      working directory — a mild foot-gun in a default terminal environment.
      We drop these rather than carry them through.
    - **Duplicates are collapsed** (first occurrence wins), so a caller PATH
      that already contains repeats is not propagated verbatim.

    For a well-formed PATH (no empties, no duplicates) the leading segment is
    byte-identical to the input and ordering is preserved; only the missing
    sane entries are appended. On Windows this is a no-op passthrough (the
    separator is ``;`` and the native PATH must not be touched).
    """
    if _IS_WINDOWS:
        return existing_path

    sane_entries = [entry for entry in _SANE_PATH.split(":") if entry]
    if not existing_path:
        return ":".join(sane_entries)

    # De-duplicate the caller PATH (first occurrence wins) and drop empty
    # entries before merging in the sane fallbacks.
    seen: set[str] = set()
    ordered_entries: list[str] = []
    for entry in existing_path.split(":"):
        if not entry or entry in seen:
            continue
        seen.add(entry)
        ordered_entries.append(entry)

    # _SANE_PATH is a static, duplicate-free constant, so a membership check
    # against the caller entries is sufficient — no need to track `seen` here.
    for entry in sane_entries:
        if entry not in seen:
            ordered_entries.append(entry)

    return ":".join(ordered_entries)


def _apply_windows_msys_bash_env_defaults(env: dict) -> None:
    """Disable MSYS argument path conversion for Git Bash subprocesses.

    Git Bash rewrites arguments that look like Unix paths (``/FO``, ``/TN``,
    ``/Create``) into ``C:/.../git/FO``-style paths, which breaks native
    Windows commands such as ``tasklist``, ``schtasks``, and ``wmic``.  Hermes
    runs terminal commands through bash on Windows, so set the standard MSYS
    opt-out by default.  Users who need conversion can override in their env.
    Refs #56700.

    ``MSYS_NO_PATHCONV`` is honored by Git for Windows bash only.  MSYS2-proper
    and Cygwin bash (which ``_find_bash`` can still return via the final
    ``shutil.which`` fallback) ignore it and honor ``MSYS2_ARG_CONV_EXCL``
    instead, so set both.  ``*`` disables all argv conversion — the semantic
    equivalent of ``MSYS_NO_PATHCONV=1``.  Also fixes ``cmd /c`` mangling
    (#56147).
    """
    if not _IS_WINDOWS:
        return
    env.setdefault("MSYS_NO_PATHCONV", "1")
    env.setdefault("MSYS2_ARG_CONV_EXCL", "*")


def _path_env_key(run_env: dict) -> str | None:
    """Return the PATH env key to update without altering Windows casing.

    Note: this is deliberately a *second* Windows guard, distinct from the
    early-return in ``_append_missing_sane_path_entries``. Its job is to pick
    the correctly-cased key (``Path`` vs ``PATH``) so completion writes back to
    the key the caller already used; the helper's guard makes that helper safe
    to call standalone (it is, e.g. in the Windows unit tests). Both are
    intentional.
    """
    if not _IS_WINDOWS:
        return "PATH"
    for key in run_env:
        if key.upper() == "PATH":
            return key
    return None


def _make_run_env(env: dict) -> dict:
    """Build a run environment with a sane PATH and provider-var stripping."""
    try:
        from tools.env_passthrough import is_env_passthrough as _is_passthrough
    except Exception:
        _is_passthrough = lambda _: False  # noqa: E731

    merged = dict(os.environ | env)
    run_env = {}
    for k, v in merged.items():
        if k.startswith(_HERMES_PROVIDER_ENV_FORCE_PREFIX):
            real_key = k[len(_HERMES_PROVIDER_ENV_FORCE_PREFIX):]
            if _is_hermes_internal_secret(real_key):
                continue
            run_env[real_key] = v
        elif _is_hermes_internal_secret(k):
            continue
        elif k not in _HERMES_PROVIDER_ENV_BLOCKLIST or _is_passthrough(k):
            run_env[k] = v
    path_key = _path_env_key(run_env)
    if path_key is not None:
        new_path = _append_missing_sane_path_entries(run_env.get(path_key, ""))
        # On Windows, ensure Git Bash's coreutils dirs (…\usr\bin etc.) are on
        # PATH.  A non-login ``bash -c`` fallback (used when ``bash -l`` is
        # broken) never sources /etc/profile, so without this cat/mktemp/mv and
        # friends are missing and every write_file/terminal call fails (empty
        # error / exit 127).  No-op off Windows and when a login snapshot is
        # healthy (the snapshot re-exports the full PATH inside the shell).
        new_path = _prepend_git_bash_dirs(new_path)
        # Ensure the hermes install dir is reachable so plugins can shell out
        # to bare ``hermes`` via the terminal tool even when the gateway was
        # launched without it on PATH (systemd, service managers, cron, etc.).
        run_env[path_key] = _prepend_hermes_bin_dir(new_path)

    _inject_context_hermes_home(run_env)

    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(run_env)

    # Bridge ContextVar-based session vars into the subprocess env (with the
    # cross-session leak guard — strips _UNSET vars when a concurrent host is
    # engaged so a sibling session's os.environ mirror can't leak in).
    _inject_session_context_env(run_env)
    # The generic terminal path is model-controlled shell. Connector bearer
    # must only flow through a dedicated allowlisted connector runner, not via
    # Popen env or the shared shell snapshot.
    _apply_profile_secret_scope_env(run_env, inject=True)

    for _marker in _ACTIVE_VENV_MARKER_VARS:
        run_env.pop(_marker, None)

    _apply_windows_msys_bash_env_defaults(run_env)

    return run_env


def _read_terminal_shell_init_config() -> tuple[list[str], bool]:
    """Return (shell_init_files, auto_source_bashrc) from config.yaml.

    Best-effort — returns sensible defaults on any failure so terminal
    execution never breaks because the config file is unreadable.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        terminal_cfg = cfg.get("terminal") or {}
        files = terminal_cfg.get("shell_init_files") or []
        if not isinstance(files, list):
            files = []
        auto_bashrc = bool(terminal_cfg.get("auto_source_bashrc", True))
        return [str(f) for f in files if f], auto_bashrc
    except Exception:
        return [], True


def _resolve_shell_init_files() -> list[str]:
    """Resolve the list of files to source before the login-shell snapshot.

    Expands ``~`` and ``${VAR}`` references and drops anything that doesn't
    exist on disk, so a missing ``~/.bashrc`` never breaks the snapshot.
    The ``auto_source_bashrc`` path runs only when the user hasn't supplied
    an explicit list — once they have, Hermes trusts them.
    """
    explicit, auto_bashrc = _read_terminal_shell_init_config()

    candidates: list[str] = []
    if explicit:
        candidates.extend(explicit)
    elif auto_bashrc and not _IS_WINDOWS:
        # Build a login-shell-ish source list so tools like n / nvm / asdf /
        # pyenv that self-install into the user's shell rc land on PATH in
        # the captured snapshot.
        #
        # ~/.profile and ~/.bash_profile run first because they have no
        # interactivity guard — installers like ``n`` and ``nvm`` append
        # their PATH export there on most distros, and a non-interactive
        # ``. ~/.profile`` picks that up.
        #
        # ~/.bashrc runs last. On Debian/Ubuntu the default bashrc starts
        # with ``case $- in *i*) ;; *) return;; esac`` and exits early
        # when sourced non-interactively, which is why sourcing bashrc
        # alone misses nvm/n PATH additions placed below that guard. We
        # still include it so users who put PATH logic in bashrc (and
        # stripped the guard, or never had one) keep working.
        candidates.extend(["~/.profile", "~/.bash_profile", "~/.bashrc"])

    resolved: list[str] = []
    for raw in candidates:
        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            continue
        if path and os.path.isfile(path):
            resolved.append(path)
    return resolved


def _prepend_shell_init(cmd_string: str, files: list[str]) -> str:
    """Prepend ``source <file>`` lines (guarded + silent) to a bash script.

    Each file is wrapped so a failing rc file doesn't abort the whole
    bootstrap: ``set +e`` keeps going on errors, ``2>/dev/null`` hides
    noisy prompts, and ``|| true`` neutralises the exit status.
    """
    if not files:
        return cmd_string

    prelude_parts = ["set +e"]
    for path in files:
        # shlex.quote isn't available here without an import; the files list
        # comes from os.path.expanduser output so it's a concrete absolute
        # path.  Escape single quotes defensively anyway.
        safe = path.replace("'", "'\\''")
        prelude_parts.append(f"[ -r '{safe}' ] && . '{safe}' 2>/dev/null || true")
    prelude = "\n".join(prelude_parts) + "\n"
    return prelude + cmd_string


class LocalEnvironment(BaseEnvironment):
    """Run commands directly on the host machine.

    Spawn-per-call: every execute() spawns a fresh bash process.
    Session snapshot preserves env vars across calls.
    CWD persists via file-based read after each command.
    """

    def __init__(self, cwd: str = "", timeout: int = 60, env: dict = None):
        cwd = _resolve_local_initial_cwd(cwd)
        super().__init__(cwd=cwd, timeout=timeout, env=env)
        self.init_session()

    def _snapshot_ephemeral_env_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                PROFILE_SCOPED_SUBPROCESS_ENV_KEYS
                | {"ZETTLAB_TURN_ID"}
                | set(super()._snapshot_ephemeral_env_keys())
            )
        )

    def _snapshot_ephemeral_env_exports(self) -> list[str]:
        """Restore live task-local context after sourcing the shell snapshot.

        A LocalEnvironment persists exported variables between terminal calls.
        Session and turn identity must not persist that way: a later request
        can reuse the environment while carrying a different ContextVar set.
        """
        exports = super()._snapshot_ephemeral_env_exports()
        public_env: dict[str, str] = {}
        _apply_profile_secret_scope_env(public_env, inject=True)
        for key in sorted(PROFILE_PUBLIC_RUNTIME_ENV_KEYS):
            value = public_env.get(key)
            if value is not None:
                exports.append(f"export {key}={shlex.quote(value)}")
        return exports

    def _wrap_command(self, command: str, cwd: str) -> str:
        run_env = _make_run_env(self.env)
        effective_cwd = _managed_terminal_cwd(cwd, env=run_env)
        _prepare_managed_command_skill_sources(command, run_env)
        return super()._wrap_command(
            _with_zettlab_turn_id(command),
            effective_cwd,
        )

    def get_temp_dir(self) -> str:
        """Return a shell-safe writable temp dir for local execution.

        Termux does not provide /tmp by default, but exposes a POSIX TMPDIR.
        Prefer POSIX-style env vars when available, keep using /tmp on regular
        Unix systems, and only fall back to tempfile.gettempdir() when it also
        resolves to a POSIX path.

        Check the environment configured for this backend first so callers can
        override the temp root explicitly (for example via terminal.env or a
        custom TMPDIR), then fall back to the host process environment.

        **Windows:** hardcoded ``/tmp`` is wrong in two ways — native Python
        can't open the path, and the Windows default temp (``%TEMP%``) often
        contains spaces (``C:\\Users\\Some Name\\AppData\\Local\\Temp``) that
        break unquoted bash interpolations.  Use a dedicated cache dir under
        ``HERMES_HOME`` instead — single-word path, guaranteed to exist, same
        string resolves in both Git Bash and native Python.
        """
        if _IS_WINDOWS:
            # Derive a Windows-safe temp dir under HERMES_HOME.  Using
            # forward slashes makes the same string work unchanged in bash
            # command interpolations AND in Python ``open()`` — Windows
            # accepts forward slashes in filesystem paths, and we control
            # the path so we can guarantee no spaces.
            try:
                from hermes_constants import get_hermes_home
                cache_dir = get_hermes_home() / "cache" / "terminal"
            except Exception:
                cache_dir = Path(tempfile.gettempdir()) / "hermes_terminal"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Force forward slashes so the same string serves both contexts.
            return str(cache_dir).replace("\\", "/")

        for env_var in ("TMPDIR", "TMP", "TEMP"):
            candidate = self.env.get(env_var) or os.environ.get(env_var)
            if candidate and candidate.startswith("/"):
                return candidate.rstrip("/") or "/"

        if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK | os.X_OK):
            return "/tmp"

        candidate = tempfile.gettempdir()
        if candidate.startswith("/"):
            return candidate.rstrip("/") or "/"

        return "/tmp"

    @staticmethod
    def _quote_cwd_for_cd(cwd: str) -> str:
        """Use native paths for Python, but Git Bash-friendly paths for cd."""
        return BaseEnvironment._quote_cwd_for_cd(_windows_to_msys_path(cwd))

    def _quote_shell_path(self, path: str) -> str:
        """Rewrite native/mixed Windows paths before quoting for Git Bash."""
        return _quote_bash_path(path)

    def _run_bash(self, cmd_string: str, *, login: bool = False,
                  timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        bash = _find_bash()
        # For login-shell invocations (used by init_session to build the
        # environment snapshot), prepend sources for the user's bashrc /
        # custom init files so tools registered outside bash_profile
        # (nvm, asdf, pyenv, …) end up on PATH in the captured snapshot.
        # Non-login invocations are already sourcing the snapshot and
        # don't need this.
        if login:
            init_files = _resolve_shell_init_files()
            if init_files:
                cmd_string = _prepend_shell_init(cmd_string, init_files)
        run_env = _make_run_env(self.env)
        managed_cwd = _managed_terminal_cwd(self.cwd, env=run_env)
        args = [bash, "-l", "-c", cmd_string] if login else [bash, "-c", cmd_string]
        args = _managed_terminal_argv(args, env=run_env)

        # Recover when the cwd has been deleted out from under us — usually by
        # a previous tool call that ran ``rm -rf`` on its own working dir
        # (issue #17558).  Popen would otherwise raise FileNotFoundError on
        # the cwd before bash starts, wedging every subsequent call until the
        # gateway restarts.
        #
        # On Windows, ``_resolve_safe_cwd`` also normalises Git Bash-style
        # POSIX paths (``/c/Users/...``) to native form so a perfectly valid
        # ``pwd -P`` result from bash isn't mistakenly treated as "missing"
        # and spammed as a warning on every command.
        safe_cwd = _resolve_safe_cwd(managed_cwd)
        if safe_cwd != managed_cwd:
            # MSYS → Windows translation alone shouldn't surface as a warning
            # (it's a benign normalization, not a recovery). Only warn when
            # the directory really doesn't exist on disk.
            normalized = (
                _msys_to_windows_path(managed_cwd)
                if _IS_WINDOWS
                else managed_cwd
            )
            if safe_cwd != normalized:
                logger.warning(
                    "LocalEnvironment cwd %r is missing on disk; "
                    "falling back to %r so terminal commands keep working.",
                    managed_cwd,
                    safe_cwd,
                )
            if managed_cwd == self.cwd:
                self.cwd = safe_cwd

        _popen_cwd = safe_cwd

        _popen_kwargs = {"creationflags": windows_hide_flags()} if _IS_WINDOWS else {}

        proc = subprocess.Popen(
            args,
            text=True,
            env=run_env,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            start_new_session=True,
            cwd=_popen_cwd,
            **_popen_kwargs,
        )
        if not _IS_WINDOWS:
            try:
                proc._hermes_pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                pass

        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)

        return proc

    def _kill_process(self, proc):
        """Kill the entire process group (all children)."""

        def _group_alive(pgid: int) -> bool:
            try:
                # POSIX-only: _IS_WINDOWS is handled before this helper is used.
                os.killpg(pgid, 0)  # windows-footgun: ok — POSIX process-group alive probe
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # The group exists, even if this process cannot signal it.
                return True

        def _wait_for_group_exit(pgid: int, timeout: float) -> bool:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                # Reap the wrapper promptly. A dead but unreaped group leader
                # still makes killpg(pgid, 0) report the group as alive.
                try:
                    proc.poll()
                except Exception:
                    pass
                if not _group_alive(pgid):
                    return True
                time.sleep(0.05)
            try:
                proc.poll()
            except Exception:
                pass
            return not _group_alive(pgid)

        try:
            if _IS_WINDOWS:
                try:
                    from gateway.status import terminate_pid

                    terminate_pid(proc.pid, force=True)
                except Exception:
                    proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            else:
                try:
                    pgid = os.getpgid(proc.pid)
                except ProcessLookupError:
                    pgid = getattr(proc, "_hermes_pgid", None)
                    if pgid is None:
                        raise

                try:
                    os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX process-group SIGTERM (guarded by _IS_WINDOWS above)
                except ProcessLookupError:
                    return

                # Wait on the process group, not just the shell wrapper. Under
                # load the wrapper can exit before grandchildren do; returning
                # at that point leaves orphaned process-group members behind.
                if _wait_for_group_exit(pgid, 1.0):
                    return

                try:
                    # POSIX-only: _IS_WINDOWS is handled by the outer branch.
                    os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX process-group SIGKILL
                except ProcessLookupError:
                    return
                _wait_for_group_exit(pgid, 2.0)
                try:
                    proc.wait(timeout=0.2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    def _update_cwd(self, result: dict):
        """Update cwd from the stdout marker emitted by the wrapped command.

        The base command wrapper already appends ``pwd -P`` to stdout inside a
        session-specific marker, so the local backend can share the same parser
        as remote backends instead of re-reading the temp file it just wrote.
        ``_extract_cwd_from_output`` keeps the local Windows normalization and
        stale-path rollback semantics intact.
        """
        self._extract_cwd_from_output(result)

    def _extract_cwd_from_output(self, result: dict):
        """Same semantics as the base class, but on Windows the value
        emitted by ``pwd -P`` inside Git Bash is in MSYS form
        (``/c/Users/x``). Normalize to native Windows form and validate
        the directory exists before assigning to ``self.cwd`` — otherwise
        ``_run_bash``'s safe-cwd recovery would warn on every subsequent
        command.

        Always defers to the base class for stripping the marker text from
        ``result["output"]`` so output formatting is identical.
        """
        # Snapshot pre-existing cwd, defer to base for parsing + marker
        # stripping, then validate / normalize whatever it assigned.
        prev_cwd = self.cwd
        super()._extract_cwd_from_output(result)
        if self.cwd != prev_cwd:
            normalized = _msys_to_windows_path(self.cwd) if _IS_WINDOWS else self.cwd
            if normalized and os.path.isdir(normalized):
                self.cwd = normalized
            else:
                # Stale / non-existent path — keep previous cwd; _run_bash
                # will resolve a safe fallback on the next call if needed.
                self.cwd = prev_cwd

    def cleanup(self):
        """Clean up temp files."""
        for f in (self._snapshot_path, self._cwd_file):
            try:
                os.unlink(f)
            except OSError:
                pass
        # Remove any orphaned atomic-write temp snapshots (snap.tmp.<bashpid>)
        # a failed/interrupted mv could have left behind (#38249).
        try:
            import glob
            for tmp in glob.glob(f"{self._snapshot_path}.tmp.*"):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        except Exception:
            pass
