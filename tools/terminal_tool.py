#!/usr/bin/env python3
"""
Terminal Tool Module

A terminal tool that executes commands in local, Docker, Modal, SSH,
Singularity, Daytona, and Vercel Sandbox environments. Supports local
execution, containerized backends, and cloud sandboxes, including managed
Modal mode.

Environment Selection (via TERMINAL_ENV environment variable):
- "local": Execute directly on the host machine (default, fastest)
- "docker": Execute in Docker containers (isolated, requires Docker)
- "modal": Execute in Modal cloud sandboxes (direct Modal or managed gateway)
- "vercel_sandbox": Execute in Vercel Sandbox cloud sandboxes

Features:
- Multiple execution backends (local, docker, modal, vercel_sandbox)
- Background task support
- VM/container lifecycle management
- Automatic cleanup after inactivity

Cloud sandbox note:
- Persistent filesystems preserve working state across sandbox recreation
- Persistent filesystems do NOT guarantee the same live sandbox or long-running processes survive cleanup, idle reaping, or Hermes exit

Usage:
    from terminal_tool import terminal_tool

    # Execute a simple command
    result = terminal_tool("ls -la")

    # Execute in background
    result = terminal_tool("python server.py", background=True)
"""

import array
import builtins
import errno
import gc
import importlib.util
import hashlib
import json
import logging
import os
import platform
import re
import select
import shlex
import signal
import stat
import time
import threading
import atexit
import shutil
import socket
import struct
import subprocess
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Mapping

from utils import env_var_enabled

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global interrupt event: set by the agent when a user interrupt arrives.
# The terminal tool polls this during command execution so it can kill
# long-running subprocesses immediately instead of blocking until timeout.
# ---------------------------------------------------------------------------
from tools.interrupt import is_interrupted, _interrupt_event  # noqa: F401 — re-exported
from tools.registry import tool_error
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)




# =============================================================================
# Custom Singularity Environment with more space
# =============================================================================

# Singularity helpers (scratch dir, SIF cache) now live in tools/environments/singularity.py
from tools.environments.singularity import _get_scratch_dir
from tools.process_security import harden_sensitive_process
from tools.tool_backend_helpers import (
    coerce_modal_mode,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_modal_backend_state,
)


def _safe_parse_import_env(
    name: str,
    default: Any,
    converter,
    type_label: str,
):
    """Parse module-level numeric env vars without breaking import.

    Terminal tool is imported by CLI, ACP, tests, and tool discovery. A single
    malformed env var must not make the whole module unloadable at import time.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name,
            raw,
            type_label,
            default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env(
    "TERMINAL_MAX_FOREGROUND_TIMEOUT",
    600,
    int,
    "integer",
)

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env(
    "TERMINAL_DISK_WARNING_GB",
    500.0,
    float,
    "number",
)
_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"
_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = (config.get("vercel_runtime") or "").strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error(
            "Vercel Sandbox runtime %r is not supported. "
            "Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
            runtime,
            supported,
        )
        return False

    disk = config.get("container_disk", 51200)
    if disk not in {0, 51200}:
        logger.error(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
            "Use the default shared setting (51200 MB).",
            disk,
        )
        return False

    if importlib.util.find_spec("vercel") is None:
        logger.error(
            "vercel is required for the Vercel Sandbox terminal backend: pip install vercel"
        )
        return False

    from agent.secret_scope import get_secret

    has_oidc = bool(get_secret("VERCEL_OIDC_TOKEN"))
    has_token = bool(get_secret("VERCEL_TOKEN"))
    has_project = bool(get_secret("VERCEL_PROJECT_ID"))
    has_team = bool(get_secret("VERCEL_TEAM_ID"))

    if has_oidc:
        return True

    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error(
            "Vercel Sandbox backend selected with token auth, but "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all "
            "be set together. VERCEL_OIDC_TOKEN is supported for one-off "
            "local development only."
        )
        return False

    logger.error(
        "Vercel Sandbox backend selected but no supported auth configuration "
        "was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID "
        "for normal use. VERCEL_OIDC_TOKEN is supported for one-off local "
        "development only."
    )
    return False


# Cache for disk usage warning to avoid full rglob scan on every call.
# The check is advisory-only — staleness for up to 5 minutes is acceptable.
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}
_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _check_disk_usage_warning():
    """Check if total disk usage exceeds warning threshold.

    Result is cached for :data:`_DISK_USAGE_CACHE_TTL` seconds (default:
    5 minutes) to avoid an expensive recursive filesystem scan on every
    terminal command.  The check is advisory-only so a stale result is
    harmless.
    """
    import time as _time_mod
    now = _time_mod.monotonic()
    if now - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        scratch_dir = _get_scratch_dir()

        # Get total size of hermes directories
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except OSError as e:
                        logger.debug("Could not stat file %s: %s", f, e)
        
        total_gb = total_bytes / (1024 ** 3)
        
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = _time_mod.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception as e:
        logger.debug("Disk usage warning check failed: %s", e, exc_info=True)
        # Don't update cache on error so the next call retries.
        return False


# Interactive sudo password cache.
#
# Scope the cache to the active session when a session key is available, then
# fall back to callback identity (ACP / CLI interactive callbacks), then the
# current thread. This prevents one interactive session from reusing another
# session's cached sudo password inside the same long-lived process.
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()

# Optional UI callbacks for interactive prompts. When set, these are called
# instead of the default /dev/tty or input() readers. The CLI registers these
# so prompts route through prompt_toolkit's event loop.
# Callback slots used by the approval prompt and sudo password prompt
# routines. Stored in thread-local state so overlapping ACP sessions —
# each running in its own ThreadPoolExecutor thread — don't stomp on
# each other's callbacks. See GHSA-qg5c-hvr5-hjgr.
#
# CLI mode is single-threaded, so each thread (the only one) holds its
# own callback exactly like before. Gateway mode resolves approvals via
# the per-session queue in tools.approval, not through these callbacks,
# so it's unaffected.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register a callback for sudo password prompts (used by CLI).

    Per-thread scope — ACP sessions that run concurrently in a
    ThreadPoolExecutor each have their own callback slot.
    """
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register a callback for dangerous command approval prompts.

    Per-thread scope — ACP sessions that run concurrently in a
    ThreadPoolExecutor each have their own callback slot. See
    GHSA-qg5c-hvr5-hjgr.
    """
    _callback_tls.approval = cb


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    try:
        from gateway.session_context import get_session_env

        session_key = get_session_env("HERMES_SESSION_KEY", "")
    except Exception:
        session_key = os.getenv("HERMES_SESSION_KEY", "")
    if session_key:
        return f"session:{session_key}"

    callback = _get_sudo_password_callback()
    if callback is not None:
        owner = getattr(callback, "__self__", None)
        func = getattr(callback, "__func__", None)
        if owner is not None and func is not None:
            return f"callback-owner:{id(owner)}:{id(func)}"
        return f"callback:{id(callback)}"

    return f"thread:{threading.get_ident()}"


def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)


def _reset_cached_sudo_passwords() -> None:
    """Clear all cached sudo passwords.

    Internal helper for tests and process teardown paths.
    """
    with _sudo_password_cache_lock:
        _sudo_password_cache.clear()

# =============================================================================
# Dangerous Command Approval System
# =============================================================================

# Dangerous command detection + approval now consolidated in tools/approval.py
from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False

    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False) -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access)


# Allowlist: characters that can legitimately appear in directory paths.
# Covers Unicode letters/digits, path separators, Windows drive/UNC separators,
# tilde, dot, hyphen, underscore, space, plus, at, equals, and comma.  Shell
# metacharacters remain rejected.  This intentionally fixes the old ASCII-only
# guard that blocked perfectly normal workdirs such as Chinese Obsidian vault
# paths while preserving the injection boundary around command execution
# (the cwd is additionally shlex-quoted before it reaches the shell; this
# allowlist is defense-in-depth).
_WORKDIR_SAFE_ASCII_CHARS = frozenset('/\\:_-.~ +@=,')


def _is_safe_workdir_char(ch: str) -> bool:
    if not ch:
        return False
    # Reject control characters (including newlines/tabs) and NUL bytes before
    # considering Unicode categories.
    if ord(ch) < 32 or ord(ch) == 127:
        return False
    return ch.isalnum() or ch in _WORKDIR_SAFE_ASCII_CHARS


def _validate_workdir(workdir: str) -> str | None:
    """Reject workdir values that don't look like a filesystem path.

    Uses an allowlist of safe characters rather than a deny-list, so novel
    shell metacharacters can't slip through.

    Returns None if safe, or an error message string if dangerous.
    """
    if not workdir:
        return None
    for ch in workdir:
        if not _is_safe_workdir_char(ch):
            return (
                f"Blocked: workdir contains disallowed character {repr(ch)}. "
                "Use a simple filesystem path without shell metacharacters."
            )
    return None


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """
    Check for sudo failure and add helpful message for messaging contexts.
    
    Returns enhanced output if sudo failed in messaging context, else original.
    """
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    
    if not is_gateway:
        return output
    
    # Check for sudo failure indicators
    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            from hermes_constants import display_hermes_home as _dhh
            return output + f"\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {_dhh()}/.env on the agent machine."
    
    return output


# sudo -S rejects a bad cached/interactive password with these messages.
_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a piped password."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


def _invalidate_cached_sudo_on_auth_failure(
    command: str | None, output: str
) -> bool:
    """Drop a session-cached sudo password after sudo rejects it.

    Env-configured ``SUDO_PASSWORD`` is left alone — that is an explicit
    operator choice, not an interactive cache entry.
    """
    if "SUDO_PASSWORD" in os.environ:
        return False
    if not _sudo_wrong_password_failure(output):
        return False
    if _count_real_sudo_invocations(command or "") == 0:
        return False
    if not _get_cached_sudo_password():
        return False
    _set_cached_sudo_password("")
    return True


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """
    Prompt user for sudo password with timeout.
    
    Returns the password if entered, or empty string if:
    - User presses Enter without input (skip)
    - Timeout expires (45s default)
    - Any error occurs
    
    Only works in interactive mode (HERMES_INTERACTIVE=1).
    If a _sudo_password_callback is registered (by the CLI), delegates to it
    so the prompt integrates with prompt_toolkit's UI.  Otherwise reads
    directly from /dev/tty with echo disabled.
    """
    import sys
    
    # Use the registered callback when available (prompt_toolkit-compatible)
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            return _sudo_cb() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password with echo disabled. Uses msvcrt on Windows, /dev/tty on Unix."""
        tty_fd = None
        old_attrs = None
        try:
            if platform.system() == "Windows":
                import msvcrt
                chars = []
                while True:
                    c = msvcrt.getwch()
                    if c in {"\r", "\n"}:
                        break
                    if c == "\x03":
                        raise KeyboardInterrupt
                    chars.append(c)
                result["password"] = "".join(chars)
            else:
                import termios
                tty_fd = os.open("/dev/tty", os.O_RDONLY)
                old_attrs = termios.tcgetattr(tty_fd)
                new_attrs = termios.tcgetattr(tty_fd)
                new_attrs[3] = new_attrs[3] & ~termios.ECHO
                termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
                chars = []
                while True:
                    b = os.read(tty_fd, 1)
                    if not b or b in {b"\n", b"\r"}:
                        break
                    chars.append(b)
                result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception as e:
                    logger.debug("Failed to restore terminal attributes: %s", e)
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception as e:
                    logger.debug("Failed to close tty fd: %s", e)
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]

def _safe_command_preview(command: Any, limit: int = 200) -> str:
    """Return a log-safe preview for possibly-invalid command values."""
    if command is None:
        return "<None>"
    if isinstance(command, str):
        return command[:limit]
    try:
        return repr(command)[:limit]
    except Exception:
        return f"<{type(command).__name__}>"

def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    i = start
    n = len(command)

    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ";|&()":
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == "\\" and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        i += 1

    return command[start:i], i


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite only real unquoted sudo command words, not plain text mentions.

    Returns the rewritten command and the number of sudo invocations rewritten.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    sudo_count = 0

    while i < n:
        ch = command[i]

        if ch.isspace():
            out.append(ch)
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            out.append(ch)
            i += 1
            command_start = True
            continue

        if ch == ")":
            out.append(ch)
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            out.append("sudo -S -p ''")
            sudo_count += 1
        else:
            out.append(token)

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return "".join(out), sudo_count


def _count_real_sudo_invocations(command: str) -> int:
    """Return how many real sudo command words appear in *command*.

    Lightweight scan that reuses the same tokeniser as
    ``_rewrite_real_sudo_invocations`` but skips the string-building, so it
    is cheap to call from the result-processing path.
    """
    count = 0
    i = 0
    n = len(command)
    command_start = True

    while i < n:
        ch = command[i]

        if ch.isspace():
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                break
            i = comment_end
            continue

        if command.startswith("&&", i) or command.startswith("||", i) or command.startswith(";;", i):
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            i += 1
            command_start = True
            continue

        if ch == ")":
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            count += 1

        if command_start and _looks_like_env_assignment(token):
            command_start = True
        else:
            command_start = False
        i = next_i

    return count


def _sudo_nopasswd_works() -> bool:
    """Return True when local sudo currently works without prompting.

    Only probes for the `local` terminal backend; Docker/SSH/Modal/etc. must
    not inherit the host's sudo state. Re-probes every call (no process-level
    cache) so an expired sudo timestamp cannot make a later command silently
    block waiting for a password.
    """
    terminal_env = os.getenv("TERMINAL_ENV", "local").strip().lower() or "local"
    if terminal_env != "local":
        return False

    try:
        probe = subprocess.run(
            ["sudo", "-n", "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _rewrite_compound_background(command: str) -> str:
    """Wrap `A && B &` (or `A || B &`) to `A && { B & }` at depth 0.

    Bash parses ``A && B &`` with `&&` tighter than `&`, so it forks a
    subshell for the whole `A && B` compound and backgrounds it. Inside
    the subshell, `B` runs foreground, so the subshell waits for `B` to
    finish. When `B` is a long-running process (`python3 -m http.server`,
    `yes > /dev/null`, anything that doesn't naturally exit), the subshell
    never exits. It leaks as a process stuck in ``wait4`` forever — and
    on the way, its open stdout pipe can prevent the terminal tool from
    returning promptly.

    Rewriting the tail to `A && { B & }` preserves `&&`'s error semantics
    (skip B if A fails) while replacing the subshell with a brace group.
    The brace group runs in the current shell (no fork), backgrounds B as
    a simple command (bash doesn't wait for it in non-interactive mode),
    and exits immediately. B runs as a normal backgrounded child, orphaned
    when the parent shell exits.

    Handles redirects (``&>``, ``2>&1``) and skips content inside quoted
    strings and parenthesised subshells. Leaves simple ``cmd &`` alone —
    that construct doesn't have the subshell-wait bug.
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    # Position in *command* just after the most recent `&&` / `||` at depth 0
    # in the current statement; -1 when no chain operator is active.
    last_chain_op_end = -1
    rewrites: list[tuple[int, int]] = []  # (chain_op_end, amp_pos)

    while i < n:
        ch = command[i]

        # Newline terminates a statement at depth 0 — reset chain state.
        # Checked before the whitespace skip so we don't miss it.
        if ch == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_op_end = -1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        # Comments (only at statement start — conservative: any `#` not inside
        # a token ends the line). `_read_shell_token` handles quoted strings
        # below so `#` inside quotes is safe.
        if ch == "#":
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        # Quoted tokens — consume whole string via the shared tokenizer.
        if ch in {"'", '"'}:
            _, next_i = _read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        # Brace groups: `{ ... }` is a group (no subshell fork), and bash
        # requires whitespace after `{`. We track depth so already-rewritten
        # output (`A && { B & }`) is idempotent — the inner `&` is part of
        # the group, not a new compound to rewrite. Also skip content inside
        # the group since `A && B &` there is separately well-formed.
        if ch == "{" and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == "\n"):
            brace_depth += 1
            i += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            # Closing a group completes a compound statement; reset chain.
            last_chain_op_end = -1
            i += 1
            continue

        # Inside parens or brace groups, skip operators — they parse in their
        # own scope. `(...)` subshells have the same bug class but are not the
        # common agent pattern; leave for a follow-up.
        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue

        # Chain operators at depth 0
        if command.startswith("&&", i) or command.startswith("||", i):
            last_chain_op_end = i + 2
            i += 2
            continue

        # Statement terminators reset the chain state
        if ch == ";":
            last_chain_op_end = -1
            i += 1
            continue

        # Single `|` (pipe) starts a new pipeline stage; don't rewrite
        # across it. `||` handled above.
        if ch == "|":
            last_chain_op_end = -1
            i += 1
            continue

        # `&` handling: distinguish `&&`, `&>`, fd redirect (`>&`, `<&`),
        # and a true backgrounding `&`.
        if ch == "&":
            # `&&` handled above; won't reach here
            if i + 1 < n and command[i + 1] == ">":
                # `&>` redirect — consume
                i += 2
                continue
            # `>&` / `<&` fd target — look back past whitespace
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                i += 1
                continue
            # Real background operator
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue

        # Regular unquoted token — advance past it via the shared tokenizer
        _, next_i = _read_shell_token(command, i)
        i = max(next_i, i + 1)

    if not rewrites:
        return command

    # Apply rewrites back-to-front so earlier indices remain valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        # Skip whitespace right after the `&&`/`||` so the brace group
        # opens flush against the inner command.
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]  # inner command + trailing space
        suffix = result[amp_pos + 1 :]
        # `{` needs a trailing space in bash; the closing `}` needs to be
        # preceded by `;` or `&` — we're providing `&` from the backgrounding.
        result = prefix + "{ " + middle + "& }" + suffix

    return result


def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """
    Transform sudo commands to use -S flag if SUDO_PASSWORD is available.

    This is a shared helper used by all execution environments to provide
    consistent sudo handling across local, SSH, and container environments.

    Returns:
        (transformed_command, sudo_stdin) where:
        - transformed_command has every bare ``sudo`` replaced with
          ``sudo -S -p ''`` so sudo reads its password from stdin.
        - sudo_stdin is the password string with a trailing newline that the
          caller must prepend to the process's stdin stream.  sudo -S reads
          exactly one line (the password) and passes the rest of stdin to the
          child command, so prepending is safe even when the caller also has
          its own stdin_data to pipe.
        - If no password is available, sudo_stdin is None and the command is
          returned unchanged so it fails gracefully with
          "sudo: a password is required".

    Callers that drive a subprocess directly (local, ssh, docker, singularity)
    should prepend sudo_stdin to their stdin_data and pass the merged bytes to
    Popen's stdin pipe.

    Callers that cannot pipe subprocess stdin (modal, daytona,
    vercel_sandbox) must embed the password in the command string
    themselves; see their execute() methods for how they handle the
    non-None sudo_stdin case.

    If SUDO_PASSWORD is not set and an interactive UI is available
    (HERMES_INTERACTIVE=1 or a registered sudo password callback):
      Prompts user for password with 45s timeout, caches for session.

    If SUDO_PASSWORD is not set and NOT interactive:
      Command runs as-is (fails gracefully with "sudo: a password is required").
    """
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    # Scope-aware read (Slack pattern): under multiplex the process env may
    # hold another profile's SUDO_PASSWORD, so honor the installed scope's
    # verdict; unscoped callers keep the legacy os.environ read.
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            _configured_password = get_secret("SUDO_PASSWORD")
        except UnscopedSecretError:
            _configured_password = os.environ.get("SUDO_PASSWORD")
    except Exception:
        _configured_password = os.environ.get("SUDO_PASSWORD")
    has_configured_password = _configured_password is not None
    sudo_password = (
        _configured_password
        if has_configured_password
        else _get_cached_sudo_password()
    )

    # Local hosts with sudoers NOPASSWD should not be forced through the
    # interactive Hermes password prompt or the sudo -S password-pipe path.
    # Scoped to the local terminal backend so Docker/SSH/Modal/etc. can't
    # inherit host sudo state. Re-probes every call (no process-lifetime
    # cache) so an expired sudo timestamp doesn't make a later command block
    # silently without Hermes prompting.
    if not has_configured_password and not sudo_password and _sudo_nopasswd_works():
        return command, None

    has_sudo_prompt_callback = _get_sudo_password_callback() is not None
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or has_sudo_prompt_callback
    )
    if not has_configured_password and not sudo_password and should_prompt_for_sudo:
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)

    if has_configured_password or sudo_password:
        # Trailing newline is required: sudo -S reads one line per invocation.
        # Compound commands (`sudo a && sudo b`) need one password line each.
        password_line = sudo_password + "\n"
        return transformed, password_line * sudo_count

    return command, None


# Environment classes now live in tools/environments/
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment as _ManagedModalEnvironment
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
import sys


_CONNECTOR_RUNTIME_SCRIPT = "connector_runtime.py"
_CONNECTOR_RUNTIME_MAX_SCRIPT_BYTES = 1024 * 1024
_CONNECTOR_RUNTIME_TRUST_MAX_PATHS = 8192
_CONNECTOR_RUNTIME_TRUST_MAX_BYTES = 64 * 1024 * 1024
_VIDEO_EDIT_RUNTIME_SCRIPTS = frozenset({
    "preference_resolver.py",
    "workflow_state.py",
    "cloud_render_business.py",
    "proactive_video.py",
    "normalize.py",
})
_CAMERA_RUNTIME_SCRIPT = "camera_connector.py"
_CAMERA_RUNTIME_RELATIVE_PATH = Path(
    "skills/camsnap/scripts/camera_connector.py"
)
_CAMERA_RUNTIME_MANIFEST_RELATIVE_PATH = Path("skills/camsnap/manifest.yaml")
_CAMERA_RUNTIME_CAPABILITY = "zettlab.camera.actions.v1"
_CAMERA_RUNTIME_MAX_MANIFEST_BYTES = 64 * 1024
_CAMERA_RUNTIME_MAX_TIMEOUT_SECONDS = 80
_CAMERA_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT = ";&|<>\n()"
_CONNECTOR_RUNTIME_SHELL_PUNCTUATION = set(_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT)
_CONNECTOR_RUNTIME_SHELL_GROUP_START = "{"
_CONNECTOR_RUNTIME_WRAPPERS = {
    "sudo",
    "env",
    "timeout",
    "exec",
    "nice",
    "nohup",
    "setsid",
    "stdbuf",
    "time",
    "command",
    "builtin",
}
_CONNECTOR_RUNTIME_WRAPPER_OPTIONS_WITH_ARG = {
    "sudo": {"-c", "--close-from", "-g", "--group", "-h", "--host", "-p", "--prompt", "-u", "--user"},
    "env": {"-a", "--argv0", "-C", "--chdir", "-S", "--split-string", "-u", "--unset"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "exec": {"-a"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-e", "--error", "-i", "--input", "-o", "--output"},
    "time": {"-f", "--format", "-o", "--output"},
}
_CONNECTOR_RUNTIME_COMMAND_SHELLS = {"bash", "dash", "sh", "zsh"}
_CONNECTOR_RUNTIME_SHELL_OPTIONS_WITH_ARG = {
    "+O",
    "+o",
    "-O",
    "-o",
    "--init-file",
    "--rcfile",
}
_CONNECTOR_RUNTIME_NESTED_SHELL_DEPTH = 8
_CONNECTOR_RUNTIME_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_CONNECTOR_RUNTIME_TIMEOUT_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[smhd])?")
_LARK_CLI_COMMAND = "lark-cli"
_LARK_CLI_MAX_TIMEOUT_SECONDS = 600
@dataclass(frozen=True)
class _ConnectorRuntimeRootAnchor:
    configured_root: Path
    resolved_root: Path
    identity: tuple[int, int]
    tree_digest: str
    file_digests: dict[str, str]


@dataclass(frozen=True)
class _ConnectorRuntimeCommand:
    argv: list[str]
    root_identity: tuple[int, int]
    script_identity: tuple[int, int]


@dataclass(frozen=True)
class _LarkCLICommand:
    args: list[str]


@dataclass(frozen=True)
class _TrustedWorkerModuleSnapshot:
    name: str
    path: str
    source: str


@dataclass(frozen=True)
class _TrustedRuntimeDirectorySnapshot:
    modules: tuple[_TrustedWorkerModuleSnapshot, ...]
    total_bytes: int


@dataclass(frozen=True)
class _TrustedWorkerSourceSnapshot:
    modules: tuple[_TrustedWorkerModuleSnapshot, ...]
    python_executable: str
    python_fingerprint: tuple[int, ...]
    worker_path: str


@dataclass(frozen=True)
class _TrustedWorkerFactoryImage:
    """Final in-process recovery root; service supervision owns gateway loss."""

    snapshot: _TrustedWorkerSourceSnapshot
    bootstrap_code: types.CodeType
    module_names: frozenset[str]
    owner_pid: int


@dataclass(frozen=True)
class _VideoEditWorkerProcessIdentity:
    """Kernel-backed identity for one process outside the gateway's child set."""

    pid: int
    start_time: Optional[int]
    pidfd: Optional[int]


@dataclass(frozen=True)
class _TrustedWorkerFactorySupervisor:
    """Disposable gateway child that owns and rebuilds the worker factory."""

    snapshot: _TrustedWorkerSourceSnapshot
    process: "_ForkedVideoEditWorkerSeed"
    channel: socket.socket
    owner_pid: int


@dataclass(frozen=True)
class _ForkedVideoEditWorkerSeed:
    """Popen-like identity handle for a process forked by the trusted tree."""

    pid: int
    identity: _VideoEditWorkerProcessIdentity
    direct_child: bool = False
    parent_control: Optional[str] = None

    def poll(self) -> Optional[int]:
        if self.direct_child:
            try:
                waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                return -getattr(signal, "SIGKILL", 9)
            if waited_pid == self.pid:
                return os.waitstatus_to_exitcode(status)
            return None
        if _video_edit_worker_process_identity_is_current(self.identity):
            return None
        return -getattr(signal, "SIGKILL", 9)

    def kill(self) -> bool:
        if self.direct_child:
            if self.poll() is not None:
                return False
            try:
                os.kill(self.pid, getattr(signal, "SIGKILL", 9))
                return True
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
        if _signal_video_edit_worker_process_identity(
            self.identity,
            getattr(signal, "SIGKILL", 9),
        ):
            return True
        return _request_video_edit_worker_parent_reap(
            self,
            parent_control=self.parent_control,
        )

    def wait(self, timeout: Optional[float] = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    "trusted video-edit worker seed",
                    float(timeout if timeout is not None else 0.0),
                )
            time.sleep(0.01)

    def close(self) -> None:
        _close_video_edit_worker_process_identity(self.identity)


_CONNECTOR_RUNTIME_ROOT_ANCHOR: Optional[_ConnectorRuntimeRootAnchor] = None
_VIDEO_EDIT_WORKER_BROKER_PID: Optional[int] = None
_VIDEO_EDIT_WORKER_BROKER_IDENTITY: Optional[_VideoEditWorkerProcessIdentity] = None
_VIDEO_EDIT_WORKER_CHANNEL: Optional[socket.socket] = None
_VIDEO_EDIT_WORKER_FACTORY_PROCESS: Optional[_ForkedVideoEditWorkerSeed] = None
_VIDEO_EDIT_WORKER_FACTORY_CHANNEL: Optional[socket.socket] = None
_VIDEO_EDIT_WORKER_SEED_PROCESS: Optional[_ForkedVideoEditWorkerSeed] = None
_VIDEO_EDIT_WORKER_SEED_CHANNEL: Optional[socket.socket] = None
_VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT: Optional[_TrustedWorkerSourceSnapshot] = None
_VIDEO_EDIT_WORKER_FACTORY_IMAGE: Optional[_TrustedWorkerFactoryImage] = None
_VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR: Optional[
    _TrustedWorkerFactorySupervisor
] = None
_VIDEO_EDIT_WORKER_LOCK = threading.RLock()
_VIDEO_EDIT_WORKER_IDLE_TIMER: Optional[threading.Timer] = None
_VIDEO_EDIT_WORKER_IDLE_GENERATION = 0
_VIDEO_EDIT_WORKER_IDLE_TIMEOUT_SECONDS = 30.0
_VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED = False
_VIDEO_EDIT_WORKER_MAX_FRAME_BYTES = 8 * 1024 * 1024
_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS = 10
_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS = 3
_VIDEO_EDIT_WORKER_FACTORY_SEED_START_MAX_ATTEMPTS = 2
_VIDEO_EDIT_WORKER_BROKER_START_MAX_ATTEMPTS = 2
_VIDEO_EDIT_WORKER_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_VIDEO_EDIT_UPLOAD_TIMEOUT_SECONDS = 3700
_PROACTIVE_VIDEO_UPLOAD_TIMEOUT_SECONDS = 10800
_VIDEO_EDIT_WORKER_SOURCE_LIMIT_BYTES = 512 * 1024
_VIDEO_EDIT_WORKER_INTERPRETER_LIMIT_BYTES = 32 * 1024 * 1024
_TRUSTED_RUNTIME_SOURCE_CACHE_MAX_BYTES = 8 * 1024 * 1024
_TRUSTED_RUNTIME_SOURCE_CACHE_MAX_DIRECTORIES = 128
# Leave two MiB of the worker's eight MiB IPC frame for JSON structure, argv,
# scoped env, and secrets. The encoded bundle check below handles escaping too.
_TRUSTED_RUNTIME_DIRECTORY_MAX_ENCODED_BYTES = 6 * 1024 * 1024
_TRUSTED_RUNTIME_SOURCE_CACHE: dict[
    tuple[str, tuple[int, int]],
    _TrustedRuntimeDirectorySnapshot,
] = {}
_TRUSTED_RUNTIME_SOURCE_CACHE_BYTES = 0
_VIDEO_EDIT_WORKER_MEMORY_BOOTSTRAP = r"""
import array
import json
import os
import select
import signal
import socket
import struct
import sys
import types

def _recv_exact(channel, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("trusted worker source channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

source_channel = socket.socket(fileno=int(sys.argv[1]))
supervisor_channel = socket.socket(fileno=int(sys.argv[2]))
interpreter_fd = int(sys.argv[3])
gateway_pid = int(sys.argv[4])
if (
    source_channel.fileno() <= 2
    or supervisor_channel.fileno() <= 2
    or source_channel.fileno() == supervisor_channel.fileno()
    or gateway_pid <= 1
):
    raise PermissionError("resident supervisor descriptor identity is invalid")
if interpreter_fd >= 0:
    os.close(interpreter_fd)
if os.getppid() != gateway_pid:
    raise PermissionError("resident supervisor parent identity changed during startup")
if sys.platform.startswith("linux") and hasattr(socket, "SO_PEERCRED"):
    peer_credentials = supervisor_channel.getsockopt(
        socket.SOL_SOCKET,
        socket.SO_PEERCRED,
        struct.calcsize("3i"),
    )
    peer_pid, _, _ = struct.unpack("3i", peer_credentials)
    if peer_pid != gateway_pid:
        raise PermissionError("resident supervisor control peer changed")
try:
    size = struct.unpack("!I", _recv_exact(source_channel, 4))[0]
    if size <= 0 or size > 1024 * 1024:
        raise ValueError("invalid trusted worker source frame")
    payload = json.loads(_recv_exact(source_channel, size).decode("utf-8"))
finally:
    source_channel.close()

modules = payload["modules"]
for item in modules:
    name = item["name"]
    if name not in {"process_security", "video_edit_runtime_worker"}:
        raise ValueError("unexpected trusted worker module")
    module = types.ModuleType(name)
    module.__file__ = item["path"]
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(item["source"], item["path"], "exec"), module.__dict__)

worker = sys.modules["video_edit_runtime_worker"]
worker_path = payload["worker_path"]
if not worker.harden_sensitive_process(no_new_privs=True, drop_ptrace=True):
    raise PermissionError("resident supervisor memory boundary is unavailable")
if not worker.enable_child_subreaper():
    raise PermissionError("resident supervisor subreaper boundary is unavailable")
memory_limit = worker.apply_worker_memory_limit(worker._MEMORY_LIMIT_BYTES)
if sys.platform.startswith("linux") and memory_limit.get("applied") is not True:
    raise PermissionError("resident supervisor memory limit is unavailable")
worker._preload_optional_runtime_modules()
for item in modules:
    item.clear()
modules.clear()
payload.clear()
sys.path[:] = []
sys.path_importer_cache.clear()
sys.meta_path[:] = [
    worker.importlib.machinery.BuiltinImporter,
    worker.importlib.machinery.FrozenImporter,
]

def _factory_loop(factory_channel, parent_guard, ready_channel, supervisor_pid):
    active_seed_pid = 0
    last_reaped_seed_pid = 0
    factory_pid = os.getpid()

    if not worker.bind_process_to_parent(
        supervisor_pid, death_signal=signal.SIGTERM
    ):
        raise PermissionError("factory parent-death boundary is unavailable")
    os.setsid()
    if not worker.harden_sensitive_process(no_new_privs=True, drop_ptrace=True):
        raise PermissionError("factory process memory boundary is unavailable")
    if not worker.enable_child_subreaper():
        raise PermissionError("factory child-subreaper boundary is unavailable")

    def _cleanup_seed(pid):
        nonlocal active_seed_pid, last_reaped_seed_pid
        if pid and active_seed_pid == pid:
            cleaned = worker._kill_and_reap_executor(pid)
            if cleaned:
                last_reaped_seed_pid = pid
                active_seed_pid = 0
            return cleaned
        if pid == last_reaped_seed_pid:
            return True
        if sys.platform.startswith("linux"):
            return worker._cleanup_linux_owned_processes(os.getpid(), None)
        return not pid

    def _refresh_seed_state():
        nonlocal active_seed_pid, last_reaped_seed_pid
        if active_seed_pid and worker._reap_exited_broker_leader(active_seed_pid):
            exited_pid = active_seed_pid
            cleaned = (
                not sys.platform.startswith("linux")
                or worker._cleanup_linux_owned_processes(os.getpid(), None)
            )
            if cleaned:
                active_seed_pid = 0
                last_reaped_seed_pid = exited_pid

    def _spawn_seed():
        nonlocal active_seed_pid
        _refresh_seed_state()
        if active_seed_pid:
            worker._send_seed_broker_response(
                factory_channel,
                {"seed_ready": False, "error": "factory already owns an active seed"},
                broker_fd=None,
            )
            return
        if sys.platform.startswith("linux") and not worker._cleanup_linux_owned_processes(
            os.getpid(), None
        ):
            worker._send_seed_broker_response(
                factory_channel,
                {"seed_ready": False, "error": "factory child invariant failed"},
                broker_fd=None,
            )
            return

        gateway_channel, seed_channel = socket.socketpair()
        try:
            seed_pid = os.fork()
        except OSError as exc:
            gateway_channel.close()
            seed_channel.close()
            worker._send_seed_broker_response(
                factory_channel,
                {"seed_ready": False, "error": f"{type(exc).__name__}: {exc}"},
                broker_fd=None,
            )
            return
        if seed_pid == 0:
            factory_channel.close()
            parent_guard.close()
            gateway_channel.close()
            try:
                if not worker.bind_process_to_parent(
                    factory_pid, death_signal=signal.SIGTERM
                ):
                    raise PermissionError("seed parent-death boundary is unavailable")
                os.setsid()
                sys.argv = [worker_path, str(seed_channel.fileno())]
                returncode = worker.main()
            except BaseException:
                returncode = 1
            finally:
                seed_channel.close()
            os._exit(returncode)

        seed_channel.close()
        try:
            gateway_channel.settimeout(10)
            ready = worker._recv_frame(gateway_channel)
            if ready.get("ready") is not True or ready.get("fork_seed") is not True:
                raise RuntimeError("trusted video-edit worker seed failed to initialize")
            gateway_channel.settimeout(None)
            active_seed_pid = seed_pid
            worker._send_seed_broker_response(
                factory_channel,
                {"seed_ready": True, "seed_pid": seed_pid, "ready": ready},
                broker_fd=gateway_channel.fileno(),
            )
        except Exception as exc:
            worker._kill_and_reap_executor(seed_pid)
            worker._send_seed_broker_response(
                factory_channel,
                {"seed_ready": False, "error": f"{type(exc).__name__}: {exc}"},
                broker_fd=None,
            )
        finally:
            gateway_channel.close()

    worker._send_frame(ready_channel, {
        "ready": True,
        "factory_ready": True,
        "fork_factory": True,
        "resident_image": True,
        "dumpable": 0 if sys.platform.startswith("linux") else None,
        "memory_limit": memory_limit,
        "accepts_secrets": False,
        "child_subreaper": True,
    })
    ready_channel.close()
    try:
        while True:
            readable, _, _ = select.select(
                [factory_channel, parent_guard],
                [],
                [],
                0.05 if active_seed_pid else None,
            )
            if parent_guard in readable:
                if not parent_guard.recv(1):
                    break
                raise PermissionError("factory parent guard received data")
            if factory_channel not in readable:
                _refresh_seed_state()
                continue
            operation = factory_channel.recv(1)
            if not operation:
                break
            if operation == b"N":
                _spawn_seed()
                continue
            if operation == b"T":
                requested_pid = struct.unpack(
                    "!Q", worker._recv_exact(factory_channel, 8)
                )[0]
                cleaned = _cleanup_seed(requested_pid)
                worker._send_frame(factory_channel, {
                    "cleanup": "stopped" if cleaned else "unknown",
                    "pid": requested_pid,
                    "reaped": cleaned,
                })
                continue
            if operation == b"Q":
                cleaned = _cleanup_seed(active_seed_pid)
                worker._send_frame(factory_channel, {"shutdown": cleaned})
                if cleaned:
                    break
                continue
            break
    finally:
        if active_seed_pid:
            _cleanup_seed(active_seed_pid)
        factory_channel.close()
        parent_guard.close()
    return 0

active_factory_pid = 0
last_reaped_factory_pid = 0
active_factory_guard = None
supervisor_pid = os.getpid()

def _close_factory_guard():
    global active_factory_guard
    guard = active_factory_guard
    active_factory_guard = None
    if guard is not None:
        guard.close()

def _cleanup_factory(pid):
    global active_factory_pid, last_reaped_factory_pid
    if pid and active_factory_pid == pid:
        cleaned = worker._kill_and_reap_executor(pid)
        if cleaned:
            last_reaped_factory_pid = pid
            active_factory_pid = 0
            _close_factory_guard()
        return cleaned
    if pid == last_reaped_factory_pid:
        return True
    if sys.platform.startswith("linux"):
        return worker._cleanup_linux_owned_processes(os.getpid(), None)
    return not pid

def _refresh_factory_state():
    global active_factory_pid, last_reaped_factory_pid
    if active_factory_pid and worker._reap_exited_broker_leader(active_factory_pid):
        exited_pid = active_factory_pid
        cleaned = (
            not sys.platform.startswith("linux")
            or worker._cleanup_linux_owned_processes(os.getpid(), None)
        )
        if cleaned:
            active_factory_pid = 0
            last_reaped_factory_pid = exited_pid
            _close_factory_guard()

def _spawn_factory():
    global active_factory_pid, active_factory_guard
    _refresh_factory_state()
    if active_factory_pid:
        worker._send_seed_broker_response(
            supervisor_channel,
            {"factory_spawned": False, "error": "supervisor already owns a factory"},
            broker_fd=None,
        )
        return
    if sys.platform.startswith("linux") and not worker._cleanup_linux_owned_processes(
        os.getpid(), None
    ):
        worker._send_seed_broker_response(
            supervisor_channel,
            {"factory_spawned": False, "error": "supervisor child invariant failed"},
            broker_fd=None,
        )
        return

    gateway_factory, factory_channel = socket.socketpair()
    supervisor_guard, factory_guard = socket.socketpair()
    ready_parent, ready_child = socket.socketpair()
    try:
        factory_pid = os.fork()
    except OSError as exc:
        for channel in (
            gateway_factory,
            factory_channel,
            supervisor_guard,
            factory_guard,
            ready_parent,
            ready_child,
        ):
            channel.close()
        worker._send_seed_broker_response(
            supervisor_channel,
            {"factory_spawned": False, "error": f"{type(exc).__name__}: {exc}"},
            broker_fd=None,
        )
        return
    if factory_pid == 0:
        supervisor_channel.close()
        gateway_factory.close()
        supervisor_guard.close()
        ready_parent.close()
        try:
            returncode = _factory_loop(
                factory_channel,
                factory_guard,
                ready_child,
                supervisor_pid,
            )
        except BaseException:
            returncode = 1
        finally:
            for channel in (factory_channel, factory_guard, ready_child):
                try:
                    channel.close()
                except OSError:
                    pass
        os._exit(returncode)

    factory_channel.close()
    factory_guard.close()
    ready_child.close()
    try:
        ready_parent.settimeout(10)
        ready = worker._recv_frame(ready_parent)
        if (
            ready.get("ready") is not True
            or ready.get("factory_ready") is not True
            or ready.get("resident_image") is not True
        ):
            raise RuntimeError("trusted video-edit worker factory failed to initialize")
        active_factory_pid = factory_pid
        active_factory_guard = supervisor_guard
        worker._send_seed_broker_response(
            supervisor_channel,
            {
                "factory_spawned": True,
                "factory_pid": factory_pid,
                "ready": ready,
            },
            broker_fd=gateway_factory.fileno(),
        )
    except Exception as exc:
        supervisor_guard.close()
        worker._kill_and_reap_executor(factory_pid)
        worker._send_seed_broker_response(
            supervisor_channel,
            {"factory_spawned": False, "error": f"{type(exc).__name__}: {exc}"},
            broker_fd=None,
        )
    finally:
        ready_parent.close()
        gateway_factory.close()

worker._send_frame(supervisor_channel, {
    "ready": True,
    "supervisor_ready": True,
    "resident_supervisor": True,
    "dumpable": 0 if sys.platform.startswith("linux") else None,
    "memory_limit": memory_limit,
    "accepts_secrets": False,
    "child_subreaper": True,
})
try:
    while True:
        readable, _, _ = select.select(
            [supervisor_channel], [], [], 0.05 if active_factory_pid else 0.25
        )
        if os.getppid() != gateway_pid:
            break
        if not readable:
            _refresh_factory_state()
            continue
        operation = supervisor_channel.recv(1)
        if not operation:
            break
        if operation == b"N":
            _spawn_factory()
            continue
        if operation == b"T":
            requested_pid = struct.unpack(
                "!Q", worker._recv_exact(supervisor_channel, 8)
            )[0]
            cleaned = _cleanup_factory(requested_pid)
            worker._send_frame(supervisor_channel, {
                "cleanup": "stopped" if cleaned else "unknown",
                "pid": requested_pid,
                "reaped": cleaned,
            })
            continue
        if operation == b"Q":
            cleaned = _cleanup_factory(active_factory_pid)
            worker._send_frame(supervisor_channel, {"shutdown": cleaned})
            if cleaned:
                break
            continue
        break
finally:
    if active_factory_pid:
        _cleanup_factory(active_factory_pid)
    _close_factory_guard()
    supervisor_channel.close()
"""
_MANAGED_TRUSTED_RUNTIME = bool(os.environ.get("ZETTLAB_PRESETS_DIR"))
_SENSITIVE_PROCESS_OS_BOUNDARY = (
    not sys.platform.startswith("linux")
    or (
        _MANAGED_TRUSTED_RUNTIME
        and harden_sensitive_process(no_new_privs=False, drop_ptrace=True)
    )
)
_MODEL_DESCENDANT_PTRACE_BOUNDARY = (
    not sys.platform.startswith("linux")
    or (_MANAGED_TRUSTED_RUNTIME and _SENSITIVE_PROCESS_OS_BOUNDARY)
)


def _is_python_executable_token(token: str) -> bool:
    name = Path(token).name.lower()
    return (
        name in {"python", "python3", "python.exe", "python3.exe"}
        or re.fullmatch(r"python3\.\d+(?:\.exe)?", name) is not None
    )


def _path_trust_rejection_reason(
    path: Path,
    *,
    enforce_cutoff: bool = True,
) -> Optional[str]:
    try:
        st = path.stat()
    except OSError:
        return "path_unavailable"
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    mode = stat.S_IMODE(st.st_mode)
    if euid == 0:
        # A root-running terminal can rewrite root-owned files even when mode
        # bits look read-only. Runtime immutability is enforced separately by
        # the startup-pinned tree/content digest; wall-clock mtime/ctime is not
        # trustworthy while a device RTC is still synchronising (codex P1).
        if st.st_uid != 0:
            return "uid_not_root"
        if mode & stat.S_IWGRP:
            return "group_writable"
        if mode & stat.S_IWOTH:
            return "world_writable"
        return None
    if euid is not None and st.st_uid == euid:
        return "owned_by_terminal_user"
    try:
        groups = set(os.getgroups())
        egid = os.getegid()  # windows-footgun: ok — guarded by try/except
        groups.add(egid)
    except Exception:
        groups = set()
    if st.st_gid in groups and mode & stat.S_IWGRP:
        return "group_writable"
    if mode & stat.S_IWOTH:
        return "world_writable"
    return None


def _path_writable_by_current_user(path: Path, *, enforce_cutoff: bool = True) -> bool:
    return _path_trust_rejection_reason(
        path,
        enforce_cutoff=enforce_cutoff,
    ) is not None


def _path_identity(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_dev, st.st_ino


def _connector_runtime_file_digest(path: Path, expected: os.stat_result) -> str:
    """Hash one no-follow regular file while pinning its open descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        expected_identity = (expected.st_dev, expected.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != expected_identity
            or before.st_size != expected.st_size
        ):
            raise OSError("presets file changed before trust snapshot")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > expected.st_size:
                raise OSError("presets file grew during trust snapshot")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != expected_identity
            or after.st_size != before.st_size
            or size != before.st_size
        ):
            raise OSError("presets file changed during trust snapshot")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _connector_runtime_tree_snapshot(root: Path) -> tuple[str, dict[str, str]]:
    """Return a bounded, wall-clock-independent snapshot of a presets tree.

    The digest pins names, file types, inode/device, owner/mode, sizes and file
    contents. It deliberately excludes mtime/ctime: NTP correcting a cold-boot
    RTC must not invalidate an otherwise unchanged official preset tree.
    """

    digest = hashlib.sha256()
    file_digests: dict[str, str] = {}
    stack = [root]
    path_count = 0
    total_bytes = 0
    while stack:
        path = stack.pop()
        st = path.lstat()
        path_count += 1
        if path_count > _CONNECTOR_RUNTIME_TRUST_MAX_PATHS:
            raise OSError("presets trust snapshot path limit exceeded")
        try:
            relative = path.relative_to(root).as_posix() or "."
        except ValueError as exc:
            raise OSError("presets trust snapshot escaped root") from exc
        file_type = stat.S_IFMT(st.st_mode)
        record = (
            relative,
            file_type,
            stat.S_IMODE(st.st_mode),
            st.st_dev,
            st.st_ino,
            st.st_uid,
            st.st_gid,
            st.st_size,
        )
        digest.update(repr(record).encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")

        if stat.S_ISREG(st.st_mode):
            if st.st_size < 0:
                raise OSError("invalid presets file size")
            total_bytes += st.st_size
            if total_bytes > _CONNECTOR_RUNTIME_TRUST_MAX_BYTES:
                raise OSError("presets trust snapshot byte limit exceeded")
            file_digest = _connector_runtime_file_digest(path, st)
            file_digests[relative] = file_digest
            digest.update(file_digest.encode("ascii"))
            digest.update(b"\0")
            continue
        if not stat.S_ISDIR(st.st_mode):
            raise OSError("presets trust snapshot contains a special path")
        with os.scandir(path) as entries:
            children = sorted(
                (Path(entry.path) for entry in entries),
                key=lambda child: child.name,
                reverse=True,
            )
        stack.extend(children)
    return digest.hexdigest(), file_digests


def _log_connector_runtime_rejection(reason: str, relative_path: str = "") -> None:
    logger.warning(
        "Connector runtime direct runner rejected: reason=%s relative_path=%s",
        reason,
        relative_path or "<unknown>",
    )


def _capture_connector_runtime_root() -> Optional[_ConnectorRuntimeRootAnchor]:
    global _CONNECTOR_RUNTIME_ROOT_ANCHOR

    presets_dir = os.environ.get("ZETTLAB_PRESETS_DIR", "")
    if not presets_dir:
        _log_connector_runtime_rejection("presets_dir_missing")
        return None
    configured_root = Path(
        os.path.expandvars(os.path.expanduser(presets_dir))
    ).absolute()
    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if anchor is not None:
        if configured_root != anchor.configured_root:
            _log_connector_runtime_rejection("presets_root_changed")
            return None
        return anchor

    try:
        resolved_root = configured_root.resolve(strict=True)
        identity = _path_identity(resolved_root)
        tree_digest, file_digests = _connector_runtime_tree_snapshot(resolved_root)
    except OSError:
        _log_connector_runtime_rejection("presets_root_unavailable")
        return None

    anchor = _ConnectorRuntimeRootAnchor(
        configured_root=configured_root,
        resolved_root=resolved_root,
        identity=identity,
        tree_digest=tree_digest,
        file_digests=file_digests,
    )
    _CONNECTOR_RUNTIME_ROOT_ANCHOR = anchor
    return anchor


# Production gateways receive ZETTLAB_PRESETS_DIR before importing this module.
# Capture the concrete version directory before any model-authored terminal call.
if os.environ.get("ZETTLAB_PRESETS_DIR"):
    _capture_connector_runtime_root()


def _connector_runtime_path_is_trusted(
    path: Path,
    presets_root: Path,
    *,
    expected_root_identity: Optional[tuple[int, int]] = None,
) -> bool:
    """Return True only for the pinned, immutable official presets tree."""
    try:
        lexical_relative = path.relative_to(presets_root)
        resolved_path = path.resolve(strict=True)
        resolved_root = presets_root.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_root)
        if expected_root_identity is not None:
            if _path_identity(resolved_root) != expected_root_identity:
                return False
    except (OSError, ValueError):
        return False

    # Shared mount ancestors may legitimately change after Hermes starts (for
    # example, creation of /volume1/subvol/.recycle). They still must have safe
    # ownership/mode, but are outside the pinned version-tree digest boundary.
    # The version root and everything below it reject symlinks and must match
    # the bounded snapshot captured before any model-authored terminal call.
    root_ancestors = list(resolved_root.parents)
    if any(
        _path_writable_by_current_user(component, enforce_cutoff=False)
        for component in root_ancestors
    ):
        return False

    current = resolved_root
    components = [resolved_root]
    for part in lexical_relative.parts:
        current = current / part
        components.append(current)
    try:
        if any(stat.S_ISLNK(component.lstat().st_mode) for component in components):
            return False
    except OSError:
        return False
    if any(
        _path_writable_by_current_user(component, enforce_cutoff=True)
        for component in components
    ):
        return False
    if expected_root_identity is None:
        return True
    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if (
        anchor is None
        or anchor.identity != expected_root_identity
        or anchor.resolved_root != resolved_root
    ):
        return False
    try:
        current_digest, _ = _connector_runtime_tree_snapshot(resolved_root)
    except OSError:
        return False
    return current_digest == anchor.tree_digest


def _connector_runtime_trust_rejection_reason(
    path: Path,
    presets_root: Path,
    *,
    expected_root_identity: Optional[tuple[int, int]] = None,
) -> Optional[str]:
    """Return a non-sensitive reason for a failed trust decision."""
    try:
        lexical_relative = path.relative_to(presets_root)
        resolved_path = path.resolve(strict=True)
        resolved_root = presets_root.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
        if expected_root_identity is not None:
            if _path_identity(resolved_root) != expected_root_identity:
                return "trust_anchor_changed"
    except OSError:
        return "path_unavailable"
    except ValueError:
        return "path_escape"

    for component in resolved_root.parents:
        reason = _path_trust_rejection_reason(component, enforce_cutoff=False)
        if reason is not None:
            return f"shared_ancestor_{reason}"

    current = resolved_root
    components = [resolved_root]
    for part in lexical_relative.parts:
        current = current / part
        components.append(current)
    try:
        if any(stat.S_ISLNK(component.lstat().st_mode) for component in components):
            return "symlink_or_special_file"
    except OSError:
        return "path_unavailable"
    for component in components:
        reason = _path_trust_rejection_reason(component, enforce_cutoff=True)
        if reason is not None:
            return reason
    if expected_root_identity is not None:
        anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
        if (
            anchor is None
            or anchor.identity != expected_root_identity
            or anchor.resolved_root != resolved_root
        ):
            return "trust_anchor_changed"
        try:
            current_digest, _ = _connector_runtime_tree_snapshot(resolved_root)
        except OSError:
            return "tree_snapshot_unavailable"
        if current_digest != anchor.tree_digest:
            return "tree_changed_since_start"
    return None


def _resolve_connector_runtime_script(raw_path: str) -> Optional[Path]:
    anchor = _capture_connector_runtime_root()
    if anchor is None:
        return None
    relative_text: Optional[str] = None
    for prefix in ("$ZETTLAB_PRESETS_DIR/", "${ZETTLAB_PRESETS_DIR}/"):
        if raw_path.startswith(prefix):
            relative_text = raw_path[len(prefix):]
            break
    else:
        normalized_raw = raw_path[2:] if raw_path.startswith("./") else raw_path
        if normalized_raw.startswith("skills/"):
            relative_text = normalized_raw
        else:
            expanded_path = Path(
                os.path.expandvars(os.path.expanduser(normalized_raw))
            ).absolute()
            for allowed_root in (anchor.configured_root, anchor.resolved_root):
                try:
                    relative_text = str(expanded_path.relative_to(allowed_root))
                    break
                except ValueError:
                    continue
            if relative_text is None:
                _log_connector_runtime_rejection("path_outside_pinned_root")
                return None

    try:
        candidate_path = anchor.resolved_root / str(relative_text)
        path = candidate_path.resolve(strict=True)
        relative = path.relative_to(anchor.resolved_root)
    except (OSError, ValueError):
        _log_connector_runtime_rejection(
            "path_unavailable_or_escaped",
            relative_text or "",
        )
        return None

    parts = path.parts
    if len(parts) < 4:
        _log_connector_runtime_rejection("invalid_layout", str(relative))
        return None
    if parts[-1] != _CONNECTOR_RUNTIME_SCRIPT:
        _log_connector_runtime_rejection("invalid_script_name", str(relative))
        return None
    if parts[-2] != "scripts" or parts[-4] != "skills":
        _log_connector_runtime_rejection("invalid_layout", str(relative))
        return None
    if not path.is_file():
        _log_connector_runtime_rejection("runner_not_regular_file", str(relative))
        return None
    if not _connector_runtime_path_is_trusted(
        candidate_path,
        anchor.resolved_root,
        expected_root_identity=anchor.identity,
    ):
        reason = _connector_runtime_trust_rejection_reason(
            candidate_path,
            anchor.resolved_root,
            expected_root_identity=anchor.identity,
        )
        _log_connector_runtime_rejection(reason or "trust_check_failed", str(relative))
        return None
    return path


def _managed_lark_cli_broker_enabled() -> bool:
    return os.environ.get("HERMES_MANAGED_GATEWAY") == "1" and os.name != "nt"


def _lex_lark_cli_command(command: str) -> Optional[list[str]]:
    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return None


def _parse_lark_cli_command(command: str) -> Optional[_LarkCLICommand]:
    """Parse one exact foreground lark-cli invocation without a shell."""

    if not _managed_lark_cli_broker_enabled():
        return None
    tokens = _lex_lark_cli_command(command)
    if not tokens or tokens[0] != _LARK_CLI_COMMAND:
        return None
    if any(
        token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION
        for token in tokens
    ):
        return None
    return _LarkCLICommand(args=tokens[1:])


def _lark_cli_command_is_present(command: str) -> bool:
    tokens = _lex_lark_cli_command(command)
    if tokens is None:
        first = command.strip().split(None, 1)[0] if command.strip() else ""
        return first == _LARK_CLI_COMMAND
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION:
            segments.append([])
        else:
            segments[-1].append(token)
    for segment in segments:
        for index, token in enumerate(segment):
            if Path(token).name != _LARK_CLI_COMMAND:
                continue
            if index == 0 or _connector_runtime_command_prefix_is_supported(
                segment[:index]
            ):
                return True
    return False


def _lark_cli_shell_guard_result(
    command: str,
    *,
    compound: bool = False,
) -> Optional[str]:
    """Keep broker-required lark-cli calls out of generic/background shells."""

    if (
        not _managed_lark_cli_broker_enabled()
        or not _lark_cli_command_is_present(command)
    ):
        return None
    code = "lark_cli_compound_command" if compound else "lark_cli_direct_only"
    message = (
        "Managed lark-cli commands must run as one direct foreground non-PTY "
        "terminal call, without shell operators or command wrappers. Retry "
        "each lark-cli command in a separate terminal tool call."
    )
    return json.dumps(
        {
            "output": "",
            "exit_code": 2,
            "error": message,
            "errorCode": code,
            "status": "error",
            "lark_cli_brokered": False,
            "lark_cli_blocked": True,
        },
        ensure_ascii=False,
    )


def _lark_cli_broker_agent_id() -> str:
    from agent.secret_scope import current_secret_scope, is_multiplex_active

    scope = current_secret_scope()
    if scope is None and is_multiplex_active():
        raise RuntimeError("lark-cli profile scope unavailable")
    agent_id = str(
        (scope or {}).get("ZET_AGENT_ID")
        or ("" if is_multiplex_active() else os.environ.get("ZET_AGENT_ID", ""))
    ).strip()
    if not agent_id:
        raise RuntimeError("lark-cli profile identity unavailable")
    return agent_id


def _lark_cli_result_json(
    *,
    output: str,
    exit_code: int,
    timed_out: bool,
) -> str:
    from agent.redact import redact_sensitive_text
    from tools.ansi_strip import strip_ansi

    normalized = strip_ansi(str(output or ""))
    try:
        from tools.tool_output_limits import get_max_bytes

        max_output_chars = get_max_bytes()
    except Exception:
        max_output_chars = 50000
    if len(normalized) > max_output_chars:
        head_chars = int(max_output_chars * 0.4)
        tail_chars = max_output_chars - head_chars
        omitted = len(normalized) - head_chars - tail_chars
        normalized = (
            normalized[:head_chars]
            + f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted] ...\n\n"
            + normalized[-tail_chars:]
        )
    normalized = (
        redact_sensitive_text(normalized.strip(), force=True, code_file=False)
        if normalized
        else ""
    )
    return json.dumps(
        {
            "output": normalized,
            "exit_code": 124 if timed_out else int(exit_code),
            "error": "Command timed out while running lark-cli" if timed_out else None,
            "status": "error" if timed_out else "completed",
            "lark_cli_brokered": True,
        },
        ensure_ascii=False,
    )


def _run_lark_cli_command_if_allowed(
    command: str,
    *,
    timeout: int,
) -> Optional[str]:
    parsed = _parse_lark_cli_command(command)
    if parsed is None:
        return _lark_cli_shell_guard_result(command, compound=True)
    try:
        normalized_timeout = max(
            1,
            min(int(timeout), _LARK_CLI_MAX_TIMEOUT_SECONDS),
        )
        from agent.credential_broker import request_lark_cli

        completed = request_lark_cli(
            _lark_cli_broker_agent_id(),
            parsed.args,
            timeout_seconds=normalized_timeout,
        )
        return _lark_cli_result_json(
            output=completed.output,
            exit_code=completed.exit_code,
            timed_out=completed.timed_out,
        )
    except Exception as exc:
        logger.warning("Managed lark-cli broker request failed: %s", type(exc).__name__)
        return json.dumps(
            {
                "output": "",
                "exit_code": -1,
                "error": str(exc),
                "errorCode": "lark_cli_broker_unavailable",
                "status": "error",
                "lark_cli_brokered": True,
            },
            ensure_ascii=False,
        )


def _parse_connector_runtime_command(command: str) -> Optional[_ConnectorRuntimeCommand]:
    """Return argv for the dedicated connector runner, or None if not exact.

    The allowlist intentionally accepts only a direct Python invocation of a
    presets skill's scripts/connector_runtime.py. Shell punctuation rejects
    compound commands such as `connector_runtime.py ... ; env`, so injected
    connector env can never be observed by a following shell fragment.
    """
    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    # Keep unquoted newlines visible as shell separators. Quoted newlines stay
    # inside their argument token, just like punctuation inside --args-json.
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None

    if len(tokens) < 2 or not _is_python_executable_token(tokens[0]):
        return None
    for token in tokens:
        if token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION:
            return None

    if Path(tokens[1]).name != _CONNECTOR_RUNTIME_SCRIPT:
        return None

    script = _resolve_connector_runtime_script(tokens[1])
    if script is None:
        return None
    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if anchor is None:
        return None
    try:
        script_identity = _path_identity(script)
    except OSError:
        _log_connector_runtime_rejection("runner_identity_unavailable")
        return None
    return _ConnectorRuntimeCommand(
        argv=[sys.executable, str(script), *tokens[2:]],
        root_identity=anchor.identity,
        script_identity=script_identity,
    )


def _connector_runtime_shell_guard_result(
    command: str,
    *,
    _nested_shell_depth: int = 0,
) -> Optional[str]:
    """Block shell-wrapped official connector runtime invocations.

    The dedicated runner intentionally accepts only one direct Python command.
    When an otherwise trusted runtime invocation is combined with another shell
    fragment, falling through to the generic terminal strips Connector context
    and produces a misleading authorization error. Fail closed instead and tell
    the agent to retry each runtime command in its own tool call.
    """
    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return None

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION:
            segments.append([])
            continue
        segments[-1].append(token)

    contains_trusted_runtime = any(
        _connector_runtime_segment_contains_trusted_invocation(segment)
        for segment in segments
    )
    if not contains_trusted_runtime and _nested_shell_depth < _CONNECTOR_RUNTIME_NESTED_SHELL_DEPTH:
        contains_trusted_runtime = any(
            _connector_runtime_segment_contains_nested_shell_invocation(
                segment,
                nested_shell_depth=_nested_shell_depth,
            )
            for segment in segments
        )
    if not contains_trusted_runtime:
        return None

    code = "connector_runtime_compound_command"
    message = (
        "Connector Runtime commands must run as one direct Python invocation "
        "in a foreground non-PTY terminal tool call, without command wrappers "
        "or shell operators. Retry each connector_runtime.py command in a "
        "separate terminal tool call."
    )
    return json.dumps({
        "output": "",
        "exit_code": 2,
        "error": message,
        "errorCode": code,
        "status": "error",
        "connector_runtime_direct": False,
        "connector_runtime_blocked": True,
        "connector_error": {
            "code": code,
            "errorCode": code,
            "message": message,
            "nextAction": {"type": "retry_single_command"},
        },
    }, ensure_ascii=False)


def _connector_runtime_segment_contains_trusted_invocation(segment: list[str]) -> bool:
    """Recognize direct or explicitly wrapped runtime command positions."""
    for index in range(len(segment) - 1):
        if not _is_python_executable_token(segment[index]):
            continue
        script_index = _connector_runtime_python_script_index(segment, index)
        if script_index is None:
            continue
        if index > 0 and not _connector_runtime_command_prefix_is_supported(segment[:index]):
            continue
        if _resolve_connector_runtime_script(segment[script_index]) is not None:
            return True
    return False


def _connector_runtime_segment_contains_nested_shell_invocation(
    segment: list[str],
    *,
    nested_shell_depth: int,
) -> bool:
    """Inspect only supported ``shell -c`` command-string positions."""
    for index, token in enumerate(segment):
        if Path(token).name.lower() not in _CONNECTOR_RUNTIME_COMMAND_SHELLS:
            continue
        if index > 0 and not _connector_runtime_command_prefix_is_supported(segment[:index]):
            continue
        nested_command = _connector_runtime_shell_command_argument(segment[index + 1:])
        if nested_command is None:
            continue
        if _connector_runtime_shell_guard_result(
            nested_command,
            _nested_shell_depth=nested_shell_depth + 1,
        ) is not None:
            return True
    return False


def _connector_runtime_shell_command_argument(arguments: list[str]) -> Optional[str]:
    """Return the command string passed to a supported shell's ``-c`` flag."""
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            return None
        if not token.startswith(("-", "+")) or token in {"-", "+"}:
            return None
        option_name = token.split("=", 1)[0]
        if option_name in _CONNECTOR_RUNTIME_SHELL_OPTIONS_WITH_ARG:
            index += 1
            if "=" not in token:
                if index >= len(arguments):
                    return None
                index += 1
            continue
        short_options = token[1:]
        if token.startswith("--") or "c" not in short_options:
            index += 1
            continue
        command_index = index + 1
        if command_index >= len(arguments):
            return None
        return arguments[command_index]
    return None


def _connector_runtime_python_script_index(
    segment: list[str],
    python_index: int,
    *,
    script_name: str = _CONNECTOR_RUNTIME_SCRIPT,
) -> Optional[int]:
    """Find a script after Python flags without interpreting ``-c``/``-m``."""
    position = python_index + 1
    while position < len(segment):
        token = segment[position]
        if token == "--":
            position += 1
            break
        if not token.startswith("-"):
            break
        option_name = token.split("=", 1)[0]
        if option_name in {"-c", "-m"}:
            return None
        position += 1
        if "=" not in token and option_name in {"-W", "-X"}:
            if position >= len(segment):
                return None
            position += 1
    if (
        position >= len(segment)
        or Path(segment[position]).name != script_name
    ):
        return None
    return position


def _connector_runtime_command_prefix_is_supported(prefix: list[str]) -> bool:
    """Accept standalone shell group openers before known wrappers."""
    position = 0
    while position < len(prefix) and prefix[position] == _CONNECTOR_RUNTIME_SHELL_GROUP_START:
        position += 1
    if position == len(prefix):
        return position > 0
    return _connector_runtime_wrapper_prefix_is_supported(prefix[position:])


def _connector_runtime_wrapper_prefix_is_supported(prefix: list[str]) -> bool:
    """Return True when prefix is only a known command-wrapper chain.

    This parser is intentionally smaller than shell parsing: it recognizes the
    wrapper vocabulary already handled by Hermes command guards plus timeout,
    and rejects unknown prefix words so data such as ``echo python3 ...`` does
    not become a Connector-runtime false positive.
    """
    position = 0
    saw_wrapper = False
    while position < len(prefix):
        if _CONNECTOR_RUNTIME_ENV_ASSIGNMENT_RE.fullmatch(prefix[position]) is not None:
            saw_wrapper = True
            position += 1
            continue
        wrapper = Path(prefix[position]).name.lower()
        if wrapper not in _CONNECTOR_RUNTIME_WRAPPERS:
            return False
        saw_wrapper = True
        position += 1

        options_with_arg = _CONNECTOR_RUNTIME_WRAPPER_OPTIONS_WITH_ARG.get(wrapper, set())
        while position < len(prefix) and prefix[position].startswith("-"):
            option = prefix[position]
            position += 1
            if option == "--":
                break
            option_name = option.split("=", 1)[0]
            if "=" not in option and option_name in options_with_arg:
                if position >= len(prefix):
                    return False
                position += 1

        if wrapper == "timeout":
            if (
                position >= len(prefix)
                or _CONNECTOR_RUNTIME_TIMEOUT_RE.fullmatch(prefix[position]) is None
            ):
                return False
            position += 1

        if wrapper in {"env", "sudo"}:
            while (
                position < len(prefix)
                and _CONNECTOR_RUNTIME_ENV_ASSIGNMENT_RE.fullmatch(prefix[position]) is not None
            ):
                position += 1

    return saw_wrapper


def _connector_runtime_result_json(
    *,
    command: str,
    output: str,
    returncode: int,
    secret_values: list[str] | None = None,
    timed_out: bool = False,
) -> str:
    from tools.ansi_strip import strip_ansi
    from agent.redact import redact_sensitive_text

    output = strip_ansi(output)
    for secret in secret_values or []:
        if secret:
            output = output.replace(secret, "[REDACTED]")
    try:
        from tools.tool_output_limits import get_max_bytes

        max_output_chars = get_max_bytes()
    except Exception:
        max_output_chars = 20000
    if len(output) > max_output_chars:
        head_chars = int(max_output_chars * 0.4)
        tail_chars = max_output_chars - head_chars
        omitted = len(output) - head_chars - tail_chars
        output = (
            output[:head_chars]
            + f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
            + f"out of {len(output)} total] ...\n\n"
            + output[-tail_chars:]
        )
    output = redact_sensitive_text(
        output.strip(),
        force=True,
        code_file=False,
    ) if output else ""
    return json.dumps({
        "output": output,
        "exit_code": 124 if timed_out else returncode,
        "error": (
            f"Command timed out while running connector runtime"
            if timed_out else None
        ),
        "connector_runtime_direct": True,
    }, ensure_ascii=False)


def _connector_runtime_isolated_sys_path(*, script: Path, cwd: Path) -> list[str]:
    """Build a Python import path that excludes model-writable command context."""
    del script
    from tools.trusted_direct_runner import isolated_python_path

    return isolated_python_path(cwd=cwd)


def _read_connector_runtime_script_bytes(
    script: Path,
    *,
    expected_identity: tuple[int, int],
    expected_digest: Optional[str] = None,
) -> bytes:
    """Freeze a verified runner before the worker drops privileges."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(script, flags)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_dev, before.st_ino) != expected_identity
            or before.st_size < 0
            or before.st_size > _CONNECTOR_RUNTIME_MAX_SCRIPT_BYTES
        ):
            raise OSError("connector runtime snapshot is not trusted")
        chunks: list[bytes] = []
        remaining = _CONNECTOR_RUNTIME_MAX_SCRIPT_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > _CONNECTOR_RUNTIME_MAX_SCRIPT_BYTES
            or (after.st_dev, after.st_ino) != expected_identity
            or after.st_size != before.st_size
            or len(payload) != before.st_size
            or (
                expected_digest is not None
                and hashlib.sha256(payload).hexdigest() != expected_digest
            )
        ):
            raise OSError("connector runtime changed while being frozen")
        return payload
    finally:
        os.close(descriptor)


def _trusted_video_edit_source_bundle(
    *,
    script: Path,
    presets_root: Path,
    expected_root_identity: tuple[int, int],
) -> dict[str, dict[str, str]]:
    """Return a pre-trust source snapshot; never reread scripts post-terminal."""
    global _TRUSTED_RUNTIME_SOURCE_CACHE_BYTES

    video_edit_scripts_root = (
        presets_root / "skills" / "video-edit-workflow-mini" / "scripts"
    )
    require_signed_video_edit_sources = script.parent == video_edit_scripts_root

    cache_key = (str(script.parent), expected_root_identity)
    with _VIDEO_EDIT_WORKER_LOCK:
        snapshot = _TRUSTED_RUNTIME_SOURCE_CACHE.get(cache_key)
        if snapshot is None:
            if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
                raise PermissionError(
                    "trusted runtime source was not captured before terminal access"
                )
            if len(_TRUSTED_RUNTIME_SOURCE_CACHE) >= (
                _TRUSTED_RUNTIME_SOURCE_CACHE_MAX_DIRECTORIES
            ):
                raise MemoryError("trusted runtime source cache directory limit reached")

            signed_digests = (
                _trusted_video_edit_release_digests()
                if require_signed_video_edit_sources
                else {}
            )
            if require_signed_video_edit_sources and not signed_digests:
                raise PermissionError(
                    "signed video-edit helper manifest is unavailable"
                )

            modules: list[_TrustedWorkerModuleSnapshot] = []
            total_bytes = 0
            for dependency in sorted(script.parent.glob("*.py")):
                if not dependency.is_file() or not _connector_runtime_path_is_trusted(
                    dependency,
                    presets_root,
                    expected_root_identity=expected_root_identity,
                ):
                    raise PermissionError(
                        f"untrusted video-edit dependency: {dependency.name}"
                    )
                source_bytes = _read_stable_trusted_worker_source(dependency)
                if require_signed_video_edit_sources:
                    relative_dependency = dependency.relative_to(
                        presets_root
                    ).as_posix()
                    expected_digest = signed_digests.get(relative_dependency)
                    if (
                        expected_digest is None
                        or hashlib.sha256(source_bytes).hexdigest()
                        != expected_digest
                    ):
                        raise PermissionError(
                            "video-edit dependency does not match signed manifest: "
                            f"{dependency.name}"
                        )
                total_bytes += len(source_bytes)
                if (
                    _TRUSTED_RUNTIME_SOURCE_CACHE_BYTES + total_bytes
                    > _TRUSTED_RUNTIME_SOURCE_CACHE_MAX_BYTES
                ):
                    raise MemoryError("trusted runtime source cache byte limit reached")
                try:
                    source = source_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise PermissionError(
                        f"trusted runtime source is not UTF-8: {dependency.name}"
                    ) from exc
                if not _connector_runtime_path_is_trusted(
                    dependency,
                    presets_root,
                    expected_root_identity=expected_root_identity,
                ):
                    raise PermissionError(
                        f"video-edit dependency changed: {dependency.name}"
                    )
                modules.append(
                    _TrustedWorkerModuleSnapshot(
                        dependency.stem,
                        str(dependency),
                        source,
                    )
                )
            snapshot = _TrustedRuntimeDirectorySnapshot(
                modules=tuple(modules),
                total_bytes=total_bytes,
            )
            if not any(module.path == str(script) for module in snapshot.modules):
                raise PermissionError(
                    "video-edit entrypoint missing from trusted snapshot"
                )
            encoded_bundle_bytes = len(json.dumps(
                {
                    module.name: {
                        "path": module.path,
                        "source": module.source,
                    }
                    for module in snapshot.modules
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
            if encoded_bundle_bytes > _TRUSTED_RUNTIME_DIRECTORY_MAX_ENCODED_BYTES:
                raise MemoryError("trusted runtime source directory exceeds IPC budget")
            _TRUSTED_RUNTIME_SOURCE_CACHE[cache_key] = snapshot
            _TRUSTED_RUNTIME_SOURCE_CACHE_BYTES += total_bytes

        bundle = {
            ("__main__" if module.path == str(script) else module.name): {
                "path": module.path,
                "source": module.source,
            }
            for module in snapshot.modules
        }
        if "__main__" not in bundle:
            raise PermissionError("runtime entrypoint was not captured before terminal access")
        return bundle


def _trusted_video_edit_release_digests() -> dict[str, str]:
    """Load only the digests from the already verified presets manifest."""
    try:
        from agent.zet_agent_response_mode import (
            trusted_video_edit_manifest_digests,
        )

        return dict(trusted_video_edit_manifest_digests())
    except Exception:
        return {}


def _preload_trusted_runtime_source_bundles() -> None:
    """Eagerly snapshot every allowlisted installed runtime before terminal use."""
    anchor = _capture_connector_runtime_root()
    if anchor is None:
        raise PermissionError("trusted presets root is unavailable")
    candidates = set(
        anchor.resolved_root.glob("skills/*/scripts/connector_runtime.py")
    )
    video_scripts = (
        anchor.resolved_root / "skills" / "video-edit-workflow-mini" / "scripts"
    )
    candidates.update(
        video_scripts / name
        for name in _VIDEO_EDIT_RUNTIME_SCRIPTS
        if (video_scripts / name).is_file()
    )
    if len(candidates) > _TRUSTED_RUNTIME_SOURCE_CACHE_MAX_DIRECTORIES:
        raise MemoryError("trusted runtime source preload count exceeded")
    for script in sorted(candidates):
        if not _connector_runtime_path_is_trusted(
            script,
            anchor.resolved_root,
            expected_root_identity=anchor.identity,
        ):
            raise PermissionError(f"untrusted runtime preload: {script}")
        _trusted_video_edit_source_bundle(
            script=script,
            presets_root=anchor.resolved_root,
            expected_root_identity=anchor.identity,
        )


def _video_edit_worker_recv_exact(channel: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("trusted video-edit worker channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _video_edit_worker_recv_frame(
    channel: socket.socket,
    *,
    timeout: float,
) -> dict[str, Any]:
    previous_timeout = channel.gettimeout()
    channel.settimeout(timeout)
    try:
        size = struct.unpack(
            "!I",
            _video_edit_worker_recv_exact(channel, 4),
        )[0]
        if size <= 0 or size > _VIDEO_EDIT_WORKER_MAX_FRAME_BYTES:
            raise ValueError("invalid trusted video-edit worker frame size")
        payload = json.loads(
            _video_edit_worker_recv_exact(channel, size).decode("utf-8")
        )
    finally:
        channel.settimeout(previous_timeout)
    if not isinstance(payload, dict):
        raise ValueError("invalid trusted video-edit worker frame")
    return payload


def _video_edit_worker_send_frame(
    channel: socket.socket,
    payload: dict[str, Any],
) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _VIDEO_EDIT_WORKER_MAX_FRAME_BYTES:
        raise ValueError("trusted video-edit worker request too large")
    channel.sendall(struct.pack("!I", len(encoded)) + encoded)


def _video_edit_worker_recv_fd_frame(
    channel: socket.socket,
    *,
    timeout: float,
) -> tuple[dict[str, Any], Optional[socket.socket]]:
    previous_timeout = channel.gettimeout()
    channel.settimeout(timeout)
    received_fds = array.array("i")
    try:
        try:
            marker, ancillary, message_flags, _address = channel.recvmsg(
                1,
                socket.CMSG_SPACE(received_fds.itemsize),
                getattr(socket, "MSG_CMSG_CLOEXEC", 0),
            )
            if message_flags & getattr(socket, "MSG_CTRUNC", 0):
                raise RuntimeError("trusted worker seed truncated broker fd metadata")
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    usable = len(data) - (len(data) % received_fds.itemsize)
                    received_fds.frombytes(data[:usable])
            payload = _video_edit_worker_recv_frame(channel, timeout=timeout)
        except Exception:
            for fd in received_fds:
                os.close(fd)
            raise
    finally:
        channel.settimeout(previous_timeout)

    if marker == b"E":
        for fd in received_fds:
            os.close(fd)
        return payload, None
    if marker != b"F" or len(received_fds) != 1:
        for fd in received_fds:
            os.close(fd)
        raise RuntimeError("trusted worker seed returned an invalid broker fd")
    fd = received_fds[0]
    try:
        os.set_inheritable(fd, False)
        return payload, socket.socket(fileno=fd)
    except Exception:
        os.close(fd)
        raise


def _read_video_edit_worker_process_start_time(pid: int) -> Optional[int]:
    """Read Linux's non-reusable process birth identity from procfs."""
    if not sys.platform.startswith("linux"):
        return None
    with Path(f"/proc/{pid}/stat").open("r", encoding="ascii") as handle:
        raw = handle.read(4096)
    closing_paren = raw.rfind(")")
    if closing_paren < 0:
        raise ValueError("invalid /proc stat comm field")
    fields = raw[closing_paren + 2 :].split()
    if len(fields) < 20:
        raise ValueError("incomplete /proc stat record")
    return int(fields[19])


def _capture_video_edit_worker_process_identity(
    pid: int,
) -> _VideoEditWorkerProcessIdentity:
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("invalid trusted worker process id")
    start_before = _read_video_edit_worker_process_start_time(pid)
    if sys.platform.startswith("linux") and start_before is None:
        raise PermissionError("trusted worker process identity is unavailable")

    descriptor: Optional[int] = None
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        try:
            descriptor = pidfd_open(pid, 0)
        except OSError as exc:
            if exc.errno not in {
                errno.EINVAL,
                errno.ENOSYS,
                errno.EPERM,
                errno.EACCES,
            }:
                raise
    try:
        start_after = _read_video_edit_worker_process_start_time(pid)
        if start_after != start_before:
            raise PermissionError("trusted worker process identity changed")
        return _VideoEditWorkerProcessIdentity(
            pid=pid,
            start_time=start_before,
            pidfd=descriptor,
        )
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _close_video_edit_worker_process_identity(
    identity: _VideoEditWorkerProcessIdentity,
) -> None:
    if identity.pidfd is None:
        return
    try:
        os.close(identity.pidfd)
    except OSError:
        pass


def _video_edit_worker_process_identity_is_current(
    identity: _VideoEditWorkerProcessIdentity,
) -> bool:
    try:
        current_start_time = _read_video_edit_worker_process_start_time(identity.pid)
    except (OSError, ValueError):
        return False
    if identity.start_time is not None:
        if current_start_time != identity.start_time:
            return False
    elif sys.platform.startswith("linux"):
        return False

    if identity.pidfd is not None:
        try:
            readable, _, _ = select.select([identity.pidfd], [], [], 0)
        except (OSError, ValueError):
            return False
        return not readable

    try:
        os.kill(identity.pid, 0)  # windows-footgun: ok -- POSIX worker only
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _video_edit_worker_process_identity_can_signal(
    identity: _VideoEditWorkerProcessIdentity,
) -> bool:
    """Require a non-reusable kernel identity before gateway-side signaling."""
    return identity.pidfd is not None or identity.start_time is not None


def _signal_video_edit_worker_process_identity(
    identity: _VideoEditWorkerProcessIdentity,
    signum: int,
) -> bool:
    """Signal only the process captured by this identity, never a reused PID."""
    if not _video_edit_worker_process_identity_can_signal(identity):
        return False
    if not _video_edit_worker_process_identity_is_current(identity):
        return False

    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if identity.pidfd is not None and pidfd_send_signal is not None:
        try:
            if not _video_edit_worker_process_identity_is_current(identity):
                return False
            pidfd_send_signal(identity.pidfd, signum, None, 0)
            return True
        except ProcessLookupError:
            return True
        except OSError:
            return False

    try:
        if not _video_edit_worker_process_identity_is_current(identity):
            return False
        os.kill(identity.pid, signum)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        return False


def _trusted_worker_stat_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _video_edit_worker_source_paths() -> tuple[tuple[str, Path], ...]:
    module_dir = Path(__file__).resolve().parent
    return (
        ("process_security", module_dir / "process_security.py"),
        ("video_edit_runtime_worker", module_dir / "video_edit_runtime_worker.py"),
    )


def _read_stable_trusted_worker_source(path: Path) -> bytes:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    path_before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(path_before.st_mode):
        raise PermissionError(f"trusted worker source is not a regular file: {path.name}")

    fd = os.open(path, flags)
    try:
        opened_before = os.fstat(fd)
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = _VIDEO_EDIT_WORKER_SOURCE_LIMIT_BYTES + 1 - total
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                break
            total += len(chunk)
            if total > _VIDEO_EDIT_WORKER_SOURCE_LIMIT_BYTES:
                raise ValueError(f"trusted worker source is too large: {path.name}")
            chunks.append(chunk)
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)

    path_after = os.stat(path, follow_symlinks=False)
    fingerprints = {
        _trusted_worker_stat_fingerprint(path_before),
        _trusted_worker_stat_fingerprint(opened_before),
        _trusted_worker_stat_fingerprint(opened_after),
        _trusted_worker_stat_fingerprint(path_after),
    }
    if len(fingerprints) != 1 or not stat.S_ISREG(opened_after.st_mode):
        raise PermissionError(f"trusted worker source changed while loading: {path.name}")
    return b"".join(chunks)


def _stable_trusted_worker_file_fingerprint(path: Path) -> tuple[int, ...]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    path_before = os.stat(path, follow_symlinks=False)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
    finally:
        os.close(fd)
    path_after = os.stat(path, follow_symlinks=False)
    fingerprints = {
        _trusted_worker_stat_fingerprint(path_before),
        _trusted_worker_stat_fingerprint(opened),
        _trusted_worker_stat_fingerprint(path_after),
    }
    if len(fingerprints) != 1 or not stat.S_ISREG(opened.st_mode):
        raise PermissionError(f"trusted executable changed while loading: {path.name}")
    return _trusted_worker_stat_fingerprint(opened)


def _capture_trusted_video_edit_worker_snapshot() -> _TrustedWorkerSourceSnapshot:
    modules: list[_TrustedWorkerModuleSnapshot] = []
    total_bytes = 0
    worker_path = ""
    for name, path in _video_edit_worker_source_paths():
        source_bytes = _read_stable_trusted_worker_source(path)
        total_bytes += len(source_bytes)
        if total_bytes > _VIDEO_EDIT_WORKER_SOURCE_LIMIT_BYTES:
            raise ValueError("trusted worker source snapshot is too large")
        try:
            source = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"trusted worker source is not UTF-8: {path.name}") from exc
        modules.append(_TrustedWorkerModuleSnapshot(name, str(path), source))
        if name == "video_edit_runtime_worker":
            worker_path = str(path)
    if not worker_path:
        raise FileNotFoundError("trusted video-edit worker entrypoint missing")

    python_executable = Path(sys.executable).resolve(strict=True)
    return _TrustedWorkerSourceSnapshot(
        modules=tuple(modules),
        python_executable=str(python_executable),
        python_fingerprint=_stable_trusted_worker_file_fingerprint(python_executable),
        worker_path=worker_path,
    )


def _trusted_video_edit_worker_snapshot_payload(
    snapshot: _TrustedWorkerSourceSnapshot,
) -> dict[str, Any]:
    return {
        "modules": [
            {"name": module.name, "path": module.path, "source": module.source}
            for module in snapshot.modules
        ],
        "worker_path": snapshot.worker_path,
    }


def _trusted_video_edit_worker_factory_image(
    snapshot: _TrustedWorkerSourceSnapshot,
) -> _TrustedWorkerFactoryImage:
    """Preload every bootstrap dependency into the gateway before trust closes."""
    global _VIDEO_EDIT_WORKER_FACTORY_IMAGE

    image = _VIDEO_EDIT_WORKER_FACTORY_IMAGE
    if (
        image is not None
        and image.owner_pid == os.getpid()
        and image.snapshot is snapshot
    ):
        return image
    if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
        raise PermissionError(
            "trusted video-edit worker gateway image was not prepared before terminal access"
        )

    expected_names = ("process_security", "video_edit_runtime_worker")
    if tuple(module.name for module in snapshot.modules) != expected_names:
        raise PermissionError("trusted video-edit worker source identity changed")

    missing = object()
    previous_modules: dict[str, object] = {}
    loaded_modules: dict[str, types.ModuleType] = {}
    modules_before = set(sys.modules)
    imported_names: set[str] = set()
    real_import = builtins.__import__

    def tracked_import(
        name: str,
        globals: Optional[dict[str, Any]] = None,
        locals: Optional[dict[str, Any]] = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        imported = real_import(name, globals, locals, fromlist, level)
        if level == 0:
            imported_names.add(name)
            for item in fromlist or ():
                qualified = f"{name}.{item}"
                if qualified in sys.modules:
                    imported_names.add(qualified)
        return imported

    tracked_builtins = dict(vars(builtins))
    tracked_builtins["__import__"] = tracked_import
    try:
        for module_snapshot in snapshot.modules:
            name = module_snapshot.name
            previous_modules[name] = sys.modules.get(name, missing)
            module = types.ModuleType(name)
            module.__file__ = module_snapshot.path
            module.__package__ = ""
            module.__dict__["__builtins__"] = tracked_builtins
            sys.modules[name] = module
            exec(
                compile(
                    module_snapshot.source,
                    module_snapshot.path,
                    "exec",
                ),
                module.__dict__,
            )
            loaded_modules[name] = module

        worker = loaded_modules["video_edit_runtime_worker"]
        real_import_module = worker.importlib.import_module

        def tracked_import_module(name: str, package: Optional[str] = None) -> Any:
            imported = real_import_module(name, package)
            imported_names.add(name)
            return imported

        worker.importlib = types.SimpleNamespace(import_module=tracked_import_module)
        worker._preload_optional_runtime_modules()
        bootstrap_code = compile(
            _VIDEO_EDIT_WORKER_MEMORY_BOOTSTRAP,
            "<trusted-video-edit-worker-supervisor>",
            "exec",
        )
        module_names = {
            "array",
            "builtins",
            "json",
            "os",
            "select",
            "signal",
            "socket",
            "struct",
            "sys",
            "types",
            *expected_names,
            *imported_names,
            *(set(sys.modules) - modules_before),
        }
        pending = list(module_names)
        while pending:
            name = pending.pop()
            module = sys.modules.get(name)
            if not isinstance(module, types.ModuleType):
                continue
            for value in vars(module).values():
                if not isinstance(value, types.ModuleType):
                    continue
                dependency = value.__name__
                if dependency in sys.modules and dependency not in module_names:
                    module_names.add(dependency)
                    pending.append(dependency)
            parts = name.split(".")
            for index in range(1, len(parts)):
                parent = ".".join(parts[:index])
                if parent in sys.modules and parent not in module_names:
                    module_names.add(parent)
                    pending.append(parent)
        for name, module in tuple(sys.modules.items()):
            spec = getattr(module, "__spec__", None)
            if getattr(spec, "origin", None) in {"built-in", "frozen"}:
                module_names.add(name)
        if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
            raise PermissionError(
                "terminal access began while trusted worker image was loading"
            )
        image = _TrustedWorkerFactoryImage(
            snapshot=snapshot,
            bootstrap_code=bootstrap_code,
            module_names=frozenset(module_names),
            owner_pid=os.getpid(),
        )
        _VIDEO_EDIT_WORKER_FACTORY_IMAGE = image
        return image
    finally:
        for name in reversed(tuple(previous_modules)):
            previous = previous_modules[name]
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _validate_trusted_video_edit_worker_interpreter(
    snapshot: _TrustedWorkerSourceSnapshot,
) -> None:
    current = _stable_trusted_worker_file_fingerprint(Path(snapshot.python_executable))
    if current != snapshot.python_fingerprint:
        raise PermissionError("trusted video-edit worker interpreter changed")


def _sealed_trusted_video_edit_worker_interpreter(
    snapshot: _TrustedWorkerSourceSnapshot,
) -> tuple[str, Optional[int]]:
    """Return an immutable Linux executable image, closing the pathname race."""
    _validate_trusted_video_edit_worker_interpreter(snapshot)
    if not sys.platform.startswith("linux"):
        return snapshot.python_executable, None
    if not hasattr(os, "memfd_create"):
        raise PermissionError("sealed trusted interpreter is unavailable")
    try:
        import fcntl as sealed_fcntl
    except ImportError as exc:
        raise PermissionError("sealed trusted interpreter is unavailable") from exc

    path = Path(snapshot.python_executable)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    path_before = os.stat(path, follow_symlinks=False)
    source_fd = os.open(path, flags)
    sealed_fd: Optional[int] = None
    try:
        opened_before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_size <= 0
            or opened_before.st_size > _VIDEO_EDIT_WORKER_INTERPRETER_LIMIT_BYTES
        ):
            raise PermissionError("trusted video-edit worker interpreter is invalid")
        sealed_fd = os.memfd_create(
            "hermes-video-worker-python",
            getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
        )
        copied = 0
        while True:
            chunk = os.read(source_fd, min(1024 * 1024, opened_before.st_size - copied))
            if not chunk:
                break
            copied += len(chunk)
            if copied > _VIDEO_EDIT_WORKER_INTERPRETER_LIMIT_BYTES:
                raise PermissionError("trusted video-edit worker interpreter is too large")
            view = memoryview(chunk)
            while view:
                written = os.write(sealed_fd, view)
                if written <= 0:
                    raise OSError("sealed interpreter copy made no progress")
                view = view[written:]
        opened_after = os.fstat(source_fd)
        path_after = os.stat(path, follow_symlinks=False)
        fingerprints = {
            _trusted_worker_stat_fingerprint(path_before),
            _trusted_worker_stat_fingerprint(opened_before),
            _trusted_worker_stat_fingerprint(opened_after),
            _trusted_worker_stat_fingerprint(path_after),
            snapshot.python_fingerprint,
        }
        if len(fingerprints) != 1 or copied != opened_after.st_size:
            raise PermissionError("trusted video-edit worker interpreter changed while sealing")
        os.fchmod(sealed_fd, 0o500)
        required_seals = (
            getattr(sealed_fcntl, "F_SEAL_WRITE", 0)
            | getattr(sealed_fcntl, "F_SEAL_GROW", 0)
            | getattr(sealed_fcntl, "F_SEAL_SHRINK", 0)
            | getattr(sealed_fcntl, "F_SEAL_SEAL", 0)
        )
        if not required_seals or not hasattr(sealed_fcntl, "F_ADD_SEALS"):
            raise PermissionError("sealed trusted interpreter is unavailable")
        sealed_fcntl.fcntl(sealed_fd, sealed_fcntl.F_ADD_SEALS, required_seals)
        applied_seals = sealed_fcntl.fcntl(sealed_fd, sealed_fcntl.F_GET_SEALS)
        if applied_seals & required_seals != required_seals:
            raise PermissionError("trusted interpreter seals were not applied")
        os.lseek(sealed_fd, 0, os.SEEK_SET)
        return f"/proc/self/fd/{sealed_fd}", sealed_fd
    except Exception:
        if sealed_fd is not None:
            os.close(sealed_fd)
        raise
    finally:
        os.close(source_fd)


def _validate_trusted_video_edit_worker_factory_supervisor(
    supervisor: _TrustedWorkerFactorySupervisor,
    snapshot: _TrustedWorkerSourceSnapshot,
) -> None:
    if supervisor.owner_pid != os.getpid() or supervisor.snapshot is not snapshot:
        raise PermissionError("trusted video-edit worker supervisor identity changed")
    if supervisor.process.poll() is not None:
        raise PermissionError("trusted video-edit worker resident supervisor was lost")
    if not _video_edit_worker_socket_peer_open(supervisor.channel):
        raise PermissionError("trusted video-edit worker supervisor channel was lost")


def _close_inherited_video_edit_worker_fds(*, keep: set[int]) -> None:
    """Close gateway descriptors in the fork child before it starts serving."""
    try:
        max_fd = int(os.sysconf("SC_OPEN_MAX"))
    except (OSError, TypeError, ValueError):
        max_fd = 65536
    start = 3
    for descriptor in sorted(fd for fd in keep if 3 <= fd < max_fd):
        if start < descriptor:
            os.closerange(start, descriptor)
        start = descriptor + 1
    if start < max_fd:
        os.closerange(start, max_fd)


def _run_trusted_video_edit_worker_supervisor_child(
    *,
    image: _TrustedWorkerFactoryImage,
    gateway_pid: int,
    parent_channel: socket.socket,
    child_channel: socket.socket,
    source_parent: socket.socket,
    source_child: socket.socket,
    worker_env: dict[str, str],
) -> None:
    """Enter the already-compiled supervisor image without an OS exec."""
    parent_channel.close()
    source_parent.close()
    child_fd = child_channel.fileno()
    source_fd = source_child.fileno()
    bootstrap_code = image.bootstrap_code
    module_names = image.module_names
    del image
    try:
        gc.disable()
        for name in tuple(sys.modules):
            if name not in module_names:
                sys.modules.pop(name, None)
        os.environ.clear()
        os.environ.update(worker_env)
        _close_inherited_video_edit_worker_fds(keep={child_fd, source_fd})
        os.setsid()  # windows-footgun: ok -- forked POSIX supervisor child
        sys.argv = [
            "hermes-resident-worker-supervisor",
            str(source_fd),
            str(child_fd),
            "-1",
            str(gateway_pid),
        ]
        exec(
            bootstrap_code,
            {
                "__builtins__": __builtins__,
                "__name__": "__main__",
            },
        )
        returncode = 0
    except BaseException:
        returncode = 1
    finally:
        for channel in (source_child, child_channel):
            try:
                channel.close()
            except OSError:
                pass
    os._exit(returncode)


def _trusted_video_edit_worker_factory_bootstrap(
    snapshot: _TrustedWorkerSourceSnapshot,
) -> _TrustedWorkerFactorySupervisor:
    """Fork the resident supervisor before the gateway becomes multithreaded."""
    global _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

    image = _trusted_video_edit_worker_factory_image(snapshot)
    supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
    if supervisor is not None:
        try:
            _validate_trusted_video_edit_worker_factory_supervisor(supervisor, snapshot)
            return supervisor
        except Exception:
            _discard_trusted_video_edit_worker_factory_supervisor()

    if _video_edit_worker_process_thread_count() != 1:
        raise PermissionError(
            "trusted video-edit worker supervisor must be prepared "
            "before gateway threads start"
        )

    parent_channel, child_channel = socket.socketpair()
    source_parent, source_child = socket.socketpair()
    gateway_pid = os.getpid()
    worker_env = _trusted_video_edit_worker_env()
    child_pid: Optional[int] = None
    process: Optional[_ForkedVideoEditWorkerSeed] = None
    try:
        child_pid = os.fork()  # windows-footgun: ok -- POSIX-gated worker path
        if child_pid == 0:
            _run_trusted_video_edit_worker_supervisor_child(
                image=image,
                gateway_pid=gateway_pid,
                parent_channel=parent_channel,
                child_channel=child_channel,
                source_parent=source_parent,
                source_child=source_child,
                worker_env=worker_env,
            )
            os._exit(1)

        identity = _capture_video_edit_worker_process_identity(child_pid)
        process = _ForkedVideoEditWorkerSeed(
            pid=child_pid,
            identity=identity,
            direct_child=True,
        )
        child_channel.close()
        source_child.close()
        _video_edit_worker_send_frame(
            source_parent,
            _trusted_video_edit_worker_snapshot_payload(snapshot),
        )
        source_parent.close()
        ready = _video_edit_worker_recv_frame(
            parent_channel,
            timeout=_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS,
        )
        memory_limit = ready.get("memory_limit") or {}
        if (
            ready.get("ready") is not True
            or ready.get("supervisor_ready") is not True
            or ready.get("resident_supervisor") is not True
            or ready.get("accepts_secrets") is not False
            or ready.get("child_subreaper") is not True
            or (
                sys.platform.startswith("linux")
                and (
                    ready.get("dumpable") != 0
                    or memory_limit.get("applied") is not True
                    or int(memory_limit.get("limit_bytes") or 0)
                    > _VIDEO_EDIT_WORKER_MEMORY_LIMIT_BYTES
                )
            )
            or process.poll() is not None
        ):
            raise RuntimeError("trusted worker resident supervisor failed to initialize")
        supervisor = _TrustedWorkerFactorySupervisor(
            snapshot=snapshot,
            process=process,
            channel=parent_channel,
            owner_pid=os.getpid(),
        )
        _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR = supervisor
        return supervisor
    except Exception:
        for pending_channel in (
            child_channel,
            source_child,
            source_parent,
            parent_channel,
        ):
            try:
                pending_channel.close()
            except OSError:
                pass
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        elif process is None and child_pid is not None and child_pid > 0:
            try:
                os.kill(child_pid, getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
            try:
                os.waitpid(child_pid, 0)
            except ChildProcessError:
                pass
        raise
    finally:
        if process is not None and _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR is None:
            process.close()


def _discard_trusted_video_edit_worker_factory_supervisor() -> None:
    """Stop the resident supervisor from its owning gateway process."""
    global _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

    supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
    _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR = None
    if supervisor is None:
        return
    try:
        if supervisor.owner_pid != os.getpid():
            return
        if supervisor.process.poll() is None:
            try:
                supervisor.channel.sendall(b"Q")
                _video_edit_worker_recv_frame(
                    supervisor.channel,
                    timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
                )
            except (EOFError, OSError, socket.timeout, ValueError):
                pass
        if supervisor.process.poll() is not None:
            return
        try:
            supervisor.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            supervisor.process.kill()
            supervisor.process.wait(timeout=2)
    finally:
        try:
            supervisor.channel.close()
        except OSError:
            pass
        supervisor.process.close()


def _discard_inherited_video_edit_worker_state() -> None:
    """Close inherited handles in a fork child without controlling parent jobs."""
    global _VIDEO_EDIT_WORKER_BROKER_IDENTITY
    global _VIDEO_EDIT_WORKER_BROKER_PID
    global _VIDEO_EDIT_WORKER_CHANNEL
    global _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    global _VIDEO_EDIT_WORKER_FACTORY_IMAGE
    global _VIDEO_EDIT_WORKER_FACTORY_PROCESS
    global _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
    global _VIDEO_EDIT_WORKER_IDLE_GENERATION
    global _VIDEO_EDIT_WORKER_IDLE_TIMER
    global _VIDEO_EDIT_WORKER_LOCK
    global _VIDEO_EDIT_WORKER_SEED_CHANNEL
    global _VIDEO_EDIT_WORKER_SEED_PROCESS

    channels = (
        _VIDEO_EDIT_WORKER_CHANNEL,
        _VIDEO_EDIT_WORKER_FACTORY_CHANNEL,
        _VIDEO_EDIT_WORKER_SEED_CHANNEL,
    )
    broker_identity = _VIDEO_EDIT_WORKER_BROKER_IDENTITY
    factory_process = _VIDEO_EDIT_WORKER_FACTORY_PROCESS
    seed_process = _VIDEO_EDIT_WORKER_SEED_PROCESS
    supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR

    _VIDEO_EDIT_WORKER_BROKER_IDENTITY = None
    _VIDEO_EDIT_WORKER_BROKER_PID = None
    _VIDEO_EDIT_WORKER_CHANNEL = None
    _VIDEO_EDIT_WORKER_FACTORY_CHANNEL = None
    _VIDEO_EDIT_WORKER_FACTORY_IMAGE = None
    _VIDEO_EDIT_WORKER_FACTORY_PROCESS = None
    _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR = None
    _VIDEO_EDIT_WORKER_IDLE_GENERATION += 1
    _VIDEO_EDIT_WORKER_IDLE_TIMER = None
    _VIDEO_EDIT_WORKER_SEED_CHANNEL = None
    _VIDEO_EDIT_WORKER_SEED_PROCESS = None
    _VIDEO_EDIT_WORKER_LOCK = threading.RLock()

    for channel in channels:
        if channel is not None:
            try:
                channel.close()
            except OSError:
                pass
    if supervisor is not None:
        try:
            supervisor.channel.close()
        except OSError:
            pass
        supervisor.process.close()
    if broker_identity is not None:
        _close_video_edit_worker_process_identity(broker_identity)
    if factory_process is not None:
        factory_process.close()
    if seed_process is not None:
        seed_process.close()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        after_in_child=_discard_inherited_video_edit_worker_state
    )


def _trusted_video_edit_worker_snapshot_for_seed_start(
) -> _TrustedWorkerSourceSnapshot:
    """Return only the startup snapshot captured before terminal trust closed."""
    global _VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT

    snapshot = _VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT
    if snapshot is not None:
        return snapshot
    if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
        raise PermissionError(
            "trusted video-edit worker source snapshot was not captured "
            "before terminal access"
        )

    snapshot = _capture_trusted_video_edit_worker_snapshot()
    if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
        raise PermissionError(
            "terminal access began while trusted worker source was loading"
        )
    _VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT = snapshot
    return snapshot


def _trusted_video_edit_worker_env() -> dict[str, str]:
    # The persistent seed never needs profile, provider, cloud, or user env.
    # Keep a fixed locale/path only; the one-shot receives its explicit env over
    # the private broker socket after the seed has forked it.
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.defpath,
        "TZ": "UTC",
    }


def _force_kill_video_edit_worker_group(
    identity: _VideoEditWorkerProcessIdentity,
) -> bool:
    """Contain an identity-stable group without signaling a reused PID/PGID."""
    if not _video_edit_worker_process_identity_can_signal(identity):
        return False
    if not _signal_video_edit_worker_process_identity(identity, signal.SIGSTOP):
        return False
    if not _video_edit_worker_process_identity_is_current(identity):
        return False
    group_signaled = True
    try:
        os.killpg(identity.pid, signal.SIGKILL)  # windows-footgun: ok -- POSIX worker
    except (PermissionError, ProcessLookupError):
        group_signaled = False
    leader_signaled = _signal_video_edit_worker_process_identity(
        identity,
        signal.SIGKILL,  # windows-footgun: ok -- POSIX worker
    )
    return group_signaled or leader_signaled


def _video_edit_worker_socket_peer_open(channel: socket.socket) -> bool:
    try:
        marker = channel.recv(
            1,
            getattr(socket, "MSG_PEEK", 0) | getattr(socket, "MSG_DONTWAIT", 0),
        )
    except (BlockingIOError, socket.timeout):
        return True
    except (AttributeError, OSError):
        return False
    return bool(marker)


def _request_video_edit_worker_parent_reap(
    process: _ForkedVideoEditWorkerSeed,
    *,
    parent_control: Optional[str],
) -> bool:
    """Use the real parent when this platform has no non-reusable PID handle."""
    if parent_control == "supervisor":
        if _VIDEO_EDIT_WORKER_FACTORY_PROCESS is not process:
            return False
        supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
        if supervisor is None:
            return False
        try:
            _validate_trusted_video_edit_worker_factory_supervisor(
                supervisor,
                supervisor.snapshot,
            )
        except PermissionError:
            return False
        channel = supervisor.channel
    elif parent_control == "factory":
        if _VIDEO_EDIT_WORKER_SEED_PROCESS is not process:
            return False
        if not _video_edit_worker_factory_is_ready():
            return False
        channel = _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
        if channel is None:
            return False
    else:
        return False

    try:
        channel.sendall(b"T" + struct.pack("!Q", process.pid))
        stopped = _video_edit_worker_recv_frame(
            channel,
            timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
        )
    except (EOFError, OSError, socket.timeout, ValueError):
        return False
    return stopped.get("pid") == process.pid and stopped.get("reaped") is True


def _video_edit_worker_factory_is_ready() -> bool:
    process = _VIDEO_EDIT_WORKER_FACTORY_PROCESS
    channel = _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
    return (
        supervisor is not None
        and supervisor.owner_pid == os.getpid()
        and supervisor.process.poll() is None
        and _video_edit_worker_socket_peer_open(supervisor.channel)
        and process is not None
        and process.poll() is None
        and channel is not None
        and _video_edit_worker_socket_peer_open(channel)
    )


def _discard_video_edit_worker_seed() -> bool:
    """Drop the active seed and ask its factory to reap the whole subtree."""
    global _VIDEO_EDIT_WORKER_SEED_CHANNEL
    global _VIDEO_EDIT_WORKER_SEED_PROCESS

    channel = _VIDEO_EDIT_WORKER_SEED_CHANNEL
    process = _VIDEO_EDIT_WORKER_SEED_PROCESS
    _VIDEO_EDIT_WORKER_SEED_CHANNEL = None
    _VIDEO_EDIT_WORKER_SEED_PROCESS = None
    if channel is not None:
        try:
            channel.close()
        except OSError:
            pass
    if process is None:
        return True

    try:
        reaped = False
        factory_channel = _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
        if _video_edit_worker_factory_is_ready() and factory_channel is not None:
            try:
                factory_channel.sendall(b"T" + struct.pack("!Q", process.pid))
                stopped = _video_edit_worker_recv_frame(
                    factory_channel,
                    timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
                )
                reaped = (
                    stopped.get("pid") == process.pid
                    and stopped.get("reaped") is True
                )
            except (EOFError, OSError, socket.timeout, ValueError):
                reaped = False
        if not reaped and process.poll() is None:
            _force_kill_video_edit_worker_group(process.identity)
        return reaped
    finally:
        process.close()


def _terminate_video_edit_worker(*, close_disk_trust: bool) -> bool:
    """Stop the active one-shot and return only seed-verified reap status."""
    global _VIDEO_EDIT_WORKER_BROKER_IDENTITY
    global _VIDEO_EDIT_WORKER_BROKER_PID
    global _VIDEO_EDIT_WORKER_CHANNEL
    global _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED

    broker_pid = _VIDEO_EDIT_WORKER_BROKER_PID
    broker_identity = _VIDEO_EDIT_WORKER_BROKER_IDENTITY
    channel = _VIDEO_EDIT_WORKER_CHANNEL
    _VIDEO_EDIT_WORKER_BROKER_IDENTITY = None
    _VIDEO_EDIT_WORKER_BROKER_PID = None
    _VIDEO_EDIT_WORKER_CHANNEL = None
    if close_disk_trust:
        _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED = True
    if channel is not None:
        try:
            channel.close()
        except OSError:
            pass

    if broker_pid is None:
        if broker_identity is not None:
            _close_video_edit_worker_process_identity(broker_identity)
        return True

    seed_channel = _VIDEO_EDIT_WORKER_SEED_CHANNEL
    if (
        broker_pid is not None
        and seed_channel is not None
        and _video_edit_worker_socket_peer_open(seed_channel)
    ):
        try:
            seed_channel.sendall(b"T" + struct.pack("!Q", broker_pid))
            stopped = _video_edit_worker_recv_frame(
                seed_channel,
                timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
            )
            if (
                stopped.get("pid") == broker_pid
                and stopped.get("cleanup") in {"stopped", "already_clean"}
                and stopped.get("reaped") is True
            ):
                if broker_identity is not None:
                    _close_video_edit_worker_process_identity(broker_identity)
                return True
            raise RuntimeError("trusted worker seed did not confirm broker reaping")
        except (EOFError, OSError, RuntimeError, socket.timeout, ValueError):
            pass

    # A gateway-side signal closes the immediate containment gap, but only the
    # seed is the broker's parent and can prove waitpid completed. Lose the seed
    # and disk trust so callers cannot mistake best-effort killing for reaping.
    if broker_identity is not None:
        _force_kill_video_edit_worker_group(broker_identity)
        _close_video_edit_worker_process_identity(broker_identity)
    _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED = True
    _discard_video_edit_worker_seed()
    return False


def _shutdown_video_edit_worker_seed() -> None:
    """Shut down both the active seed and its pre-terminal factory."""
    global _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    global _VIDEO_EDIT_WORKER_FACTORY_PROCESS
    global _VIDEO_EDIT_WORKER_SEED_CHANNEL
    global _VIDEO_EDIT_WORKER_SEED_PROCESS

    _discard_video_edit_worker_seed()
    channel = _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    process = _VIDEO_EDIT_WORKER_FACTORY_PROCESS
    _VIDEO_EDIT_WORKER_FACTORY_CHANNEL = None
    _VIDEO_EDIT_WORKER_FACTORY_PROCESS = None
    _VIDEO_EDIT_WORKER_SEED_CHANNEL = None
    _VIDEO_EDIT_WORKER_SEED_PROCESS = None
    if channel is not None and process is not None and process.poll() is None:
        try:
            channel.sendall(b"Q")
            _video_edit_worker_recv_frame(
                channel,
                timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
            )
        except (EOFError, OSError, socket.timeout, ValueError):
            pass
    if channel is not None:
        try:
            channel.close()
        except OSError:
            pass
    if process is None:
        return
    try:
        if process.poll() is None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
                if supervisor is not None:
                    try:
                        _validate_trusted_video_edit_worker_factory_supervisor(
                            supervisor,
                            supervisor.snapshot,
                        )
                        supervisor.channel.sendall(
                            b"T" + struct.pack("!Q", process.pid)
                        )
                        _video_edit_worker_recv_frame(
                            supervisor.channel,
                            timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
                        )
                    except (EOFError, OSError, PermissionError, socket.timeout, ValueError):
                        pass
                if process.poll() is None:
                    _force_kill_video_edit_worker_group(process.identity)
                process.wait(timeout=2)
    finally:
        process.close()


def _ensure_video_edit_worker_factory_started() -> None:
    global _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    global _VIDEO_EDIT_WORKER_FACTORY_PROCESS

    if _video_edit_worker_factory_is_ready():
        return
    if (
        _VIDEO_EDIT_WORKER_FACTORY_PROCESS is not None
        or _VIDEO_EDIT_WORKER_FACTORY_CHANNEL is not None
    ):
        _shutdown_video_edit_worker_seed()
    if not _ensure_sensitive_runtime_boundary():
        raise PermissionError("Hermes process memory boundary is unavailable")
    if os.name != "posix" or not hasattr(socket.socket, "sendmsg"):
        raise PermissionError("trusted video-edit worker requires POSIX fd isolation")

    snapshot = _trusted_video_edit_worker_snapshot_for_seed_start()
    supervisor = _trusted_video_edit_worker_factory_bootstrap(snapshot)
    _validate_trusted_video_edit_worker_factory_supervisor(supervisor, snapshot)
    supervisor.channel.sendall(b"N")
    response, factory_channel = _video_edit_worker_recv_fd_frame(
        supervisor.channel,
        timeout=_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS,
    )
    factory_pid = response.get("factory_pid")
    ready = response.get("ready") or {}
    memory_limit = ready.get("memory_limit") or {}
    if (
        factory_channel is None
        or response.get("factory_spawned") is not True
        or not isinstance(factory_pid, int)
        or factory_pid <= 0
        or ready.get("ready") is not True
        or ready.get("factory_ready") is not True
        or ready.get("fork_factory") is not True
        or ready.get("resident_image") is not True
        or ready.get("accepts_secrets") is not False
        or ready.get("child_subreaper") is not True
        or (
            sys.platform.startswith("linux")
            and (
                ready.get("dumpable") != 0
                or memory_limit.get("applied") is not True
                or int(memory_limit.get("limit_bytes") or 0)
                > _VIDEO_EDIT_WORKER_MEMORY_LIMIT_BYTES
            )
        )
    ):
        if factory_channel is not None:
            factory_channel.close()
        raise RuntimeError(
            str(response.get("error") or "resident supervisor could not spawn factory")
        )
    try:
        identity = _capture_video_edit_worker_process_identity(factory_pid)
    except Exception:
        try:
            supervisor.channel.sendall(b"T" + struct.pack("!Q", factory_pid))
            _video_edit_worker_recv_frame(
                supervisor.channel,
                timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
            )
        except (EOFError, OSError, socket.timeout, ValueError):
            pass
        factory_channel.close()
        raise
    _VIDEO_EDIT_WORKER_FACTORY_PROCESS = _ForkedVideoEditWorkerSeed(
        factory_pid,
        identity,
        parent_control="supervisor",
    )
    _VIDEO_EDIT_WORKER_FACTORY_CHANNEL = factory_channel


def _spawn_video_edit_worker_seed_from_factory() -> None:
    global _VIDEO_EDIT_WORKER_SEED_CHANNEL
    global _VIDEO_EDIT_WORKER_SEED_PROCESS

    factory_channel = _VIDEO_EDIT_WORKER_FACTORY_CHANNEL
    if not _video_edit_worker_factory_is_ready() or factory_channel is None:
        raise PermissionError("trusted video-edit worker factory unavailable")
    factory_channel.sendall(b"N")
    response, seed_channel = _video_edit_worker_recv_fd_frame(
        factory_channel,
        timeout=_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS,
    )
    seed_pid = response.get("seed_pid")
    ready = response.get("ready") or {}
    memory_limit = ready.get("memory_limit") or {}
    if (
        seed_channel is None
        or response.get("seed_ready") is not True
        or not isinstance(seed_pid, int)
        or seed_pid <= 0
        or ready.get("ready") is not True
        or ready.get("fork_seed") is not True
        or ready.get("accepts_secrets") is not False
        or ready.get("child_subreaper") is not True
        or (
            sys.platform.startswith("linux")
            and (
                ready.get("dumpable") != 0
                or memory_limit.get("applied") is not True
                or int(memory_limit.get("limit_bytes") or 0)
                > _VIDEO_EDIT_WORKER_MEMORY_LIMIT_BYTES
            )
        )
    ):
        if seed_channel is not None:
            seed_channel.close()
        if isinstance(seed_pid, int) and seed_pid > 0:
            try:
                factory_channel.sendall(b"T" + struct.pack("!Q", seed_pid))
                _video_edit_worker_recv_frame(
                    factory_channel,
                    timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
                )
            except (EOFError, OSError, socket.timeout, ValueError):
                pass
        raise RuntimeError(
            str(response.get("error") or "trusted worker factory could not spawn seed")
        )
    try:
        identity = _capture_video_edit_worker_process_identity(seed_pid)
    except Exception:
        seed_channel.close()
        try:
            factory_channel.sendall(b"T" + struct.pack("!Q", seed_pid))
            _video_edit_worker_recv_frame(
                factory_channel,
                timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
            )
        except (EOFError, OSError, socket.timeout, ValueError):
            pass
        raise
    _VIDEO_EDIT_WORKER_SEED_PROCESS = _ForkedVideoEditWorkerSeed(
        seed_pid,
        identity,
        parent_control="factory",
    )
    _VIDEO_EDIT_WORKER_SEED_CHANNEL = seed_channel


def _ensure_video_edit_worker_seed_started() -> None:
    for attempt in range(_VIDEO_EDIT_WORKER_FACTORY_SEED_START_MAX_ATTEMPTS):
        process = _VIDEO_EDIT_WORKER_SEED_PROCESS
        channel = _VIDEO_EDIT_WORKER_SEED_CHANNEL
        try:
            if not _video_edit_worker_factory_is_ready():
                if process is not None or channel is not None:
                    _terminate_video_edit_worker(close_disk_trust=False)
                    _discard_video_edit_worker_seed()
                _ensure_video_edit_worker_factory_started()
                process = _VIDEO_EDIT_WORKER_SEED_PROCESS
                channel = _VIDEO_EDIT_WORKER_SEED_CHANNEL
            if (
                process is not None
                and process.poll() is None
                and channel is not None
                and _video_edit_worker_socket_peer_open(channel)
            ):
                return
            if process is not None or channel is not None:
                _terminate_video_edit_worker(close_disk_trust=False)
                _discard_video_edit_worker_seed()
            _spawn_video_edit_worker_seed_from_factory()
            return
        except (
            AttributeError,
            EOFError,
            OSError,
            PermissionError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            supervisor = _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR
            snapshot = _VIDEO_EDIT_WORKER_SOURCE_SNAPSHOT
            if (
                supervisor is None
                or snapshot is None
                or (
                    _VIDEO_EDIT_WORKER_FACTORY_PROCESS is None
                    and _VIDEO_EDIT_WORKER_FACTORY_CHANNEL is None
                )
            ):
                raise
            _validate_trusted_video_edit_worker_factory_supervisor(
                supervisor,
                snapshot,
            )
            _shutdown_video_edit_worker_seed()
            if attempt + 1 >= _VIDEO_EDIT_WORKER_FACTORY_SEED_START_MAX_ATTEMPTS:
                raise


def _spawn_video_edit_worker_broker() -> None:
    global _VIDEO_EDIT_WORKER_BROKER_IDENTITY
    global _VIDEO_EDIT_WORKER_BROKER_PID
    global _VIDEO_EDIT_WORKER_CHANNEL

    for attempt in range(_VIDEO_EDIT_WORKER_BROKER_START_MAX_ATTEMPTS):
        seed_channel = _VIDEO_EDIT_WORKER_SEED_CHANNEL
        seed_process = _VIDEO_EDIT_WORKER_SEED_PROCESS
        if (
            seed_channel is None
            or seed_process is None
            or seed_process.poll() is not None
        ):
            raise PermissionError("trusted video-edit worker seed unavailable")

        broker_channel: Optional[socket.socket] = None
        try:
            seed_channel.sendall(b"S")
            response, broker_channel = _video_edit_worker_recv_fd_frame(
                seed_channel,
                timeout=_VIDEO_EDIT_WORKER_START_TIMEOUT_SECONDS,
            )
            broker_pid = response.get("broker_pid")
            if (
                broker_channel is None
                or response.get("broker_ready") is not True
                or not isinstance(broker_pid, int)
                or broker_pid <= 0
            ):
                raise RuntimeError(
                    str(
                        response.get("error")
                        or "trusted worker seed could not spawn broker"
                    )
                )
            try:
                identity = _capture_video_edit_worker_process_identity(broker_pid)
            except (OSError, ValueError):
                try:
                    seed_channel.sendall(b"T" + struct.pack("!Q", broker_pid))
                    _video_edit_worker_recv_frame(
                        seed_channel,
                        timeout=_VIDEO_EDIT_WORKER_CLEANUP_ACK_TIMEOUT_SECONDS,
                    )
                except (EOFError, OSError, socket.timeout, ValueError):
                    pass
                raise
            _VIDEO_EDIT_WORKER_BROKER_IDENTITY = identity
            _VIDEO_EDIT_WORKER_BROKER_PID = broker_pid
            _VIDEO_EDIT_WORKER_CHANNEL = broker_channel
            return
        except (AttributeError, EOFError, OSError, RuntimeError, TypeError, ValueError):
            if broker_channel is not None:
                try:
                    broker_channel.close()
                except OSError:
                    pass
            _discard_video_edit_worker_seed()
            if attempt + 1 >= _VIDEO_EDIT_WORKER_BROKER_START_MAX_ATTEMPTS:
                raise
            try:
                _ensure_video_edit_worker_seed_started()
            except Exception as recovery_error:
                raise RuntimeError(
                    "trusted video-edit worker seed recovery unavailable"
                ) from recovery_error


def _ensure_video_edit_worker_started() -> None:
    with _VIDEO_EDIT_WORKER_LOCK:
        _ensure_video_edit_worker_seed_started()
        if _VIDEO_EDIT_WORKER_BROKER_PID is not None or _VIDEO_EDIT_WORKER_CHANNEL is not None:
            _terminate_video_edit_worker(close_disk_trust=False)


def _cancel_video_edit_worker_idle_recycle() -> None:
    """Cancel the current idle deadline while the lifecycle lock is held."""
    global _VIDEO_EDIT_WORKER_IDLE_GENERATION
    global _VIDEO_EDIT_WORKER_IDLE_TIMER

    timer = _VIDEO_EDIT_WORKER_IDLE_TIMER
    _VIDEO_EDIT_WORKER_IDLE_TIMER = None
    _VIDEO_EDIT_WORKER_IDLE_GENERATION += 1
    if timer is not None:
        timer.cancel()


def _video_edit_worker_process_thread_count() -> int:
    """Return the native thread count used to guard the supervisor fork."""
    if sys.platform.startswith("linux"):
        try:
            return len(os.listdir("/proc/self/task"))
        except OSError:
            pass
    return threading.active_count()


def _reap_video_edit_worker_resident_tree(*, stop_supervisor: bool) -> None:
    """Reap transient workers; stop the clean supervisor only at process exit."""
    cleanup_steps = [
        (
            "broker",
            lambda: _terminate_video_edit_worker(close_disk_trust=False),
        ),
        ("seed_factory", _shutdown_video_edit_worker_seed),
    ]
    if stop_supervisor:
        cleanup_steps.append(
            ("supervisor", _discard_trusted_video_edit_worker_factory_supervisor)
        )
    for layer, cleanup in cleanup_steps:
        try:
            cleanup()
        except Exception as exc:
            logger.warning(
                "Trusted video-edit %s cleanup failed: %s",
                layer,
                type(exc).__name__,
            )


def _recycle_video_edit_worker_after_idle(generation: int) -> None:
    """Reap transient worker layers after a bounded inactive interval."""
    global _VIDEO_EDIT_WORKER_IDLE_GENERATION
    global _VIDEO_EDIT_WORKER_IDLE_TIMER

    with _VIDEO_EDIT_WORKER_LOCK:
        if (
            generation != _VIDEO_EDIT_WORKER_IDLE_GENERATION
            or _VIDEO_EDIT_WORKER_IDLE_TIMER is None
        ):
            return
        _VIDEO_EDIT_WORKER_IDLE_TIMER = None
        _VIDEO_EDIT_WORKER_IDLE_GENERATION += 1
        _reap_video_edit_worker_resident_tree(stop_supervisor=False)


def _schedule_video_edit_worker_idle_recycle() -> None:
    """Keep the trusted tree warm briefly, then release its bounded RSS."""
    global _VIDEO_EDIT_WORKER_IDLE_GENERATION
    global _VIDEO_EDIT_WORKER_IDLE_TIMER

    _cancel_video_edit_worker_idle_recycle()
    if all(
        value is None
        for value in (
            _VIDEO_EDIT_WORKER_FACTORY_SUPERVISOR,
            _VIDEO_EDIT_WORKER_FACTORY_PROCESS,
            _VIDEO_EDIT_WORKER_SEED_PROCESS,
            _VIDEO_EDIT_WORKER_BROKER_PID,
            _VIDEO_EDIT_WORKER_CHANNEL,
        )
    ):
        return
    generation = _VIDEO_EDIT_WORKER_IDLE_GENERATION
    timer = threading.Timer(
        max(float(_VIDEO_EDIT_WORKER_IDLE_TIMEOUT_SECONDS), 0.0),
        _recycle_video_edit_worker_after_idle,
        args=(generation,),
    )
    timer.daemon = True
    _VIDEO_EDIT_WORKER_IDLE_TIMER = timer
    try:
        timer.start()
    except Exception as exc:
        _VIDEO_EDIT_WORKER_IDLE_TIMER = None
        _VIDEO_EDIT_WORKER_IDLE_GENERATION += 1
        timer.cancel()
        logger.warning(
            "Trusted video-edit idle recycle timer failed: %s",
            type(exc).__name__,
        )
        _reap_video_edit_worker_resident_tree(stop_supervisor=False)


def _run_video_edit_worker(
    payload: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    with _VIDEO_EDIT_WORKER_LOCK:
        _cancel_video_edit_worker_idle_recycle()
        try:
            _ensure_video_edit_worker_started()
            _spawn_video_edit_worker_broker()
            channel = _VIDEO_EDIT_WORKER_CHANNEL
            broker_pid = _VIDEO_EDIT_WORKER_BROKER_PID
            if channel is None or _VIDEO_EDIT_WORKER_BROKER_PID is None:
                _terminate_video_edit_worker(close_disk_trust=True)
                raise RuntimeError("trusted video-edit worker unavailable")
            try:
                _video_edit_worker_send_frame(
                    channel,
                    {
                        "operation": "run",
                        "executor_timeout_seconds": max(float(timeout), 0.1),
                        **payload,
                    },
                )
                response = _video_edit_worker_recv_frame(
                    channel,
                    timeout=max(float(timeout) + 1.0, 1.1),
                )
            except socket.timeout:
                reaped = _terminate_video_edit_worker(close_disk_trust=False)
                return {
                    "stdout": "",
                    "stderr": "trusted video-edit executor timed out",
                    "returncode": 124,
                    "worker": {
                        "one_shot": True,
                        "pid": broker_pid,
                        "call_index": 1,
                        "reaped": reaped,
                    },
                }
            except (EOFError, OSError, ValueError):
                reaped = _terminate_video_edit_worker(close_disk_trust=False)
                return {
                    "stdout": "",
                    "stderr": "trusted video-edit executor terminated by memory/security limit",
                    "returncode": 1,
                    "worker": {
                        "one_shot": True,
                        "pid": broker_pid,
                        "call_index": 1,
                        "reaped": reaped,
                    },
                }
            reaped = _terminate_video_edit_worker(close_disk_trust=False)
            if not reaped:
                return {
                    "stdout": "",
                    "stderr": "trusted video-edit executor cleanup could not be verified",
                    "returncode": 1,
                    "worker": {
                        "one_shot": True,
                        "pid": broker_pid,
                        "call_index": 1,
                        "reaped": False,
                    },
                }
            if not all(key in response for key in ("stdout", "stderr", "returncode")):
                _close_video_edit_worker_disk_trust()
                raise RuntimeError("invalid trusted video-edit worker response")
            worker = response.get("worker")
            if (
                not isinstance(worker, dict)
                or worker.get("one_shot") is not True
                or worker.get("pid") != broker_pid
            ):
                _close_video_edit_worker_disk_trust()
                raise RuntimeError("invalid trusted video-edit worker identity")
            worker["reaped"] = True
            return response
        finally:
            _schedule_video_edit_worker_idle_recycle()


def _stop_video_edit_worker() -> None:
    with _VIDEO_EDIT_WORKER_LOCK:
        _cancel_video_edit_worker_idle_recycle()
        _reap_video_edit_worker_resident_tree(stop_supervisor=True)


def _close_video_edit_worker_disk_trust() -> None:
    global _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED

    with _VIDEO_EDIT_WORKER_LOCK:
        _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED = True


def _ensure_sensitive_runtime_boundary() -> bool:
    """Close process inspection before trusted tokens can enter memory.

    Disk trust controls future source reads, not this already-imported OS
    primitive. A frozen worker image may establish the process boundary after
    an ordinary terminal command has closed all filesystem trust.
    """
    global _MODEL_DESCENDANT_PTRACE_BOUNDARY
    global _SENSITIVE_PROCESS_OS_BOUNDARY

    if _SENSITIVE_PROCESS_OS_BOUNDARY and _MODEL_DESCENDANT_PTRACE_BOUNDARY:
        return True
    boundary = harden_sensitive_process(no_new_privs=False, drop_ptrace=True)
    _SENSITIVE_PROCESS_OS_BOUNDARY = boundary
    _MODEL_DESCENDANT_PTRACE_BOUNDARY = boundary
    return boundary


def _late_prepare_video_edit_worker_before_terminal() -> None:
    """Freeze trusted code and fork its supervisor before gateway threads start."""
    if not os.environ.get("ZETTLAB_PRESETS_DIR"):
        return
    with _VIDEO_EDIT_WORKER_LOCK:
        if _VIDEO_EDIT_WORKER_DISK_TRUST_CLOSED:
            return
        snapshot = _trusted_video_edit_worker_snapshot_for_seed_start()
        _trusted_video_edit_worker_factory_image(snapshot)
        _preload_trusted_runtime_source_bundles()
        _trusted_video_edit_worker_factory_bootstrap(snapshot)


atexit.register(_stop_video_edit_worker)

# Production gateways set the pinned presets root before tool discovery. Freeze
# the trusted image and create its single-threaded supervisor before gateway
# adapters start threads. Transient factory/seed/broker processes remain lazy.
if os.environ.get("ZETTLAB_PRESETS_DIR"):
    try:
        _late_prepare_video_edit_worker_before_terminal()
    except Exception as exc:
        logger.warning(
            "Trusted video-edit image failed during tool initialization: %s",
            type(exc).__name__,
        )


def _run_connector_runtime_command_if_allowed(
    command: str,
    *,
    cwd: str,
    timeout: int,
) -> Optional[str]:
    parsed = _parse_connector_runtime_command(command)
    if parsed is None:
        return _connector_runtime_shell_guard_result(command)
    if not _ensure_sensitive_runtime_boundary():
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": "Connector runtime process memory boundary is unavailable",
            "connector_runtime_direct": True,
        }, ensure_ascii=False)
    argv = parsed.argv

    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    script = Path(argv[1])
    expected_digest: Optional[str] = None
    try:
        relative_script = script.relative_to(anchor.resolved_root).as_posix()
        expected_digest = anchor.file_digests.get(relative_script)
        identities_match = (
            anchor is not None
            and expected_digest is not None
            and _path_identity(anchor.resolved_root) == parsed.root_identity
            and _path_identity(script) == parsed.script_identity
            and _connector_runtime_path_is_trusted(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
        )
    except OSError:
        identities_match = False
    if not identities_match:
        reason = None
        if anchor is not None:
            reason = _connector_runtime_trust_rejection_reason(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
        _log_connector_runtime_rejection(reason or "identity_changed_before_exec")
        return None
    try:
        script_bytes = _read_connector_runtime_script_bytes(
            script,
            expected_identity=parsed.script_identity,
            expected_digest=expected_digest,
        )
    except OSError:
        _log_connector_runtime_rejection("script_snapshot_failed")
        return None

    secret_values: list[str] = []
    try:
        from tools.environments.local import build_connector_runtime_env

        connector_env = build_connector_runtime_env()
        from tools.environments.local import _sanitize_subprocess_env

        run_env = _sanitize_subprocess_env(os.environ)
        run_env.pop("PYTHONPATH", None)
        secret_values = [
            connector_env.get("ZETTLAB_CONNECTORS_AUTH_TOKEN", ""),
            connector_env.get("ZETTLAB_CONNECTORS_URL", ""),
        ]
        run_cwd = cwd if cwd and os.path.isdir(cwd) else os.getcwd()
        from tools.trusted_direct_runner import run_trusted_python_script

        completed = run_trusted_python_script(
            script=script,
            argv=argv[1:],
            cwd=Path(run_cwd),
            base_env=run_env,
            injected_env=connector_env,
            timeout=timeout,
            secret_values=secret_values,
            script_bytes=script_bytes,
        )
        return _connector_runtime_result_json(
            command=command,
            output=completed.output,
            returncode=completed.returncode,
            secret_values=secret_values,
            timed_out=completed.timed_out,
        )
    except Exception as e:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Connector runtime execution failed: {type(e).__name__}: {e}",
            "connector_runtime_direct": True,
        }, ensure_ascii=False)


_AGENT_CREATOR_SCRIPT = "create_agent.py"
_AGENT_CREATOR_RELATIVE_PATH = Path(
    "skills/agent-creator/scripts/create_agent.py"
)
_AGENT_CREATOR_MANIFEST_RELATIVE_PATH = Path(
    "skills/agent-creator/manifest.yaml"
)
_AGENT_CREATOR_ACTION_TOKEN_FD_CAPABILITY = (
    "zettlab.agent_action_token_fd.v1"
)
_AGENT_CREATOR_MAX_PAYLOAD_BYTES = 1024 * 1024
_AGENTCOMPUTER_MAX_STDIN_BYTES = 4 * 1024 * 1024
_AGENT_CREATOR_MAX_SCRIPT_BYTES = 1024 * 1024
_AGENT_CREATOR_MAX_MANIFEST_BYTES = 64 * 1024
_AGENT_CREATOR_MAX_RUNTIME_CAPABILITIES = 32
_AGENT_CREATOR_PAYLOAD_KEYS = frozenset({
    "name",
    "soul_identity",
    "soul_style",
    "greeting",
    "user_entries",
    "memory_entries",
})
_AGENTCOMPUTER_CLI_VALUE_FLAGS = {
    ("file", "list"): frozenset({"--path", "--offset", "--limit"}),
    ("file", "stat"): frozenset({"--path"}),
    ("file", "read"): frozenset({"--path", "--offset", "--limit"}),
    ("file", "write"): frozenset({"--path"}),
    ("file", "mkdir"): frozenset({"--path"}),
    ("file", "rename"): frozenset({"--source", "--target"}),
    ("file", "copy"): frozenset({"--source", "--target"}),
    ("file", "move"): frozenset({"--source", "--target"}),
    ("file", "delete"): frozenset({"--path"}),
    ("file", "search"): frozenset({"--path", "--query", "--limit"}),
    ("system", "status"): frozenset(),
    ("system", "overview"): frozenset(),
    ("system", "device"): frozenset(),
    ("system", "pools"): frozenset(),
    ("system", "disks"): frozenset(),
    ("system", "smart-status"): frozenset({"--device"}),
    ("system", "smart-info"): frozenset({"--device"}),
    ("system", "network"): frozenset(),
    ("system", "time"): frozenset(),
}
_AGENTCOMPUTER_CLI_BOOL_FLAGS = {
    ("file", "write"): frozenset({"--stdin", "--overwrite", "--parents"}),
    ("file", "mkdir"): frozenset({"--parents"}),
    ("file", "copy"): frozenset({"--overwrite"}),
    ("file", "move"): frozenset({"--overwrite"}),
}
_AGENTCOMPUTER_CLI_REQUIRED_FLAGS = {
    ("file", "stat"): frozenset({"--path"}),
    ("file", "read"): frozenset({"--path"}),
    ("file", "write"): frozenset({"--path", "--stdin"}),
    ("file", "mkdir"): frozenset({"--path"}),
    ("file", "rename"): frozenset({"--source", "--target"}),
    ("file", "copy"): frozenset({"--source", "--target"}),
    ("file", "move"): frozenset({"--source", "--target"}),
    ("file", "delete"): frozenset({"--path"}),
    ("file", "search"): frozenset({"--query"}),
    ("system", "smart-status"): frozenset({"--device"}),
    ("system", "smart-info"): frozenset({"--device"}),
}
_AGENTCOMPUTER_CLI_PATH_FLAGS = frozenset({"--path", "--source", "--target"})
_AGENTCOMPUTER_CLI_MUTATIONS = frozenset({
    ("file", "write"),
    ("file", "mkdir"),
    ("file", "copy"),
    ("file", "rename"),
    ("file", "move"),
    ("file", "delete"),
})
_AGENT_CREATOR_HEREDOC_RE = re.compile(
    r"^(?P<command>.+?)\s+<<\s*"
    r"(?P<quote>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]{0,31})(?P=quote)\s*$"
)


@dataclass(frozen=True)
class _AgentCreatorCommand:
    argv: list[str]
    root_identity: tuple[int, int]
    script_identity: tuple[int, int]
    stdin_text: Optional[str]
    approval_operation: Optional[str]


def _agent_creator_blocked_result(
    code: str,
    message: str,
    *,
    direct: bool = False,
) -> str:
    return json.dumps({
        "output": "",
        "exit_code": 2,
        "error": message,
        "errorCode": code,
        "status": "error",
        "agent_creator_direct": direct,
        "agent_creator_blocked": True,
    }, ensure_ascii=False)


def _agent_creator_shell_guard_result(command: str) -> Optional[str]:
    """Fail closed when a reserved creator invocation is not exactly allowed."""

    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION:
            segments.append([])
            continue
        segments[-1].append(token)

    contains_creator_invocation = any(
        _agent_creator_segment_contains_invocation(segment)
        for segment in segments
    )
    if not contains_creator_invocation:
        contains_creator_invocation = any(
            _agent_creator_segment_contains_nested_shell_invocation(segment)
            for segment in segments
        )
    if not contains_creator_invocation:
        return None
    return _agent_creator_blocked_result(
        "agent_creator_command_blocked",
        (
            "Agent Creator must run as one direct Python invocation of the "
            "canonical presets script. Only preflight or create --payload "
            "with a bounded JSON object is allowed; wrappers, non-canonical "
            "paths, extra arguments, and shell operators are rejected."
        ),
    )


@dataclass(frozen=True)
class _VideoEditRuntimeCommand:
    argv: list[str]
    root_identity: tuple[int, int]
    script_identity: tuple[int, int]


@dataclass(frozen=True)
class _CameraRuntimeCommand:
    argv: list[str]
    root_identity: tuple[int, int]
    script_identity: tuple[int, int]


def _video_edit_runtime_timeout(
    parsed: _VideoEditRuntimeCommand,
    requested_timeout: int,
) -> int:
    if (
        Path(parsed.argv[1]).name == "cloud_render_business.py"
        and _cloud_render_business_subcommand(parsed.argv[2:]) == "upload"
    ):
        return max(requested_timeout, _VIDEO_EDIT_UPLOAD_TIMEOUT_SECONDS)
    if (
        Path(parsed.argv[1]).name == "proactive_video.py"
        and _cloud_render_business_subcommand(parsed.argv[2:]) == "upload"
    ):
        return max(requested_timeout, _PROACTIVE_VIDEO_UPLOAD_TIMEOUT_SECONDS)
    return requested_timeout


def _cloud_render_business_subcommand(arguments: list[str]) -> Optional[str]:
    """Return the argparse subcommand after supported global options."""
    position = 0
    options_with_value = {"--agent-id", "--base-url", "--timeout"}
    while position < len(arguments):
        token = arguments[position]
        if token == "--":
            position += 1
            break
        option_name = token.split("=", 1)[0]
        if option_name not in options_with_value:
            break
        position += 1
        if "=" not in token:
            if position >= len(arguments):
                return None
            position += 1
    if position >= len(arguments) or arguments[position].startswith("-"):
        return None
    return arguments[position]


_PROACTIVE_VIDEO_MANIFEST_ID_RE = re.compile(r"pvm_[A-Za-z0-9_-]{32}")


def _exact_cli_option(
    arguments: list[str],
    position: int,
    name: str,
) -> tuple[Optional[str], int]:
    """Read one exact long option without argparse abbreviations."""
    if position >= len(arguments):
        return None, position
    token = arguments[position]
    if token == name:
        if position + 1 >= len(arguments):
            return None, position
        return arguments[position + 1], position + 2
    prefix = name + "="
    if token.startswith(prefix):
        return token[len(prefix) :], position + 1
    return None, position


def _proactive_video_arguments_match_receipt(
    arguments: list[str],
    *,
    expected_agent_id: str,
    turn_id: str,
) -> bool:
    """Allow only the manifest-bound proactive helper CLI grammar."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", expected_agent_id):
        return False
    agent_id, position = _exact_cli_option(arguments, 0, "--agent-id")
    if agent_id != expected_agent_id or position >= len(arguments):
        return False
    subcommand = arguments[position]
    if subcommand not in {"resolve", "upload", "create-project", "report"}:
        return False
    manifest_id, position = _exact_cli_option(
        arguments,
        position + 1,
        "--manifest-id",
    )
    if not manifest_id or not _PROACTIVE_VIDEO_MANIFEST_ID_RE.fullmatch(manifest_id):
        return False
    if subcommand != "report":
        return position == len(arguments)
    workflow_state, position = _exact_cli_option(
        arguments,
        position,
        "--workflow-state",
    )
    expected_state = _proactive_video_workflow_state_path(
        expected_agent_id,
        turn_id,
    )
    return workflow_state == expected_state and position == len(arguments)


def _proactive_video_workflow_state_path(agent_id: str, turn_id: str) -> str:
    return str(
        Path("/volume1/subvol/agents/data")
        / agent_id
        / "output"
        / f"proactive-{turn_id}"
        / ".video-edit-workflow-mini"
        / "workflow_state.json"
    )


def _proactive_preference_finalizer_arguments_match_receipt(
    arguments: list[str],
    *,
    expected_agent_id: str,
    turn_id: str,
) -> bool:
    """Accept the one no-memory finalizer form owned by proactive runs."""
    return arguments == [
        "finalize-success",
        "--workflow-state",
        _proactive_video_workflow_state_path(expected_agent_id, turn_id),
        "--memory-commit-state",
        "skipped",
        "--sidecar-state",
        "skipped",
    ]


def _video_edit_runtime_claims_match_receipt(
    parsed: _VideoEditRuntimeCommand,
    trusted_env: Mapping[str, str],
) -> bool:
    """Bind model-supplied business routing claims to the frozen receipt."""
    script_name = Path(parsed.argv[1]).name
    execution_policy = str(
        trusted_env.get("HERMES_EXECUTION_POLICY", "") or ""
    ).strip().lower()
    turn_id = str(trusted_env.get("HERMES_TURN_ID", "") or "").strip()
    gateway_session_key = str(
        trusted_env.get("HERMES_GATEWAY_SESSION_KEY", "") or ""
    ).strip()
    proactive_receipt = bool(
        re.fullmatch(r"pvm-[0-9a-f]{24}", turn_id)
        and gateway_session_key == f"proactive-{turn_id}"
    )
    expected_agent_id = str(trusted_env.get("ZET_AGENT_ID", "") or "").strip()
    if script_name == "proactive_video.py":
        return proactive_receipt and _proactive_video_arguments_match_receipt(
            parsed.argv[2:],
            expected_agent_id=expected_agent_id,
            turn_id=turn_id,
        )
    if execution_policy == "silent_automation":
        # Silent turns have no interactive fallback.  Every terminal operation
        # must stay inside the proactive manifest wrapper; otherwise a valid
        # proof with an ordinary/stale session key could reach the legacy
        # normalize, preference, or cloud upload helpers.
        if script_name == "preference_resolver.py" and proactive_receipt:
            return _proactive_preference_finalizer_arguments_match_receipt(
                parsed.argv[2:],
                expected_agent_id=expected_agent_id,
                turn_id=turn_id,
            )
        return False
    if proactive_receipt:
        if script_name == "normalize.py":
            return False
        if script_name == "preference_resolver.py":
            # The proactive wrapper owns preference reads and upload strategy.
            # The model only needs the deterministic success finalizer after
            # the verified download; every other resolver subcommand could
            # re-read or mutate user memory outside the frozen manifest flow.
            return _proactive_preference_finalizer_arguments_match_receipt(
                parsed.argv[2:],
                expected_agent_id=expected_agent_id,
                turn_id=turn_id,
            )
        if (
            script_name == "cloud_render_business.py"
            and _cloud_render_business_subcommand(parsed.argv[2:])
            in {"upload", "create-project"}
        ):
            return False

    if script_name != "cloud_render_business.py":
        return True

    if not expected_agent_id:
        return False

    agent_ids: list[str] = []
    arguments = parsed.argv[2:]
    position = 0
    while position < len(arguments):
        token = arguments[position]
        option_name = token.split("=", 1)[0]
        if option_name.startswith("--") and "--base-url".startswith(option_name):
            # argparse accepts unambiguous long-option abbreviations by
            # default, so --base and --base=<url> are equivalent to the
            # forbidden --base-url override inside the signed helper.
            return False
        if (
            option_name != "--agent-id"
            and option_name.startswith("--")
            and "--agent-id".startswith(option_name)
        ):
            # An abbreviated second declaration (for example --agent) would
            # be accepted by argparse and could override the exact bound ID.
            return False
        if token == "--agent-id":
            position += 1
            if position >= len(arguments):
                return False
            agent_ids.append(arguments[position])
        elif token.startswith("--agent-id="):
            agent_ids.append(token.split("=", 1)[1])
        position += 1

    # The signed workflow always names its profile explicitly. The base URL is
    # deliberately not model-configurable; the signed helper owns the loopback
    # business endpoint default.
    return (
        len(agent_ids) == 1
        and agent_ids[0] == expected_agent_id
    )


def _resolve_video_edit_runtime_script(raw_path: str) -> Optional[Path]:
    anchor = _capture_connector_runtime_root()
    if anchor is None:
        return None
    relative_text: Optional[str] = None
    for prefix in ("$ZETTLAB_PRESETS_DIR/", "${ZETTLAB_PRESETS_DIR}/"):
        if raw_path.startswith(prefix):
            relative_text = raw_path[len(prefix):]
            break
    if relative_text is None:
        expanded = Path(os.path.expandvars(os.path.expanduser(raw_path))).absolute()
        for allowed_root in (anchor.configured_root, anchor.resolved_root):
            try:
                relative_text = str(expanded.relative_to(allowed_root))
                break
            except ValueError:
                continue
    if relative_text is None:
        return None

    expected_prefix = Path("skills/video-edit-workflow-mini/scripts")
    relative = Path(relative_text)
    if relative.parent != expected_prefix or relative.name not in _VIDEO_EDIT_RUNTIME_SCRIPTS:
        return None
    candidate = anchor.resolved_root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(anchor.resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or not _connector_runtime_path_is_trusted(
        candidate,
        anchor.resolved_root,
        expected_root_identity=anchor.identity,
    ):
        return None
    return resolved


def _parse_video_edit_runtime_command(command: str) -> Optional[_VideoEditRuntimeCommand]:
    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    if len(tokens) < 2 or not _is_python_executable_token(tokens[0]):
        return None
    if any(
        token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION
        for token in tokens
    ):
        return None
    if Path(tokens[1]).name not in _VIDEO_EDIT_RUNTIME_SCRIPTS:
        return None
    script = _resolve_video_edit_runtime_script(tokens[1])
    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if script is None or anchor is None:
        return None
    try:
        script_identity = _path_identity(script)
    except OSError:
        return None
    return _VideoEditRuntimeCommand(
        argv=[sys.executable, str(script), *tokens[2:]],
        root_identity=anchor.identity,
        script_identity=script_identity,
    )


def _resolve_camera_runtime_script(raw_path: str) -> Optional[Path]:
    anchor = _capture_connector_runtime_root()
    if anchor is None:
        return None
    relative_text: Optional[str] = None
    for prefix in ("$ZETTLAB_PRESETS_DIR/", "${ZETTLAB_PRESETS_DIR}/"):
        if raw_path.startswith(prefix):
            relative_text = raw_path[len(prefix):]
            break
    if relative_text is None:
        expanded = Path(os.path.expandvars(os.path.expanduser(raw_path))).absolute()
        for allowed_root in (anchor.configured_root, anchor.resolved_root):
            try:
                relative_text = str(expanded.relative_to(allowed_root))
                break
            except ValueError:
                continue
    if relative_text is None or Path(relative_text) != _CAMERA_RUNTIME_RELATIVE_PATH:
        return None
    candidate = anchor.resolved_root / _CAMERA_RUNTIME_RELATIVE_PATH
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(anchor.resolved_root)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or not _connector_runtime_path_is_trusted(
        candidate,
        anchor.resolved_root,
        expected_root_identity=anchor.identity,
    ):
        return None
    return resolved


def _camera_runtime_arguments_allowed(arguments: list[str]) -> bool:
    if arguments == ["list"]:
        return True
    if (
        len(arguments) == 3
        and arguments[0] in {"snap", "doctor"}
        and arguments[1] == "--camera-id"
        and _CAMERA_ID_RE.fullmatch(arguments[2]) is not None
    ):
        return True
    if (
        len(arguments) in {3, 5}
        and arguments[0] == "clip"
        and arguments[1] == "--camera-id"
        and _CAMERA_ID_RE.fullmatch(arguments[2]) is not None
    ):
        if len(arguments) == 3:
            return True
        return (
            arguments[3] == "--duration"
            and arguments[4].isdigit()
            and 1 <= int(arguments[4]) <= 60
        )
    return False


def _parse_camera_runtime_command(command: str) -> Optional[_CameraRuntimeCommand]:
    lexer = shlex.shlex(
        command.strip(),
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    if (
        len(tokens) < 3
        or not _is_python_executable_token(tokens[0])
        or Path(tokens[1]).name != _CAMERA_RUNTIME_SCRIPT
        or any(
            token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION
            for token in tokens
        )
        or not _camera_runtime_arguments_allowed(tokens[2:])
    ):
        return None
    script = _resolve_camera_runtime_script(tokens[1])
    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if script is None or anchor is None:
        return None
    try:
        script_identity = _path_identity(script)
    except OSError:
        return None
    return _CameraRuntimeCommand(
        argv=[sys.executable, str(script), *tokens[2:]],
        root_identity=anchor.identity,
        script_identity=script_identity,
    )


def _camera_runtime_manifest_allows(anchor: _ConnectorRuntimeRootAnchor) -> bool:
    manifest = anchor.resolved_root / _CAMERA_RUNTIME_MANIFEST_RELATIVE_PATH
    try:
        manifest_digest = anchor.file_digests.get(
            _CAMERA_RUNTIME_MANIFEST_RELATIVE_PATH.as_posix()
        )
        if manifest_digest is None or not _connector_runtime_path_is_trusted(
            manifest,
            anchor.resolved_root,
            expected_root_identity=anchor.identity,
        ):
            return False
        raw = _read_connector_runtime_script_bytes(
            manifest,
            expected_identity=_path_identity(manifest),
            expected_digest=manifest_digest,
        )
        if len(raw) > _CAMERA_RUNTIME_MAX_MANIFEST_BYTES:
            return False
        import yaml

        loaded = yaml.safe_load(raw.decode("utf-8"))
        return bool(
            isinstance(loaded, dict)
            and loaded.get("id") == "camsnap"
            and loaded.get("required_scopes") == ["hardware.camera:read"]
            and _CAMERA_RUNTIME_CAPABILITY
            in (loaded.get("runtime_capabilities") or [])
        )
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def _camera_runtime_shell_guard_result(command: str) -> Optional[str]:
    if _CAMERA_RUNTIME_SCRIPT not in command:
        return None
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": (
            "Camera actions must run as one exact foreground Python helper "
            "command with a registered camera_id and no shell operators, "
            "wrappers, host, credential, URL, output path, discovery, or watch input."
        ),
        "camera_runtime_direct": False,
        "camera_runtime_blocked": True,
    }, ensure_ascii=False)


def _agent_creator_segment_contains_invocation(segment: list[str]) -> bool:
    """Recognize creator scripts only where the shell would execute them."""

    for index, token in enumerate(segment):
        if Path(token).name != _AGENT_CREATOR_SCRIPT:
            continue
        if index == 0 or _connector_runtime_command_prefix_is_supported(
            segment[:index]
        ):
            return True

    for index, token in enumerate(segment):
        if not _is_python_executable_token(token):
            continue
        script_index = _connector_runtime_python_script_index(
            segment,
            index,
            script_name=_AGENT_CREATOR_SCRIPT,
        )
        if script_index is None:
            continue
        if index > 0 and not _connector_runtime_command_prefix_is_supported(
            segment[:index]
        ):
            continue
        return True
    return False


def _agent_creator_segment_contains_nested_shell_invocation(
    segment: list[str],
    *,
    nested_shell_depth: int = 0,
) -> bool:
    if nested_shell_depth >= _CONNECTOR_RUNTIME_NESTED_SHELL_DEPTH:
        return False
    for index, token in enumerate(segment):
        if Path(token).name.lower() not in _CONNECTOR_RUNTIME_COMMAND_SHELLS:
            continue
        if index > 0 and not _connector_runtime_command_prefix_is_supported(
            segment[:index]
        ):
            continue
        nested_command = _connector_runtime_shell_command_argument(
            segment[index + 1:]
        )
        if nested_command is None:
            continue
        lexer = shlex.shlex(
            nested_command.strip(),
            posix=True,
            punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
        )
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            nested_tokens = list(lexer)
        except ValueError:
            continue
        nested_segments: list[list[str]] = [[]]
        for nested_token in nested_tokens:
            if (
                nested_token
                and set(nested_token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION
            ):
                nested_segments.append([])
                continue
            nested_segments[-1].append(nested_token)
        if any(
            _agent_creator_segment_contains_invocation(nested_segment)
            for nested_segment in nested_segments
        ):
            return True
        if any(
            _agent_creator_segment_contains_nested_shell_invocation(
                nested_segment,
                nested_shell_depth=nested_shell_depth + 1,
            )
            for nested_segment in nested_segments
        ):
            return True
    return False


def _log_agent_creator_rejection(reason: str) -> None:
    logger.warning(
        "Agent Creator direct runner rejected: reason=%s",
        reason,
    )


def _resolve_agent_creator_script(raw_path: str) -> Optional[Path]:
    """Resolve only the fixed creator script below the pinned presets root."""

    anchor = _capture_connector_runtime_root()
    if anchor is None:
        return None

    expected = _AGENT_CREATOR_RELATIVE_PATH
    relative: Optional[Path] = None
    for prefix in ("$ZETTLAB_PRESETS_DIR/", "${ZETTLAB_PRESETS_DIR}/"):
        if raw_path.startswith(prefix):
            relative = Path(raw_path[len(prefix):])
            break
    else:
        supplied = Path(raw_path)
        if raw_path == expected.as_posix():
            relative = expected
        elif not supplied.is_absolute():
            return None
        else:
            expanded = Path(
                os.path.expandvars(os.path.expanduser(raw_path))
            ).absolute()
            for allowed_root in (anchor.configured_root, anchor.resolved_root):
                try:
                    relative = expanded.relative_to(allowed_root)
                    break
                except ValueError:
                    continue

    if relative is None or relative != expected or ".." in relative.parts:
        return None

    candidate = anchor.resolved_root / expected
    try:
        resolved = candidate.resolve(strict=True)
        if resolved.relative_to(anchor.resolved_root) != expected:
            return None
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    if not _connector_runtime_path_is_trusted(
        candidate,
        anchor.resolved_root,
        expected_root_identity=anchor.identity,
    ):
        reason = _connector_runtime_trust_rejection_reason(
            candidate,
            anchor.resolved_root,
            expected_root_identity=anchor.identity,
        )
        _log_agent_creator_rejection(reason or "trust_check_failed")
        return None
    return resolved


def _split_agent_creator_heredoc(
    command: str,
) -> tuple[str, Optional[str]]:
    """Return the direct command line and optional validated heredoc body."""

    if "\n" not in command:
        return command.strip(), None
    first_line, remainder = command.split("\n", 1)
    first_line = first_line.rstrip("\r")
    match = _AGENT_CREATOR_HEREDOC_RE.fullmatch(first_line)
    if match is None:
        raise ValueError("unsupported stdin shape")

    delimiter = match.group("delimiter")
    suffix = f"\n{delimiter}"
    if remainder.endswith(suffix + "\n"):
        payload = remainder[: -len(suffix + "\n")]
    elif remainder.endswith(suffix):
        payload = remainder[: -len(suffix)]
    else:
        raise ValueError("missing heredoc terminator")
    if not payload:
        raise ValueError("empty payload")
    return match.group("command").strip(), payload


def _validate_agent_creator_payload(payload: str) -> str:
    if len(payload.encode("utf-8")) > _AGENT_CREATOR_MAX_PAYLOAD_BYTES:
        raise ValueError("payload too large")

    def reject_constant(value: str):
        raise ValueError(f"invalid JSON constant {value}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("payload must be one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("payload must be one JSON object")
    if any(
        not isinstance(key, str) or key not in _AGENT_CREATOR_PAYLOAD_KEYS
        for key in value
    ):
        raise ValueError("payload contains unsupported fields")
    try:
        normalized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("payload must be one JSON object") from exc
    if len(normalized.encode("utf-8")) > _AGENT_CREATOR_MAX_PAYLOAD_BYTES:
        raise ValueError("payload too large")
    return normalized


def _validate_agentcomputer_cli_args(args: list[str]) -> bool:
    """Validate the exact zettctl grammar before any secret is acquired."""

    if len(args) < 2 or len(args) > 24:
        raise ValueError("unsupported AgentComputer CLI command")
    command = (args[0], args[1])
    value_flags = _AGENTCOMPUTER_CLI_VALUE_FLAGS.get(command)
    if value_flags is None:
        raise ValueError("unsupported AgentComputer CLI command")
    bool_flags = _AGENTCOMPUTER_CLI_BOOL_FLAGS.get(command, frozenset())
    seen: set[str] = set()
    index = 2
    while index < len(args):
        flag = args[index]
        if flag in seen:
            raise ValueError("duplicate AgentComputer CLI flag")
        if flag in bool_flags:
            seen.add(flag)
            index += 1
            continue
        if flag not in value_flags or index + 1 >= len(args):
            raise ValueError("unsupported AgentComputer CLI flag")
        value = args[index + 1]
        if (
            not value
            or value.startswith("--")
            or "\x00" in value
            or len(value.encode("utf-8")) > 4096
        ):
            raise ValueError("invalid AgentComputer CLI value")
        if flag in _AGENTCOMPUTER_CLI_PATH_FLAGS:
            parts = value.split("/")
            if value.startswith("/") or "\\" in value or ".." in parts:
                raise ValueError("AgentComputer file paths must be workspace-relative")
        seen.add(flag)
        index += 2
    required = _AGENTCOMPUTER_CLI_REQUIRED_FLAGS.get(command, frozenset())
    if not required.issubset(seen):
        raise ValueError("required AgentComputer CLI flag is missing")
    return command == ("file", "write")


def _parse_agent_creator_command(command: str) -> Optional[_AgentCreatorCommand]:
    try:
        command_line, heredoc_payload = _split_agent_creator_heredoc(command)
    except ValueError:
        return None

    lexer = shlex.shlex(
        command_line,
        posix=True,
        punctuation_chars=_CONNECTOR_RUNTIME_SHELL_PUNCTUATION_TEXT,
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    if len(tokens) < 3 or not _is_python_executable_token(tokens[0]):
        return None
    if any(
        token and set(token) <= _CONNECTOR_RUNTIME_SHELL_PUNCTUATION
        for token in tokens
    ):
        return None
    if Path(tokens[1]).name != _AGENT_CREATOR_SCRIPT:
        return None

    script = _resolve_agent_creator_script(tokens[1])
    if script is None:
        return None

    args = tokens[2:]
    stdin_text: Optional[str] = None
    approval_operation: Optional[str] = None
    if args in (["preflight"], ["list"]):
        if heredoc_payload is not None:
            return None
    elif len(args) == 3 and args[:2] == ["create", "--payload"]:
        if args[2] == "-":
            if heredoc_payload is None:
                return None
            try:
                stdin_text = _validate_agent_creator_payload(heredoc_payload) + "\n"
            except ValueError:
                return None
        else:
            if heredoc_payload is not None:
                return None
            try:
                args[2] = _validate_agent_creator_payload(args[2])
            except ValueError:
                return None
        approval_operation = "agent.create"
    elif args and args[0] == "cli":
        cli_args = args[1:]
        try:
            requires_stdin = _validate_agentcomputer_cli_args(cli_args)
        except (UnicodeEncodeError, ValueError):
            return None
        cli_operation = tuple(cli_args[:2])
        if cli_operation in _AGENTCOMPUTER_CLI_MUTATIONS:
            approval_operation = ".".join(cli_operation)
        if requires_stdin:
            if heredoc_payload is None:
                return None
            try:
                payload_size = len(heredoc_payload.encode("utf-8"))
            except UnicodeEncodeError:
                return None
            if (
                payload_size == 0
                or payload_size > _AGENTCOMPUTER_MAX_STDIN_BYTES
                or "\x00" in heredoc_payload
            ):
                return None
            stdin_text = heredoc_payload
        elif heredoc_payload is not None:
            return None
    else:
        return None

    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if anchor is None:
        return None
    try:
        script_identity = _path_identity(script)
    except OSError:
        return None
    return _AgentCreatorCommand(
        argv=[sys.executable, str(script), *args],
        root_identity=anchor.identity,
        script_identity=script_identity,
        stdin_text=stdin_text,
        approval_operation=approval_operation,
    )


def _request_agentcomputer_mutation_approval(
    parsed: _AgentCreatorCommand,
) -> Optional[str]:
    """Require a fresh human decision before a data-changing CLI operation."""

    operation = parsed.approval_operation
    if operation is None:
        return None

    from tools.approval import request_tool_approval

    fingerprint = hashlib.sha256()
    for value in parsed.argv[2:]:
        encoded = value.encode("utf-8")
        fingerprint.update(len(encoded).to_bytes(8, "big"))
        fingerprint.update(encoded)
    stdin_bytes = (parsed.stdin_text or "").encode("utf-8")
    fingerprint.update(len(stdin_bytes).to_bytes(8, "big"))
    fingerprint.update(stdin_bytes)

    fingerprint_hex = fingerprint.hexdigest()
    shell_argv = shlex.join(["agentcomputer", *parsed.argv[2:]])
    shell_argv_sha256 = hashlib.sha256(shell_argv.encode("utf-8")).hexdigest()
    max_argv_display_chars = 2048
    if len(shell_argv) > max_argv_display_chars:
        shell_argv_display = (
            shell_argv[:max_argv_display_chars]
            + "\n[argv display truncated: "
            + f"chars={len(shell_argv)} sha256={shell_argv_sha256}]"
        )
    else:
        shell_argv_display = shell_argv
    stdin_sha256 = hashlib.sha256(stdin_bytes).hexdigest()
    stdin_text = parsed.stdin_text or ""
    max_preview_chars = 512
    if len(stdin_text) <= max_preview_chars:
        stdin_preview = json.dumps(stdin_text, ensure_ascii=True)
    else:
        head_chars = 320
        tail_chars = 128
        omitted_chars = len(stdin_text) - head_chars - tail_chars
        stdin_preview = (
            json.dumps(stdin_text[:head_chars], ensure_ascii=True)
            + "\n[stdin preview truncated: "
            + f"chars={len(stdin_text)} omitted={omitted_chars}]\n"
            + json.dumps(stdin_text[-tail_chars:], ensure_ascii=True)
        )
    display_target = (
        f"argv: {shell_argv_display}\n"
        f"stdin: bytes={len(stdin_bytes)} sha256={stdin_sha256}\n"
        f"stdin preview: {stdin_preview}\n"
        f"approval fingerprint: sha256={fingerprint_hex}"
    )

    approval = request_tool_approval(
        "agentcomputer_cli",
        f"AgentComputer {operation} modifies AgentComputer user data.",
        rule_key=(
            f"agentcomputer:{operation}:{fingerprint_hex}"
        ),
        approval_callback=_get_approval_callback(),
        one_shot=True,
        allow_yolo_bypass=False,
        display_target=display_target,
    )
    if approval.get("approved"):
        return None

    pending = approval.get("status") in {
        "approval_required",
        "pending_approval",
    }
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": "" if pending else approval.get(
            "message",
            f"AgentComputer {operation} was not approved.",
        ),
        "status": "pending_approval" if pending else "blocked",
        "approval_pending": pending,
        "approval_id": approval.get("approval_id"),
        "command": approval.get("command", f"agentcomputer {operation}"),
        "description": approval.get(
            "description",
            f"AgentComputer {operation} modifies AgentComputer user data.",
        ),
        "pattern_key": approval.get("pattern_key", f"agentcomputer:{operation}"),
        "smart_denied": approval.get("smart_denied", False),
        "allow_permanent": False,
        "agent_creator_direct": True,
    }, ensure_ascii=False)


def _read_verified_agent_creator_file(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    max_bytes: int,
    expected_digest: Optional[str] = None,
) -> bytes:
    """Freeze one trusted regular file through a non-following descriptor."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        file_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or (file_stat.st_dev, file_stat.st_ino) != expected_identity
            or file_stat.st_size > max_bytes
        ):
            raise OSError("agent creator trusted file identity invalid")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, max_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise OSError("agent creator trusted file too large")
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != expected_identity
            or after.st_size != file_stat.st_size
            or len(payload) != file_stat.st_size
            or (
                expected_digest is not None
                and hashlib.sha256(payload).hexdigest() != expected_digest
            )
        ):
            raise OSError("agent creator trusted file changed")
        return payload
    finally:
        os.close(descriptor)


def _read_verified_agent_creator_script(
    script: Path,
    *,
    expected_identity: tuple[int, int],
    expected_digest: Optional[str] = None,
) -> bytes:
    """Freeze the verified script source before any scoped secret is injected."""

    return _read_verified_agent_creator_file(
        script,
        expected_identity=expected_identity,
        max_bytes=_AGENT_CREATOR_MAX_SCRIPT_BYTES,
        expected_digest=expected_digest,
    )


def _agent_creator_manifest_supports_action_token_fd(
    anchor: _ConnectorRuntimeRootAnchor,
) -> bool:
    """Validate the preset ABI before acquiring or injecting a scoped token."""

    manifest = anchor.resolved_root / _AGENT_CREATOR_MANIFEST_RELATIVE_PATH
    if not _connector_runtime_path_is_trusted(
        manifest,
        anchor.resolved_root,
        expected_root_identity=anchor.identity,
    ):
        _log_agent_creator_rejection("manifest_trust_check_failed")
        return False

    try:
        manifest_identity = _path_identity(manifest)
        manifest_digest = anchor.file_digests.get(
            manifest.relative_to(anchor.resolved_root).as_posix()
        )
        if manifest_digest is None:
            raise OSError("manifest absent from startup trust snapshot")
        raw = _read_verified_agent_creator_file(
            manifest,
            expected_identity=manifest_identity,
            max_bytes=_AGENT_CREATOR_MAX_MANIFEST_BYTES,
            expected_digest=manifest_digest,
        )

        import yaml

        loaded = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("manifest root must be a mapping")
        capabilities = loaded.get("runtime_capabilities")
        if (
            not isinstance(capabilities, list)
            or len(capabilities) > _AGENT_CREATOR_MAX_RUNTIME_CAPABILITIES
            or not all(
                isinstance(capability, str)
                and 0 < len(capability) <= 128
                for capability in capabilities
            )
        ):
            raise ValueError("runtime_capabilities must be a bounded string list")
        if (
            _AGENT_CREATOR_ACTION_TOKEN_FD_CAPABILITY
            not in capabilities
        ):
            raise ValueError("action token FD capability missing")
        if (
            _path_identity(anchor.resolved_root) != anchor.identity
            or _path_identity(manifest) != manifest_identity
            or not _connector_runtime_path_is_trusted(
                manifest,
                anchor.resolved_root,
                expected_root_identity=anchor.identity,
            )
        ):
            raise OSError("manifest trust identity changed")
    except Exception:
        _log_agent_creator_rejection("manifest_capability_unavailable")
        return False
    return True


_VIDEO_EDIT_PLAN_PREPARATION_ACTIONS = frozenset({
    "resolve",
    "finalize",
    "freeze",
    "select-upload",
    "plan-migrate",
})


def _is_video_edit_plan_preparation_command(command: str) -> bool:
    """Allow only bounded preference preparation while App Plan mode is active."""
    parsed = _parse_video_edit_runtime_command(command)
    return bool(
        parsed is not None
        and Path(parsed.argv[1]).name == "preference_resolver.py"
        and len(parsed.argv) >= 3
        and parsed.argv[2] in _VIDEO_EDIT_PLAN_PREPARATION_ACTIONS
    )


def _video_edit_runtime_shell_guard_result(command: str) -> Optional[str]:
    if not any(name in command for name in _VIDEO_EDIT_RUNTIME_SCRIPTS):
        return None
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": (
            "Trusted video-edit scripts must run as one direct foreground "
            "python command without shell operators, wrappers, PTY, or background execution."
        ),
        "video_edit_runtime_direct": False,
        "video_edit_runtime_blocked": True,
    }, ensure_ascii=False)


def _run_agent_creator_command_if_allowed(
    command: str,
    *,
    cwd: str,
    timeout: int,
) -> Optional[str]:
    del cwd
    parsed = _parse_agent_creator_command(command)
    if parsed is None:
        return _agent_creator_shell_guard_result(command)

    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    if anchor is None:
        return _agent_creator_blocked_result(
            "agent_creator_identity_changed",
            "Agent Creator trust identity changed before execution.",
            direct=True,
        )
    script = Path(parsed.argv[1])
    expected_script_digest: Optional[str] = None
    try:
        expected_script_digest = anchor.file_digests.get(
            script.relative_to(anchor.resolved_root).as_posix()
        )
        identities_match = (
            expected_script_digest is not None
            and _path_identity(anchor.resolved_root) == parsed.root_identity
            and _path_identity(script) == parsed.script_identity
            and _connector_runtime_path_is_trusted(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
        )
    except OSError:
        identities_match = False
    if not identities_match:
        return _agent_creator_blocked_result(
            "agent_creator_identity_changed",
            "Agent Creator trust identity changed before execution.",
            direct=True,
        )

    approval_result = _request_agentcomputer_mutation_approval(parsed)
    if approval_result is not None:
        return approval_result

    try:
        script_bytes = _read_verified_agent_creator_script(
            script,
            expected_identity=parsed.script_identity,
            expected_digest=expected_script_digest,
        )
        if (
            _path_identity(anchor.resolved_root) != parsed.root_identity
            or not _connector_runtime_path_is_trusted(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
        ):
            raise OSError("agent creator trust identity changed")
    except OSError:
        return _agent_creator_blocked_result(
            "agent_creator_identity_changed",
            "Agent Creator trust identity changed before execution.",
            direct=True,
        )

    if not _agent_creator_manifest_supports_action_token_fd(anchor):
        return _agent_creator_blocked_result(
            "agent_creator_runtime_capability_unavailable",
            (
                "Agent Creator is unavailable because the installed preset "
                "does not support the scoped authorization channel."
            ),
            direct=True,
        )

    try:
        from tools.environments.local import build_agent_creator_runtime_env

        creator_env = build_agent_creator_runtime_env()
    except Exception:
        return _agent_creator_blocked_result(
            "agent_creator_scope_unavailable",
            "Agent Creator is unavailable because its scoped authorization is missing.",
            direct=True,
        )

    token = creator_env.pop("ZETTLAB_AGENT_ACTION_TOKEN", "")
    turn_id = creator_env.get("ZETTLAB_TURN_ID", "")
    try:
        from tools.environments.local import _sanitize_subprocess_env
        from tools.trusted_direct_runner import run_trusted_python_script

        run_env = _sanitize_subprocess_env(os.environ)
        run_env.pop("ZETTLAB_TURN_ID", None)
        completed = run_trusted_python_script(
            script=script,
            argv=parsed.argv[1:],
            cwd=anchor.resolved_root,
            base_env=run_env,
            injected_env=creator_env,
            injected_secrets={"ZETTLAB_AGENT_ACTION_TOKEN": token},
            timeout=timeout,
            stdin_text=parsed.stdin_text,
            secret_values=(token, turn_id),
            script_bytes=script_bytes,
            stdlib_only=True,
        )
    except Exception:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": "Agent Creator execution failed.",
            "agent_creator_direct": True,
        }, ensure_ascii=False)

    error = None
    if completed.timed_out:
        error = "Command timed out while running Agent Creator."
    elif completed.interrupted:
        error = "Agent Creator was interrupted."
    return json.dumps({
        "output": completed.output,
        "exit_code": completed.returncode,
        "error": error,
        "agent_creator_direct": True,
    }, ensure_ascii=False)


def _run_video_edit_runtime_command_if_allowed(
    command: str,
    *,
    cwd: str,
    timeout: int,
) -> Optional[str]:
    parsed = _parse_video_edit_runtime_command(command)
    if parsed is None:
        return _video_edit_runtime_shell_guard_result(command)

    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    script = Path(parsed.argv[1])
    try:
        identities_match = (
            anchor is not None
            and _path_identity(anchor.resolved_root) == parsed.root_identity
            and _path_identity(script) == parsed.script_identity
            and _connector_runtime_path_is_trusted(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
        )
    except OSError:
        identities_match = False
    if not identities_match:
        return _video_edit_runtime_shell_guard_result(command)

    secret_values: list[str] = []
    try:
        from tools.environments.local import build_video_edit_runtime_env

        trusted_env = build_video_edit_runtime_env()
        if not _video_edit_runtime_claims_match_receipt(parsed, trusted_env):
            return _video_edit_runtime_shell_guard_result(command)
        # Keep both identities inside the trusted worker: HERMES_SESSION_KEY is
        # the current lineage, while HERMES_GATEWAY_SESSION_KEY binds helper
        # authorization to the stable App/profile session across compaction.
        secret_values = [
            trusted_env.get("ZETTLAB_BUSINESS_EXECUTION_TOKEN", ""),
            trusted_env.get("ZETTLAB_AGENT_ACTION_TOKEN", ""),
        ]
        trusted_secrets = {
            key: trusted_env.pop(key)
            for key in (
                "ZETTLAB_BUSINESS_EXECUTION_TOKEN",
                "ZETTLAB_AGENT_ACTION_TOKEN",
            )
            if trusted_env.get(key)
        }
        trusted_context = {
            key: trusted_env[key]
            for key in (
                "HERMES_TURN_ID",
                "HERMES_SESSION_KEY",
                "HERMES_GATEWAY_SESSION_KEY",
                "ZETTLAB_EXECUTION_SCOPE_DIGEST",
                "ZETTLAB_EXECUTION_REQUEST_DIGEST",
            )
            if trusted_env.get(key)
        }
        run_cwd = cwd if cwd and os.path.isdir(cwd) else os.getcwd()
        payload = {
            "script": parsed.argv[1],
            "argv": parsed.argv[1:],
            "env": trusted_env,
            "context": trusted_context,
            "secrets": trusted_secrets,
            "cwd": run_cwd,
            "source_bundle": _trusted_video_edit_source_bundle(
                script=script,
                presets_root=anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            ),
        }
        # The upload helper permits one 1800s transfer plus one retry. Keep the
        # generic terminal cap unchanged while allowing this bounded operation
        # to finish both attempts.
        worker_timeout = _video_edit_runtime_timeout(parsed, timeout)
        completed = _run_video_edit_worker(
            payload,
            timeout=worker_timeout,
        )
        returncode = int(completed["returncode"])
        result = json.loads(_connector_runtime_result_json(
            command=command,
            output=(completed["stdout"] or "") + (completed["stderr"] or ""),
            returncode=returncode,
            secret_values=secret_values,
            timed_out=returncode == 124,
        ))
        result.pop("connector_runtime_direct", None)
        result["video_edit_runtime_direct"] = True
        return json.dumps(result, ensure_ascii=False)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        result = json.loads(_connector_runtime_result_json(
            command=command,
            output=stdout + stderr,
            returncode=124,
            secret_values=secret_values,
            timed_out=True,
        ))
        result.pop("connector_runtime_direct", None)
        result["video_edit_runtime_direct"] = True
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Trusted video-edit execution failed: {type(exc).__name__}: {exc}",
            "video_edit_runtime_direct": True,
        }, ensure_ascii=False)


def _run_camera_runtime_command_if_allowed(
    command: str,
    *,
    cwd: str,
    timeout: int,
) -> Optional[str]:
    parsed = _parse_camera_runtime_command(command)
    if parsed is None:
        return _camera_runtime_shell_guard_result(command)
    if not _ensure_sensitive_runtime_boundary():
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": "Camera runtime process memory boundary is unavailable",
            "camera_runtime_direct": True,
        }, ensure_ascii=False)

    anchor = _CONNECTOR_RUNTIME_ROOT_ANCHOR
    script = Path(parsed.argv[1])
    expected_digest: Optional[str] = None
    try:
        expected_digest = anchor.file_digests.get(
            script.relative_to(anchor.resolved_root).as_posix()
        )
        identities_match = (
            anchor is not None
            and expected_digest is not None
            and _path_identity(anchor.resolved_root) == parsed.root_identity
            and _path_identity(script) == parsed.script_identity
            and _connector_runtime_path_is_trusted(
                script,
                anchor.resolved_root,
                expected_root_identity=parsed.root_identity,
            )
            and _camera_runtime_manifest_allows(anchor)
        )
    except (OSError, AttributeError):
        identities_match = False
    if not identities_match:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": "Camera runtime package identity or capability is unavailable",
            "camera_runtime_direct": True,
        }, ensure_ascii=False)

    try:
        script_bytes = _read_connector_runtime_script_bytes(
            script,
            expected_identity=parsed.script_identity,
            expected_digest=expected_digest,
        )
        from tools.environments.local import build_camera_runtime_env
        from tools.trusted_direct_runner import run_trusted_python_script

        trusted_env = build_camera_runtime_env()
        trusted_secrets = {
            key: trusted_env.pop(key)
            for key in (
                "ZETTLAB_AGENT_ACTION_TOKEN",
                "ZETTLAB_BUSINESS_EXECUTION_TOKEN",
            )
        }
        secret_values = list(trusted_secrets.values())
        run_cwd = cwd if cwd and os.path.isdir(cwd) else os.getcwd()
        completed = run_trusted_python_script(
            script=script,
            argv=parsed.argv[1:],
            cwd=Path(run_cwd),
            base_env={},
            injected_env=trusted_env,
            injected_secrets=trusted_secrets,
            timeout=max(1, min(timeout, _CAMERA_RUNTIME_MAX_TIMEOUT_SECONDS)),
            secret_values=secret_values,
            script_bytes=script_bytes,
            stdlib_only=True,
        )
        payload = json.loads(_connector_runtime_result_json(
            command=command,
            output=completed.output,
            returncode=completed.returncode,
            secret_values=secret_values,
            timed_out=completed.timed_out,
        ))
        payload.pop("connector_runtime_direct", None)
        payload["camera_runtime_direct"] = True
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Camera runtime execution failed: {type(exc).__name__}",
            "camera_runtime_direct": True,
        }, ensure_ascii=False)


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id). Pair with notify_on_complete=true for bounded tasks; leave silent only for servers/daemons that never exit. Never use nohup/setsid/trailing '&' — use background=true so Hermes tracks the process. After starting a server, verify readiness with a health check, then act in a separate call; no blind sleep loops. Manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd. On the local backend, managed platform runtimes may expose the semantic 'agent_output' workdir for the current agent's output directory. When a command changes the session cwd (cd, pushd), the result includes a "cwd" field — trust it instead of prefixing every command with 'cd'.
PTY: set pty=true for interactive CLIs (they hang without it). Pipe git output to cat if it might page.
"""

# Global state for environment lifecycle management
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_environment_profile_owners: Dict[str, str] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper (issue #20561).
# Set when _maybe_reap_docker_orphans first runs; concurrent _create_environment
# calls for parallel subagents won't re-trigger the sweep.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps long-Exited containers labeled ``hermes-agent=1`` for the current
    profile that match the issue #20561 leak class — containers left behind
    by Hermes processes that exited without firing ``atexit`` (SIGKILL,
    OOM, terminal-window-close). The reaper is conservative by default:
    only Exited containers older than ``2 × lifetime_seconds`` and scoped to
    the current profile.

    Gates:

    * ``terminal.docker_orphan_reaper: false`` disables it entirely (the
      operator opted out — usually because they're running multiple
      Hermes processes in the same profile and don't trust the
      conservative defaults).
    * ``_docker_orphan_reaper_ran`` flag — sweep runs once per Python
      interpreter, not on every subagent / RL-rollout / parallel
      ``terminal()`` call.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    # Cheap double-checked-locking: read without the lock, take the lock
    # only on first run, recheck inside.
    if _docker_orphan_reaper_ran:
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × lifetime_seconds gives sibling Hermes processes a generous grace
    # window. Floor at 60s so an operator with TERMINAL_LIFETIME_SECONDS=0
    # doesn't get an instant-reap that races their own setup.
    # ``container_config`` only carries container_* keys, so read
    # lifetime_seconds from the env var the rest of the module uses.
    try:
        lifetime = int(os.getenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
    lifetime = max(60, lifetime)
    max_age = lifetime * 2

    try:
        from tools.environments.docker import (
            reap_orphan_containers, _get_active_profile_name,
        )
    except ImportError:
        return
    try:
        profile = _get_active_profile_name()
        removed = reap_orphan_containers(
            max_age_seconds=max_age, profile_filter=profile,
        )
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )
    except Exception as e:
        # Never fail the env-creation path because of a janitor problem.
        logger.debug("Docker orphan reaper raised: %s", e)


# Per-task environment overrides registry.
# Allows environments (e.g., TerminalBench2Env) to specify a custom Docker/Modal
# image for a specific task_id BEFORE the agent loop starts. When the terminal or
# file tools create a new sandbox for that task_id, they check this registry first
# and fall back to the TERMINAL_MODAL_IMAGE (etc.) env var if no override is set.
#
# This is never exposed to the model -- only infrastructure code calls it.
# Thread-safe because each task_id is unique per rollout.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}

_MANAGED_PROFILE_REGISTRY_PREFIX = "managed-profile:"


def _canonical_managed_profile_home(profile_home: object | None = None) -> str | None:
    """Resolve the canonical owner used to isolate multiplex terminal state."""
    if profile_home is None:
        if os.environ.get("HERMES_MANAGED_GATEWAY") != "1":
            return None
        try:
            from hermes_constants import get_hermes_home

            profile_home = get_hermes_home()
        except Exception:
            return None
    raw = str(profile_home or "").strip()
    if not raw or "\x00" in raw:
        return None
    try:
        return os.path.normcase(
            os.path.realpath(os.path.abspath(os.path.expanduser(raw)))
        )
    except (OSError, ValueError):
        return None


def _managed_profile_registry_prefix(profile_home: object | None = None) -> str:
    """Return an opaque, stable registry prefix for one profile owner."""
    canonical = _canonical_managed_profile_home(profile_home)
    if canonical is None:
        return ""
    import hashlib

    digest = hashlib.sha256(
        canonical.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return f"{_MANAGED_PROFILE_REGISTRY_PREFIX}{digest}:"


def _managed_profile_registry_key(
    task_id: Optional[str],
    profile_home: object | None = None,
) -> str:
    """Bind a task/session key to its canonical multiplex profile owner."""
    raw = str(task_id or "default")
    prefix = _managed_profile_registry_prefix(profile_home)
    return f"{prefix}{raw}" if prefix else raw

# ── Per-session cwd records (cwd rearchitecture, step 1) ────────────────────
#
# The durable source of truth for "which directory is THIS session working
# in". Keyed by the raw session/task key (NOT the collapsed container id):
# the terminal env is shared across sessions, so any cwd state stored on the
# env is a global mutable timeshared between sessions — the root cause of the
# wrong-worktree bug class (env.cwd_owner stamping, _last_known_cwd, and the
# ownership ladder in file_tools are all patches over that misplacement).
#
# Step 1 (this change): dual-write only. Every site that learns a session's
# live cwd (post-command tracking, cwd-override registration) also records it
# here. Readers still use the legacy env.cwd ladder. Later steps flip
# file_tools and _resolve_command_cwd to read this store, then delete the
# env-side tracking + ownership guards.
_session_cwd: Dict[str, str] = {}
_session_cwd_lock = threading.Lock()


def record_session_cwd(session_key: Optional[str], cwd: Optional[str]) -> None:
    """Record *cwd* as the working directory of *session_key*.

    Called wherever a session's live cwd becomes known: after a terminal
    command completes (the env's post-command tracking has just parsed the
    resulting cwd) and when a surface registers a workspace cwd override.
    Empty/None session keys collapse to ``"default"`` (single-session CLI).
    Non-string / empty cwds are ignored.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = _managed_profile_registry_key(session_key)
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd


def get_session_cwd(session_key: Optional[str]) -> Optional[str]:
    """Return the recorded working directory for *session_key*, if any.

    No fallback chain here on purpose: callers decide what an absent record
    means (config default, TERMINAL_CWD seed, process cwd). ``None``/empty
    keys read the ``"default"`` record.
    """
    key = _managed_profile_registry_key(session_key)
    with _session_cwd_lock:
        return _session_cwd.get(key)


def clear_session_cwd(session_key: str) -> None:
    """Drop a session's cwd record (session teardown)."""
    key = _managed_profile_registry_key(session_key)
    with _session_cwd_lock:
        _session_cwd.pop(key, None)


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """
    Register environment overrides for a specific task/rollout.

    Called by Atropos environments before the agent loop to configure
    per-task sandbox settings (e.g., a custom Dockerfile for the Modal image).

    Supported override keys:
        - modal_image: str -- Path to Dockerfile or Docker Hub image name
        - docker_image: str -- Docker image name
        - cwd: str -- Working directory inside the sandbox

    Args:
        task_id: The rollout's unique task identifier
        overrides: Dict of config keys to override
    """
    raw_task_key = _managed_profile_registry_key(task_id)
    _task_env_overrides[raw_task_key] = overrides

    # If a live environment already exists for this task, a freshly registered
    # ``cwd`` override (e.g. the ACP client switching the editor's project root
    # mid-session via ``session/load`` / ``session/resume``) must take effect
    # immediately. The session record is what commands resolve against;
    # the live env's cwd is also updated so env-side seeding stays consistent.
    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        # A registered workspace cwd IS the session's working directory until
        # a `cd` changes it.
        record_session_cwd(task_id, new_cwd)
        # The live env is cached under the raw task_id for per-session surfaces
        # (ACP/gateway/dashboard) and under the collapsed container id for
        # isolation-keyed rollouts. Try the raw id first, then the container id,
        # so a CWD-only override (which collapses to "default") still finds and
        # updates the originating session's env.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(raw_task_key) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """
    Clear environment overrides for a task after rollout completes.

    Called during cleanup to avoid stale entries accumulating.
    """
    _task_env_overrides.pop(_managed_profile_registry_key(task_id), None)
    clear_session_cwd(task_id)


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """
    Map a tool-call ``task_id`` to the container/sandbox key used by
    ``_active_environments``.

    The top-level agent passes ``task_id=None`` and lands on ``"default"``.
    ``delegate_task`` children pass their own subagent ID so that
    file-state tracking, the active-subagents registry, and TUI events stay
    distinct per child -- but we deliberately collapse that ID back to
    ``"default"`` here so subagents share the parent's long-lived container
    (one bash, one /workspace, one set of installed packages).

    Exception: RL / benchmark environments (TerminalBench2, HermesSweEnv, ...)
    call ``register_task_env_overrides(task_id, {...})`` to request a
    per-task Docker/Modal image. When an override is registered for a
    task_id, we honour it by returning the task_id unchanged -- those
    rollouts need their own isolated sandbox, which is the whole point of
    the override.

    CWD-only overrides (registered by the ACP adapter for workspace
    tracking) are *not* isolation signals — they should not cause each
    session to spin up its own container.  Only overrides containing
    backend-specific image keys or ``env_type`` trigger isolation.
    """
    _ISOLATION_KEYS = frozenset({
        "docker_image", "modal_image", "singularity_image",
        "daytona_image", "env_type",
    })
    raw_task_key = _managed_profile_registry_key(task_id)
    if task_id and raw_task_key in _task_env_overrides:
        overrides = _task_env_overrides[raw_task_key]
        if set(overrides.keys()) & _ISOLATION_KEYS:
            return raw_task_key
    return _managed_profile_registry_key("default")


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container so per-session surfaces (ACP/gateway/
    dashboard) don't each spin up their own sandbox. Callers that need the
    override (terminal command setup, file-tool cwd resolution) must therefore
    read the raw id FIRST and only fall back to the collapsed container id, or
    the originating session's override is silently dropped. This is the single
    source of that lookup so the terminal and file layers can't drift apart.
    """
    raw = _managed_profile_registry_key(task_id)
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(task_id))
        or {}
    )


# Configuration from environment variables

def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an environment variable with *converter*, raising a clear error on bad values.

    Without this wrapper, a single malformed env var (e.g. TERMINAL_TIMEOUT=5m)
    causes an unhandled ValueError that kills every terminal command.
    """
    raw = os.getenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """Return the current working directory, tolerating a deleted CWD.

    ``os.getcwd()`` raises FileNotFoundError when the process's working
    directory has been removed out from under it (e.g. a scratch workspace
    that was cleaned up mid-session). Fall back to TERMINAL_CWD, then the
    user's home directory, so terminal setup never crashes on a stale CWD.
    """
    try:
        return os.getcwd()
    except FileNotFoundError:
        return os.getenv("TERMINAL_CWD") or os.path.expanduser("~")


# Path prefixes that identify a *host* working directory which cannot exist
# inside a container sandbox. Covers POSIX user dirs and Windows drive paths
# (``C:\Users\...`` / ``C:/Users/...``) — the latter is how a Windows host's
# cwd looks when it leaks toward a Linux container's ``-w`` flag.
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _is_ssh_remote_tilde_cwd(backend: str, cwd: str) -> bool:
    """Return True when *cwd* is a tilde path that the remote SSH shell must
    expand itself, so the Hermes host/container must NOT ``expanduser`` it.

    SSH ``cwd`` is interpreted by the *remote* shell (``cd ~`` / ``cd ~/x``
    over ``ssh ... bash -c``). Expanding ``~`` locally would rewrite it to the
    Hermes host HOME (often ``/opt/data`` under Docker) and inject a
    nonexistent path into the remote session. Only ``~`` / ``~/...`` on the
    ``ssh`` backend qualify; absolute remote paths still pass through
    unchanged, and every other backend keeps expanding locally.
    """
    if (backend or "").strip().lower() != "ssh":
        return False
    return cwd == "~" or cwd.startswith("~/")


def _is_unusable_container_cwd(cwd: str) -> bool:
    """Return True if *cwd* is a host/relative path that won't work as the
    working directory inside a container sandbox.

    A container's cwd must be an absolute path that exists *inside* the
    sandbox (e.g. ``/workspace`` or ``/root``). A host path (``/home/user``,
    ``C:\\Users\\me``) or a relative path (``.``, ``src/``) is meaningless to
    ``docker run -w`` and makes the container fail to start (exit 125).
    """
    if not cwd:
        return False
    if any(cwd.startswith(p) for p in _HOST_CWD_PREFIXES):
        return True
    # Relative paths (".", "src/") can't be a container workdir either. Windows
    # drive paths are absolute on Windows but os.path.isabs() is False on a
    # POSIX host, so they're already caught by the prefix check above.
    if not os.path.isabs(cwd):
        return True
    return False


# One-shot guard for the config-fallback bridge below.  Purely an
# optimization: after the first attempt either TERMINAL_ENV is set (bridge
# succeeded — merged config always carries terminal.backend) or the import
# failed and retrying every call would be wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    terminal_tool reads ALL terminal settings from os.environ (TERMINAL_*).
    The CLI (cli.py ``env_mappings``), the gateway (gateway/run.py
    ``_terminal_env_map``), and TUI/dashboard PTY launches
    (``apply_terminal_config_to_env``) bridge ``terminal.*`` config into env
    vars at startup — but processes that skip all of those paths (``hermes
    serve`` / the Desktop app backend's in-process agents, the desktop cron
    ticker, ACP) used to silently fall back to the local backend even when
    config.yaml selects ``terminal.backend: docker``, running commands on the
    host the user intended to sandbox (#63141, #54449, #61115, #65696).

    Explicit terminal config keys win: when config.yaml has a ``terminal``
    section, each key present there overrides its matching env value (which may
    be stale from ``hermes setup``). Environment values for omitted terminal
    keys are preserved. When no terminal section exists, exported/.env values
    keep working unchanged.
    """
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    try:
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        # If config.yaml has an explicit terminal section, bridge with
        # override enabled. The helper only overrides env vars for keys present
        # in that raw section; merged defaults remain backfill-only. Without a
        # terminal section, preserve an existing TERMINAL_ENV selection or
        # backfill defaults when no selection exists.
        raw_config = read_raw_config()
        has_terminal_section = isinstance(raw_config.get("terminal"), dict)

        if has_terminal_section:
            # Explicit terminal keys in config.yaml win over matching env values.
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            # No terminal section in config.yaml, TERMINAL_ENV not set —
            # backfill from config defaults
            apply_terminal_config_to_env(env=None, override=False)
    except Exception:
        # Never let a config problem take the terminal tool down — the
        # historical local default still applies.
        logger.debug("terminal config → env fallback bridge failed", exc_info=True)


def _get_env_config() -> Dict[str, Any]:
    """Get terminal environment configuration from environment variables."""
    # Default image with Python and Node.js for maximum compatibility
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    _ensure_terminal_env_bridged()
    env_type = os.getenv("TERMINAL_ENV", "local")
    
    mount_docker_cwd = os.getenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false").lower() in {"true", "1", "yes"}
    container_backend = env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}
    docker_backend = env_type == "docker"

    # Docker/container-only env vars may be bridged from config.yaml even when
    # the active backend is local/ssh.  Do not parse their JSON/numeric payloads
    # until a backend that can consume them is selected; a stale or invalid
    # Docker value should not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number")
        container_memory = _parse_env_var("TERMINAL_CONTAINER_MEMORY", "5120")
        container_disk = _parse_env_var("TERMINAL_CONTAINER_DISK", "51200")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
        docker_shm_size = os.getenv("TERMINAL_DOCKER_SHM_SIZE", "1g")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []
        docker_shm_size = "1g"

    # Default cwd: local uses the host's current directory, ssh uses the
    # remote home, Vercel uses its documented workspace root, and everything
    # else starts in the backend's default root-like cwd.
    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    elif env_type == "vercel_sandbox":
        default_cwd = _VERCEL_SANDBOX_DEFAULT_CWD
    else:
        default_cwd = "/root"

    # Read TERMINAL_CWD but sanity-check it for container backends.
    # If Docker cwd passthrough is explicitly enabled, remap the host path to
    # /workspace and track the original host path separately. Otherwise keep the
    # normal sandbox behavior and discard host paths.
    cwd = os.getenv("TERMINAL_CWD", default_cwd)
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = os.getenv("TERMINAL_CWD") or _safe_getcwd()
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            any(candidate.startswith(p) for p in _HOST_CWD_PREFIXES)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif env_type in _CONTAINER_BACKENDS and cwd:
        # Host paths and relative paths that won't work inside containers
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(os.getenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": os.getenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": os.getenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": os.getenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": os.getenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "vercel_runtime": os.getenv("TERMINAL_VERCEL_RUNTIME", "").strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": os.getenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": os.getenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": os.getenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting (true by default for non-local backends); local is always opt-in.
        # Per-backend env vars override if explicitly set.
        "ssh_persistent": os.getenv(
            "TERMINAL_SSH_PERSISTENT",
            os.getenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ).lower() in {"true", "1", "yes"},
        "local_persistent": os.getenv("TERMINAL_LOCAL_PERSISTENT", "false").lower() in {"true", "1", "yes"},
        # Container resource config (applies to docker, singularity, modal,
        # daytona, and vercel_sandbox -- ignored for local/ssh)
        "container_cpu": container_cpu,
        "container_memory": container_memory,     # MB (default 5GB)
        "container_disk": container_disk,        # MB (default 50GB)
        "container_persistent": os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"true", "1", "yes"},
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": os.getenv("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false").lower() in {"true", "1", "yes"},
        "docker_network": os.getenv("TERMINAL_DOCKER_NETWORK", "true").lower() in {"true", "1", "yes"},
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process container reuse (issue #20561).  The docs claim
        # "ONE long-lived container shared across sessions" — this toggle
        # makes that real by probing for a labeled container at startup and
        # attaching to it instead of always starting a fresh one.  Set to
        # ``false`` for hard per-process isolation (no reuse, container is
        # removed on exit).
        "docker_persist_across_processes": os.getenv(
            "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"
        ).lower() in {"true", "1", "yes"},
        # Startup orphan reaper for hermes-tagged containers left behind by
        # crashed / SIGKILL'd previous processes that bypassed atexit.
        # Conservative: only sweeps Exited containers older than 2× the
        # idle-reap window AND scoped to the current profile. Issue #20561.
        "docker_orphan_reaper": os.getenv(
            "TERMINAL_DOCKER_ORPHAN_REAPER", "true"
        ).lower() in {"true", "1", "yes"},
    }


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: str = None):
    """
    Create an execution environment for sandboxed command execution.
    
    Args:
        env_type: One of "local", "docker", "singularity", "modal",
            "daytona", "vercel_sandbox", "ssh"
        image: Docker/Singularity/Modal image name (ignored for local/ssh/vercel)
        cwd: Working directory
        timeout: Default command timeout
        ssh_config: SSH connection config (for env_type="ssh")
        container_config: Resource config for container backends (cpu, memory, disk, persistent)
        task_id: Task identifier for environment reuse and snapshot keying
        host_cwd: Optional host working directory to bind into Docker when explicitly enabled
        
    Returns:
        Environment instance with execute() method
    """
    cc = container_config or {}
    cpu = cc.get("container_cpu", 1)
    memory = cc.get("container_memory", 5120)
    disk = cc.get("container_disk", 51200)
    persistent = cc.get("container_persistent", True)
    volumes = cc.get("docker_volumes", [])
    docker_forward_env = cc.get("docker_forward_env", [])
    docker_env = cc.get("docker_env", {})
    docker_extra_args = cc.get("docker_extra_args", [])
    docker_network = cc.get("docker_network", True)

    if env_type == "local":
        return _LocalEnvironment(cwd=cwd, timeout=timeout)
    
    elif env_type == "docker":
        # One-shot orphan reaper: clean up labeled containers left behind by
        # prior Hermes processes that hit SIGKILL / OOM / a closed terminal
        # before the atexit cleanup hook could run.  Gated to once per
        # process so concurrent _create_environment calls (parallel
        # subagents, RL benchmarks) don't run the reaper N times.
        # Disable via ``terminal.docker_orphan_reaper: false`` (issue #20561).
        _maybe_reap_docker_orphans(cc)
        return _DockerEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
            volumes=volumes,
            host_cwd=host_cwd,
            auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
            forward_env=docker_forward_env,
            env=docker_env,
            run_as_host_user=cc.get("docker_run_as_host_user", False),
            network=docker_network,
            extra_args=docker_extra_args,
            persist_across_processes=cc.get("docker_persist_across_processes", True),
            shm_size=cc.get("docker_shm_size", "1g"),
        )
    
    elif env_type == "singularity":
        return _SingularityEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=cpu, memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "modal":
        sandbox_kwargs = {}
        if cpu > 0:
            sandbox_kwargs["cpu"] = cpu
        if memory > 0:
            sandbox_kwargs["memory"] = memory
        if disk > 0:
            try:
                import inspect, modal
                if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                    sandbox_kwargs["ephemeral_disk"] = disk
            except Exception:
                pass

        modal_state = _get_modal_backend_state(cc.get("modal_mode"))

        if modal_state["selected_backend"] == "managed":
            return _ManagedModalEnvironment(
                image=image, cwd=cwd, timeout=timeout,
                modal_sandbox_kwargs=sandbox_kwargs,
                persistent_filesystem=persistent, task_id=task_id,
            )

        if modal_state["selected_backend"] != "direct":
            if modal_state["managed_mode_blocked"]:
                raise ValueError(
                    "Modal backend is configured for managed mode, but "
                    "Nous Tool Gateway access is not currently available and no direct "
                    "Modal credentials/config were found. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                    + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
                )
            if modal_state["mode"] == "managed":
                raise ValueError(
                    "Modal backend is configured for managed mode, but the managed tool gateway is unavailable. "
                    + nous_tool_gateway_unavailable_message(
                        "managed Modal execution",
                    )
                )
            if modal_state["mode"] == "direct":
                raise ValueError(
                    "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."
                )
            message = "Modal backend selected but no direct Modal credentials/config was found."
            if managed_nous_tools_enabled():
                message = (
                    "Modal backend selected but no direct Modal credentials/config or managed tool gateway was found."
                )
            raise ValueError(message)

        return _ModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )
    
    elif env_type == "daytona":
        # Lazy import so daytona SDK is only required when backend is selected.
        from tools.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment
        return _DaytonaEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            cpu=int(cpu), memory=memory, disk=disk,
            persistent_filesystem=persistent, task_id=task_id,
        )

    elif env_type == "vercel_sandbox":
        from tools.environments.vercel_sandbox import (
            VercelSandboxEnvironment as _VercelSandboxEnvironment,
        )
        return _VercelSandboxEnvironment(
            runtime=cc.get("vercel_runtime") or None,
            cwd=cwd,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            disk=disk,
            persistent_filesystem=persistent,
            task_id=task_id,
        )

    elif env_type == "ssh":
        if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
            raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
        return _SSHEnvironment(
            host=ssh_config["host"],
            user=ssh_config["user"],
            port=ssh_config.get("port", 22),
            key_path=ssh_config.get("key", ""),
            cwd=cwd,
            timeout=timeout,
        )

    else:
        raise ValueError(
            f"Unknown environment type: {env_type}. Use 'local', 'docker', "
            f"'singularity', 'modal', 'daytona', 'vercel_sandbox', or 'ssh'"
        )


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    current_time = time.time()

    # Check the process registry -- skip cleanup for sandboxes with active
    # background processes (their _last_activity gets refreshed to keep them alive).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            profile_owner = _environment_profile_owners.get(task_id)
            if profile_owner:
                has_active = process_registry.has_active_processes_for_profile(
                    task_id, profile_owner
                )
            else:
                has_active = process_registry.has_active_processes(task_id)
            if has_active:
                _last_activity[task_id] = current_time  # Keep sandbox alive
    except ImportError:
        pass

    # Phase 1: collect stale entries and remove them from tracking dicts while
    # holding the lock.  Do NOT call env.cleanup() inside the lock -- Modal and
    # Docker teardown can block for 10-15s, which would stall every concurrent
    # terminal/file tool call waiting on _env_lock.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                _environment_profile_owners.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        # Also purge per-task creation locks for cleaned-up tasks
        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the actual sandboxes OUTSIDE the lock so other tool calls
    # are not blocked while Modal/Docker sandboxes shut down.
    for task_id, env in envs_to_stop:
        # Invalidate stale file_ops cache entry (Bug fix: prevents
        # ShellFileOperations from referencing a dead sandbox)
        try:
            from tools.file_tools import clear_file_ops_cache
            clear_file_ops_cache(task_id)
        except ImportError:
            pass

        try:
            if hasattr(env, 'cleanup'):
                env.cleanup()
            elif hasattr(env, 'stop'):
                env.stop()
            elif hasattr(env, 'terminate'):
                env.terminate()

            logger.info("Cleaned up inactive environment for task: %s", task_id)

        except Exception as e:
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower():
                logger.info("Environment for task %s already cleaned up", task_id)
            else:
                logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e, exc_info=True)

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def get_active_env(task_id: str):
    """Return the active BaseEnvironment for *task_id*, or None."""
    lookup = _resolve_container_task_id(task_id)
    raw_task_key = _managed_profile_registry_key(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(raw_task_key)


def is_persistent_env(task_id: str) -> bool:
    """Return True if the active environment for task_id is configured for
    cross-turn persistence (``persistent_filesystem=True``).

    Used by the agent loop to skip per-turn teardown for backends whose whole
    point is to survive between turns (docker with ``container_persistent``,
    daytona, modal, etc.). Non-persistent backends (e.g. Morph) still get torn
    down at end-of-turn to prevent leakage. The idle reaper
    (``_cleanup_inactive_envs``) handles persistent envs once they exceed
    ``terminal.lifetime_seconds``.
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    return bool(getattr(env, "_persistent", False))




def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id, _already_scoped=True)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(
    task_id: str,
    *,
    force_remove: bool = False,
    _already_scoped: bool = False,
):
    """Manually clean up a specific environment by task_id.

    *force_remove* (default False) is forwarded to backends that accept it
    — currently only ``DockerEnvironment``. The default of False matches
    session-lifecycle semantics: this function is called from
    ``AIAgent.close()`` (TUI session close, gateway session teardown) and the
    per-turn cleanup branch for non-persistent envs, both of which should
    honor the user's persist-mode preference. Stopping the container here
    would defeat the "ONE long-lived container shared across sessions"
    contract — exactly the bug Ben reported when the container was killed
    on every TUI session close.

    Pass ``force_remove=True`` for actual user-initiated teardown
    (e.g. ``/reset``-style flows that haven't been wired yet, or future
    "destroy my sandbox" commands).

    The idle reaper passes the env through ``env.cleanup()`` directly (not
    via this function), so persist-mode idle envs are similarly no-op'd —
    only the orphan reaper at next startup reclaims them.
    """
    # Remove from tracking dicts while holding the lock, but defer the
    # actual (potentially slow) env.cleanup() call to outside the lock
    # so other tool calls aren't blocked.
    registry_key = task_id if _already_scoped else _resolve_container_task_id(task_id)
    env = None
    with _env_lock:
        env = _active_environments.pop(registry_key, None)
        _last_activity.pop(registry_key, None)
        _environment_profile_owners.pop(registry_key, None)

    # Clean up per-task creation lock
    with _creation_locks_lock:
        _creation_locks.pop(registry_key, None)

    # Invalidate stale file_ops cache entry
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(registry_key)
    except ImportError:
        pass

    if env is None:
        return

    try:
        if hasattr(env, 'cleanup'):
            # Pass force_remove only if the env's cleanup() accepts it
            # (DockerEnvironment after issue #20561; other backends don't).
            import inspect
            sig = inspect.signature(env.cleanup)
            if "force_remove" in sig.parameters:
                env.cleanup(force_remove=force_remove)
            else:
                env.cleanup()
        elif hasattr(env, 'stop'):
            env.stop()
        elif hasattr(env, 'terminate'):
            env.terminate()

        logger.info("Manually cleaned up environment for task: %s", registry_key)

    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", registry_key)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", registry_key, e)


def cleanup_managed_profile_environments(profile_home: object) -> int:
    """Destroy terminal state owned by one unloaded multiplex profile."""
    prefix = _managed_profile_registry_prefix(profile_home)
    if not prefix:
        return 0

    with _env_lock:
        active_keys = [
            key for key in _active_environments if key.startswith(prefix)
        ]
    for key in active_keys:
        cleanup_vm(key, force_remove=True, _already_scoped=True)

    with _session_cwd_lock:
        for key in list(_session_cwd):
            if key.startswith(prefix):
                _session_cwd.pop(key, None)
    for key in list(_task_env_overrides):
        if key.startswith(prefix):
            _task_env_overrides.pop(key, None)
    with _creation_locks_lock:
        for key in list(_creation_locks):
            if key.startswith(prefix):
                _creation_locks.pop(key, None)
    return len(active_keys)


def _atexit_cleanup():
    """Stop cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        # Snapshot the env objects BEFORE cleanup_all_environments empties
        # the dict; we need them to wait on docker cleanup threads after the
        # registry has been cleared.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        # Block briefly so docker stop/rm actually completes before the
        # interpreter exits. Issue #20561 — without this join, the daemon
        # cleanup threads were getting torn down mid-`docker stop`, leaving
        # Exited containers piled up on the host.
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is None:
                continue
            try:
                wait_fn(timeout=15.0)
            except Exception as e:  # never block shutdown on a bad backend
                logger.debug("wait_for_cleanup raised on exit: %s", e)

atexit.register(_atexit_cleanup)


# =============================================================================
# Exit Code Context for Common CLI Tools
# =============================================================================
# Many Unix commands use non-zero exit codes for informational purposes, not
# to indicate failure.  The model sees a raw exit_code=1 from `grep` and
# wastes a turn investigating something that just means "no matches".
# This lookup adds a human-readable note so the agent can move on.

def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Return a human-readable note when a non-zero exit code is non-erroneous.

    Returns None when the exit code is 0 or genuinely signals an error.
    The note is appended to the tool result so the model doesn't waste
    turns investigating expected exit codes.
    """
    if exit_code == 0:
        return None

    # Extract the last command in a pipeline/chain — that determines the
    # exit code.  Handles  `cmd1 && cmd2`, `cmd1 | cmd2`, `cmd1; cmd2`.
    # Deliberately simple: split on shell operators and take the last piece.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Get base command name (first word), stripping env var assignments
    # like  VAR=val cmd ...
    words = last_segment.split()
    base_cmd = ""
    for w in words:
        if "=" in w and not w.startswith("-"):
            continue  # skip VAR=val
        base_cmd = w.split("/")[-1]  # handle /usr/bin/grep -> grep
        break

    if not base_cmd:
        return None

    # Command-specific semantics
    semantics: dict[str, dict[int, str]] = {
        # grep/rg/ag/ack: 1=no matches found (normal), 2+=real error
        "grep":  {1: "No matches found (not an error)"},
        "egrep": {1: "No matches found (not an error)"},
        "fgrep": {1: "No matches found (not an error)"},
        "rg":    {1: "No matches found (not an error)"},
        "ag":    {1: "No matches found (not an error)"},
        "ack":   {1: "No matches found (not an error)"},
        # diff: 1=files differ (expected), 2+=real error
        "diff":  {1: "Files differ (expected, not an error)"},
        "colordiff": {1: "Files differ (expected, not an error)"},
        # find: 1=some dirs inaccessible but results may still be valid
        "find":  {1: "Some directories were inaccessible (partial results may still be valid)"},
        # test/[: 1=condition is false (expected)
        "test":  {1: "Condition evaluated to false (expected, not an error)"},
        "[":     {1: "Condition evaluated to false (expected, not an error)"},
        # curl: common non-error codes
        "curl":  {
            6: "Could not resolve host",
            7: "Failed to connect to host",
            22: "HTTP response code indicated error (e.g. 404, 500)",
            28: "Operation timed out",
        },
        # git: 1 is context-dependent but often normal (e.g. git diff with changes)
        "git":   {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
    }

    cmd_semantics = semantics.get(base_cmd)
    if cmd_semantics and exit_code in cmd_semantics:
        return cmd_semantics[exit_code]

    return None


def _command_requires_pipe_stdin(command: str) -> bool:
    """Return True when PTY mode would break stdin-driven commands.

    Some CLIs change behavior when stdin is a TTY. In particular,
    `gh auth login --with-token` expects the token to arrive via piped stdin and
    waits for EOF; when we launch it under a PTY, `process.submit()` only sends a
    newline, so the command appears to hang forever with no visible progress.
    """
    normalized = " ".join(command.lower().split())
    return (
        normalized.startswith("gh auth login")
        and "--with-token" in normalized
    )


_SHELL_LEVEL_BACKGROUND_RE = re.compile(
    r"(?:^|[;&|]\s*|&&\s*|\|\|\s*|\$\(\s*)(?:nohup|disown|setsid)\b", re.IGNORECASE | re.MULTILINE
)
_INLINE_BACKGROUND_AMP_RE = re.compile(r"\s&\s")
_TRAILING_BACKGROUND_AMP_RE = re.compile(r"\s&\s*(?:#.*)?$")


def _strip_quotes(command: str) -> str:
    """Remove single- and double-quoted content so regex checks don't match inside strings.

    This prevents false positives when keywords like 'nohup' or 'setsid' appear
    in commit messages, Python -c code, echo arguments, or PR body text.
    Also strips backtick-quoted content and heredoc-style inline text.
    """
    # Remove single-quoted strings (no escaping inside single quotes in shell)
    result = re.sub(r"'[^']*'", "''", command)
    # Remove double-quoted strings (handle escaped quotes)
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)
    # Remove backtick-quoted strings
    result = re.sub(r"`[^`]*`", "``", result)
    return result


_LONG_LIVED_FOREGROUND_PATTERNS = (
    re.compile(r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+compose\s+up\b", re.IGNORECASE),
    re.compile(r"\bnext\s+dev\b", re.IGNORECASE),
    re.compile(r"\bvite(?:\s|$)", re.IGNORECASE),
    re.compile(r"\bnodemon\b", re.IGNORECASE),
    re.compile(r"\buvicorn\b", re.IGNORECASE),
    re.compile(r"\bgunicorn\b", re.IGNORECASE),
    re.compile(r"\bpython(?:3)?\s+-m\s+http\.server\b", re.IGNORECASE),
)


def _looks_like_help_or_version_command(command: str) -> bool:
    """Return True for informational invocations that should never be blocked."""
    normalized = " ".join(command.lower().split())
    return (
        " --help" in normalized
        or normalized.endswith(" -h")
        or " --version" in normalized
        or normalized.endswith(" -v")
    )


def _foreground_background_guidance(command: str) -> str | None:
    """Suggest background mode when a foreground command looks long-lived.

    Prevents workflows that start a server/watch process and then stall before
    follow-up checks or test commands run.
    """
    if _looks_like_help_or_version_command(command):
        return None

    # Strip quoted content so keywords inside strings/arguments don't trigger
    # false positives (e.g., git commit -m "... setsid ...", python3 -c "os.setsid").
    unquoted = _strip_quotes(command)

    if _SHELL_LEVEL_BACKGROUND_RE.search(unquoted):
        return (
            "Foreground command uses shell-level background wrappers (nohup/disown/setsid). "
            "Re-send WITHOUT the wrapper as terminal(command=\"<cmd>\", background=true, "
            "notify_on_complete=true) so Hermes tracks the process, then run readiness "
            "checks and tests in separate commands."
        )

    if _INLINE_BACKGROUND_AMP_RE.search(unquoted) or _TRAILING_BACKGROUND_AMP_RE.search(unquoted):
        return (
            "Foreground command uses '&' backgrounding. Re-send WITHOUT the '&' as "
            "terminal(command=\"<cmd>\", background=true) — add notify_on_complete=true "
            "for bounded jobs — then run health checks and tests in follow-up terminal calls."
        )

    for pattern in _LONG_LIVED_FOREGROUND_PATTERNS:
        if pattern.search(unquoted):
            return (
                "This foreground command appears to start a long-lived server/watch process. "
                "Run it with background=true, verify readiness (health endpoint/log signal), "
                "then execute tests in a separate command."
            )

    return None


def _resolve_notification_flag_conflict(
    *,
    notify_on_complete: bool,
    watch_patterns,
    background: bool,
) -> tuple:
    """Decide what to do when both notify_on_complete and watch_patterns are set.

    These flags produce duplicate, delayed notifications when combined — one
    notification per watch-pattern match AND one on process exit, with async
    delivery that can spam the user long after the process ends. When both are
    set, we drop watch_patterns in favor of notify_on_complete (the more useful
    "let me know when it's done" signal) and return a human-readable note.

    Returns:
        (watch_patterns_to_use, conflict_note). conflict_note is "" when there
        is no conflict.
    """
    if background and notify_on_complete and watch_patterns:
        note = (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
        return None, note
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
) -> str:
    """Return the cwd for a command. Explicit ``workdir=`` overrides everything.

    Otherwise the session's own cwd RECORD (``get_session_cwd``) wins — it is
    written after every completed command for this session, so it IS the
    session's ``cd`` state, with no shared-env ambiguity: another session's
    ``cd`` lands in another record and can't affect us. A session with no
    record yet (first command) runs in ``default_cwd`` (config/override cwd),
    which is also what seeds a fresh environment.
    """
    if workdir:
        return workdir
    return get_session_cwd(session_key) or default_cwd


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    _runtime_agent_output_workdir: bool = False,
) -> str:
    """
    Execute a command in the configured terminal environment.

    Args:
        command: The command to execute
        background: Whether to run in background (default: False)
        timeout: Command timeout in seconds (default: from config)
        task_id: Unique identifier for environment isolation (optional)
        session_id: Conversation/session identifier for durable observability
        force: If True, skip dangerous command check (use after user confirms)
        workdir: Working directory for this command (optional, uses session cwd if not set)
        pty: If True, use pseudo-terminal for interactive CLI tools (local backend only)
        notify_on_complete: If True and background=True, you'll be notified exactly once when the process exits. The right choice for almost every long task. MUTUALLY EXCLUSIVE with watch_patterns.
        watch_patterns: List of strings to watch for in background output. HARD rate limit: 1 notification per 15s per process. After 3 strike windows in a row, watch_patterns is disabled and the session is auto-promoted to notify_on_complete. Use ONLY for rare, one-shot mid-process signals on long-lived processes (server readiness, migration-done markers). NEVER use in loops/batch jobs — error patterns there will hit the strike limit and get disabled. MUTUALLY EXCLUSIVE with notify_on_complete — set one, not both.

    Returns:
        str: JSON string with output, exit_code, and error fields

    Examples:
        # Execute a simple command
        >>> result = terminal_tool(command="ls -la /tmp")

        # Run a background task
        >>> result = terminal_tool(command="python server.py", background=True)

        # With custom timeout
        >>> result = terminal_tool(command="long_task.sh", timeout=300)
        
        # Force run after user confirmation
        # Note: force parameter is internal only, not exposed to model API
    """
    try:
        if not isinstance(command, str):
            logger.warning(
                "Rejected invalid terminal command value: %s",
                type(command).__name__,
            )
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            }, ensure_ascii=False)

        try:
            from tools.runtime_workdir import (
                AGENT_OUTPUT_WORKDIR,
                resolve_runtime_workdir,
            )

            requested_agent_output = (
                _runtime_agent_output_workdir or workdir == AGENT_OUTPUT_WORKDIR
            )
            workdir = resolve_runtime_workdir(workdir)
        except ValueError as exc:
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": str(exc),
                    "error_type": "runtime_workdir",
                    "status": "error",
                },
                ensure_ascii=False,
            )

        # Get configuration
        config = _get_env_config()
        env_type = config["env_type"]
        if requested_agent_output and env_type != "local":
            return json.dumps(
                {
                    "output": "",
                    "exit_code": -1,
                    "error": (
                        "workdir 'agent_output' is available only with the local "
                        "terminal backend; configure an explicit backend-visible "
                        "workdir for container or remote execution"
                    ),
                    "error_type": "runtime_workdir",
                    "status": "error",
                },
                ensure_ascii=False,
            )

        # Use task_id for environment isolation. By default all subagent
        # task_ids collapse back to "default" so the top-level agent and
        # every delegate_task child share one container; only task_ids with
        # a registered env override (RL benchmarks) get isolated sandboxes.
        effective_task_id = _resolve_container_task_id(task_id)
        raw_task_key = _managed_profile_registry_key(task_id)
        environment_profile_owner = None
        if os.environ.get("HERMES_MANAGED_GATEWAY") == "1":
            environment_profile_owner = _canonical_managed_profile_home()

        # Check per-task overrides (set by environments like TerminalBench2Env)
        # before falling back to global env var config. ``resolve_task_overrides``
        # reads the raw task id first then the collapsed container id, so a
        # CWD-only override (which collapses ``effective_task_id`` to
        # ``"default"``) is still found under its originating session id while
        # isolation-keyed RL/benchmark overrides keep resolving as before.
        overrides = resolve_task_overrides(task_id)
        
        # Select image based on env type, with per-task override support
        if env_type == "docker":
            image = overrides.get("docker_image") or config["docker_image"]
        elif env_type == "singularity":
            image = overrides.get("singularity_image") or config["singularity_image"]
        elif env_type == "modal":
            image = overrides.get("modal_image") or config["modal_image"]
        elif env_type == "daytona":
            image = overrides.get("daytona_image") or config["daytona_image"]
        else:
            image = ""

        cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
        # A per-task cwd override (registered by the gateway/TUI for workspace
        # tracking, or by RL/benchmark envs) wins over config["cwd"] — but
        # config["cwd"] was already sanitized for container backends in
        # _get_env_config() while the override is raw. On a container backend a
        # raw host path (e.g. a Windows desktop session's C:\Users\<user>, or a
        # POSIX /home/<user>) reaches `docker run -w <host-path>` and the
        # container fails to start (exit 125). Re-apply the same host/relative
        # path guard to the *resolved* cwd so the override can't bypass it.
        # Valid in-container override paths (RL/benchmark sandboxes that set
        # cwd to /workspace, /root, etc.) are absolute non-host paths and pass
        # through untouched.
        if env_type in _CONTAINER_BACKENDS and _is_unusable_container_cwd(cwd):
            if cwd != config["cwd"]:
                logger.info(
                    "Ignoring host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, config["cwd"],
                )
            cwd = config["cwd"]
        default_timeout = config["timeout"]

        # Validate an explicit timeout before it flows into deadline math.
        # ``timeout or default`` silently turns 0 into the default (0 can't mean
        # "no timeout" here), and a negative value is truthy so it would sail
        # through to ``deadline = now + timeout`` and fire an immediate,
        # nonsensical "-Ns" timeout. Reject non-positive values outright.
        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )
        effective_timeout = timeout or default_timeout

        # Reject foreground commands where the model explicitly requests
        # a timeout above FOREGROUND_MAX_TIMEOUT — nudge it toward background.
        if not background and timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            return tool_error(
                f"Foreground timeout {timeout}s exceeds the maximum of "
                f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                f"notify_on_complete=true for long-running commands."
            )

        # Guardrail: long-lived server/watch commands should run as managed
        # background sessions, not foreground shell hacks.
        if not background:
            guidance = _foreground_background_guidance(command)
            if guidance:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                }, ensure_ascii=False)

        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        if not background and not pty:
            camera_runtime_result = _run_camera_runtime_command_if_allowed(
                command,
                cwd=workdir or cwd,
                timeout=effective_timeout,
            )
            if camera_runtime_result is not None:
                return camera_runtime_result
            video_edit_runtime_result = _run_video_edit_runtime_command_if_allowed(
                command,
                cwd=workdir or cwd,
                timeout=effective_timeout,
            )
            if video_edit_runtime_result is not None:
                return video_edit_runtime_result
            agent_creator_result = _run_agent_creator_command_if_allowed(
                command,
                cwd=workdir or cwd,
                timeout=effective_timeout,
            )
            if agent_creator_result is not None:
                return agent_creator_result
            connector_runtime_result = _run_connector_runtime_command_if_allowed(
                command,
                cwd=workdir or cwd,
                timeout=effective_timeout,
            )
            if connector_runtime_result is not None:
                return connector_runtime_result
        else:
            lark_cli_result = _lark_cli_shell_guard_result(command)
            if lark_cli_result is not None:
                return lark_cli_result
            camera_runtime_result = _camera_runtime_shell_guard_result(command)
            if camera_runtime_result is not None:
                return camera_runtime_result
            video_edit_runtime_result = _video_edit_runtime_shell_guard_result(command)
            if video_edit_runtime_result is not None:
                return video_edit_runtime_result
            agent_creator_result = _agent_creator_shell_guard_result(command)
            if agent_creator_result is not None:
                return agent_creator_result
            connector_runtime_result = _connector_runtime_shell_guard_result(command)
            if connector_runtime_result is not None:
                return connector_runtime_result

        try:
            _late_prepare_video_edit_worker_before_terminal()
        except Exception as exc:
            logger.warning(
                "Trusted video-edit worker late preload failed before terminal: %s",
                type(exc).__name__,
            )
        _close_video_edit_worker_disk_trust()

        # Start cleanup thread
        _start_cleanup_thread()

        # Get or create environment.
        # Use a per-task creation lock so concurrent tool calls for the same
        # task_id wait for the first one to finish creating the sandbox,
        # instead of each creating their own (wasting Modal resources).
        env: Any = None
        with _env_lock:
            # Prefer the collapsed container id, but fall back to an env cached
            # under the raw task_id. Per-session surfaces (ACP/gateway/dashboard)
            # with a CWD-only override collapse to "default" for container
            # sharing, yet an env may already be cached under the originating
            # task_id; honor it instead of spawning a duplicate.
            _existing_key = (
                effective_task_id if effective_task_id in _active_environments
                else (raw_task_key if raw_task_key in _active_environments else None)
            )
            if _existing_key is not None:
                _last_activity[_existing_key] = time.time()
                if environment_profile_owner:
                    _environment_profile_owners[_existing_key] = (
                        environment_profile_owner
                    )
                env = _active_environments[_existing_key]
                needs_creation = False
            else:
                needs_creation = True

        if needs_creation:
            # Per-task lock: only one thread creates the sandbox, others wait
            with _creation_locks_lock:
                if effective_task_id not in _creation_locks:
                    _creation_locks[effective_task_id] = threading.Lock()
                task_lock = _creation_locks[effective_task_id]

            with task_lock:
                # Double-check after acquiring the per-task lock
                with _env_lock:
                    _existing_key = (
                        effective_task_id if effective_task_id in _active_environments
                        else (raw_task_key if raw_task_key in _active_environments else None)
                    )
                    if _existing_key is not None:
                        _last_activity[_existing_key] = time.time()
                        if environment_profile_owner:
                            _environment_profile_owners[_existing_key] = (
                                environment_profile_owner
                            )
                        env = _active_environments[_existing_key]
                        needs_creation = False

                if needs_creation:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        ssh_config = None
                        if env_type == "ssh":
                            ssh_config = {
                                "host": config.get("ssh_host", ""),
                                "user": config.get("ssh_user", ""),
                                "port": config.get("ssh_port", 22),
                                "key": config.get("ssh_key", ""),
                                "persistent": config.get("ssh_persistent", False),
                            }

                        container_config = None
                        if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
                            container_config = {
                                "container_cpu": config.get("container_cpu", 1),
                                "container_memory": config.get("container_memory", 5120),
                                "container_disk": config.get("container_disk", 51200),
                                "container_persistent": config.get("container_persistent", True),
                                "modal_mode": config.get("modal_mode", "auto"),
                                "vercel_runtime": config.get("vercel_runtime", ""),
                                "docker_volumes": config.get("docker_volumes", []),
                                "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
                                "docker_forward_env": config.get("docker_forward_env", []),
                                "docker_env": config.get("docker_env", {}),
                                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                                "docker_extra_args": config.get("docker_extra_args", []),
                                "docker_shm_size": config.get("docker_shm_size", "1g"),
                                "docker_network": config.get("docker_network", True),
                                "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
                                "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
                            }

                        local_config = None
                        if env_type == "local":
                            local_config = {
                                "persistent": config.get("local_persistent", False),
                            }

                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=ssh_config,
                            container_config=container_config,
                            local_config=local_config,
                            task_id=effective_task_id,
                            host_cwd=config.get("host_cwd"),
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": f"Terminal tool disabled: environment creation failed ({e})",
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        if environment_profile_owner:
                            _environment_profile_owners[effective_task_id] = (
                                environment_profile_owner
                            )
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        assert env is not None  # all creation failure paths return above

        # The session key that drives cwd records: get_current_session_key()'s
        # contextvar doesn't cross tool-worker threads, so fall back to the raw
        # task_id (which IS the session_key for the top-level agent) — a
        # stable, thread-safe anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        # Hard-block: gateway lifecycle commands (systemctl/launchctl/hermes
        # restart|stop targeting hermes-gateway) must never run inside the
        # gateway process itself. The restart would SIGTERM the gateway, which
        # kills this very subprocess before it can complete — the service may
        # never restart. This mirrors the `hermes gateway restart` guard in
        # hermes_cli/gateway.py and the cron-path guard in hermes_cli/cron.py,
        # but applies unconditionally (force=True cannot help here).
        if os.environ.get("_HERMES_GATEWAY") == "1":
            from cron.lifecycle_guard import (
                contains_gateway_lifecycle_command_or_referenced_script,
                contains_launchctl_submit_command,
            )
            if contains_launchctl_submit_command(command):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: launchctl submit/bootstrap registers a persistent "
                        "KeepAlive job and is unsafe from inside the gateway process. "
                        "Use Hermes cron for one-shot delayed work, or install an "
                        "explicit LaunchAgent from a separate shell."
                    ),
                    "status": "error",
                }, ensure_ascii=False)
            guard_cwd_base = get_session_cwd(session_key)
            if guard_cwd_base is None:
                guard_cwd_base = getattr(env, "cwd", None) or cwd
            guard_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=guard_cwd_base,
                session_key=session_key,
            )

            def _read_script_in_env(script_path: str) -> Optional[str]:
                """Best-effort script read; uses env.execute only when local read fails.

                For local backends the script path is on the host filesystem. For
                SSH/Modal/Daytona the same path is remote; the local read misses, so we
                fall back to ``env.execute('cat ...')``.
                """
                if env is None:
                    return None
                try:
                    local_path = Path(script_path).expanduser()
                    if not local_path.is_absolute():
                        local_path = Path(guard_cwd) / local_path
                    if local_path.is_file():
                        metadata = local_path.stat()
                        if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= 1024 * 1024:
                            data = local_path.read_bytes()
                            if len(data) <= 1024 * 1024:
                                return data.decode("utf-8", errors="replace")
                except Exception:
                    pass
                # Remote / sandboxed backend: read via the environment's shell.
                try:
                    result = env.execute(f"cat {shlex.quote(script_path)}")
                    if result.get("returncode", -1) == 0:
                        return result.get("output", "")
                except Exception:
                    pass
                return None

            if contains_gateway_lifecycle_command_or_referenced_script(
                command,
                cwd=guard_cwd,
                read_remote_script=_read_script_in_env,
            ):
                return json.dumps({
                    "output": "",
                    "exit_code": 1,
                    "error": (
                        "Blocked: command or referenced script cannot restart or stop "
                        "the gateway from inside the gateway process. The gateway would "
                        "kill this command before it could complete (SIGTERM propagates "
                        "to child processes). Run `hermes gateway restart` from a "
                        "separate shell outside the running gateway."
                    ),
                    "status": "error",
                }, ensure_ascii=False)

        # Pre-exec security checks (tirith + dangerous command detection)
        # Skip check if force=True (user has confirmed they want to run it)
        approval_note = None
        # True when the user explicitly approved this run (or pre-confirmed via
        # force).  Drives the clean-interrupt-slate clear before env.execute so
        # an approved command can't be SIGINT-killed by a bit that landed during
        # the approval-wait (see clear_current_thread_interrupt).
        _approved_run = bool(force)
        if not force:
            approval = _check_all_guards(
                command, env_type,
                has_host_access=_docker_has_host_access(config),
            )
            if not approval["approved"]:
                # Check if this is an approval_required (gateway ask mode)
                if approval.get("status") == "pending_approval":
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": "",
                        "status": "pending_approval",
                        "approval_pending": True,
                        "command": approval.get("command", command),
                        "description": approval.get("description", "command flagged"),
                        "pattern_key": approval.get("pattern_key", ""),
                        "smart_denied": approval.get("smart_denied", False),
                        "allow_permanent": approval.get("allow_permanent", True),
                    }, ensure_ascii=False)
                # Command was blocked
                desc = approval.get("description", "command flagged")
                fallback_msg = (
                    f"Command denied: {desc}. "
                    "Use the approval prompt to allow it, or rephrase the command."
                )
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": approval.get("message", fallback_msg),
                    "status": "blocked"
                }, ensure_ascii=False)
            # Track whether approval was explicitly granted by the user
            if approval.get("user_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command required approval ({desc}) and was approved by the user."
                _approved_run = True
            elif approval.get("smart_approved"):
                desc = approval.get("description", "flagged as dangerous")
                approval_note = f"Command was flagged ({desc}) and auto-approved by smart approval."

        # Validate workdir against shell injection
        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        # Managed lark-cli is intentionally brokered only after the ordinary
        # terminal approval pass. That preserves the CLI's read/write/high-risk
        # approval semantics while keeping OAuth files out of the model shell.
        if not background and not pty and env_type == "local":
            lark_cli_result = _run_lark_cli_command_if_allowed(
                command,
                timeout=effective_timeout,
            )
            if lark_cli_result is not None:
                return lark_cli_result

        # Prepare command for execution
        pty_disabled_reason = None
        effective_pty = pty
        if pty and _command_requires_pipe_stdin(command):
            effective_pty = False
            pty_disabled_reason = (
                "PTY disabled for this command because it expects piped stdin/EOF "
                "(for example gh auth login --with-token). For local background "
                "processes, call process(action='close') after writing so it receives "
                "EOF."
            )

        # The session key is already computed above the gateway guard.
        if background:
            # Spawn a tracked background process via the process registry.
            # For local backends: uses subprocess.Popen with output buffering.
            # For non-local backends: runs inside the sandbox via env.execute().
            from tools.process_registry import process_registry

            effective_cwd = _resolve_command_cwd(
                workdir=workdir,
                default_cwd=cwd,
                session_key=session_key,
            )
            try:
                if env_type == "local":
                    proc_session = process_registry.spawn_local(
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                        env_vars=env.env if hasattr(env, 'env') else None,
                        use_pty=effective_pty,
                    )
                else:
                    proc_session = process_registry.spawn_via_env(
                        env=env,
                        command=command,
                        cwd=effective_cwd,
                        task_id=effective_task_id,
                        session_key=session_key,
                    )

                result_data = {
                    "output": "Background process started",
                    "session_id": proc_session.id,
                    "pid": proc_session.pid,
                    "exit_code": 0,
                    "error": None,
                }
                # Background spawns detached and returns exit_code 0 immediately;
                # it never inline-polls is_interrupted(), so the stale-bit kill
                # cannot occur here and this note never co-occurs with rc=130.
                if approval_note:
                    result_data["approval"] = approval_note
                if pty_disabled_reason:
                    result_data["pty_note"] = pty_disabled_reason

                # Nudge: background=True without notify_on_complete=True OR
                # watch_patterns is a silent process. The agent has NO way to
                # learn it finished short of calling process(action="poll"/"wait")
                # explicitly. That's correct only for genuine long-lived
                # processes that never exit (servers, watchers). For every
                # bounded task (tests, builds, CI pollers, deploys, batch
                # jobs) the agent almost certainly wanted notification and
                # forgot the flag. May 2026 PR #31231 incident: bg CI poller
                # ran fine, exited green, agent never noticed — user had to
                # surface the result. Cheap nudge here costs ~one read for
                # server cases (false positive) and prevents silent
                # blindness for bounded-task cases (false negative).
                if background and not notify_on_complete and not watch_patterns:
                    result_data["hint"] = (
                        "background=true without notify_on_complete=true means "
                        "this process runs SILENTLY — you will not be told when "
                        "it exits. If this is a bounded task (test suite, build, "
                        "CI poller, deploy, anything with a defined end), you "
                        "almost certainly wanted notify_on_complete=true so the "
                        "system pings you on exit. Re-launch with "
                        "notify_on_complete=true, or call process(action='poll') "
                        "/ process(action='wait') yourself to learn the outcome. "
                        "Only ignore this hint for genuine long-lived processes "
                        "that never exit (servers, watchers, daemons)."
                    )

                # Nudge: homebrewed CI watcher built from `gh pr view`
                # `--json statusCheckRollup` or `gh pr checks` piped through
                # `jq` is the #1 cause of silent CI-watcher failures in
                # hermes-agent dev work. May 2026 PRs that surfaced this
                # exact failure mode: #31329, #31448, #31695, #31709, #31745,
                # #32264, #33131. Failure modes seen:
                #   * `gh pr view --json statusCheckRollup --jq ...` with
                #     `from_entries` choking on null `conclusion` keys, loop
                #     silently exits with empty status, never terminates.
                #   * `for i in $(seq 1 60); do ... 2>&1` block-buffered stdout
                #     never flushed to background-process capture; SIGTERM
                #     cuts the buffer before flush; `process(action='log')`
                #     returns total_lines=0 forever.
                #   * conclusion vs. status field confusion: filtering for
                #     `PENDING` in `.conclusion` while in-progress checks have
                #     empty conclusion → poller declares all-green while 18/23
                #     checks still IN_PROGRESS.
                #   * grepping for TTY-only banners ("All checks were
                #     successful") that never appear when stdout is piped.
                # The canonical patterns in the green-ci-policy skill avoid
                # every one of these — drive the loop off exit codes or on
                # tab-separated `awk -F"\t" "$2==\"pending\""` (column 2).
                # The detector here is deliberately narrow: it flags the
                # statusCheckRollup JSON-API path and the `gh pr checks` +
                # jq combination, but NOT the canonical column-2 awk
                # poller (which uses awk on tabs, not as a generic
                # stdout parser). When we detect the homebrew shape, point
                # the agent at the canonical snippet rather than letting
                # it ship another broken poller.
                if background and command:
                    _gh = ("gh pr view" in command or "gh pr checks" in command)
                    _has_jq = (
                        " jq " in command or "| jq" in command or "$(jq" in command
                    )
                    _bad_shape = (
                        # The JSON-API anti-pattern. Even without jq, going
                        # through `--json statusCheckRollup` + parsing puts
                        # you in conclusion-vs-status field hell.
                        "statusCheckRollup" in command
                        # gh pr checks piped to jq is also wrong — `gh pr
                        # checks` doesn't emit JSON, so any `| jq` here is
                        # confused intent. The canonical column-2 poller
                        # uses awk-on-tabs, not jq.
                        or (_gh and _has_jq)
                    )
                    if _bad_shape:
                        existing = result_data.get("hint", "")
                        canonical_hint = (
                            "This looks like a homebrewed CI poller built from "
                            "`gh pr view --json statusCheckRollup` and/or "
                            "`gh pr checks | jq`. That shape has burned us "
                            "repeatedly in hermes-agent dev work (PRs #31329, "
                            "#31448, #31695, #31709, #31745, #32264, #33131) — "
                            "stdout buffering kills output capture, jq null-key "
                            "edge cases silently exit the loop, conclusion-vs-"
                            "status field confusion exits early with bogus "
                            "all-green verdicts, TTY-only summary banners "
                            "never appear when piped. Use the canonical "
                            "snippets in the green-ci-policy skill instead: "
                            "the exit-code-driven `gh pr checks $PR >/dev/null` "
                            "(rc 0 = green, 8 = pending, else fail) for "
                            "exit-on-first-fail behavior, or the column-2 "
                            "awk-on-tabs poller "
                            "(`awk -F\"\\t\" \"$2==\\\"pending\\\"\"`) for "
                            "sharded matrices. Load skill_view("
                            "name='github/hermes-agent-dev', "
                            "file_path='references/green-ci-policy.md') for "
                            "the verbatim snippets. If you must roll a custom "
                            "loop with rich structured output, write each tick "
                            "to a known file (`tee -a /tmp/ci.log`) and rely "
                            "on `process(action='log')` to read THAT file — "
                            "do not rely on background-process stdout capture "
                            "for line-buffered shell loops."
                        )
                        result_data["hint"] = (
                            existing + "\n\n" + canonical_hint if existing
                            else canonical_hint
                        )

                # Populate routing metadata on the session so that
                # watch-pattern and completion notifications can be
                # routed back to the correct chat/thread.
                if background and (notify_on_complete or watch_patterns):
                    from gateway.session_context import (
                        async_delivery_supported as _async_ok,
                        get_session_env as _gse,
                    )

                    # Finite sessions (stateless HTTP requests and one-shot
                    # Kanban workers) cannot route a completion back to the
                    # agent after the turn/process ends. Refuse the promise:
                    # drop the flags and tell the agent to poll.
                    if not _async_ok():
                        notify_on_complete = False
                        watch_patterns = None
                        result_data["notify_on_complete"] = False
                        result_data["notify_unsupported"] = (
                            "notify_on_complete / watch_patterns are not available in "
                            "this session — it cannot receive an async completion after "
                            "the turn ends (a one-shot runner such as `hermes -z`, a "
                            "cron job, a Kanban worker, or a stateless HTTP endpoint). "
                            "The process is "
                            "running in the background; retrieve its result with "
                            "process(action='poll') or process(action='wait')."
                        )
                        logger.info(
                            "background proc %s: async delivery unsupported on this "
                            "session; notify_on_complete/watch_patterns disabled",
                            proc_session.id,
                        )
                    else:
                        _gw_platform = _gse("HERMES_SESSION_PLATFORM", "")
                        if _gw_platform:
                            _gw_chat_id = _gse("HERMES_SESSION_CHAT_ID", "")
                            _gw_thread_id = _gse("HERMES_SESSION_THREAD_ID", "")
                            _gw_user_id = _gse("HERMES_SESSION_USER_ID", "")
                            _gw_user_name = _gse("HERMES_SESSION_USER_NAME", "")
                            _gw_message_id = _gse("HERMES_SESSION_MESSAGE_ID", "")
                            proc_session.watcher_platform = _gw_platform
                            proc_session.watcher_chat_id = _gw_chat_id
                            proc_session.watcher_user_id = _gw_user_id
                            proc_session.watcher_user_name = _gw_user_name
                            proc_session.watcher_thread_id = _gw_thread_id
                            proc_session.watcher_message_id = _gw_message_id

                # Mutual exclusion: if both notify_on_complete and watch_patterns
                # are set, drop watch_patterns. The combination produces duplicate
                # notifications (one per match + one on exit) that deliver
                # asynchronously and can spam the user long after the process ends.
                # notify_on_complete is the more useful signal for "let me know
                # when the task finishes"; watch_patterns should be reserved for
                # standalone mid-process signals on long-lived processes.
                watch_patterns, conflict_note = _resolve_notification_flag_conflict(
                    notify_on_complete=bool(notify_on_complete),
                    watch_patterns=watch_patterns,
                    background=bool(background),
                )
                if conflict_note:
                    logger.warning("background proc %s: %s", proc_session.id, conflict_note)
                    result_data["watch_patterns_ignored"] = conflict_note

                # Mark for agent notification on completion
                if notify_on_complete and background:
                    proc_session.notify_on_complete = True
                    result_data["notify_on_complete"] = True

                    # In gateway mode, auto-register a fast watcher so the
                    # gateway can detect completion and trigger a new agent
                    # turn.  CLI mode uses the completion_queue directly.
                    if proc_session.watcher_platform:
                        proc_session.watcher_interval = 5
                        process_registry.pending_watchers.append({
                            "session_id": proc_session.id,
                            "check_interval": 5,
                            "session_key": session_key,
                            "profile_owner": proc_session.profile_owner,
                            "platform": proc_session.watcher_platform,
                            "chat_id": proc_session.watcher_chat_id,
                            "user_id": proc_session.watcher_user_id,
                            "user_name": proc_session.watcher_user_name,
                            "thread_id": proc_session.watcher_thread_id,
                            "message_id": proc_session.watcher_message_id,
                            "notify_on_complete": True,
                        })

                # Set watch patterns for output monitoring
                if watch_patterns and background:
                    proc_session.watch_patterns = list(watch_patterns)
                    result_data["watch_patterns"] = proc_session.watch_patterns

                return json.dumps(result_data, ensure_ascii=False)
            except Exception as e:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Failed to start background process: {str(e)}"
                }, ensure_ascii=False)
        else:
            # Run foreground command with retry logic
            max_retries = 3
            retry_count = 0
            result = None
            command_cwd = None

            # Clean interrupt slate for an approved command, ONCE before the
            # retry loop: drop a stale bit that landed on this thread during the
            # approval-wait so it can't SIGINT the just-approved run.  Do NOT
            # re-clear inside the loop -- a genuine interrupt arriving during the
            # backoff sleep between retries must survive and abort the command
            # (caught by the next attempt's _wait_for_process poll loop -> 130).
            if _approved_run:
                from tools.interrupt import clear_current_thread_interrupt
                clear_current_thread_interrupt()

            while retry_count <= max_retries:
                try:
                    command_cwd = _resolve_command_cwd(
                        workdir=workdir,
                        default_cwd=cwd,
                        session_key=session_key,
                    )
                    execute_kwargs = {
                        "timeout": effective_timeout,
                        "cwd": command_cwd,
                        # Foreground model-facing output: cap retention while
                        # streaming (head/tail window) so a verbose command
                        # can't OOM the gateway before truncation (#64435).
                        # Internal env.execute() consumers (file ops cat
                        # reads, RPC reads) intentionally stay unbounded.
                        "bounded_capture": True,
                    }
                    result = env.execute(command, **execute_kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "timeout" in error_str:
                        return json.dumps({
                            "output": "",
                            "exit_code": 124,
                            "error": f"Command timed out after {effective_timeout} seconds"
                        }, ensure_ascii=False)
                    
                    # Retry on transient errors
                    if retry_count < max_retries:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                       wait_time, retry_count, max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                        time.sleep(wait_time)
                        continue
                    
                    logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                 max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                    return json.dumps({
                        "output": "",
                        "exit_code": -1,
                        "error": f"Command execution failed: {type(e).__name__}: {str(e)}"
                    }, ensure_ascii=False)
                
                # Got a result
                break

            # Dual-write (cwd rearch step 1): the env's post-command tracking
            # (marker parse / local sync) has just updated env.cwd with the
            # directory this command finished in. That cwd belongs to THIS
            # session — record it under the session key so the durable record
            # never depends on the shared env surviving or on who drives the
            # env next.
            #
            # BUT: a per-command ``workdir`` override is transient by contract
            # (docstring: "Working directory for this command"). Recording it
            # would hijack the session's durable cwd for every later command
            # that doesn't pass ``workdir``. Skip the dual-write in that case.
            if not workdir:
                record_session_cwd(session_key, getattr(env, "cwd", None))

            # Extract output
            output = result.get("output", "")
            returncode = result.get("returncode", 0)
            # Spill metadata from the bounded collector: present only when
            # output overflowed the capture window (see _wait_for_process).
            spill_total_chars = result.get("output_total_chars")
            spill_file_path = result.get("full_output_path")

            # Add helpful message for sudo failures in messaging context
            output = _handle_sudo_failure(output, env_type)

            sudo_auth_failed = _sudo_wrong_password_failure(output)
            sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
                command, output
            )
            if sudo_cache_cleared:
                has_sudo_prompt_callback = _get_sudo_password_callback() is not None
                if has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE"):
                    output += (
                        "\n\n⚠️ Sudo authentication failed — cached password "
                        "cleared. You will be prompted again on the next sudo "
                        "command."
                    )

            # Foreground terminal output canonicalization seam: process capture
            # is already bounded by BaseEnvironment before sudo checks and hooks
            # run. Plugins may replace that bounded string; replacements are
            # still subject to the final output limit below.
            # The hook is fail-open, and the first valid string return wins.
            try:
                from hermes_cli.lifecycle import invoke_hook
                hook_results = invoke_hook(
                    "transform_terminal_output",
                    command=command,
                    output=output,
                    returncode=returncode,
                    task_id=effective_task_id or "",
                    env_type=env_type,
                )
                for hook_result in hook_results:
                    if isinstance(hook_result, str):
                        output = hook_result
                        break
            except Exception:
                pass
            
            # Truncate output if too long, keeping both head and tail
            from tools.tool_output_limits import get_max_bytes
            MAX_OUTPUT_CHARS = get_max_bytes()
            if len(output) > MAX_OUTPUT_CHARS:
                head_chars = int(MAX_OUTPUT_CHARS * 0.4)  # 40% head (error messages often appear early)
                tail_chars = MAX_OUTPUT_CHARS - head_chars  # 60% tail (most recent/relevant output)
                omitted = len(output) - head_chars - tail_chars
                truncated_notice = (
                    f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
                    f"out of {len(output)} total] ...\n\n"
                )
                output = output[:head_chars] + truncated_notice + output[-tail_chars:]

            # Strip ANSI escape sequences so the model never sees terminal
            # formatting — prevents it from copying escapes into file writes.
            from tools.ansi_strip import strip_ansi
            output = strip_ansi(output)

            # Redact secrets from command output. For source/config dumps
            # (MAX_TOKENS=100, "apiKey": "x" fixtures, postgresql:// f-string
            # templates) the ENV/JSON/template passes are skipped to avoid
            # false positives (code_file=True). But for env-dump commands
            # (env/printenv/set/export/declare) the output IS a KEY=value
            # credential dump, so redact_terminal_output runs the ENV pass
            # (code_file=False) to mask opaque tokens with no vendor prefix.
            # Real prefixes, auth headers, JWTs, private keys are masked in
            # both modes. See issue #43025.
            from agent.redact import redact_terminal_output
            output = redact_terminal_output(output.strip(), command) if output else ""

            # Interpret non-zero exit codes that aren't real errors
            # (e.g. grep=1 means "no matches", diff=1 means "files differ")
            exit_note = _interpret_exit_code(command, returncode)

            # Output-pattern failure hints: map well-known error shapes
            # (command-not-found, ModuleNotFoundError, gh field drift,
            # merge conflicts, ...) to one short recovery hint so the model
            # fixes the root cause on the next call instead of spending
            # turns on re-diagnosis. See tools/terminal_hints.py.
            failure_hint = None
            if returncode != 0 and not exit_note:
                try:
                    from tools.terminal_hints import annotate_failure
                    failure_hint = annotate_failure(command, returncode, output)
                except Exception:
                    failure_hint = None

            result_dict = {
                "output": output,
                "exit_code": returncode,
                "error": None,
            }
            # cwd echo: when the command changed the session's working
            # directory (cd, pushd, ...), tell the model where it ended up.
            # Production mining shows 60% of terminal calls carry a
            # defensive 'cd X && ' prefix because the model can't see cwd
            # state; echoing it on change removes the guesswork (pattern
            # borrowed from crush's <cwd> injection).
            try:
                post_cwd = getattr(env, "cwd", None)
                if post_cwd and command_cwd and os.path.realpath(str(post_cwd)) != os.path.realpath(str(command_cwd)):
                    result_dict["cwd"] = str(post_cwd)
            except Exception:
                pass
            # Truncation metadata (codex/opencode/goose pattern): report the
            # pre-truncation size and a spill-file handle so the model can
            # retrieve the omitted middle with read_file/search_files instead
            # of re-running the command. The spill was written raw by the
            # collector; redact it here with the same pass as the visible
            # output so no secret persists unmasked on disk.
            if spill_file_path:
                try:
                    _sp = Path(spill_file_path)
                    raw_spill = _sp.read_text(encoding="utf-8", errors="replace")
                    _sp.write_text(
                        redact_terminal_output(strip_ansi(raw_spill), command),
                        encoding="utf-8", errors="replace",
                    )
                    result_dict["output_total_chars"] = spill_total_chars
                    result_dict["full_output_path"] = spill_file_path
                    result_dict["truncation_note"] = (
                        "Output exceeded the capture window (head+tail shown). "
                        f"Full output ({spill_total_chars:,} chars) saved to "
                        f"{spill_file_path} — search it with search_files or page it "
                        "with read_file instead of re-running the command."
                    )
                except Exception:
                    logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
                    try:
                        Path(spill_file_path).unlink()
                    except OSError:
                        pass
            try:
                from agent.verification_evidence import record_terminal_result

                evidence = record_terminal_result(
                    command=command,
                    cwd=command_cwd,
                    session_id=session_id or task_id or effective_task_id or "default",
                    exit_code=returncode,
                    output=output,
                )
                if evidence:
                    result_dict["verification_evidence"] = {
                        "status": evidence.get("status"),
                        "kind": evidence.get("kind"),
                        "scope": evidence.get("scope"),
                        "canonical_command": evidence.get("canonical_command"),
                    }
            except Exception:
                logger.debug("verification evidence recording failed", exc_info=True)
            if approval_note:
                # Treat rc=130 as an interrupt only when the executor's marker is
                # present.  A command can legitimately exit 130 on its own
                # (e.g. `bash -c 'exit 130'`); _wait_for_process returns the
                # child's natural returncode there with no marker, and that must
                # NOT be relabelled as a user interrupt in the audit note.
                if returncode == 130 and "[Command interrupted]" in output:
                    # Approved command was interrupted mid-run by a genuine Stop.
                    # Keep the audit trail but never imply success: the bare
                    # "...approved by the user." note must not co-occur with the
                    # interrupt exit code (satisfies the 3-part-signature DONE).
                    result_dict["approval"] = approval_note.rstrip(".") + ", then interrupted."
                else:
                    result_dict["approval"] = approval_note
            if exit_note:
                result_dict["exit_code_meaning"] = exit_note
            if failure_hint:
                result_dict["hint"] = failure_hint
            if sudo_auth_failed:
                result_dict["sudo_auth_failed"] = True
            if sudo_cache_cleared:
                result_dict["sudo_cache_cleared"] = True

            return json.dumps(result_dict, ensure_ascii=False)

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        logger.error("terminal_tool exception:\n%s", tb_str)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}",
            "traceback": tb_str,
            "status": "error"
        }, ensure_ascii=False)


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
        env_type = config["env_type"]

        if env_type == "local":
            return True

        elif env_type == "docker":
            from tools.environments.docker import find_docker
            docker = find_docker()
            if not docker:
                logger.error("Docker executable not found in PATH or common install locations")
                return False
            result = subprocess.run([docker, "version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            return result.returncode == 0

        elif env_type == "singularity":
            executable = shutil.which("apptainer") or shutil.which("singularity")
            if executable:
                result = subprocess.run([executable, "--version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
                return result.returncode == 0
            return False

        elif env_type == "ssh":
            if not config.get("ssh_host") or not config.get("ssh_user"):
                logger.error(
                    "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
                    "are not both set. Configure both or switch TERMINAL_ENV to 'local'."
                )
                return False
            return True

        elif env_type == "modal":
            modal_state = _get_modal_backend_state(config.get("modal_mode"))
            if modal_state["selected_backend"] == "managed":
                return True

            if modal_state["selected_backend"] != "direct":
                if modal_state["managed_mode_blocked"]:
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but "
                        "Nous Tool Gateway access is not currently available and no direct "
                        "Modal credentials/config were found. %s Choose "
                        "TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                if modal_state["mode"] == "managed":
                    logger.error(
                        "Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed "
                        "tool gateway is unavailable. %s",
                        nous_tool_gateway_unavailable_message(
                            "managed Modal execution",
                        ),
                    )
                    return False
                elif modal_state["mode"] == "direct":
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=managed/auto."
                        )
                    else:
                        logger.error(
                            "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                            "Modal credentials/config were found. Configure Modal or choose "
                            "TERMINAL_MODAL_MODE=auto."
                        )
                    return False
                else:
                    if managed_nous_tools_enabled():
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config or managed "
                            "tool gateway was found. Configure Modal, set up the managed gateway, "
                            "or choose a different TERMINAL_ENV."
                        )
                    else:
                        logger.error(
                            "Modal backend selected but no direct Modal credentials/config was found. "
                            "Configure Modal or choose a different TERMINAL_ENV."
                        )
                    return False

            if importlib.util.find_spec("modal") is None:
                logger.error("modal is required for direct modal terminal backend: pip install modal")
                return False

            return True

        elif env_type == "vercel_sandbox":
            return _check_vercel_sandbox_requirements(config)

        elif env_type == "daytona":
            from daytona import Daytona  # noqa: F401 — SDK presence check
            from agent.secret_scope import get_secret
            return get_secret("DAYTONA_API_KEY") is not None

        else:
            logger.error(
                "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
                "modal, daytona, vercel_sandbox, ssh.",
                env_type,
            )
            return False
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


if __name__ == "__main__":
    # Simple test when run directly
    print("Terminal Tool Module")
    print("=" * 50)
    
    config = _get_env_config()
    print("\nCurrent Configuration:")
    print(f"  Environment type: {config['env_type']}")
    print(f"  Docker image: {config['docker_image']}")
    print(f"  Modal image: {config['modal_image']}")
    print(f"  Working directory: {config['cwd']}")
    print(f"  Default timeout: {config['timeout']}s")
    print(f"  Lifetime: {config['lifetime_seconds']}s")

    if not check_terminal_requirements():
        print("\n❌ Requirements not met. Please check the messages above.")
        sys.exit(1)

    print("\n✅ All requirements met!")
    print("\nAvailable Tool:")
    print("  - terminal_tool: Execute commands in sandboxed environments")

    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_tool(command='ls -la')")
    print("  ")
    print("  # Run a background task")
    print("  result = terminal_tool(command='python server.py', background=True)")

    print("\nEnvironment Variables:")
    default_img = "nikolaik/python-nodejs:python3.11-nodejs20"
    print(
        "  TERMINAL_ENV: "
        f"{os.getenv('TERMINAL_ENV', 'local')} "
        "(local/docker/singularity/modal/daytona/vercel_sandbox/ssh)"
    )
    print(f"  TERMINAL_DOCKER_IMAGE: {os.getenv('TERMINAL_DOCKER_IMAGE', default_img)}")
    print(f"  TERMINAL_SINGULARITY_IMAGE: {os.getenv('TERMINAL_SINGULARITY_IMAGE', f'docker://{default_img}')}")
    print(f"  TERMINAL_MODAL_IMAGE: {os.getenv('TERMINAL_MODAL_IMAGE', default_img)}")
    print(f"  TERMINAL_DAYTONA_IMAGE: {os.getenv('TERMINAL_DAYTONA_IMAGE', default_img)}")
    print(f"  TERMINAL_CWD: {os.getenv('TERMINAL_CWD', _safe_getcwd())}")
    from hermes_constants import display_hermes_home as _dhh
    print(f"  TERMINAL_SANDBOX_DIR: {os.getenv('TERMINAL_SANDBOX_DIR', f'{_dhh()}/sandboxes')}")
    print(f"  TERMINAL_TIMEOUT: {os.getenv('TERMINAL_TIMEOUT', '60')}")
    print(f"  TERMINAL_LIFETIME_SECONDS: {os.getenv('TERMINAL_LIFETIME_SECONDS', '300')}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify_on_complete=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; use background=true for longer commands.",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path), or 'agent_output' when the platform exposes a managed output directory for the current agent and the local terminal backend is active. Defaults to the session working directory."
            },
            "pty": {
                "type": "boolean",
                "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                "default": False
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "With background=true: get exactly one notification when the process exits. The right choice for nearly every bounded long task — set it and keep working. MUTUALLY EXCLUSIVE with watch_patterns (watch_patterns is dropped when both are set).",
                "default": False
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Strings to watch for in background output. ONLY for rare one-shot mid-process signals on processes that never exit (e.g. ['Application startup complete'] on a server). NOT for end-of-run markers (use notify_on_complete) and NOT for per-iteration patterns like 'ERROR' in loops — rate-limited to 1 notification/15s; repeated over-firing auto-disables it and falls back to notify-on-exit. When in doubt, use notify_on_complete. MUTUALLY EXCLUSIVE with notify_on_complete."
            }
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    from tools.runtime_workdir import AGENT_OUTPUT_ARG

    # registry 解析层注入的内部标记到此为止：pop 掉避免作为业务参数外溢。
    runtime_agent_output = bool(args.pop(AGENT_OUTPUT_ARG, False))
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=args.get("notify_on_complete", False),
        watch_patterns=args.get("watch_patterns"),
        _runtime_agent_output_workdir=runtime_agent_output,
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
