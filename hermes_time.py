"""
Timezone-aware clock for Hermes.

Provides a single ``now()`` helper that returns a timezone-aware datetime
based on the user's configured IANA timezone (e.g. ``Asia/Kolkata``).

Resolution order:
  1. ``HERMES_TIMEZONE`` environment variable
  2. ``timezone`` key in ``~/.hermes/config.yaml``
  3. OS system tz: ``/etc/localtime`` symlink → ``/etc/timezone`` → ``timedatectl``
  4. Falls back to the server's local time (``datetime.now().astimezone()``)

Resolution is fingerprint-gated: a change to any source (env / config.yaml /
/etc/timezone / /etc/localtime) is picked up on the next call WITHOUT a
restart, so an APP → timedatectl → /etc/timezone change propagates live.

Invalid timezone values log a warning and fall back safely — Hermes never
crashes due to a bad timezone string.
"""

import logging
import os
import subprocess
from datetime import datetime
from hermes_constants import get_config_path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python 3.8 fallback (shouldn't be needed — Hermes requires 3.9+)
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

# OS timezone sources — module-level so tests can monkeypatch.
ETC_TIMEZONE = "/etc/timezone"
ETC_LOCALTIME = "/etc/localtime"
_ZONEINFO_MARKERS = ("/usr/share/zoneinfo/", "/var/db/timezone/zoneinfo/")

# Cached state — re-resolved when the source fingerprint changes (live tz),
# so an on-disk timezone change takes effect on the next call without a
# restart. Call reset_cache() to force re-resolution.
_cached_tz: Optional[ZoneInfo] = None
_cached_tz_name: Optional[str] = None
_cached_fp = None


def reset_cache() -> None:
    """Force re-resolution on the next call (tests / after config edits)."""
    global _cached_tz, _cached_tz_name, _cached_fp
    _cached_tz = None
    _cached_tz_name = None
    _cached_fp = None


def _is_zettlab_device_mode() -> bool:
    return os.getenv("ZET_AGENT_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def _read_config_timezone() -> str:
    try:
        # Prefer the shared cached raw-config reader (mtime/size-keyed cache +
        # libyaml C loader) — a direct yaml.safe_load of a large config.yaml
        # costs ~100ms+ and this used to run inside the FIRST system prompt
        # build, on the time-to-first-token critical path.
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
        except Exception:
            import yaml
            config_path = get_config_path()
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {}
        if cfg:
            # Managed scope: an administrator can pin ``timezone`` too. Overlay
            # via the shared helper (fail-open) since this reads config.yaml directly.
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            tz_cfg = cfg.get("timezone", "")
            if isinstance(tz_cfg, str) and tz_cfg.strip():
                return tz_cfg.strip()
    except Exception:
        pass
    return ""


def _read_os_timezone() -> str:
    # /etc/localtime (what the OS clock uses) wins; stale-prone /etc/timezone is a fallback.
    for reader in (_read_localtime_symlink, _read_etc_timezone, _read_timedatectl):
        tz_os = reader()
        if tz_os:
            return tz_os
    return ""


def _resolve_timezone_name() -> str:
    """Read the configured IANA timezone string (or empty string)."""
    # 1. Environment variable (highest priority — explicit operator override).
    tz_env = os.getenv("HERMES_TIMEZONE", "").strip()
    if tz_env:
        return tz_env

    # 2. Zettlab device mode: APP changes the OS timezone via timedatectl, so
    #    OS timezone must win over any stale non-committed config.yaml value.
    if _is_zettlab_device_mode():
        tz_os = _read_os_timezone()
        if tz_os:
            return tz_os

    # 3. config.yaml ``timezone`` key (generic Hermes behavior).
    tz_cfg = _read_config_timezone()
    if tz_cfg:
        return tz_cfg

    # 4. OS system timezone fallback.
    tz_os = _read_os_timezone()
    if tz_os:
        return tz_os

    return ""


def _read_etc_timezone() -> str:
    try:
        with open(ETC_TIMEZONE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return line.strip()
    except OSError:
        pass
    return ""


def _read_localtime_symlink() -> str:
    try:
        target = os.readlink(ETC_LOCALTIME)
    except OSError:
        return ""
    target = target.replace("\\", "/")
    for marker in _ZONEINFO_MARKERS:
        idx = target.find(marker)
        if idx >= 0:
            cand = target[idx + len(marker):]
            if cand.startswith("posix/"):
                cand = cand[len("posix/"):]
            return cand
    return ""


def _read_timedatectl() -> str:
    try:
        out = subprocess.run(
            ["timedatectl", "show", "-p", "Timezone", "--value"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _get_zoneinfo(name: str) -> Optional[ZoneInfo]:
    """Validate and return a ZoneInfo, or None if invalid.

    ZoneInfo only accepts real tzdata keys and rejects absolute paths / ``..``
    traversal, so a validated name is always a safe IANA string (the gate that
    keeps a bogus /etc/timezone or symlink target out of the code-exec TZ=).
    """
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (KeyError, Exception) as exc:
        logger.warning(
            "Invalid timezone '%s': %s. Falling back to server local time.",
            name, exc,
        )
        return None


def _source_fingerprint():
    """Composite (env, config.yaml, /etc/timezone, /etc/localtime) signature.

    lstat on /etc/localtime catches a retargeted symlink; a missing path
    contributes (False,) so appearance/disappearance flips the fingerprint.
    When file sources are unavailable, timedatectl's value is included so that
    non-Debian systems where /etc/localtime is a copied tzfile still refresh
    live instead of pinning the fallback until reset_cache().
    """
    def stat_fp(path, follow):
        try:
            st = os.stat(path) if follow else os.lstat(path)
        except OSError:
            return (False,)
        return (True, st.st_ino, st.st_mtime_ns, st.st_size)

    etc_timezone_fp = stat_fp(ETC_TIMEZONE, True)
    localtime_fp = stat_fp(ETC_LOCALTIME, False)
    timedatectl_fp = None
    if not etc_timezone_fp[0] and not localtime_fp[0]:
        timedatectl_fp = _read_timedatectl()

    return (
        os.getenv("HERMES_TIMEZONE", "").strip(),
        os.getenv("ZET_AGENT_ENABLED", "").strip().lower(),
        stat_fp(str(get_config_path()), True),
        etc_timezone_fp,
        localtime_fp,
        timedatectl_fp,
    )


def _refresh() -> None:
    """Re-resolve tz/name when the source fingerprint changed."""
    global _cached_tz, _cached_tz_name, _cached_fp
    fp = _source_fingerprint()
    if fp == _cached_fp:
        return
    name = _resolve_timezone_name()
    tz = _get_zoneinfo(name)
    _cached_tz = tz
    # Only expose a name that validated to a real zone — never a bare/invalid
    # string — so callers like the code-exec TZ= can use it directly.
    _cached_tz_name = name if tz is not None else None
    _cached_fp = fp


def get_timezone() -> Optional[ZoneInfo]:
    """Return the user's configured ZoneInfo, or None (meaning server-local).

    Re-resolved when the source fingerprint changes. Call ``reset_cache()`` to
    force re-resolution.
    """
    _refresh()
    return _cached_tz


def get_timezone_name() -> Optional[str]:
    """Return the validated IANA timezone name, or None when none resolves
    (or only the server-local fallback applies)."""
    _refresh()
    return _cached_tz_name


def now() -> datetime:
    """
    Return the current time as a timezone-aware datetime.

    If a valid timezone is configured, returns wall-clock time in that zone.
    Otherwise returns the server's local time (via ``astimezone()``).
    """
    tz = get_timezone()
    if tz is not None:
        return datetime.now(tz)
    # No timezone configured — use server-local (still tz-aware)
    return datetime.now().astimezone()
