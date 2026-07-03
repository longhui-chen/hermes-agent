"""Shared constants for Hermes Agent.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
import sys
import sysconfig
from contextvars import ContextVar, Token
from pathlib import Path


_profile_fallback_warned: bool = False
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar[str | object] = ContextVar(
    "_HERMES_HOME_OVERRIDE", default=_UNSET
)


def set_hermes_home_override(path: str | Path | None) -> Token:
    """Set a context-local Hermes home override and return its reset token.

    This is for in-process, per-task scoping.  It deliberately does not mutate
    ``os.environ`` because that is shared by every thread in the process.
    """
    value: str | object = _UNSET if path is None else str(path)
    return _HERMES_HOME_OVERRIDE.set(value)


def reset_hermes_home_override(token: Token) -> None:
    """Restore the previous context-local Hermes home override."""
    _HERMES_HOME_OVERRIDE.reset(token)


def get_hermes_home_override() -> str | None:
    """Return the active context-local Hermes home override, if any."""
    override = _HERMES_HOME_OVERRIDE.get()
    if override is _UNSET or not override:
        return None
    return str(override)


def _get_platform_default_hermes_home() -> Path:
    """Return the platform-native default Hermes home path."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: platform-native path).

    Reads HERMES_HOME env var, falls back to the platform-native default.
    This is the single source of truth — all other copies should import this.

    When ``HERMES_HOME`` is unset but an ``active_profile`` file indicates
    a non-default profile is active, logs a loud one-shot warning to
    ``errors.log`` so cross-profile data corruption is diagnosable instead
    of silent.  Behavior is unchanged otherwise — we still return
    the platform-native default — because raising here would brick 30+ module-level
    callers that import this at load time.  Subprocess spawners are
    expected to propagate ``HERMES_HOME`` explicitly (see the systemd
    template in ``hermes_cli/gateway.py`` and the kanban dispatcher in
    ``hermes_cli/kanban_db.py``).  See https://github.com/NousResearch/hermes-agent/issues/18594.
    """
    override = get_hermes_home_override()
    if override:
        return Path(override)

    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val)

    # Guard: if a non-default profile is sticky-active, warn once that
    # the fallback to the default profile is almost certainly wrong.
    global _profile_fallback_warned
    if not _profile_fallback_warned:
        try:
            fallback_home = _get_platform_default_hermes_home()
            active_path = fallback_home / "active_profile"
            active = active_path.read_text().strip() if active_path.exists() else ""
        except (UnicodeDecodeError, OSError):
            active = ""
        if active and active != "default":
            _profile_fallback_warned = True
            # Write directly to stderr.  We intentionally do NOT route this
            # through ``logging`` because (a) this function is called at
            # module-import time from 30+ sites, often before logging is
            # configured, and (b) root-logger propagation would double-emit
            # on consoles where a StreamHandler is already attached.
            msg = (
                f"[HERMES_HOME fallback] HERMES_HOME is unset but active "
                f"profile is {active!r}. Falling back to {fallback_home}, which "
                f"is the DEFAULT profile — not {active!r}. Any data this "
                f"process writes will land in the wrong profile. The "
                f"subprocess spawner should pass HERMES_HOME explicitly "
                f"(see issue #18594)."
            )
            try:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            except Exception:
                pass

    return _get_platform_default_hermes_home()


def get_default_hermes_root() -> Path:
    """Return the root Hermes directory for profile-level operations.

    In standard deployments this is the platform-native Hermes home
    (``~/.hermes`` on POSIX, ``%LOCALAPPDATA%\\hermes`` on native Windows).

    In Docker or custom deployments where ``HERMES_HOME`` points outside
    ``~/.hermes`` (e.g. ``/opt/data``), returns ``HERMES_HOME`` directly
    — that IS the root.

    In profile mode where ``HERMES_HOME`` is ``<root>/profiles/<name>``,
    returns ``<root>`` so that ``profile list`` can see all profiles.
    Works both for standard (``~/.hermes/profiles/coder``) and Docker
    (``/opt/data/profiles/coder``) layouts.

    Import-safe — no dependencies beyond stdlib.
    """
    native_home = _get_platform_default_hermes_home()
    env_home = os.environ.get("HERMES_HOME", "")
    if not env_home:
        return native_home
    env_path = Path(env_home)
    try:
        env_path.resolve().relative_to(native_home.resolve())
        # HERMES_HOME is under ~/.hermes (normal or profile mode)
        return native_home
    except ValueError:
        pass

    # Docker / custom deployment.
    # Check if this is a profile path: <root>/profiles/<name>
    # If the immediate parent dir is named "profiles", the root is
    # the grandparent — this covers Docker profiles correctly.
    if env_path.parent.name == "profiles":
        return env_path.parent.parent

    # Not a profile path — HERMES_HOME itself is the root
    return env_path


def _get_packaged_data_dir(name: str) -> Path | None:
    """Return an installed data-files directory if one exists.

    Used to discover bundled skills/optional-skills when Hermes is installed
    from a wheel that emitted them via setuptools data_files.
    """
    candidates = []
    for scheme in ("data", "purelib", "platlib"):
        raw = sysconfig.get_path(scheme)
        if raw:
            candidates.append(Path(raw) / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``HERMES_OPTIONAL_SKILLS``.
    """
    override = os.getenv("HERMES_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "optional-skills"


def get_optional_mcps_dir(default: Path | None = None) -> Path:
    """Return the optional-mcps directory, honoring package-manager wrappers.

    Mirrors :func:`get_optional_skills_dir` for the MCP catalog (Nous-approved
    Model Context Protocol servers shipped with the repo but disabled by
    default). Packaged installs may ship ``optional-mcps`` outside the Python
    package tree and expose it via ``HERMES_OPTIONAL_MCPS``.
    """
    override = os.getenv("HERMES_OPTIONAL_MCPS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("optional-mcps")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "optional-mcps"


def get_bundled_skills_dir(default: Path | None = None) -> Path:
    """Return the bundled skills directory for source and packaged installs.

    Resolution order:
        1. ``HERMES_BUNDLED_SKILLS`` env var (Nix wrapper / explicit override)
        2. Wheel-installed ``<sysconfig data>/skills`` (pip install path)
        3. Caller-supplied ``default`` (typically the source-checkout path)
        4. ``<HERMES_HOME>/skills`` last-resort
    """
    override = os.getenv("HERMES_BUNDLED_SKILLS", "").strip()
    if override:
        return Path(override)
    packaged = _get_packaged_data_dir("skills")
    if packaged is not None:
        return packaged
    if default is not None:
        return default
    return get_hermes_home() / "skills"


def get_hermes_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a Hermes subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to HERMES_HOME (e.g. ``"cache/images"``).
        old_name: Legacy path relative to HERMES_HOME (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_hermes_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def display_hermes_home() -> str:
    """Return a user-friendly display string for the current HERMES_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.hermes``
        profile:  ``~/.hermes/profiles/coder``
        custom:   ``/opt/hermes-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.hermes``.  For code that needs a real ``Path``, use
    :func:`get_hermes_home` instead.
    """
    home = get_hermes_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


def secure_parent_dir(path: Path) -> None:
    """Chmod ``0o700`` on the parent directory of *path*, but only if safe.

    Refuses to chmod ``/`` or any top-level directory (resolved parent with
    fewer than 3 parts, i.e. ``/`` or any direct child like ``/usr``) to
    prevent catastrophic host bricking when ``HERMES_HOME`` or other path
    env vars resolve to an unexpected location.

    See https://github.com/NousResearch/hermes-agent/issues/25821.
    """
    parent = path.parent.resolve()
    # Refuse root and its direct children (/usr, /home, /var, /tmp, …).
    if parent == Path("/") or len(parent.parts) < 3:
        return
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass


def _norm_home_path(path: str | None) -> str:
    """Return a comparable absolute path string, or ``""`` for empty input."""
    raw = (path or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(raw)))
    except Exception:
        return os.path.normcase(raw)


def _same_home_path(a: str | None, b: str | None) -> bool:
    """Return True when *a* and *b* name the same home dir, symlinks resolved.

    The path-carrying fallback marker and the inherited HOME can reach the same
    profile home through different spellings when the Hermes tree is symlinked
    (``/data/hermes`` -> ``/mnt/vol/hermes``): the marker is written via one
    path, HOME read via the canonical one. A plain ``_norm_home_path`` (abspath,
    no symlink resolution) then sees them as different and the marker is wrongly
    treated as stale. Canonicalize both ends first; fall back to the textual
    compare when they already match textually (covers non-existent paths, where
    ``realpath`` is a no-op anyway).
    """
    if _norm_home_path(a) == _norm_home_path(b):
        return True
    ra, rb = a or "", b or ""
    if not ra or not rb:
        return False
    try:
        return os.path.normcase(os.path.realpath(ra)) == os.path.normcase(os.path.realpath(rb))
    except OSError:
        return False


def _profile_home_path(env: dict[str, str] | None = None) -> str | None:
    """Return ``{HERMES_HOME}/home`` when the profile-home directory exists."""
    hermes_home = get_hermes_home_override() or (env or {}).get("HERMES_HOME") or os.getenv("HERMES_HOME")
    if not hermes_home:
        # cron/systemd lines often launch hermes with neither HOME nor
        # HERMES_HOME. All state reads still resolve through
        # get_hermes_home()'s platform default, so the HOME policy must see
        # the same directory — otherwise the missing-HOME fallback in
        # get_subprocess_home() silently never fires for exactly the
        # environments it exists for (ZET-1938 cron variant).
        try:
            hermes_home = str(get_hermes_home())
        except Exception:
            # Path.home() can fail on POSIX with no HOME and no passwd
            # entry; no resolvable Hermes home means no profile home.
            return None
    profile_home = os.path.join(hermes_home, "home")
    if os.path.isdir(profile_home):
        return profile_home
    return None


def _is_profile_home(candidate: str | None, profile_home: str | None) -> bool:
    # Symlink-aware (via _same_home_path), matching the path-carrying marker
    # comparison in _home_fallback_marked. Without this the two diverge on a
    # symlinked profile tree: a reverse-symlink HOME (a symlinked spelling of
    # {HERMES_HOME}/home) would not be recognized here as the profile home, so
    # get_real_home() would hand the profile home back as the OS real home and
    # a home_mode=real child would get HOME pinned to the profile dir.
    return bool(candidate and profile_home and _same_home_path(candidate, profile_home))


_HOME_FALLBACK_MARKER = "HERMES_HOME_FALLBACK"


def _child_env_home(env: dict[str, str]) -> str:
    """Return the HOME the child process will actually see.

    An explicit env-dict entry wins even when empty: a caller passing
    ``{"HOME": ""}`` is describing a child that launches with a blank HOME,
    and the host process's own HOME must not mask that.
    """
    if "HOME" in env:
        return str(env["HOME"] or "").strip()
    return str(os.getenv("HOME") or "").strip()


def _home_fallback_marker_raw(env: dict[str, str]) -> str:
    """Return the raw ``HERMES_HOME_FALLBACK`` value the child will inherit.

    An explicit env-dict entry wins over ``os.environ`` (mirrors
    :func:`_child_env_home`): a caller building a child env is describing that
    child's marker, not the host process's.
    """
    raw = env[_HOME_FALLBACK_MARKER] if _HOME_FALLBACK_MARKER in env else os.getenv(_HOME_FALLBACK_MARKER, "")
    return str(raw or "").strip()


def _looks_like_profile_home(path: str | None) -> bool:
    """Return True when *path* has the ``{HERMES_HOME}/home`` shape.

    The missing-HOME fallback always injects ``os.path.join(hermes_home,
    "home")``, so a fallback-injected HOME's final path component is literally
    ``home``. A real OS-account home almost never is (it is the username:
    ``alice``, ``root``, …). This is the only signal available for a legacy
    bare-``"1"`` marker, which — unlike the modern marker — does not record the
    source path it injected.

    A real, top-level OS home whose basename happens to be ``home`` — ``/home``,
    ``/srv/home``, ``/mnt/home`` — must NOT match: a genuine ``{HERMES_HOME}/home``
    profile home lives under a real Hermes tree, so its parent is never the
    filesystem root or a direct child of it. Reuse the same shallow-parent
    threshold :func:`secure_parent_dir` uses (parent with fewer than 3 parts),
    so a legacy ``"1"`` marker riding on ``HOME=/home`` is not mistaken for a
    fallback injection and does not hijack HOME to the profile dir.
    """
    norm = _norm_home_path(path)
    if not norm or os.path.basename(norm.rstrip("/\\")) != "home":
        return False
    parent = os.path.dirname(norm.rstrip("/\\"))
    # ``/home`` -> parent ``/`` (1 part); ``/srv/home`` -> ``/srv`` (2 parts).
    # A real profile home's parent (``.../profiles/coder``) has more.
    return len(Path(parent).parts) >= 3


def _home_fallback_marked(env: dict[str, str]) -> bool:
    """Return True when HOME in *env* was injected by the missing-HOME fallback.

    The marker records the source profile-home *path* it injected (older
    releases wrote a bare ``"1"``). It only vouches for a HOME it still equals:
    if an intermediate layer reset HOME to some other value (a wrapper's
    ``export HOME=/home/user``, ``sudo -E``, or a tool passing
    ``env_vars={"HOME": …}``) without clearing the inherited marker, the marker
    is stale and must NOT be trusted — otherwise the auto-mode cross-profile
    branch would hijack that explicit HOME back to this hop's profile home,
    pointing the child's ``~``-addressed credential stores at the wrong dir.

    * Path-carrying marker: fallback-injected iff ``marker == current HOME``.
    * Legacy ``"1"`` marker (no source path): honored when the inherited HOME
      is *exactly* this hop's ``{HERMES_HOME}/home`` (same-profile, robust even
      for a shallow single-segment HERMES_HOME like ``/tmp`` → ``/tmp/home``),
      OR — for a cross-profile A→B hop where HOME is profile A's home, unknown
      to this hop — has the deep ``{HERMES_HOME}/home`` shape. A HOME reset to a
      real user home (basename ``alice``/``root``/…, or a top-level ``/home``)
      is neither, so it is treated as user-pinned, closing the hijack.
    """
    raw = _home_fallback_marker_raw(env)
    if not raw:
        return False
    current_home = _child_env_home(env)
    if raw == "1":
        # Legacy marker without source info. Same-profile: exact match against
        # this hop's profile home (covers shallow HERMES_HOME the shape guard's
        # parts>=3 threshold would reject). Cross-profile A→B: fall back to the
        # deep {HERMES_HOME}/home shape, since A's home is unknown here.
        if _is_profile_home(current_home, _profile_home_path(env)):
            return True
        return _looks_like_profile_home(current_home)
    return _same_home_path(raw, current_home)


def _iter_real_home_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Return likely OS-user home candidates in trust order."""
    env = env or {}
    candidates: list[str] = []
    explicit = str(env.get("HERMES_REAL_HOME") or os.getenv("HERMES_REAL_HOME", "")).strip()
    if explicit:
        candidates.append(explicit)
    home = str(env.get("HOME") or os.getenv("HOME", "")).strip()
    if home:
        candidates.append(home)
    try:
        import pwd

        pw_home = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX-only module inside try/except
        if pw_home:
            candidates.append(pw_home)
    except Exception:
        pass
    userprofile = str(env.get("USERPROFILE") or os.getenv("USERPROFILE", "")).strip()
    if userprofile:
        candidates.append(userprofile)
    drive = str(env.get("HOMEDRIVE") or os.getenv("HOMEDRIVE", "")).strip()
    path = str(env.get("HOMEPATH") or os.getenv("HOMEPATH", "")).strip()
    if drive and path:
        candidates.append(f"{drive}{path}" if path.startswith(("\\", "/")) else os.path.join(drive, path))
    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        candidates.append(expanded)
    return candidates


def get_real_home(env: dict[str, str] | None = None) -> str:
    """Return the OS user's real home directory, avoiding Hermes profile HOME.

    ``HERMES_HOME`` scopes Hermes state. ``HOME`` is reserved for the OS/user
    account and the many external CLIs that store credentials under ``~``.
    If a parent process is already running with ``HOME={HERMES_HOME}/home``,
    this helper repairs back to the account home when possible.
    """
    profile_home = _profile_home_path(env)
    seen: set[str] = set()
    for candidate in _iter_real_home_candidates(env):
        key = _norm_home_path(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if not _is_profile_home(candidate, profile_home):
            return candidate
    return "/tmp"


def _resolve_subprocess_home(env: dict[str, str] | None = None) -> tuple[str | None, bool]:
    """Resolve the subprocess HOME override for :func:`get_subprocess_home`.

    Returns ``(home, from_missing_home_fallback)``; the flag is True only
    when *home* came from the POSIX missing-HOME fallback, so
    :func:`apply_subprocess_home_env` can mark the injection for descendants.
    """
    env = env or {}
    profile_home = _profile_home_path(env)
    mode = str(env.get("TERMINAL_HOME_MODE") or os.getenv("TERMINAL_HOME_MODE", "auto")).strip().lower() or "auto"
    if mode in {"isolated", "profile_home", "profile-home"}:
        mode = "profile"
    if mode in {"host", "user", "real_home", "real-home"}:
        mode = "real"

    if mode == "profile":
        return profile_home, False

    real_home = get_real_home(env)
    current_home = _child_env_home(env)
    if mode == "real":
        return (real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None), False

    if profile_home and is_container():
        return profile_home, False
    if sys.platform != "win32" and not current_home and profile_home:
        # "Keep the real user HOME" needs a HOME to keep. systemd system
        # services (and cron) start Hermes with no HOME anywhere in the
        # process chain; without an override, ``~``-addressed credential
        # stores (e.g. lark-cli's ~/.lark-cli, ~/.local/share/lark-cli)
        # resolve nowhere. Fall back to the profile home — the pre-v0.17
        # directory-gated behavior — so those stores stay reachable even
        # when the terminal.home_mode pin was never delivered (ZET-1938).
        # Deliberately no real-HOME fallback here: inventing a HOME the
        # parent never had is a separate policy decision. POSIX-only:
        # Windows hosts never carry HOME (only USERPROFILE), so a missing
        # HOME there is the normal state, not the systemd/cron failure —
        # pinning it would redirect MSYS/git-bash tools (git, ssh, gh)
        # away from the real ~/.gitconfig and ~/.ssh on every install.
        return profile_home, True
    if _home_fallback_marked(env) and not _is_profile_home(current_home, profile_home):
        # HOME was fallback-injected by an ancestor (the marker vouches it is
        # not user-pinned), but this hop's profile home differs from the
        # inherited HOME — i.e. HERMES_HOME was switched A→B between hops
        # (set_hermes_home_override / an env-dict swap) while HOME kept
        # pointing at profile A's home. Left alone, B's children would
        # read/write A's ~-addressed credential stores (cross-profile leak).
        # Re-point HOME at B's own profile home when it exists. No target
        # profile home for B means there is nothing better to inject — keep
        # the inherited HOME rather than inventing one.
        if profile_home:
            return profile_home, True
        return None, False
    if _is_profile_home(current_home, profile_home):
        if _home_fallback_marked(env):
            # This profile HOME was injected by the fallback above, one
            # process level up. "Repairing" it would flip descendants back
            # to the real-HOME guess the fallback exists to avoid,
            # re-breaking every second hop of a nested hermes chain.
            return None, False
        return (real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None), False
    return None, False


def get_subprocess_home(env: dict[str, str] | None = None) -> str | None:
    """Return a subprocess ``HOME`` override, if one should be applied.

    Policy is controlled by ``terminal.home_mode`` (bridged to
    ``TERMINAL_HOME_MODE``):

    * ``auto`` (default): host installs keep the real user HOME; containers use
      ``{HERMES_HOME}/home`` for persistent state. If a host parent already has
      HOME pointed at the profile home, repair subprocesses back to real HOME —
      unless ``HERMES_HOME_FALLBACK`` marks it as fallback-injected. That marker
      records the *source* profile home, so a hop that switched HERMES_HOME
      A→B while inheriting profile-A's HOME re-points HOME at B's own profile
      home instead of leaking A's credential dir. POSIX hosts launched with no
      HOME at all (systemd system services, cron) fall back to the profile home
      when it exists; Windows hosts (which never set HOME) are left untouched.
    * ``real``: always prefer the real OS-user HOME.
    * ``profile``: use ``{HERMES_HOME}/home`` when it exists, preserving the
      older strict per-profile tool-config isolation.
    """
    return _resolve_subprocess_home(env)[0]


def apply_subprocess_home_env(env: dict[str, str]) -> None:
    """Apply Hermes' subprocess HOME contract to *env* in-place."""
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
    home, from_fallback = _resolve_subprocess_home(env)
    if home:
        env["HOME"] = home
        if from_fallback:
            # Mark the injection so nested hermes levels can tell "the
            # fallback put HOME here" apart from "the user pinned HOME
            # here" — only the latter is repaired back to the real HOME.
            # Record the injected profile-home path (not a bare "1"). The
            # marker's presence tells a nested hop this HOME is fallback-
            # injected; that hop then compares the inherited HOME against its
            # own profile home to decide keep (same profile) vs. re-inject
            # (cross profile), closing the cross-profile credential leak.
            # Storing the path (rather than "1") also keeps the marker
            # self-describing for debugging.
            env[_HOME_FALLBACK_MARKER] = home
            return
    # No fallback injection this hop. If a marker is riding along but no longer
    # vouches for the HOME the child will actually see (an intermediate layer
    # reset HOME out from under it), it is stale — drop it so a deeper hop is
    # not misled into re-hijacking the now-explicit HOME. The same-profile keep
    # path leaves the marker in place because there it still matches HOME.
    if _HOME_FALLBACK_MARKER in env and not _home_fallback_marked(env):
        del env[_HOME_FALLBACK_MARKER]


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid levels: "none", "minimal", "low", "medium", "high", "xhigh".
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def is_termux() -> bool:
    """Return True when running inside a Termux (Android) environment.

    Checks ``TERMUX_VERSION`` (set by Termux) or the Termux-specific
    ``PREFIX`` path.  Import-safe — no heavy deps.
    """
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """Return True when running inside WSL (Windows Subsystem for Linux).

    Checks ``/proc/version`` for the ``microsoft`` marker that both WSL1
    and WSL2 inject.  Result is cached for the process lifetime.
    Import-safe — no heavy deps.
    """
    global _wsl_detected
    if _wsl_detected is not None:
        return _wsl_detected
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            _wsl_detected = "microsoft" in f.read().lower()
    except Exception:
        _wsl_detected = False
    return _wsl_detected


_container_detected: bool | None = None


def is_container() -> bool:
    """Return True when running inside a container.

    Recognizes Docker (``/.dockerenv``), Podman (``/run/.containerenv``),
    and — via ``/proc/1/cgroup`` — the docker/podman/lxc cgroup-v1 markers.

    cgroup v2 collapses ``/proc/1/cgroup`` to a single ``0::/`` line with no
    runtime marker, so containerd/CRI-O runtimes (the common case on
    Kubernetes/k3s) were previously missed. To cover those, also check:
      * ``KUBERNETES_SERVICE_HOST`` env var — set in every Kubernetes pod.
      * ``kubepods`` / ``containerd`` / ``crio`` markers in ``/proc/1/cgroup``.
      * the same markers in ``/proc/self/mountinfo`` (cgroup-v2 fallback).

    Result is cached for the process lifetime.  Import-safe — no heavy deps.

    See: NousResearch/hermes-agent#47111
    """
    global _container_detected
    if _container_detected is not None:
        return _container_detected
    if os.path.exists("/.dockerenv"):
        _container_detected = True
        return True
    if os.path.exists("/run/.containerenv"):
        _container_detected = True
        return True
    # Kubernetes always injects this into pod containers; absent on hosts.
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        _container_detected = True
        return True
    _CGROUP_MARKERS = ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio")
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cgroup = f.read()
            if any(marker in cgroup for marker in _CGROUP_MARKERS):
                _container_detected = True
                return True
    except OSError:
        pass
    # cgroup v2: /proc/1/cgroup is just "0::/" with no marker. The container
    # runtime still shows up in the mount table (overlay rootfs, runtime mount
    # paths), so scan mountinfo as a last resort.
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            mountinfo = f.read()
            if any(marker in mountinfo for marker in ("kubepods", "containerd", "crio")):
                _container_detected = True
                return True
    except OSError:
        pass
    _container_detected = False
    return False


# ─── Well-Known Paths ─────────────────────────────────────────────────────────


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under HERMES_HOME.

    Replaces the ``get_hermes_home() / "config.yaml"`` pattern repeated
    in 7+ files (skill_utils.py, hermes_logging.py, hermes_time.py, etc.).
    """
    return get_hermes_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under HERMES_HOME."""
    return get_hermes_home() / "skills"



def get_env_path() -> Path:
    """Return the path to the ``.env`` file under HERMES_HOME."""
    return get_hermes_home() / ".env"


# ─── Network Preferences ─────────────────────────────────────────────────────


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 connections.

    On servers with broken or unreachable IPv6, Python tries AAAA records
    first and hangs for the full TCP timeout before falling back to IPv4.
    This affects httpx, requests, urllib, the OpenAI SDK — everything that
    uses ``socket.getaddrinfo``.

    When *force* is True, patches ``getaddrinfo`` so that calls with
    ``family=AF_UNSPEC`` (the default) resolve as ``AF_INET`` instead,
    skipping IPv6 entirely.  If no A record exists, falls back to the
    original unfiltered resolution so pure-IPv6 hosts still work.

    Safe to call multiple times — only patches once.
    Set ``network.force_ipv4: true`` in ``config.yaml`` to enable.
    """
    if not force:
        return

    import socket

    # Guard against double-patching
    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return

    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(
                    host, port, socket.AF_INET, type, proto, flags
                )
            except socket.gaierror:
                # No A record — fall back to full resolution (pure-IPv6 hosts)
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)

    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


# ─── Streaming Response Constants ────────────────────────────────────────────

# Response ID for partial stream stubs used during error recovery
PARTIAL_STREAM_STUB_ID = "partial-stream-stub"

FINISH_REASON_LENGTH = "length"


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
