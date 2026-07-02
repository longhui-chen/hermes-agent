"""Tests for subprocess HOME handling in profile mode.

Hermes state stays profile-scoped through HERMES_HOME. Host subprocesses should
keep the user's real HOME by default so external CLIs find existing credentials.
Containers still use the profile home for persistence, and users can explicitly
opt into profile HOME isolation on the host.

See: https://github.com/NousResearch/hermes-agent/issues/25114
See: https://github.com/NousResearch/hermes-agent/issues/36144
See: https://github.com/NousResearch/hermes-agent/issues/29015
"""

import os
import threading
from pathlib import Path

import hermes_constants



# ---------------------------------------------------------------------------
# get_subprocess_home()
# ---------------------------------------------------------------------------

class TestGetSubprocessHome:
    """Unit tests for hermes_constants.get_subprocess_home()."""

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    def _container_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    def test_returns_none_when_hermes_home_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_returns_none_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # No home/ subdirectory created
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_host_auto_keeps_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        """Host installs should not hide real ~/.ssh, ~/.gitconfig, ~/.azure, etc."""
        self._host_mode(monkeypatch)
        real_home = tmp_path / "real-home"
        hermes_home = real_home / ".hermes" / "profiles" / "coder"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_host_auto_keeps_root_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        """A present HOME (even /root) is kept untouched on hosts."""
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_host_auto_falls_back_to_profile_home_when_home_env_missing(self, tmp_path, monkeypatch):
        """systemd system services launch Hermes with no HOME at all (ZET-1938).

        With nothing to keep, auto must fall back to the profile home so
        ``~``-addressed credential stores stay reachable.
        """
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_host_auto_falls_back_to_profile_home_when_home_env_empty(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HOME", "")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_host_auto_env_dict_without_home_falls_back_to_profile_home(self, tmp_path, monkeypatch):
        """The explicit env-dict input path honors the same missing-HOME fallback."""
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home({"HERMES_HOME": str(hermes_home)}) == str(profile_home)

    def test_host_auto_missing_home_without_profile_home_returns_none(self, tmp_path, monkeypatch):
        """No profile home means no fallback — auto must not invent a HOME."""
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # No home/ subdirectory created
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_windows_host_auto_missing_home_keeps_home_unset(self, tmp_path, monkeypatch):
        """Windows hosts never carry HOME (only USERPROFILE), so the
        missing-HOME fallback must stay POSIX-only — otherwise every Windows
        install gets its subprocess HOME pinned to the profile home and
        MSYS/git-bash tools (git, ssh, gh) lose the real ~/.gitconfig, ~/.ssh.
        """
        self._host_mode(monkeypatch)
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("USERPROFILE", r"C:\Users\alice")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None

    def test_windows_host_auto_env_dict_without_home_keeps_home_unset(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.delenv("HOME", raising=False)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        from hermes_constants import get_subprocess_home
        env = {"HERMES_HOME": str(hermes_home), "USERPROFILE": r"C:\Users\alice"}
        assert get_subprocess_home(env) is None

    def test_real_mode_with_missing_home_still_returns_real_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        (hermes_home / "home").mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(real_home)

    def test_profile_mode_with_missing_home_returns_profile_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_container_auto_with_missing_home_uses_profile_home(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_container_auto_uses_profile_home_when_home_dir_exists(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_returns_profile_specific_path(self, tmp_path, monkeypatch):
        """Explicit profile mode keeps the old per-profile HOME behavior."""
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_dir.mkdir(parents=True)
        profile_home = profile_dir / "home"
        profile_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_real_mode_repairs_parent_home_already_pointing_at_profile(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home = profile_dir / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("TERMINAL_HOME_MODE", "real")
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))

        from hermes_constants import get_subprocess_home, get_real_home

        assert get_real_home() == str(real_home)
        assert get_subprocess_home() == str(real_home)

    def test_real_home_falls_back_to_os_account_when_home_is_profile(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        profile_dir = tmp_path / ".hermes" / "profiles" / "coder"
        profile_home = profile_dir / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        monkeypatch.setenv("HOME", str(profile_home))

        from hermes_constants import get_real_home

        assert get_real_home() != str(profile_home)

    def test_two_profiles_get_different_homes(self, tmp_path, monkeypatch):
        self._container_mode(monkeypatch)
        base = tmp_path / ".hermes" / "profiles"
        for name in ("alpha", "beta"):
            p = base / name
            p.mkdir(parents=True)
            (p / "home").mkdir()

        from hermes_constants import get_subprocess_home

        monkeypatch.setenv("HERMES_HOME", str(base / "alpha"))
        home_a = get_subprocess_home()

        monkeypatch.setenv("HERMES_HOME", str(base / "beta"))
        home_b = get_subprocess_home()

        assert home_a is not None
        assert home_b is not None
        assert home_a != home_b
        assert home_a.endswith("alpha/home")
        assert home_b.endswith("beta/home")

    def test_context_override_is_thread_local(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        profile = tmp_path / "profile"
        root.mkdir()
        profile.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(root))

        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        ready = threading.Event()
        release = threading.Event()
        seen: list[str] = []

        def read_from_other_thread():
            ready.set()
            release.wait(timeout=5)
            seen.append(str(get_hermes_home()))

        thread = threading.Thread(target=read_from_other_thread)
        thread.start()
        assert ready.wait(timeout=5)

        token = set_hermes_home_override(profile)
        try:
            assert get_hermes_home() == profile
            release.set()
            thread.join(timeout=5)
        finally:
            reset_hermes_home_override(token)
            release.set()

        assert seen == [str(root)]
        assert get_hermes_home() == root


class TestMissingHomeFallbackNestedChain:
    """The missing-HOME fallback must survive nested Hermes invocations.

    ``apply_subprocess_home_env()`` marks a fallback-injected HOME with
    ``HERMES_HOME_FALLBACK=<profile home path>``. Without the marker, a
    nested hermes level sees ``HOME == profile home`` and "repairs" it back
    to the real-HOME guess (``/root`` on the ZET-1938 box) — re-breaking
    every second hop of the process chain.
    """

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)

    def _profile_home(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        return hermes_home, profile_home

    def test_fallback_injection_sets_marker(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        monkeypatch.delenv("HOME", raising=False)
        hermes_home, profile_home = self._profile_home(tmp_path)

        from hermes_constants import apply_subprocess_home_env
        env = {"HERMES_HOME": str(hermes_home), "PATH": "/usr/bin"}
        apply_subprocess_home_env(env)

        assert env["HOME"] == str(profile_home)
        # The marker records the source profile home it injected, so a nested
        # level can tell same-profile (keep) from cross-profile (re-inject).
        assert env["HERMES_HOME_FALLBACK"] == str(profile_home)

    def test_nested_apply_does_not_flip_fallback_home_back(self, tmp_path, monkeypatch):
        """Layer 1: no-HOME parent gets the profile home. Layer 2: a nested
        hermes building its own child env must keep that HOME, not repair it
        to the real-HOME guess."""
        self._host_mode(monkeypatch)
        monkeypatch.delenv("HOME", raising=False)
        hermes_home, profile_home = self._profile_home(tmp_path)

        from hermes_constants import apply_subprocess_home_env
        env1 = {"HERMES_HOME": str(hermes_home), "PATH": "/usr/bin"}
        apply_subprocess_home_env(env1)
        assert env1["HOME"] == str(profile_home)

        env2 = dict(env1)  # nested hermes inherits layer-1's env
        apply_subprocess_home_env(env2)

        assert env2["HOME"] == str(profile_home)
        assert env2["HERMES_HOME_FALLBACK"] == str(profile_home)

    def test_repair_still_flips_unmarked_profile_home(self, tmp_path, monkeypatch):
        """A user who manually pinned HOME to the profile home (no marker)
        keeps the existing auto-mode repair behavior."""
        self._host_mode(monkeypatch)
        hermes_home, profile_home = self._profile_home(tmp_path)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(profile_home))
        monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(real_home)

    def test_real_mode_overrides_fallback_marker(self, tmp_path, monkeypatch):
        """Explicit ``terminal.home_mode=real`` is user policy — it wins over
        the fallback marker."""
        self._host_mode(monkeypatch)
        monkeypatch.delenv("HOME", raising=False)
        hermes_home, profile_home = self._profile_home(tmp_path)
        real_home = tmp_path / "real-home"
        real_home.mkdir()

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(profile_home),
            # Legacy "1" marker still recognized (backward compat).
            "HERMES_HOME_FALLBACK": "1",
            "HERMES_REAL_HOME": str(real_home),
            "TERMINAL_HOME_MODE": "real",
        }
        assert get_subprocess_home(env) == str(real_home)


class TestFallbackMarkerCrossProfile:
    """The fallback marker must carry the *source* profile home so an A→B
    HERMES_HOME switch does not leak A's credential dir into B's children.

    Scenario: a parent injected ``HOME=A/home`` via the missing-HOME
    fallback (marker records ``A/home``). A later hop switches HERMES_HOME
    to profile B but inherits the parent's ``HOME=A/home``. Because
    ``HOME`` is non-empty, the missing-HOME branch never fires; because
    ``HOME`` (A/home) != B's profile home, the same-profile keep never
    fires either — so the old code returned None and B's child read/wrote
    A's ``~/.lark-cli`` etc. The marker source lets us detect the mismatch
    and re-point HOME at B's own profile home instead.
    """

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)

    def _two_profiles(self, tmp_path):
        base = tmp_path / ".hermes" / "profiles"
        a = base / "alpha"
        b = base / "beta"
        (a / "home").mkdir(parents=True)
        (b / "home").mkdir(parents=True)
        return a, b

    def test_switch_a_to_b_reinjects_b_profile_home(self, tmp_path, monkeypatch):
        """A→B: child inherits HOME=A/home (marked source A/home) but its
        own HERMES_HOME is B — resolve to B/home, not A/home."""
        self._host_mode(monkeypatch)
        a, b = self._two_profiles(tmp_path)
        a_home = a / "home"
        b_home = b / "home"
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(b),          # this hop is profile B
            "HOME": str(a_home),            # inherited from profile-A parent
            "HERMES_HOME_FALLBACK": str(a_home),  # marker records A's home
        }
        assert get_subprocess_home(env) == str(b_home)

    def test_switch_a_to_b_apply_updates_marker(self, tmp_path, monkeypatch):
        """After re-injection the marker must record B's home, so the next
        hop treats it as same-profile (keep) rather than re-injecting again."""
        self._host_mode(monkeypatch)
        a, b = self._two_profiles(tmp_path)
        a_home = a / "home"
        b_home = b / "home"
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import apply_subprocess_home_env
        env = {
            "HERMES_HOME": str(b),
            "HOME": str(a_home),
            "HERMES_HOME_FALLBACK": str(a_home),
            "PATH": "/usr/bin",
        }
        apply_subprocess_home_env(env)
        assert env["HOME"] == str(b_home)
        assert env["HERMES_HOME_FALLBACK"] == str(b_home)

        # Next hop stays in B: same-profile keep, no further churn.
        env2 = dict(env)
        apply_subprocess_home_env(env2)
        assert env2["HOME"] == str(b_home)
        assert env2["HERMES_HOME_FALLBACK"] == str(b_home)

    def test_same_profile_two_hops_does_not_flip(self, tmp_path, monkeypatch):
        """Regression guard for the existing anti-flip behavior when the
        marker carries a path: A→A must keep A/home, never repair."""
        self._host_mode(monkeypatch)
        a, _ = self._two_profiles(tmp_path)
        a_home = a / "home"
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(a),
            "HOME": str(a_home),
            "HERMES_HOME_FALLBACK": str(a_home),
            "HERMES_REAL_HOME": str(real_home),
        }
        assert get_subprocess_home(env) is None  # keep A/home, no override

    def test_legacy_marker_cross_profile_reinjects_current_profile_home(self, tmp_path, monkeypatch):
        """A legacy ``"1"`` marker carries no source path. When the inherited
        HOME does not match this hop's profile home, treat it as a cross-
        profile fallback and re-inject the current profile home anyway."""
        self._host_mode(monkeypatch)
        a, b = self._two_profiles(tmp_path)
        a_home = a / "home"
        b_home = b / "home"
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(b),
            "HOME": str(a_home),
            "HERMES_HOME_FALLBACK": "1",  # legacy, no source info
        }
        assert get_subprocess_home(env) == str(b_home)

    def test_legacy_marker_same_profile_is_kept(self, tmp_path, monkeypatch):
        """A legacy ``"1"`` marker whose inherited HOME already equals this
        hop's profile home is same-profile — keep it, do not repair."""
        self._host_mode(monkeypatch)
        a, _ = self._two_profiles(tmp_path)
        a_home = a / "home"
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(a),
            "HOME": str(a_home),
            "HERMES_HOME_FALLBACK": "1",
            "HERMES_REAL_HOME": str(real_home),
        }
        assert get_subprocess_home(env) is None

    def test_cross_profile_without_target_profile_home_keeps_inherited(self, tmp_path, monkeypatch):
        """If the switched-to profile has no home/ dir, there is nothing to
        re-inject — keep the inherited HOME rather than inventing one."""
        self._host_mode(monkeypatch)
        a, b = self._two_profiles(tmp_path)
        a_home = a / "home"
        # Remove B's home/ so profile_home resolves to None for B.
        import shutil
        shutil.rmtree(b / "home")
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(b),
            "HOME": str(a_home),
            "HERMES_HOME_FALLBACK": str(a_home),
        }
        assert get_subprocess_home(env) is None  # keep A/home, nothing better


class TestStaleMarkerDoesNotHijackExplicitRealHome:
    """A stale fallback marker must not hijack an explicitly-set real HOME.

    The marker records the *source* profile-home path it injected. A nested
    hermes process always inherits that marker in ``os.environ``. When an
    intermediate layer explicitly resets HOME back to the user's real home
    (``export HOME=/home/user`` in a wrapper, ``sudo -E``, or a tool passing
    ``env_vars={"HOME": "/real/path"}`` through ``process_registry``) but does
    NOT clear the marker, the next auto-mode hop sees:

      * ``current_home`` = the explicit real home (≠ any profile home),
      * a non-empty (stale) marker,

    and — with the old "non-empty marker ⇒ fallback-injected" rule — wrongly
    concludes HOME was fallback-injected and cross-profile, hijacking HOME to
    this hop's profile home. That points the child's ``~/.ssh``,
    ``~/.gitconfig``, ``~/.lark-cli`` at the profile dir instead of the real
    user home, and pollutes ``HERMES_REAL_HOME`` too.

    The marker only vouches for a HOME it actually equals. When the marker
    value differs from the current HOME, the HOME was reset out from under the
    marker — treat it as user-pinned and leave it alone.
    """

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)

    def _profile_and_real(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        # A different profile home the stale marker points back at.
        old_profile = tmp_path / "old" / ".hermes" / "home"
        old_profile.mkdir(parents=True)
        return hermes_home, profile_home, real_home, old_profile

    def test_get_subprocess_home_keeps_explicit_real_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home, real_home, old_profile = self._profile_and_real(tmp_path)
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(real_home),                 # explicitly reset to real home
            "HERMES_HOME_FALLBACK": str(old_profile),  # stale marker, ≠ HOME
        }
        # Must NOT hijack HOME to this hop's profile home.
        assert get_subprocess_home(env) is None

    def test_apply_subprocess_home_env_keeps_explicit_real_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home, real_home, old_profile = self._profile_and_real(tmp_path)
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import apply_subprocess_home_env
        env = {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(real_home),
            "HERMES_HOME_FALLBACK": str(old_profile),
            "PATH": "/usr/bin",
        }
        apply_subprocess_home_env(env)
        assert env["HOME"] == str(real_home)
        assert env["HERMES_REAL_HOME"] == str(real_home)

    def test_make_run_env_keeps_explicit_real_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home, real_home, old_profile = self._profile_and_real(tmp_path)
        # os.environ carries the stale marker + real HOME, as a nested hermes
        # would inherit after a wrapper reset HOME.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("HERMES_HOME_FALLBACK", str(old_profile))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})
        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_sanitize_subprocess_env_keeps_explicit_real_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home, real_home, old_profile = self._profile_and_real(tmp_path)
        # process_registry passes env_vars={"HOME": real} as extra_env while
        # os.environ still carries the inherited stale marker.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_HOME_FALLBACK", str(old_profile))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)
        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_stale_marker_is_cleared_from_explicit_real_home_env(self, tmp_path, monkeypatch):
        """When HOME is a non-fallback external value, the stale marker must be
        dropped so it cannot mislead an even deeper hop into re-hijacking."""
        self._host_mode(monkeypatch)
        hermes_home, profile_home, real_home, old_profile = self._profile_and_real(tmp_path)
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import apply_subprocess_home_env
        env = {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(real_home),
            "HERMES_HOME_FALLBACK": str(old_profile),
            "PATH": "/usr/bin",
        }
        apply_subprocess_home_env(env)
        assert env["HOME"] == str(real_home)
        assert env.get("HERMES_HOME_FALLBACK") in (None, "")


class TestProfileHomeFromPlatformDefault:
    """cron lines run hermes with neither HOME nor HERMES_HOME set; the
    guard must locate the profile home through get_hermes_home()'s platform
    default instead of silently going dead (ZET-1938 cron variant)."""

    def _bare_env(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)

    def test_cron_shaped_env_falls_back_via_platform_default(self, tmp_path, monkeypatch):
        self._bare_env(monkeypatch)
        default_home = tmp_path / ".hermes"
        profile_home = default_home / "home"
        profile_home.mkdir(parents=True)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: default_home)

        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() == str(profile_home)

    def test_platform_default_without_home_dir_returns_none(self, tmp_path, monkeypatch):
        self._bare_env(monkeypatch)
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        # No home/ subdirectory created
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: default_home)

        from hermes_constants import get_subprocess_home
        assert get_subprocess_home() is None


class TestEnvDictHomeShadowing:
    """An explicit env-dict ``HOME`` key must not be shadowed by the host
    process env: ``{"HOME": ""}`` describes a child launching with a blank
    HOME even when the host itself has one."""

    def _host_mode(self, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.delenv("HERMES_HOME_FALLBACK", raising=False)

    def _profile_home(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        profile_home = hermes_home / "home"
        profile_home.mkdir(parents=True)
        return hermes_home, profile_home

    def test_host_home_present_env_dict_without_key_keeps_host_home(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, _ = self._profile_home(tmp_path)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))

        from hermes_constants import get_subprocess_home
        assert get_subprocess_home({"HERMES_HOME": str(hermes_home)}) is None

    def test_host_home_present_env_dict_empty_home_triggers_fallback(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home = self._profile_home(tmp_path)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))  # host HOME must not mask env's ""

        from hermes_constants import get_subprocess_home
        env = {"HERMES_HOME": str(hermes_home), "HOME": ""}
        assert get_subprocess_home(env) == str(profile_home)

    def test_no_host_home_env_dict_empty_home_triggers_fallback(self, tmp_path, monkeypatch):
        self._host_mode(monkeypatch)
        hermes_home, profile_home = self._profile_home(tmp_path)
        monkeypatch.delenv("HOME", raising=False)

        from hermes_constants import get_subprocess_home
        env = {"HERMES_HOME": str(hermes_home), "HOME": ""}
        assert get_subprocess_home(env) == str(profile_home)

    def test_real_mode_env_dict_empty_home_still_injects_real_home(self, tmp_path, monkeypatch):
        """real mode: a child described with HOME="" needs the injection even
        when the host HOME already equals the real home (old code compared
        against the host value and skipped it)."""
        self._host_mode(monkeypatch)
        hermes_home, _ = self._profile_home(tmp_path)
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setenv("HOME", str(real_home))

        from hermes_constants import get_subprocess_home
        env = {
            "HERMES_HOME": str(hermes_home),
            "HOME": "",
            "HERMES_REAL_HOME": str(real_home),
            "TERMINAL_HOME_MODE": "real",
        }
        assert get_subprocess_home(env) == str(real_home)


# ---------------------------------------------------------------------------
# _make_run_env() injection
# ---------------------------------------------------------------------------

class TestMakeRunEnvHomeInjection:
    """Verify _make_run_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", str(real_home))
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        # No home/ subdirectory
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/root"

    def test_no_injection_when_hermes_home_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from tools.environments.local import _make_run_env
        result = _make_run_env({})

        assert result["HOME"] == "/home/user"

    def test_context_override_bridges_to_subprocess_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        root = tmp_path / "root"
        profile = tmp_path / "profile"
        root.mkdir()
        profile.mkdir()
        (profile / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(root))
        monkeypatch.setenv("HOME", "/root")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.environments.local import _make_run_env

        token = set_hermes_home_override(profile)
        try:
            result = _make_run_env({})
        finally:
            reset_hermes_home_override(token)

        assert result["HERMES_HOME"] == str(profile)
        assert result["HOME"] == str(profile / "home")


# ---------------------------------------------------------------------------
# _sanitize_subprocess_env() injection
# ---------------------------------------------------------------------------

class TestSanitizeSubprocessEnvHomeInjection:
    """Verify _sanitize_subprocess_env() applies the subprocess HOME policy."""

    def test_host_auto_preserves_real_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(real_home)
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_profile_mode_injects_profile_home_when_profile_home_exists(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        real_home = tmp_path / "real-home"
        real_home.mkdir()
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": str(real_home), "PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(hermes_home / "home")
        assert result["HERMES_REAL_HOME"] == str(real_home)

    def test_no_injection_when_home_dir_missing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"HOME": "/root", "PATH": "/usr/bin"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == "/root"

    def test_missing_home_injects_profile_home(self, tmp_path, monkeypatch):
        """systemd-shaped parent env (no HOME anywhere) gets the profile home."""
        hermes_home = tmp_path / "hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"PATH": "/usr/bin", "USER": "root"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert result["HOME"] == str(hermes_home / "home")

    def test_windows_missing_home_does_not_inject_profile_home(self, tmp_path, monkeypatch):
        """Windows cmd/PowerShell/service env (USERPROFILE, no HOME) must not
        get HOME pinned — MSYS tools would resolve ~ into the profile home."""
        hermes_home = tmp_path / "hermes"
        (hermes_home / "home").mkdir(parents=True)
        monkeypatch.setattr(hermes_constants, "is_container", lambda: False)
        monkeypatch.setattr(hermes_constants.sys, "platform", "win32")
        monkeypatch.delenv("HOME", raising=False)
        monkeypatch.delenv("TERMINAL_HOME_MODE", raising=False)
        monkeypatch.delenv("HERMES_REAL_HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        base_env = {"PATH": "/usr/bin", "USERPROFILE": r"C:\Users\alice"}
        from tools.environments.local import _sanitize_subprocess_env
        result = _sanitize_subprocess_env(base_env)

        assert "HOME" not in result

    def test_context_override_bridges_to_background_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hermes_constants, "is_container", lambda: True)
        root = tmp_path / "root"
        profile = tmp_path / "profile"
        root.mkdir()
        profile.mkdir()
        (profile / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(root))

        base_env = {"HOME": "/root", "PATH": "/usr/bin"}
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.environments.local import _sanitize_subprocess_env

        token = set_hermes_home_override(profile)
        try:
            result = _sanitize_subprocess_env(base_env)
        finally:
            reset_hermes_home_override(token)

        assert result["HERMES_HOME"] == str(profile)
        assert result["HOME"] == str(profile / "home")


# ---------------------------------------------------------------------------
# Profile bootstrap
# ---------------------------------------------------------------------------

class TestProfileBootstrap:
    """Verify new profiles get a home/ subdirectory."""

    def test_profile_dirs_includes_home(self):
        from hermes_cli.profiles import _PROFILE_DIRS
        assert "home" in _PROFILE_DIRS

    def test_create_profile_bootstraps_home_dir(self, tmp_path, monkeypatch):
        """create_profile() should create home/ inside the profile dir."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(home))

        from hermes_cli.profiles import create_profile
        profile_dir = create_profile("testbot", no_alias=True)
        assert (profile_dir / "home").is_dir()


# ---------------------------------------------------------------------------
# Python process HOME unchanged
# ---------------------------------------------------------------------------

class TestPythonProcessUnchanged:
    """Confirm the Python process's own HOME is never modified."""

    def test_path_home_unchanged_after_subprocess_home_resolved(
        self, tmp_path, monkeypatch
    ):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "home").mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        original_home = os.environ.get("HOME")
        original_path_home = str(Path.home())

        from hermes_constants import get_subprocess_home
        sub_home = get_subprocess_home()

        # Resolving subprocess HOME must not mutate the Python process env.
        assert sub_home in (None, str(hermes_home / "home"), original_home)
        assert os.environ.get("HOME") == original_home
        assert str(Path.home()) == original_path_home
