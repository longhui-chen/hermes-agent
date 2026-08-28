"""Guards for ``get_external_skills_dirs`` mtime-based memo.

``get_external_skills_dirs()`` is called once per skill during banner
construction and tool registration — on a typical install that's 120+
calls.  Without caching, each call re-reads + YAML-parses the full
config.yaml (~85ms each, 10+ seconds total).  This test pins the
behavior: first call parses, subsequent calls return cached result,
cache invalidates when config.yaml's mtime changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import skill_utils
from agent.skill_utils import (
    _external_dirs_cache_clear,
    get_external_skills_dirs,
)


@pytest.fixture
def hermes_home_with_config(tmp_path, monkeypatch):
    """Isolated ``~/.hermes/`` with a config.yaml referencing one external dir."""
    home = tmp_path / ".hermes"
    home.mkdir()
    external = tmp_path / "external_skills"
    external.mkdir()

    config = home / "config.yaml"
    config.write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {external}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()
    yield home, external, config
    _external_dirs_cache_clear()






def test_cache_invalidates_on_mtime_change(hermes_home_with_config):
    """A config.yaml edit invalidates the cache on the next call."""
    _home, external, config = hermes_home_with_config
    other = external.parent / "other_skills"
    other.mkdir()

    # Prime cache with original contents.
    first = get_external_skills_dirs()
    assert first == [external.resolve()]

    # Rewrite config; bump mtime forward explicitly so filesystems with
    # coarse mtime granularity still register the change on fast test
    # systems.
    config.write_text(
        "skills:\n"
        f"  external_dirs:\n"
        f"    - {other}\n",
        encoding="utf-8",
    )
    stat = config.stat()
    future = stat.st_atime + 10
    os.utime(config, (future, future))

    second = get_external_skills_dirs()
    assert second == [other.resolve()]


def test_cache_invalidates_when_external_symlink_target_changes(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    presets = tmp_path / "presets"
    v1 = presets / "v1" / "skills"
    v2 = presets / "v2" / "skills"
    v1.mkdir(parents=True)
    v2.mkdir(parents=True)
    current = presets / "current"
    current.symlink_to(presets / "v1", target_is_directory=True)
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {current}/skills\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()

    assert get_external_skills_dirs() == [v1.resolve()]
    current.unlink()
    current.symlink_to(presets / "v2", target_is_directory=True)

    assert get_external_skills_dirs() == [v2.resolve()]


def test_cache_invalidates_when_missing_external_dir_appears(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    presets = tmp_path / "presets"
    current = presets / "current"
    (home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {current}/skills\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()

    assert get_external_skills_dirs() == []

    v1 = presets / "v1" / "skills"
    v1.mkdir(parents=True)
    current.symlink_to(presets / "v1", target_is_directory=True)

    assert get_external_skills_dirs() == [v1.resolve()]


def test_returns_empty_when_config_missing(tmp_path, monkeypatch):
    """No config file → empty list, cached as empty."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _external_dirs_cache_clear()

    assert get_external_skills_dirs() == []




def test_cache_key_is_per_config_path(tmp_path, monkeypatch):
    """Two different HERMES_HOMEs keep separate cache entries."""
    home_a = tmp_path / "home_a" / ".hermes"
    home_a.mkdir(parents=True)
    ext_a = tmp_path / "ext_a"
    ext_a.mkdir()
    (home_a / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_a}\n", encoding="utf-8"
    )

    home_b = tmp_path / "home_b" / ".hermes"
    home_b.mkdir(parents=True)
    ext_b = tmp_path / "ext_b"
    ext_b.mkdir()
    (home_b / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {ext_b}\n", encoding="utf-8"
    )

    _external_dirs_cache_clear()

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [ext_a.resolve()]

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert get_external_skills_dirs() == [ext_b.resolve()]

    # And switching back still works — both entries coexist in the cache.
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert get_external_skills_dirs() == [ext_a.resolve()]


def test_self_contained_marker_disables_external_dirs(hermes_home_with_config):
    """A verified self-contained profile never scans the shared preset root."""
    home, _external, _config = hermes_home_with_config
    marker = home / ".zettlab-self-contained-agent"
    marker.touch()

    assert get_external_skills_dirs() == []

def test_self_contained_marker_bypasses_existing_cache(hermes_home_with_config):
    """Publishing the marker takes effect without a config/mtime change."""
    home, external, _config = hermes_home_with_config
    assert get_external_skills_dirs() == [external.resolve()]

    (home / ".zettlab-self-contained-agent").touch()

    assert get_external_skills_dirs() == []


@pytest.mark.parametrize("marker_kind", ["symlink", "directory"])
def test_invalid_self_contained_marker_fails_closed(
    hermes_home_with_config, marker_kind, tmp_path
):
    """A marker that is not a regular file must not re-enable shared skills."""
    home, _external, _config = hermes_home_with_config
    marker = home / ".zettlab-self-contained-agent"
    if marker_kind == "symlink":
        target = tmp_path / "marker-target"
        target.touch()
        marker.symlink_to(target)
    else:
        marker.mkdir()

    assert get_external_skills_dirs() == []


def test_unreadable_self_contained_marker_fails_closed(hermes_home_with_config, monkeypatch):
    """An inspection failure is conservative and hides external skills."""
    home, _external, _config = hermes_home_with_config
    marker = home / ".zettlab-self-contained-agent"
    marker.touch()

    real_open = skill_utils.os.open

    def fail_marker_open(path, flags, *args, **kwargs):
        if os.fspath(path) == os.fspath(marker):
            raise OSError("simulated marker read failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(skill_utils.os, "open", fail_marker_open)
    assert get_external_skills_dirs() == []


def test_unstatable_self_contained_marker_fails_closed(hermes_home_with_config, monkeypatch):
    """An fd stat failure is also isolated without escaping the hook."""
    home, _external, _config = hermes_home_with_config
    marker = home / ".zettlab-self-contained-agent"
    marker.touch()

    real_fstat = skill_utils.os.fstat

    def fail_marker_fstat(fd):
        raise OSError("simulated marker stat failure")

    monkeypatch.setattr(skill_utils.os, "fstat", fail_marker_fstat)
    assert get_external_skills_dirs() == []
    monkeypatch.setattr(skill_utils.os, "fstat", real_fstat)
