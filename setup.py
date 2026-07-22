from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.egg_info import egg_info as _egg_info


REPO_ROOT = Path(__file__).parent.resolve()


def _source_tree_is_writable() -> bool:
    probe = REPO_ROOT / ".setuptools-write-probe"
    try:
        with probe.open("w", encoding="utf-8") as handle:
            handle.write("")
        probe.unlink()
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def _temporary_build_dir(kind: str) -> str:
    return tempfile.mkdtemp(prefix=f"hermes-agent-{kind}-")


def _would_write_under_source(path_value: str | None) -> bool:
    if path_value is None:
        return True
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


class ReadOnlySourceBuild(_build):
    def finalize_options(self) -> None:
        if (
            not _source_tree_is_writable()
            and _would_write_under_source(self.build_base)
        ):
            self.build_base = _temporary_build_dir("build")
        super().finalize_options()


class ReadOnlySourceEggInfo(_egg_info):
    def finalize_options(self) -> None:
        if (
            not _source_tree_is_writable()
            and _would_write_under_source(self.egg_base)
        ):
            self.egg_base = _temporary_build_dir("egg-info")
        super().finalize_options()


def _data_file_tree(root_name: str) -> list[tuple[str, list[str]]]:
    root = REPO_ROOT / root_name
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(REPO_ROOT)
        grouped[str(rel_path.parent)].append(str(rel_path))
    return sorted(grouped.items())


setup(
    cmdclass={
        "build": ReadOnlySourceBuild,
        "egg_info": ReadOnlySourceEggInfo,
    },
    # NOTE: data_files is declared HERE (programmatically) rather than in
    # pyproject.toml's [tool.setuptools.data-files] on purpose. A wheel must ship
    # the bundled skill TREES (skills/, optional-skills/) with their nested
    # category/skill structure preserved — static pyproject data-files globs
    # flatten everything into one target dir, which breaks skill discovery.
    # _data_file_tree() walks each tree and emits one (target_dir, files) pair
    # per subdirectory, preserving structure, and picks up new skills
    # automatically (no per-skill list to maintain).
    #
    # Because pyproject fields take precedence over setup.py, [tool.setuptools.
    # data-files] MUST stay absent from pyproject.toml — otherwise it silently
    # overrides this list and the wheel ships zero skills. locales/ lives here too
    # for the same reason (it used to be the lone pyproject entry; #27632/#35374).
    data_files=[
        *_data_file_tree("skills"),
        *_data_file_tree("optional-skills"),
        # i18n catalogs (locales/ is a bare data dir, not a package). Without these
        # in the wheel, sealed installs surface raw i18n keys (#27632/#35374/#23943).
        *_data_file_tree("locales"),
        # Ship the seed policy + fallback manifest alongside skills/ so a wheel
        # install resolves them (else the seed filter silently no-ops).
        ("config", ["config/skill_seed_policy.json", "config/seed_fallback_manifest.txt"]),
    ]
)
