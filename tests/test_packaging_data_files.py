"""Guard the wheel payload for bundled skills + the seed policy.

A wheel must ship the bundled skill trees (skills/, optional-skills/) with
structure, plus config/skill_seed_policy.json and the
fallback manifest, or a wheel install seeds 0 skills. setuptools takes data_files
from pyproject.toml's [tool.setuptools.data-files] in preference to setup.py, and
static pyproject globs cannot preserve the nested tree — so the trees are declared
programmatically in setup.py and the pyproject table MUST stay absent. These tests
fail loudly if either invariant is broken (the trap is silent: the build still
succeeds, it just drops every skill).
"""
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _data_file_tree(root_name: str):
    """Mirror of setup.py:_data_file_tree — one (target_dir, files) per subdir."""
    root = REPO / root_name
    grouped = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            grouped[str(path.relative_to(REPO).parent)].append(str(path.relative_to(REPO)))
    return sorted(grouped.items())


def test_pyproject_has_no_data_files_table():
    """If [tool.setuptools.data-files] reappears it overrides setup.py and the
    wheel silently ships zero skills (pyproject precedence over setup.py)."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    ts = data.get("tool", {}).get("setuptools", {})
    assert "data-files" not in ts, (
        "[tool.setuptools.data-files] is back in pyproject.toml — it takes "
        "precedence over setup.py and would drop every bundled skill from the "
        "wheel. Declare data_files in setup.py instead (see setup.py rationale)."
    )


def test_setup_py_declares_all_skill_trees():
    """setup.py must ship the skill trees + policy + locales via data_files."""
    setup_src = (REPO / "setup.py").read_text(encoding="utf-8")
    for tree in ("skills", "optional-skills", "locales"):
        assert f'_data_file_tree("{tree}")' in setup_src, f"setup.py drops {tree}/ from the wheel"
    assert "skill_seed_policy.json" in setup_src
    assert "seed_fallback_manifest.txt" in setup_src


def test_skill_trees_are_nonempty_and_structured():
    """Each declared tree resolves to real files with category structure."""
    for tree in ("skills", "optional-skills"):
        entries = _data_file_tree(tree)
        assert entries, f"{tree}/ produced no data_files entries"
        # structure preserved: more than one distinct target dir (categories/skills)
        assert len({d for d, _ in entries}) > 1, f"{tree}/ would be flattened"
    assert (REPO / "config" / "skill_seed_policy.json").is_file()
    assert (REPO / "config" / "seed_fallback_manifest.txt").is_file()


if __name__ == "__main__":
    test_pyproject_has_no_data_files_table()
    test_setup_py_declares_all_skill_trees()
    test_skill_trees_are_nonempty_and_structured()
    print("packaging data_files guards OK")
