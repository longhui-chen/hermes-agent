"""Tests for ruff lint config — guards against accidental rule removal.

PLW1514 (unspecified-encoding) was enabled after a debug session on
Windows turned up three separate UTF-8 regressions in execute_code.
The rule catches bare ``open()`` / ``read_text()`` / ``write_text()``
calls that default to locale encoding — cp1252 on Windows — which
silently corrupts non-ASCII content.

These tests ensure:
  1. PLW1514 stays in ``[tool.ruff.lint.select]``
  2. pyproject.toml has ``preview = true`` (required — PLW1514 is a
     preview rule in ruff 0.15.x)

If someone removes either, ``ruff check`` stops enforcing UTF-8-explicit
opens and we're back to the original Windows-regression trap.

(The zettlab fork runs its own CI and does not ship upstream's
``.github/workflows/lint.yml``, so the workflow-shape assertions that
upstream keeps here were dropped.)
"""

from __future__ import annotations

import pathlib

import pytest

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover — 3.10 and earlier
    import tomli as tomllib  # type: ignore

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_pyproject() -> dict:
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


class TestRuffConfig:
    def test_plw1514_is_in_select_list(self):
        """pyproject.toml must keep PLW1514 in [tool.ruff.lint.select]."""
        cfg = _load_pyproject()
        selected = (
            cfg.get("tool", {})
            .get("ruff", {})
            .get("lint", {})
            .get("select", [])
        )
        assert "PLW1514" in selected, (
            "PLW1514 (unspecified-encoding) was removed from "
            "[tool.ruff.lint.select].  This rule blocks bare open() calls "
            "that default to locale encoding on Windows — removing it "
            "re-opens a class of UTF-8 bugs we already paid to close.  "
            "If you genuinely want to remove it, delete this test in the "
            "same commit so the intent is deliberate."
        )

    def test_preview_mode_enabled(self):
        """PLW1514 is a preview rule in ruff 0.15.x — preview=true is
        required for it to actually run."""
        cfg = _load_pyproject()
        ruff_cfg = cfg.get("tool", {}).get("ruff", {})
        assert ruff_cfg.get("preview") is True, (
            "[tool.ruff] preview=true is required — PLW1514 is a preview "
            "rule and silently becomes a no-op without it.  If this ever "
            "becomes a stable rule, you can drop preview=true but must "
            "verify PLW1514 still fires in a sample test run first."
        )
