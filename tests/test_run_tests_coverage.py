"""Unit tests for the --coverage mode of scripts/run_tests_parallel.py.

The runner lives under scripts/ (not an importable package), so we load it by
path. We test the two pieces of new logic that don't require actually spawning
coverage: the per-file command wrapping and the best-effort finalize step.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "run_tests_parallel.py"
_spec = importlib.util.spec_from_file_location("run_tests_parallel_under_test", _RUNNER)
rtp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rtp)


def test_build_pytest_cmd_plain():
    assert rtp._build_pytest_cmd(Path("tests/x.py"), ["-v"], coverage=False) == [
        sys.executable, "-m", "pytest", "tests/x.py", "-v",
    ]


def test_build_pytest_cmd_wraps_with_coverage():
    assert rtp._build_pytest_cmd(Path("tests/x.py"), ["-v"], coverage=True) == [
        sys.executable, "-m", "coverage", "run", "-m", "pytest", "tests/x.py", "-v",
    ]


def test_finalize_coverage_emits_percent(tmp_path, monkeypatch, capsys):
    """combine→json→parse: cov.json is written and its percent is surfaced."""
    def _fake_run(cmd, **kwargs):
        verb = cmd[3]  # [python, -m, coverage, <verb>, ...]
        if verb == "json":
            out = cmd[cmd.index("-o") + 1]
            Path(out).write_text(json.dumps({"totals": {"percent_covered": 42.5}}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rtp.subprocess, "run", _fake_run)
    rtp._finalize_coverage(tmp_path)

    cov_json = tmp_path / "cov.json"
    assert cov_json.exists()
    assert json.loads(cov_json.read_text())["totals"]["percent_covered"] == 42.5
    assert "42.5%" in capsys.readouterr().out


def test_finalize_coverage_best_effort_on_failure(tmp_path, monkeypatch, capsys):
    """A coverage tooling failure (or timeout) must not raise — best effort."""
    def _fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(rtp.subprocess, "run", _fake_run)
    rtp._finalize_coverage(tmp_path)  # must not raise

    assert not (tmp_path / "cov.json").exists()
    assert "coverage json failed" in capsys.readouterr().err
