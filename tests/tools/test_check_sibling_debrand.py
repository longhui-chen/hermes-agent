"""Regression tests for tools/check_sibling_debrand.py — the guard that keeps the
de-branded sibling `zettlab-memo-setup` brand-clean and in sync with upstream
`hermes-agent`. Covers each exit code (0/1/2/3), the --update-baseline gate, and
the round-trip, against temp upstream/sibling/baseline (module globals patched)."""
import json

import pytest

from tools import check_sibling_debrand as mod

# Minimal upstream skill (branded) and its de-branded sibling. Same section
# headers (modulo the product name); the sibling carries zero brand tokens.
UP = """---
name: hermes-agent
author: Hermes Agent + Teknium
---
# Hermes Agent

Hermes Agent by Nous Research. See https://hermes-agent.nousresearch.com/docs/.

## CLI Reference

```
hermes setup
```

## Troubleshooting

Run hermes doctor.
"""

SIB = """---
name: zettlab-memo-setup
author: Zettlab
---
# Zettlab Memo

Zettlab Memo runtime.

## CLI Reference

```
hermes setup
```

## Troubleshooting

Run hermes doctor.
"""


@pytest.fixture
def patched(tmp_path, monkeypatch):
    up = tmp_path / "up"
    up.mkdir()
    (up / "SKILL.md").write_text(UP, encoding="utf-8")
    sib = tmp_path / "sib"
    sib.mkdir()
    (sib / "SKILL.md").write_text(SIB, encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "UPSTREAM", up)
    monkeypatch.setattr(mod, "SIBLING", sib)
    monkeypatch.setattr(mod, "BASELINE", baseline)
    return up, sib, baseline


def _record_baseline(baseline, up):
    baseline.write_text(json.dumps({"upstream_sha256": mod._content_hash(up)}), encoding="utf-8")


def test_ok(patched):
    up, sib, baseline = patched
    assert mod._update_baseline() == 0
    assert baseline.exists()
    assert mod.main() == 0


def test_brand_leak_rc1(patched):
    up, sib, baseline = patched
    _record_baseline(baseline, up)
    (sib / "SKILL.md").write_text(SIB + "\nBuilt on Hermes Agent.\n", encoding="utf-8")
    assert mod.main() == 1


def test_file_drift_rc2(patched):
    up, sib, baseline = patched
    (up / "references").mkdir()
    (up / "references" / "extra.md").write_text("# Extra\n", encoding="utf-8")
    _record_baseline(baseline, up)  # hash matches -> isolate the structural drift
    assert mod.main() == 2


def test_header_drift_rc2(patched):
    up, sib, baseline = patched
    (up / "SKILL.md").write_text(UP + "\n## New Upstream Section\n\nbody\n", encoding="utf-8")
    _record_baseline(baseline, up)  # hash matches -> isolate the header drift
    assert mod.main() == 2


def test_in_section_content_drift_rc3(patched):
    """The whole point of the provenance hash: an in-section upstream change that
    adds no header and no file must still be caught."""
    up, sib, baseline = patched
    assert mod._update_baseline() == 0
    (up / "SKILL.md").write_text(UP + "\nAn extra in-section sentence, no new header.\n", encoding="utf-8")
    assert mod.main() == 3


def test_missing_baseline_rc3(patched):
    up, sib, baseline = patched
    assert not baseline.exists()
    assert mod.main() == 3


def test_update_baseline_refuses_on_dirty_sibling(patched):
    up, sib, baseline = patched
    (sib / "SKILL.md").write_text(SIB + "\nHermes Agent leak.\n", encoding="utf-8")
    assert mod._update_baseline() == 1  # brand leak -> refuse
    assert not baseline.exists()  # nothing rubber-stamped


def test_update_baseline_refuses_on_structural_drift(patched):
    up, sib, baseline = patched
    (up / "SKILL.md").write_text(UP + "\n## New Upstream Section\n\nbody\n", encoding="utf-8")
    assert mod._update_baseline() == 2  # sibling lacks the section -> refuse
    assert not baseline.exists()


def test_roundtrip_resyncs_to_green(patched):
    up, sib, baseline = patched
    assert mod._update_baseline() == 0
    assert mod.main() == 0
    # upstream changes in-section -> drift
    (up / "SKILL.md").write_text(UP + "\nextra line\n", encoding="utf-8")
    assert mod.main() == 3
    # re-record baseline (sibling still brand-clean + structurally parallel) -> green
    assert mod._update_baseline() == 0
    assert mod.main() == 0


def test_junk_files_ignored(patched):
    """A stray .DS_Store / __pycache__ in either tree must not register as drift."""
    up, sib, baseline = patched
    assert mod._update_baseline() == 0
    (up / ".DS_Store").write_text("junk", encoding="utf-8")
    (sib / "__pycache__").mkdir()
    (sib / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    assert mod.main() == 0
