"""Contract manifest for the Zet Agent SSE extension (chat-ui-b0-1-guard-manifests).

Runs on every PR without any secret: the committed manifest must equal what
``ast`` finds in ``zet_agent.py``, every frame type must have a golden in the
mirrored contract snapshot (and vice versa), the doc must list every type, and
the generated field tables in the doc must match the golden set.
"""

from __future__ import annotations

import os

import pytest

from tests.gateway import chat_ui_contract as cuc


def _update_requested() -> bool:
    return os.environ.get("CHAT_UI_MANIFEST_UPDATE") == "1"


def test_manifest_is_up_to_date():
    snap = cuc.load_snapshot()
    manifest = cuc.build_manifest(snap)
    want = cuc.manifest_text(manifest)
    if _update_requested():
        cuc.MANIFEST_PATH.write_text(want, encoding="utf-8")
        return
    have = cuc.MANIFEST_PATH.read_text(encoding="utf-8") if cuc.MANIFEST_PATH.exists() else ""
    assert have == want, (
        f"{cuc.MANIFEST_PATH.relative_to(cuc.REPO_ROOT)} is stale; regenerate: {cuc.MANIFEST_GENERATED_BY}"
    )


def test_every_payload_type_is_a_string_literal():
    scan = cuc.scan_payload_types()
    assert scan["unparsed"] == [], (
        "payload.type must be a string literal so the contract guard can see it; "
        f"dynamic assignments: {scan['unparsed']}"
    )


def test_payload_types_match_golden_and_doc():
    snap = cuc.load_snapshot()
    golden = snap["payload"]["golden"]["hermes"]
    scan = cuc.scan_payload_types()
    golden_types = {k for k in golden if k not in cuc.NON_TYPE_GOLDENS}
    ignore = set(snap["payload"].get("known", {}).get("hermes", {}).get("ignore_type_literals", []))
    emitted = set(scan["payload_types"]) - ignore
    assert emitted - golden_types == set(), f"emitted without golden: {sorted(emitted - golden_types)}"
    assert golden_types - emitted == set(), f"golden without producer: {sorted(golden_types - emitted)}"
    allow = set(snap["payload"].get("known", {}).get("hermes", {}).get("ignore_type_literals", []))
    unlisted = set(scan["ignored_type_literals"]) - allow
    assert unlisted == set(), (
        f"undotted type literals not on the audited allowlist (known-differences hermes.ignore_type_literals): {sorted(unlisted)}; "
        "a new frame needs golden + doc, a non-frame literal needs the allowlist"
    )
    documented = set(cuc.documented_types())
    assert emitted - documented == set(), f"undocumented in {cuc.CONTRACT_DOC.name}: {sorted(emitted - documented)}"
    # static shape: a literal frame never writes a key the golden does not document
    for t, shape in scan["frame_shapes"].items():
        extra = set(shape["keys"]) - set(golden[t])
        assert extra == set(), f"{t}: producer writes {sorted(extra)} but golden/hermes/{t}.json lacks them"


def test_doc_field_tables_are_generated_from_golden():
    snap = cuc.load_snapshot()
    golden = snap["payload"]["golden"]["hermes"]
    text = cuc.CONTRACT_DOC.read_text(encoding="utf-8")
    names = cuc.fenced_names(text)
    expected = sorted(k for k in golden if k not in ("hermes-error", "finish-chunk"))
    assert sorted(names) == expected, f"doc fences {sorted(names)} != golden frames {expected}"
    regenerated = cuc.apply_fences(text, golden)
    if _update_requested():
        cuc.CONTRACT_DOC.write_text(regenerated, encoding="utf-8")
        return
    assert regenerated == text, "generated field tables in the contract doc are stale; regenerate with CHAT_UI_MANIFEST_UPDATE=1"


@pytest.mark.parametrize("sample", [
    ('payload = {"type": "hermes.new_frame", "a": 1}', ["hermes.new_frame"], []),
    ("payload = dict(type='hermes.dict_frame', a=1)", ["hermes.dict_frame"], []),
    ('frame["type"] = "context.compaction"', ["context.compaction"], []),
    ('frame["type"] = kind_from_upstream', [], ["line 1"]),
    ("payload = dict(type=kind)", [], ["line 1"]),
    ('payload = {"type": kind}', [], ["line 1"]),
    ('payload = {"type": "disabled"}', [], []),
])
def test_scanner_is_closed_set(sample):
    src, want_types, want_unparsed_prefixes = sample
    scan = cuc.scan_payload_types(src)
    assert scan["payload_types"] == want_types
    assert len(scan["unparsed"]) == len(want_unparsed_prefixes)
    for got, prefix in zip(scan["unparsed"], want_unparsed_prefixes):
        assert got.startswith(prefix)
