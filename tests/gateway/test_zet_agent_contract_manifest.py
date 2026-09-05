"""Validate the committed Hermes Chat UI manifest against its snapshot pin."""

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_manifest_is_pinned_to_committed_golden_snapshot():
    manifest = json.loads((ROOT / "schemas/chat-ui.manifest.json").read_text())
    snapshot = json.loads((ROOT / "schemas/chat-ui-golden.snapshot.json").read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["repo"] == "hermes-agent"
    assert manifest["snapshot_sha256"] == snapshot["sha256"]
    assert {"hermes.approval", "hermes.clarify"} <= set(manifest["hermes"]["payload_types"])
