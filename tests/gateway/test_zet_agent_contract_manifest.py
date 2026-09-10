"""Validate the committed Hermes Chat UI manifest against its producers.

The manifest is generated from the cross-repo snapshot, but the PR guard must
also prove that the producer still emits every declared payload type and that
the frozen golden/doc field sets have not drifted.  Keeping these checks here
means a stale, hand-edited JSON file cannot make the contract job green.
"""

import json
import os
from pathlib import Path


ROOT = Path(__file__).parents[2]
PRODUCER_FILES = (
    ROOT / "gateway" / "platforms" / "zet_agent.py",
    ROOT / "gateway" / "platforms" / "api_server.py",
)


def test_manifest_is_pinned_to_committed_golden_snapshot():
    manifest = json.loads((ROOT / "schemas/chat-ui.manifest.json").read_text())
    snapshot = json.loads((ROOT / "schemas/chat-ui-golden.snapshot.json").read_text())
    update = os.environ.get("CHAT_UI_MANIFEST_UPDATE") == "1"
    if update:
        # Refresh provenance only; producer facts below must validate before writing.
        manifest["snapshot_sha256"] = snapshot["sha256"]
    hermes = manifest["hermes"]
    golden = snapshot["payload"]["golden"]["hermes"]
    assert manifest["manifest_version"] == 1
    assert manifest["repo"] == "hermes-agent"
    assert manifest["snapshot_sha256"] == snapshot["sha256"]
    payload_types = set(hermes["payload_types"])
    # Some pass-through payloads (currently context.compaction) have no
    # stable field shape; every declared shape must nevertheless be a payload.
    assert set(hermes["frame_shapes"]) <= payload_types
    assert payload_types <= set(hermes["documented_types"])
    assert {"hermes.approval", "hermes.clarify"} <= payload_types

    producer_source = "\n".join(path.read_text() for path in PRODUCER_FILES)
    for payload_type in payload_types:
        # Producers use both literal dict values and comparisons/assignments;
        # requiring the exact wire string in source catches removed/renamed
        # emitters without depending on a fragile AST shape.
        assert payload_type in producer_source, f"producer no longer emits {payload_type}"
        assert payload_type in golden, f"golden sample missing {payload_type}"

        if payload_type not in hermes["frame_shapes"]:
            continue
        declared_keys = set(hermes["frame_shapes"][payload_type]["keys"])
        golden_keys = set(golden[payload_type])
        assert declared_keys <= golden_keys, (
            f"manifest fields for {payload_type} are not covered by its golden sample"
        )

    if update:
        (ROOT / "schemas/chat-ui.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
