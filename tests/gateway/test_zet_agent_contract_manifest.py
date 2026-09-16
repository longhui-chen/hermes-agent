"""Validate the committed Hermes Chat UI manifest against its producers.

The manifest is generated from the cross-repo snapshot, but the PR guard must
also prove that the producer still emits every declared payload type and that
the frozen golden/doc field sets have not drifted.  Keeping these checks here
means a stale, hand-edited JSON file cannot make the contract job green.
"""

import ast
import hashlib
import json
import os
import re

from tests.gateway.chat_ui_contract import CHAT_UI_GOLDEN_SNAPSHOT_SHA256
from gateway.platforms.zet_agent_bt import identity_fields
from pathlib import Path


ROOT = Path(__file__).parents[2]
PRODUCER_FILES = (
    ROOT / "gateway" / "platforms" / "zet_agent.py",
    ROOT / "gateway" / "platforms" / "api_server.py",
)


def test_manifest_is_pinned_to_committed_golden_snapshot():
    manifest = json.loads((ROOT / "schemas/chat-ui.manifest.json").read_text())
    snapshot = json.loads((ROOT / "schemas/chat-ui-golden.snapshot.json").read_text())
    digest = hashlib.sha256(json.dumps(snapshot["payload"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert digest == snapshot["sha256"] == CHAT_UI_GOLDEN_SNAPSHOT_SHA256
    update = os.environ.get("CHAT_UI_MANIFEST_UPDATE") == "1"
    identities = snapshot["payload"]["known"]["hermes"]["identity_fields"]
    assert identity_fields() == identities
    for kind, path in identities.items():
        if update:
            manifest["hermes"]["frame_shapes"][kind]["identity_field"] = path
        assert manifest["hermes"]["frame_shapes"][kind]["identity_field"] == path
        if path != "fixed":
            value = snapshot["payload"]["golden"]["hermes"][kind]
            for part in path.split("."):
                value = value[part]
            assert isinstance(value, str) and value
    hermes = manifest["hermes"]
    golden = snapshot["payload"]["golden"]["hermes"]
    assert manifest["manifest_version"] == 1
    assert manifest["repo"] == "hermes-agent"
    assert manifest["snapshot_sha256"] == snapshot["sha256"] or update
    payload_types = set(hermes["payload_types"])
    assert set(hermes["frame_shapes"]) <= payload_types
    assert payload_types <= set(hermes["documented_types"])
    assert {"hermes.approval", "hermes.clarify"} <= payload_types

    compression = ast.parse((ROOT / "agent/conversation_compression.py").read_text())
    compaction_keys = set()
    for node in ast.walk(compression):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_emit_structured_status" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "context.compaction"):
            assert isinstance(node.args[1], ast.Dict)
            for key in node.args[1].keys:
                assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                compaction_keys.add(key.value)
    assert compaction_keys
    compaction_keys |= {"type", "turn_id", "index"}
    if update:
        hermes["frame_shapes"]["context.compaction"]["keys"] = sorted(compaction_keys)
        hermes["frame_shapes_unknown"] = [kind for kind in hermes["frame_shapes_unknown"] if kind != "context.compaction"]
    assert set(hermes["frame_shapes"]["context.compaction"]["keys"]) == compaction_keys
    assert not set(identities).intersection(hermes["frame_shapes_unknown"])
    assert not set(identities).intersection(snapshot["payload"]["known"]["hermes"].get("frame_shape_exempt", {}))

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

    document = ROOT / "docs/zet-agent-sse-extension-contract.md"
    source = document.read_text()
    for kind in re.findall(r"<!-- generated:begin (.*?) -->", source):
        fields = []
        for name, value in golden[kind].items():
            wire = ("boolean" if isinstance(value, bool) else "string" if isinstance(value, str)
                    else "number" if isinstance(value, (int, float)) else "array" if isinstance(value, list)
                    else "object" if isinstance(value, dict) else "null")
            sample = json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("|", "&#124;")
            fields.append(f"| `{name}` | {wire} | `{sample}` |")
        block = "\n".join([f"<!-- generated:begin {kind} -->",
            f"Fields from `contracts/chat-ui/v1/golden/hermes/{kind}.json` (do not edit; regenerated by the manifest test):",
            "", "| Field | Wire type | Golden sample |", "|---|---|---|", *fields,
            f"<!-- generated:end {kind} -->"])
        pattern = r"<!-- generated:begin " + re.escape(kind) + r" -->.*?<!-- generated:end " + re.escape(kind) + r" -->"
        current = re.search(pattern, source, re.S).group(0)
        assert update or current == block, f"stale generated fields: {kind}"
        if update:
            source = re.sub(pattern, lambda _: block, source, flags=re.S)
    if update:
        document.write_text(source)
        manifest["snapshot_sha256"] = snapshot["sha256"]
        manifest["hermes"]["snapshot_sha256"] = snapshot["sha256"]
        (ROOT / "schemas/chat-ui.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
