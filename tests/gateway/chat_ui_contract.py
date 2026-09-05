"""chat-ui contract v1 helpers for hermes-agent (chat-ui-b0-1-guard-manifests).

hermes-agent produces the event: hermes.tool.progress extension frames.
Instead of letting the root guard parse zet_agent.py with regular
expressions, this module enumerates every payload.type literal with the
real Python parser (:mod:`ast`), records the static shape of every literal
frame, and emits the contract manifest the root guard reads
(schemas/chat-ui.manifest.json). The root contract itself is mirrored as
schemas/chat-ui-golden.snapshot.json with its sha pinned below, so the
tests in this package need no cross-repo checkout.

Regenerate the manifest and the doc fences::

    CHAT_UI_MANIFEST_UPDATE=1 python -m pytest -q tests/gateway/test_zet_agent_contract_manifest.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ZET_AGENT_PY = REPO_ROOT / "gateway" / "platforms" / "zet_agent.py"
CONTRACT_DOC = REPO_ROOT / "docs" / "zet-agent-sse-extension-contract.md"
SNAPSHOT_PATH = REPO_ROOT / "schemas" / "chat-ui-golden.snapshot.json"
MANIFEST_PATH = REPO_ROOT / "schemas" / "chat-ui.manifest.json"
ATTACHMENT_SCHEMA_PATH = REPO_ROOT / "schemas" / "chatproto.schema.json"

MANIFEST_VERSION = 1
MANIFEST_REPO = "hermes-agent"
MANIFEST_GENERATED_BY = (
    "CHAT_UI_MANIFEST_UPDATE=1 python -m pytest -q tests/gateway/test_zet_agent_contract_manifest.py"
)

# Pinned sha256 of the mirrored contract snapshot payload
# (zettlab-product-dev: node scripts/chat-ui-contract/snapshot.mjs --print-sha).
# Bump together with schemas/chat-ui-golden.snapshot.json on a reviewed contract change.
CHAT_UI_GOLDEN_SNAPSHOT_SHA256 = "a1f5df4f21381fb04f36a73ab1bc5c75b1430e84fa75e75fb46165aabfe4d0a0"

# Golden files under golden/hermes that are not keyed by payload.type.
NON_TYPE_GOLDENS = ("tool-frame.running", "tool-frame.completed", "tool-frame.error", "hermes-error", "finish-chunk")


def canonical_json(value: Any) -> str:
    """Byte-for-byte what scripts/chat-ui-contract/snapshot.mjs hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_snapshot() -> dict:
    raw = SNAPSHOT_PATH.read_text(encoding="utf-8")
    snap = json.loads(raw)
    assert snap.get("snapshot_version") == 1, f"snapshot_version={snap.get('snapshot_version')} want 1"
    assert re.fullmatch(r"[0-9a-f]{40}", str(snap.get("root_commit") or "")), (
        f"root_commit={snap.get('root_commit')!r}: the mirror must record the zettlab-product-dev commit it was taken at"
    )
    digest = hashlib.sha256(canonical_json(snap["payload"]).encode("utf-8")).hexdigest()
    assert digest == snap["sha256"], (
        f"mirror payload sha256={digest} but file says {snap['sha256']}; the mirror was edited by hand"
    )
    assert snap["sha256"] == CHAT_UI_GOLDEN_SNAPSHOT_SHA256, (
        f"mirror sha256={snap['sha256']} want pinned {CHAT_UI_GOLDEN_SNAPSHOT_SHA256}; "
        "update schemas/chat-ui-golden.snapshot.json and CHAT_UI_GOLDEN_SNAPSHOT_SHA256 together"
    )
    return snap


def is_frame_type(literal: str) -> bool:
    """Frame types are dotted (hermes.todo) or the one legacy bare name."""
    return "." in literal or literal == "steer_dropped"


def scan_payload_types(src: str | None = None) -> dict:
    """Enumerate every payload.type the adapter can emit, using ast.

    Covers dict literals ({"type": "x"}), subscript assignments
    (frame["type"] = "x") and dict(type="x") calls. Any type whose
    value is not a string constant is reported in unparsed — the manifest
    test turns that into a failure, so a dynamic frame type cannot hide.
    frame_shapes records the constant keys of every dict literal that
    carries a type (skipped when the literal has a **spread).
    """
    text = src if src is not None else ZET_AGENT_PY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    types: set[str] = set()
    ignored: set[str] = set()
    unparsed: list[str] = []
    shapes: dict[str, set[str]] = {}
    partial: set[str] = set()

    def record(literal: str, keys: list[str] | None, line: int) -> None:
        if not is_frame_type(literal):
            ignored.add(literal)
            return
        types.add(literal)
        if keys is None:
            partial.add(literal)
        else:
            shapes.setdefault(literal, set()).update(keys)

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "type":
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        spread = any(k is None for k in node.keys)
                        keys = None if spread else [k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                        record(value.value, keys, node.lineno)
                    else:
                        unparsed.append(f"line {node.lineno}: dict literal type={ast.unparse(value)}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "type"
                ):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        record(node.value.value, None, node.lineno)
                    else:
                        unparsed.append(f"line {node.lineno}: {ast.unparse(target)} = {ast.unparse(node.value)}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            for kw in node.keywords:
                if kw.arg == "type":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        keys = [k.arg for k in node.keywords if k.arg] if not any(k.arg is None for k in node.keywords) else None
                        record(kw.value.value, keys, node.lineno)
                    else:
                        unparsed.append(f"line {node.lineno}: dict(type={ast.unparse(kw.value)})")
    return {
        "payload_types": sorted(types),
        "ignored_type_literals": sorted(ignored),
        "unparsed": unparsed,
        "frame_shapes": {t: {"keys": sorted(k)} for t, k in sorted(shapes.items()) if t not in partial},
        "frame_shapes_unknown": sorted(t for t in types if t in partial or t not in shapes),
    }


def documented_types(doc_text: str | None = None) -> list[str]:
    text = doc_text if doc_text is not None else CONTRACT_DOC.read_text(encoding="utf-8")
    return sorted({m for m in re.findall(r"`([a-z][a-z0-9_.]*)`", text) if is_frame_type(m)})


def attachment_schema_sha256() -> str:
    return hashlib.sha256(ATTACHMENT_SCHEMA_PATH.read_bytes()).hexdigest()


def build_manifest(snapshot: dict) -> dict:
    scan = scan_payload_types()
    return {
        "manifest_version": MANIFEST_VERSION,
        "repo": MANIFEST_REPO,
        "generated_by": MANIFEST_GENERATED_BY,
        "snapshot_sha256": snapshot["sha256"],
        "attachment_schema_sha256": attachment_schema_sha256(),
        "hermes": {
            "payload_types": scan["payload_types"],
            "documented_types": documented_types(),
            "unparsed": scan["unparsed"],
            "frame_shapes": scan["frame_shapes"],
            "frame_shapes_unknown": scan["frame_shapes_unknown"],
            "ignored_type_literals": scan["ignored_type_literals"],
        },
    }


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


# ---------- doc fences (field tables generated from the golden set) ----------

FENCE_BEGIN = "<!-- generated:begin {name} -->"
FENCE_END = "<!-- generated:end {name} -->"
FENCE_RE = re.compile(r"<!-- generated:begin (?P<name>[^ ]+) -->\n(?P<body>.*?)<!-- generated:end (?P=name) -->", re.S)


def _wire_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def render_fence(name: str, frame: dict) -> str:
    """The field list of one golden frame, as the doc's generated block."""
    rows = "\n".join(f"| `{k}` | {_wire_type(v)} | `{json.dumps(v, ensure_ascii=False)[:60]}` |" for k, v in frame.items())
    body = (
        f"Fields from `contracts/chat-ui/v1/golden/hermes/{name}.json` (do not edit; regenerated by the manifest test):\n\n"
        "| Field | Wire type | Golden sample |\n|---|---|---|\n" + rows + "\n"
    )
    return FENCE_BEGIN.format(name=name) + "\n" + body + FENCE_END.format(name=name)


def apply_fences(doc_text: str, golden_hermes: dict) -> str:
    """Replace every fenced block in the doc with the freshly rendered one."""

    def repl(m: re.Match) -> str:
        name = m.group("name")
        frame = golden_hermes.get(name)
        if frame is None:
            raise AssertionError(f"doc fence {name!r} has no golden/hermes/{name}.json")
        return render_fence(name, frame)

    return FENCE_RE.sub(repl, doc_text)


def fenced_names(doc_text: str) -> list[str]:
    return [m.group("name") for m in FENCE_RE.finditer(doc_text)]
