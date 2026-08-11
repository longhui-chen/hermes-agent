"""Pinned mirror of local-server's authoritative attachment schema."""

import hashlib
import json
from pathlib import Path


EXPECTED_SCHEMA_SHA256 = "0c56b93ea0f0d673a51afa938a04c2a4fe3c93fdd790fa91a0e10e99de85337a"


def test_chatproto_schema_mirror_matches_hermes_wire_contract():
    raw = (Path(__file__).parents[2] / "schemas" / "chatproto.schema.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SCHEMA_SHA256
    schema = json.loads(raw)
    assert schema["$id"] == "https://schemas.zettlab.com/chatproto/attachment-v1.schema.json"
    definitions = schema["$defs"]

    attachment = {
        "id": "att-1",
        "kind": "channel.connect",
        "v": 1,
        "state": "active",
        "payload": {"channel_kind": "feishu"},
        "actions": [{"id": "connect", "style": "primary"}],
        "expires_at": 1785920400000,
    }
    assert set(definitions["ChatAttachmentWire"]["required"]) <= set(attachment)
    assert set(attachment) <= set(definitions["ChatAttachmentWire"]["properties"])

    action = {
        "attachment_id": "att-1",
        "action_id": "connect",
        "action_token": "token-1",
        "turn_id": "turn-1",
        "payload": {"source": "card"},
    }
    assert set(action) == set(definitions["AttachmentActionData"]["properties"])
    assert definitions["ChatAttachmentResolution"]["properties"]["at"]["type"] == "integer"
    assert definitions["AttachmentUpsertEvent"]["properties"]["type"]["const"] == "attachment.upsert"
    assert definitions["AttachmentActionAckEvent"]["properties"]["type"]["const"] == "attachment.action.ack"
