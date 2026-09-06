"""B1 Hermes producer conformance against the pinned root golden snapshot."""

import json
from pathlib import Path

import pytest

from gateway.platforms.zet_agent import _clarify_sentinel


SNAPSHOT = Path(__file__).parents[2] / "schemas" / "chat-ui-golden.snapshot.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(SNAPSHOT.read_text())["payload"]["golden"]["hermes"]


@pytest.mark.parametrize(
    "kind, state, reason",
    [
        (kind, state, reason)
        for kind in ("clarify", "approval")
        for state, reason in (
            ("expired", "timeout"),
            ("cancelled", "turn_interrupted"),
            ("cancelled", "session_reset"),
            ("cancelled", "delivery_failed"),
        )
    ],
)
def test_terminal_variant_is_registered_and_bounded(golden, kind, state, reason):
    key = f"hermes.{kind}.{state}.{reason}"
    assert key in golden
    payload = golden[key]
    assert payload["type"] == f"hermes.{kind}"
    assert payload["state"] == state
    assert payload["state_reason"] == reason
    sentinel = _clarify_sentinel(kind, "a" * 64, state, reason, "clarify could not be delivered")
    assert len(sentinel.encode("utf-8")) <= 160

