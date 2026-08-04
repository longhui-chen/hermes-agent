"""Manifest session/turn correlation fields (delegation-app-foundation).

zettlab-local-server tails ``cache/delegation/live/<id>/`` to attribute live
transcripts to a chat session/turn. The manifest must carry the dispatching
gateway session_key + turn_id when they exist, and must NOT leak the approval
helper's literal ``"default"`` sentinel into CLI dispatches.
"""

import json

from gateway.session_context import set_zettlab_turn_id
from tools.approval import reset_current_session_key, set_current_session_key
from tools.delegation_live_log import create_live_transcripts, live_transcript_root


def _manifest(deleg_id):
    return json.loads(
        (live_transcript_root() / deleg_id / "manifest.json").read_text()
    )


def test_manifest_omits_context_when_unset():
    deleg_id, _writers, _paths = create_live_transcripts([{"goal": "g"}])
    manifest = _manifest(deleg_id)
    # Especially: no literal "default" from get_current_session_key()'s
    # default parameter.
    assert "session_key" not in manifest
    assert "turn_id" not in manifest


def test_manifest_includes_gateway_context_when_set():
    token = set_current_session_key("agent:main:zet_agent:dm:zettlab:u1:agentA:1")
    set_zettlab_turn_id("t_abc123")
    try:
        deleg_id, _writers, _paths = create_live_transcripts([{"goal": "g"}])
        manifest = _manifest(deleg_id)
        assert manifest["session_key"] == "agent:main:zet_agent:dm:zettlab:u1:agentA:1"
        assert manifest["turn_id"] == "t_abc123"
    finally:
        reset_current_session_key(token)
        set_zettlab_turn_id("")


def test_manifest_survives_context_helper_failure(monkeypatch):
    """Correlation is best-effort: a broken helper must not kill dispatch."""
    import tools.delegation_live_log as live_log

    def _boom():
        raise RuntimeError("no context")

    monkeypatch.setattr(live_log, "_dispatch_context_fields", _boom)
    # create_live_transcripts wraps manifest writing in best-effort handling;
    # transcripts must still be created.
    deleg_id, writers, paths = create_live_transcripts([{"goal": "g"}])
    assert deleg_id and len(paths) == 1
