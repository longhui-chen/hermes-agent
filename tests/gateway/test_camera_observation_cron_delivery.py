import json
import os
import time

import pytest


@pytest.mark.parametrize("success", [True, False])
def test_camera_summary_survives_many_frames_in_source_memo(tmp_path, monkeypatch, success):
    import hermes_state
    import gateway.platforms.zet_agent_cron as cron

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    agent = "camera-agent"
    bucket = tmp_path / "agents" / "data" / agent / "output" / "source-memo"
    bucket.mkdir(parents=True)
    for index in range(40):
        (bucket / f"camera-a{index:03}-frame.jpg").write_bytes(b"frame")
    summary = bucket / "camera-z-summary.json"
    events = bucket / "camera-z-events.json"
    events.write_text(json.dumps({"events": [{"event_id": "hit", "occurred_at": "2026-09-09T12:00:00Z"}], "analysis_complete": success}))
    result = {"analysis_complete": success, "summary_path": str(summary), "coverage": {"frames": 120}, "analysis": {"events_path": str(events)}}
    summary.write_text(json.dumps(result))
    db = hermes_state.SessionDB()
    sid = "cron_observe_20260909_120000"
    db.create_session(sid, source="cron", user_id="owner")
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (time.time() - 10, sid))
    db.append_message(sid, role="tool", tool_name="terminal", content=json.dumps({"data": result}))
    db.close()
    job = {"id": "observe", "name": "Observation", "origin": {"platform": "zet_agent", "chat_id": f"zettlab:owner:{agent}:source-memo"}}
    body = cron._build_typed_message_content(job, "observe", success, None if success else "interrupted", None)
    metadata = json.loads(body.split("```cron-summary\n", 1)[1].split("```", 1)[0])
    attachments = metadata["attachments"]
    assert len(attachments) <= cron._CRON_ATTACHMENT_LIMIT
    assert attachments[0]["path"] == str(summary.resolve())
    assert attachments[0]["mime"] == "application/json"
    assert attachments[1]["path"] == str(events.resolve())
    assert attachments[1]["mime"] == "application/json"
    assert metadata["last_run_result"] == ("success" if success else "failed")


@pytest.mark.parametrize("case", ["old", "foreign", "symlink", "script"])
def test_explicit_summary_reference_keeps_existing_attachment_rejections(tmp_path, case):
    import gateway.platforms.zet_agent_cron as cron

    root = tmp_path / "agents" / "data" / "owner-agent" / "output" / "memo"
    root.mkdir(parents=True)
    foreign = tmp_path / "agents" / "data" / "foreign-agent" / "output" / "memo" / "summary.json"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("{}")
    path = root / ("helper.py" if case == "script" else "summary.json")
    if case == "symlink":
        path.symlink_to(foreign)
    else:
        path.write_text("{}")
    if case == "old":
        os.utime(path, (1, 1))
    if case == "foreign":
        path = foreign
    assert cron._cron_attachment_for_path(str(path), [root.resolve()], set(), time.time() - 60, allow_external=True) is None
