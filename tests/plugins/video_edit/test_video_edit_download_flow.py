from __future__ import annotations

import json

import pytest

from gateway.session_context import (
    pop_execution_session_key,
    push_execution_session_key,
)
from plugins.video_edit import client, paths, state, tools


@pytest.fixture
def isolated_download_flow(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    monkeypatch.setattr("tools.runtime_workdir.agent_output_dir", lambda: str(output))
    return output


def test_interactive_download_flow_allocates_once_and_reuses_across_lineage_tip(
    isolated_download_flow,
    monkeypatch,
):
    workflow = state.workflow_id("turn-download-flow", "agent-a")
    state.update(
        workflow,
        "agent-a",
        {
            "task_id": "turn-download-flow",
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-download-flow",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "status": "completed",
        },
    )
    session_dir = isolated_download_flow / "app-session-flow"
    session_dir.mkdir()
    existing = paths.result_path(
        "agent-a",
        "hangzhou-vlog.mp4",
        allow_existing=True,
        session_id="app-session-flow",
    )
    existing.write_bytes(b"previous-edit")
    downloads = []

    def fake_download(_url, target):
        downloads.append(target)
        target.write_bytes(b"new-edit")
        return client.file_evidence(target)

    monkeypatch.setattr(client, "download", fake_download)
    token = push_execution_session_key("zettlab:user-a:agent-a:app-session-flow")
    try:
        first = json.loads(
            tools.handle_download_result(
                {"workflow_id": workflow, "filename": "hangzhou-vlog.mp4"},
                agent_id="agent-a",
                session_id="api-lineage-tip-one",
            )
        )
    finally:
        pop_execution_session_key(token)

    assert first["ok"] is True
    delivered_path = paths.validate_output_file(
        first["output"],
        "agent-a",
        session_id="app-session-flow",
    )
    assert delivered_path != existing
    assert delivered_path.name.startswith("hangzhou-vlog-")
    assert delivered_path.read_bytes() == b"new-edit"
    assert existing.read_bytes() == b"previous-edit"

    token = push_execution_session_key("zettlab:user-a:agent-a:app-session-flow")
    try:
        reused = json.loads(
            tools.handle_download_result(
                {"workflow_id": workflow, "filename": "ignored-on-resume.mp4"},
                agent_id="agent-a",
                session_id="api-lineage-tip-two",
            )
        )
    finally:
        pop_execution_session_key(token)

    assert reused["ok"] is True
    assert reused["reused"] is True
    assert reused["output"] == first["output"]
    assert downloads == [delivered_path]


def test_interactive_download_flow_rehomes_legacy_flat_checkpoint_once(
    isolated_download_flow,
    monkeypatch,
):
    workflow = state.workflow_id("turn-legacy-download", "agent-a")
    legacy = paths.result_path("agent-a", "legacy-vlog.mp4")
    legacy.write_bytes(b"legacy-render")
    state.update(
        workflow,
        "agent-a",
        {
            "task_id": "turn-legacy-download",
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-legacy-download",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(legacy),
            "status": "delivered",
        },
    )
    monkeypatch.setattr(
        client,
        "download",
        lambda *_args, **_kwargs: pytest.fail("a committed result must not redownload"),
    )

    token = push_execution_session_key("zettlab:user-a:agent-a:app-session-legacy")
    try:
        delivered = json.loads(
            tools.handle_download_result(
                {"workflow_id": workflow},
                agent_id="agent-a",
                session_id="api-lineage-tip",
            )
        )
    finally:
        pop_execution_session_key(token)

    expected = isolated_download_flow / "app-session-legacy" / "legacy-vlog.mp4"
    assert delivered["ok"] is True
    assert delivered["reused"] is True
    assert delivered["output"] == str(expected)
    assert expected.read_bytes() == b"legacy-render"
    assert not legacy.exists()
    checkpoint = state.get(workflow, "agent-a")
    assert checkpoint["output_path"] == str(expected)
    assert checkpoint["output_session_id"] == "app-session-legacy"


def test_interactive_download_flow_rejects_checkpoint_from_another_session(
    isolated_download_flow,
    monkeypatch,
):
    workflow = state.workflow_id("turn-cross-session-download", "agent-a")
    other_dir = isolated_download_flow / "other-app-session"
    other_dir.mkdir()
    other = other_dir / "other-vlog.mp4"
    other.write_bytes(b"other-session-render")
    state.update(
        workflow,
        "agent-a",
        {
            "task_id": "turn-cross-session-download",
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-cross-session-download",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(other),
            "status": "delivered",
        },
    )
    monkeypatch.setattr(
        client,
        "download",
        lambda *_args, **_kwargs: pytest.fail("cross-session output must not download"),
    )

    token = push_execution_session_key("zettlab:user-a:agent-a:app-session-current")
    try:
        rejected = json.loads(
            tools.handle_download_result(
                {"workflow_id": workflow},
                agent_id="agent-a",
                session_id="api-lineage-tip",
            )
        )
    finally:
        pop_execution_session_key(token)

    assert rejected["reason_code"] == "workflow_unavailable"
    assert rejected["retryable"] is False
    assert other.read_bytes() == b"other-session-render"
