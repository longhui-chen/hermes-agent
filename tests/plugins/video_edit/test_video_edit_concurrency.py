from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import threading
import time

import pytest

from plugins.video_edit import client, normalizer, preferences, state, tools


@pytest.fixture
def isolated_video_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    monkeypatch.setattr("tools.runtime_workdir.agent_output_dir", lambda: str(output))
    return home, output


def _write_video(path: Path, marker: bytes = b"") -> None:
    brand = b"qt  " if path.suffix.lower() == ".mov" else b"isom"
    def box(kind: bytes, payload: bytes) -> bytes:
        return (len(payload) + 8).to_bytes(4, "big") + kind + payload

    hdlr = box(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    path.write_bytes(
        box(b"ftyp", brand + b"\x00\x00\x00\x00" + brand)
        + box(b"moov", box(b"trak", box(b"mdia", hdlr)))
        + marker
    )


def _hold_upload_lock(
    workflow_id: str,
    agent_id: str,
    entered: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with state.upload_lock(workflow_id, agent_id):
        entered.set()
        assert release.wait(timeout=5)


def test_interactive_repeat_preserves_checkpoint_and_first_resolved_values(
    isolated_video_home,
):
    args = {"task_id": "repeat-progressed", "scene": "travel"}
    first = json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))
    workflow_id = first["workflow_id"]
    tools.state.update(
        workflow_id,
        "agent-a",
        {
            "source_paths": ["/persisted/source.mov"],
            "object_keys": ["assets/persisted-source"],
            "project_id": "project-persisted",
            "project": {"status": "completed"},
            "status": "completed",
        },
    )
    preferences.update(
        "agent-a",
        "global",
        "travel",
        "hard",
        "set",
        {"style": "later-memory"},
    )

    repeated = json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))
    checkpoint = tools.state.get(workflow_id, "agent-a")

    assert repeated["ok"] is True
    assert repeated["workflow_id"] == workflow_id
    assert repeated["preferences"] == first["preferences"]
    assert repeated["sources"] == first["sources"]
    assert repeated["memory_hit"] == first["memory_hit"]
    assert checkpoint["source_paths"] == ["/persisted/source.mov"]
    assert checkpoint["object_keys"] == ["assets/persisted-source"]
    assert checkpoint["project_id"] == "project-persisted"
    assert checkpoint["status"] == "completed"


@pytest.mark.parametrize(
    "changed",
    [
        {"scene": "different"},
        {"preferences": {"style": "different"}},
        {"silent": True},
    ],
)
def test_interactive_resolve_conflict_does_not_mutate_checkpoint(
    isolated_video_home,
    changed,
):
    args = {
        "task_id": "conflicting-repeat",
        "scene": "travel",
        "preferences": {"style": "original"},
        "silent": False,
    }
    first = json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))
    workflow_id = first["workflow_id"]
    tools.state.update(
        workflow_id,
        "agent-a",
        {
            "source_paths": ["/persisted/source.mov"],
            "object_keys": ["assets/persisted-source"],
            "project_id": "project-persisted",
            "status": "completed",
        },
    )
    before = tools.state.get(workflow_id, "agent-a")

    conflicting = json.loads(
        tools.handle_preferences_resolve(args | changed, agent_id="agent-a")
    )

    assert conflicting["reason_code"] == "workflow_unavailable"
    assert conflicting["retryable"] is False
    assert conflicting["next"] is None
    assert tools.state.get(workflow_id, "agent-a") == before


def test_interactive_resolve_concurrent_conflict_has_one_winner(
    isolated_video_home,
    monkeypatch,
):
    original_resolve = preferences.resolve
    conflict_waiting = threading.Event()
    release_conflict = threading.Event()

    def gated_resolve(agent_id, scene, explicit, *, silent=False):
        if scene == "conflict":
            conflict_waiting.set()
            assert release_conflict.wait(timeout=2)
        return original_resolve(agent_id, scene, explicit, silent=silent)

    monkeypatch.setattr(preferences, "resolve", gated_resolve)
    winner_args = {
        "task_id": "concurrent-conflict",
        "scene": "winner",
        "preferences": {"style": "winner"},
    }
    conflict_args = {
        "task_id": "concurrent-conflict",
        "scene": "conflict",
        "preferences": {"style": "conflict"},
    }

    with ThreadPoolExecutor(max_workers=2) as pool:
        conflict = pool.submit(
            lambda: json.loads(
                tools.handle_preferences_resolve(conflict_args, agent_id="agent-a")
            )
        )
        assert conflict_waiting.wait(timeout=2)
        winner = pool.submit(
            lambda: json.loads(
                tools.handle_preferences_resolve(winner_args, agent_id="agent-a")
            )
        )
        winner_result = winner.result(timeout=2)
        release_conflict.set()
        conflict_result = conflict.result(timeout=2)

    checkpoint = tools.state.get(winner_result["workflow_id"], "agent-a")
    assert winner_result["ok"] is True
    assert conflict_result["reason_code"] == "workflow_unavailable"
    assert checkpoint["resolve_request"] == tools._interactive_request_identity(
        winner_args, winner_args["task_id"]
    )


def test_interactive_resolve_concurrent_same_identity_is_idempotent(
    isolated_video_home,
):
    args = {
        "task_id": "concurrent-repeat",
        "scene": "travel",
        "preferences": {"style": "steady"},
        "silent": True,
    }
    start = threading.Barrier(3)

    def invoke():
        start.wait(timeout=2)
        return json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke) for _ in range(2)]
        start.wait(timeout=2)
        results = [future.result(timeout=2) for future in futures]

    assert all(result["ok"] is True for result in results)
    assert len({result["workflow_id"] for result in results}) == 1
    assert results[0]["preferences"] == results[1]["preferences"]


def test_interactive_legacy_checkpoint_is_adopted_without_resetting_progress(
    isolated_video_home,
):
    args = {
        "task_id": "legacy-interactive",
        "scene": "travel",
        "preferences": {"style": "vintage"},
    }
    resolved = preferences.resolve(
        "agent-a", args["scene"], args["preferences"], silent=False
    )
    workflow_id = state.workflow_id(args["task_id"], "agent-a")
    state.update(
        workflow_id,
        "agent-a",
        {
            "task_id": args["task_id"],
            "scene": resolved["scene"],
            "preferences": resolved["preferences"],
            "preference_sources": resolved["sources"],
            "source_paths": ["/legacy/source.mov"],
            "object_keys": ["assets/legacy-source"],
            "project_id": "project-legacy",
            "status": "completed",
        },
    )

    adopted = json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))
    checkpoint = state.get(workflow_id, "agent-a")

    assert adopted["ok"] is True
    assert checkpoint["resolve_request"] == tools._interactive_request_identity(
        args, args["task_id"]
    )
    assert checkpoint["source_paths"] == ["/legacy/source.mov"]
    assert checkpoint["object_keys"] == ["assets/legacy-source"]
    assert checkpoint["project_id"] == "project-legacy"
    assert checkpoint["status"] == "completed"


def test_interactive_legacy_migration_rejects_matching_proactive_checkpoint(
    isolated_video_home,
):
    args = {
        "task_id": "legacy-proactive-boundary",
        "scene": "travel",
        "preferences": {"style": "vintage"},
    }
    resolved = preferences.resolve(
        "agent-a", args["scene"], args["preferences"], silent=False
    )
    workflow_id = state.workflow_id(args["task_id"], "agent-a")
    state.update(
        workflow_id,
        "agent-a",
        {
            "task_id": args["task_id"],
            "scene": resolved["scene"],
            "preferences": resolved["preferences"],
            "preference_sources": resolved["sources"],
            "source_paths": ["/proactive/source.mov"],
            "object_keys": ["assets/proactive-source"],
            "project_id": "project-proactive",
            "proactive": True,
            "status": "completed",
        },
    )
    before = state.get(workflow_id, "agent-a")

    result = json.loads(tools.handle_preferences_resolve(args, agent_id="agent-a"))

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None
    assert state.get(workflow_id, "agent-a") == before


def test_upload_lock_rejects_a_symlinked_bucket_before_media_work(
    isolated_video_home,
    monkeypatch,
    tmp_path,
):
    workflow_id = state.workflow_id("upload-lock-symlink", "agent-a")
    state.update(workflow_id, "agent-a", {"task_id": "upload-lock-symlink"})
    bucket = (
        int.from_bytes(hashlib.sha256(workflow_id.encode()).digest()[:2], "big")
        % state.UPLOAD_LOCK_BUCKETS
    )
    lock_path = (
        isolated_video_home[0] / "video_edit" / "agent-a" / f".upload-{bucket:02d}.lock"
    )
    outside = tmp_path / "outside-lock"
    outside.write_text("sentinel", encoding="utf-8")
    lock_path.symlink_to(outside)
    monkeypatch.setattr(
        normalizer,
        "normalize_files",
        lambda *_args, **_kwargs: pytest.fail("symlinked lock must fail closed"),
    )
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail("symlinked lock must fail closed"),
    )

    result = json.loads(
        tools.handle_upload_assets({"workflow_id": workflow_id}, agent_id="agent-a")
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_concurrent_normalized_upload_reuses_completed_checkpoint(
    isolated_video_home,
    monkeypatch,
    tmp_path,
):
    source = isolated_video_home[1] / "agent-a" / "source.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"source")
    workflow_id = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "concurrent-normalized"}, agent_id="agent-a"
        )
    )["workflow_id"]
    normalizing = threading.Event()
    release_normalizer = threading.Event()
    normalized_calls: list[str] = []
    upload_calls: list[list[str]] = []

    def slow_normalize(files, workflow):
        normalized_calls.append(workflow)
        normalizing.set()
        assert release_normalizer.wait(timeout=2)
        output = tmp_path / "normalized.mp4"
        output.write_bytes(b"normalized")
        return [output]

    def upload(files, **_kwargs):
        upload_calls.append([path.name for path in files])
        return {"data": {"uploads": [{"object_key": "assets/source"}]}}

    monkeypatch.setattr(normalizer, "generation", lambda: "test-generation")
    monkeypatch.setattr(normalizer, "normalize_files", slow_normalize)
    monkeypatch.setattr(client, "upload", upload)
    args = {"workflow_id": workflow_id, "files": [str(source)], "normalize": True}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            lambda: json.loads(tools.handle_upload_assets(args, agent_id="agent-a"))
        )
        assert normalizing.wait(timeout=2)
        second = pool.submit(
            lambda: json.loads(tools.handle_upload_assets(args, agent_id="agent-a"))
        )
        time.sleep(0.05)
        assert not second.done()
        release_normalizer.set()
        results = [first.result(timeout=2), second.result(timeout=2)]

    assert normalized_calls == [workflow_id]
    assert upload_calls == [["normalized.mp4"]]
    assert sorted(bool(result.get("reused")) for result in results) == [False, True]
    assert all(result["uploaded"] == 1 for result in results)
    assert state.get(workflow_id, "agent-a")["object_keys"] == ["assets/source"]


@pytest.mark.skipif(
    not hasattr(multiprocessing, "get_context"), reason="no process context"
)
def test_upload_lock_blocks_a_second_process(isolated_video_home):
    context = multiprocessing.get_context("fork")
    workflow_id = state.workflow_id("cross-process-lock", "agent-a")
    entered = context.Event()
    release = context.Event()
    child = context.Process(
        target=_hold_upload_lock,
        args=(workflow_id, "agent-a", entered, release),
    )
    child.start()
    assert entered.wait(timeout=2)
    acquired = threading.Event()

    def wait_for_lock():
        with state.upload_lock(workflow_id, "agent-a"):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter = pool.submit(wait_for_lock)
        assert not acquired.wait(timeout=0.1)
        release.set()
        assert acquired.wait(timeout=2)
        waiter.result(timeout=2)
    child.join(timeout=2)

    assert child.exitcode == 0
