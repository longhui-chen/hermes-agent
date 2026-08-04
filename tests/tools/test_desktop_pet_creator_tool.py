from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from agent import secret_scope  # noqa: E402
from gateway.session_context import (  # noqa: E402
    pop_current_turn_reference_image,
    push_current_turn_reference_image,
)
from tools import desktop_pet_creator_tool as pet_tool  # noqa: E402


SESSION_ID = "zettlab:local:agent-a:session-a"
TASK_ID = "task-a"
TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.fixture
def pet_env(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    output_root = tmp_path / "files" / "agents" / "data" / "agent-a" / "output"
    home.mkdir()
    output_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope(
        {
            "ZET_AGENT_ID": "agent-a",
            "ZET_AGENT_OUTPUT_ROOT": str(output_root),
        }
    )
    pet_tool._reset_state_for_tests()
    try:
        yield {
            "home": home,
            "output_dir": output_root / "session-a",
        }
    finally:
        pet_tool._reset_state_for_tests()
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


def _fake_drafts(tmp_path: Path):
    captured = {}

    def generate(
        concept,
        *,
        n=3,
        style="auto",
        reference_images=None,
        provider=None,
        on_draft=None,
        is_cancelled=None,
    ):
        del concept, style, provider, on_draft
        captured["reference_images"] = reference_images
        paths = []
        for index in range(n):
            assert not is_cancelled()
            path = tmp_path / f"generated-{index}.png"
            Image.new("RGBA", (64, 64), (20 + index, 80, 160, 255)).save(path)
            paths.append(path)
        return paths

    generate.captured = captured
    return generate


def _call(args: dict) -> dict:
    return json.loads(
        pet_tool.desktop_pet_creator(
            args,
            task_id=TASK_ID,
            session_id=SESSION_ID,
        )
    )


def test_schema_never_exposes_paths_or_array_indexes():
    properties = pet_tool.DESKTOP_PET_CREATOR_SCHEMA["parameters"]["properties"]

    assert "reference_paths" not in properties
    assert "output_dir" not in properties
    assert "draft_index" not in properties
    assert "candidate_id" in properties


def test_status_creates_safe_session_output_directory(monkeypatch, pet_env):
    monkeypatch.setattr("agent.pet.generate.imagegen.list_sprite_providers", lambda: [])
    assert not pet_env["output_dir"].exists()

    status = _call({"action": "status"})

    assert status["success"] is True
    assert status["output_capable"] is True
    assert pet_env["output_dir"].is_dir()


def test_zettlab_generation_rejects_model_supplied_local_path_strings():
    from agent.pet.generate import imagegen

    class FakeProvider:
        def generate(self, prompt, **kwargs):
            raise AssertionError("provider must not receive an arbitrary local path")

    provider = imagegen.SpriteProvider(
        name="zettlab",
        provider=FakeProvider(),
        supports_references=True,
    )

    with pytest.raises(imagegen.GenerationError, match="current-turn media"):
        imagegen.generate(
            "an original desktop pet",
            reference_images=["/private/tmp/model-supplied.png"],
            provider=provider,
        )


def test_draft_uses_only_current_turn_context_and_manifest_survives_reset(
    monkeypatch, pet_env, tmp_path
):
    generate = _fake_drafts(tmp_path)
    monkeypatch.setattr("agent.pet.generate.generate_base_drafts", generate)
    context_token = push_current_turn_reference_image(TINY_PNG_DATA_URL)
    try:
        drafted = _call(
            {
                "action": "draft",
                "concept": "an original tiny blue robot fox",
                "count": 2,
            }
        )
    finally:
        pop_current_turn_reference_image(context_token)

    assert drafted["success"] is True
    assert drafted["status"] == "drafted"
    assert len(drafted["drafts"]) == 2
    assert generate.captured["reference_images"] == [TINY_PNG_DATA_URL]
    candidate_ids = [item["candidate_id"] for item in drafted["drafts"]]
    assert len(set(candidate_ids)) == 2
    assert all(Path(item["path"]).parent == pet_env["output_dir"] for item in drafted["drafts"])

    pet_tool._reset_state_for_tests()
    resumed = _call({"action": "status", "token": drafted["token"]})

    assert resumed["success"] is True
    assert [item["candidate_id"] for item in resumed["drafts"]] == candidate_ids


def test_refine_uses_private_candidate_and_keeps_active_set_bounded(
    monkeypatch, pet_env, tmp_path
):
    generate = _fake_drafts(tmp_path)
    monkeypatch.setattr("agent.pet.generate.generate_base_drafts", generate)
    drafted = _call(
        {
            "action": "draft",
            "concept": "an original desk companion",
            "count": 3,
        }
    )
    original_ids = [item["candidate_id"] for item in drafted["drafts"]]

    refined = _call(
        {
            "action": "refine",
            "token": drafted["token"],
            "candidate_id": original_ids[1],
            "instruction": "make the ears rounder",
        }
    )

    assert refined["success"] is True
    assert len(refined["drafts"]) == 1
    assert refined["drafts"][0]["candidate_id"] not in original_ids
    references = generate.captured["reference_images"]
    assert len(references) == 1
    assert isinstance(references[0], Path)
    assert references[0].parent.name == "candidates"

    resumed = _call({"action": "status", "token": drafted["token"]})
    assert len(resumed["drafts"]) == 3
    assert refined["drafts"][0]["candidate_id"] in {
        item["candidate_id"] for item in resumed["drafts"]
    }


def test_scope_mismatch_cannot_resume_task(monkeypatch, pet_env, tmp_path):
    monkeypatch.setattr(
        "agent.pet.generate.generate_base_drafts", _fake_drafts(tmp_path)
    )
    drafted = _call(
        {"action": "draft", "concept": "an original desk companion", "count": 1}
    )

    result = json.loads(
        pet_tool.desktop_pet_creator(
            {"action": "status", "token": drafted["token"]},
            task_id="task-b",
            session_id="zettlab:local:agent-a:session-b",
        )
    )

    assert result["success"] is False
    assert result["error"] == "task token is invalid or expired"


def test_concurrent_hatch_is_rejected_before_touching_task_storage(
    monkeypatch, pet_env, tmp_path
):
    monkeypatch.setattr(
        "agent.pet.generate.generate_base_drafts", _fake_drafts(tmp_path)
    )
    drafted = _call(
        {"action": "draft", "concept": "an original desk companion", "count": 1}
    )
    task_dir = pet_env["home"] / "cache" / "desktop-pet-creator" / drafted["token"]
    staging = task_dir / "hatched"
    stale_cache = task_dir / "hatch-rows" / ("a" * 32)
    staging.mkdir()
    stale_cache.mkdir(parents=True)
    staging_marker = staging / "active-hatch.txt"
    cache_marker = stale_cache / "complete.json"
    staging_marker.write_text("active", encoding="utf-8")
    cache_marker.write_text("active", encoding="utf-8")
    monkeypatch.setattr(
        pet_tool,
        "_reserve",
        lambda _token: (_ for _ in ()).throw(ValueError("desktop-pet task is already running")),
    )

    result = _call(
        {
            "action": "hatch",
            "token": drafted["token"],
            "candidate_id": drafted["drafts"][0]["candidate_id"],
            "name": "Blue Byte",
        }
    )

    assert result["success"] is False
    assert "already running" in result["error"]
    assert staging_marker.read_text(encoding="utf-8") == "active"
    assert cache_marker.read_text(encoding="utf-8") == "active"


def test_failed_rehatch_clears_stale_success_metadata(monkeypatch, pet_env, tmp_path):
    from agent.pet.generate.imagegen import GenerationError

    monkeypatch.setattr(
        "agent.pet.generate.generate_base_drafts", _fake_drafts(tmp_path)
    )
    drafted = _call(
        {"action": "draft", "concept": "an original desk companion", "count": 1}
    )
    token = drafted["token"]

    def seed_previous_hatch(manifest):
        manifest.update(
            {
                "status": "exported",
                "slug": "old-pet",
                "display_name": "Old Pet",
                "description": "stale",
                "pet_dir": "/tmp/old-pet",
                "spritesheet_path": "/tmp/old-pet/spritesheet.webp",
                "states": ["idle"],
                "preview": {"path": "/tmp/old-preview.webp"},
                "export": {"path": "/tmp/old-pet.desktop-pet.zip"},
            }
        )

    pet_tool._update_manifest(
        token,
        task_id=TASK_ID,
        session_id=SESSION_ID,
        update=seed_previous_hatch,
    )

    def fake_hatch(**kwargs):
        del kwargs
        raise GenerationError(
            "missing required animation row(s): failed; generation failures: "
            "failed after 3 attempt(s): media generation service timed out"
        )

    monkeypatch.setattr("agent.pet.generate.hatch_pet", fake_hatch)
    result = _call(
        {
            "action": "hatch",
            "token": token,
            "candidate_id": drafted["drafts"][0]["candidate_id"],
            "name": "Blue Byte",
        }
    )

    assert result["success"] is False
    assert "failed after 3 attempt(s)" in result["error"]
    status = _call({"action": "status", "token": token})
    assert status["status"] == "drafted"
    assert "failed after 3 attempt(s)" in status["last_error"]
    assert "preview" not in status
    assert "export" not in status


def test_manifest_progress_status_and_cancel_are_serialized(
    monkeypatch, pet_env, tmp_path
):
    monkeypatch.setattr(
        "agent.pet.generate.generate_base_drafts", _fake_drafts(tmp_path)
    )
    drafted = _call(
        {"action": "draft", "concept": "an original desk companion", "count": 1}
    )
    token = drafted["token"]
    errors: list[Exception] = []
    barrier = threading.Barrier(3)

    def write_progress() -> None:
        try:
            barrier.wait()
            for index in range(25):
                pet_tool._update_manifest(
                    token,
                    task_id=TASK_ID,
                    session_id=SESSION_ID,
                    update=lambda manifest, current=index: manifest.__setitem__(
                        "hatch_progress", {"completed_states": [f"state-{current}"]}
                    ),
                )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def read_status() -> None:
        try:
            barrier.wait()
            for _ in range(25):
                assert _call({"action": "status", "token": token})["success"] is True
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    writer = threading.Thread(target=write_progress)
    reader = threading.Thread(target=read_status)
    writer.start()
    reader.start()
    barrier.wait()
    cancelled = _call({"action": "cancel", "token": token})
    writer.join()
    reader.join()

    assert errors == []
    assert cancelled["status"] == "cancelled"
    final = _call({"action": "status", "token": token})
    assert final["status"] == "cancelled"
    assert final["hatch_progress"]["completed_states"] == ["state-24"]
    task_dir = pet_env["home"] / "cache" / "desktop-pet-creator" / token
    assert list(task_dir.glob(".manifest-*.part")) == []
