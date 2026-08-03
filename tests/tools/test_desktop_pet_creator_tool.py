from __future__ import annotations

import json
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
