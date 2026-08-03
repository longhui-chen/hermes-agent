from __future__ import annotations

import json
import threading
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import model_tools  # noqa: E402
from agent import secret_scope  # noqa: E402
from agent.pet import store  # noqa: E402
from agent.pet.generate import HatchResult  # noqa: E402
from tools import desktop_pet_creator_tool as pet_tool  # noqa: E402


SESSION_ID = "zettlab:local:agent-a:session-flow"
TASK_ID = "task-flow"


def _call(args: dict) -> dict:
    return json.loads(
        model_tools.handle_function_call(
            function_name="desktop_pet_creator",
            function_args=args,
            task_id=TASK_ID,
            session_id=SESSION_ID,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
        )
    )


def test_desktop_pet_candidate_hatch_export_flow(monkeypatch, tmp_path):
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

    draft_source = tmp_path / "draft-source.png"
    Image.new("RGBA", (96, 96), (80, 120, 220, 255)).save(draft_source)

    def fake_drafts(
        concept,
        *,
        n=3,
        style="auto",
        reference_images=None,
        provider=None,
        on_draft=None,
        is_cancelled=None,
    ):
        del concept, style, reference_images, provider, on_draft
        assert not is_cancelled()
        return [draft_source for _ in range(n)]

    def fake_hatch(
        *,
        base_image,
        slug,
        display_name="",
        description="",
        concept="",
        style="auto",
        on_progress=None,
        provider=None,
        is_cancelled=None,
        staging_dir=None,
    ):
        del base_image, concept, style, provider, on_progress
        assert not is_cancelled()
        root = Path(staging_dir)
        pet_dir = root / "blue-byte"
        pet_dir.mkdir()
        spritesheet = pet_dir / "spritesheet.webp"
        Image.new("RGBA", (1536, 1872), (80, 120, 220, 0)).save(
            spritesheet,
            format="WEBP",
            lossless=True,
        )
        (pet_dir / "pet.json").write_text(
            json.dumps(
                {
                    "id": "blue-byte",
                    "displayName": display_name,
                    "description": description,
                    "spritesheetPath": "spritesheet.webp",
                    "createdBy": "generator",
                }
            ),
            encoding="utf-8",
        )
        return HatchResult(
            slug="blue-byte",
            display_name=display_name,
            spritesheet=spritesheet,
            states=["idle", "running-right", "running-left", "waving"],
            validation={"ok": True},
        )

    monkeypatch.setattr("agent.pet.generate.generate_base_drafts", fake_drafts)
    monkeypatch.setattr("agent.pet.generate.hatch_pet", fake_hatch)

    try:
        drafted = _call(
            {
                "action": "draft",
                "concept": "an original tiny blue robot fox",
                "count": 2,
            }
        )
        assert drafted["success"] is True
        candidate_id = drafted["drafts"][1]["candidate_id"]

        selected = _call(
            {
                "action": "select",
                "token": drafted["token"],
                "candidate_id": candidate_id,
            }
        )
        assert selected["selected_candidate_id"] == candidate_id

        hatched = _call(
            {
                "action": "hatch",
                "token": drafted["token"],
                "candidate_id": candidate_id,
                "name": "Blue Byte",
                "description": "A friendly original robot fox.",
            }
        )
        assert hatched["success"] is True
        assert hatched["preview"]["candidate_id"] == candidate_id
        assert Path(hatched["preview"]["path"]).is_file()
        assert store.load_pet("blue-byte") is None
        assert store.installed_pets() == []

        exported = _call({"action": "export", "token": drafted["token"]})
        assert exported["success"] is True
        assert exported["filename"] == "blue-byte.desktop-pet.zip"
        package = Path(exported["path"])
        assert package.read_bytes().startswith(b"PK\x03\x04")
        with zipfile.ZipFile(package) as archive:
            assert set(archive.namelist()) == {
                "blue-byte/pet.json",
                "blue-byte/spritesheet.webp",
            }

        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
        from tools.registry import registry

        entry = registry.get_entry("desktop_pet_creator")
        assert entry is not None
        assert entry.toolset == "desktop_pet"
        assert "desktop_pet_creator" not in _HERMES_CORE_TOOLS
        assert TOOLSETS["desktop_pet"]["tools"] == ["desktop_pet_creator"]
        assert "desktop_pet_creator" in TOOLSETS["hermes-zet-agent"]["tools"]
        assert "desktop_pet_creator" not in TOOLSETS["hermes-cli"]["tools"]
        assert "desktop_pet_creator" not in TOOLSETS["hermes-cron"]["tools"]
    finally:
        pet_tool._reset_state_for_tests()
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_desktop_pet_draft_flow_propagates_profile_secret_scope(monkeypatch, tmp_path):
    from agent.pet.generate import imagegen

    home = tmp_path / "hermes"
    output_root = tmp_path / "files" / "agents" / "data" / "agent-a" / "output"
    home.mkdir()
    output_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())

    def fake_generate(
        prompt,
        *,
        n=1,
        reference_images=None,
        provider=None,
        prefix="pet",
        aspect_ratio="square",
    ):
        del prompt, n, reference_images, provider, aspect_ratio
        assert secret_scope.get_secret("PET_TEST_PROFILE") == "agent-a-secret"
        path = tmp_path / f"{prefix}-{threading.get_ident()}.png"
        Image.new("RGBA", (96, 96), (80, 120, 220, 255)).save(path)
        return [path]

    monkeypatch.setattr(imagegen, "generate", fake_generate)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope(
        {
            "ZET_AGENT_ID": "agent-a",
            "ZET_AGENT_OUTPUT_ROOT": str(output_root),
            "PET_TEST_PROFILE": "agent-a-secret",
        }
    )
    pet_tool._reset_state_for_tests()
    try:
        drafted = _call(
            {
                "action": "draft",
                "concept": "an original scoped robot pet",
                "count": 2,
            }
        )
    finally:
        pet_tool._reset_state_for_tests()
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert drafted["success"] is True
    assert len(drafted["drafts"]) == 2
