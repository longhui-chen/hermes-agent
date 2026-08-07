"""Tests for pet generation: deterministic atlas ops, store register, orchestration.

No network/API calls — image generation is mocked with synthetic strips so the
whole pipeline (segmentation → compose → validate → register → adopt) is
exercised hermetically.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("HERMES_RUN_SLOW_PET_TESTS") != "1",
    reason=(
        "pet generation image-processing suite is opt-in; run with "
        "HERMES_RUN_SLOW_PET_TESTS=1 scripts/run_tests.sh tests/agent/test_pet_generate.py"
    ),
)

from agent.pet.generate import atlas

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402


def _strip(n_blobs: int, *, transparent: bool = True, bg=(0, 255, 0, 255), size=(208, 208)) -> Image.Image:
    """A horizontal strip with *n_blobs* clearly-separated colored ellipses."""
    w = size[0] * n_blobs
    h = size[1]
    base = (0, 0, 0, 0) if transparent else bg
    img = Image.new("RGBA", (w, h), base)
    draw = ImageDraw.Draw(img)
    for i in range(n_blobs):
        cx = i * size[0] + size[0] // 2
        cy = h // 2
        r = size[0] // 3
        color = (40 + i * 30 % 200, 80, 200 - i * 20 % 180, 255)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    return img


# ───────────────────────── frame extraction ─────────────────────────


def test_extract_strip_frames_transparent_returns_centered_cells():
    frames = atlas.extract_strip_frames(_strip(6), 6)
    assert len(frames) == 6
    for frame in frames:
        assert frame.size == (atlas.CELL_WIDTH, atlas.CELL_HEIGHT)
        # Background corners must be transparent.
        assert frame.getpixel((0, 0))[3] == 0
        # Something is drawn.
        assert frame.getchannel("A").getextrema()[1] > 0




def test_remove_background_defringes_antialiased_edge():
    # The contaminated antialiased ring where sprite meets backdrop survives the
    # key (it's a blend, too far from pure magenta). Defringe shaves that 1px ring:
    # the keyed silhouette comes back eroded ~1px on every side, core intact.
    img = Image.new("RGBA", (200, 200), (255, 0, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((50, 50, 149, 149), fill=(40, 200, 60, 255))  # 100x100 green
    keyed = atlas.remove_background(img)
    bbox = keyed.getbbox()
    assert bbox is not None
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    assert 96 <= w <= 99 and 96 <= h <= 99  # ~1px shaved per side
    assert keyed.getpixel((100, 100))[3] > 0  # core intact






















# ───────────────────────── atlas compose / validate ─────────────────────────


def _frames_for_all_states() -> dict[str, list]:
    out: dict[str, list] = {}
    for state, _row, count in atlas.ROW_SPECS:
        out[state] = atlas.extract_strip_frames(_strip(count), count)
    return out










def test_validate_atlas_rejects_postage_stamp_sprite():
    sheet = Image.new("RGBA", (atlas.ATLAS_WIDTH, atlas.ATLAS_HEIGHT), (0, 0, 0, 0))
    frame = Image.new("RGBA", (atlas.CELL_WIDTH, atlas.CELL_HEIGHT), (0, 0, 0, 0))
    ImageDraw.Draw(frame).rectangle((86, 174, 106, 201), fill=(220, 240, 255, 255))

    for _state, row, count in atlas.ROW_SPECS:
        for col in range(count):
            sheet.alpha_composite(frame, (col * atlas.CELL_WIDTH, row * atlas.CELL_HEIGHT))

    result = atlas.validate_atlas(sheet)

    assert not result["ok"]
    assert any("too small" in e for e in result["errors"])








def test_normalize_cells_uses_consistent_pose_scale_for_motion_rows():
    # A jump row needs a taller union crop than idle, but the pet itself should
    # not shrink just because the motion envelope is taller.
    idle = Image.new("RGBA", (160, 180), (0, 0, 0, 0))
    jump_low = Image.new("RGBA", (160, 180), (0, 0, 0, 0))
    jump_high = Image.new("RGBA", (160, 180), (0, 0, 0, 0))
    ImageDraw.Draw(idle).rectangle((50, 80, 110, 160), fill=(80, 120, 220, 255))
    ImageDraw.Draw(jump_low).rectangle((50, 80, 110, 160), fill=(220, 120, 80, 255))
    ImageDraw.Draw(jump_high).rectangle((50, 60, 110, 140), fill=(220, 120, 80, 255))

    normalized = atlas.normalize_cells({"idle": [idle], "jumping": [jump_low, jump_high]})
    idle_box = normalized["idle"][0].getbbox()
    jump_box = normalized["jumping"][0].getbbox()

    assert idle_box is not None
    assert jump_box is not None
    idle_h = idle_box[3] - idle_box[1]
    jump_h = jump_box[3] - jump_box[1]
    assert abs(idle_h - jump_h) <= 8


# ───────────────────────── store register / adopt ─────────────────────────


def test_slugify_and_unique_slug():
    from agent.pet import store

    assert store.slugify("My Cool Pet!") == "my-cool-pet"
    assert store.slugify("   ") == "pet"
    first = store.unique_slug("Robo")
    (store.pets_dir() / first).mkdir(parents=True)
    assert store.unique_slug("Robo") == "robo-2"


def test_register_local_pet_appears_and_is_adoptable():
    from agent.pet import store

    sheet = atlas.compose_atlas(_frames_for_all_states())
    pet = store.register_local_pet(sheet, slug="Sparky", display_name="Sparky", description="zappy")
    assert pet.slug == "sparky"
    assert pet.exists
    assert any(p.slug == "sparky" for p in store.installed_pets())

    # install_pet returns the on-disk pet without ever hitting the manifest.
    adopted = store.install_pet("sparky")
    assert adopted.slug == "sparky"
    assert adopted.display_name == "Sparky"


def test_register_local_pet_is_generated_and_exports_zip():
    import io
    import zipfile

    from agent.pet import store

    sheet = atlas.compose_atlas(_frames_for_all_states())
    store.register_local_pet(sheet, slug="zippy", display_name="Zippy")
    assert store.load_pet("zippy").generated is True  # createdBy=generator

    filename, data = store.export_pet("zippy")
    assert filename == "zippy.zip"
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert "zippy/pet.json" in names
    assert any(n.startswith("zippy/spritesheet") for n in names)






# ───────────────────────── orchestration (mocked imagegen) ─────────────────────────




def test_generate_base_drafts_hardens_opaque_background(monkeypatch, tmp_path):
    """A provider that ignores background=transparent still yields a cutout."""
    from agent.pet.generate import imagegen, orchestrate

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        # Solid-green backdrop with a blob — i.e. the provider painted a backdrop.
        p = tmp_path / f"{prefix}_opaque.png"
        _strip(1, transparent=False, bg=(0, 255, 0, 255)).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    drafts = orchestrate.generate_base_drafts("a fox", n=1)
    assert len(drafts) == 1

    with Image.open(drafts[0]) as out:
        rgba = out.convert("RGBA")
    # The keyed backdrop is now transparent (corner pixel fully see-through).
    assert rgba.getpixel((0, 0))[3] == 0
    # The pet blob in the center is still opaque.
    assert rgba.getpixel((rgba.width // 2, rgba.height // 2))[3] > 0


def test_hatch_pet_propagates_secret_scope_to_generation_workers(monkeypatch, tmp_path):
    from agent import secret_scope
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate

    base = tmp_path / "base.png"
    _strip(1).save(base)

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        del prompt, n, reference_images, provider, aspect_ratio
        assert secret_scope.get_secret("PET_TEST_PROFILE") == "profile-a"
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        path = tmp_path / f"{prefix}.png"
        _strip(count).save(path)
        return [path]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)
    monkeypatch.setattr(store, "pets_dir", lambda: tmp_path / "pets")

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope_token = secret_scope.set_secret_scope({"PET_TEST_PROFILE": "profile-a"})
    try:
        result = orchestrate.hatch_pet(
            base_image=base,
            slug="scoped-pet",
            concept="a scoped robot pet",
        )
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert result.slug == "scoped-pet"
    assert result.validation["ok"] is True


def test_hatch_pet_generates_animation_rows_one_at_a_time(monkeypatch, tmp_path):
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate

    base = tmp_path / "base.png"
    _strip(1).save(base)
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_generate(
        prompt,
        *,
        n=1,
        reference_images=None,
        provider=None,
        prefix="pet",
        aspect_ratio="square",
    ):
        nonlocal active, max_active
        del prompt, n, reference_images, provider, aspect_ratio
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.02)
            state = prefix.replace("pet_row_", "")
            count = atlas_mod.FRAME_COUNTS.get(state, 6)
            path = tmp_path / f"{prefix}.png"
            _strip(count).save(path)
            return [path]
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)
    monkeypatch.setattr(store, "pets_dir", lambda: tmp_path / "pets")

    result = orchestrate.hatch_pet(
        base_image=base,
        slug="bounded-pet",
        concept="a bounded robot pet",
    )

    assert result.validation["ok"] is True
    assert max_active == orchestrate._MAX_PARALLEL_HATCH_ROWS == 1


def test_hatch_pet_retry_reuses_completed_animation_rows(monkeypatch, tmp_path):
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate
    from agent.pet.generate.imagegen import GenerationError

    base = tmp_path / "base.png"
    _strip(1).save(base)
    cache = tmp_path / "row-cache"
    cache.mkdir()
    calls: Counter[str] = Counter()
    allow_running_right = False

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
        state = prefix.replace("pet_row_", "")
        calls[state] += 1
        if state == "running-right" and not allow_running_right:
            raise GenerationError("temporary upstream failure")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        path = tmp_path / f"{prefix}.png"
        _strip(count).save(path)
        return [path]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)
    monkeypatch.setattr(store, "pets_dir", lambda: tmp_path / "pets")

    with pytest.raises(GenerationError, match="running-right"):
        orchestrate.hatch_pet(
            base_image=base,
            slug="resumable-pet",
            concept="a resumable robot pet",
            row_cache_dir=cache,
        )

    generated_states = {
        state for state, _, _ in atlas_mod.ROW_SPECS if state != "running-left"
    }
    assert calls["running-right"] == orchestrate._ROW_GEN_ATTEMPTS
    assert all(calls[state] == 1 for state in generated_states - {"running-right"})

    allow_running_right = True
    result = orchestrate.hatch_pet(
        base_image=base,
        slug="resumable-pet",
        concept="a resumable robot pet",
        row_cache_dir=cache,
    )

    assert result.validation["ok"] is True
    assert calls["running-right"] == orchestrate._ROW_GEN_ATTEMPTS + 1
    assert all(calls[state] == 1 for state in generated_states - {"running-right"})
    assert {path.name for path in cache.iterdir()} == generated_states


def test_hatch_pet_reports_only_durable_rows_as_ready(monkeypatch, tmp_path):
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate

    base = tmp_path / "base.png"
    _strip(1).save(base)
    cache = tmp_path / "row-cache"
    cache.mkdir()

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
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        path = tmp_path / f"{prefix}.png"
        _strip(count).save(path)
        return [path]

    real_save = orchestrate._save_cached_row

    def fake_save(cache_root, state, frames):
        if state == "waving":
            raise OSError("cache unavailable")
        real_save(cache_root, state, frames)

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)
    monkeypatch.setattr(orchestrate, "_save_cached_row", fake_save)
    monkeypatch.setattr(store, "pets_dir", lambda: tmp_path / "pets")
    events: list[tuple[str, str]] = []

    result = orchestrate.hatch_pet(
        base_image=base,
        slug="durable-pet",
        concept="a durable robot pet",
        row_cache_dir=cache,
        on_progress=lambda event, detail: events.append((event, detail)),
    )

    assert result.validation["ok"] is True
    assert ("row-ready", "idle") in events
    assert ("row-ready", "waving") not in events


def test_cached_row_rejects_small_compressed_image_with_large_decoded_size(tmp_path):
    import json

    from agent.pet.generate import orchestrate

    cache = tmp_path / "row-cache"
    row = cache / "idle"
    row.mkdir(parents=True)
    Image.new("RGBA", (3000, 3000), (0, 0, 0, 0)).save(row / "frame-0.png")
    (row / "complete.json").write_text(
        json.dumps({"version": 1, "state": "idle", "count": 1}),
        encoding="utf-8",
    )

    assert (row / "frame-0.png").stat().st_size < orchestrate._MAX_CACHED_FRAME_BYTES
    assert orchestrate._load_cached_row(cache, "idle", 1) is None


def test_hatch_pixel_budget_covers_all_generated_rows():
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import orchestrate

    generated_row_count = len(atlas_mod.ROW_SPECS) - 1
    assert orchestrate._MAX_HATCH_FRAME_PIXELS == (
        orchestrate._MAX_CACHED_ROW_PIXELS * generated_row_count
    )


def test_hatch_pet_end_to_end(monkeypatch, tmp_path):
    from agent.pet import store
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate

    base = tmp_path / "base.png"
    _strip(1).save(base)

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        # Return a synthetic row strip; frame count is inferable from the spec.
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        p = tmp_path / f"{prefix}.png"
        _strip(count).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    events: list[tuple[str, str]] = []
    result = orchestrate.hatch_pet(
        base_image=base,
        slug="mocky",
        display_name="Mocky",
        description="a test pet",
        concept="a fox",
        on_progress=lambda ev, detail: events.append((ev, detail)),
    )

    assert result.slug == "mocky"
    assert result.validation["ok"]
    assert set(result.states) == {s for s, _, _ in atlas_mod.ROW_SPECS}
    assert ("compose", "") in events
    # The pet is on disk and adoptable.
    assert store.load_pet("mocky").exists


def test_stage_hatched_pet_writes_package_without_installing(monkeypatch, tmp_path):
    import json

    from agent.pet import store
    from agent.pet.generate import orchestrate

    installed = tmp_path / "installed"
    staging = tmp_path / "staging"
    installed.mkdir()
    staging.mkdir()
    monkeypatch.setattr(store, "pets_dir", lambda: installed)
    sheet = atlas.compose_atlas(_frames_for_all_states())

    slug, display_name, spritesheet = orchestrate._stage_hatched_pet(
        sheet,
        staging_dir=staging,
        slug="Blue Byte",
        display_name="Blue Byte",
        description="An original test pet.",
    )

    metadata = json.loads((spritesheet.parent / "pet.json").read_text())
    assert slug == "blue-byte"
    assert display_name == "Blue Byte"
    assert spritesheet.is_file()
    assert metadata["id"] == "blue-byte"
    assert metadata["spritesheetPath"] == "spritesheet.webp"
    assert list(installed.iterdir()) == []


def test_hatch_pet_retries_timeout_three_times_without_idle_fallback(monkeypatch, tmp_path):
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate
    from agent.pet.generate.imagegen import GenerationError

    base = tmp_path / "base.png"
    _strip(1).save(base)
    cache = tmp_path / "row-cache"
    cache.mkdir()
    idle_calls = 0

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        nonlocal idle_calls
        if prefix == "pet_row_idle":
            idle_calls += 1
            raise GenerationError("media generation service timed out; retryable=false")
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        p = tmp_path / f"{prefix}.png"
        _strip(count).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    with pytest.raises(GenerationError, match=r"idle after 3 attempt\(s\).*timed out"):
        orchestrate.hatch_pet(
            base_image=base,
            slug="strict-idle",
            concept="a fox",
            row_cache_dir=cache,
        )

    assert idle_calls == orchestrate._ROW_GEN_ATTEMPTS == 3


def test_hatch_pet_rejects_missing_required_animation_rows(monkeypatch, tmp_path):
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate
    from agent.pet.generate.imagegen import GenerationError

    base = tmp_path / "base.png"
    _strip(1).save(base)

    failed_calls = 0

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        nonlocal failed_calls
        if prefix == "pet_row_failed":
            failed_calls += 1
            raise GenerationError("bad row")
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        p = tmp_path / f"{prefix}.png"
        _strip(count).save(p)
        return [p]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    with pytest.raises(GenerationError, match=r"failed after 3 attempt\(s\)"):
        orchestrate.hatch_pet(base_image=base, slug="broken", concept="a fox")

    assert failed_calls == orchestrate._ROW_GEN_ATTEMPTS == 3


def test_hatch_pet_does_not_retry_explicit_nonretryable_row_error(monkeypatch, tmp_path):
    from agent.pet.generate import atlas as atlas_mod
    from agent.pet.generate import imagegen, orchestrate
    from agent.pet.generate.imagegen import GenerationError

    base = tmp_path / "base.png"
    _strip(1).save(base)
    failed_calls = 0

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        nonlocal failed_calls
        if prefix == "pet_row_failed":
            failed_calls += 1
            raise GenerationError("content policy rejection; retryable=false")
        state = prefix.replace("pet_row_", "")
        count = atlas_mod.FRAME_COUNTS.get(state, 6)
        path = tmp_path / f"{prefix}.png"
        _strip(count).save(path)
        return [path]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)

    with pytest.raises(GenerationError, match=r"failed after 1 attempt\(s\)"):
        orchestrate.hatch_pet(base_image=base, slug="blocked", concept="a fox")

    assert failed_calls == 1




class _FakeImgProvider:
    def __init__(self, name, available=True):
        self.name = name
        self._available = available

    def is_available(self):
        return self._available




def test_list_sprite_providers_marks_default(monkeypatch):
    """Lists only available ref-capable backends, flagging the default pick."""
    from agent.pet.generate import imagegen

    registry = {"openai": _FakeImgProvider("openai"), "nous": _FakeImgProvider("nous")}
    monkeypatch.setattr(imagegen, "_discover", lambda: None)
    monkeypatch.setattr("agent.image_gen_registry.get_active_provider", lambda: registry["openai"])
    monkeypatch.setattr("agent.image_gen_registry.get_provider", lambda name: registry.get(name))

    listed = imagegen.list_sprite_providers()
    names = {p["name"] for p in listed}
    assert names == {"openai", "nous"}
    # Every entry carries a display label (no quality note — all backends are equal).
    assert all(p["label"] for p in listed)
    assert all("note" not in p for p in listed)
    assert [p["name"] for p in listed if p["default"]] == ["openai"]
    # Listed in preference order: Nous Portal before OpenAI.
    assert [p["name"] for p in listed] == ["nous", "openai"]


def test_generate_retries_without_transparent_background(monkeypatch, tmp_path):
    """A model that rejects background=transparent still produces images."""
    from agent.pet.generate import imagegen

    saved = tmp_path / "img.png"
    _strip(1).save(saved)
    calls: list[dict] = []

    class FakeProvider:
        def generate(self, prompt, **kwargs):
            calls.append(kwargs)
            if kwargs.get("background") == "transparent":
                return {"success": False, "error": "Transparent background is not supported for this model."}
            return {"success": True, "image": str(saved)}

    sprite = imagegen.SpriteProvider(name="openai", provider=FakeProvider(), supports_references=False)

    out = imagegen.generate("a fox", n=2, provider=sprite)
    assert len(out) == 2
    # First variant probes transparent (rejected) then retries opaque; the second
    # variant skips the transparent probe entirely.
    backgrounds = [c.get("background") for c in calls]
    assert backgrounds == ["transparent", None, None]
