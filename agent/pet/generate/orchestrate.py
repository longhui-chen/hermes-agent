"""Pet generation orchestration — the base-draft → hatch flow.

Two steps, mirroring the UX across every surface:

1. :func:`generate_base_drafts` — a handful of prompt-only "what should this pet
   look like" variants. Cheap; the user picks one (or retries for a fresh set).
2. :func:`hatch_pet` — takes the chosen base and generates one grounded row
   strip per Hermes state, slices each into frames, composes the atlas, validates
   it, and writes the pet into the store.

Splitting it this way bounds cost (a few cheap base calls per round; eight
generated state rows plus one mirrored row happen only for the pet you keep)
and gives each UI a natural preview/loading point.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.pet.generate import atlas, imagegen, prompts
from agent.pet.generate.imagegen import GenerationError, SpriteProvider

logger = logging.getLogger(__name__)

# (event, detail) — e.g. ("row", "idle"), ("compose", ""), ("save", "<slug>").
ProgressFn = Callable[[str, str], None]

# Image generations are independent network calls. Drafts use a small bounded
# pool, while hatch rows stay at one active provider call so each completed state
# can be checkpointed before the next one starts.
_MAX_PARALLEL_DRAFT_GENERATIONS = 2
_MAX_PARALLEL_HATCH_ROWS = 1
# Retry a failed state within the same hatch call. The final attempt uses lenient
# equal-slot slicing so touching poses still get one recovery path.
_ROW_GEN_ATTEMPTS = 3
_REQUIRED_STATES = frozenset(state for state, _, _ in atlas.ROW_SPECS)
_ROW_CACHE_VERSION = 1
_MAX_CACHED_FRAME_BYTES = 4 * 1024 * 1024
_MAX_CACHED_ROW_BYTES = 16 * 1024 * 1024
_MAX_CACHED_TASK_BYTES = 64 * 1024 * 1024
_MAX_CACHED_FRAME_DIMENSION = 4096
_MAX_CACHED_ROW_PIXELS = 8_000_000
# The hatch may hold all eight generated rows before normalizing the atlas.
# Keep the aggregate bound consistent with the already-enforced per-row bound;
# running-left is mirrored only after this accounting pass.
_MAX_HATCH_FRAME_PIXELS = _MAX_CACHED_ROW_PIXELS * (len(atlas.ROW_SPECS) - 1)


@dataclass(frozen=True)
class HatchResult:
    """Outcome of a successful :func:`hatch_pet`."""

    slug: str
    display_name: str
    spritesheet: Path
    states: list[str]
    validation: dict


def _resolve_row_cache(row_cache_dir: str | Path | None) -> Path | None:
    """Resolve an existing, caller-owned cache directory without following links."""
    if row_cache_dir is None:
        return None
    root = Path(row_cache_dir).expanduser()
    if root.is_symlink() or not root.is_dir():
        raise GenerationError("pet row cache directory is unavailable")
    return root.resolve(strict=True)


def _load_cached_row(cache_root: Path | None, state: str, count: int) -> list | None:
    """Load one complete cached animation row; partial or unsafe entries miss."""
    if cache_root is None:
        return None
    from PIL import Image

    directory = cache_root / state
    marker = directory / "complete.json"
    try:
        if directory.is_symlink() or not directory.is_dir():
            return None
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 4096:
            return None
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata != {"version": _ROW_CACHE_VERSION, "state": state, "count": count}:
            return None

        frames = []
        total_bytes = 0
        total_pixels = 0
        for index in range(count):
            path = directory / f"frame-{index}.png"
            if path.is_symlink() or not path.is_file():
                return None
            size = path.stat().st_size
            total_bytes += size
            if size > _MAX_CACHED_FRAME_BYTES or total_bytes > _MAX_CACHED_ROW_BYTES:
                return None
            with Image.open(path) as opened:
                if (
                    opened.width <= 0
                    or opened.height <= 0
                    or opened.width > _MAX_CACHED_FRAME_DIMENSION
                    or opened.height > _MAX_CACHED_FRAME_DIMENSION
                ):
                    return None
                total_pixels += opened.width * opened.height
                if total_pixels > _MAX_CACHED_ROW_PIXELS:
                    return None
                frames.append(opened.convert("RGBA").copy())
        return frames
    except (OSError, ValueError, json.JSONDecodeError):
        logger.warning("pet hatch: ignoring invalid cached row %r", state)
        return None


def _save_cached_row(cache_root: Path | None, state: str, frames: list) -> None:
    """Atomically publish one complete animation row to the task-private cache."""
    if cache_root is None:
        return
    destination = cache_root / state
    temporary = cache_root / f".{state}.part"
    if destination.is_symlink() or temporary.is_symlink():
        raise GenerationError("pet row cache entry is unavailable")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(mode=0o700)
    try:
        total_bytes = 0
        total_pixels = 0
        for index, frame in enumerate(frames):
            if (
                frame.width <= 0
                or frame.height <= 0
                or frame.width > _MAX_CACHED_FRAME_DIMENSION
                or frame.height > _MAX_CACHED_FRAME_DIMENSION
            ):
                raise GenerationError("generated pet frame dimensions are invalid")
            total_pixels += frame.width * frame.height
            if total_pixels > _MAX_CACHED_ROW_PIXELS:
                raise GenerationError("generated pet row exceeds decoded pixel limits")
            path = temporary / f"frame-{index}.png"
            frame.convert("RGBA").save(path, format="PNG")
            size = path.stat().st_size
            total_bytes += size
            if size > _MAX_CACHED_FRAME_BYTES or total_bytes > _MAX_CACHED_ROW_BYTES:
                raise GenerationError("generated pet row exceeds cache limits")
        (temporary / "complete.json").write_text(
            json.dumps(
                {"version": _ROW_CACHE_VERSION, "state": state, "count": len(frames)},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        if _row_cache_bytes(cache_root) > _MAX_CACHED_TASK_BYTES:
            shutil.rmtree(destination, ignore_errors=True)
            raise GenerationError("desktop-pet row cache exceeds task size limits")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _row_cache_bytes(cache_root: Path) -> int:
    """Return bounded task-cache bytes without following directory symlinks."""
    total = 0
    for directory in cache_root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_symlink() or not path.is_file():
                continue
            total += path.stat().st_size
            if total > _MAX_CACHED_TASK_BYTES:
                return total
    return total


def _harden_transparency(path: Path) -> Path:
    """Key out any solid backdrop the provider painted; save as an RGBA PNG.

    ``background=transparent`` is requested on every call, but image models honor
    it inconsistently — some still paint a flat (often near-white) backdrop. We
    run the same chroma-key pass the row extractor uses so every base draft the
    user picks between (and the reference the rows are grounded on) is a clean
    cutout. Best-effort: a decode failure leaves the original untouched.
    """
    from PIL import Image

    try:
        with Image.open(path) as opened:
            keyed = atlas.remove_background(opened.convert("RGBA"))
        # Zero the RGB of any leftover semi-transparent edge pixels so a keyed
        # draft has no colored halo when composited on the dark UI.
        keyed = atlas._clear_transparent_rgb(keyed)
        out = path.with_suffix(".png")
        keyed.save(out, format="PNG")
        return out
    except Exception as exc:  # noqa: BLE001 - cosmetic; fall back to the raw image
        logger.debug("base draft transparency hardening failed for %s: %s", path, exc)
        return path


def generate_base_drafts(
    concept: str,
    *,
    n: int = 4,
    style: str = "auto",
    reference_images: list[str | Path] | None = None,
    provider: SpriteProvider | None = None,
    on_draft: Callable[[int, Path], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[Path]:
    """Generate *n* candidate base looks for *concept*; returns image paths.

    Each draft is hardened to a transparent cutout (see :func:`_harden_transparency`).
    Drafts are generated concurrently and *on_draft(index, path)* fires as each
    one finishes (not at the end) so callers can stream previews to the UI
    instead of leaving it blank until the whole batch is done.

    *is_cancelled*, when supplied, is polled cooperatively: a draft that hasn't
    started yet is skipped, and once it trips we stop staging/streaming further
    drafts and cancel any queued work (already-in-flight provider calls can't be
    hard-killed, but their results are dropped).
    """
    # A user reference image (e.g. their own pet) grounds every draft, so it
    # needs a reference-capable provider — same requirement as the row passes.
    refs = reference_images or None
    sprite = provider or imagegen.resolve_provider(require_references=bool(refs))
    cancelled = is_cancelled or (lambda: False)

    # Each draft is its own one-shot generation, run concurrently so the user
    # waits for one image, not N. A single draft failing must not sink the set.
    # Each gets a distinct variation nudge so the options aren't near-duplicates.
    logger.info("pet generate: drafting %d base looks for %r (style=%s)", n, concept, style)

    def _one(index: int) -> tuple[int, Path | None, str | None]:
        if cancelled():
            return index, None, None
        t0 = time.monotonic()
        variation = prompts.BASE_VARIATIONS[index % len(prompts.BASE_VARIATIONS)]
        prompt = prompts.build_base_prompt(concept, style=style, variation=variation)
        try:
            out = imagegen.generate(prompt, n=1, reference_images=refs, provider=sprite, prefix="pet_base")
        except Exception as exc:  # noqa: BLE001 - tolerate a single failed draft
            logger.warning("pet generate: draft %d failed after %.1fs: %s", index, time.monotonic() - t0, exc)
            return index, None, str(exc)
        if not out:
            logger.warning("pet generate: draft %d produced no image", index)
            return index, None, "the image provider returned no image"
        logger.info("pet generate: draft %d ready in %.1fs", index, time.monotonic() - t0)
        return index, _harden_transparency(out[0]), None

    workers = max(1, min(n, _MAX_PARALLEL_DRAFT_GENERATIONS))
    results: dict[int, Path] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_context().run, _one, i) for i in range(n)]
        # as_completed runs in *this* (the caller's) thread, so on_draft — and any
        # gateway event it emits — inherits the request's bound transport, unlike
        # the worker threads above.
        for fut in as_completed(futures):
            if cancelled():
                logger.info("pet generate: cancelled — dropping remaining drafts")
                for pending in futures:
                    pending.cancel()
                break
            index, path, err = fut.result()
            if path is None:
                if err:
                    errors.append(err)
                continue
            results[index] = path
            if on_draft is not None:
                try:
                    on_draft(index, path)
                except Exception as exc:  # noqa: BLE001 - progress is best-effort
                    logger.debug("on_draft callback failed: %s", exc)

    drafts = [results[i] for i in sorted(results)]
    if not drafts and not cancelled():
        # Surface *why* — every draft failed for a reason (a content-policy refusal
        # on a name like "minion", a provider/auth error, …); the most common one
        # is the representative cause. Far more useful than "no usable drafts".
        raise GenerationError(_drafts_failed_reason(errors))
    return drafts


def _drafts_failed_reason(errors: list[str]) -> str:
    """The representative reason a draft round produced nothing, humanized."""
    if not errors:
        return "image generation produced no usable drafts"
    from collections import Counter

    return _humanize_image_error(Counter(errors).most_common(1)[0][0])


def _humanize_image_error(error: str) -> str:
    """Turn a raw provider error into a friendly, actionable sentence.

    The big one is moderation: image models refuse trademarked characters and
    real people (e.g. "minion"), which reads as an opaque 400 otherwise.
    """
    low = error.lower()
    if any(s in low for s in ("moderation_blocked", "safety system", "content policy", "content_policy")):
        return (
            "The image provider blocked this prompt — its safety filter rejects "
            "trademarked characters and real people. Try an original description."
        )
    if any(s in low for s in ("api key", "unauthorized", "401", "auth")):
        return "The image provider rejected the request — check your API key in Settings → Providers."
    if "rate limit" in low or "429" in low:
        return "The image provider is rate-limiting — wait a moment and try again."
    # Otherwise the first line, trimmed of the noisy provider envelope.
    return error.splitlines()[0].strip()[:200]


def _row_error_is_retryable(error: Exception) -> bool:
    """Retry timeouts even when an upstream transport labels them non-retryable."""
    message = str(error).casefold()
    if "timed out" in message or "timeout" in message:
        return True
    return "retryable=false" not in message


def hatch_pet(
    *,
    base_image: str | Path,
    slug: str,
    display_name: str = "",
    description: str = "",
    concept: str = "",
    style: str = "auto",
    on_progress: ProgressFn | None = None,
    provider: SpriteProvider | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    staging_dir: str | Path | None = None,
    row_cache_dir: str | Path | None = None,
) -> HatchResult:
    """Turn an approved base image into a full Hermes pet.

    Generates a grounded row strip per state, extracts frames, composes +
    validates the atlas. By default the result is registered in the profile's
    pet store. When *staging_dir* is supplied, the package is written beneath
    that existing non-symlink directory without installing it. When
    *row_cache_dir* is supplied, each completed state is persisted there and
    reused by later retries. All nine animation states are required; a state that
    cannot be generated after its in-call retries fails the hatch.

    *is_cancelled*, when supplied, is polled cooperatively: rows that haven't
    started are skipped, queued rows are cancelled, and once every row is done we
    abort (raising :class:`GenerationError`) before composing/saving so a stopped
    hatch never writes a half-built pet.
    """
    base = Path(base_image)
    if not base.is_file():
        raise GenerationError(f"base image not found: {base}")

    sprite = provider or imagegen.resolve_provider(require_references=True)
    progress = on_progress or (lambda *_: None)
    cancelled = is_cancelled or (lambda: False)
    label = concept or display_name or slug
    row_cache = _resolve_row_cache(row_cache_dir)

    frames_by_state: dict[str, list] = {}
    decoded_frame_pixels = 0
    total_rows = len(atlas.ROW_SPECS)
    logger.info("pet hatch %r: generating %d animation rows", slug, total_rows)

    # Generate each grounded row through a bounded pool. The production limit is
    # deliberately one: finish and checkpoint a state before starting the next.
    def _gen_row(
        spec: tuple[str, int, int],
    ) -> tuple[str, list | None, bool, int, Exception | None]:
        state, _row, count = spec
        if cancelled():
            return state, None, False, 0, None
        cached = _load_cached_row(row_cache, state, count)
        if cached is not None:
            logger.info("pet hatch %r: row %r restored from cache", slug, state)
            return state, cached, True, 0, None
        t0 = time.monotonic()
        last_exc: Exception | None = None
        attempts_used = 0
        # Self-healing: a model occasionally returns a row whose poses are touching
        # (no clean gutters), which slices badly. We retry such rolls; only the
        # final attempt falls back to lenient ``auto`` slicing so a stubborn row
        # still yields *something* rather than dropping the whole row.
        for attempt in range(_ROW_GEN_ATTEMPTS):
            if cancelled():
                return state, None, False, attempts_used, None
            attempts_used = attempt + 1
            strict = attempt < _ROW_GEN_ATTEMPTS - 1
            try:
                strips = imagegen.generate(
                    prompts.build_row_prompt(state, count, label, style=style),
                    n=1,
                    reference_images=[base],
                    provider=sprite,
                    prefix=f"pet_row_{state}",
                    # Wider canvas → each frame gets real horizontal room, so winged
                    # poses keep a full, healthy size and still leave clean gutters.
                    aspect_ratio="landscape",
                )
                # ``components`` requires clean per-pose gutters (raises otherwise),
                # so a touching roll is rejected and regenerated; the last attempt
                # uses ``auto`` (equal-slot fallback, never raises). Raw (fit=False)
                # so normalize_cells registers the whole pet at once.
                method = "components" if strict else "auto"
                frames = atlas.extract_strip_frames(strips[0], count, method=method, fit=False)
                logger.info(
                    "pet hatch %r: row %r ready in %.1fs (attempt %d)",
                    slug, state, time.monotonic() - t0, attempt + 1,
                )
                durable = True
                if row_cache is not None:
                    try:
                        _save_cached_row(row_cache, state, frames)
                    except Exception as exc:  # noqa: BLE001 - current run can continue
                        durable = False
                        logger.warning(
                            "pet hatch %r: row %r cache write failed: %s", slug, state, exc
                        )
                return state, frames, durable, attempts_used, None
            except Exception as exc:  # noqa: BLE001 - retried; one bad row is tolerated
                last_exc = exc
                logger.warning(
                    "pet hatch %r: row %r attempt %d/%d failed: %s",
                    slug, state, attempt + 1, _ROW_GEN_ATTEMPTS, exc,
                )
                if not _row_error_is_retryable(exc):
                    break
        logger.warning(
            "pet hatch %r: row %r gave up after %.1fs: %s",
            slug, state, time.monotonic() - t0, last_exc,
        )
        return state, None, False, attempts_used, last_exc

    # running-left is derived by mirroring running-right (guaranteed-consistent
    # and one fewer generation), so we don't generate it directly.
    generated_specs = [spec for spec in atlas.ROW_SPECS if spec[0] != "running-left"]

    workers = max(1, min(len(generated_specs), _MAX_PARALLEL_HATCH_ROWS))
    done = 0
    row_failures: dict[str, tuple[int, Exception]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy_context().run, _gen_row, spec) for spec in generated_specs]
        # as_completed runs on the caller (request) thread, so progress events
        # emitted here inherit the request transport — unlike the worker threads.
        for fut in as_completed(futures):
            if cancelled():
                logger.info("pet hatch %r: cancelled — dropping remaining rows", slug)
                for pending in futures:
                    pending.cancel()
                break
            state, frames, durable, attempts_used, row_error = fut.result()
            done += 1
            progress("row", f"{state}:{done}:{total_rows}")
            if frames:
                decoded_frame_pixels += sum(frame.width * frame.height for frame in frames)
                if decoded_frame_pixels > _MAX_HATCH_FRAME_PIXELS:
                    raise GenerationError("generated pet animation exceeds decoded pixel limits")
                frames_by_state[state] = frames
                if durable:
                    progress("row-ready", state)
            elif row_error is not None:
                row_failures[state] = (attempts_used, row_error)

    if cancelled():
        raise GenerationError("hatch cancelled")

    # Derive running-left from the approved running-right row (per-frame mirror,
    # preserving order/timing). Missing running-right is rejected below; a pet
    # without its canonical walk cycle is a failed hatch, not a shippable mascot.
    right = frames_by_state.get("running-right")
    if right:
        done += 1
        progress("row", f"running-left:{done}:{total_rows}")
        frames_by_state["running-left"] = atlas.mirror_frames(right)
        logger.info("pet hatch %r: row 'running-left' mirrored from running-right", slug)
    else:
        logger.warning("pet hatch %r: no running-right to mirror; left walk left empty", slug)

    progress("compose", "")
    logger.info("pet hatch %r: composing atlas from %d states", slug, len(frames_by_state))
    # One shared scale + baseline across every state so the pet never slides or
    # pulses size between frames; compose just packs the normalized cells.
    sheet = atlas.compose_atlas(atlas.normalize_cells(frames_by_state))
    validation = atlas.validate_atlas(sheet)
    if not validation["ok"]:
        raise GenerationError("; ".join(validation["errors"]) or "atlas validation failed")
    filled_states = set(validation["filled_states"])
    missing_required = sorted(_REQUIRED_STATES - filled_states)
    if missing_required:
        failure_details = []
        for state in missing_required:
            failure = row_failures.get(state)
            if failure is None:
                continue
            attempts, error = failure
            failure_details.append(
                f"{state} after {attempts} attempt(s): {_humanize_image_error(str(error))}"
            )
        detail = f"; generation failures: {'; '.join(failure_details)}" if failure_details else ""
        raise GenerationError(
            f"missing required animation row(s): {', '.join(missing_required)}{detail}"
        )

    progress("save", slug)
    logger.info("pet hatch %r: saving pet", slug)
    if staging_dir is None:
        from agent.pet import store

        pet = store.register_local_pet(
            sheet,
            slug=slug,
            display_name=display_name or slug,
            description=description,
        )
        saved_slug = pet.slug
        saved_display_name = pet.display_name
        spritesheet = pet.spritesheet
    else:
        saved_slug, saved_display_name, spritesheet = _stage_hatched_pet(
            sheet,
            staging_dir=Path(staging_dir),
            slug=slug,
            display_name=display_name,
            description=description,
        )
    return HatchResult(
        slug=saved_slug,
        display_name=saved_display_name,
        spritesheet=spritesheet,
        states=validation["filled_states"],
        validation=validation,
    )


def _stage_hatched_pet(
    spritesheet,
    *,
    staging_dir: Path,
    slug: str,
    display_name: str,
    description: str,
) -> tuple[str, str, Path]:
    """Write one generated pet package outside the installed pet store."""
    from agent.pet import store

    safe_slug = store.slugify(slug)
    directory: Path | None = None
    try:
        root = staging_dir.expanduser()
        if root.is_symlink() or not root.is_dir():
            raise GenerationError("pet staging directory is unavailable")
        root = root.resolve(strict=True)
        directory = root / safe_slug
        if directory.exists() or directory.is_symlink():
            raise GenerationError("pet staging output already exists")
        directory.mkdir(mode=0o700)
        sprite_path = directory / "spritesheet.webp"
        sprite_partial = directory / "spritesheet.webp.part"
        metadata_path = directory / "pet.json"
        metadata_partial = directory / "pet.json.part"
        try:
            store._write_spritesheet(spritesheet, sprite_partial)
            sprite_partial.replace(sprite_path)
            metadata_partial.write_text(
                json.dumps(
                    {
                        "id": safe_slug,
                        "displayName": display_name or safe_slug,
                        "description": description or "",
                        "spritesheetPath": sprite_path.name,
                        "createdBy": "generator",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            metadata_partial.replace(metadata_path)
        finally:
            sprite_partial.unlink(missing_ok=True)
            metadata_partial.unlink(missing_ok=True)
        return safe_slug, display_name or safe_slug, sprite_path
    except Exception as exc:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError(f"could not stage generated pet '{safe_slug}': {exc}") from exc
