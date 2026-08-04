"""Zettlab video generation backend via local-server ai-proxy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    success_response,
)
from plugins import zettlab_media_client as media_client


class ZettlabVideoGenProvider(VideoGenProvider):
    @property
    def name(self) -> str:
        return "zettlab"

    @property
    def display_name(self) -> str:
        return "Zettlab"

    def is_available(self) -> bool:
        return media_client.is_available("video")

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for model in media_client.list_models("video"):
            out.append({
                "id": model["id"],
                "display": model.get("display_name") or model["id"],
                "speed": "",
                "strengths": ", ".join(model.get("modalities") or []),
                "price": "Zettlab credits",
                "modalities": model.get("modalities") or ["text"],
            })
        return out

    def default_model(self) -> Optional[str]:
        return media_client.default_model("video")

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Zettlab",
            "badge": "included",
            "tag": "Device cloud-account video generation via local-server ai-proxy",
            "env_vars": [],
        }

    def capabilities(self) -> Dict[str, Any]:
        try:
            cap, model = media_client.selected_model_capability("video")
        except Exception:
            return super().capabilities()
        modalities: List[str] = []
        aspect_ratios: List[str] = []
        resolutions: List[str] = []
        durations: List[int] = []
        if isinstance(model, dict):
            modalities = media_client.supported_modalities(cap, model)
            for value in model.get("aspect_ratios") or []:
                if isinstance(value, str) and value not in aspect_ratios:
                    aspect_ratios.append(value)
            for value in model.get("resolutions") or []:
                if isinstance(value, str) and value not in resolutions:
                    resolutions.append(value)
            for value in model.get("durations") or []:
                if isinstance(value, int) and value not in durations:
                    durations.append(value)
        return {
            "modalities": modalities,
            "aspect_ratios": aspect_ratios or [DEFAULT_ASPECT_RATIO],
            "resolutions": resolutions or [DEFAULT_RESOLUTION],
            "max_duration": max(durations) if durations else 10,
            "min_duration": min(durations) if durations else 1,
            "supports_audio": False,
            "supports_negative_prompt": False,
            "max_reference_images": 0,
            "image_input_description": (
                "Pass one PNG, JPEG, or WebP image as a base64 Data URI, local file "
                "path/file URL, or HTTP(S) URL."
            ),
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        if not prompt:
            return error_response(
                error="prompt is required.",
                error_type="missing_prompt",
                provider="zettlab",
                model=model or "",
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        resolved_model_value, model_capability = media_client.resolve_model_with_capability("video", model)
        resolved_model = str(resolved_model_value or "").strip()
        if not resolved_model:
            return error_response(
                error="No Zettlab video generation model is available from ai-gateway capabilities",
                error_type="no_model",
                provider="zettlab",
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        effective_aspect_ratio = aspect_ratio
        effective_resolution = resolution
        effective_duration: Optional[int] = None
        try:
            session_id = kwargs.get("_task_id")
            if duration is not None:
                effective_duration = int(duration)
            allowed_durations: List[int] = []
            if isinstance(model_capability, dict):
                for candidate in model_capability.get("durations") or []:
                    if isinstance(candidate, int) and candidate > 0 and candidate not in allowed_durations:
                        allowed_durations.append(candidate)
                allowed_aspect_ratios = [
                    value.strip()
                    for value in model_capability.get("aspect_ratios") or []
                    if isinstance(value, str) and value.strip()
                ]
                if allowed_aspect_ratios and effective_aspect_ratio not in allowed_aspect_ratios:
                    effective_aspect_ratio = allowed_aspect_ratios[0]
                allowed_resolutions = [
                    value.strip()
                    for value in model_capability.get("resolutions") or []
                    if isinstance(value, str) and value.strip()
                ]
                if allowed_resolutions and effective_resolution not in allowed_resolutions:
                    effective_resolution = allowed_resolutions[0]
            if allowed_durations:
                if effective_duration is None or effective_duration <= 0:
                    effective_duration = allowed_durations[0]
                elif effective_duration not in allowed_durations:
                    effective_duration = min(
                        allowed_durations,
                        key=lambda candidate: (abs(candidate - effective_duration), candidate),
                    )
            input_image = media_client.inline_image_input(
                image_url,
                reference_image_urls,
                model_capability,
            )
            configured_modalities = media_client.normalized_modalities(model_capability)
            if not input_image and "text" not in configured_modalities:
                if "image" in configured_modalities:
                    return error_response(
                        error="An image input is required for this Zettlab video generation model.",
                        error_type="missing_image",
                        provider="zettlab",
                        model=resolved_model,
                        prompt=prompt,
                        aspect_ratio=effective_aspect_ratio,
                    )
                return error_response(
                    error="The Zettlab video model exposes no supported input modality.",
                    error_type="unsupported_capability",
                    provider="zettlab",
                    model=resolved_model,
                    prompt=prompt,
                    aspect_ratio=effective_aspect_ratio,
                )
            if negative_prompt or audio is not None or seed is not None:
                return error_response(
                    error="negative_prompt, audio, and seed are not enabled for Zettlab video generation.",
                    error_type="unsupported_parameter",
                    provider="zettlab",
                    model=resolved_model,
                    prompt=prompt,
                    aspect_ratio=effective_aspect_ratio,
                )
            payload: Dict[str, Any] = {
                "output_count": 1,
                "aspect_ratio": effective_aspect_ratio,
                "resolution": effective_resolution,
            }
            if input_image:
                payload["input_image"] = input_image
            if effective_duration is not None:
                payload["duration"] = effective_duration
            job = media_client.create_and_wait(
                media_type="video",
                model=resolved_model,
                prompt=prompt,
                timeout_seconds=media_client.timeout_from_model_capability("video", model_capability),
                payload=payload,
                session_id=session_id,
            )
            video = media_client.first_asset_location(
                job,
                prefer_local=bool(session_id),
            )
        except Exception as exc:
            return error_response(
                error=f"Zettlab video generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="zettlab",
                model=resolved_model,
                prompt=prompt,
                aspect_ratio=effective_aspect_ratio,
            )

        return success_response(
            video=video,
            model=resolved_model,
            prompt=prompt,
            modality="image" if input_image else "text",
            aspect_ratio=effective_aspect_ratio,
            duration=effective_duration or 0,
            provider="zettlab",
            extra={
                "job_id": job.get("job_id"),
                "assets": job.get("assets") or [],
            },
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(ZettlabVideoGenProvider())
