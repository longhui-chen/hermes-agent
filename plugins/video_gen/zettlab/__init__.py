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
        modalities: List[str] = ["text"]
        aspect_ratios: List[str] = []
        resolutions: List[str] = []
        durations: List[int] = []
        if isinstance(model, dict):
            for value in model.get("modalities") or []:
                if isinstance(value, str) and value not in modalities:
                    modalities.append(value)
            for value in model.get("aspect_ratios") or []:
                if isinstance(value, str) and value not in aspect_ratios:
                    aspect_ratios.append(value)
            for value in model.get("resolutions") or []:
                if isinstance(value, str) and value not in resolutions:
                    resolutions.append(value)
            for value in model.get("durations") or []:
                if isinstance(value, int) and value not in durations:
                    durations.append(value)
        limits = cap.get("limits") if isinstance(cap, dict) else {}
        max_refs = 0
        if isinstance(limits, dict):
            max_refs = int(limits.get("max_remote_media_inputs") or 0)
        return {
            "modalities": modalities,
            "aspect_ratios": aspect_ratios or [DEFAULT_ASPECT_RATIO],
            "resolutions": resolutions or [DEFAULT_RESOLUTION],
            "max_duration": max(durations) if durations else 10,
            "min_duration": min(durations) if durations else 1,
            "supports_audio": False,
            "supports_negative_prompt": True,
            "max_reference_images": max(0, max_refs - 1),
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

        try:
            effective_duration = int(duration or 0)
            if effective_duration <= 0 and isinstance(model_capability, dict):
                for candidate in model_capability.get("durations") or []:
                    if isinstance(candidate, int) and candidate > 0:
                        effective_duration = candidate
                        break
            inputs = media_client.remote_inputs(image_url, reference_image_urls)
            parameters: Dict[str, Any] = {}
            if negative_prompt:
                parameters["negative_prompt"] = negative_prompt
            if audio is not None:
                parameters["audio"] = bool(audio)
            if seed is not None:
                parameters["seed"] = seed
            job = media_client.create_and_wait(
                media_type="video",
                model=resolved_model,
                prompt=prompt,
                payload={
                    "output_count": 1,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "duration": effective_duration,
                    "remote_media_inputs": inputs,
                    "parameters": parameters,
                },
            )
            video = media_client.first_asset_url(job)
        except Exception as exc:
            return error_response(
                error=f"Zettlab video generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="zettlab",
                model=resolved_model,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        return success_response(
            video=video,
            model=resolved_model,
            prompt=prompt,
            modality="image" if image_url or reference_image_urls else "text",
            aspect_ratio=aspect_ratio,
            duration=effective_duration,
            provider="zettlab",
            extra={
                "job_id": job.get("job_id"),
                "assets": job.get("assets") or [],
            },
        )


def register(ctx) -> None:
    ctx.register_video_gen_provider(ZettlabVideoGenProvider())
