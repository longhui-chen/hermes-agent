"""Zettlab image generation backend via local-server ai-proxy."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    normalize_reference_images,
    resolve_aspect_ratio,
    success_response,
)
from plugins import zettlab_media_client as media_client


class ZettlabImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "zettlab"

    @property
    def display_name(self) -> str:
        return "Zettlab"

    def is_available(self) -> bool:
        return media_client.is_available("image")

    def list_models(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for model in media_client.list_models("image"):
            out.append({
                "id": model["id"],
                "display": model.get("display_name") or model["id"],
                "speed": "",
                "strengths": ", ".join(model.get("modalities") or []),
                "price": "Zettlab credits",
            })
        return out

    def default_model(self) -> Optional[str]:
        return media_client.default_model("image")

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Zettlab",
            "badge": "included",
            "tag": "Device cloud-account image generation via local-server ai-proxy",
            "env_vars": [],
        }

    def capabilities(self) -> Dict[str, Any]:
        try:
            cap, model = media_client.selected_model_capability("image")
        except Exception:
            return {"modalities": ["text"], "max_reference_images": 0}
        modalities: List[str] = ["text"]
        if isinstance(model, dict):
            for value in model.get("modalities") or []:
                if isinstance(value, str) and value not in modalities:
                    modalities.append(value)
        limits = cap.get("limits") if isinstance(cap, dict) else {}
        max_refs = 0
        if isinstance(limits, dict):
            max_refs = int(limits.get("max_remote_media_inputs") or 0)
        return {"modalities": modalities, "max_reference_images": max(0, max_refs - 1)}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="zettlab",
                aspect_ratio=aspect,
            )

        model = str(media_client.resolve_model("image", kwargs.get("model")) or "").strip()
        if not model:
            return error_response(
                error="No Zettlab image generation model is available from ai-gateway capabilities",
                error_type="no_model",
                provider="zettlab",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            refs = normalize_reference_images(reference_image_urls)
            inputs = media_client.remote_inputs(image_url, refs)
            output_count = int(kwargs.get("num_images") or kwargs.get("output_count") or 1)
            job = media_client.create_and_wait(
                media_type="image",
                model=model,
                prompt=prompt,
                payload={
                    "output_count": max(1, output_count),
                    "aspect_ratio": aspect,
                    "remote_media_inputs": inputs,
                },
            )
            image = media_client.first_asset_url(job)
        except Exception as exc:
            return error_response(
                error=f"Zettlab image generation failed: {exc}",
                error_type=type(exc).__name__,
                provider="zettlab",
                model=model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image,
            model=model,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="zettlab",
            modality="image" if image_url or reference_image_urls else "text",
            extra={
                "job_id": job.get("job_id"),
                "assets": job.get("assets") or [],
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ZettlabImageGenProvider())
