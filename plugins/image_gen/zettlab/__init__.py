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


_GATEWAY_ASPECT_PREFERENCES = {
    "landscape": ("16:9", "4:3", "3:2", "21:9"),
    "square": ("1:1",),
    "portrait": ("9:16", "3:4", "2:3"),
}


def _capability_strings(model_capability: Optional[Dict[str, Any]], key: str) -> List[str]:
    if not isinstance(model_capability, dict):
        return []
    values = model_capability.get(key)
    if not isinstance(values, list):
        return []
    return [value.strip() for value in values if isinstance(value, str) and value.strip()]


def _gateway_aspect_ratio(aspect: str, model_capability: Optional[Dict[str, Any]]) -> str:
    allowed = _capability_strings(model_capability, "aspect_ratios")
    if not allowed or aspect in allowed:
        return aspect
    for candidate in _GATEWAY_ASPECT_PREFERENCES.get(aspect, ()):
        if candidate in allowed:
            return candidate
    return allowed[0]


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
        return {
            "modalities": media_client.supported_modalities(cap, model),
            "max_reference_images": 0,
            "image_input_description": (
                "Pass one PNG, JPEG, or WebP image as a base64 Data URI, local file "
                "path/file URL, or HTTP(S) URL."
            ),
        }

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

        model_value, model_capability = media_client.resolve_model_with_capability("image", kwargs.get("model"))
        model = str(model_value or "").strip()
        if not model:
            return error_response(
                error="No Zettlab image generation model is available from ai-gateway capabilities",
                error_type="no_model",
                provider="zettlab",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            session_id = kwargs.get("_task_id")
            refs = normalize_reference_images(reference_image_urls)
            input_image = media_client.inline_image_input(
                image_url,
                refs,
                model_capability,
            )
            configured_modalities = media_client.normalized_modalities(model_capability)
            if not input_image and "text" not in configured_modalities:
                if "image" in configured_modalities:
                    return error_response(
                        error="An image input is required for this Zettlab image generation model.",
                        error_type="missing_image",
                        provider="zettlab",
                        model=model,
                        prompt=prompt,
                        aspect_ratio=aspect,
                    )
                return error_response(
                    error="The Zettlab image model exposes no supported input modality.",
                    error_type="unsupported_capability",
                    provider="zettlab",
                    model=model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            gateway_aspect = _gateway_aspect_ratio(aspect, model_capability)
            payload: Dict[str, Any] = {
                "output_count": 1,
                "aspect_ratio": gateway_aspect,
            }
            if input_image:
                payload["input_image"] = input_image
            resolutions = _capability_strings(model_capability, "resolutions")
            if resolutions:
                payload["resolution"] = resolutions[0]
            job = media_client.create_and_wait(
                media_type="image",
                model=model,
                prompt=prompt,
                timeout_seconds=media_client.timeout_from_model_capability("image", model_capability),
                payload=payload,
                session_id=session_id,
            )
            image = media_client.first_asset_location(
                job,
                prefer_local=bool(session_id),
            )
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
            modality="image" if input_image else "text",
            extra={
                "job_id": job.get("job_id"),
                "assets": job.get("assets") or [],
            },
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(ZettlabImageGenProvider())
