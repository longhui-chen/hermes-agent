"""Zettlab managed TTS backend via local-server ai-proxy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from agent.tts_provider import TTSProvider
from plugins import zettlab_media_client as media_client


_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 120.0
_WRITE_TIMEOUT_SECONDS = 30.0
_POOL_TIMEOUT_SECONDS = 5.0
_FORMAT_SUFFIXES = {
    "aac": ".aac",
    "flac": ".flac",
    "mp3": ".mp3",
    "opus": ".ogg",
    "pcm": ".pcm",
    "wav": ".wav",
}


def _format_values(model_capability: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for value in model_capability.get("formats") or []:
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in _FORMAT_SUFFIXES and normalized not in out:
                out.append(normalized)
    return out


class ZettlabTTSProvider(TTSProvider):
    @property
    def name(self) -> str:
        return "zettlab"

    @property
    def display_name(self) -> str:
        return "Zettlab"

    def is_available(self) -> bool:
        try:
            media_client.action_headers()
            media_client.base_url("tts")
        except Exception:
            return False
        if not media_client.is_available("tts"):
            return False
        try:
            _, model = media_client.resolve_model_with_capability("tts")
        except Exception:
            return False
        return isinstance(model, dict) and bool(_format_values(model))

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": model["id"],
                "display": model.get("display_name") or model["id"],
                "formats": _format_values(model),
                "default_voice": model.get("default_voice"),
                "supports_speed": model.get("supports_speed") is True,
            }
            for model in media_client.list_models("tts")
        ]

    def default_model(self) -> Optional[str]:
        return media_client.default_model("tts")

    def default_voice(self) -> Optional[str]:
        _, model = media_client.resolve_model_with_capability("tts")
        if not isinstance(model, dict):
            return None
        voice = model.get("default_voice")
        return voice.strip() if isinstance(voice, str) and voice.strip() else None

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Zettlab",
            "badge": "included",
            "tag": "Device cloud-account TTS via local-server ai-proxy",
            "env_vars": [],
        }

    @property
    def voice_compatible(self) -> bool:
        try:
            _, model = media_client.resolve_model_with_capability("tts")
        except Exception:
            return False
        return isinstance(model, dict) and "opus" in _format_values(model)

    def synthesize(
        self,
        text: str,
        output_path: str,
        *,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        speed: Optional[float] = None,
        format: str = "mp3",
        **extra: Any,
    ) -> str:
        del extra
        model_id, model_capability = media_client.resolve_model_with_capability(
            "tts", model
        )
        if not model_id or not isinstance(model_capability, dict):
            raise ValueError(
                "No valid Zettlab TTS model is available from ai-gateway capabilities"
            )

        formats = _format_values(model_capability)
        if not formats:
            raise ValueError(
                f"Zettlab TTS model {model_id!r} has no supported response format"
            )
        requested_format = str(format or "mp3").strip().lower()
        if requested_format == "ogg":
            requested_format = "opus"
        if requested_format not in formats:
            requested_format = formats[0]

        selected_voice = str(voice or "").strip()
        if not selected_voice:
            selected_voice = str(model_capability.get("default_voice") or "").strip()
        if not selected_voice:
            raise ValueError(
                f"Zettlab TTS model {model_id!r} has no default voice capability"
            )

        token = media_client.action_headers()[media_client.ACTION_TOKEN_HEADER]
        from tools.tts_tool import _generate_openai_tts

        http_client = httpx.Client(
            trust_env=False,
            timeout=httpx.Timeout(
                connect=_CONNECT_TIMEOUT_SECONDS,
                read=_READ_TIMEOUT_SECONDS,
                write=_WRITE_TIMEOUT_SECONDS,
                pool=_POOL_TIMEOUT_SECONDS,
            ),
            follow_redirects=False,
        )
        output = Path(output_path)
        expected_suffix = _FORMAT_SUFFIXES.get(
            requested_format, f".{requested_format}"
        )
        if output.suffix.lower() != expected_suffix:
            output = output.with_suffix(expected_suffix)
        try:
            return _generate_openai_tts(
                text,
                str(output),
                {},
                api_key=token,
                base_url=media_client.base_url("tts"),
                model=model_id,
                voice=selected_voice,
                speed=(speed if model_capability.get("supports_speed") is True else None),
                stream_response=True,
                client_kwargs={"http_client": http_client, "max_retries": 0},
                label="Zettlab TTS",
            )
        finally:
            http_client.close()


def register(ctx) -> None:
    ctx.register_tts_provider(ZettlabTTSProvider())
