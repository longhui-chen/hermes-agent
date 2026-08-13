from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import tts_registry
from plugins import zettlab_media_client as media_client
from plugins.tts.zettlab import ZettlabTTSProvider
from tools import tts_tool


def _tts_capability(*, supports_speed: bool = True) -> dict:
    return {
        "enabled": True,
        "default_model": "public-default",
        "models": [
            {
                "id": "public-fast",
                "display_name": "Public Fast",
                "formats": ["mp3"],
                "default_voice": "voice-fast",
                "supports_speed": False,
            },
            {
                "id": "public-default",
                "display_name": "Public Default",
                "formats": ["mp3", "opus"],
                "default_voice": "voice-default",
                "supports_speed": supports_speed,
            },
        ],
    }


@pytest.fixture(autouse=True)
def _reset_tts_registry():
    tts_registry._reset_for_tests()
    try:
        yield
    finally:
        tts_registry._reset_for_tests()


def test_tts_capability_requires_complete_new_contract(monkeypatch):
    valid = _tts_capability()
    monkeypatch.setattr(media_client, "type_capability", lambda media_type: valid)
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )

    provider = ZettlabTTSProvider()
    assert provider.is_available() is True
    assert provider.default_model() == "public-default"
    assert provider.default_voice() == "voice-default"
    assert provider.list_models()[1] == {
        "id": "public-default",
        "display": "Public Default",
        "formats": ["mp3", "opus"],
        "default_voice": "voice-default",
        "supports_speed": True,
    }

    invalid_sections = [
        {},
        {**valid, "enabled": False},
        {**valid, "default_model": None},
        {**valid, "default_model": "missing"},
        {**valid, "models": []},
        {
            **valid,
            "models": [{"id": "public-default", "formats": ["mp3"]}],
        },
    ]
    for section in invalid_sections:
        monkeypatch.setattr(
            media_client, "type_capability", lambda media_type, value=section: value
        )
        assert provider.is_available() is False
        assert provider.default_model() is None
        assert provider.list_models() == []


def test_tts_model_resolution_order_and_no_list_first_fallback(monkeypatch):
    section = _tts_capability()
    monkeypatch.setattr(media_client, "type_capability", lambda media_type: section)
    monkeypatch.setattr(media_client, "_config_section", lambda media_type: {})

    assert media_client.resolve_model("tts", "public-fast") == "public-fast"
    assert media_client.resolve_model("tts") == "public-default"
    assert media_client.resolve_model("tts", "missing") is None

    monkeypatch.setattr(
        media_client, "_config_section", lambda media_type: {"model": "public-fast"}
    )
    assert media_client.resolve_model("tts") == "public-fast"

    no_default = dict(section)
    no_default.pop("default_model")
    monkeypatch.setattr(media_client, "type_capability", lambda media_type: no_default)
    assert media_client.resolve_model("tts") is None


def test_zettlab_provider_uses_capability_transport_contract(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        media_client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: (
            "public-default",
            _tts_capability()["models"][1],
        ),
    )
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/api/v1/ai-proxy/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )

    def fake_generate(text, output_path, config, **kwargs):
        captured.update(text=text, output_path=output_path, config=config, **kwargs)
        Path(output_path).write_bytes(b"OggS-audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_generate)

    cancel_event = threading.Event()
    cancel_token = tts_tool._TTS_CANCEL_EVENT.set(cancel_event)
    try:
        result = ZettlabTTSProvider().synthesize(
            "hello",
            str(tmp_path / "speech.mp3"),
            format="opus",
            speed=1.25,
        )
    finally:
        tts_tool._TTS_CANCEL_EVENT.reset(cancel_token)

    assert result == str(tmp_path / "speech.ogg")
    assert captured["api_key"] == "action-token"
    assert captured["base_url"].endswith("/api/v1/ai-proxy/v1")
    assert captured["model"] == "public-default"
    assert captured["voice"] == "voice-default"
    assert captured["speed"] == 1.25
    assert captured["stream_response"] is True
    assert captured["cancel_event"] is cancel_event
    assert captured["client_kwargs"]["max_retries"] == 0
    http_client = captured["client_kwargs"]["http_client"]
    assert http_client._trust_env is False
    assert http_client.follow_redirects is False


def test_zettlab_provider_omits_speed_when_capability_disables_it(
    monkeypatch,
    tmp_path,
):
    model_capability = _tts_capability(supports_speed=False)["models"][1]
    monkeypatch.setattr(
        media_client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: ("public-default", model_capability),
    )
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )
    captured = {}

    def fake_generate(text, output_path, config, **kwargs):
        captured.update(kwargs)
        Path(output_path).write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_generate)

    ZettlabTTSProvider().synthesize(
        "hello", str(tmp_path / "speech.mp3"), speed=2.0
    )
    assert captured["speed"] is None


@pytest.mark.parametrize(
    ("formats", "expected_format", "expected_suffix"),
    [
        (["mp3"], "mp3", ".mp3"),
        (["opus"], "opus", ".ogg"),
        (["wav"], "wav", ".wav"),
        (["flac"], "flac", ".flac"),
    ],
)
def test_zettlab_provider_negotiates_supported_format(
    monkeypatch,
    tmp_path,
    formats,
    expected_format,
    expected_suffix,
):
    model_capability = {
        **_tts_capability()["models"][0],
        "formats": formats,
    }
    monkeypatch.setattr(
        media_client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: ("public-fast", model_capability),
    )
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )
    captured = {}

    def fake_generate(text, output_path, config, **kwargs):
        captured["output_path"] = output_path
        Path(output_path).write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_generate)

    result = ZettlabTTSProvider().synthesize(
        "hello", str(tmp_path / "speech.mp3"), format="flac"
    )

    assert result.endswith(expected_suffix)
    assert captured["output_path"] == result
    assert tts_tool._tts_response_format_from_path(result) == expected_format


def test_text_to_speech_auto_dispatches_to_zettlab_plugin(monkeypatch, tmp_path):
    capability = _tts_capability()
    tts_registry.register_provider(ZettlabTTSProvider())
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_zettlab_tool_gateway",
        lambda vendor: SimpleNamespace(vendor="zettlab-tts"),
    )
    monkeypatch.setattr(tts_tool, "_resolve_profile_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(
        tts_tool,
        "_load_tts_config",
        lambda: {"provider": "edge", "_provider_is_default": True},
    )
    monkeypatch.setattr(media_client, "type_capability", lambda media_type: capability)
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )

    def fake_generate(text, output_path, config, **kwargs):
        Path(output_path).write_bytes(b"audio")
        return output_path

    monkeypatch.setattr(tts_tool, "_generate_openai_tts", fake_generate)

    result = json.loads(
        tts_tool.text_to_speech_tool(
            "hello",
            output_path=str(tmp_path / "speech.mp3"),
        )
    )

    assert result["success"] is True
    assert result["provider"] == "zettlab"
    assert result["file_path"] == str(tmp_path / "speech.mp3")
    assert tts_tool.check_tts_requirements() is True


def test_zettlab_tool_is_unavailable_when_capability_is_missing(monkeypatch):
    tts_registry.register_provider(ZettlabTTSProvider())
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_zettlab_tool_gateway",
        lambda vendor: SimpleNamespace(vendor="zettlab-tts"),
    )
    monkeypatch.setattr(tts_tool, "_resolve_profile_openai_audio_api_key", lambda: "")
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {})
    monkeypatch.setattr(media_client, "type_capability", lambda media_type: {})

    assert tts_tool._get_provider({}) == "zettlab"
    assert tts_tool.check_tts_requirements() is False


def test_zettlab_default_respects_disabled_plugin(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_zettlab_tool_gateway",
        lambda vendor: SimpleNamespace(vendor="zettlab-tts"),
    )
    monkeypatch.setattr(tts_tool, "_resolve_profile_openai_audio_api_key", lambda: "")

    assert tts_tool._get_provider({}) == "edge"


def test_tts_gateway_runtime_uses_share_action_origin(monkeypatch):
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AI_PROXY_BASE_URL", raising=False)
    monkeypatch.setenv(
        "ZETTLAB_AGENT_SHARE_ACTION_URL",
        "http://127.0.0.1:9430/api/v1/internal/action",
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "same-origin-token")

    base_url, headers = media_client.tts_gateway_runtime()

    assert base_url == "http://127.0.0.1:9430/api/v1/ai-proxy/v1"
    assert headers == {media_client.ACTION_TOKEN_HEADER: "same-origin-token"}


def test_unresolved_openai_secret_ref_does_not_override_zettlab(monkeypatch):
    tts_registry.register_provider(ZettlabTTSProvider())
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tts_tool,
        "resolve_zettlab_tool_gateway",
        lambda vendor: SimpleNamespace(vendor="zettlab-tts"),
    )
    monkeypatch.setattr(tts_tool, "_resolve_profile_openai_audio_api_key", lambda: "")

    config = {"openai": {"api_key": "${env:OPENAI_API_KEY}"}}
    assert tts_tool._get_provider(config) == "zettlab"
    with pytest.raises(ValueError, match="Neither tts.openai.api_key"):
        tts_tool._resolve_openai_audio_client_config(
            {"provider": "openai", "use_gateway": False, **config}
        )


def test_zettlab_provider_is_unavailable_without_action_token(monkeypatch):
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (_ for _ in ()).throw(
            media_client.ZettlabMediaError("action token is missing")
        ),
    )

    assert ZettlabTTSProvider().is_available() is False


def test_zettlab_provider_is_unavailable_without_supported_format(monkeypatch):
    monkeypatch.setattr(
        media_client,
        "tts_gateway_runtime",
        lambda: (
            "http://127.0.0.1:9090/v1",
            {media_client.ACTION_TOKEN_HEADER: "action-token"},
        ),
    )
    monkeypatch.setattr(media_client, "is_available", lambda media_type: True)
    monkeypatch.setattr(
        media_client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: (
            "public-default",
            {
                **_tts_capability()["models"][1],
                "formats": ["aac", "pcm"],
            },
        ),
    )

    assert ZettlabTTSProvider().is_available() is False
