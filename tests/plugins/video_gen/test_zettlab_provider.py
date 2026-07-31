from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from plugins.video_gen.zettlab import ZettlabVideoGenProvider, register


PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(
    b"\x89PNG\r\n\x1a\nsource"
).decode("ascii")


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def _capabilities():
    return {
        "video": {
            "enabled": True,
            "models": [{
                "id": "seedance-v1",
                "display_name": "Seedance V1",
                "modalities": ["text", "image"],
                "aspect_ratios": ["16:9", "9:16"],
                "resolutions": ["720p"],
                "durations": [5, 10],
            }],
            "limits": {
                "provider_timeout_seconds": 1200,
                "finalization_timeout_seconds": 600,
                "max_inline_image_bytes": 5 * 1024 * 1024,
            },
        },
    }


def test_zettlab_video_provider_reads_capabilities(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client._SESSION,
        "get",
        lambda url, timeout, allow_redirects, stream: _Resp(_capabilities()),
    )

    provider = ZettlabVideoGenProvider()
    assert provider.is_available() is True
    assert provider.default_model() == "seedance-v1"
    assert provider.list_models()[0]["display"] == "Seedance V1"
    caps = provider.capabilities()
    assert caps["aspect_ratios"] == ["16:9", "9:16"]
    assert caps["resolutions"] == ["720p"]
    assert caps["max_duration"] == 10


def test_zettlab_video_capabilities_use_only_selected_model(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "selected",
        "models": [
            {"id": "selected", "modalities": ["text"], "aspect_ratios": ["16:9"], "resolutions": ["720p"], "durations": [5]},
            {"id": "other", "modalities": ["image"], "aspect_ratios": ["9:16"], "resolutions": ["1080p"], "durations": [10]},
        ],
        "limits": {"max_remote_media_inputs": 1},
    })

    caps = ZettlabVideoGenProvider().capabilities()
    assert caps["modalities"] == ["text"]
    assert caps["aspect_ratios"] == ["16:9"]
    assert caps["resolutions"] == ["720p"]
    assert caps["min_duration"] == caps["max_duration"] == 5


def test_zettlab_video_capabilities_preserve_image_only_modality(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "selected_model_capability", lambda media_type: ({
        "limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
    }, {
        "id": "image-only",
        "modalities": ["image"],
        "durations": [5],
    }))

    assert ZettlabVideoGenProvider().capabilities()["modalities"] == ["image"]


def test_zettlab_video_capabilities_hide_image_without_inline_limit(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "selected_model_capability", lambda media_type: ({
        "limits": {},
    }, {
        "id": "image-only",
        "modalities": ["image"],
        "durations": [5],
    }))

    assert ZettlabVideoGenProvider().capabilities()["modalities"] == []


def test_zettlab_video_generate_creates_media_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    monkeypatch.setattr(client._SESSION, "get", lambda url, **kwargs: _Resp(_capabilities()))

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        assert allow_redirects is False
        assert stream is True
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _Resp({
            "job_id": "job-video-1",
            "status": "done",
            "assets": [{
                "asset_id": "asset-video-1",
                "url": "https://cdn.example/video.mp4",
                "content_type": "video/mp4",
                "duration_sec": 5,
            }],
        })

    monkeypatch.setattr(client._SESSION, "post", fake_post)

    got = ZettlabVideoGenProvider().generate(
        "make a short clip",
        model="seedance-v1",
        duration=5,
        aspect_ratio="16:9",
        resolution="720p",
        image_url=PNG_DATA_URI,
    )

    assert got["success"] is True
    assert got["video"] == "https://cdn.example/video.mp4"
    assert got["provider"] == "zettlab"
    assert got["job_id"] == "job-video-1"
    assert captured["url"].endswith("/media/generation-jobs")
    assert captured["headers"]["X-Scene-Type"] == "media_generation"
    assert captured["headers"]["X-Zettlab-Agent-Action-Token"] == "media-token"
    assert captured["json"]["media_type"] == "video"
    assert captured["json"]["model"] == "seedance-v1"
    assert captured["json"]["duration"] == 5
    assert "parameters" not in captured["json"]
    assert captured["json"]["input_image"] == PNG_DATA_URI
    assert "remote_media_inputs" not in captured["json"]


def test_zettlab_video_rejects_disabled_custom_parameters(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client._SESSION, "get", lambda url, **kwargs: _Resp(_capabilities()))
    monkeypatch.setattr(
        client,
        "create_and_wait",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unsupported request should not be sent")),
    )

    got = ZettlabVideoGenProvider().generate(
        "make a short clip",
        model="seedance-v1",
        negative_prompt="blurry",
    )

    assert got["success"] is False
    assert got["error_type"] == "unsupported_parameter"


def test_zettlab_video_generate_uses_gateway_default_when_model_is_omitted(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    capability_calls = 0

    def fake_capability(media_type):
        nonlocal capability_calls
        capability_calls += 1
        return {
            "enabled": True,
            "default_model": "seedance-default",
            "models": [{"id": "seedance-default", "durations": [5, 10]}],
        }

    monkeypatch.setattr(client, "type_capability", fake_capability)

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        assert allow_redirects is False
        assert stream is True
        captured.update(json)
        return _Resp({"job_id": "job-video-default", "status": "done", "assets": [{"url": "https://cdn.example/default.mp4"}]})

    monkeypatch.setattr(client._SESSION, "post", fake_post)
    got = ZettlabVideoGenProvider().generate("make video")
    assert got["success"] is True
    assert captured["model"] == "seedance-default"
    assert captured["duration"] == 5
    assert got["duration"] == 5
    assert capability_calls == 1


def test_zettlab_video_generate_normalizes_duration_to_nearest_supported_value(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: ("seedance-v1", {
            "id": "seedance-v1",
            "modalities": ["text"],
            "durations": [5, 10],
        }),
    )
    captured = {}

    def fake_create_and_wait(**kwargs):
        captured.update(kwargs["payload"])
        return {"job_id": "job-nearest", "status": "done", "assets": [{"url": "https://cdn.example/nearest.mp4"}]}

    monkeypatch.setattr(client, "create_and_wait", fake_create_and_wait)

    got = ZettlabVideoGenProvider().generate("make video", duration=8)

    assert got["success"] is True
    assert captured["duration"] == 10
    assert got["duration"] == 10


def test_zettlab_video_omits_unspecified_duration_and_normalizes_model_options(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: ("restricted-video", {
            "id": "restricted-video",
            "modalities": ["text"],
            "aspect_ratios": ["9:16"],
            "resolutions": ["1080p"],
        }),
    )
    captured = {}

    def fake_create_and_wait(**kwargs):
        captured.update(kwargs["payload"])
        return {"job_id": "job-restricted", "status": "done", "assets": [{"url": "https://cdn.example/restricted.mp4"}]}

    monkeypatch.setattr(client, "create_and_wait", fake_create_and_wait)

    got = ZettlabVideoGenProvider().generate("make vertical video")

    assert got["success"] is True
    assert "duration" not in captured
    assert captured["aspect_ratio"] == "9:16"
    assert captured["resolution"] == "1080p"
    assert got["aspect_ratio"] == "9:16"


def test_zettlab_video_image_only_model_requires_image_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: ("image-only", {
            "id": "image-only",
            "modalities": ["image"],
            "durations": [5],
        }),
    )
    monkeypatch.setattr(
        client,
        "create_and_wait",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("request should not be sent")),
    )

    got = ZettlabVideoGenProvider().generate("animate this")

    assert got["success"] is False
    assert got["error_type"] == "missing_image"


def test_zettlab_video_rejects_remote_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: (
            "seedance-v1",
            {
                "id": "seedance-v1",
                "modalities": ["text", "image"],
                "durations": [5],
                "_type_limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
            },
        ),
    )
    got = ZettlabVideoGenProvider().generate("make video", image_url="https://example.com/source.png")
    assert got["success"] is False
    assert got["error_type"] == "ZettlabMediaError"
    assert "local image path or data URI" in got["error"]


def test_first_asset_url_accepts_legacy_top_level_shortcut_without_assets():
    from plugins import zettlab_media_client as client

    assert client.first_asset_url({"status": "done", "video": "https://cdn.example/legacy.mp4"}) == (
        "https://cdn.example/legacy.mp4"
    )


@pytest.mark.parametrize(
    ("assets", "message"),
    [
        ([], "completed without assets"),
        (["bad-shape"], "invalid shape"),
        ([{}], "no retrievable URL"),
    ],
)
def test_first_asset_url_errors_preserve_job_id(assets, message):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match=rf"{message}.*job_id=job-assets"):
        client.first_asset_url({"job_id": "job-assets", "status": "done", "assets": assets})


def test_register_calls_video_provider_registry():
    calls = []
    register(SimpleNamespace(register_video_gen_provider=lambda provider: calls.append(provider)))
    assert isinstance(calls[0], ZettlabVideoGenProvider)
