from __future__ import annotations

from types import SimpleNamespace

from plugins.video_gen.zettlab import ZettlabVideoGenProvider, register


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
                "max_remote_media_inputs": 1,
                "provider_timeout_seconds": 1200,
                "finalization_timeout_seconds": 600,
            },
        },
    }


def test_zettlab_video_provider_reads_capabilities(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout: _Resp(_capabilities()))

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


def test_zettlab_video_generate_creates_media_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    monkeypatch.setattr(client._SESSION, "get", lambda url, **kwargs: _Resp(_capabilities()))

    def fake_post(url, json, headers, timeout):
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
        image_url="https://example.com/source.png",
        duration=5,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt="blurry",
        seed=42,
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
    assert captured["json"]["parameters"] == {"negative_prompt": "blurry", "seed": 42}
    assert captured["json"]["remote_media_inputs"] == [
        {"url": "https://example.com/source.png", "role": "source"},
    ]


def test_zettlab_video_generate_uses_gateway_default_when_model_is_omitted(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "seedance-default",
        "models": [{"id": "seedance-default"}],
    })

    def fake_post(url, json, headers, timeout):
        captured.update(json)
        return _Resp({"job_id": "job-video-default", "status": "done", "assets": [{"url": "https://cdn.example/default.mp4"}]})

    monkeypatch.setattr(client._SESSION, "post", fake_post)
    got = ZettlabVideoGenProvider().generate("make video")
    assert got["success"] is True
    assert captured["model"] == "seedance-default"


def test_zettlab_video_rejects_non_https_remote_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "resolve_model", lambda media_type, requested=None: "seedance-v1")
    got = ZettlabVideoGenProvider().generate("make video", image_url="/tmp/source.png")
    assert got["success"] is False
    assert got["error_type"] == "ZettlabMediaError"
    assert "https URL" in got["error"]


def test_register_calls_video_provider_registry():
    calls = []
    register(SimpleNamespace(register_video_gen_provider=lambda provider: calls.append(provider)))
    assert isinstance(calls[0], ZettlabVideoGenProvider)
