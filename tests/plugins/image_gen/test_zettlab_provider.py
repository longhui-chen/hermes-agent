from __future__ import annotations

from types import SimpleNamespace

from plugins.image_gen.zettlab import ZettlabImageGenProvider, register


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_zettlab_image_provider_reads_capabilities(monkeypatch):
    from plugins import zettlab_media_client as client

    def fake_get(url, timeout):
        assert url == "http://127.0.0.1:9090/api/v1/ai-proxy/v1/media/generation-capabilities"
        assert timeout == client.CAPABILITY_TIMEOUT
        return _Resp({
            "image": {
                "enabled": True,
                "models": [{
                    "id": "seedream-v4",
                    "display_name": "Seedream V4",
                    "modalities": ["text", "image"],
                }],
                "limits": {"max_remote_media_inputs": 4},
            },
            "video": {"enabled": False, "models": []},
        })

    monkeypatch.setattr(client.requests, "get", fake_get)

    provider = ZettlabImageGenProvider()
    assert provider.is_available() is True
    assert provider.default_model() == "seedream-v4"
    assert provider.list_models()[0]["display"] == "Seedream V4"
    assert provider.capabilities()["max_reference_images"] == 3


def test_zettlab_image_generate_creates_media_job(monkeypatch):
    from plugins import zettlab_media_client as client

    captured = {}

    def fake_get(url, **kwargs):
        return _Resp({
            "image": {
                "enabled": True,
                "models": [{"id": "seedream-v4", "modalities": ["text", "image"]}],
                "limits": {"provider_timeout_seconds": 300, "finalization_timeout_seconds": 600},
            },
        })

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Resp({
            "job_id": "job-1",
            "status": "done",
            "assets": [{
                "asset_id": "asset-1",
                "url": "https://cdn.example/image.png",
                "content_type": "image/png",
            }],
        })

    monkeypatch.setattr(client.requests, "get", fake_get)
    monkeypatch.setattr(client.requests, "post", fake_post)

    got = ZettlabImageGenProvider().generate(
        "make a product shot",
        aspect_ratio="square",
        image_url="https://example.com/source.png",
        reference_image_urls=["https://example.com/ref.png"],
        model="seedream-v4",
        num_images=2,
    )

    assert got["success"] is True
    assert got["image"] == "https://cdn.example/image.png"
    assert got["provider"] == "zettlab"
    assert got["job_id"] == "job-1"
    assert captured["url"].endswith("/media/generation-jobs")
    assert captured["headers"]["X-Scene-Type"] == "media_generation"
    assert captured["json"]["media_type"] == "image"
    assert captured["json"]["model"] == "seedream-v4"
    assert captured["json"]["output_count"] == 2
    assert captured["json"]["remote_media_inputs"] == [
        {"url": "https://example.com/source.png", "role": "source"},
        {"url": "https://example.com/ref.png", "role": "reference"},
    ]


def test_zettlab_image_rejects_non_https_remote_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "default_model", lambda media_type: "seedream-v4")
    got = ZettlabImageGenProvider().generate("make image", image_url="http://example.com/a.png")
    assert got["success"] is False
    assert got["error_type"] == "ZettlabMediaError"
    assert "https URL" in got["error"]


def test_register_calls_image_provider_registry():
    calls = []
    register(SimpleNamespace(register_image_gen_provider=lambda provider: calls.append(provider)))
    assert isinstance(calls[0], ZettlabImageGenProvider)
