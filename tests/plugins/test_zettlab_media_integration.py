from __future__ import annotations

import json


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_image_generate_tool_dispatches_to_zettlab_provider(monkeypatch):
    from agent import image_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.image_gen.zettlab import ZettlabImageGenProvider
    from tools import image_generation_tool as image_tool

    image_gen_registry._reset_for_tests()
    image_gen_registry.register_provider(ZettlabImageGenProvider())
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "zettlab")
    monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)

    captured = {}

    def fake_post(url, json, headers, timeout, allow_redirects):
        assert allow_redirects is False
        captured["json"] = json
        return _Resp({
            "job_id": "job-image",
            "status": "done",
            "assets": [{"url": "https://cdn.example/image.png"}],
        })

    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects: _Resp({
        "image": {
            "enabled": True,
            "default_model": "seedream-v4",
            "models": [{"id": "seedream-v4"}],
        },
    }))
    monkeypatch.setattr(client._SESSION, "post", fake_post)

    raw = image_tool._handle_image_generate({
        "prompt": "make a product shot",
        "aspect_ratio": "square",
    })
    got = json.loads(raw)
    assert got["success"] is True
    assert got["provider"] == "zettlab"
    assert got["image"] == "https://cdn.example/image.png"
    assert captured["json"]["media_type"] == "image"
    assert captured["json"]["model"] == "seedream-v4"


def test_image_only_model_requires_input_through_generation_tool(monkeypatch):
    from agent import image_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.image_gen.zettlab import ZettlabImageGenProvider
    from tools import image_generation_tool as image_tool

    image_gen_registry._reset_for_tests()
    image_gen_registry.register_provider(ZettlabImageGenProvider())
    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "zettlab")
    monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects: _Resp({
        "image": {
            "enabled": True,
            "default_model": "image-only",
            "models": [{"id": "image-only", "modalities": ["image"]}],
        },
    }))
    monkeypatch.setattr(
        client._SESSION,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("request should not be sent")),
    )

    got = json.loads(image_tool._handle_image_generate({"prompt": "edit this image"}))

    assert got["success"] is False
    assert got["error_type"] == "missing_image"


def test_video_generate_tool_dispatches_to_zettlab_provider(monkeypatch):
    from agent import video_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import video_generation_tool as video_tool

    video_gen_registry._reset_for_tests()
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(video_tool, "_read_configured_video_provider", lambda: "zettlab")
    monkeypatch.setattr(video_tool, "_read_configured_video_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)

    captured = {}

    def fake_post(url, json, headers, timeout, allow_redirects):
        assert allow_redirects is False
        captured["json"] = json
        return _Resp({
            "job_id": "job-video",
            "status": "done",
            "assets": [{"url": "https://cdn.example/video.mp4"}],
        })

    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects: _Resp({
        "video": {
            "enabled": True,
            "default_model": "seedance-v1",
            "models": [{
                "id": "seedance-v1",
                "aspect_ratios": ["9:16"],
                "resolutions": ["1080p"],
            }],
        },
    }))
    monkeypatch.setattr(client._SESSION, "post", fake_post)

    raw = video_tool._handle_video_generate({
        "prompt": "make a clip",
        "aspect_ratio": "16:9",
    })
    got = json.loads(raw)
    assert got["success"] is True
    assert got["provider"] == "zettlab"
    assert got["video"] == "https://cdn.example/video.mp4"
    assert captured["json"]["media_type"] == "video"
    assert captured["json"]["model"] == "seedance-v1"
    assert captured["json"]["aspect_ratio"] == "9:16"
    assert captured["json"]["resolution"] == "1080p"
    assert "duration" not in captured["json"]
