from __future__ import annotations

import json
import threading


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

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        assert allow_redirects is False
        assert stream is True
        captured["json"] = json
        return _Resp({
            "job_id": "job-image",
            "status": "done",
            "assets": [{"url": "https://cdn.example/image.png"}],
        })

    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects, stream: _Resp({
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


def test_image_tool_hidden_when_selected_zettlab_capability_is_disabled(monkeypatch):
    from agent import image_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.image_gen.zettlab import ZettlabImageGenProvider
    from tools import image_generation_tool as image_tool

    image_gen_registry._reset_for_tests()
    image_gen_registry.register_provider(ZettlabImageGenProvider())
    monkeypatch.setenv("FAL_KEY", "legacy-fal-key")
    monkeypatch.setattr(
        image_tool, "_read_configured_image_provider", lambda: "zettlab"
    )
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        client._SESSION,
        "get",
        lambda *args, **kwargs: _Resp({
            "image": {
                "enabled": False,
                "default_model": None,
                "models": [],
            },
        }),
    )

    assert image_tool.check_image_generation_requirements() is False


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
    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects, stream: _Resp({
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

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        assert allow_redirects is False
        assert stream is True
        captured["json"] = json
        return _Resp({
            "job_id": "job-video",
            "status": "done",
            "assets": [{"url": "https://cdn.example/video.mp4"}],
        })

    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects, stream: _Resp({
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


def test_local_image_dispatches_through_both_generation_tools(tmp_path, monkeypatch):
    from agent import image_gen_registry, video_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.image_gen.zettlab import ZettlabImageGenProvider
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import image_generation_tool as image_tool
    from tools import video_generation_tool as video_tool

    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nsource-image")
    task_id = "zettlab:user:main:session-local"
    image_gen_registry._reset_for_tests()
    video_gen_registry._reset_for_tests()
    image_gen_registry.register_provider(ZettlabImageGenProvider())
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "zettlab")
    monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr(video_tool, "_read_configured_video_provider", lambda: "zettlab")
    monkeypatch.setattr(video_tool, "_read_configured_video_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)

    capabilities = {
        "image": {
            "enabled": True,
            "default_model": "seedream-v4",
            "models": [{"id": "seedream-v4", "modalities": ["text", "image"]}],
            "limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
        },
        "video": {
            "enabled": True,
            "default_model": "seedance-v1",
            "models": [{
                "id": "seedance-v1",
                "modalities": ["text", "image"],
                "durations": [5],
            }],
            "limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
        },
    }
    requests = []
    def fake_get(url, timeout, allow_redirects, stream, headers=None):
        if url.endswith("/media/generation-capabilities"):
            return _Resp(capabilities)
        media_type = "image" if url.endswith("job-image") else "video"
        extension = "png" if media_type == "image" else "mp4"
        return _Resp({
            "job_id": f"job-{media_type}",
            "status": "done",
            "assets": [{"url": f"https://cdn.example/generated.{extension}"}],
        })

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        requests.append(json)
        media_type = json["media_type"]
        extension = "png" if media_type == "image" else "mp4"
        return _Resp({
            "job_id": f"job-{media_type}",
            "status": "done",
            "assets": [{"url": f"https://cdn.example/generated.{extension}"}],
        })

    monkeypatch.setattr(client._SESSION, "post", fake_post)

    assert client.first_asset_location(
        {
            "job_id": "job-source",
            "assets": [{"local_path": str(image_path), "persisted": True}],
        },
        prefer_local=True,
    ) == str(image_path)

    image_result = json.loads(image_tool._handle_image_generate({
        "prompt": "edit this image",
        "image_url": str(image_path),
    }, task_id=task_id))
    video_result = json.loads(video_tool._handle_video_generate({
        "prompt": "animate this image",
        "image_url": str(image_path),
        "duration": 5,
    }, task_id=task_id))

    assert image_result["success"] is True
    assert image_result["modality"] == "image"
    assert video_result["success"] is True
    assert video_result["modality"] == "image"
    assert [request["media_type"] for request in requests] == ["image", "video"]
    assert all(request["input_image"].startswith("data:image/png;base64,") for request in requests)
    assert all("remote_media_inputs" not in request for request in requests)


def test_generated_media_local_artifact_flow(monkeypatch):
    from agent import image_gen_registry, video_gen_registry
    from plugins import zettlab_media_client as client
    from plugins.image_gen.zettlab import ZettlabImageGenProvider
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import image_generation_tool as image_tool
    from tools import video_generation_tool as video_tool

    image_gen_registry._reset_for_tests()
    video_gen_registry._reset_for_tests()
    image_gen_registry.register_provider(ZettlabImageGenProvider())
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(image_tool, "_read_configured_image_provider", lambda: "zettlab")
    monkeypatch.setattr(image_tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr(video_tool, "_read_configured_video_provider", lambda: "zettlab")
    monkeypatch.setattr(video_tool, "_read_configured_video_model", lambda: None)
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)

    capabilities = {
        "image": {
            "enabled": True,
            "default_model": "seedream-v4",
            "models": [{"id": "seedream-v4"}],
        },
        "video": {
            "enabled": True,
            "default_model": "seedance-v1",
            "models": [{"id": "seedance-v1"}],
        },
    }
    create_headers = []
    finalize_headers = []

    def fake_get(url, timeout, allow_redirects, stream, headers=None):
        if url.endswith("/media/generation-capabilities"):
            return _Resp(capabilities)
        finalize_headers.append(dict(headers or {}))
        media_type = "image" if url.endswith("job-image") else "video"
        extension = "png" if media_type == "image" else "mp4"
        return _Resp({
            "job_id": f"job-{media_type}",
            "status": "done",
            "media_type": media_type,
            "assets": [{
                "url": f"https://cdn.example/{media_type}.{extension}",
                "local_path": f"/volume1/agents/data/main/output/session-local/{media_type}.{extension}",
                "persisted": True,
            }],
        })

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        create_headers.append(dict(headers))
        media_type = json["media_type"]
        extension = "png" if media_type == "image" else "mp4"
        return _Resp({
            "job_id": f"job-{media_type}",
            "status": "done",
            "media_type": media_type,
            "assets": [{
                "url": f"https://cdn.example/{media_type}.{extension}",
            }],
        })

    monkeypatch.setattr(client._SESSION, "post", fake_post)

    image = json.loads(image_tool._handle_image_generate(
        {"prompt": "make an image"},
        task_id="zettlab:user:main:session-local",
    ))
    video = json.loads(video_tool._handle_video_generate(
        {"prompt": "make a video"},
        task_id="zettlab:user:main:session-local",
    ))

    assert image["image"].endswith("/session-local/image.png")
    assert video["video"].endswith("/session-local/video.mp4")
    assert all(
        client.ARTIFACT_SESSION_HEADER not in headers
        for headers in create_headers
    )
    assert all(
        headers["X-Task-Id"] == "zettlab:user:main:session-local"
        for headers in create_headers
    )
    assert all(
        headers[client.ARTIFACT_SESSION_HEADER] == "zettlab:user:main:session-local"
        for headers in finalize_headers
    )


def test_concurrent_generation_flow_uses_bounded_http_workers(monkeypatch):
    from plugins import zettlab_media_client as client

    barrier = threading.Barrier(2)
    calls = []

    class FlowWorker:
        def __init__(self, job_id):
            self.job_id = job_id

        def request(self, method, url, *, deadline, **kwargs):
            calls.append(kwargs["json"]["prompt"])
            barrier.wait(timeout=1)
            return _Resp({"job_id": self.job_id, "status": "done", "assets": []})

        def close(self):
            return None

    session = client._MediaHTTPSession(
        workers=[FlowWorker("job-a"), FlowWorker("job-b")]
    )
    monkeypatch.setattr(client, "_SESSION", session)
    monkeypatch.setattr(client, "base_url", lambda media_type: "http://127.0.0.1/test")
    monkeypatch.setattr(client, "action_headers", lambda: {"X-Test": "token"})
    results = []
    errors = []

    def generate(prompt):
        try:
            results.append(client.create_and_wait(
                media_type="image",
                model="seedream-v4",
                prompt=prompt,
                payload={},
                timeout_seconds=2,
            ))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=generate, args=("pet-a",)),
        threading.Thread(target=generate, args=("pet-b",)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
    finally:
        session.close()

    assert errors == []
    assert sorted(calls) == ["pet-a", "pet-b"]
    assert sorted(result["job_id"] for result in results) == ["job-a", "job-b"]


def test_generated_media_falls_back_to_remote_url_with_older_local_server(monkeypatch):
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
    def fake_get(url, *args, **kwargs):
        if url.endswith("/media/generation-capabilities"):
            return _Resp({
                "image": {
                    "enabled": True,
                    "default_model": "seedream-v4",
                    "models": [{"id": "seedream-v4"}],
                },
            })
        return _Resp({
            "job_id": "job-old-server",
            "status": "done",
            "assets": [{"url": "https://cdn.example/generated.png"}],
        })

    monkeypatch.setattr(client._SESSION, "get", fake_get)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({
        "job_id": "job-old-server",
        "status": "done",
        "assets": [{"url": "https://cdn.example/generated.png"}],
    }))

    got = json.loads(image_tool._handle_image_generate(
        {"prompt": "make an image"},
        task_id="zettlab:user:main:session-local",
    ))

    assert got["success"] is True
    assert got["image"] == "https://cdn.example/generated.png"


def test_async_media_uses_short_polls_then_remaining_budget_for_artifact_finalization(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({
        "job_id": "job-video",
        "status": "running",
    }))
    calls = []

    def fake_get(url, headers, timeout, allow_redirects, stream):
        calls.append({"headers": dict(headers), "timeout": timeout})
        if len(calls) == 1:
            return _Resp({
                "job_id": "job-video",
                "status": "done",
                "assets": [{"url": "https://cdn.example/video.mp4"}],
            })
        if len(calls) == 2:
            raise client.requests.ConnectionError("temporary finalization failure")
        return _Resp({
            "job_id": "job-video",
            "status": "done",
            "assets": [{
                "url": "https://cdn.example/video.mp4",
                "local_path": "/volume1/agents/data/main/output/session-local/video.mp4",
                "persisted": True,
            }],
        })

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    job = client.create_and_wait(
        media_type="video",
        model="seedance-v1",
        prompt="make a video",
        payload={},
        session_id="zettlab:user:main:session-local",
        timeout_seconds=120,
    )

    assert client.ARTIFACT_SESSION_HEADER not in calls[0]["headers"]
    assert calls[0]["timeout"] <= client.REQUEST_TIMEOUT
    assert calls[1]["headers"][client.ARTIFACT_SESSION_HEADER].endswith("session-local")
    assert calls[1]["timeout"] > client.REQUEST_TIMEOUT
    assert calls[2]["headers"][client.ARTIFACT_SESSION_HEADER].endswith("session-local")
    assert client.first_asset_local_path(job).endswith("video.mp4")


def test_zet_agent_exposes_video_tool_when_gateway_capability_is_enabled(monkeypatch):
    from agent import video_gen_registry
    from hermes_cli.tools_config import _get_platform_tools
    import model_tools
    from plugins import zettlab_media_client as client
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import video_generation_tool as video_tool

    video_gen_registry._reset_for_tests()
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr(video_tool, "_read_configured_video_provider", lambda: "zettlab")
    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects, stream: _Resp({
        "video": {
            "enabled": True,
            "default_model": "seedance-v1",
            "models": [{
                "id": "seedance-v1",
                "modalities": ["text"],
                "aspect_ratios": ["16:9"],
                "resolutions": ["720p"],
                "durations": [4],
            }],
        },
    }))

    enabled = _get_platform_tools(
        {
            "video_gen": {"provider": "zettlab"},
            "platform_toolsets": {"zet_agent": ["hermes-zet-agent", "cronjob"]},
        },
        "zet_agent",
        include_default_mcp_servers=False,
    )
    model_tools._clear_tool_defs_cache()
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=sorted(enabled),
        quiet_mode=True,
        # Progressive tool search may defer plugin schemas from the eager
        # model-facing list.  This assertion verifies the capability-filtered
        # source catalog before that presentation layer is applied.
        skip_tool_search_assembly=True,
    )

    assert "video_generate" in {item["function"]["name"] for item in definitions}


def test_read_only_vault_hides_zettlab_video_even_when_capability_is_enabled(monkeypatch):
    from agent import video_gen_registry
    from hermes_cli.tools_config import _get_platform_tools
    import model_tools
    from plugins import zettlab_media_client as client
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import video_generation_tool as video_tool

    video_gen_registry._reset_for_tests()
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    monkeypatch.setattr(
        "hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        video_tool, "_read_configured_video_provider", lambda: "zettlab"
    )
    monkeypatch.setattr(
        client._SESSION,
        "get",
        lambda *args, **kwargs: _Resp({
            "video": {
                "enabled": True,
                "default_model": "seedance-v1",
                "models": [{"id": "seedance-v1"}],
            },
        }),
    )

    enabled = _get_platform_tools(
        {
            "video_gen": {"provider": "zettlab"},
            "platform_toolsets": {
                "zet_agent": [
                    "markdown_vault",
                    "todo",
                    "clarify",
                    "no_mcp",
                    "video_gen",
                ]
            },
        },
        "zet_agent",
        include_default_mcp_servers=False,
    )
    model_tools._clear_tool_defs_cache()
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=sorted(enabled),
        quiet_mode=True,
    )

    assert "video_gen" not in enabled
    assert "video_generate" not in {
        item["function"]["name"] for item in definitions
    }


def test_zet_agent_hides_video_tool_when_zettlab_is_disabled_even_if_another_provider_is_available(monkeypatch):
    from agent import video_gen_registry
    from agent.video_gen_provider import VideoGenProvider
    from hermes_cli.tools_config import _get_platform_tools
    import model_tools
    from plugins import zettlab_media_client as client
    from plugins.video_gen.zettlab import ZettlabVideoGenProvider
    from tools import video_generation_tool as video_tool

    class _AvailableThirdPartyProvider(VideoGenProvider):
        @property
        def name(self):
            return "third-party"

        def generate(self, prompt, **kwargs):
            raise AssertionError("configured Zettlab provider must remain authoritative")

    video_gen_registry._reset_for_tests()
    video_gen_registry.register_provider(ZettlabVideoGenProvider())
    video_gen_registry.register_provider(_AvailableThirdPartyProvider())
    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", lambda *args, **kwargs: None)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(video_tool, "_read_configured_video_provider", lambda: "zettlab")
    monkeypatch.setattr(client._SESSION, "get", lambda url, timeout, allow_redirects, stream: _Resp({
        "video": {
            "enabled": False,
            "default_model": "",
            "models": [],
        },
    }))

    enabled = _get_platform_tools(
        {
            "video_gen": {"provider": "zettlab"},
            "platform_toolsets": {"zet_agent": ["hermes-zet-agent", "cronjob"]},
        },
        "zet_agent",
        include_default_mcp_servers=False,
    )
    model_tools._clear_tool_defs_cache()
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=sorted(enabled),
        quiet_mode=True,
    )

    assert "video_generate" not in {item["function"]["name"] for item in definitions}
