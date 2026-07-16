from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

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

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    provider = ZettlabImageGenProvider()
    assert provider.is_available() is True
    assert provider.default_model() == "seedream-v4"
    assert provider.list_models()[0]["display"] == "Seedream V4"
    assert provider.capabilities()["max_reference_images"] == 3


def test_zettlab_provider_uses_gateway_default_model(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "seedream-pro",
        "models": [{"id": "seedream-fast"}, {"id": "seedream-pro"}],
    })

    assert client.default_model("image") == "seedream-pro"


def test_zettlab_provider_default_model_uses_one_capability_snapshot(monkeypatch):
    from plugins import zettlab_media_client as client

    calls = 0

    def capability(media_type):
        nonlocal calls
        calls += 1
        return {"enabled": True, "models": [{"id": "legacy-first"}]}

    monkeypatch.setattr(client, "type_capability", capability)
    assert client.default_model("image") == "legacy-first"
    assert calls == 1


def test_zettlab_provider_rejects_disabled_or_empty_capability(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "type_capability", lambda media_type: {"enabled": False, "models": [{"id": "hidden"}]})
    assert client.default_model("image") is None
    monkeypatch.setattr(client, "type_capability", lambda media_type: {"enabled": True, "default_model": "missing", "models": []})
    assert client.default_model("image") is None
    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "missing",
        "models": [{"id": "catalog-first"}],
    })
    assert client.default_model("image") is None


def test_zettlab_provider_validates_local_model_override(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "_config_section", lambda media_type: {"model": "local-override"})
    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "gateway-default",
        "models": [{"id": "local-override"}, {"id": "gateway-default"}],
    })
    assert client.default_model("image") == "local-override"

    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "gateway-default",
        "models": [{"id": "gateway-default"}],
    })
    assert client.default_model("image") is None


def test_zettlab_image_generate_creates_media_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
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

    monkeypatch.setattr(client._SESSION, "get", fake_get)
    monkeypatch.setattr(client._SESSION, "post", fake_post)

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
    assert captured["headers"]["X-Zettlab-Agent-Action-Token"] == "media-token"
    assert captured["json"]["media_type"] == "image"
    assert captured["json"]["model"] == "seedream-v4"
    assert captured["json"]["output_count"] == 2
    assert captured["json"]["remote_media_inputs"] == [
        {"url": "https://example.com/source.png", "role": "source"},
        {"url": "https://example.com/ref.png", "role": "reference"},
    ]


def test_zettlab_image_generate_uses_gateway_default_when_model_is_omitted(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "seedream-default",
        "models": [{"id": "seedream-default"}],
    })

    def fake_post(url, json, headers, timeout):
        captured.update(json)
        return _Resp({"job_id": "job-default", "status": "done", "assets": [{"url": "https://cdn.example/default.png"}]})

    monkeypatch.setattr(client._SESSION, "post", fake_post)
    got = ZettlabImageGenProvider().generate("make image")
    assert got["success"] is True
    assert captured["model"] == "seedream-default"


def test_zettlab_image_rejects_non_https_remote_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "resolve_model", lambda media_type, requested=None: "seedream-v4")
    got = ZettlabImageGenProvider().generate("make image", image_url="http://example.com/a.png")
    assert got["success"] is False
    assert got["error_type"] == "ZettlabMediaError"
    assert "https URL" in got["error"]


@pytest.mark.parametrize("value", [
    "https://localhost/a.png",
    "https://127.0.0.1/a.png",
    "https://example.com:8443/a.png",
    "https://example.com/a.png#fragment",
])
def test_zettlab_remote_input_matches_gateway_url_policy(value):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError):
        client.validate_remote_url(value, label="image_url")


def test_zettlab_ai_proxy_rejects_non_loopback_base_url(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AI_PROXY_BASE_URL", "https://attacker.example/ai-proxy/v1")
    with pytest.raises(client.ZettlabMediaError, match="loopback"):
        client.base_url("image")


def test_zettlab_ai_proxy_ignores_environment_proxies(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    settings = client._SESSION.merge_environment_settings(
        client.DEFAULT_BASE_URL,
        {},
        None,
        None,
        None,
    )

    assert client._SESSION.trust_env is False
    assert settings["proxies"] == {}


def test_zettlab_poll_retries_transient_error_without_cleanup(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-retry", "status": "running"}))
    polls = iter([requests.ConnectionError("temporary"), _Resp({"job_id": "job-retry", "status": "done", "assets": [{"url": "https://cdn.example/done.png"}]})])

    def fake_get(*args, **kwargs):
        result = next(polls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client._SESSION, "get", fake_get)
    monkeypatch.setattr(client._SESSION, "delete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cleanup should not run")))

    job = client.create_and_wait(media_type="image", model="seedream-v4", prompt="retry", payload={}, timeout_seconds=10)
    assert job["status"] == "done"


def test_zettlab_poll_interrupt_deletes_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-interrupt", "status": "running"}))
    monkeypatch.setattr(client, "is_interrupted", lambda: True)
    deleted = []
    monkeypatch.setattr(client._SESSION, "delete", lambda url, **kwargs: deleted.append(url) or _Resp({}))

    with pytest.raises(client.ZettlabMediaError, match="interrupted"):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="stop", payload={}, timeout_seconds=10)
    assert deleted and deleted[0].endswith("/media/generation-jobs/job-interrupt")


def test_zettlab_poll_exhaustion_deletes_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-error", "status": "running"}))
    monkeypatch.setattr(client._SESSION, "get", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    deleted = []
    monkeypatch.setattr(client._SESSION, "delete", lambda url, **kwargs: deleted.append(url) or _Resp({}))

    with pytest.raises(requests.ConnectionError):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="fail", payload={}, timeout_seconds=10)
    assert deleted and deleted[0].endswith("/media/generation-jobs/job-error")


def test_register_calls_image_provider_registry():
    calls = []
    register(SimpleNamespace(register_image_gen_provider=lambda provider: calls.append(provider)))
    assert isinstance(calls[0], ZettlabImageGenProvider)
