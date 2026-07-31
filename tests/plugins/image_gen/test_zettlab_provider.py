from __future__ import annotations

import base64
import io
import multiprocessing
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import requests

from plugins.image_gen.zettlab import ZettlabImageGenProvider, _gateway_aspect_ratio, register


PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(
    b"\x89PNG\r\n\x1a\nsource"
).decode("ascii")


def _spawn_parent_watchdog_probe(output):
    from plugins import zettlab_media_client as client

    context = multiprocessing.get_context("spawn")
    child = context.Process(target=client._watch_parent, args=(os.getpid(),))
    child.start()
    output.put(child.pid)
    time.sleep(30)


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _ErrorResp(_Resp):
    def __init__(self, status_code, data):
        super().__init__(data)
        self.status_code = status_code
        self.content = b"error"

    def raise_for_status(self):
        raise requests.HTTPError(response=self)


def test_zettlab_image_provider_reads_capabilities(monkeypatch):
    from plugins import zettlab_media_client as client

    def fake_get(url, timeout, allow_redirects, stream):
        assert url == "http://127.0.0.1:9090/api/v1/ai-proxy/v1/media/generation-capabilities"
        assert 0 < timeout <= client.CAPABILITY_TIMEOUT
        assert allow_redirects is False
        assert stream is True
        return _Resp({
            "image": {
                "enabled": True,
                "models": [{
                    "id": "seedream-v4",
                    "display_name": "Seedream V4",
                    "modalities": ["text", "image"],
                }],
                "limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
            },
            "video": {"enabled": False, "models": []},
        })

    monkeypatch.setattr(client._SESSION, "get", fake_get)

    provider = ZettlabImageGenProvider()
    assert provider.is_available() is True
    assert provider.default_model() == "seedream-v4"
    assert provider.list_models()[0]["display"] == "Seedream V4"
    assert provider.capabilities()["max_reference_images"] == 0


def test_zettlab_capabilities_response_is_bounded_and_closed(monkeypatch):
    from plugins import zettlab_media_client as client

    response = requests.Response()
    response.status_code = 200
    raw = io.BytesIO(b"x" * (client.MAX_CAPABILITY_RESPONSE_BYTES + 1))
    response.raw = raw
    monkeypatch.setattr(client._SESSION, "get", lambda *args, **kwargs: response)

    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.get_capabilities("image")

    assert raw.closed is True


def test_zettlab_image_capabilities_preserve_image_only_modality(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "selected_model_capability", lambda media_type: ({
        "limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
    }, {
        "id": "image-only",
        "modalities": [" IMAGE "],
    }))

    assert ZettlabImageGenProvider().capabilities()["modalities"] == ["image"]


@pytest.mark.parametrize("invalid_limit", [None, 0, -1, True, "5242880"])
def test_zettlab_image_capabilities_hide_image_with_invalid_inline_limit(
    monkeypatch,
    invalid_limit,
):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(client, "selected_model_capability", lambda media_type: ({
        "limits": {"max_inline_image_bytes": invalid_limit},
    }, {
        "id": "image-only",
        "modalities": [" IMAGE "],
    }))

    assert ZettlabImageGenProvider().capabilities()["modalities"] == []


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
                "models": [{
                    "id": "seedream-v4",
                    "modalities": ["text", "image"],
                    "aspect_ratios": ["1:1", "16:9", "9:16"],
                    "resolutions": ["2K"],
                }],
                "limits": {
                    "provider_timeout_seconds": 300,
                    "finalization_timeout_seconds": 600,
                    "max_inline_image_bytes": 5 * 1024 * 1024,
                },
            },
        })

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        captured["allow_redirects"] = allow_redirects
        captured["stream"] = stream
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
        image_url=PNG_DATA_URI,
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
    assert captured["allow_redirects"] is False
    assert captured["stream"] is True
    assert captured["json"]["media_type"] == "image"
    assert captured["json"]["model"] == "seedream-v4"
    assert captured["json"]["output_count"] == 1
    assert captured["json"]["aspect_ratio"] == "1:1"
    assert captured["json"]["resolution"] == "2K"
    assert captured["json"]["input_image"] == PNG_DATA_URI
    assert "remote_media_inputs" not in captured["json"]


@pytest.mark.parametrize(
    ("aspect", "allowed", "expected"),
    [
        ("landscape", ["4:3", "16:9"], "16:9"),
        ("square", ["16:9", "1:1"], "1:1"),
        ("portrait", ["3:4", "9:16"], "9:16"),
        ("landscape", ["4:3"], "4:3"),
        ("square", ["square"], "square"),
    ],
)
def test_gateway_aspect_ratio_uses_model_capabilities(aspect, allowed, expected):
    assert _gateway_aspect_ratio(aspect, {"aspect_ratios": allowed}) == expected


def test_zettlab_image_generate_uses_gateway_default_when_model_is_omitted(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    captured = {}
    monkeypatch.setattr(client, "type_capability", lambda media_type: {
        "enabled": True,
        "default_model": "seedream-default",
        "models": [{"id": "seedream-default"}],
    })

    def fake_post(url, json, headers, timeout, allow_redirects, stream):
        assert allow_redirects is False
        assert stream is True
        captured.update(json)
        return _Resp({"job_id": "job-default", "status": "done", "assets": [{"url": "https://cdn.example/default.png"}]})

    monkeypatch.setattr(client._SESSION, "post", fake_post)
    got = ZettlabImageGenProvider().generate("make image")
    assert got["success"] is True
    assert captured["model"] == "seedream-default"


def test_zettlab_image_rejects_remote_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: (
            "seedream-v4",
            {
                "id": "seedream-v4",
                "modalities": ["text", "image"],
                "_type_limits": {"max_inline_image_bytes": 5 * 1024 * 1024},
            },
        ),
    )
    got = ZettlabImageGenProvider().generate("make image", image_url="http://example.com/a.png")
    assert got["success"] is False
    assert got["error_type"] == "ZettlabMediaError"
    assert "local image path or data URI" in got["error"]


def test_zettlab_image_only_model_requires_image_input(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client,
        "resolve_model_with_capability",
        lambda media_type, requested=None: (
            "image-only",
            {
                "id": "image-only",
                "modalities": [" IMAGE "],
                "_type_limits": {"max_inline_image_bytes": True},
            },
        ),
    )
    monkeypatch.setattr(
        client,
        "create_and_wait",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("request should not be sent")),
    )

    got = ZettlabImageGenProvider().generate("edit this image")

    assert got["success"] is False
    assert got["error_type"] == "missing_image"


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


def test_zettlab_media_client_uses_profile_scoped_proxy_origin_and_action_token(monkeypatch):
    from agent import secret_scope
    from plugins import zettlab_media_client as client

    previous_multiplex = secret_scope.is_multiplex_active()
    monkeypatch.setattr(client, "_config_section", lambda media_type: {})
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:9999/other-profile")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "other-token")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token",
    })
    try:
        assert client.base_url("image") == "http://127.0.0.1:9420/api/v1/ai-proxy/v1"
        assert client.action_headers() == {
            "X-Zettlab-Agent-Action-Token": "profile-token",
        }
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_zettlab_media_client_explicit_profile_proxy_base_wins(monkeypatch):
    from agent import secret_scope
    from plugins import zettlab_media_client as client

    previous_multiplex = secret_scope.is_multiplex_active()
    monkeypatch.setattr(client, "_config_section", lambda media_type: {})
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({
        "ZETTLAB_AI_PROXY_BASE_URL": "http://127.0.0.1:9430/custom/ai-proxy/v1",
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:9420/api/v1/internal/chat/append",
    })
    try:
        assert client.base_url("video") == "http://127.0.0.1:9430/custom/ai-proxy/v1"
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


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
        assert kwargs["allow_redirects"] is False
        result = next(polls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client._SESSION, "get", fake_get)
    monkeypatch.setattr(client._SESSION, "delete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cleanup should not run")))

    job = client.create_and_wait(media_type="image", model="seedream-v4", prompt="retry", payload={}, timeout_seconds=10)
    assert job["status"] == "done"


def test_zettlab_create_timeout_uses_capability_budget(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_timeout_from_capability", lambda media_type: 900)
    received = {}

    def fake_post(*args, **kwargs):
        received["timeout"] = kwargs["timeout"]
        return _Resp({"job_id": "job-sync", "status": "done", "assets": [{"url": "https://cdn.example/done.png"}]})

    monkeypatch.setattr(client._SESSION, "post", fake_post)
    job = client.create_and_wait(media_type="image", model="seedream-v4", prompt="slow", payload={})

    assert 899 <= received["timeout"] <= 900
    assert job["status"] == "done"


def test_zettlab_http_worker_kills_timed_out_process_and_recovers(monkeypatch):
    from plugins import zettlab_media_client as client

    class FakeProcess:
        def __init__(self):
            self.alive = True
            self.terminated = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            return None

        def kill(self):
            self.alive = False

        def close(self):
            return None

    class FakeConnection:
        def __init__(self, result=None):
            self.result = result
            self.closed = False

        def send(self, request):
            raise AssertionError("parent must not synchronously write the request pipe")

        def poll(self, timeout):
            if self.result is None:
                time.sleep(timeout)
                return False
            return True

        def recv(self):
            return self.result

        def close(self):
            self.closed = True

    timed_out_process = FakeProcess()
    timed_out_connection = FakeConnection()
    recovered_process = FakeProcess()
    recovered_process.alive = False
    recovered_connection = FakeConnection({
        "status_code": 200,
        "body": b'{"job_id":"recovered","status":"done"}',
    })
    attempts = iter([
        (timed_out_process, timed_out_connection),
        (recovered_process, recovered_connection),
    ])
    worker = client._MediaHTTPWorker()

    def start_next_worker(request, deadline):
        if worker._process is None:
            worker._process, worker._connection = next(attempts)

    monkeypatch.setattr(worker, "_ensure_started", start_next_worker)
    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError, match="deadline exceeded"):
        worker.request("POST", "http://127.0.0.1/media/generation-jobs", deadline=time.monotonic() + 0.05)
    assert time.monotonic() - started < 0.15
    assert timed_out_process.terminated is True
    assert timed_out_connection.closed is True

    response = worker.request("GET", "http://127.0.0.1/media/generation-jobs/recovered", deadline=time.monotonic() + 1)
    assert client._bounded_response_json(response, client.MAX_MEDIA_RESPONSE_BYTES)["job_id"] == "recovered"
    worker.close()


def test_zettlab_http_worker_times_out_during_spawn_and_recovers():
    from plugins import zettlab_media_client as client

    class FakeProcess:
        def __init__(self, start_delay=0):
            self.start_delay = start_delay
            self.alive = False
            self.terminated = False

        def start(self):
            time.sleep(self.start_delay)
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            return None

        def kill(self):
            self.alive = False

        def close(self):
            return None

    class FakeConnection:
        def __init__(self, result=None):
            self.result = result

        def poll(self, timeout):
            return self.result is not None

        def recv(self):
            return self.result

        def close(self):
            return None

    slow_process = FakeProcess(start_delay=0.2)
    fast_process = FakeProcess()
    attempts = iter([
        (slow_process, FakeConnection(), FakeConnection()),
        (fast_process, FakeConnection({"status_code": 200, "body": b'{"status":"done"}'}), FakeConnection()),
    ])

    class FakeContext:
        def RawArray(self, typecode, size):
            return bytearray(size)

        def Pipe(self):
            self.process, parent, child = next(attempts)
            return parent, child

        def Process(self, **kwargs):
            return self.process

    worker = client._MediaHTTPWorker()
    worker._context = FakeContext()
    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError, match="start deadline exceeded"):
        worker.request("POST", "http://127.0.0.1/media/generation-jobs", deadline=time.monotonic() + 0.05)
    assert time.monotonic() - started < 0.15

    response = worker.request("GET", "http://127.0.0.1/media/generation-jobs/recovered", deadline=time.monotonic() + 1)
    assert client._bounded_response_json(response, client.MAX_MEDIA_RESPONSE_BYTES)["status"] == "done"
    deadline = time.monotonic() + 1
    while not slow_process.terminated and time.monotonic() < deadline:
        time.sleep(0.01)
    assert slow_process.terminated is True
    worker.close()


def test_zettlab_http_worker_bounds_permanently_stalled_starters():
    from plugins import zettlab_media_client as client

    release = threading.Event()

    class StalledProcess:
        alive = False
        terminated = False

        def start(self):
            release.wait(timeout=2)
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            return None

        def kill(self):
            self.alive = False

        def close(self):
            return None

    class FakeConnection:
        def close(self):
            return None

    processes = [StalledProcess(), StalledProcess()]
    attempts = iter(processes)

    class FakeContext:
        def RawArray(self, typecode, size):
            return bytearray(size)

        def Pipe(self):
            return FakeConnection(), FakeConnection()

        def Process(self, **kwargs):
            return next(attempts)

    worker = client._MediaHTTPWorker()
    worker._context = FakeContext()
    try:
        for _ in range(2):
            with pytest.raises(client.ZettlabMediaDeadlineError, match="start deadline exceeded"):
                worker.request("GET", "http://127.0.0.1/media/generation-capabilities", deadline=time.monotonic() + 0.03)
        started = time.monotonic()
        with pytest.raises(client.ZettlabMediaDeadlineError, match="starter capacity exhausted"):
            worker.request("GET", "http://127.0.0.1/media/generation-capabilities", deadline=time.monotonic() + 1)
        assert time.monotonic() - started < 0.1
    finally:
        release.set()

    deadline = time.monotonic() + 1
    while not all(process.terminated for process in processes) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert all(process.terminated for process in processes)
    assert client._STARTER_CAPACITY.acquire(timeout=1)
    assert client._STARTER_CAPACITY.acquire(timeout=1)
    client._STARTER_CAPACITY.release()
    client._STARTER_CAPACITY.release()


def test_zettlab_http_worker_cleans_up_after_prespawn_failure():
    from plugins import zettlab_media_client as client

    class BrokenContext:
        def RawArray(self, typecode, size):
            raise OSError("shared memory unavailable")

    worker = client._MediaHTTPWorker()
    worker._context = BrokenContext()

    with pytest.raises(requests.ConnectionError, match="failed to start"):
        worker.request("GET", "http://127.0.0.1/media/generation-capabilities", deadline=time.monotonic() + 1)

    assert worker._process is None
    assert worker._connection is None


def test_zettlab_http_worker_cleanup_continues_after_individual_errors():
    from plugins import zettlab_media_client as client

    class BrokenConnection:
        def close(self):
            raise OSError("close failed")

    class BrokenProcess:
        alive = True
        killed = False
        closed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            raise OSError("terminate failed")

        def join(self, timeout=None):
            return None

        def kill(self):
            self.killed = True
            self.alive = False

        def close(self):
            self.closed = True

    worker = client._MediaHTTPWorker()
    process = BrokenProcess()
    worker._connection = BrokenConnection()
    worker._process = process

    worker._reset(force=True)

    assert process.killed is True
    assert process.closed is True
    assert worker._process is None
    assert worker._connection is None


def test_zettlab_http_worker_recovers_after_real_header_stall():
    from plugins import zettlab_media_client as client

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/slow":
                time.sleep(1)
                return
            body = b'{"job_id":"real-worker","status":"done"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    worker = client._MediaHTTPWorker()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(client.ZettlabMediaDeadlineError, match="deadline exceeded"):
            worker.request("GET", base + "/slow", deadline=time.monotonic() + 0.3)
        response = worker.request("GET", base + "/ok", deadline=time.monotonic() + 3)
        assert client._bounded_response_json(response, client.MAX_MEDIA_RESPONSE_BYTES)["job_id"] == "real-worker"
    finally:
        worker.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)


@pytest.mark.live_system_guard_bypass
def test_zettlab_parent_watchdog_exits_after_parent_is_killed():
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    parent = context.Process(target=_spawn_parent_watchdog_probe, args=(output,))
    parent.start()
    child_pid = output.get(timeout=5)
    try:
        parent.kill()
        parent.join(timeout=2)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("media HTTP child survived its parent")
    finally:
        if parent.is_alive():
            parent.kill()
            parent.join(timeout=1)
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output.close()


def test_zettlab_http_worker_resets_after_pipe_poll_error(monkeypatch):
    from plugins import zettlab_media_client as client

    class FakeProcess:
        alive = True
        terminated = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def join(self, timeout=None):
            return None

        def kill(self):
            self.alive = False

        def close(self):
            return None

    class BrokenConnection:
        closed = False

        def send(self, request):
            raise AssertionError("parent must not synchronously write the request pipe")

        def poll(self, timeout):
            raise OSError("broken pipe")

        def close(self):
            self.closed = True

    worker = client._MediaHTTPWorker()
    process = FakeProcess()
    connection = BrokenConnection()
    worker._process = process
    worker._connection = connection
    monkeypatch.setattr(worker, "_ensure_started", lambda request, deadline: None)

    with pytest.raises(requests.ConnectionError, match="poll failed"):
        worker.request("GET", "http://127.0.0.1/media/generation-jobs/job-pipe", deadline=time.monotonic() + 1)

    assert process.terminated is True
    assert connection.closed is True
    assert worker._process is None


def test_zettlab_error_response_reads_only_bounded_stream():
    from plugins import zettlab_media_client as client

    response = requests.Response()
    response.status_code = 502
    response.raw = io.BytesIO(b"x" * (client.MAX_ERROR_RESPONSE_BYTES + 1024))

    error = client._response_error(response)

    assert "HTTP 502" in str(error)
    assert response.raw.tell() == client.MAX_ERROR_RESPONSE_BYTES + 1


def test_zettlab_create_preserves_failed_job_semantics(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_timeout_from_capability", lambda media_type: 10)
    monkeypatch.setattr(
        client._SESSION,
        "post",
        lambda *args, **kwargs: _Resp({
            "job_id": "job-billing",
            "status": "failed",
            "error_code": "upstream_insufficient_credits",
            "error_message": "media generation failed",
            "retryable": False,
        }),
    )

    with pytest.raises(
        client.ZettlabMediaError,
        match=r"code=upstream_insufficient_credits.*retryable=false.*job_id=job-billing",
    ):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="billing", payload={})


def test_zettlab_create_preserves_structured_http_error(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_timeout_from_capability", lambda media_type: 10)
    monkeypatch.setattr(
        client._SESSION,
        "post",
        lambda *args, **kwargs: _ErrorResp(402, {
            "error": {"code": "insufficient_credits", "message": "credits exhausted", "retryable": False},
            "job_id": "job-402",
        }),
    )

    with pytest.raises(
        client.ZettlabMediaError,
        match=r"HTTP 402.*code=insufficient_credits.*retryable=false.*job_id=job-402",
    ):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="billing", payload={})


def test_zettlab_poll_interrupt_preserves_active_job(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-interrupt", "status": "running"}))
    monkeypatch.setattr(client, "is_interrupted", lambda: True)
    monkeypatch.setattr(client._SESSION, "delete", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("active job must be preserved")))

    with pytest.raises(client.ZettlabMediaError, match=r"interrupted; job_id=job-interrupt"):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="stop", payload={}, timeout_seconds=10)


def test_zettlab_poll_attempt_timeout_retries_until_total_deadline(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-slow-poll", "status": "running"}))
    attempts = 0

    def timeout_first_poll(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise client.ZettlabMediaDeadlineError("single poll timed out")
        return _Resp({"job_id": "job-slow-poll", "status": "done", "assets": [{"url": "https://cdn.example/done.png"}]})

    monkeypatch.setattr(client._SESSION, "get", timeout_first_poll)
    job = client.create_and_wait(media_type="image", model="seedream-v4", prompt="retry", payload={}, timeout_seconds=10)

    assert job["status"] == "done"
    assert attempts == 2


def test_zettlab_malformed_poll_response_preserves_job_id(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-malformed", "status": "running"}))
    response = requests.Response()
    response.status_code = 200
    response.raw = io.BytesIO(b"{not-json")
    monkeypatch.setattr(client._SESSION, "get", lambda *args, **kwargs: response)

    with pytest.raises(client.ZettlabMediaError, match=r"not valid JSON.*job_id=job-malformed"):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="bad", payload={}, timeout_seconds=10)


def test_zettlab_poll_transport_failure_preserves_job_id(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-pipe", "status": "running"}))
    monkeypatch.setattr(
        client._SESSION,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("pipe failed")),
    )

    with pytest.raises(client.ZettlabMediaError, match=r"timed out.*job_id=job-pipe"):
        client.create_and_wait(media_type="image", model="seedream-v4", prompt="pipe", payload={}, timeout_seconds=0.02)


def test_zettlab_poll_continues_past_transient_failures(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "media-token")
    monkeypatch.setattr(client, "_interruptible_sleep", lambda delay: None)
    monkeypatch.setattr(client._SESSION, "post", lambda *args, **kwargs: _Resp({"job_id": "job-error", "status": "running"}))
    polls = iter([
        requests.ConnectionError("offline-1"),
        requests.ConnectionError("offline-2"),
        requests.ConnectionError("offline-3"),
        requests.ConnectionError("offline-4"),
        requests.ConnectionError("offline-5"),
        _Resp({"job_id": "job-error", "status": "done", "assets": [{"url": "https://cdn.example/recovered.png"}]}),
    ])

    def fake_get(*args, **kwargs):
        result = next(polls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(client._SESSION, "get", fake_get)
    job = client.create_and_wait(media_type="image", model="seedream-v4", prompt="recover", payload={}, timeout_seconds=10)
    assert job["status"] == "done"


def test_register_calls_image_provider_registry():
    calls = []
    register(SimpleNamespace(register_image_gen_provider=lambda provider: calls.append(provider)))
    assert isinstance(calls[0], ZettlabImageGenProvider)
