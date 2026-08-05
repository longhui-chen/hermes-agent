"""Capability-probe cost and failure visibility.

Regression cover for a field incident: the probe ran through the media worker
pool, so every tool-definition pass spawned a subprocess that re-imported the
Hermes entry point. On device that raced ``CAPABILITY_TIMEOUT``, and losing the
race removed ``image_generate`` / ``video_generate`` from the tool list with no
log line anywhere — the model then told users the capability did not exist.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest
import requests


CAPABILITIES = {
    "image": {
        "enabled": True,
        "default_model": "seedream-v4",
        "models": [{"id": "seedream-v4", "modalities": ["text"]}],
    },
}


class _Resp:
    def __init__(self, data):
        self._data = data
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self._data

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    from plugins import zettlab_media_client as module

    monkeypatch.setattr(module, "base_url", lambda media_type: "http://127.0.0.1:19090/api/v1/ai-proxy/v1")
    return module


def _explode(*_args, **_kwargs):
    raise AssertionError("capability probe must not use the media worker pool")


def test_probe_does_not_use_the_subprocess_worker_pool(client, monkeypatch):
    """The worker pool costs an interpreter + entry-point re-import per call.

    A loopback GET with no body and a size-capped response does not need that
    isolation, and paying for it is what made the probe time out on device.
    """
    monkeypatch.setattr(client._SESSION, "get", _explode)
    monkeypatch.setattr(client._SESSION, "post", _explode)
    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _Resp(CAPABILITIES))

    assert client.get_capabilities("image")["image"]["enabled"] is True


def test_successful_probe_is_reused_within_the_ttl(client, monkeypatch):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", fake_get)

    assert client.get_capabilities("image")["image"]["enabled"] is True
    assert client.get_capabilities("image")["image"]["enabled"] is True
    assert client.is_available("image") is True
    assert len(calls) == 1, "cached capability document should serve later probes"


def test_failed_probe_is_not_cached(client, monkeypatch):
    """A blip must not hide the tool for the whole TTL.

    Negative caching here would reproduce the incident with a longer fuse: one
    slow probe would strip the tool for every turn in the next minute.
    """
    attempts = []

    def flaky_get(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.ConnectionError("probe blipped")
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", flaky_get)

    assert client.is_available("image") is False
    assert client.is_available("image") is True
    assert len(attempts) == 2


def test_cached_document_is_isolated_from_callers(client, monkeypatch):
    """Callers get their own copy; mutating it must not poison the cache."""
    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _Resp(CAPABILITIES))

    first = client.get_capabilities("image")
    first["image"]["enabled"] = False
    first["image"]["models"].clear()

    assert client.get_capabilities("image")["image"]["enabled"] is True
    assert client.is_available("image") is True


def test_invalidate_capability_cache_forces_a_refetch(client, monkeypatch):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", fake_get)

    client.get_capabilities("image")
    client.invalidate_capability_cache()
    client.get_capabilities("image")

    assert len(calls) == 2


def test_probe_response_is_closed(client, monkeypatch):
    responses = []

    def fake_get(*_args, **_kwargs):
        resp = _Resp(CAPABILITIES)
        responses.append(resp)
        return resp

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", fake_get)
    client.get_capabilities("image")

    assert responses and responses[0].closed is True


def test_probe_timeout_is_env_overridable(client, monkeypatch):
    """Every constant on this path was hard-coded, leaving ops no lever."""
    seen = {}

    def fake_get(url, timeout, allow_redirects, stream):
        seen["timeout"] = timeout
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", fake_get)

    client.get_capabilities("image")
    assert seen["timeout"] == client.CAPABILITY_TIMEOUT

    client.invalidate_capability_cache()
    monkeypatch.setenv("ZETTLAB_MEDIA_CAPABILITY_TIMEOUT", "12.5")
    client.get_capabilities("image")
    assert seen["timeout"] == 12.5


def test_cache_is_bounded(client, monkeypatch):
    """Keys embed the profile-scoped base URL, so the key space is not fixed.

    An unbounded resident cache of 256KB documents is exactly what the 2GB
    device budget forbids.
    """
    urls = iter([f"http://127.0.0.1:{19090 + i}/api/v1/ai-proxy/v1" for i in range(50)])
    monkeypatch.setattr(client, "base_url", lambda media_type: next(urls))
    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _Resp(CAPABILITIES))

    for _ in range(50):
        client.get_capabilities("image")

    assert len(client._capability_cache) <= client.MAX_CAPABILITY_CACHE_ENTRIES


def test_stalled_body_read_is_cancelled_at_the_deadline(client, monkeypatch):
    """The budget must hold even when the read itself never returns.

    `raw.read(n)` has BufferedReader semantics — it comes back only once n bytes
    have arrived or the stream ends. A peer dribbling one byte every few seconds
    keeps each recv under the socket timeout while stalling the read as a whole,
    so checking a clock *between* reads is useless: control never returns to do
    it. The bound has to be imposed from outside, which is what this asserts.
    """
    released = threading.Event()

    class _Stalled:
        def __init__(self):
            self.closed = False

        def read(self, _n, decode_content=False):
            # Blocks like the real thing; only a close() gets us out.
            released.wait(30)
            return b"{}"

    def _stream_resp():
        resp = requests.Response()
        resp.status_code = 200
        resp.raw = _Stalled()
        resp.close = lambda: released.set()
        return resp

    monkeypatch.setattr(client, "_capability_timeout", lambda: 0.2)
    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _stream_resp())

    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError):
        client.get_capabilities("image")
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"deadline must cancel the stalled read, took {elapsed:.1f}s"
    assert released.is_set(), "expiry must close the response to unblock the reader"


def test_stalled_probe_degrades_to_tool_unavailable(client, monkeypatch, caplog):
    """A stalled probe must hide the tool, never hang agent construction."""
    released = threading.Event()

    class _Stalled:
        def read(self, _n, decode_content=False):
            released.wait(30)
            return b"{}"

    def _stream_resp():
        resp = requests.Response()
        resp.status_code = 200
        resp.raw = _Stalled()
        resp.close = lambda: released.set()
        return resp

    monkeypatch.setattr(client, "_capability_timeout", lambda: 0.2)
    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _stream_resp())

    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="plugins.zettlab_media_client"):
        assert client.is_available("image") is False
    assert time.monotonic() - started < 5.0
    assert any("capability probe failed" in record.getMessage() for record in caplog.records)


def test_oversized_body_is_still_rejected(client, monkeypatch):
    """The chunked reader must keep the original size cap."""

    class _Flood:
        def read(self, n, decode_content=False):
            return b"x" * n

    def _stream_resp():
        resp = requests.Response()
        resp.status_code = 200
        resp.raw = _Flood()
        return resp

    monkeypatch.setattr(client._CAPABILITY_SESSION, "get", lambda *a, **k: _stream_resp())

    with pytest.raises(client.ZettlabMediaError):
        client.get_capabilities("image")


def test_probe_failure_is_logged(client, monkeypatch, caplog):
    """Silent False here is what made the incident undiagnosable."""
    monkeypatch.setattr(
        client._CAPABILITY_SESSION,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )

    with caplog.at_level(logging.WARNING, logger="plugins.zettlab_media_client"):
        assert client.is_available("image") is False

    assert any("capability probe failed" in record.getMessage()
               for record in caplog.records)


def test_profile_scoped_check_failure_is_logged(caplog):
    """A profile-scoped check that returns False used to log nothing at all.

    ``image_generate`` sits behind exactly such a check, so the tool vanished
    from the model's tool list without a single line to explain it.
    """
    from agent import secret_scope
    from tools.registry import _check_fn_cached

    def check_returns_false():
        return False

    check_returns_false._profile_scope_sensitive = True

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        with caplog.at_level(logging.WARNING, logger="tools.registry"):
            assert _check_fn_cached(check_returns_false) is False
    finally:
        secret_scope.set_multiplex_active(previous)

    assert any("returned False" in record.getMessage()
               for record in caplog.records)


def test_profile_scoped_check_is_never_cached():
    """Authorization must re-evaluate every pass, even though probes cost.

    The fix for probe cost belongs behind the check (a cached capability
    document), never in front of it — a cached verdict would let a revoked
    grant keep a tool alive.
    """
    from agent import secret_scope
    from tools.registry import _check_fn_cached

    verdicts = iter([True, False])

    def revocable_check():
        return next(verdicts)

    revocable_check._profile_scope_sensitive = True

    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        assert _check_fn_cached(revocable_check) is True
        assert _check_fn_cached(revocable_check) is False, "revoke must land immediately"
    finally:
        secret_scope.set_multiplex_active(previous)
