"""Capability-probe cost and failure visibility.

Regression cover for a field incident: the probe ran through the media worker
pool, so every tool-definition pass spawned a subprocess that re-imported the
Hermes entry point. On device that raced ``CAPABILITY_TIMEOUT``, and losing the
race removed ``image_generate`` / ``video_generate`` from the tool list with no
log line anywhere — the model then told users the capability did not exist.
"""

from __future__ import annotations

import logging
import socket
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
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _Resp(CAPABILITIES))

    assert client.get_capabilities("image")["image"]["enabled"] is True


def test_successful_probe_is_reused_within_the_ttl(client, monkeypatch):
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(1)
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", fake_get)

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

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", flaky_get)

    assert client.is_available("image") is False
    assert client.is_available("image") is True
    assert len(attempts) == 2


def test_cached_document_is_isolated_from_callers(client, monkeypatch):
    """Callers get their own copy; mutating it must not poison the cache."""
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _Resp(CAPABILITIES))

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

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", fake_get)

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

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", fake_get)
    client.get_capabilities("image")

    assert responses and responses[0].closed is True


def test_probe_timeout_is_env_overridable(client, monkeypatch):
    """Every constant on this path was hard-coded, leaving ops no lever."""
    seen = {}

    def fake_get(url, timeout, allow_redirects, stream):
        seen["timeout"] = timeout
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", fake_get)

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
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _Resp(CAPABILITIES))

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
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _stream_resp())

    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError):
        client.get_capabilities("image")
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"deadline must cancel the stalled read, took {elapsed:.1f}s"
    assert released.is_set(), "expiry must close the response to unblock the reader"


def test_stalled_headers_phase_is_bounded(client, monkeypatch):
    """The budget must cover `session.get()`, not just the body read.

    `get()` only returns once the status line and headers have arrived, so a
    peer trickling headers stalls there — before any response object exists to
    close. Bounding only the body read leaves this hole open, and it hangs agent
    construction just the same.
    """
    entered = threading.Event()

    def stalled_get(*_args, **_kwargs):
        entered.set()
        time.sleep(30)  # never returns a response within the budget

    monkeypatch.setattr(client, "_capability_timeout", lambda: 0.2)
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", stalled_get)

    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError):
        client.get_capabilities("image")
    elapsed = time.monotonic() - started

    assert entered.is_set()
    assert elapsed < 5.0, f"headers stall must not hang the caller, took {elapsed:.1f}s"


def test_concurrent_probes_share_one_request(client, monkeypatch):
    """Agents built concurrently must not each fire the same loopback GET.

    A multiplexed gateway builds several agents at once and every build probes
    image and video, so a cold cache would otherwise multiply one identical
    request by the number of agents in flight.
    """
    calls = []
    release = threading.Event()

    def slow_get(*_args, **_kwargs):
        calls.append(1)
        release.wait(5)
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", slow_get)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(client.get_capabilities("image")))
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    time.sleep(0.2)  # let every caller attach to the in-flight probe
    release.set()
    for thread in threads:
        thread.join(10)

    assert len(calls) == 1, f"concurrent callers must share one probe, saw {len(calls)}"
    assert len(results) == 6
    assert all(r["image"]["enabled"] is True for r in results)


def test_concurrent_probes_are_never_denied(client, monkeypatch):
    """No caller may be turned away just because others are probing.

    An earlier revision capped concurrent probes with a semaphore and failed
    fast once it was full. That turned a healthy local-server into hidden
    media tools for whichever agents lost the race — the very symptom this
    change exists to remove.
    """
    urls = iter([f"http://127.0.0.1:{19090 + i}/api/v1/ai-proxy/v1" for i in range(8)])
    url_lock = threading.Lock()

    def next_url(_media_type):
        with url_lock:
            return next(urls)

    barrier = threading.Barrier(8, timeout=10)

    def concurrent_get(*_args, **_kwargs):
        barrier.wait()  # every probe must be in flight at the same moment
        return _Resp(CAPABILITIES)

    monkeypatch.setattr(client, "base_url", next_url)
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", concurrent_get)

    failures = []
    results = []

    def probe():
        try:
            results.append(client.get_capabilities("image"))
        except Exception as exc:  # noqa: BLE001 — recorded as a failure below
            failures.append(exc)

    threads = [threading.Thread(target=probe) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)

    assert not failures, f"no caller may be denied a probe: {failures}"
    assert len(results) == 8


def test_deadline_shuts_the_socket_down_so_the_probe_ends(client, monkeypatch):
    """The deadline must terminate the probe, not just stop waiting on it.

    This is the property the whole transport choice buys, and only a peer that
    keeps the socket *busy* tests it: this one dribbles the status line a byte
    at a time, faster than the idle timeout, so no socket timeout will ever
    fire and nothing ends the exchange on its own. The caller has to return on
    time *and* the probe thread has to be gone shortly after — otherwise
    abandoned probes pile up, and any cap on them eventually wedges the tool
    for good.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = []
    stop = threading.Event()

    def accept_and_dribble():
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        accepted.append(conn)
        try:
            for byte in b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n":
                if stop.wait(0.1):
                    return
                conn.sendall(bytes([byte]))  # never completes the headers
        except OSError:
            pass

    acceptor = threading.Thread(target=accept_and_dribble, daemon=True)
    acceptor.start()

    monkeypatch.setattr(client, "base_url", lambda mt: f"http://127.0.0.1:{port}/api/v1/ai-proxy/v1")
    monkeypatch.setattr(client, "_capability_timeout", lambda: 0.3)

    before = {t.ident for t in threading.enumerate() if t.name == "zettlab-capability-probe"}
    started = time.monotonic()
    try:
        with pytest.raises(client.ZettlabMediaDeadlineError):
            client.get_capabilities("image")
        elapsed = time.monotonic() - started
        assert elapsed < 3.0, f"caller must return on its deadline, took {elapsed:.1f}s"

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            live = {t.ident for t in threading.enumerate()
                    if t.name == "zettlab-capability-probe"} - before
            if not live:
                break
            time.sleep(0.05)
        assert not live, "cancelled probe must actually end, not linger on the socket"
    finally:
        stop.set()
        for conn in accepted:
            conn.close()
        listener.close()
        acceptor.join(1)


def test_deadline_shuts_detached_response_socket_so_slow_body_probe_ends(
    client, monkeypatch
):
    """Cancellation must reach a socket detached by ``Connection: close``.

    ``HTTPConnection.getresponse()`` clears ``conn.sock`` after headers when
    the response cannot be reused, while ``HTTPResponse`` keeps the socket
    alive through its buffered reader. A peer that then dribbles the body can
    strand the probe unless the transport retained that socket separately.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    accepted = []
    body_started = threading.Event()
    stop = threading.Event()

    def accept_and_dribble_body():
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        accepted.append(conn)
        try:
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 1000000\r\n"
                b"Connection: close\r\n\r\n"
            )
            body_started.set()
            while not stop.wait(0.1):
                conn.sendall(b"{")
        except OSError:
            pass

    acceptor = threading.Thread(target=accept_and_dribble_body, daemon=True)
    acceptor.start()

    monkeypatch.setattr(
        client,
        "base_url",
        lambda mt: f"http://127.0.0.1:{port}/api/v1/ai-proxy/v1",
    )
    monkeypatch.setattr(client, "_capability_timeout", lambda: 0.3)

    before = {
        t.ident
        for t in threading.enumerate()
        if t.name == "zettlab-capability-probe"
    }
    started = time.monotonic()
    try:
        with pytest.raises(client.ZettlabMediaDeadlineError):
            client.get_capabilities("image")
        elapsed = time.monotonic() - started
        assert body_started.is_set(), "the peer must reach the slow-body phase"
        assert elapsed < 3.0, f"caller must return on its deadline, took {elapsed:.1f}s"

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            live = {
                t.ident
                for t in threading.enumerate()
                if t.name == "zettlab-capability-probe"
            } - before
            if not live:
                break
            time.sleep(0.05)
        assert not live, "cancelled slow-body probe must not retain its detached socket"
    finally:
        stop.set()
        for conn in accepted:
            conn.close()
        listener.close()
        acceptor.join(1)


def test_capability_response_closes_its_retained_socket_and_raw_stream(client):
    """Response cleanup releases both views of a detached connection."""

    class _Socket:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, how):
            self.shutdown_calls.append(how)

    class _Raw:
        status = 200
        reason = "OK"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    sock = _Socket()
    raw = _Raw()
    resp = client._CapabilityResponse(conn=None, raw=raw, sock=sock)

    resp.close()

    assert sock.shutdown_calls == [socket.SHUT_RDWR]
    assert raw.closed is True


def test_capability_response_enforces_the_size_cap(client):
    """The real transport's reader keeps the 256KB cap."""

    class _Flood:
        def read(self, n):
            return b"x" * n

    resp = client._CapabilityResponse(conn=None, raw=_Flood())
    with pytest.raises(client.ZettlabMediaError):
        resp.json()


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
    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _stream_resp())

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

    monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", lambda *a, **k: _stream_resp())

    with pytest.raises(client.ZettlabMediaError):
        client.get_capabilities("image")


def test_probe_failure_is_logged(client, monkeypatch, caplog):
    """Silent False here is what made the incident undiagnosable."""
    monkeypatch.setattr(
        client._CAPABILITY_TRANSPORT,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")),
    )

    with caplog.at_level(logging.WARNING, logger="plugins.zettlab_media_client"):
        assert client.is_available("image") is False

    assert any("capability probe failed" in record.getMessage()
               for record in caplog.records)


def _run_profile_scoped(check, caplog, level):
    from agent import secret_scope
    from tools.registry import _check_fn_cached

    check._profile_scope_sensitive = True
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        with caplog.at_level(level, logger="tools.registry"):
            return _check_fn_cached(check)
    finally:
        secret_scope.set_multiplex_active(previous)


def test_profile_scoped_check_returning_false_is_not_a_warning(caplog):
    """A plain False is the designed answer, not an alarm.

    Every optional tool whose prerequisite is simply absent answers False here,
    on every agent construction and uncached. Logging each one at warning
    rotated the real alarms out of agent.log within minutes, so the expected
    case belongs at debug — while still leaving a trace to follow.
    """
    def check_returns_false():
        return False

    assert _run_profile_scoped(check_returns_false, caplog, logging.DEBUG) is False

    messages = [r for r in caplog.records if "returned False" in r.getMessage()]
    assert messages, "the expected case must still leave a trace"
    assert all(r.levelno == logging.DEBUG for r in messages)


def test_profile_scoped_check_that_raises_is_a_warning(caplog):
    """A raise is a malfunction, and the check may have had no chance to log.

    ``image_generate`` sits behind such a check, so this is the line that
    explains a tool vanishing for a reason nobody chose.
    """
    def check_explodes():
        raise RuntimeError("scope lookup failed")

    assert _run_profile_scoped(check_explodes, caplog, logging.WARNING) is False

    assert any(r.levelno == logging.WARNING and "raised" in r.getMessage()
               for r in caplog.records)


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
