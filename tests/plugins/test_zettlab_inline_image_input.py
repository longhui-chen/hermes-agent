from __future__ import annotations

import base64
import os
import time

import pytest


PNG = b"\x89PNG\r\n\x1a\ninline-image"
JPEG = b"\xff\xd8\xffinline-image"
WEBP = b"RIFF\x0c\x00\x00\x00WEBPinline-image"


def _capability(limit: int = 1024):
    return {
        "modalities": ["text", "image"],
        "_type_limits": {"max_inline_image_bytes": limit},
    }


@pytest.mark.parametrize(
    ("raw", "mime"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")],
)
def test_inline_image_input_encodes_supported_local_file(tmp_path, monkeypatch, raw, mime):
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "source.bin"
    image_path.write_bytes(raw)

    got = client.inline_image_input(str(image_path), None, _capability())

    assert got == f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def test_inline_image_input_accepts_matching_data_uri():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    assert client.inline_image_input(value, None, _capability()) == value


def test_inline_image_input_normalizes_gateway_modality():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    capability = _capability()
    capability["modalities"] = [" IMAGE "]

    assert client.inline_image_input(value, None, capability) == value


@pytest.mark.parametrize("modalities", [[], ["future-mode"], [" FUTURE-MODE "]])
def test_normalized_modalities_ignore_unknown_values(modalities):
    from plugins import zettlab_media_client as client

    assert client.normalized_modalities({"modalities": modalities}) == []


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("https://example.com/source.png", "local image path or data URI"),
        ("data:image/jpeg;base64," + base64.b64encode(PNG).decode("ascii"), "does not match"),
        ("data:image/png;base64,not-base64!", "valid base64"),
    ],
)
def test_inline_image_input_rejects_unsafe_or_invalid_values(value, message):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match=message):
        client.inline_image_input(value, None, _capability())


@pytest.mark.parametrize(
    "value",
    [
        r"\\server\share\image.png",
        r"//server/share/image.png",
        r"\\?\UNC\server\share\image.png",
        r"\\.\C:\image.png",
        r"\??\C:\image.png",
        r"\Device\Mup\server\share\image.png",
    ],
)
def test_inline_image_input_rejects_windows_network_and_device_paths_before_read(
    monkeypatch,
    value,
):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client._FILE_WORKER,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe Windows path must be rejected before file handling")
        ),
    )

    with pytest.raises(client.ZettlabMediaError, match="network and device paths"):
        client.inline_image_input(value, None, _capability())


def test_inline_image_input_rejects_oversize_and_multiple_inputs():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.inline_image_input(value, None, _capability(limit=len(PNG) - 1))
    with pytest.raises(client.ZettlabMediaError, match="exactly one image"):
        client.inline_image_input(value, [value], _capability())


def test_inline_image_input_fails_closed_without_gateway_limit():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    with pytest.raises(client.ZettlabMediaError, match="not enabled"):
        client.inline_image_input(value, None, {"modalities": ["text", "image"]})


def test_inline_image_input_clamps_gateway_limit_to_local_hard_cap():
    from plugins import zettlab_media_client as client

    assert client._inline_image_limit(_capability(limit=16 * 1024 * 1024)) == (
        client.MAX_INLINE_IMAGE_BYTES
    )


def test_inline_image_input_rejects_oversize_file_before_encoding(tmp_path, monkeypatch):
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "oversize.png"
    image_path.write_bytes(PNG + b"x" * client.MAX_INLINE_IMAGE_BYTES)
    monkeypatch.setattr(
        base64,
        "b64encode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversize file must be rejected before encoding")
        ),
    )

    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.inline_image_input(
            str(image_path),
            None,
            _capability(limit=16 * 1024 * 1024),
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable")
def test_inline_image_input_rejects_fifo_without_blocking(tmp_path):
    from plugins import zettlab_media_client as client

    fifo_path = tmp_path / "image.fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(client.ZettlabMediaError, match="regular file"):
        client.inline_image_input(str(fifo_path), None, _capability())


def test_inline_image_input_preserves_file_safety_guard_in_worker(tmp_path):
    from plugins import zettlab_media_client as client

    blocked_path = tmp_path / ".env"
    blocked_path.write_bytes(PNG)

    with pytest.raises(ValueError, match="Access denied"):
        client.inline_image_input(str(blocked_path), None, _capability())


def test_media_file_worker_terminates_stalled_read_and_recovers(monkeypatch):
    from plugins import zettlab_media_client as client

    class FakeProcess:
        def __init__(self, *, alive=True):
            self.alive = alive
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

        def poll(self, timeout):
            if self.result is None:
                time.sleep(timeout)
                return False
            return True

        def recv(self):
            return self.result

        def close(self):
            self.closed = True

    stalled_process = FakeProcess()
    stalled_connection = FakeConnection()
    recovered_process = FakeProcess(alive=False)
    recovered_connection = FakeConnection({"length": len(PNG)})
    attempts = iter(
        [
            (stalled_process, stalled_connection, b""),
            (recovered_process, recovered_connection, PNG),
        ]
    )
    worker = client._MediaFileWorker()

    def start_next_worker(source, limit, deadline):
        process, connection, payload = next(attempts)
        worker._process = process
        worker._connection = connection
        worker._buffer = bytearray(limit)
        worker._buffer[: len(payload)] = payload

    monkeypatch.setattr(worker, "_ensure_started", start_next_worker)

    started = time.monotonic()
    with pytest.raises(client.ZettlabMediaDeadlineError, match="deadline exceeded"):
        worker.read("/stalled/image.png", 1024, deadline=time.monotonic() + 0.05)
    assert time.monotonic() - started < 0.15
    assert stalled_process.terminated is True
    assert stalled_connection.closed is True

    assert worker.read("/recovered/image.png", 1024, deadline=time.monotonic() + 1) == PNG
    worker.close()


def test_media_file_worker_interrupt_terminates_read(monkeypatch):
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

    class FakeConnection:
        closed = False

        def poll(self, timeout):
            return False

        def close(self):
            self.closed = True

    worker = client._MediaFileWorker()
    process = FakeProcess()
    connection = FakeConnection()

    def start_worker(source, limit, deadline):
        worker._process = process
        worker._connection = connection
        worker._buffer = bytearray(limit)

    monkeypatch.setattr(worker, "_ensure_started", start_worker)
    monkeypatch.setattr(client, "is_interrupted", lambda: True)

    with pytest.raises(client.ZettlabMediaError, match="interrupted"):
        worker.read("/stalled/image.png", 1024, deadline=time.monotonic() + 1)

    assert process.terminated is True
    assert connection.closed is True


def test_media_worker_parent_watch_uses_cross_platform_parent_sentinel(monkeypatch):
    from plugins import zettlab_media_client as client

    class FakeParent:
        def __init__(self):
            self.states = iter([True, False])

        def is_alive(self):
            return next(self.states)

    exits = []
    monkeypatch.setattr(client.multiprocessing, "parent_process", lambda: FakeParent())
    monkeypatch.setattr(client.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        client.os,
        "getppid",
        lambda: (_ for _ in ()).throw(
            AssertionError("multiprocessing parent sentinel must avoid PPID polling")
        ),
    )
    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(client.os, "_exit", fake_exit)

    with pytest.raises(SystemExit):
        client._watch_parent(12345)

    assert exits == [1]


def test_media_http_session_accepts_base64_sized_request(monkeypatch):
    from plugins import zettlab_media_client as client

    sentinel = object()
    monkeypatch.setattr(client._HTTP_WORKER, "request", lambda *args, **kwargs: sentinel)
    got = client._SESSION.post(
        "http://127.0.0.1:9090/media/generation-jobs",
        json={"input_image": "A" * (2 * 1024 * 1024)},
        timeout=1,
        allow_redirects=False,
    )
    assert got is sentinel

    with pytest.raises(client.ZettlabMediaError, match="request exceeds maximum size"):
        client._SESSION.post(
            "http://127.0.0.1:9090/media/generation-jobs",
            json={"input_image": "A" * client.MAX_MEDIA_REQUEST_BYTES},
            timeout=1,
            allow_redirects=False,
        )
