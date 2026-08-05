from __future__ import annotations

import base64
import os
import time

import pytest


PNG = b"\x89PNG\r\n\x1a\ninline-image"
JPEG = b"\xff\xd8\xffinline-image"
WEBP = b"RIFF\x0c\x00\x00\x00WEBPinline-image"


def _capability(limit: int = 1024, *, supports_url: bool = False):
    capability = {
        "modalities": ["text", "image"],
        "_type_limits": {"max_inline_image_bytes": limit},
    }
    if supports_url:
        capability["supports_input_image_url"] = True
    return capability


@pytest.mark.parametrize(
    ("raw", "mime"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")],
)
def test_inline_image_input_encodes_supported_local_file(tmp_path, monkeypatch, raw, mime):
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "source.bin"
    image_path.write_bytes(raw)

    got = client.inline_image_input(
        str(image_path),
        None,
        _capability(),
    )

    assert got == f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def test_inline_image_input_accepts_file_url(tmp_path):
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "source.png"
    image_path.write_bytes(PNG)

    got = client.inline_image_input(image_path.as_uri(), None, _capability())

    assert got == f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"


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


def test_supported_modalities_accept_url_only_image_capability():
    from plugins import zettlab_media_client as client

    assert client.supported_modalities(
        {"limits": {}},
        {"modalities": ["image"], "supports_input_image_url": True},
    ) == ["image"]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("data:image/jpeg;base64," + base64.b64encode(PNG).decode("ascii"), "does not match"),
        ("data:image/png;base64,not-base64!", "valid base64"),
    ],
)
def test_inline_image_input_rejects_unsafe_or_invalid_values(value, message):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match=message):
        client.inline_image_input(value, None, _capability())


def test_inline_image_input_passes_https_url_without_network(monkeypatch):
    from plugins import zettlab_media_client as client

    source = "https://images.example.com/source.png?token=signed-value"
    monkeypatch.setattr(
        client._SESSION,
        "request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("remote URL must not be fetched")
        ),
    )
    monkeypatch.setattr(
        client._FILE_WORKER,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("remote URL must not reach the file worker")
        ),
    )

    assert client.inline_image_input(
        source,
        None,
        _capability(supports_url=True),
    ) == source


def test_inline_image_input_fails_closed_when_url_capability_is_missing():
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match="not enabled"):
        client.inline_image_input(
            "https://images.example.com/source.png",
            None,
            _capability(),
        )


@pytest.mark.parametrize(
    "value",
    [
        "http://images.example.com/source.png",
        "https://user:pass@images.example.com/source.png",
        "https://images.example.com/source.png#fragment",
        "https://images.example.com/source image.png",
        "https://localhost/source.png",
        "https://images.localhost/source.png",
        "https://10.0.0.5/source.png",
        "https://[::1]/source.png",
        "https://images.example.com:8443/source.png",
    ],
)
def test_inline_image_input_rejects_invalid_remote_url(value):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match="valid HTTPS URL"):
        client.inline_image_input(
            value,
            None,
            _capability(supports_url=True),
        )


def test_inline_image_input_rejects_oversize_remote_url():
    from plugins import zettlab_media_client as client

    value = "https://images.example.com/" + "a" * client.MAX_INPUT_IMAGE_URL_BYTES
    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.inline_image_input(
            value,
            None,
            _capability(supports_url=True),
        )


def test_inline_image_input_rejects_relative_path_before_read(monkeypatch):
    from plugins import zettlab_media_client as client

    monkeypatch.setattr(
        client._FILE_WORKER,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("relative path must not reach file worker")
        ),
    )

    with pytest.raises(client.ZettlabMediaError, match="absolute path"):
        client.inline_image_input("source.png", None, _capability())


@pytest.mark.parametrize("backend", ["ssh", "docker", "singularity", "modal", "daytona"])
def test_inline_image_input_rejects_nonlocal_terminal_backends(
    tmp_path,
    monkeypatch,
    backend,
):
    from plugins import zettlab_media_client as client
    from tools import file_tools

    image_path = tmp_path / "source.png"
    image_path.write_bytes(PNG)
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task_id: backend)
    monkeypatch.setattr(
        client._FILE_WORKER,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-local path must not reach file worker")
        ),
    )

    with pytest.raises(client.ZettlabMediaError, match=f"{backend} terminal backend"):
        client.inline_image_input(
            str(image_path),
            None,
            _capability(),
            task_id="remote-session",
        )


def test_inline_image_input_rejects_registered_ssh_environment(tmp_path, monkeypatch):
    from plugins import zettlab_media_client as client
    from tools import file_tools, terminal_tool

    class SSHEnvironment:
        pass

    task_id = "ssh-media-session"
    image_path = tmp_path / "source.png"
    image_path.write_bytes(PNG)
    monkeypatch.setattr(
        client._FILE_WORKER,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SSH path must not reach host file worker")
        ),
    )
    with terminal_tool._env_lock:
        previous = terminal_tool._active_environments.get(task_id)
        terminal_tool._active_environments[task_id] = SSHEnvironment()
    try:
        assert file_tools._terminal_env_type_for_task(task_id) == "ssh"
        with pytest.raises(client.ZettlabMediaError, match="ssh terminal backend"):
            client.inline_image_input(
                str(image_path),
                None,
                _capability(),
                task_id=task_id,
            )
    finally:
        with terminal_tool._env_lock:
            if previous is None:
                terminal_tool._active_environments.pop(task_id, None)
            else:
                terminal_tool._active_environments[task_id] = previous


def test_host_read_resolver_returns_canonical_target_and_identity(tmp_path, monkeypatch):
    from tools import file_tools

    target_path = tmp_path / "target.png"
    (tmp_path / "unused").mkdir()
    target_path.write_bytes(PNG)
    monkeypatch.setattr(file_tools, "_terminal_env_type_for_task", lambda _task_id: "local")

    resolved, identity = file_tools.resolve_host_read_path_for_task(
        str(tmp_path / "unused" / ".." / "target.png"),
        "local-session",
    )
    target_stat = target_path.stat()

    assert resolved == target_path.resolve(strict=True)
    assert identity == (target_stat.st_dev, target_stat.st_ino)


def test_inline_image_input_passes_read_context_to_worker(
    tmp_path,
    monkeypatch,
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "source.png"
    profile_home = tmp_path / "profile-home"
    image_path.write_bytes(PNG)
    seen = {}

    def read(
        source,
        limit,
        *,
        deadline,
        task_id="default",
        terminal_backend="local",
        managed_hermes_roots=(),
        hermes_home_override=None,
    ):
        seen.update(
            source=source,
            limit=limit,
            deadline=deadline,
            task_id=task_id,
            terminal_backend=terminal_backend,
            managed_hermes_roots=managed_hermes_roots,
            hermes_home_override=hermes_home_override,
        )
        return PNG

    monkeypatch.setattr(client._FILE_WORKER, "read", read)

    token = set_hermes_home_override(profile_home)
    try:
        got = client.inline_image_input(
            str(image_path),
            None,
            _capability(),
            task_id="local-session",
        )
    finally:
        reset_hermes_home_override(token)

    assert got.startswith("data:image/png;base64,")
    assert seen["source"] == str(image_path)
    assert seen["task_id"] == "local-session"
    assert seen["terminal_backend"] == "local"
    assert seen["managed_hermes_roots"]
    assert seen["hermes_home_override"] == str(profile_home)


def test_inline_image_parent_preflight_does_not_resolve_or_stat_path(
    tmp_path,
    monkeypatch,
):
    from plugins import zettlab_media_client as client
    from tools import file_tools

    image_path = tmp_path / "source.png"
    image_path.write_bytes(PNG)
    monkeypatch.setattr(
        file_tools,
        "resolve_host_read_path_for_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("path I/O must run only inside the spawned worker")
        ),
    )
    monkeypatch.setattr(client._FILE_WORKER, "read", lambda *_args, **_kwargs: PNG)

    got = client.inline_image_input(str(image_path), None, _capability())

    assert got.startswith("data:image/png;base64,")


def test_windows_reparse_component_is_rejected_before_canonical_resolution(
    tmp_path,
    monkeypatch,
):
    from tools import file_tools

    reparse_path = tmp_path / "junction" / "image.png"
    real_stat = file_tools.os.stat

    def fake_stat(path, *, follow_symlinks=True):
        if str(path).endswith("junction") and follow_symlinks is False:
            return type("ReparseStat", (), {"st_file_attributes": 0x400})()
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(file_tools.sys, "platform", "win32")
    monkeypatch.setattr(file_tools.os, "stat", fake_stat)

    with pytest.raises(ValueError, match="reparse paths"):
        file_tools._reject_windows_reparse_components(reparse_path)


@pytest.mark.parametrize(
    "value",
    [
        r"\\server\share\image.png",
        r"//server/share/image.png",
        r"\\?\UNC\server\share\image.png",
        r"\\.\C:\image.png",
        r"\??\C:\image.png",
        r"\Device\Mup\server\share\image.png",
        "file:////server/share/image.png",
        "file:///%2Fserver/share/image.png",
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


def test_inline_image_input_confines_managed_sibling_profiles(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins import zettlab_media_client as client
    from tools import file_tools

    managed_root = tmp_path / "hermes-home"
    profile_a = managed_root / "profiles" / "profile-a"
    profile_b = managed_root / "profiles" / "profile-b"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    own_image = profile_a / "own.png"
    sibling_image = profile_b / "sibling.png"
    own_image.write_bytes(PNG)
    sibling_image.write_bytes(PNG)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    monkeypatch.setattr(file_tools, "_MANAGED_CLAW_HERMES_ROOTS", (str(managed_root),))

    token = set_hermes_home_override(profile_a)
    try:
        own = client.inline_image_input(
            str(own_image),
            None,
            _capability(),
            task_id="session-a",
        )
        assert own.startswith("data:image/png;base64,")
        with pytest.raises(client.ZettlabMediaError, match="sibling profile"):
            client.inline_image_input(
                str(sibling_image),
                None,
                _capability(),
                task_id="session-a",
            )
    finally:
        reset_hermes_home_override(token)


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
        client._FILE_WORKER.read(
            str(fifo_path),
            1024,
            deadline=time.monotonic() + 1,
        )


def test_inline_image_input_preserves_file_safety_guard_in_worker(tmp_path):
    from plugins import zettlab_media_client as client

    blocked_path = tmp_path / ".env"
    blocked_path.write_bytes(PNG)

    with pytest.raises(ValueError, match="Access denied"):
        client._FILE_WORKER.read(
            str(blocked_path),
            1024,
            deadline=time.monotonic() + 1,
        )


def test_inline_image_input_does_not_follow_symlink(tmp_path):
    from plugins import zettlab_media_client as client

    target_path = tmp_path / "target.png"
    target_path.write_bytes(PNG)
    symlink_path = tmp_path / "generated.png"
    try:
        symlink_path.symlink_to(target_path)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable")
    with pytest.raises(ValueError, match="symbolic link"):
        client._FILE_WORKER.read(
            str(symlink_path),
            1024,
            deadline=time.monotonic() + 1,
        )


def test_media_file_worker_rejects_identity_changed_after_authorization(tmp_path):
    from plugins import zettlab_media_client as client

    authorized_path = tmp_path / "authorized.png"
    replacement_path = tmp_path / "replacement.png"
    authorized_path.write_bytes(PNG)
    replacement_path.write_bytes(PNG)
    authorized_stat = authorized_path.stat()

    with pytest.raises(client.ZettlabMediaError, match="changed after authorization"):
        client._read_authorized_media_file(
            str(replacement_path),
            1024,
            (authorized_stat.st_dev, authorized_stat.st_ino),
            bytearray(1024),
        )


def test_media_file_worker_terminates_stalled_preflight_and_recovers(monkeypatch):
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

    def start_next_worker(
        source,
        limit,
        task_id,
        terminal_backend,
        managed_hermes_roots,
        hermes_home_override,
        deadline,
    ):
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

    def start_worker(
        source,
        limit,
        task_id,
        terminal_backend,
        managed_hermes_roots,
        hermes_home_override,
        deadline,
    ):
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


def test_media_http_session_accepts_base64_sized_request():
    from plugins import zettlab_media_client as client

    sentinel = object()
    class FakeWorker:
        def request(self, *args, **kwargs):
            return sentinel

        def close(self):
            return None

    session = client._MediaHTTPSession(workers=[FakeWorker()])
    try:
        got = session.post(
            "http://127.0.0.1:9090/media/generation-jobs",
            json={"input_image": "A" * (2 * 1024 * 1024)},
            timeout=1,
            allow_redirects=False,
        )
        assert got is sentinel

        with pytest.raises(client.ZettlabMediaError, match="request exceeds maximum size"):
            session.post(
                "http://127.0.0.1:9090/media/generation-jobs",
                json={"input_image": "A" * client.MAX_MEDIA_REQUEST_BYTES},
                timeout=1,
                allow_redirects=False,
            )
    finally:
        session.close()
