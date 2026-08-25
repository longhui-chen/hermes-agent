from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from plugins.video_edit import client, normalizer, paths


_ASF_HEADER = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")
_ASF_VIDEO_STREAM = bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b")
_MATROSKA_HEADER = b"\x1a\x45\xdf\xa3\x42\x82\x88matroska"


def _install_trusted_normalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    monkeypatch.setattr(normalizer, "_PRESETS_ANCHOR", None)
    presets = tmp_path / "presets"
    script = (
        presets
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / "normalize.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# trusted test adapter\n", encoding="utf-8")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets))
    return script


def _delayed_video_sample(suffix: str) -> bytes:
    padding = b"\x00" * (paths.VIDEO_HEADER_BYTES * 2)
    if suffix == ".mkv":
        return _MATROSKA_HEADER + b"\xec\x60\x00" + padding + b"\x83\x81\x01"
    if suffix == ".avi":
        body = b"AVI " + b"JUNK" + len(padding).to_bytes(4, "little") + padding
        body += b"LIST\x10\x00\x00\x00strhvids"
        return b"RIFF" + len(body).to_bytes(4, "little") + body
    if suffix == ".asf":
        return _ASF_HEADER + padding + _ASF_VIDEO_STREAM
    if suffix == ".ts":
        packet_size = 188
        sample = bytearray(packet_size * 30)
        for index in range(30):
            sample[packet_size * index] = 0x47
        marker = packet_size * 25 + 8
        sample[marker : marker + 4] = b"\x00\x00\x01\xe0"
        return bytes(sample)
    raise AssertionError(f"unsupported fixture suffix: {suffix}")


@pytest.mark.parametrize(
    ("suffix", "expected_mime"),
    [
        (".mkv", "video/x-matroska"),
        (".avi", "video/x-msvideo"),
        (".asf", "video/x-ms-asf"),
        (".ts", "video/mp2t"),
    ],
)
def test_descriptor_accepts_video_evidence_beyond_the_first_header(
    tmp_path: Path,
    suffix: str,
    expected_mime: str,
) -> None:
    source = tmp_path / f"delayed{suffix}"
    source.write_bytes(_delayed_video_sample(suffix))
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.validate_video_descriptor(source, descriptor) == expected_mime
    finally:
        os.close(descriptor)


def test_descriptor_probe_is_bounded_when_large_container_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-void.mkv"
    source.write_bytes(
        _MATROSKA_HEADER
        + b"\xec\x30\x00\x00"
        + b"\x00" * (paths.VIDEO_PROBE_BYTES + paths.VIDEO_HEADER_BYTES)
        + b"\x83\x81\x01"
    )
    descriptor = os.open(source, os.O_RDONLY)
    read_sizes: list[int] = []
    real_read = os.read
    monkeypatch.setattr(
        paths.os,
        "read",
        lambda fd, size: read_sizes.append(size) or real_read(fd, size),
    )
    try:
        os.lseek(descriptor, 7, os.SEEK_SET)
        assert (
            paths.validate_video_descriptor(source, descriptor)
            == "video/x-matroska"
        )
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 7
    finally:
        os.close(descriptor)

    assert sum(read_sizes) == paths.VIDEO_PROBE_BYTES
    assert max(read_sizes) <= 64 * 1024


def test_complete_small_audio_container_is_delegated_to_packaged_probe(
    tmp_path: Path,
) -> None:
    source = tmp_path / "audio-only.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"\x83\x81\x02")
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/x-matroska",
            False,
        )
    finally:
        os.close(descriptor)


def test_descriptor_does_not_treat_a_short_probe_as_valid_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "short-read.mkv"
    source.write_bytes(
        _MATROSKA_HEADER + b"\x00" * (paths.VIDEO_PROBE_BYTES * 2)
    )
    descriptor = os.open(source, os.O_RDONLY)
    real_read = os.read
    calls = 0

    def short_read(fd: int, size: int) -> bytes:
        nonlocal calls
        calls += 1
        if calls > 1:
            return b""
        return real_read(fd, size)

    monkeypatch.setattr(paths.os, "read", short_read)
    try:
        with pytest.raises(paths.VideoPathError, match="supported video"):
            paths.validate_video_descriptor(source, descriptor)
    finally:
        os.close(descriptor)


def test_packaged_probe_uses_fixed_argv_and_returns_inode_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "ambiguous.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"payload")
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "must-not-pass")
    captured: dict = {}

    def run(command, *, env, timeout, pass_fds):
        captured.update(
            command=command,
            env=env,
            timeout=timeout,
            pass_fds=pass_fds,
            script_inode=os.fstat(pass_fds[0]).st_ino,
        )
        payload = {
            "ok": True,
            "items": [{"category": "direct_only", "direct_ok": True}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), ""), False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", run)
    info = source.stat()

    identities = normalizer.inspect_files([source], "workflow-1")

    assert identities == [
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    ]
    assert captured["command"][0] == sys.executable
    assert captured["command"][1] != str(script)
    assert captured["command"][1].endswith(f"/{captured['pass_fds'][0]}")
    assert captured["script_inode"] == script.stat().st_ino
    assert captured["command"][2] == "--inspect-input"
    assert len(captured["command"]) == 4
    assert captured["timeout"] == normalizer.MEDIA_INSPECTION_TIMEOUT_SECONDS
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in captured["env"]
    assert not list(tmp_path.glob(".hermes-video-input-*"))


def test_packaged_probe_maps_missing_video_stream_to_path_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "audio-only.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"\x83\x81\x02")
    _install_trusted_normalizer(tmp_path, monkeypatch)
    payload = {
        "ok": True,
        "items": [
            {
                "category": "unsupported",
                "direct_ok": False,
                "reason": "PROBE_FAILED",
            }
        ],
    }
    monkeypatch.setattr(
        normalizer,
        "_run_bounded_subprocess",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(command, 0, json.dumps(payload), ""),
            False,
        ),
    )

    with pytest.raises(paths.VideoPathError, match="supported video"):
        normalizer.inspect_files([source], "workflow-1")

    assert not list(tmp_path.glob(".hermes-video-input-*"))


def test_inspection_capability_error_is_distinct_from_media_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "old-helper.mov"
    source.write_bytes(
        (16).to_bytes(4, "big")
        + b"ftyp"
        + b"qt  \x00\x00\x00\x00qt  "
    )
    _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        normalizer,
        "_run_bounded_subprocess",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(
                command,
                2,
                "",
                "usage: normalize.py [-h]\nnormalize.py: error: unrecognized arguments: --inspect-input",
            ),
            False,
        ),
    )

    with pytest.raises(
        normalizer.NormalizerUnavailableError,
        match="capability is unavailable",
    ):
        normalizer.inspect_files([source], "workflow-1")


def test_old_helper_capability_error_reaches_raw_upload_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "old-helper-upload.mov"
    source.write_bytes(
        (16).to_bytes(4, "big")
        + b"ftyp"
        + b"qt  \x00\x00\x00\x00qt  "
    )
    _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        normalizer,
        "_run_bounded_subprocess",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(
                command,
                2,
                "",
                "usage: normalize.py [-h]\nnormalize.py: error: unrecognized arguments: --inspect-input",
            ),
            False,
        ),
    )

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-old"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            return None

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="old-helper-upload",
    )

    assert client.extract_upload_keys(result) == ["asset-old"]


def test_unavailable_helper_is_reported_before_source_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "read-only-source.mov"
    source.write_bytes(b"not inspected")
    pin_calls: list[Path] = []

    def unavailable():
        raise normalizer.NormalizeError("video normalizer is unavailable")

    monkeypatch.setattr(normalizer, "_presets_anchor", unavailable)
    monkeypatch.setattr(
        normalizer,
        "_pin_input",
        lambda path, *_args: pin_calls.append(path),
    )

    with pytest.raises(
        normalizer.NormalizerUnavailableError,
        match="normalizer is unavailable",
    ):
        normalizer.inspect_files([source], "workflow-1")

    assert pin_calls == []


def test_read_only_source_snapshot_uses_local_raw_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "read-only-probe.mov"
    source.write_bytes(
        (16).to_bytes(4, "big")
        + b"ftyp"
        + b"isom\x00\x00\x02\x00isommp42"
    )
    _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        normalizer,
        "_pin_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizeError("video normalization input cannot be pinned")
        ),
    )

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-read-only"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            return None

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="read-only-probe",
    )

    assert client.extract_upload_keys(result) == ["asset-read-only"]


@pytest.mark.parametrize(
    ("filename", "prefix"),
    [
        ("large-audio.mkv", _MATROSKA_HEADER + b"\x83\x81\x02"),
        ("large-forged.asf", _ASF_HEADER),
    ],
)
def test_large_inconclusive_container_requires_probe_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    prefix: bytes,
) -> None:
    source = tmp_path / filename
    source.write_bytes(prefix + b"\x00" * (paths.VIDEO_PROBE_BYTES + 1))
    inspected: list[tuple[list[Path], str]] = []
    connections: list[bool] = []

    def reject(sources, workflow_id):
        inspected.append((list(sources), workflow_id))
        raise paths.VideoPathError("input is not a supported video file")

    monkeypatch.setattr(normalizer, "inspect_files", reject)
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(paths.VideoPathError, match="supported video"):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    assert inspected == [([source], "source-v1")]
    assert connections == []


def test_container_marker_cannot_bypass_packaged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "forged-marker.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"metadata\x83\x81\x01")
    connections: list[bool] = []
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            paths.VideoPathError("input is not a supported video file")
        ),
    )
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(paths.VideoPathError, match="supported video"):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    assert connections == []


def test_missing_packaged_probe_allows_locally_proven_video_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "probe-unavailable.mov"
    source.write_bytes(
        (16).to_bytes(4, "big")
        + b"ftyp"
        + b"isom\x00\x00\x02\x00isommp42"
    )

    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizerUnavailableError(
                "video media inspection capability is unavailable"
            )
        ),
    )

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-local"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            return None

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="probe-unavailable",
    )

    assert client.extract_upload_keys(result) == ["asset-local"]


def test_media_probe_failure_does_not_use_raw_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "probe-failed.mov"
    source.write_bytes(
        (16).to_bytes(4, "big")
        + b"ftyp"
        + b"isom\x00\x00\x02\x00isommp42"
    )
    connections: list[bool] = []
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizeError("video media inspection failed")
        ),
    )
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(normalizer.NormalizeError, match="inspection failed"):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="probe-failed",
        )

    assert connections == []


def test_missing_packaged_probe_rejects_inconclusive_local_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "probe-unavailable-large.mkv"
    source.write_bytes(
        _MATROSKA_HEADER
        + b"\xec\x30\x00\x00"
        + b"\x00" * (paths.VIDEO_PROBE_BYTES + paths.VIDEO_HEADER_BYTES)
    )
    connections: list[bool] = []
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizeError("video normalizer is unavailable")
        ),
    )
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(normalizer.NormalizeError, match="inspection is unavailable"):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="probe-unavailable-large",
        )

    assert connections == []


def test_successful_probe_is_bound_to_the_opened_inode_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large-delayed.mkv"
    source.write_bytes(
        _MATROSKA_HEADER
        + b"\x00" * (paths.VIDEO_PROBE_BYTES + 1)
        + b"\x83\x81\x01"
    )

    def inspect(sources, _workflow_id):
        inspected = list(sources)
        assert inspected == [source]
        info = source.stat()
        return [(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)]

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-1"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.sent = 0

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            return None

        def send(self, chunk):
            self.sent += len(chunk)

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(normalizer, "inspect_files", inspect)
    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="source-v1",
    )

    assert client.extract_upload_keys(result) == ["asset-1"]
