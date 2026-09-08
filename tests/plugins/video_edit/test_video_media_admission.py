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
_EBML_HEADER_ID = b"\x1a\x45\xdf\xa3"
_EBML_SEGMENT_ID = b"\x18\x53\x80\x67"
_EBML_TRACKS_ID = b"\x16\x54\xae\x6b"
_EBML_TRACK_ENTRY_ID = b"\xae"
_EBML_TRACK_TYPE_ID = b"\x83"
_EBML_VOID_ID = b"\xec"


def _ebml_size(value: int) -> bytes:
    for width in range(1, 9):
        if value <= (1 << (7 * width)) - 2:
            return ((1 << (7 * width)) | value).to_bytes(width, "big")
    raise ValueError("EBML fixture is too large")


def _ebml_element(element_id: bytes, payload: bytes) -> bytes:
    return element_id + _ebml_size(len(payload)) + payload


def _ebml_header(doctype: bytes) -> bytes:
    payload = b"".join(
        (
            _ebml_element(b"\x42\x86", b"\x01"),
            _ebml_element(b"\x42\xf7", b"\x01"),
            _ebml_element(b"\x42\xf2", b"\x04"),
            _ebml_element(b"\x42\xf3", b"\x08"),
            _ebml_element(b"\x42\x82", doctype),
            _ebml_element(b"\x42\x87", b"\x04"),
            _ebml_element(b"\x42\x85", b"\x02"),
        )
    )
    return _ebml_element(_EBML_HEADER_ID, payload)


def _ebml_track_entry(track_type: int, extra: bytes = b"") -> bytes:
    codec_id = b"V_VP9" if track_type == 1 else b"A_OPUS"
    payload = b"".join(
        (
            _ebml_element(b"\xd7", b"\x01"),
            _ebml_element(b"\x73\xc5", b"\x01"),
            _ebml_element(_EBML_TRACK_TYPE_ID, bytes([track_type])),
            _ebml_element(b"\x86", codec_id),
            extra,
        )
    )
    return _ebml_element(_EBML_TRACK_ENTRY_ID, payload)


def _ebml_sample(
    doctype: bytes = b"matroska",
    *,
    track_type: int = 1,
    before_tracks: bytes = b"",
    after_tracks: bytes = b"",
    unknown_segment: bool = True,
) -> bytes:
    tracks = _ebml_element(
        _EBML_TRACKS_ID,
        _ebml_track_entry(track_type),
    )
    segment_payload = before_tracks + tracks + after_tracks
    segment_size = (
        b"\x01" + b"\xff" * 7
        if unknown_segment
        else _ebml_size(len(segment_payload))
    )
    return (
        _ebml_header(doctype)
        + _EBML_SEGMENT_ID
        + segment_size
        + segment_payload
    )


_MATROSKA_HEADER = (
    _ebml_header(b"matroska")
    + _EBML_SEGMENT_ID
    + b"\x01"
    + b"\xff" * 7
)


def _iso_box(kind: bytes, payload: bytes) -> bytes:
    return (len(payload) + 8).to_bytes(4, "big") + kind + payload


def _iso_largesize_box(kind: bytes, payload: bytes) -> bytes:
    return b"\x00\x00\x00\x01" + kind + (len(payload) + 16).to_bytes(8, "big") + payload


def _iso_zero_size_box(kind: bytes, payload: bytes) -> bytes:
    return b"\x00\x00\x00\x00" + kind + payload


def _iso_ftyp(brand: bytes = b"qt  ") -> bytes:
    payload = brand + b"\x00\x00\x00\x00" + brand
    return _iso_box(b"ftyp", payload)


def _iso_track_sample(handler_type: bytes, brand: bytes = b"qt  ") -> bytes:
    hdlr = _iso_box(
        b"hdlr",
        b"\x00\x00\x00\x00"  # version + flags
        + b"\x00\x00\x00\x00"  # pre_defined
        + handler_type
        + b"\x00" * 12,  # reserved + empty name
    )
    mdia = _iso_box(b"mdia", hdlr)
    trak = _iso_box(b"trak", mdia)
    return _iso_ftyp(brand) + _iso_box(b"moov", trak)


def _iso_video_sample() -> bytes:
    return _iso_track_sample(b"vide")


def _iso_audio_sample() -> bytes:
    return _iso_track_sample(b"soun", brand=b"isom")


def _iso_tail_sample(handler_type: bytes = b"vide", brand: bytes = b"qt  ") -> bytes:
    ftyp = _iso_ftyp(brand)
    moov = _iso_track_sample(handler_type, brand=brand)[len(ftyp) :]
    mdat_payload = b"\x00" * (paths.VIDEO_PROBE_BYTES + 8192)
    return ftyp + _iso_box(b"mdat", mdat_payload) + moov


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


def _normalized_output(path: Path) -> normalizer.NormalizedOutput:
    info = path.stat()
    return normalizer.NormalizedOutput(
        path=path,
        identity=(
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ),
    )


def _install_successful_upload_connection(
    monkeypatch: pytest.MonkeyPatch,
    connections: list[bool],
) -> None:
    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-normalized"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            connections.append(True)

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


def _delayed_video_sample(suffix: str) -> bytes:
    padding = b"\x00" * (paths.VIDEO_HEADER_BYTES * 2)
    if suffix == ".mkv":
        return _ebml_sample(
            before_tracks=_ebml_element(_EBML_VOID_ID, padding)
        )
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
        _ebml_sample(
            before_tracks=_ebml_element(
                _EBML_VOID_ID,
                b"\x00" * (
                    paths.VIDEO_PROBE_BYTES + paths.VIDEO_HEADER_BYTES
                ),
            )
        )
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
    source.write_bytes(_ebml_sample(track_type=2))
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/x-matroska",
            False,
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "sample",
    [
        _MATROSKA_HEADER
        + _ebml_element(_EBML_VOID_ID, b"metadata\x83\x81\x01"),
        _MATROSKA_HEADER
        + _ebml_element(
            _EBML_TRACKS_ID,
            _ebml_element(_EBML_TRACK_TYPE_ID, b"\x01"),
        ),
        _MATROSKA_HEADER
        + _ebml_element(
            _EBML_TRACKS_ID,
            _ebml_track_entry(
                1,
                _ebml_element(_EBML_TRACK_TYPE_ID, b"\x01"),
            ),
        ),
        _MATROSKA_HEADER
        + _EBML_TRACKS_ID
        + b"\xff"
        + _ebml_track_entry(1),
        _MATROSKA_HEADER + _EBML_TRACKS_ID + b"\x40",
        _ebml_header(b"matroska")
        + _EBML_SEGMENT_ID
        + _ebml_size(1024)
        + _ebml_element(_EBML_TRACKS_ID, _ebml_track_entry(1)),
    ],
    ids=[
        "marker-in-void",
        "track-type-outside-entry",
        "duplicate-track-type",
        "unknown-tracks-size",
        "truncated-size-vint",
        "segment-size-past-eof",
    ],
)
def test_ebml_video_track_proof_fails_closed_on_invalid_hierarchy(
    sample: bytes,
) -> None:
    with pytest.raises(paths.VideoPathError, match="supported video"):
        paths.validate_video_sample(Path("invalid.mkv"), sample)


def test_ebml_doctype_must_be_a_direct_header_child() -> None:
    forged_header = _ebml_element(
        _EBML_HEADER_ID,
        _ebml_element(
            _EBML_VOID_ID,
            _ebml_element(b"\x42\x82", b"matroska"),
        ),
    )
    sample = (
        forged_header
        + _EBML_SEGMENT_ID
        + b"\x01"
        + b"\xff" * 7
        + _ebml_element(_EBML_TRACKS_ID, _ebml_track_entry(1))
    )

    with pytest.raises(paths.VideoPathError, match="supported video"):
        paths.validate_video_sample(Path("forged.mkv"), sample)


@pytest.mark.parametrize("suffix", [".mov", ".mp4"])
def test_iso_bmff_brand_without_video_track_is_inconclusive(
    tmp_path: Path,
    suffix: str,
) -> None:
    source = tmp_path / f"ftyp-only{suffix}"
    source.write_bytes(_iso_ftyp())
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime" if suffix == ".mov" else "video/mp4",
            False,
        )
    finally:
        os.close(descriptor)


def test_iso_bmff_brand_without_video_track_is_rejected_by_sample_validator() -> None:
    with pytest.raises(paths.VideoPathError, match="supported video"):
        paths.validate_video_sample(Path("fake.mov"), _iso_ftyp())


def test_iso_bmff_proven_requires_video_handler_type(
    tmp_path: Path,
) -> None:
    source = tmp_path / "video.mov"
    source.write_bytes(_iso_video_sample())
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            True,
        )
    finally:
        os.close(descriptor)


def test_iso_bmff_video_track_cannot_cross_container_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wrong-container.mkv"
    source.write_bytes(_iso_video_sample())
    descriptor = os.open(source, os.O_RDONLY)
    try:
        with pytest.raises(paths.VideoPathError, match="supported video"):
            paths.validate_video_descriptor(source, descriptor)
    finally:
        os.close(descriptor)


def test_iso_bmff_vide_bytes_outside_handler_do_not_prove_a_track(
    tmp_path: Path,
) -> None:
    source = tmp_path / "forged-vide.mov"
    source.write_bytes(_iso_ftyp() + _iso_box(b"free", b"vide"))
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
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
        pin = Path(command[3])
        captured.update(
            command=command,
            env=env,
            timeout=timeout,
            pass_fds=pass_fds,
            script_inode=os.fstat(pass_fds[0]).st_ino,
            pin_parent=pin.parent,
            pin_inode=os.stat(pin, follow_symlinks=False).st_ino,
            pin_nlink=os.stat(source).st_nlink,
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
        (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            source.stat().st_ctime_ns,
        )
    ]
    assert captured["command"][0] == sys.executable
    assert captured["command"][1] != str(script)
    assert captured["command"][1].endswith(f"/{captured['pass_fds'][0]}")
    # The media is handed over as a private hard-link pin inside the
    # local-server task-cache subtree, never through /proc/<pid>/fd, so a
    # non-dumpable gateway without CAP_SYS_PTRACE can still be probed.
    assert captured["pin_parent"] == tmp_path / ".cache" / "tasks" / "hermes-video-inspect"
    assert Path(captured["command"][3]).name.startswith(".hermes-video-inspect-")
    assert captured["pin_inode"] == source.stat().st_ino
    assert captured["pin_nlink"] == 2
    assert captured["script_inode"] == script.stat().st_ino
    assert captured["command"][2] == "--inspect-input"
    assert len(captured["command"]) == 4
    assert len(captured["pass_fds"]) == 1
    assert captured["timeout"] == normalizer.MEDIA_INSPECTION_TIMEOUT_SECONDS
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in captured["env"]
    assert not list(tmp_path.glob(".hermes-video-input-*"))
    assert not list(tmp_path.rglob(".hermes-video-inspect-*"))
    assert not (tmp_path / ".cache").exists()
    assert source.stat().st_nlink == 1


@pytest.mark.skipif(os.name != "posix", reason="fd path requires procfs")
def test_packaged_probe_fd_path_survives_helper_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "child-readable.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"child-readable-payload")
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    script.write_text(
        "import argparse, json, subprocess, sys\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--inspect-input', action='append', default=[])\n"
        "args = parser.parse_args()\n"
        "child = subprocess.run([sys.executable, '-c', "
        "\"from pathlib import Path; import sys; "
        "print(len(Path(sys.argv[1]).read_bytes()))\", args.inspect_input[0]], "
        "capture_output=True, text=True)\n"
        "if child.returncode != 0:\n"
        "    print(child.stderr, file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
        "print(json.dumps({'ok': True, 'items': "
        "[{'category': 'direct_only'}]}))\n",
        encoding="utf-8",
    )

    identities = normalizer.inspect_files([source], "workflow-child-fd")

    assert identities[0][:4] == (
        source.stat().st_dev,
        source.stat().st_ino,
        source.stat().st_size,
        source.stat().st_mtime_ns,
    )


def test_packaged_probe_identity_handles_duplicate_hard_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "same-inode-a.mkv"
    alias = tmp_path / "same-inode-b.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"payload")
    os.link(source, alias)
    _install_trusted_normalizer(tmp_path, monkeypatch)

    def run(command, **_kwargs):
        payload = {
            "ok": True,
            "items": [
                {"category": "direct_only", "direct_ok": True},
                {"category": "direct_only", "direct_ok": True},
            ],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), ""), False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", run)
    identities = normalizer.inspect_files([source, alias], "workflow-1")

    assert len(identities) == 2
    assert identities[0][:4] == identities[1][:4]
    assert identities[0][4] == source.stat().st_ctime_ns
    assert not list(tmp_path.glob(".hermes-video-input-*"))


def test_packaged_probe_rejects_source_change_during_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "changed-during-probe.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"payload")
    _install_trusted_normalizer(tmp_path, monkeypatch)

    def run(command, **_kwargs):
        source.write_bytes(source.read_bytes() + b"changed")
        payload = {
            "ok": True,
            "items": [{"category": "direct_only", "direct_ok": True}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), ""), False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", run)
    with pytest.raises(normalizer.NormalizeError, match="input changed"):
        normalizer.inspect_files([source], "workflow-1")

    assert not list(tmp_path.glob(".hermes-video-input-*"))


def test_packaged_probe_rejects_same_size_mutation_with_restored_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "same-size-mutation.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"original-payload")
    _install_trusted_normalizer(tmp_path, monkeypatch)
    before = source.stat()

    def run(command, **_kwargs):
        with source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            current = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(b"X" if current != b"X" else b"Y")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        payload = {
            "ok": True,
            "items": [{"category": "direct_only", "direct_ok": True}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), ""), False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", run)
    with pytest.raises(normalizer.NormalizeError, match="input changed"):
        normalizer.inspect_files([source], "workflow-1")

    assert source.stat().st_size == before.st_size
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert source.stat().st_ctime_ns != before.st_ctime_ns


def test_packaged_probe_maps_missing_video_stream_to_path_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "audio-only.mkv"
    source.write_bytes(_ebml_sample(track_type=2))
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


def test_admitted_normalized_identity_skips_packaged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "normalized.mp4"
    source.write_bytes(_iso_video_sample())
    output = _normalized_output(source)
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: pytest.fail(
            "normalized output must not run the raw-media probe"
        ),
    )
    connections: list[bool] = []
    _install_successful_upload_connection(monkeypatch, connections)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="normalized-output",
        normalized_outputs=[output],
    )

    assert client.extract_upload_keys(result) == ["asset-normalized"]
    assert connections == [True]


@pytest.mark.parametrize(
    "change",
    ["same_inode_mutation", "inode_replacement", "symlink_replacement"],
)
def test_stale_normalized_identity_fails_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    source = tmp_path / "normalized.mp4"
    source.write_bytes(_iso_video_sample())
    output = _normalized_output(source)
    before = source.stat()
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(_iso_video_sample())
    if change == "same_inode_mutation":
        with source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b"X")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "inode_replacement":
        os.replace(replacement, source)
    else:
        source.unlink()
        source.symlink_to(replacement)
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: pytest.fail(
            "stale normalized output must not run the raw-media probe"
        ),
    )
    connections: list[bool] = []
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(normalizer.NormalizeError):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="stale-normalized-output",
            normalized_outputs=[output],
        )

    assert connections == []


def test_raw_direct_upload_still_runs_packaged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw-direct.mov"
    source.write_bytes(_iso_video_sample())
    inspected: list[tuple[list[Path], str]] = []

    def inspect(sources, workflow_id):
        inspected.append((list(sources), workflow_id))
        return [_normalized_output(source).identity]

    monkeypatch.setattr(normalizer, "inspect_files", inspect)
    connections: list[bool] = []
    _install_successful_upload_connection(monkeypatch, connections)

    result = client.upload(
        [source],
        agent_id="agent-a",
        replay_scope="raw-direct",
    )

    assert client.extract_upload_keys(result) == ["asset-normalized"]
    assert inspected == [([source], "raw-direct")]
    assert connections == [True]


@pytest.mark.parametrize(
    ("sample", "error", "message"),
    [
        pytest.param(
            _iso_ftyp(),
            normalizer.NormalizeError,
            "inspection is unavailable",
            id="ftyp-only",
        ),
        pytest.param(
            _iso_audio_sample(),
            normalizer.NormalizeError,
            "inspection is unavailable",
            id="isom-audio-track",
        ),
    ],
)
def test_old_helper_capability_error_rejects_unproven_iso_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample: bytes,
    error: type[Exception],
    message: str,
) -> None:
    source = tmp_path / "old-helper-upload.mov"
    source.write_bytes(sample)
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

    connections: list[bool] = []

    class Connection:
        def __init__(self, *_args, **_kwargs):
            connections.append(True)

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    with pytest.raises(error, match=message):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="old-helper-upload",
        )

    assert connections == []


def test_old_helper_capability_error_allows_proven_video_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "old-helper-video.mov"
    source.write_bytes(_iso_video_sample())
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
        replay_scope="old-helper-video",
    )

    assert client.extract_upload_keys(result) == ["asset-old"]


def test_local_fallback_rejects_same_size_mutation_with_restored_mtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "local-fallback-mutation.mov"
    source.write_bytes(_iso_video_sample())
    before = source.stat()
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizerUnavailableError("capability is unavailable")
        ),
    )
    real_inspect = client.inspect_video_descriptor

    def inspect(path: Path, descriptor: int):
        result = real_inspect(path, descriptor)
        with source.open("r+b") as stream:
            stream.seek(-1, os.SEEK_END)
            current = stream.read(1)
            stream.seek(-1, os.SEEK_CUR)
            stream.write(b"X" if current != b"X" else b"Y")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(client, "inspect_video_descriptor", inspect)
    connections: list[bool] = []

    class Connection:
        def __init__(self, *_args, **_kwargs):
            connections.append(True)

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    with pytest.raises(client.VideoClientError, match="source changed"):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="local-fallback-mutation",
        )

    assert connections == []


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
    source.write_bytes(_iso_video_sample())
    _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(
        normalizer,
        "_run_bounded_subprocess",
        lambda command, **_kwargs: (
            subprocess.CompletedProcess(
                command,
                2,
                "",
                "usage: normalize.py [-h]\n"
                "normalize.py: error: unrecognized arguments: --inspect-input",
            ),
            False,
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
        ("large-audio.mkv", _ebml_sample(track_type=2)),
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


@pytest.mark.parametrize(
    "capability_error",
    [
        normalizer.NormalizerUnavailableError(
            "video media inspection capability is unavailable"
        ),
        normalizer.NormalizeError("video normalizer is unavailable"),
    ],
    ids=["missing-helper", "old-helper"],
)
def test_container_marker_cannot_bypass_missing_packaged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability_error: normalizer.NormalizeError,
) -> None:
    source = tmp_path / "forged-marker.mkv"
    source.write_bytes(
        _ebml_sample(
            track_type=2,
            before_tracks=_ebml_element(
                _EBML_VOID_ID,
                b"metadata\x83\x81\x01",
            ),
        )
    )
    connections: list[bool] = []
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            capability_error
        ),
    )
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(
        normalizer.NormalizeError,
        match="inspection is unavailable",
    ):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    assert connections == []


@pytest.mark.parametrize(
    ("filename", "doctype", "unknown_segment"),
    [
        ("video.mkv", b"matroska", False),
        ("video.webm", b"webm", True),
    ],
)
def test_structured_ebml_video_is_locally_proven_without_packaged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    doctype: bytes,
    unknown_segment: bool,
) -> None:
    source = tmp_path / filename
    source.write_bytes(
        _ebml_sample(doctype, unknown_segment=unknown_segment)
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
            return b'{"data":{"uploads":[{"object_key":"asset-ebml"}]}}'

    connections: list[bool] = []

    class Connection:
        def __init__(self, *_args, **_kwargs):
            connections.append(True)

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
        replay_scope="structured-ebml",
    )

    assert client.extract_upload_keys(result) == ["asset-ebml"]
    assert connections == [True]


def test_missing_packaged_probe_allows_locally_proven_video_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "probe-unavailable.mov"
    source.write_bytes(_iso_video_sample())

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


def test_missing_packaged_probe_allows_tail_moov_video_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tail-moov.mov"
    source.write_bytes(_iso_tail_sample())
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
            return b'{"data":{"uploads":[{"object_key":"asset-tail"}]}}'

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
        replay_scope="tail-moov",
    )

    assert client.extract_upload_keys(result) == ["asset-tail"]


@pytest.mark.parametrize(
    ("mdat_builder", "moov_builder"),
    [
        pytest.param(_iso_box, _iso_box, id="32-bit"),
        pytest.param(_iso_largesize_box, _iso_box, id="mdat-largesize"),
        pytest.param(_iso_box, _iso_largesize_box, id="moov-largesize"),
        pytest.param(_iso_box, _iso_zero_size_box, id="moov-zero-size"),
    ],
)
def test_tail_moov_descriptor_supports_top_level_size_encodings(
    tmp_path: Path,
    mdat_builder,
    moov_builder,
) -> None:
    source = tmp_path / "tail-size-encoding.mov"
    ftyp = _iso_ftyp()
    moov = _iso_video_sample()[len(ftyp) :]
    source.write_bytes(
        ftyp
        + mdat_builder(
            b"mdat",
            b"\x00" * (paths.VIDEO_PROBE_BYTES + 8192),
        )
        + moov_builder(b"moov", moov[8:])
    )
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            True,
        )
    finally:
        os.close(descriptor)


def test_tail_moov_scanner_seeks_over_mdat_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tail-read-offsets.mov"
    ftyp = _iso_ftyp()
    moov = _iso_video_sample()[len(ftyp) :]
    mdat = _iso_box(b"mdat", b"\x00" * (paths.VIDEO_PROBE_BYTES + 8192))
    source.write_bytes(ftyp + mdat + moov)
    mdat_start = len(ftyp)
    payload_start = mdat_start + 8
    moov_start = mdat_start + len(mdat)
    real_read = paths.os.read
    reads: list[tuple[int, int]] = []

    def read(fd: int, size: int) -> bytes:
        offset = paths.os.lseek(fd, 0, os.SEEK_CUR)
        reads.append((offset, size))
        return real_read(fd, size)

    monkeypatch.setattr(paths.os, "read", read)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            True,
        )
    finally:
        os.close(descriptor)

    assert (mdat_start, 8) in reads
    assert not any(
        payload_start <= offset < moov_start
        for offset, _size in reads[1:]
    )


@pytest.mark.parametrize("corruption", ["trailing-top-level", "nested-child"])
def test_tail_moov_rejects_structurally_incomplete_boxes(
    tmp_path: Path,
    corruption: str,
) -> None:
    ftyp = _iso_ftyp()
    mdat = _iso_box(b"mdat", b"\x00" * (paths.VIDEO_PROBE_BYTES + 8192))
    valid_trak = _iso_video_sample()[len(ftyp) + 8 :]
    if corruption == "trailing-top-level":
        moov = _iso_box(b"moov", valid_trak)
        malformed = b"\x00\x00\x00\x20junk"
    else:
        moov = _iso_box(b"moov", valid_trak + b"\x00\x00\x00\x20junk")
        malformed = b""
    source = tmp_path / f"tail-malformed-{corruption}.mov"
    source.write_bytes(ftyp + mdat + moov + malformed)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            False,
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "malformed_tail",
    [b"abc", b"\x00\x00\x00\x04junk", b"\x00\x00\x00\x20junk"],
)
def test_fast_start_moov_rejects_structurally_incomplete_boxes(
    tmp_path: Path,
    malformed_tail: bytes,
) -> None:
    source = tmp_path / "fast-malformed-tail.mov"
    source.write_bytes(_iso_video_sample() + malformed_tail)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            False,
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "malformed_trak",
    [
        pytest.param(_iso_box(b"trak", b""), id="empty-trak"),
        pytest.param(_iso_box(b"trak", _iso_box(b"tkhd", b"\x00" * 4)), id="missing-mdia"),
        pytest.param(
            _iso_box(b"trak", _iso_box(b"mdia", _iso_box(b"mdhd", b"\x00" * 4))),
            id="missing-hdlr",
        ),
        pytest.param(_iso_box(b"trak", b"\x00\x00\x00\x20junk"), id="truncated-child"),
    ],
)
def test_moov_rejects_malformed_sibling_trak_after_video_track(
    tmp_path: Path,
    malformed_trak: bytes,
) -> None:
    ftyp = _iso_ftyp()
    valid_trak = _iso_video_sample()[len(ftyp) + 8 :]
    source = tmp_path / "malformed-sibling-trak.mov"
    source.write_bytes(ftyp + _iso_box(b"moov", valid_trak + malformed_trak))
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            False,
        )
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param("ftyp", id="duplicate-ftyp"),
        pytest.param("mdia", id="duplicate-mdia"),
        pytest.param("hdlr", id="duplicate-hdlr"),
    ],
)
def test_iso_bmff_rejects_duplicate_required_boxes(
    tmp_path: Path,
    duplicate: str,
) -> None:
    ftyp = _iso_ftyp()
    video_handler = _iso_box(b"hdlr", b"\x00" * 8 + b"vide")
    audio_handler = _iso_box(b"hdlr", b"\x00" * 8 + b"soun")
    if duplicate == "ftyp":
        payload = ftyp + ftyp + _iso_video_sample()[len(ftyp) :]
    elif duplicate == "mdia":
        trak = _iso_box(
            b"trak",
            _iso_box(b"mdia", video_handler)
            + _iso_box(b"mdia", audio_handler),
        )
        payload = ftyp + _iso_box(b"moov", trak)
    else:
        trak = _iso_box(
            b"trak",
            _iso_box(b"mdia", video_handler + audio_handler),
        )
        payload = ftyp + _iso_box(b"moov", trak)

    source = tmp_path / f"duplicate-{duplicate}.mov"
    source.write_bytes(payload)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            False,
        )
    finally:
        os.close(descriptor)


def test_tail_moov_rejects_unscanned_top_level_box_overflow(
    tmp_path: Path,
) -> None:
    ftyp = _iso_ftyp()
    moov = _iso_video_sample()[len(ftyp) :]
    mdat = _iso_box(b"mdat", b"\x00" * (paths.VIDEO_PROBE_BYTES + 8192))
    free_boxes = b"".join(
        _iso_box(b"free", b"")
        for _ in range(paths._ISO_BMFF_MAX_TOP_LEVEL_BOXES)
    )
    source = tmp_path / "tail-box-limit.mov"
    source.write_bytes(ftyp + mdat + moov + free_boxes)
    descriptor = os.open(source, os.O_RDONLY)
    try:
        assert paths.inspect_video_descriptor(source, descriptor) == (
            "video/quicktime",
            False,
        )
    finally:
        os.close(descriptor)


def test_missing_packaged_probe_rejects_tail_moov_audio_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "tail-audio.mov"
    source.write_bytes(_iso_tail_sample(b"soun", brand=b"isom"))
    monkeypatch.setattr(
        normalizer,
        "inspect_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            normalizer.NormalizerUnavailableError(
                "video media inspection capability is unavailable"
            )
        ),
    )
    connections: list[bool] = []
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(
        normalizer.NormalizeError,
        match="inspection is unavailable",
    ):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="tail-audio",
        )

    assert connections == []


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
        _ebml_header(b"matroska")
        + _EBML_SEGMENT_ID
        + b"\x01"
        + b"\xff" * 7
        + _ebml_element(
            _EBML_VOID_ID,
            b"\x00" * (
                paths.VIDEO_PROBE_BYTES + paths.VIDEO_HEADER_BYTES
            ),
        )
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
        return [
            (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        ]

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


def test_upload_rejects_incomplete_probe_identity_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "incomplete-identity.mkv"
    source.write_bytes(_MATROSKA_HEADER + b"video")

    def inspect(sources, _workflow_id):
        info = list(sources)[0].stat()
        return [(info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)]

    monkeypatch.setattr(normalizer, "inspect_files", inspect)
    connections: list[bool] = []

    class Connection:
        def __init__(self, *_args, **_kwargs):
            connections.append(True)

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    with pytest.raises(client.VideoClientError, match="inspection is invalid"):
        client.upload(
            [source],
            agent_id="agent-a",
            replay_scope="incomplete-identity",
        )

    assert connections == []
