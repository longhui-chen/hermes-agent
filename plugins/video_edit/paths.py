"""Profile-scoped paths and bounded filesystem validation for video editing."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RAW_ROOTS = ("/volume1/subvol/data", "/volume1/data")
_MAX_PATH_BYTES = 4096
MAX_TASK_ID_LENGTH = 256
VIDEO_HEADER_BYTES = 4096
VIDEO_PROBE_BYTES = 1024 * 1024

_VIDEO_FORMATS: dict[str, tuple[str, frozenset[str]]] = {
    ".3g2": ("video/3gpp2", frozenset({"iso-bmff"})),
    ".3gp": ("video/3gpp", frozenset({"iso-bmff"})),
    ".asf": ("video/x-ms-asf", frozenset({"asf"})),
    ".avi": ("video/x-msvideo", frozenset({"avi"})),
    ".flv": ("video/x-flv", frozenset({"flv"})),
    ".m2ts": ("video/mp2t", frozenset({"mpeg-ts"})),
    ".m4v": ("video/mp4", frozenset({"iso-bmff"})),
    ".mkv": ("video/x-matroska", frozenset({"matroska"})),
    ".mov": ("video/quicktime", frozenset({"iso-bmff"})),
    ".mp4": ("video/mp4", frozenset({"iso-bmff"})),
    ".mpeg": ("video/mpeg", frozenset({"mpeg-ps"})),
    ".mpg": ("video/mpeg", frozenset({"mpeg-ps"})),
    ".mts": ("video/mp2t", frozenset({"mpeg-ts"})),
    ".mxf": ("application/mxf", frozenset({"mxf"})),
    ".ogv": ("video/ogg", frozenset({"ogg-video"})),
    ".ts": ("video/mp2t", frozenset({"mpeg-ts"})),
    ".webm": ("video/webm", frozenset({"webm"})),
    ".wmv": ("video/x-ms-wmv", frozenset({"asf"})),
}

_ASF_HEADER = bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")
_ASF_VIDEO_STREAM = bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b")
_MXF_HEADER_PREFIX = bytes.fromhex("060e2b34020501010d0102010102")
_ISO_AUDIO_BRANDS = {b"F4A ", b"F4B ", b"M4A ", b"M4B ", b"M4P "}
_ISO_VIDEO_BRANDS = {
    b"avc1",
    b"avc2",
    b"avc3",
    b"avc4",
    b"dash",
    b"iso2",
    b"iso3",
    b"iso4",
    b"iso5",
    b"iso6",
    b"iso7",
    b"iso8",
    b"iso9",
    b"isom",
    b"M4V ",
    b"M4VH",
    b"M4VP",
    b"mp41",
    b"mp42",
    b"msdh",
    b"msix",
    b"qt  ",
}


class VideoPathError(ValueError):
    pass


def safe_id(value: Any, *, fallback: str = "default") -> str:
    text = str(value or "").strip()
    if not text or not _PROFILE_RE.fullmatch(text):
        return fallback
    return text


def agent_id_from_kwargs(kwargs: dict | None = None) -> str:
    kwargs = kwargs or {}
    for key in ("agent_id", "profile_id", "agent"):
        value = kwargs.get(key)
        if isinstance(value, str) and _PROFILE_RE.fullmatch(value.strip()):
            return value.strip()
    try:
        from agent.secret_scope import get_secret

        for key in ("ZET_AGENT_ID", "ZETTLAB_AGENT_ID", "AGENT_ID"):
            value = str(get_secret(key, "") or "").strip()
            if _PROFILE_RE.fullmatch(value):
                return value
    except Exception:
        pass
    try:
        from hermes_cli.profiles import get_active_profile_name

        return safe_id(get_active_profile_name())
    except Exception:
        return "default"


def task_id_from_kwargs(kwargs: dict | None = None) -> str:
    kwargs = kwargs or {}
    for key in ("task_id", "turn_id", "session_id"):
        value = str(kwargs.get(key) or "").strip()
        if value:
            return value[:MAX_TASK_ID_LENGTH]
    try:
        from gateway.session_context import zettlab_turn_id

        value = zettlab_turn_id()
        if value:
            return value[:MAX_TASK_ID_LENGTH]
    except Exception:
        pass
    return "interactive"


def state_root() -> Path:
    raw_home = Path(get_hermes_home())
    if raw_home.is_symlink():
        raise VideoPathError("Hermes state root is a symlink")
    root = raw_home.resolve() / "video_edit"
    if root.exists() and root.is_symlink():
        raise VideoPathError("video edit state root is a symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink():
        raise VideoPathError("video edit state root is a symlink")
    return root


def state_path(name: str, agent_id: str) -> Path:
    if not re.fullmatch(r"[a-z_]{1,48}\.json", name):
        raise VideoPathError("invalid video edit state file")
    profile_root = state_root() / safe_id(agent_id)
    if profile_root.is_symlink():
        raise VideoPathError("video edit profile state root is a symlink")
    profile_root.mkdir(mode=0o700, parents=False, exist_ok=True)
    return profile_root / name


def _under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def _iso_bmff_is_video(sample: bytes) -> bool:
    offset = 0
    while offset + 8 <= len(sample):
        size = int.from_bytes(sample[offset : offset + 4], "big")
        kind = sample[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if offset + 16 > len(sample):
                return False
            size = int.from_bytes(sample[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = len(sample) - offset
        if size < header_size or offset + size > len(sample):
            return False
        if kind == b"ftyp":
            payload = sample[offset + header_size : offset + size]
            if len(payload) < 8 or (len(payload) - 8) % 4:
                return False
            if payload[:4] in _ISO_AUDIO_BRANDS:
                return False
            brands = [payload[:4]] + [
                payload[index : index + 4]
                for index in range(8, len(payload), 4)
            ]
            return any(
                brand in _ISO_VIDEO_BRANDS
                or brand.startswith(b"3gp")
                or brand.startswith(b"3g2")
                for brand in brands
            )
        offset += size
    return False


def _ebml_doctype(sample: bytes) -> str:
    marker = b"\x42\x82"
    offset = sample.find(marker, 4)
    if offset < 0 or offset + len(marker) >= len(sample):
        return ""
    size_offset = offset + len(marker)
    first = sample[size_offset]
    width = next((index for index in range(1, 9) if first & (1 << (8 - index))), 0)
    if not width or size_offset + width > len(sample):
        return ""
    size = first & ((1 << (8 - width)) - 1)
    for byte in sample[size_offset + 1 : size_offset + width]:
        size = (size << 8) | byte
    value_offset = size_offset + width
    if size <= 0 or value_offset + size > len(sample):
        return ""
    try:
        return sample[value_offset : value_offset + size].decode("ascii").lower()
    except UnicodeDecodeError:
        return ""


def _mpeg_ts_has_sync(sample: bytes, packet_size: int, sync_offset: int) -> bool:
    sync_positions = [sync_offset + packet_size * index for index in range(3)]
    return not any(
        position >= len(sample) or sample[position] != 0x47
        for position in sync_positions
    )


def _mpeg_ts_has_video(sample: bytes, packet_size: int, sync_offset: int) -> bool:
    if not _mpeg_ts_has_sync(sample, packet_size, sync_offset):
        return False
    return any(
        sample[index : index + 3] == b"\x00\x00\x01"
        and 0xE0 <= sample[index + 3] <= 0xEF
        for index in range(len(sample) - 3)
    )


def _detect_video_container_signature(sample: bytes) -> str:
    if _iso_bmff_is_video(sample):
        return "iso-bmff"
    if sample.startswith(b"\x1a\x45\xdf\xa3"):
        doctype = _ebml_doctype(sample)
        if doctype in {"matroska", "webm"}:
            return doctype
    if (
        len(sample) >= 12
        and sample.startswith(b"RIFF")
        and sample[8:12] == b"AVI "
    ):
        return "avi"
    if (
        len(sample) >= 9
        and sample.startswith(b"FLV\x01")
        and sample[4] & 0x01
        and int.from_bytes(sample[5:9], "big") >= 9
    ):
        return "flv"
    if sample.startswith(_ASF_HEADER):
        return "asf"
    if sample.startswith(_MXF_HEADER_PREFIX):
        return "mxf"
    if sample.startswith(b"OggS"):
        return "ogg-video"
    if sample.startswith(b"\x00\x00\x01\xba"):
        return "mpeg-ps"
    if _mpeg_ts_has_sync(sample, 188, 0) or _mpeg_ts_has_sync(sample, 192, 4):
        return "mpeg-ts"
    return ""


def _detect_video_container(sample: bytes) -> str:
    container = _detect_video_container_signature(sample)
    if container in {"iso-bmff", "flv", "mxf"}:
        return container
    if container in {"matroska", "webm"} and b"\x83\x81\x01" in sample:
        return container
    if container == "avi" and b"vids" in sample:
        return container
    if container == "asf" and _ASF_VIDEO_STREAM in sample:
        return container
    if container == "ogg-video" and b"\x80theora" in sample:
        return container
    if container == "mpeg-ps" and any(
        sample[index : index + 3] == b"\x00\x00\x01"
        and 0xE0 <= sample[index + 3] <= 0xEF
        for index in range(len(sample) - 3)
    ):
        return container
    if container == "mpeg-ts" and (
        _mpeg_ts_has_video(sample, 188, 0)
        or _mpeg_ts_has_video(sample, 192, 4)
    ):
        return container
    return ""


def video_media_type(path: Path) -> str:
    details = _VIDEO_FORMATS.get(path.suffix.lower())
    if details is None:
        raise VideoPathError("input is not a supported video file")
    return details[0]


def validate_video_sample(path: Path, sample: bytes) -> str:
    details = _VIDEO_FORMATS.get(path.suffix.lower())
    if details is None or _detect_video_container(sample) not in details[1]:
        raise VideoPathError("input is not a supported video file")
    return details[0]


def inspect_video_descriptor(path: Path, descriptor: int) -> tuple[str, bool]:
    """Return the declared MIME and whether bounded bytes prove a video track."""
    details = _VIDEO_FORMATS.get(path.suffix.lower())
    if details is None:
        raise VideoPathError("input is not a supported video file")
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        sample = os.read(descriptor, VIDEO_HEADER_BYTES)
        if _detect_video_container(sample) in details[1]:
            return details[0], True

        size = os.fstat(descriptor).st_size
        probe_limit = min(size, VIDEO_PROBE_BYTES)
        chunks = [sample]
        remaining = probe_limit - len(sample)
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        sample = b"".join(chunks)
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)
    if _detect_video_container(sample) in details[1]:
        return details[0], True
    if (
        len(sample) == probe_limit
        and _detect_video_container_signature(sample) in details[1]
    ):
        return details[0], False
    raise VideoPathError("input is not a supported video file")


def validate_video_descriptor(path: Path, descriptor: int) -> str:
    return inspect_video_descriptor(path, descriptor)[0]


def _validate_video_file(path: Path, expected: os.stat_result) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != expected.st_dev
            or opened.st_ino != expected.st_ino
            or opened.st_size != expected.st_size
        ):
            raise VideoPathError("input file changed during validation")
        validate_video_descriptor(path, descriptor)
    finally:
        os.close(descriptor)


def validate_input_file(raw: str, agent_id: str) -> Path:
    if not isinstance(raw, str) or len(raw.encode()) > _MAX_PATH_BYTES:
        raise VideoPathError("input path is invalid")
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        raise VideoPathError("input path must be absolute")
    try:
        if candidate.is_symlink():
            raise VideoPathError("input symlink is not allowed")
        resolved = candidate.resolve(strict=True)
        stat_result = resolved.stat()
    except (OSError, ValueError) as exc:
        raise VideoPathError("input file is unavailable") from exc
    if not resolved.is_file() or stat_result.st_size <= 0:
        raise VideoPathError("input file is unavailable")
    raw_ok = any(_under(resolved, Path(root).resolve()) for root in _RAW_ROOTS)
    output = output_root(agent_id)
    if not raw_ok and not _under(resolved, output):
        raise VideoPathError("input path is outside the media workspace")
    try:
        _validate_video_file(resolved, stat_result)
    except VideoPathError:
        raise
    except OSError as exc:
        raise VideoPathError("input file is unavailable") from exc
    return resolved


def validate_output_file(raw: str, agent_id: str, *, session_id: str = "") -> Path:
    """Validate a persisted result strictly inside the agent output bucket."""
    if not isinstance(raw, str) or len(raw.encode()) > _MAX_PATH_BYTES:
        raise VideoPathError("output checkpoint path is invalid")
    candidate = Path(raw.strip())
    if not candidate.is_absolute() or candidate.is_symlink():
        raise VideoPathError("output checkpoint path is invalid")
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise VideoPathError("output checkpoint is unavailable")
    except (OSError, ValueError) as exc:
        raise VideoPathError("output checkpoint is unavailable") from exc
    root = output_root(agent_id).resolve()
    boundary = root
    if session_id:
        bucket = safe_id(session_id, fallback="")
        if not bucket:
            raise VideoPathError("output checkpoint session is invalid")
        boundary = (root / bucket).resolve()
    if not _under(resolved, boundary):
        raise VideoPathError("output checkpoint is outside the agent output bucket")
    return resolved


def output_root(agent_id: str) -> Path:
    try:
        from tools.runtime_workdir import agent_output_dir

        raw = agent_output_dir()
    except Exception:
        raw = None
    if not raw:
        raw = os.environ.get("ZET_AGENT_OUTPUT_DIR", "")
    if not raw or not os.path.isabs(raw):
        raise VideoPathError("agent output directory is unavailable")
    raw_root = Path(raw)
    if raw_root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    root = raw_root.resolve()
    if root.exists() and root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    if root.is_symlink():
        raise VideoPathError("agent output directory is a symlink")
    # local-server injects ZET_AGENT_OUTPUT_DIR as
    # <agents-data>/<agent_id>/output for the active multiplex profile. Adding
    # agent_id again would create output/<agent_id>/... and make the first
    # child look like a session bucket to produced-file validation.
    if not safe_id(agent_id, fallback=""):
        raise VideoPathError("agent id is invalid")
    return root


def result_path(
    agent_id: str,
    filename: str,
    *,
    allow_existing: bool = False,
    session_id: str = "",
) -> Path:
    name = Path(str(filename or "video-edit.mp4").strip()).name
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
        raise VideoPathError("invalid result filename")
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    root = output_root(agent_id)
    if session_id:
        bucket = safe_id(session_id, fallback="")
        if not bucket:
            raise VideoPathError("result session is invalid")
        root = root / bucket
        root.mkdir(mode=0o750, parents=False, exist_ok=True)
        if root.is_symlink():
            raise VideoPathError("result session directory is a symlink")
    candidate = root / name
    if candidate.is_symlink():
        raise VideoPathError("result path is unavailable")
    target = candidate.resolve()
    if not _under(target, root):
        raise VideoPathError("result path is unavailable")
    if target.exists() and (not allow_existing or not target.is_file()):
        raise VideoPathError("result path is unavailable")
    return target
