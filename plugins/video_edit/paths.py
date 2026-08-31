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
_ISO_BMFF_MAX_BOX_DEPTH = 8
_ISO_BMFF_MAX_TOP_LEVEL_BOXES = 4096
_ISO_BMFF_MAX_CHILD_BOXES = 4096
_ISO_BMFF_MAX_MOOV_BYTES = VIDEO_PROBE_BYTES
_ISO_BMFF_READ_CHUNK_BYTES = 64 * 1024
_EBML_HEADER_ID = b"\x1a\x45\xdf\xa3"
_EBML_DOCTYPE_ID = b"\x42\x82"
_EBML_SEGMENT_ID = b"\x18\x53\x80\x67"
_EBML_TRACKS_ID = b"\x16\x54\xae\x6b"
_EBML_TRACK_ENTRY_ID = b"\xae"
_EBML_TRACK_TYPE_ID = b"\x83"
_EBML_MAX_ELEMENTS = 4096


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


def _iso_bmff_boxes(
    sample: bytes,
    start: int,
    end: int,
):
    """Yield complete ISO-BMFF child boxes inside one bounded parent."""
    offset = start
    while offset < end:
        if end - offset < 8:
            return
        size = int.from_bytes(sample[offset : offset + 4], "big")
        kind = sample[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if end - offset < 16:
                return
            size = int.from_bytes(sample[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or size > end - offset:
            return
        box_end = offset + size
        yield kind, offset + header_size, box_end
        offset = box_end


def _iso_bmff_complete_boxes(
    sample: bytes,
    start: int,
    end: int,
) -> list[tuple[bytes, int, int]] | None:
    """Return all boxes only when the bounded parent is structurally complete."""
    if start < 0 or end < start or end > len(sample):
        return None
    boxes: list[tuple[bytes, int, int]] = []
    offset = start
    while offset < end:
        if len(boxes) >= _ISO_BMFF_MAX_CHILD_BOXES:
            return None
        if end - offset < 8:
            return None
        size = int.from_bytes(sample[offset : offset + 4], "big")
        kind = sample[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if end - offset < 16:
                return None
            size = int.from_bytes(sample[offset + 8 : offset + 16], "big")
            header_size = 16
        elif size == 0:
            size = end - offset
        if size < header_size or size > end - offset:
            return None
        box_end = offset + size
        boxes.append((kind, offset + header_size, box_end))
        offset = box_end
    return boxes


def _iso_bmff_mdia_video_status(
    sample: bytes,
    start: int,
    end: int,
    depth: int,
) -> tuple[bool, bool]:
    if depth > _ISO_BMFF_MAX_BOX_DEPTH:
        return False, False
    boxes = _iso_bmff_complete_boxes(sample, start, end)
    if boxes is None:
        return False, False
    handler_count = 0
    has_video = False
    for kind, payload_start, box_end in boxes:
        if kind != b"hdlr":
            continue
        handler_count += 1
        if handler_count > 1:
            return False, False
        if box_end - payload_start < 12:
            return False, False
        # FullBox(version + flags), pre_defined, then handler_type.
        if sample[payload_start + 8 : payload_start + 12] == b"vide":
            has_video = True
    return handler_count == 1, has_video


def _iso_bmff_mdia_has_video_handler(
    sample: bytes,
    start: int,
    end: int,
    depth: int,
) -> bool:
    valid, has_video = _iso_bmff_mdia_video_status(sample, start, end, depth)
    return valid and has_video


def _iso_bmff_trak_video_status(
    sample: bytes,
    start: int,
    end: int,
    depth: int,
) -> tuple[bool, bool]:
    if depth > _ISO_BMFF_MAX_BOX_DEPTH:
        return False, False
    boxes = _iso_bmff_complete_boxes(sample, start, end)
    if boxes is None:
        return False, False
    mdia_count = 0
    has_video = False
    for kind, payload_start, box_end in boxes:
        if kind != b"mdia":
            continue
        mdia_count += 1
        if mdia_count > 1:
            return False, False
        valid, mdia_has_video = _iso_bmff_mdia_video_status(
            sample, payload_start, box_end, depth + 1
        )
        if not valid:
            return False, False
        has_video = has_video or mdia_has_video
    return mdia_count == 1, has_video


def _iso_bmff_trak_has_video_handler(
    sample: bytes,
    start: int,
    end: int,
    depth: int,
) -> bool:
    valid, has_video = _iso_bmff_trak_video_status(sample, start, end, depth)
    return valid and has_video


def _iso_bmff_moov_has_video_track(
    sample: bytes,
    start: int,
    end: int,
) -> bool:
    boxes = _iso_bmff_complete_boxes(sample, start, end)
    if boxes is None:
        return False
    has_video = False
    for trak_kind, trak_start, trak_end in boxes:
        if trak_kind != b"trak":
            continue
        valid, trak_has_video = _iso_bmff_trak_video_status(
            sample, trak_start, trak_end, 2
        )
        if not valid:
            return False
        has_video = has_video or trak_has_video
    return has_video


def _iso_bmff_has_video_track(sample: bytes) -> bool:
    """Require an actual ``hdlr=vide`` track, not just an ISO brand."""
    if not _iso_bmff_is_video(sample):
        return False
    # ``sample`` is intentionally only a bounded prefix.  A valid fast-start
    # moov may be followed by an incomplete mdat tail in that prefix; the
    # descriptor-aware path performs whole-file top-level validation when the
    # prefix cannot prove the track.
    for kind, payload_start, box_end in _iso_bmff_boxes(
        sample, 0, len(sample)
    ):
        if kind == b"moov" and _iso_bmff_moov_has_video_track(
            sample, payload_start, box_end
        ):
            return True
    return False


def _iso_bmff_read_top_level_header(
    descriptor: int,
    offset: int,
    file_size: int,
) -> tuple[bytes, int, int] | None:
    """Read one bounded top-level header and return kind, end, header size."""
    if offset < 0 or offset > file_size - 8:
        return None
    os.lseek(descriptor, offset, os.SEEK_SET)
    header = os.read(descriptor, 8)
    if len(header) < 8:
        return None
    size = int.from_bytes(header[:4], "big")
    kind = header[4:8]
    header_size = 8
    if size == 1:
        largesize = os.read(descriptor, 8)
        if len(largesize) < 8:
            return None
        size = int.from_bytes(largesize, "big")
        header_size = 16
    elif size == 0:
        size = file_size - offset
    if size < header_size or size > file_size - offset:
        return None
    return kind, offset + size, header_size


def _iso_bmff_read_range(
    descriptor: int,
    offset: int,
    size: int,
) -> bytes | None:
    """Read one bounded box without touching unrelated media payload."""
    if size < 0 or size > _ISO_BMFF_MAX_MOOV_BYTES:
        return None
    os.lseek(descriptor, offset, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(
            descriptor,
            min(_ISO_BMFF_READ_CHUNK_BYTES, remaining),
        )
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iso_bmff_has_video_track_descriptor(
    descriptor: int,
    file_size: int,
) -> bool:
    """Find a bounded ``moov`` by seeking over trusted top-level sizes."""
    offset = 0
    saw_ftyp = False
    saw_moov = False
    has_video_track = False
    for _ in range(_ISO_BMFF_MAX_TOP_LEVEL_BOXES):
        header = _iso_bmff_read_top_level_header(
            descriptor, offset, file_size
        )
        if header is None:
            return False
        kind, box_end, header_size = header
        if kind == b"ftyp":
            if saw_ftyp:
                return False
            saw_ftyp = True
        elif kind == b"moov":
            if not saw_ftyp or saw_moov:
                return False
            saw_moov = True
            box_size = box_end - offset
            raw = _iso_bmff_read_range(descriptor, offset, box_size)
            if raw is None:
                return False
            # The header parser has already checked the box size; only parse
            # complete child boxes inside this bounded moov snapshot.
            has_video_track = _iso_bmff_moov_has_video_track(
                raw, header_size, len(raw)
            )
        if box_end <= offset:
            return False
        offset = box_end
        if offset == file_size:
            break
    if offset != file_size:
        return False
    return saw_moov and has_video_track


def _ebml_vint_width(first: int, maximum: int) -> int:
    if first <= 0:
        return 0
    return next(
        (
            width
            for width in range(1, maximum + 1)
            if first & (1 << (8 - width))
        ),
        0,
    )


def _ebml_read_id(
    sample: bytes,
    offset: int,
    end: int,
) -> tuple[bytes, int] | None:
    available_end = min(end, len(sample))
    if offset < 0 or offset >= available_end:
        return None
    width = _ebml_vint_width(sample[offset], 4)
    if not width or offset + width > available_end:
        return None
    raw = sample[offset : offset + width]
    data = int.from_bytes(raw, "big") & ((1 << (7 * width)) - 1)
    if data in {0, (1 << (7 * width)) - 1}:
        return None
    return raw, offset + width


def _ebml_read_size(
    sample: bytes,
    offset: int,
    end: int,
) -> tuple[int | None, int] | None:
    available_end = min(end, len(sample))
    if offset < 0 or offset >= available_end:
        return None
    width = _ebml_vint_width(sample[offset], 8)
    if not width or offset + width > available_end:
        return None
    value = sample[offset] & ((1 << (8 - width)) - 1)
    for byte in sample[offset + 1 : offset + width]:
        value = (value << 8) | byte
    unknown = (1 << (7 * width)) - 1
    return (None if value == unknown else value), offset + width


def _ebml_element_header(
    sample: bytes,
    offset: int,
    parent_end: int,
) -> tuple[bytes, int, int | None] | None:
    element_id = _ebml_read_id(sample, offset, parent_end)
    if element_id is None:
        return None
    raw_id, size_offset = element_id
    size_value = _ebml_read_size(sample, size_offset, parent_end)
    if size_value is None:
        return None
    size, payload_start = size_value
    if payload_start > parent_end:
        return None
    if size is None:
        return raw_id, payload_start, None
    if size > parent_end - payload_start:
        return None
    return raw_id, payload_start, payload_start + size


def _ebml_header(sample: bytes) -> tuple[str, int] | None:
    root = _ebml_element_header(sample, 0, len(sample))
    if root is None:
        return None
    element_id, payload_start, payload_end = root
    if element_id != _EBML_HEADER_ID or payload_end is None:
        return None
    offset = payload_start
    doctype = ""
    for _ in range(_EBML_MAX_ELEMENTS):
        if offset == payload_end:
            break
        child = _ebml_element_header(sample, offset, payload_end)
        if child is None:
            return None
        child_id, child_start, child_end = child
        if child_end is None or child_end > len(sample):
            return None
        if child_id == _EBML_DOCTYPE_ID:
            if doctype or not 1 <= child_end - child_start <= 32:
                return None
            try:
                doctype = sample[child_start:child_end].decode("ascii").lower()
            except UnicodeDecodeError:
                return None
        offset = child_end
    if offset != payload_end or not doctype:
        return None
    return doctype, payload_end


def _ebml_doctype(sample: bytes) -> str:
    header = _ebml_header(sample)
    return header[0] if header is not None else ""


def _ebml_track_entry_video_status(
    sample: bytes,
    start: int,
    end: int,
) -> tuple[bool, bool]:
    if start < 0 or end < start or end > len(sample):
        return False, False
    offset = start
    track_type: int | None = None
    for _ in range(_EBML_MAX_ELEMENTS):
        if offset == end:
            break
        child = _ebml_element_header(sample, offset, end)
        if child is None:
            return False, False
        child_id, payload_start, payload_end = child
        if payload_end is None:
            return False, False
        if child_id == _EBML_TRACK_TYPE_ID:
            if track_type is not None or not 1 <= payload_end - payload_start <= 8:
                return False, False
            track_type = int.from_bytes(sample[payload_start:payload_end], "big")
        offset = payload_end
    if offset != end or track_type is None:
        return False, False
    return True, track_type == 1


def _ebml_tracks_video_status(
    sample: bytes,
    start: int,
    end: int,
) -> tuple[bool, bool]:
    if start < 0 or end < start or end > len(sample):
        return False, False
    offset = start
    track_entries = 0
    has_video = False
    for _ in range(_EBML_MAX_ELEMENTS):
        if offset == end:
            break
        child = _ebml_element_header(sample, offset, end)
        if child is None:
            return False, False
        child_id, payload_start, payload_end = child
        if payload_end is None:
            return False, False
        if child_id == _EBML_TRACK_ENTRY_ID:
            track_entries += 1
            valid, entry_has_video = _ebml_track_entry_video_status(
                sample, payload_start, payload_end
            )
            if not valid:
                return False, False
            has_video = has_video or entry_has_video
        offset = payload_end
    if offset != end or not track_entries:
        return False, False
    return True, has_video


def _ebml_segment_has_video_track(
    sample: bytes,
    start: int,
    end: int,
) -> bool:
    offset = start
    for _ in range(_EBML_MAX_ELEMENTS):
        if offset >= end or offset >= len(sample):
            return False
        child = _ebml_element_header(sample, offset, end)
        if child is None:
            return False
        child_id, payload_start, payload_end = child
        if child_id == _EBML_TRACKS_ID:
            if payload_end is None or payload_end > len(sample):
                return False
            valid, has_video = _ebml_tracks_video_status(
                sample, payload_start, payload_end
            )
            return valid and has_video
        if payload_end is None:
            return False
        offset = payload_end
    return False


def _ebml_has_video_track(
    sample: bytes,
    file_size: int,
) -> bool:
    header = _ebml_header(sample)
    if header is None or file_size < len(sample):
        return False
    doctype, offset = header
    if doctype not in {"matroska", "webm"} or offset > file_size:
        return False
    for _ in range(_EBML_MAX_ELEMENTS):
        if offset >= file_size or offset >= len(sample):
            return False
        element = _ebml_element_header(sample, offset, file_size)
        if element is None:
            return False
        element_id, payload_start, payload_end = element
        if element_id == _EBML_SEGMENT_ID:
            segment_end = file_size if payload_end is None else payload_end
            return _ebml_segment_has_video_track(
                sample, payload_start, segment_end
            )
        if payload_end is None:
            return False
        offset = payload_end
    return False


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


def _detect_video_container(
    sample: bytes,
    *,
    file_size: int | None = None,
) -> str:
    container = _detect_video_container_signature(sample)
    if container == "iso-bmff":
        return container if _iso_bmff_has_video_track(sample) else ""
    if container in {"flv", "mxf"}:
        return container
    if container in {"matroska", "webm"}:
        physical_size = len(sample) if file_size is None else file_size
        return container if _ebml_has_video_track(sample, physical_size) else ""
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
        size = os.fstat(descriptor).st_size
        signature = _detect_video_container_signature(sample)
        if signature == "iso-bmff":
            if "iso-bmff" not in details[1]:
                raise VideoPathError("input is not a supported video file")
            if _iso_bmff_has_video_track_descriptor(descriptor, size):
                return details[0], True
            # A recognized ISO brand without a bounded, structurally valid
            # video track is intentionally inconclusive; the upload caller
            # will fail closed when the packaged probe is unavailable.
            return details[0], False

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
    if _detect_video_container(sample, file_size=size) in details[1]:
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
    output = output_root(agent_id)
    uploads = upload_root(agent_id)
    workspace_roots = [
        *(Path(root).resolve() for root in _RAW_ROOTS),
        output.resolve(),
        uploads,
    ]
    if not any(_under(resolved, root) for root in workspace_roots):
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


def upload_root(agent_id: str) -> Path:
    """Return the active agent's App attachment workspace.

    local-server places inbound channel attachments beside the agent output
    bucket at ``<agents-data>/<agent>/uploads``.  Keep this sibling scoped to
    the same validated output parent so one profile cannot select another
    profile's attachments.
    """
    output = output_root(agent_id)
    candidate = output.parent / "uploads"
    if candidate.is_symlink():
        raise VideoPathError("agent upload directory is a symlink")
    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise VideoPathError("agent upload directory is unavailable") from exc
    if candidate.exists() and not candidate.is_dir():
        raise VideoPathError("agent upload directory is unavailable")
    return resolved


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
