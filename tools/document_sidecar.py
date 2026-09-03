"""Resolve the runtime-owned canonical Markdown for binary documents.

This module deliberately contains no parser.  The runtime document pipeline
owns extraction and publishes the artifact path on the source file.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


DOC_MD_PATH_XATTR = "user.zettlab.doc_md_path.v1"
FILE_ID_XATTR = "user.zettlab.file_id.v2"

# Upgraded devices retain agent uploads below the legacy profile root while
# local-server publishes canonical artifacts below the current data root.
# Keep this compatibility pair exact; arbitrary cross-root pointers remain
# invalid even when they happen to end in `.doc/content.md`.
_MANAGED_LEGACY_SOURCE_ROOT = Path("/volume1/agents/data")
_MANAGED_CACHE_ROOT = Path("/volume1/subvol")

PARSED_DOCUMENT_EXTENSIONS = frozenset({
    ".doc",
    ".docx",
    ".epub",
    ".key",
    ".mobi",
    ".msg",
    ".numbers",
    ".odp",
    ".ods",
    ".odt",
    ".pages",
    ".pdf",
    ".pps",
    ".ppsx",
    ".ppt",
    ".pptx",
    ".prc",
    ".rtf",
    ".xls",
    ".xlsx",
})


class CanonicalDocumentUnavailable(Exception):
    """The runtime has not published a readable canonical artifact yet."""


def is_parsed_document(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in PARSED_DOCUMENT_EXTENSIONS


def _bucket_prefix(file_id: str) -> tuple[str, str]:
    if len(file_id) >= 4:
        return file_id[:2], file_id[2:4]
    if len(file_id) >= 2:
        return file_id[:2], "00"
    return "00", "00"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_canonical_document(path: str | os.PathLike[str]) -> Path:
    """Return the validated canonical Markdown artifact for *path*."""

    source = Path(path)
    try:
        raw = os.getxattr(source, DOC_MD_PATH_XATTR)
        raw_file_id = os.getxattr(source, FILE_ID_XATTR)
    except (AttributeError, OSError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc

    try:
        target = Path(os.fsdecode(raw))
        file_id = os.fsdecode(raw_file_id).strip()
    except (TypeError, UnicodeError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc

    # The runtime contract is deliberately narrower than "any Markdown path":
    #
    #   <base>/.cache/<fs>/<aa>/<bb>/<file-id>/<mtime>-<size>/.doc/content.md
    #
    # The xattrs live on a user-owned source file, so a suffix-only check would
    # still be an arbitrary-file indirection primitive. Pin the full cache
    # shape and bind it to this source's file id plus content snapshot.
    if (
        not target.is_absolute()
        or target.name != "content.md"
        or target.parent.name != ".doc"
        or len(target.parents) < 8
        or target.parents[6].name != ".cache"
        or not file_id.isdecimal()
        or int(file_id) <= 0
    ):
        raise CanonicalDocumentUnavailable(str(source))
    try:
        source_real = source.resolve(strict=True)
        source_stat = source_real.stat()
        target_stat = target.lstat()
        target_real = target.resolve(strict=True)
        cache_root = target.parents[7].resolve(strict=True)
        same_root = _is_within(source_real, cache_root)
        managed_legacy_pair = cache_root == _MANAGED_CACHE_ROOT and _is_within(
            source_real, _MANAGED_LEGACY_SOURCE_ROOT
        )
        aa, bb = _bucket_prefix(file_id)
        valid_snapshots = {
            f"{source_stat.st_mtime_ns}-{source_stat.st_size}",
            f"{int(source_stat.st_mtime)}-{source_stat.st_size}",
        }
        if (
            not (same_root or managed_legacy_pair)
            or target.parents[2].name != file_id
            or target.parents[3].name != bb
            or target.parents[4].name != aa
            or target.parents[1].name not in valid_snapshots
            or target_real != target
            or not stat.S_ISREG(target_stat.st_mode)
            or not os.access(target, os.R_OK)
        ):
            raise CanonicalDocumentUnavailable(str(source))
    except (OSError, ValueError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc
    return target
