"""Resolve the runtime-owned canonical Markdown for binary documents.

This module deliberately contains no parser.  The runtime document pipeline
owns extraction and publishes the artifact path on the source file.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


DOC_MD_PATH_XATTR = "user.zettlab.doc_md_path.v1"

PARSED_DOCUMENT_EXTENSIONS = frozenset(
    {
        ".doc", ".docx", ".epub", ".key", ".mobi", ".msg",
        ".numbers", ".odp", ".ods", ".odt", ".pages", ".pdf",
        ".pps", ".ppsx", ".ppt", ".pptx", ".prc", ".rtf",
        ".xls", ".xlsx",
    }
)


class CanonicalDocumentUnavailable(Exception):
    """The runtime has not published a readable canonical artifact yet."""


def is_parsed_document(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in PARSED_DOCUMENT_EXTENSIONS


def resolve_canonical_document(path: str | os.PathLike[str]) -> Path:
    """Return the validated canonical Markdown artifact for *path*."""

    source = Path(path)
    try:
        raw = os.getxattr(source, DOC_MD_PATH_XATTR)
    except (AttributeError, OSError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc

    try:
        target = Path(os.fsdecode(raw))
    except (TypeError, UnicodeError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc

    # The runtime contract is deliberately narrower than "any Markdown path":
    #
    #   <base>/.cache/<fs>/<aa>/<bb>/<file-id>/<mtime>-<size>/.doc/content.md
    #
    # The xattr lives on a user-owned source file, so a suffix-only check would
    # still be an arbitrary-file indirection primitive. Pin the full cache
    # shape and require the source to live below the same runtime base root.
    if (
        not target.is_absolute()
        or target.name != "content.md"
        or target.parent.name != ".doc"
        or len(target.parents) < 8
        or target.parents[6].name != ".cache"
    ):
        raise CanonicalDocumentUnavailable(str(source))
    try:
        source_real = source.resolve(strict=True)
        target_stat = target.lstat()
        target_real = target.resolve(strict=True)
        cache_root = target.parents[7].resolve(strict=True)
        source_real.relative_to(cache_root)
        if (
            target_real != target
            or not stat.S_ISREG(target_stat.st_mode)
            or not os.access(target, os.R_OK)
        ):
            raise CanonicalDocumentUnavailable(str(source))
    except (OSError, ValueError) as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc
    return target
