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
    # canonical artifacts always end in .doc/content.md.  Do not turn a
    # user-controlled xattr into an arbitrary-file indirection primitive.
    if (
        not target.is_absolute()
        or target.name != "content.md"
        or target.parent.name != ".doc"
    ):
        raise CanonicalDocumentUnavailable(str(source))
    try:
        target_stat = target.lstat()
        if not stat.S_ISREG(target_stat.st_mode) or not os.access(target, os.R_OK):
            raise CanonicalDocumentUnavailable(str(source))
    except OSError as exc:
        raise CanonicalDocumentUnavailable(str(source)) from exc
    return target
