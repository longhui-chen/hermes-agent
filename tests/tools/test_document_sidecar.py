import json
import os

import pytest

from tools.document_sidecar import (
    CanonicalDocumentUnavailable,
    DOC_MD_PATH_XATTR,
    FILE_ID_XATTR,
    is_parsed_document,
    resolve_canonical_document,
)
from tools.file_tools import clear_file_ops_cache, read_file_tool


def test_document_types_use_runtime_parsing_contract():
    assert is_parsed_document("report.pdf")
    assert is_parsed_document("book.DOCX")
    assert is_parsed_document("slides.pptx")
    assert not is_parsed_document("notes.md")
    assert not is_parsed_document("notebook.ipynb")


def test_resolve_canonical_document_requires_runtime_pointer(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    monkeypatch.setattr(
        os,
        "getxattr",
        lambda *_: (_ for _ in ()).throw(OSError("missing")),
        raising=False,
    )

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_read_file_reads_canonical_markdown_not_binary_source(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-binary-source")
    source_stat = source.stat()
    sidecar = (
        tmp_path
        / ".cache"
        / "local"
        / "12"
        / "34"
        / "1234"
        / f"{source_stat.st_mtime_ns}-{source_stat.st_size}"
        / ".doc"
        / "content.md"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("# Canonical\n\nparsed once", encoding="utf-8")

    def fake_getxattr(path, key):
        assert str(path) == str(source)
        if key == DOC_MD_PATH_XATTR:
            return os.fsencode(sidecar)
        if key == FILE_ID_XATTR:
            return b"1234"
        raise OSError("missing")

    monkeypatch.setattr(os, "getxattr", fake_getxattr, raising=False)
    clear_file_ops_cache()
    result = json.loads(read_file_tool(str(source)))

    assert "Canonical" in result["content"]
    assert "PDF-binary-source" not in result["content"]


def test_resolve_canonical_document_allows_managed_legacy_upload_root(
    tmp_path, monkeypatch
):
    source = tmp_path / "agents" / "data" / "main" / "uploads" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    source_stat = source.stat()
    cache_root = tmp_path / "subvol"
    sidecar = (
        cache_root
        / ".cache"
        / "local"
        / "12"
        / "34"
        / "1234"
        / f"{source_stat.st_mtime_ns}-{source_stat.st_size}"
        / ".doc"
        / "content.md"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("legacy upload parsed", encoding="utf-8")

    def fake_getxattr(_path, key):
        return os.fsencode(sidecar) if key == DOC_MD_PATH_XATTR else b"1234"

    monkeypatch.setattr(os, "getxattr", fake_getxattr, raising=False)
    monkeypatch.setattr(
        "tools.document_sidecar._MANAGED_LEGACY_SOURCE_ROOT",
        tmp_path / "agents" / "data",
    )
    monkeypatch.setattr("tools.document_sidecar._MANAGED_CACHE_ROOT", cache_root)

    assert resolve_canonical_document(source) == sidecar


def test_resolve_canonical_document_rejects_unbound_file_id(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    source_stat = source.stat()
    sidecar = (
        tmp_path
        / ".cache"
        / "local"
        / "56"
        / "78"
        / "5678"
        / f"{source_stat.st_mtime_ns}-{source_stat.st_size}"
        / ".doc"
        / "content.md"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("another document", encoding="utf-8")

    def fake_getxattr(_path, key):
        return os.fsencode(sidecar) if key == DOC_MD_PATH_XATTR else b"1234"

    monkeypatch.setattr(os, "getxattr", fake_getxattr, raising=False)

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_resolve_canonical_document_rejects_stale_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    sidecar = (
        tmp_path
        / ".cache"
        / "local"
        / "12"
        / "34"
        / "1234"
        / "1-4"
        / ".doc"
        / "content.md"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("stale document", encoding="utf-8")

    def fake_getxattr(_path, key):
        return os.fsencode(sidecar) if key == DOC_MD_PATH_XATTR else b"1234"

    monkeypatch.setattr(os, "getxattr", fake_getxattr, raising=False)

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_resolve_canonical_document_rejects_arbitrary_markdown_pointer(
    tmp_path, monkeypatch
):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    unrelated = tmp_path / "secrets.md"
    unrelated.write_text("not a runtime artifact", encoding="utf-8")
    monkeypatch.setattr(
        os, "getxattr", lambda *_: os.fsencode(unrelated), raising=False
    )

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_resolve_canonical_document_rejects_suffix_only_pointer(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    unrelated = tmp_path / "private" / ".doc" / "content.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not a runtime artifact", encoding="utf-8")
    monkeypatch.setattr(
        os, "getxattr", lambda *_: os.fsencode(unrelated), raising=False
    )

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_read_file_refuses_document_when_canonical_artifact_is_missing(
    tmp_path, monkeypatch
):
    source = tmp_path / "report.docx"
    source.write_bytes(b"zip")
    monkeypatch.setattr(
        os,
        "getxattr",
        lambda *_: (_ for _ in ()).throw(OSError("missing")),
        raising=False,
    )

    result = json.loads(read_file_tool(str(source)))

    assert "Canonical document content is not available yet" in result["error"]
    assert "do not parse the source with another tool" in result["error"]
