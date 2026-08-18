import json
import os

import pytest

from tools.document_sidecar import (
    CanonicalDocumentUnavailable,
    DOC_MD_PATH_XATTR,
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
    monkeypatch.setattr(os, "getxattr", lambda *_: (_ for _ in ()).throw(OSError("missing")), raising=False)

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_read_file_reads_canonical_markdown_not_binary_source(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-binary-source")
    sidecar = (
        tmp_path
        / ".cache"
        / "local"
        / "12"
        / "34"
        / "1234"
        / "1-18"
        / ".doc"
        / "content.md"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("# Canonical\n\nparsed once", encoding="utf-8")

    def fake_getxattr(path, key):
        assert str(path) == str(source)
        assert key == DOC_MD_PATH_XATTR
        return os.fsencode(sidecar)

    monkeypatch.setattr(os, "getxattr", fake_getxattr, raising=False)
    clear_file_ops_cache()
    result = json.loads(read_file_tool(str(source)))

    assert "Canonical" in result["content"]
    assert "PDF-binary-source" not in result["content"]


def test_resolve_canonical_document_rejects_arbitrary_markdown_pointer(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    unrelated = tmp_path / "secrets.md"
    unrelated.write_text("not a runtime artifact", encoding="utf-8")
    monkeypatch.setattr(os, "getxattr", lambda *_: os.fsencode(unrelated), raising=False)

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_resolve_canonical_document_rejects_suffix_only_pointer(tmp_path, monkeypatch):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF")
    unrelated = tmp_path / "private" / ".doc" / "content.md"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("not a runtime artifact", encoding="utf-8")
    monkeypatch.setattr(os, "getxattr", lambda *_: os.fsencode(unrelated), raising=False)

    with pytest.raises(CanonicalDocumentUnavailable):
        resolve_canonical_document(source)


def test_read_file_refuses_document_when_canonical_artifact_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "report.docx"
    source.write_bytes(b"zip")
    monkeypatch.setattr(os, "getxattr", lambda *_: (_ for _ in ()).throw(OSError("missing")), raising=False)

    result = json.loads(read_file_tool(str(source)))

    assert "Canonical document content is not available yet" in result["error"]
    assert "do not parse the source with another tool" in result["error"]
