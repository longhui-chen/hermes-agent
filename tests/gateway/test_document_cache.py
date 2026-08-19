"""
Tests for document cache utilities in gateway/platforms/base.py.

Covers: get_document_cache_dir, cache_document_from_bytes,
        cleanup_document_cache, SUPPORTED_DOCUMENT_TYPES.
"""

import os
import time
from pathlib import Path

import pytest

from gateway.platforms.base import (
    SUPPORTED_DOCUMENT_TYPES,
    cache_document_from_bytes,
    cleanup_document_cache,
    get_audio_cache_dir,
    get_document_cache_dir,
    get_image_cache_dir,
    get_video_cache_dir,
)

# ---------------------------------------------------------------------------
# Fixture: redirect DOCUMENT_CACHE_DIR to a temp directory for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point the module-level DOCUMENT_CACHE_DIR to a fresh tmp_path."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.AUDIO_CACHE_DIR", tmp_path / "audio_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.IMAGE_CACHE_DIR", tmp_path / "image_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )


# ---------------------------------------------------------------------------
# TestGetDocumentCacheDir
# ---------------------------------------------------------------------------

class TestGetDocumentCacheDir:
    def test_creates_directory(self, tmp_path):
        cache_dir = get_document_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# TestCacheDocumentFromBytes
# ---------------------------------------------------------------------------

class TestCacheDocumentFromBytes:
    def test_basic_caching(self):
        data = b"hello world"
        path = cache_document_from_bytes(data, "test.txt")
        assert os.path.exists(path)
        assert Path(path).read_bytes() == data

    def test_filename_preserved_in_path(self):
        path = cache_document_from_bytes(b"data", "report.pdf")
        assert "report.pdf" in os.path.basename(path)

    def test_empty_filename_uses_fallback(self):
        path = cache_document_from_bytes(b"data", "")
        assert "document" in os.path.basename(path)

    def test_document_uses_same_inbound_size_gate(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.base.get_inbound_media_max_bytes", lambda: 3,
        )

        with pytest.raises(ValueError, match="4 bytes > 3 bytes"):
            cache_document_from_bytes(b"data", "report.pdf")

    def test_filename_controls_are_removed_and_length_is_bounded(self):
        hostile = "quarterly\nreport\x1b[31m_" + ("x" * 400) + ".pdf"
        path = Path(cache_document_from_bytes(b"data", hostile))

        assert "\n" not in path.name and "\x1b" not in path.name
        assert len(path.name.encode()) <= 255
        assert path.suffix == ".pdf"


# ---------------------------------------------------------------------------
# TestCleanupDocumentCache
# ---------------------------------------------------------------------------

class TestCleanupDocumentCache:
    def test_removes_old_files(self, tmp_path):
        cache_dir = get_document_cache_dir()
        old_file = cache_dir / "old.txt"
        old_file.write_text("old")
        # Set modification time to 48 hours ago
        old_mtime = time.time() - 48 * 3600
        os.utime(old_file, (old_mtime, old_mtime))

        removed = cleanup_document_cache(max_age_hours=24)
        assert removed == 1
        assert not old_file.exists()


class TestSharedMediaCacheBudget:
    def test_oldest_file_is_evicted_before_total_bytes_exceed_cap(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_TOTAL_BYTES", 10,
            raising=False,
        )
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_FILES", 100,
            raising=False,
        )
        first = Path(cache_document_from_bytes(b"1111", "first.bin"))
        second = Path(cache_document_from_bytes(b"2222", "second.bin"))
        assert first.exists() and second.exists(), "夹具必须先把两个旧附件落盘"
        old = time.time() - 25 * 60 * 60
        os.utime(first, (old, old))

        newest = Path(cache_document_from_bytes(b"3333", "newest.bin"))
        files = [
            p
            for cache_dir in (
                get_image_cache_dir(), get_audio_cache_dir(),
                get_video_cache_dir(), get_document_cache_dir(),
            )
            for p in cache_dir.iterdir()
            if p.is_file()
        ]

        assert newest.exists()
        assert not first.exists(), "达到总字节预算时必须淘汰最旧媒体缓存"
        assert sum(p.stat().st_size for p in files) <= 10

    def test_tiny_files_cannot_grow_cache_entry_count_without_bound(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_TOTAL_BYTES", 1024,
            raising=False,
        )
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_FILES", 2,
            raising=False,
        )
        paths = [
            Path(cache_document_from_bytes(b"0", "0.bin")),
            Path(cache_document_from_bytes(b"1", "1.bin")),
        ]
        old = time.time() - 25 * 60 * 60
        os.utime(paths[0], (old, old))
        paths.append(Path(cache_document_from_bytes(b"2", "2.bin")))

        assert sum(path.exists() for path in paths) == 2
        assert not paths[0].exists() and paths[-1].exists()

    def test_recent_inflight_file_is_not_evicted(self, monkeypatch):
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_TOTAL_BYTES", 6,
            raising=False,
        )
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_CACHE_MAX_FILES", 2,
            raising=False,
        )
        inflight = Path(cache_document_from_bytes(b"1111", "inflight.bin"))
        assert inflight.exists(), "夹具必须先注册仍可能被 agent 打开的近期附件"

        with pytest.raises(ValueError, match="capacity is full"):
            cache_document_from_bytes(b"2222", "new.bin")

        assert inflight.exists(), "容量满时不得删除仍在 24 小时 TTL 内的附件"


# ---------------------------------------------------------------------------
# TestSupportedDocumentTypes
# ---------------------------------------------------------------------------

class TestSupportedDocumentTypes:
    def test_all_extensions_have_mime_types(self):
        for ext, mime in SUPPORTED_DOCUMENT_TYPES.items():
            assert ext.startswith("."), f"{ext} missing leading dot"
            assert "/" in mime, f"{mime} is not a valid MIME type"


# ---------------------------------------------------------------------------
# TestCacheMediaBytes — the unified, platform-agnostic caching primitive
# ---------------------------------------------------------------------------

# 1x1 transparent PNG (passes cache_image_from_bytes validation)
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)


class TestCacheMediaBytes:
    def test_pdf_routes_to_document(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"%PDF-1.4 body", filename="report.pdf", mime_type="application/pdf")
        assert result is not None
        assert result.kind == "document"
        assert result.media_type == "application/pdf"
        assert "report.pdf" in result.display_name
        assert os.path.exists(result.path)
        assert "report.pdf" in result.context_note()

    def test_png_routes_to_image(self):
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(_PNG_1PX, filename="photo.png", mime_type="image/png")
        assert result is not None
        assert result.kind == "image"
        assert result.media_type == "image/png"
        assert os.path.exists(result.path)


    def test_unknown_document_cached_as_octet_stream(self):
        """Unknown file types are cached (not dropped) so the agent can inspect them.

        Authorization to message the agent is the gate, not the file extension.
        """
        from gateway.platforms.base import cache_media_bytes
        result = cache_media_bytes(b"MZ", filename="program.exe", mime_type="application/x-msdownload")
        assert result is not None
        assert result.kind == "document"
        # Caller-supplied MIME is preserved when present.
        assert result.media_type == "application/x-msdownload"
        assert os.path.exists(result.path)
