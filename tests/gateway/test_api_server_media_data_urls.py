"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import (  # noqa: E402
    _StreamingMediaDeltaFilter,
    _resolve_media_to_data_urls,
)

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_missing_file_is_replaced_without_leaking_host_path(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        out = _resolve_media_to_data_urls(text)
        self.assertNotIn("/nonexistent/path", out)
        self.assertIn("Couldn't deliver", out)

    def test_non_image_path_is_not_exposed(self):
        text = "MEDIA:/nonexistent/archive.zip"
        out = _resolve_media_to_data_urls(text)
        self.assertNotIn("/nonexistent/archive.zip", out)

    def test_lowercase_marker_is_also_filtered(self):
        out = _resolve_media_to_data_urls("media:/nonexistent/private.png")
        self.assertNotIn("/nonexistent/private.png", out)


def test_stream_filter_handles_marker_split_across_deltas(tmp_path):
    image = tmp_path / "stream.png"
    image.write_bytes(_PNG_BYTES)
    stream_filter = _StreamingMediaDeltaFilter()

    output = []
    output.extend(stream_filter.feed("done ME"))
    output.extend(stream_filter.feed(f"DIA:{image}"))
    output.extend(stream_filter.feed("\nnext"))
    output.extend(stream_filter.finish())
    rendered = "".join(output)

    assert "done " in rendered and "next" in rendered
    assert "data:image/png;base64," in rendered
    assert "MEDIA:" not in rendered and str(image) not in rendered


def test_stream_filter_never_leaks_missing_host_path():
    stream_filter = _StreamingMediaDeltaFilter()
    output = stream_filter.feed("MEDIA:/secret/missing.png")
    output += stream_filter.finish()
    rendered = "".join(output)

    assert "/secret/missing.png" not in rendered
    assert "Couldn't deliver" in rendered


if __name__ == "__main__":
    unittest.main()
