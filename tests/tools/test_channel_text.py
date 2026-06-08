"""Tests for tools/channel_text.py — shared channel-send chunking."""

from tools.channel_text import CHANNEL_SEND_MAX_RUNES, chunk_channel_text


def test_short_text_single_unchanged_chunk():
    assert chunk_channel_text("喝水啦 💧") == ["喝水啦 💧"]


def test_long_text_split_under_caps():
    chunks = chunk_channel_text("水" * 9000)
    assert len(chunks) >= 2
    # every chunk must be strictly under local-server's 4000-rune endpoint cap
    # (the whole point — otherwise the send is rejected as "text too long")...
    assert all(len(c) < 4000 for c in chunks), [len(c) for c in chunks]
    # ...and within the helper's own bound (headroom for (N/M) indicators).
    assert all(len(c) <= CHANNEL_SEND_MAX_RUNES for c in chunks), [len(c) for c in chunks]


def test_nothing_silently_dropped():
    # truncate_message only adds (N/M) indicators; it must never drop content.
    text = "水" * 9000
    total = sum(len(c) for c in chunk_channel_text(text))
    assert total >= len(text)


def test_empty_returns_single_chunk():
    # Empty/whitespace → one chunk (the original) so the caller still posts and
    # the endpoint surfaces "text is empty" rather than a silent no-op.
    assert chunk_channel_text("") == [""]
    assert chunk_channel_text("   ") == ["   "]


def test_non_str_input_coerced():
    assert chunk_channel_text(12345) == ["12345"]
