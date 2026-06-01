"""Shared text chunking for Zettlab channel sends.

This module is intentionally side-effect free. In particular, tools can import
it without importing gateway.platforms.zet_agent_cron, whose module import
auto-installs cron monkeypatches.
"""

from typing import List


# Must stay strictly under local-server's per-message rune cap
# (internal_channels.maxChannelSendTextLen = 4000); the headroom absorbs the
# "(N/M)" part indicators and code-fence reopening that truncate_message adds.
CHANNEL_SEND_MAX_RUNES = 3900


def chunk_channel_text(content) -> List[str]:
    """Split content into <=CHANNEL_SEND_MAX_RUNES-rune chunks for channel send.

    Reuses the same smart splitter as native Hermes platform delivery
    (BasePlatformAdapter.truncate_message — preserves code-block boundaries,
    adds (N/M) indicators). Falls back to a hard rune split if that import is
    unavailable. Empty/whitespace chunks are dropped. Always returns >=1 chunk
    for non-empty input so the caller still posts something.
    """
    text = content if isinstance(content, str) else str(content)
    chunks = None
    try:
        from gateway.platforms.base import BasePlatformAdapter
        chunks = BasePlatformAdapter.truncate_message(text, CHANNEL_SEND_MAX_RUNES)
    except Exception:
        runes = list(text)
        chunks = [
            "".join(runes[i:i + CHANNEL_SEND_MAX_RUNES])
            for i in range(0, len(runes), CHANNEL_SEND_MAX_RUNES)
        ]
    chunks = [c for c in (chunks or []) if c and c.strip()]
    if not chunks:
        # text was empty/whitespace — let the single send surface the
        # endpoint's "text is empty" rejection rather than silently no-op.
        return [text]
    return chunks
