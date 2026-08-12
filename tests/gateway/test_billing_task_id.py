"""Unit tests for credit-ledger key mapping (gateway.session_context).

Two keys are derived from one session, and they must NOT be the same value:
``billing_usage_id`` is the ledger key (one chat turn / one cron run) while
``billing_conversation_id`` is ai-gateway's routing + prompt-cache key and stays
at conversation granularity. Non-NAS sessions map to '' in both, so neither is
ever stamped on a third-party-provider call.
"""

from urllib.parse import unquote

from gateway.session_context import (
    _VAR_MAP,
    billing_conversation_id,
    billing_conversation_id_for,
    billing_task_id,
    billing_task_id_for,
    billing_task_title_encoded,
    billing_usage_id,
    billing_usage_id_for,
    pop_billing_usage_id,
    pop_zettlab_turn_title,
    push_billing_usage_id,
    push_zettlab_turn_title,
    set_current_session_id,
    set_zettlab_turn_id,
    summarize_turn_title,
)


def test_interactive_session_passes_through():
    assert billing_task_id_for("zettlab:u1:agent-a:abc123") == "zettlab:u1:agent-a:abc123"


def test_cron_session_keeps_its_per_run_timestamp():
    # F1: cron_<job>_<YYYYMMDD>_<HHMMSS> is the ledger key as-is, so every run
    # of the job gets its own usage card instead of collapsing into one.
    assert (
        billing_task_id_for("cron_4b2628798006_20260624_104233")
        == "cron_4b2628798006_20260624_104233"
    )
    assert (
        billing_usage_id_for("cron_4b2628798006_20260625_010101")
        == "cron_4b2628798006_20260625_010101"
    )


def test_cron_conversation_id_collapses_to_stable_per_job_id():
    # The routing/cache key keeps its pre-F1 value: all runs of one job share an
    # upstream and a prompt-cache bucket.
    assert (
        billing_conversation_id_for("cron_4b2628798006_20260624_104233")
        == "cron_4b2628798006"
    )
    assert (
        billing_conversation_id_for("cron_4b2628798006_20260625_010101")
        == "cron_4b2628798006"
    )


def test_non_nas_and_empty_sessions_are_not_attributed():
    for session_id in ("local-session-xyz", "", None):
        assert billing_task_id_for(session_id) == ""  # type: ignore[arg-type]
        assert billing_usage_id_for(session_id) == ""  # type: ignore[arg-type]
        assert billing_conversation_id_for(session_id) == ""  # type: ignore[arg-type]


def test_billing_task_id_reads_current_session_context():
    set_current_session_id("cron_job123_20260624_104233")
    try:
        assert billing_task_id() == "cron_job123_20260624_104233"
        assert billing_conversation_id() == "cron_job123"
    finally:
        set_current_session_id("")


def test_usage_id_appends_the_turn_segment_for_chat_sessions():
    assert (
        billing_usage_id_for("zettlab:u1:agent-a:abc123", "turn-7")
        == "zettlab:u1:agent-a:abc123:tturn-7"
    )


def test_usage_id_falls_back_to_the_session_without_a_turn():
    # CLI / gateway platforms never bind a turn id: attribution stays exactly
    # where it was before per-turn keys existed.
    assert (
        billing_usage_id_for("zettlab:u1:agent-a:abc123", "")
        == "zettlab:u1:agent-a:abc123"
    )
    assert (
        billing_usage_id_for("zettlab:u1:agent-a:abc123", "   ")
        == "zettlab:u1:agent-a:abc123"
    )


def test_usage_id_digests_a_turn_id_with_separators_or_whitespace():
    # ':' delimits the key's segments and whitespace is illegal in a header
    # value. Such ids degrade to a short digest instead of dropping characters:
    # dropping would let two distinct turn ids collide on one ledger card.
    usage_id = billing_usage_id_for("zettlab:u1:agent-a:abc", "turn: 7\tb")
    assert usage_id.startswith("zettlab:u1:agent-a:abc:t")
    # The session part must stay recoverable: exactly one ':t' separator beyond
    # the session's own three colons.
    assert usage_id.count(":") == 4
    assert usage_id.isascii()
    assert not any(c.isspace() for c in usage_id)
    # Deterministic, and injective where the old strip-based scheme collided.
    assert usage_id == billing_usage_id_for("zettlab:u1:agent-a:abc", "turn: 7\tb")
    assert (
        billing_usage_id_for("zettlab:u1:agent-a:abc", "turn:1")
        != billing_usage_id_for("zettlab:u1:agent-a:abc", "turn1")
    )


def test_usage_id_digests_a_non_ascii_turn_id():
    # metadata.turn_id's charset is deliberately unrestricted upstream
    # (api_server._extract_turn_id), but X-Task-Id must stay a legal ASCII
    # header value — httpx raises UnicodeEncodeError otherwise, which would
    # fail every model call of the turn.
    usage_id = billing_usage_id_for("zettlab:u1:agent-a:abc", "轮次一🌀")
    assert usage_id.isascii()
    assert usage_id.startswith("zettlab:u1:agent-a:abc:t")
    segment = usage_id.rsplit(":t", 1)[1]
    assert len(segment) == 12
    assert all(c in "0123456789abcdef" for c in segment)


def test_usage_id_length_cap_is_exclusive_at_120():
    base = "zettlab:u1:agent-a:abc"
    room = 120 - len(base) - len(":t")
    at_cap = "x" * room
    # Exactly 120 chars composed: the verbatim segment survives.
    assert billing_usage_id_for(base, at_cap) == f"{base}:t{at_cap}"
    # One char past the cap: the segment degrades to the 12-hex digest.
    over_cap = "x" * (room + 1)
    hashed = billing_usage_id_for(base, over_cap)
    assert hashed != f"{base}:t{over_cap}"
    assert len(hashed) == len(base) + len(":t") + 12
    assert hashed == billing_usage_id_for(base, over_cap)


def test_usage_id_hashes_an_over_long_turn_segment():
    session_id = "zettlab:u1:agent-a:" + "s" * 40
    usage_id = billing_usage_id_for(session_id, "t" * 120)
    assert usage_id.startswith(session_id + ":t")
    assert len(usage_id) <= 120
    # Deterministic: the same turn id always yields the same short segment.
    assert usage_id == billing_usage_id_for(session_id, "t" * 120)
    assert usage_id != billing_usage_id_for(session_id, "u" * 120)


def test_usage_id_reads_the_current_turn_context():
    set_current_session_id("zettlab:u1:agent-a:abc")
    set_zettlab_turn_id("turn-42")
    try:
        assert billing_usage_id() == "zettlab:u1:agent-a:abc:tturn-42"
        # 🔴 The routing key must NOT pick up the turn segment.
        assert billing_conversation_id() == "zettlab:u1:agent-a:abc"
    finally:
        set_zettlab_turn_id("")
        set_current_session_id("")


def test_captured_usage_id_overrides_the_ambient_context():
    # Background workers (title generation) carry the turn key across a thread
    # boundary that ContextVars do not follow.
    set_current_session_id("zettlab:u1:agent-a:abc")
    token = push_billing_usage_id("zettlab:u1:agent-a:abc:tfirst-turn")
    try:
        assert billing_usage_id() == "zettlab:u1:agent-a:abc:tfirst-turn"
        assert billing_conversation_id() == "zettlab:u1:agent-a:abc"
    finally:
        pop_billing_usage_id(token)
        set_current_session_id("")
    assert billing_usage_id() == ""


def test_billing_task_title_encoded_percent_encodes_cron_job_name():
    # run_job sets HERMES_CRON_TASK_TITLE to the job name; HTTP headers are
    # ASCII-only so a CJK title must be percent-encoded (ai-api QueryUnescape-
    # decodes it once). Assert it's ASCII-safe and round-trips, without
    # hard-coding the UTF-8 byte sequence.
    _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("站立提醒 (每分钟)")
    try:
        enc = billing_task_title_encoded()
        assert enc.isascii() and " " not in enc
        assert unquote(enc) == "站立提醒 (每分钟)"
    finally:
        _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("")


def test_billing_task_title_encoded_empty_when_unset():
    _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("")
    assert billing_task_title_encoded() == ""


def test_billing_task_title_encoded_uses_the_bound_turn_title():
    _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("")
    token = push_zettlab_turn_title("帮我整理这周的照片")
    try:
        enc = billing_task_title_encoded()
        assert enc.isascii()
        assert unquote(enc) == "帮我整理这周的照片"
    finally:
        pop_zettlab_turn_title(token)
    assert billing_task_title_encoded() == ""


def test_cron_job_name_wins_over_a_turn_title():
    _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("站立提醒")
    token = push_zettlab_turn_title("不应出现")
    try:
        assert unquote(billing_task_title_encoded()) == "站立提醒"
    finally:
        pop_zettlab_turn_title(token)
        _VAR_MAP["HERMES_CRON_TASK_TITLE"].set("")


def test_summarize_turn_title_collapses_whitespace_and_truncates():
    assert summarize_turn_title("  帮我   整理\n照片  ") == "帮我 整理 照片"
    assert summarize_turn_title(None) == ""
    long_title = summarize_turn_title("x" * 200)
    assert long_title == "x" * 60
