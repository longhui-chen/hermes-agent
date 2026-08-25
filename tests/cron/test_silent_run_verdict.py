"""Regression: silent/skipped cron runs must not push a card into the chat.

TB-20260817-007. Two skip branches wrote a doc the old markdown-sniffing guard
could not recognize (wake-gate: no ``**Status:** silent`` line; no-script-output:
empty doc), so they leaked a ``cron-summary`` card. That bumps
``chat_sessions.updated_at`` in local-server, which both re-floats the
conversation and marks it unread — the two are one write
(``IsRead = read_at_ms >= updated_at``).

The fix keys off the scheduler's own verdict instead of the saved text.
"""

from __future__ import annotations

import pytest

from cron.scheduler import SILENT_MARKER, _is_cron_silence_response


JOB_ID = "silent-verdict-job"


@pytest.fixture
def card_layer(monkeypatch):
    """The Zettlab card layer with its per-run caches isolated."""
    import gateway.platforms.zet_agent_cron as z

    monkeypatch.setattr(z, "_LATEST_OUTPUT", {}, raising=False)
    monkeypatch.setattr(z, "_LATEST_SILENT", {}, raising=False)
    return z


def _publish(card_layer, job_id, doc, final_response, success=True):
    """Mirror run_one_job's ordering: record the verdict, then save the doc."""
    card_layer._LATEST_SILENT[job_id] = success and _is_cron_silence_response(final_response)
    card_layer._LATEST_OUTPUT[job_id] = doc


# The exact docs each skip branch hands back, paired with its final_response.
WAKE_GATE_DOC = (
    "# Cron Job: poll\n\n"
    "**Job ID:** j1\n"
    "**Run Time:** 2026-08-17 09:06:16\n\n"
    "Script gate returned `wakeAgent=false` — agent skipped.\n"
)
NO_AGENT_GATE_DOC = (
    "# Cron Job: poll\n\n"
    "**Job ID:** j1\n"
    "**Mode:** no_agent (script)\n"
    "**Status:** silent (wakeAgent=false)\n"
)
NO_AGENT_EMPTY_DOC = (
    "# Cron Job: poll\n\n"
    "**Job ID:** j1\n"
    "**Mode:** no_agent (script)\n"
    "**Status:** silent (empty output)\n"
)
AGENT_SILENT_DOC = "# Cron Job: poll\n\n## Response\n\n[SILENT]\n"


@pytest.mark.parametrize(
    "label,doc",
    [
        ("agent-path wake-gate skip", WAKE_GATE_DOC),
        ("no_agent wakeAgent=false", NO_AGENT_GATE_DOC),
        ("no_agent empty stdout", NO_AGENT_EMPTY_DOC),
        ("agent replied [SILENT]", AGENT_SILENT_DOC),
        ("script produced no output", ""),
    ],
)
def test_every_skip_branch_suppresses_the_card(card_layer, label, doc):
    _publish(card_layer, JOB_ID, doc, SILENT_MARKER)
    assert card_layer._is_silent_run(JOB_ID) is True, f"{label} would push a card"


def test_real_output_still_pushes_a_card(card_layer):
    """The guard must not swallow runs that produced something to show."""
    doc = "# Cron Job: poll\n\n## Response\n\n会议纪要已生成：3 个待办事项\n"
    _publish(card_layer, JOB_ID, doc, "会议纪要已生成：3 个待办事项")
    assert card_layer._is_silent_run(JOB_ID) is False


def test_failed_run_still_pushes_a_card(card_layer):
    """A failure delivers even though its body is not user-authored output."""
    doc = "# Cron Job: poll\n\n## Error\n\nscript exited 1\n"
    _publish(card_layer, JOB_ID, doc, "script exited 1", success=False)
    assert card_layer._is_silent_run(JOB_ID) is False


def test_verdict_does_not_leak_into_the_next_run(card_layer):
    """mark_job_run drains per-run state; a stale True would mute real output."""
    _publish(card_layer, JOB_ID, WAKE_GATE_DOC, SILENT_MARKER)
    assert card_layer._is_silent_run(JOB_ID) is True

    card_layer._LATEST_OUTPUT.pop(JOB_ID, None)
    card_layer._LATEST_SILENT.pop(JOB_ID, None)

    card_layer._LATEST_OUTPUT[JOB_ID] = "# Cron Job: poll\n\n## Response\n\n真实结果\n"
    assert card_layer._is_silent_run(JOB_ID) is False


def test_falls_back_to_doc_sniffing_without_a_verdict(card_layer):
    """No recorded verdict (older scheduler / partial patch) keeps legacy behavior."""
    card_layer._LATEST_OUTPUT[JOB_ID] = NO_AGENT_GATE_DOC
    assert card_layer._is_silent_run(JOB_ID) is True
