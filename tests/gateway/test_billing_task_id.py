"""Unit tests for credit-ledger task-id mapping (gateway.session_context).

Interactive sessions pass through; cron sessions collapse to a stable per-job
id so every run aggregates into one ledger task card; non-NAS sessions map to ''
so X-Task-Id is never stamped on third-party-provider calls.
"""

from gateway.session_context import (
    billing_task_id,
    billing_task_id_for,
    set_current_session_id,
)


def test_interactive_session_passes_through():
    assert billing_task_id_for("zettlab:u1:agent-a:abc123") == "zettlab:u1:agent-a:abc123"


def test_cron_session_collapses_to_stable_per_job_id():
    # cron_<job>_<YYYYMMDD>_<HHMMSS> -> cron_<job>; every run of the job collapses
    # to the same id regardless of the per-run timestamp.
    assert billing_task_id_for("cron_4b2628798006_20260624_104233") == "cron_4b2628798006"
    assert billing_task_id_for("cron_4b2628798006_20260625_010101") == "cron_4b2628798006"


def test_non_nas_and_empty_sessions_are_not_attributed():
    assert billing_task_id_for("local-session-xyz") == ""
    assert billing_task_id_for("") == ""
    assert billing_task_id_for(None) == ""  # type: ignore[arg-type]


def test_billing_task_id_reads_current_session_context():
    set_current_session_id("cron_job123_20260624_104233")
    try:
        assert billing_task_id() == "cron_job123"
    finally:
        set_current_session_id("")
