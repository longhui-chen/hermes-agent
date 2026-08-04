"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: data-layer semantics
# ---------------------------------------------------------------------------


def test_create_job_no_agent_requires_script(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


def test_update_job_roundtrips_no_agent_flag(hermes_env):
    from cron.jobs import create_job, update_job, get_job

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local")

    update_job(job["id"], {"no_agent": False})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False

    update_job(job["id"], {"no_agent": True})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool: API-layer validation
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_no_agent_without_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "no_agent=True requires a script" in result.get("error", "")


# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


# ---------------------------------------------------------------------------
# scheduler.run_job: calendar reminder (notify-only) short-circuit
# ---------------------------------------------------------------------------
# Calendar reminders are written straight into jobs.json by
# zettlab-local-server (source="calendar", a one-shot "once" schedule whose
# `content` is the reminder text). They are NOT created via create_job, so
# these tests build the job dict directly — exactly the shape the device
# writes.


def _calendar_job(content="开会：项目评审 16:00", **extra):
    job = {
        "id": "cal_job_1",
        "name": "团队例会",
        "source": "calendar",
        "calendar_provider": "google_calendar",
        "calendar_connection_id": "conn-1",
        "calendar_series_id": "standup_series",
        "calendar_original_start": "2026-06-26T07:00:00Z",
        "content": content,
        "no_agent": True,
        "schedule": {"kind": "once", "run_at": "2026-06-26T07:00:00Z"},
        "deliver": "local",
    }
    job.update(extra)
    return job


def test_run_job_legacy_calendar_content_is_suppressed(hermes_env):
    """Legacy/broad calendar jobs cannot bypass the exact planner contract."""
    from cron.scheduler import run_job, SILENT_MARKER

    job = _calendar_job(content="开会：项目评审 16:00")
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER
    assert "开会：项目评审 16:00" not in doc


def test_run_job_calendar_no_script_required(hermes_env):
    """A calendar job carries no script; it must NOT trip the no_agent
    'requires a script' guard — the calendar branch precedes it."""
    from cron.scheduler import run_job

    job = _calendar_job()
    assert "script" not in job
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None  # NOT "no_agent=True but no script is set"


def test_run_job_calendar_empty_content_is_silent(hermes_env):
    """No content to deliver → SILENT_MARKER suppresses delivery."""
    from cron.scheduler import run_job, SILENT_MARKER

    job = _calendar_job(content="   ", name="")
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER


def test_run_job_calendar_never_invokes_aiagent(hermes_env):
    """Calendar reminders must NOT import/construct the AIAgent (no LLM spend)."""
    job = _calendar_job()
    with patch("run_agent.AIAgent") as ai_mock:
        from cron.scheduler import run_job

        run_job(job)
    ai_mock.assert_not_called()


def test_run_job_calendar_near_miss_is_always_silent(hermes_env):
    """A calendar marker near-miss is quarantined/silent, never generic execution."""
    from cron.scheduler import run_job, SILENT_MARKER

    job = _calendar_job(no_agent=False, schedule={"kind": "cron", "expr": "* * * * *"})
    success, _doc, final_response, error = run_job(job)
    assert success is True
    assert final_response == SILENT_MARKER
    assert error is None
    assert "开会：项目评审 16:00" not in final_response


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output
