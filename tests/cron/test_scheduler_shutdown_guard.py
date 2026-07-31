"""Regression coverage for #58720 / #55924 — cron scheduling races
interpreter finalization.

When the gateway tears down (SIGTERM from ``hermes update`` /
``hermes gateway stop`` / systemd restart, or an OOM-kill), a cron tick can
still fire. Once the Python interpreter is finalizing, ``concurrent.futures``
refuses new work with ``RuntimeError: cannot schedule new futures after
interpreter shutdown`` and asyncio's default executor is gone. The cron
delivery + dispatch paths used to hit that unguarded, crashing the tick and
spraying a traceback into ``errors.log`` on every restart-race.

The fix adds ``_interpreter_shutting_down()`` and guards the scheduling
sites so they skip gracefully with a warning instead of raising.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestInterpreterShuttingDownHelper:
    def test_true_when_finalizing(self):
        from cron.scheduler import _interpreter_shutting_down

        with patch("sys.is_finalizing", return_value=True):
            assert _interpreter_shutting_down() is True

    def test_false_when_not_finalizing_and_no_exc(self):
        from cron.scheduler import _interpreter_shutting_down

        with patch("sys.is_finalizing", return_value=False):
            assert _interpreter_shutting_down() is False

    def test_matches_shutdown_error_text_as_fallback(self):
        """The concurrent.futures module-global flag can be set a hair before
        ``sys.is_finalizing()`` flips — matching the error text catches that
        race so a shutdown RuntimeError isn't misread as a real failure."""
        from cron.scheduler import _interpreter_shutting_down

        exc = RuntimeError("cannot schedule new futures after interpreter shutdown")
        with patch("sys.is_finalizing", return_value=False):
            assert _interpreter_shutting_down(exc) is True

    def test_unrelated_error_is_not_shutdown(self):
        from cron.scheduler import _interpreter_shutting_down

        exc = RuntimeError("some other problem")
        with patch("sys.is_finalizing", return_value=False):
            assert _interpreter_shutting_down(exc) is False


class TestStandaloneDeliverySkipsDuringShutdown:
    def _telegram_cfg(self):
        from gateway.config import Platform

        pconfig = MagicMock()
        pconfig.enabled = True
        mock_cfg = MagicMock()
        mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
        return mock_cfg

    def test_standalone_path_skips_without_scheduling(self):
        """With the interpreter finalizing, the standalone delivery path must
        skip BEFORE attempting to schedule the send — no ``_send_to_platform``
        call, a graceful warning-level skip, and an error string returned
        (not a raised exception)."""
        from cron.scheduler import _deliver_result

        job = {
            "id": "gov-job",
            "name": "model-governor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=self._telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("sys.is_finalizing", return_value=True):
            result = _deliver_result(job, "daily report body")

        send_mock.assert_not_called()
        assert result is not None
        assert "shutting down" in result

    def test_normal_delivery_still_works_when_not_finalizing(self):
        """Guard must not regress the happy path: a normal (non-finalizing)
        run still delivers via the standalone send."""
        from cron.scheduler import _deliver_result

        job = {
            "id": "gov-job",
            "name": "model-governor",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        send_mock = AsyncMock(return_value={"success": True})
        with patch("gateway.config.load_gateway_config", return_value=self._telegram_cfg()), \
             patch("tools.send_message_tool._send_to_platform", new=send_mock), \
             patch("sys.is_finalizing", return_value=False):
            result = _deliver_result(job, "daily report body")

        send_mock.assert_called_once()
        assert result is None


class TestSourceGuardrail:
    @pytest.fixture
    def source(self) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[2] / "cron" / "scheduler.py"
        ).read_text(encoding="utf-8")

    def test_helper_defined(self, source):
        assert "def _interpreter_shutting_down(" in source
        assert "#58720" in source

    def test_helper_guards_dispatch_submit(self, source):
        """The tick dispatch (``_submit_with_guard``) must consult the guard so
        a tick that races teardown skips instead of crashing."""
        idx_submit = source.find("def _submit_with_guard(")
        assert idx_submit >= 0
        tail = source[idx_submit:idx_submit + 1600]
        assert "_interpreter_shutting_down(" in tail

    def test_helper_guards_standalone_delivery(self, source):
        """The standalone delivery path must consult the guard before
        scheduling ``asyncio.run`` / a fresh pool."""
        idx = source.find("Standalone path: run the async send")
        assert idx >= 0
        # The guard appears shortly before the standalone send comment.
        window = source[max(0, idx - 600):idx]
        assert "_interpreter_shutting_down()" in window


class TestTickSkipsBeforeMutatingDuringShutdown:
    """A tick racing teardown must bail before any next_run_at write, so the
    skipped occurrence stays due for the next healthy tick's catch-up."""

    @pytest.fixture
    def storage(self, tmp_path, monkeypatch):
        import cron.executions as executions_module
        import cron.jobs as jobs_module
        import hermes_time

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")
        monkeypatch.setattr(
            executions_module, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
        )
        (tmp_path / "cron").mkdir(parents=True, exist_ok=True)
        hermes_time.reset_cache()
        yield tmp_path
        hermes_time.reset_cache()

    def _make_due_daily_job(self):
        from datetime import datetime, timedelta, timezone

        from cron.jobs import create_job, load_jobs, save_jobs

        create_job(prompt="draw daily", schedule="0 9 * * *", name="daily")
        jobs = load_jobs()
        due_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        jobs[0]["next_run_at"] = due_at
        save_jobs(jobs)
        return jobs[0]["id"], due_at

    def test_shutdown_tick_leaves_due_jobs_untouched(self, storage):
        from datetime import datetime, timezone

        import cron.scheduler as scheduler
        from cron.executions import list_executions
        from cron.jobs import get_due_jobs, load_jobs

        job_id, due_at = self._make_due_daily_job()
        assert [j["id"] for j in get_due_jobs()] == [job_id]

        with patch("sys.is_finalizing", return_value=True):
            dispatched = scheduler.tick(verbose=False)

        assert dispatched == 0
        assert list_executions(job_id=job_id) == []

        job = next(j for j in load_jobs() if j["id"] == job_id)
        assert job.get("last_run_at") in (None, "")
        assert job["next_run_at"] == due_at
        assert datetime.fromisoformat(job["next_run_at"]) <= datetime.now(timezone.utc)
        assert [j["id"] for j in get_due_jobs()] == [job_id]
