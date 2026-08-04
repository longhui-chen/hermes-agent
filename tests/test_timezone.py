"""
Tests for timezone support (hermes_time module + integration points).

Covers:
  - Valid timezone applies correctly
  - Invalid timezone falls back safely (no crash, warning logged)
  - execute_code child env receives TZ
  - Cron uses timezone-aware now()
  - Backward compatibility with naive timestamps
"""

import os
import logging
import sys
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import hermes_time


def _reset_hermes_time_cache():
    """Reset the hermes_time module cache."""
    hermes_time.reset_cache()


# =========================================================================
# hermes_time.now() — core helper
# =========================================================================

class TestHermesTimeNow:
    """Test the timezone-aware now() helper."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_valid_timezone_applies(self):
        """With a valid IANA timezone, now() returns time in that zone."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # IST is UTC+5:30
        offset = result.utcoffset()
        assert offset == timedelta(hours=5, minutes=30)

    def test_utc_timezone(self):
        """UTC timezone works."""
        os.environ["HERMES_TIMEZONE"] = "UTC"
        result = hermes_time.now()
        assert result.utcoffset() == timedelta(0)

    def test_us_eastern(self):
        """US/Eastern timezone works (DST-aware zone)."""
        os.environ["HERMES_TIMEZONE"] = "America/New_York"
        result = hermes_time.now()
        assert result.tzinfo is not None
        # Offset is -5h or -4h depending on DST
        offset_hours = result.utcoffset().total_seconds() / 3600
        assert offset_hours in {-5, -4}






class TestGetTimezone:
    """Test get_timezone()."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_returns_zoneinfo_for_valid(self):
        os.environ["HERMES_TIMEZONE"] = "Europe/London"
        tz = hermes_time.get_timezone()
        assert isinstance(tz, ZoneInfo)
        assert str(tz) == "Europe/London"


    def _isolate_os_sources(self, tmp_path, monkeypatch):
        """Decouple from the host OS tz so 'None' means 'nothing resolved',
        not 'CI host happens to have no tz'. OS-tz live read is covered by
        TestOSTimezoneLiveRead."""
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent2"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        monkeypatch.setattr(hermes_time, "_read_timedatectl", lambda: "")
        hermes_time.reset_cache()

    def test_returns_none_for_empty(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        self._isolate_os_sources(tmp_path, monkeypatch)
        tz = hermes_time.get_timezone()
        assert tz is None

    def test_returns_none_for_invalid(self, tmp_path, monkeypatch):
        os.environ["HERMES_TIMEZONE"] = "Not/A/Timezone"
        self._isolate_os_sources(tmp_path, monkeypatch)
        tz = hermes_time.get_timezone()
        assert tz is None



# =========================================================================
# execute_code child env — TZ injection
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="UDS not available on Windows")
class TestCodeExecutionTZ:
    """Verify TZ env var is passed to sandboxed child process via real execute_code."""

    @pytest.fixture(autouse=True)
    def _import_execute_code(self, monkeypatch):
        """Lazy-import execute_code to avoid pulling in firecrawl at collection time."""
        # Force local backend — other tests in the same xdist worker may leak
        # TERMINAL_ENV=modal/docker which causes modal.exception.AuthError.
        monkeypatch.setenv("TERMINAL_ENV", "local")
        try:
            from tools.code_execution_tool import execute_code
            self._execute_code = execute_code
        except ImportError:
            pytest.skip("tools.code_execution_tool not importable (missing deps)")

    def teardown_method(self):
        os.environ.pop("HERMES_TIMEZONE", None)

    def _mock_handle(self, function_name, function_args, task_id=None, user_task=None):
        import json as _json
        return _json.dumps({"error": f"unexpected tool call: {function_name}"})

    def test_tz_injected_when_configured(self):
        """When an IANA tz resolves, child process sees TZ env var.

        Patches hermes_time.get_timezone_name so this is decoupled from the
        CI host /etc/timezone. The local execution path (env_type == "local",
        the default here) is what sets child TZ. Verified alongside
        leak-prevention in one subprocess call so we don't pay the subprocess
        startup cost twice (each execute_code spawns a real subprocess ~3s).
        """
        import json as _json
        probe = (
            'import os; '
            'print("TZ=" + os.environ.get("TZ", "NOT_SET")); '
            'print("HERMES_TIMEZONE=" + os.environ.get("HERMES_TIMEZONE", "NOT_SET"))'
        )
        with patch("hermes_time.get_timezone_name", return_value="Asia/Kolkata"), \
             patch("model_tools.handle_function_call", side_effect=self._mock_handle):
            result = _json.loads(self._execute_code(
                code=probe, task_id="tz-combined-test", enabled_tools=[],
            ))
        assert result["status"] == "success"
        assert "TZ=Asia/Kolkata" in result["output"]
        assert "HERMES_TIMEZONE=NOT_SET" in result["output"], (
            "HERMES_TIMEZONE should not leak into child env (only TZ)"
        )

    def test_tz_not_injected_when_no_iana(self):
        """No resolvable IANA tz → child has no TZ (patch decouples from host /etc/timezone)."""
        import json as _json
        with patch("hermes_time.get_timezone_name", return_value=None), \
             patch("model_tools.handle_function_call", side_effect=self._mock_handle):
            result = _json.loads(self._execute_code(
                code='import os; print(os.environ.get("TZ", "NOT_SET"))',
                task_id="tz-test-empty",
                enabled_tools=[],
            ))
        assert result["status"] == "success"
        assert "NOT_SET" in result["output"]


# =========================================================================
# Cron timezone-aware scheduling
# =========================================================================

class TestCronTimezone:
    """Verify cron paths use timezone-aware now()."""

    def setup_method(self):
        _reset_hermes_time_cache()

    def teardown_method(self):
        _reset_hermes_time_cache()
        os.environ.pop("HERMES_TIMEZONE", None)

    def test_parse_schedule_duration_uses_tz_aware_now(self):
        """parse_schedule('30m') should produce a tz-aware run_at."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        from cron.jobs import parse_schedule
        result = parse_schedule("30m")
        run_at = datetime.fromisoformat(result["run_at"])
        # The stored timestamp should be tz-aware
        assert run_at.tzinfo is not None

    def test_compute_next_run_tz_aware(self):
        """compute_next_run returns tz-aware timestamps."""
        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        from cron.jobs import compute_next_run
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        next_dt = datetime.fromisoformat(result)
        assert next_dt.tzinfo is not None


    def test_ensure_aware_naive_preserves_absolute_time(self):
        """_ensure_aware must preserve the absolute instant for naive datetimes.

        Regression: the old code used replace(tzinfo=hermes_tz) which shifted
        absolute time when system-local tz != Hermes tz.  The fix interprets
        naive values as system-local wall time, then converts.
        """
        from cron.jobs import _ensure_aware

        os.environ["HERMES_TIMEZONE"] = "Asia/Kolkata"
        _reset_hermes_time_cache()

        # Create a naive datetime — will be interpreted as system-local time
        naive_dt = datetime(2026, 3, 11, 12, 0, 0)

        result = _ensure_aware(naive_dt)

        # The result should be in Kolkata tz
        assert result.tzinfo is not None

        # The UTC equivalent must match what we'd get by correctly interpreting
        # the naive dt as system-local time first, then converting
        system_tz = datetime.now().astimezone().tzinfo
        expected_utc = naive_dt.replace(tzinfo=system_tz).astimezone(timezone.utc)
        actual_utc = result.astimezone(timezone.utc)
        assert actual_utc == expected_utc, (
            f"Absolute time shifted: expected {expected_utc}, got {actual_utc}"
        )



    def test_get_due_jobs_naive_cross_timezone(self, tmp_path, monkeypatch):
        """Naive past timestamps must be detected as due even when Hermes tz
        is behind system local tz — the scenario that triggered #806."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        # Use a Hermes timezone far behind UTC so that the numeric wall time
        # of the naive timestamp exceeds _hermes_now's wall time — this would
        # have caused a false "not due" with the old replace(tzinfo=...) approach.
        os.environ["HERMES_TIMEZONE"] = "Pacific/Midway"  # UTC-11
        _reset_hermes_time_cache()

        from cron.jobs import create_job, load_jobs, save_jobs, get_due_jobs
        create_job(prompt="Cross-tz job", schedule="every 1h")
        jobs = load_jobs()

        # Force a naive past timestamp (system-local wall time, 10 min ago)
        naive_past = (datetime.now() - timedelta(seconds=30)).isoformat()
        jobs[0]["next_run_at"] = naive_past
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 1, (
            "Naive past timestamp should be due regardless of Hermes timezone"
        )

    def test_create_job_stores_tz_aware_timestamps(self, tmp_path, monkeypatch):
        """New jobs store timezone-aware created_at and next_run_at."""
        import cron.jobs as jobs_module
        monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
        monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
        monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")

        os.environ["HERMES_TIMEZONE"] = "US/Eastern"
        _reset_hermes_time_cache()

        from cron.jobs import create_job
        job = create_job(prompt="TZ test", schedule="every 2h")

        created = datetime.fromisoformat(job["created_at"])
        assert created.tzinfo is not None

        next_run = datetime.fromisoformat(job["next_run_at"])
        assert next_run.tzinfo is not None


# =========================================================================
# OS system-tz live read + fingerprint-gated cache (实时透传)
# =========================================================================

class TestOSTimezoneLiveRead:
    def setup_method(self):
        hermes_time.reset_cache()

    def teardown_method(self):
        hermes_time.reset_cache()
        os.environ.pop("HERMES_TIMEZONE", None)
        os.environ.pop("ZET_AGENT_ENABLED", None)

    def test_reads_etc_timezone_when_no_env(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        tzfile = tmp_path / "timezone"
        tzfile.write_text("Asia/Tokyo\n")
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent"))
        # config.yaml 不应有 timezone：指到空目录
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() == "Asia/Tokyo"
        assert hermes_time.now().utcoffset() == timedelta(hours=9)

    def test_localtime_symlink_wins_over_stale_etc_timezone(self, tmp_path, monkeypatch):
        """ZET-2185: /etc/timezone can go stale (e.g. an OTA writes 'Etc/UTC')
        while /etc/localtime still points at the real zone. The authoritative
        /etc/localtime symlink must win — otherwise the agent reasons in UTC and
        schedules cron 8h off. Reproduces the board28 (CEO device) state."""
        os.environ.pop("HERMES_TIMEZONE", None)
        os.environ.pop("ZET_AGENT_ENABLED", None)
        # Stale Debian file left at UTC.
        tzfile = tmp_path / "timezone"
        tzfile.write_text("Etc/UTC\n")
        # Authoritative symlink → Asia/Shanghai. Only the readlink() target
        # string is parsed (for the zoneinfo marker), so it need not resolve.
        localtime = tmp_path / "localtime"
        os.symlink("../usr/share/zoneinfo/Asia/Shanghai", localtime)
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(localtime))
        # Stub config read directly: _read_config_timezone goes through
        # read_raw_config() (the real config path, NOT get_config_path), so
        # patching get_config_path alone leaves the test coupled to the host's
        # actual hermes config. Force it empty so OS-tz resolution is exercised.
        monkeypatch.setattr(hermes_time, "_read_config_timezone", lambda: "")
        monkeypatch.setattr(hermes_time, "_read_timedatectl", lambda: "")
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() == "Asia/Shanghai"
        assert hermes_time.now().utcoffset() == timedelta(hours=8)

    def test_generic_config_timezone_wins_over_os_timezone(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        os.environ.pop("ZET_AGENT_ENABLED", None)
        tzfile = tmp_path / "timezone"
        tzfile.write_text("Asia/Tokyo\n")
        config = tmp_path / "config.yaml"
        config.write_text("timezone: Europe/London\n")
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: config)
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() == "Europe/London"

    def test_zettlab_device_os_timezone_wins_over_stale_config(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        os.environ["ZET_AGENT_ENABLED"] = "true"
        tzfile = tmp_path / "timezone"
        tzfile.write_text("Asia/Tokyo\n")
        config = tmp_path / "config.yaml"
        config.write_text("timezone: Europe/London\n")
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: config)
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() == "Asia/Tokyo"

    def test_picks_up_change_after_file_rewrite(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        tzfile = tmp_path / "timezone"
        tzfile.write_text("Asia/Shanghai\n")
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() == "Asia/Shanghai"
        # 重写文件（mtime/size 改变）→ 下一次调用必须重解析，无需 reset_cache
        tzfile.write_text("Asia/Tokyo\n")
        os.utime(tzfile, ns=(0, 0))  # 强制 mtime_ns 与上次不同
        assert hermes_time.get_timezone_name() == "Asia/Tokyo"

    def test_file_appearance_triggers_resolve(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        tzfile = tmp_path / "timezone"  # 起初不存在
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tzfile))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        monkeypatch.setattr(hermes_time, "_read_timedatectl", lambda: "")
        hermes_time.reset_cache()
        assert hermes_time.get_timezone_name() is None
        tzfile.write_text("Europe/Paris\n")  # 从无到有
        assert hermes_time.get_timezone_name() == "Europe/Paris"

    def test_timedatectl_fallback_change_triggers_resolve(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        current = {"tz": "Asia/Shanghai"}
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent2"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        monkeypatch.setattr(hermes_time, "_read_timedatectl", lambda: current["tz"])
        hermes_time.reset_cache()

        assert hermes_time.get_timezone_name() == "Asia/Shanghai"
        current["tz"] = "Asia/Tokyo"
        assert hermes_time.get_timezone_name() == "Asia/Tokyo"

    def test_get_timezone_name_returns_none_for_abbreviation_fallback(self, tmp_path, monkeypatch):
        os.environ.pop("HERMES_TIMEZONE", None)
        monkeypatch.setattr(hermes_time, "ETC_TIMEZONE", str(tmp_path / "nonexistent"))
        monkeypatch.setattr(hermes_time, "ETC_LOCALTIME", str(tmp_path / "nonexistent2"))
        monkeypatch.setattr(hermes_time, "get_config_path", lambda: tmp_path / "no_config.yaml")
        monkeypatch.setattr(hermes_time, "_read_timedatectl", lambda: "")
        hermes_time.reset_cache()
        # 没有任何 IANA 源 → name 为 None，但 now() 仍 tz-aware（server-local 兜底）
        assert hermes_time.get_timezone_name() is None
        assert hermes_time.now().tzinfo is not None
