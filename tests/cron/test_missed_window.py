"""设备停机错过窗口（失败场景）的回归测试。

覆盖 cron/jobs.py 两条行为：
  - ``_compute_grace_seconds``：宽限窗口 = 周期一半，clamp 在 120s(2min) ~ 7200s(2h)
  - ``get_due_jobs``：recurring 任务 next_run_at 在过去时
      · 超过 grace → fast-forward 跳到下一个未来时刻、不进 due（网关停机不积压补推）
      · 在 grace 内 → 进 due（补跑一次）
"""

import os
from datetime import datetime, timedelta, timezone

import pytest

import hermes_time
import cron.jobs as jobs_module
from cron.jobs import (
    _compute_grace_seconds,
    create_job,
    get_due_jobs,
    load_jobs,
    save_jobs,
)


def _reset_hermes_time_cache():
    hermes_time._cached_tz = None
    hermes_time._cached_tz_name = None
    hermes_time._cache_resolved = False


@pytest.fixture
def cron_storage(tmp_path, monkeypatch):
    """把 cron 存储重定向到 tmp，并把时区钉成 UTC 让 now() 确定。"""
    monkeypatch.setattr(jobs_module, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs_module, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_module, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setenv("HERMES_TIMEZONE", "UTC")
    _reset_hermes_time_cache()
    yield tmp_path
    _reset_hermes_time_cache()


class TestComputeGraceSeconds:
    """周期一半，clamp 在 120s ~ 7200s。"""

    def test_interval_within_clamp(self):
        # every 10min → 600//2 = 300s
        assert _compute_grace_seconds({"kind": "interval", "minutes": 10}) == 300

    def test_interval_below_min_clamped_to_120(self):
        # every 1min → 30s → clamp 到 MIN 120
        assert _compute_grace_seconds({"kind": "interval", "minutes": 1}) == 120

    def test_interval_above_max_clamped_to_7200(self):
        # every 600min(10h) → 18000s → clamp 到 MAX 7200(2h)
        assert _compute_grace_seconds({"kind": "interval", "minutes": 600}) == 7200

    @pytest.mark.skipif(not jobs_module.HAS_CRONITER, reason="croniter 未安装")
    def test_cron_daily_clamped_to_2h(self):
        # 每天周期 86400 → 43200 → clamp 7200
        assert _compute_grace_seconds({"kind": "cron", "expr": "0 9 * * *"}) == 7200

    @pytest.mark.skipif(not jobs_module.HAS_CRONITER, reason="croniter 未安装")
    def test_cron_every_5min_within_clamp(self):
        # 周期 300 → 150
        assert _compute_grace_seconds({"kind": "cron", "expr": "*/5 * * * *"}) == 150

    def test_unknown_or_missing_kind_falls_back_to_min(self):
        assert _compute_grace_seconds({"kind": "once"}) == 120
        assert _compute_grace_seconds({}) == 120


class TestGetDueJobsMissedWindow:
    """网关停机错过窗口：超过 grace → fast-forward 不补推；grace 内 → 补跑一次。

    用 ``every 1h`` → grace = 3600//2 = 1800s。
    """

    def test_stale_run_fast_forwarded_not_backlogged(self, cron_storage):
        create_job(prompt="x", schedule="every 1h")
        jobs = load_jobs()
        # next_run_at 在 3 小时前，远超 grace(1800s) —— 模拟网关停机错过
        jobs[0]["next_run_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=3)
        ).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()

        # 不积压补推：这条不进 due
        assert due == []
        # fast-forward：next_run_at 被推到未来
        new_next = datetime.fromisoformat(load_jobs()[0]["next_run_at"])
        assert new_next > datetime.now(timezone.utc)

    def test_recent_miss_within_grace_catches_up(self, cron_storage):
        create_job(prompt="x", schedule="every 1h")
        jobs = load_jobs()
        # next_run_at 刚过 60s，在 grace(1800s) 内 —— 应当补跑一次
        jobs[0]["next_run_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=60)
        ).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()

        assert len(due) == 1
