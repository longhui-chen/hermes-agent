"""Tests for gateway/platforms/zet_agent_goals.py — the zet_agent goal-loop
driver (5th host driver for hermes_cli.goals).

Covers: sidecar/index round-trip, projection mapping, action dispatch
(create/pause/resume/clear), the post-turn evaluation flow (continue / done /
paused verdicts → advance report), user-interrupt pause semantics, and the
compaction sidecar migration. The judge is always mocked — no LLM calls.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME (mirrors tests/hermes_cli/test_goals.py)."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # DEFAULT_DB_PATH 在 hermes_state 模块导入时固化（get_hermes_home() 的
    # import-time 求值），仅改 env 不会让 SessionDB() 落到本测试的 home ——
    # 不 patch 的话所有测试共享第一次导入时的 state.db，互相污染。
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


class FakeAdapter:
    """Minimal stand-in for ZetAgentAdapter: just the attributes the driver touches."""

    def __init__(self):
        self._background_tasks = set()
        self._session_run_lock = threading.Lock()
        self._active_session_agents = {}

    def _check_auth(self, request):
        return None


@pytest.fixture
def driver(hermes_home):
    from gateway.platforms.zet_agent_goals import ZetGoalDriver

    return ZetGoalDriver(FakeAdapter())


@pytest.fixture
def reports(driver, monkeypatch):
    """Capture advance reports instead of POSTing (fail-open path untested here)."""
    captured = []

    def fake_report(session_id, proj, *, continuation=None):
        captured.append({"session_id": session_id, "proj": dict(proj), "continuation": continuation})

    monkeypatch.setattr(driver, "report", fake_report)
    # report_in_thread 直接同步执行，测试不依赖线程时序。
    monkeypatch.setattr(driver, "report_in_thread", fake_report)
    return captured


SID = "zettlab:u1:agent-a:s1"


def _create(driver, sid=SID, text="整理下载目录 直到没有散落文件", goal_id="g_test", max_rounds=8):
    return driver._apply_action_sync(sid, "create", {
        "goal_id": goal_id,
        "text": text,
        "max_rounds": max_rounds,
        "app_session_id": sid,
    })


class TestSidecarAndIndex:
    def test_create_persists_sidecar_and_index(self, driver):
        proj = _create(driver)
        assert proj["goal_id"] == "g_test"
        assert proj["state"] == "running"
        assert proj["max_rounds"] == 8
        side = driver._load_sidecar(SID)
        assert side["goal_id"] == "g_test"
        assert side["app_session_id"] == SID
        assert SID in driver._index()

    def test_clear_removes_from_index(self, driver):
        _create(driver)
        proj = driver._apply_action_sync(SID, "clear", {})
        assert proj["state"] == "cleared"
        assert SID not in driver._index()

    def test_migrate_sidecar_follows_compaction(self, driver):
        _create(driver)
        new_sid = "zettlab:u1:agent-a:s1--rotated"
        driver._migrate_sidecar(SID, new_sid)
        assert driver._load_sidecar(new_sid)["goal_id"] == "g_test"
        assert new_sid in driver._index()
        assert SID not in driver._index()
        # 迁移后 wire session id 仍是稳定的 app session id。
        assert driver._wire_session_id(new_sid) == SID


class TestProjection:
    def test_states_map_to_wire_values(self, driver):
        from hermes_cli.goals import GoalManager

        _create(driver)
        assert driver.projection(SID)["state"] == "running"

        GoalManager(SID).pause("stop it")
        assert driver.projection(SID)["state"] == "paused"

        GoalManager(SID).resume()
        assert driver.projection(SID)["state"] == "running"

        GoalManager(SID).mark_done("all good")
        proj = driver.projection(SID)
        assert proj["state"] == "done"
        assert proj["summary"] == "all good"

    def test_no_goal_projects_cleared(self, driver):
        proj = driver.projection("zettlab:u1:agent-a:never-had-goal")
        assert proj["state"] == "cleared"


class TestResumeRoundContinuity:
    def test_user_resume_keeps_round_counter(self, driver, reports):
        """第 3 轮暂停后点继续，必须从第 4 轮接着数 —— 上游 resume() 默认
        重置 turns_used 会让 App 轮次显示跳回第 1 轮。"""
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        mgr = GoalManager(SID)
        st = mgr.state
        st.turns_used = 3
        save_goal(SID, st)
        driver._apply_action_sync(SID, "pause", {"reason": "user"})

        proj = driver._apply_action_sync(SID, "resume", {})

        assert GoalManager(SID).state.turns_used == 3, "非预算暂停的 resume 不得重置轮数"
        # resume 重踢的续轮上报应是「下一轮」= 4。
        assert proj["round"] == 4
        assert reports and reports[-1]["proj"]["round"] == 4
        assert reports[-1]["continuation"]

    def test_soft_pause_counts_finished_continuation_round(self, driver, reports):
        """第 2 轮跑着时按暂停（软暂停不打断本轮）：本轮跑完必须计轮，
        resume 后从第 3 轮继续 —— 不计的话轮次号原地重复。"""
        from gateway.platforms.zet_agent_goals import CONTINUATION_MARKER
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        mgr = GoalManager(SID)
        st = mgr.state
        st.turns_used = 1  # 第 1 轮已计，第 2 轮在跑
        save_goal(SID, st)
        driver._apply_action_sync(SID, "pause", {"reason": "user"})

        # 第 2 轮（continuation 驱动）跑完，post-turn hook 落在 paused 上：
        driver._after_turn_sync(SID, CONTINUATION_MARKER + " your standing goal]\nGoal: x", "第二段内容")

        assert GoalManager(SID).state.turns_used == 2, "软暂停期间跑完的 continuation 轮必须计数"
        assert reports and reports[-1]["proj"]["round"] == 2, "暂停投影应同步到第 2 轮"

        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["round"] == 3, "继续后应从第 3 轮开始"

    def test_soft_pause_does_not_count_user_interjection(self, driver, reports):
        """暂停期间的用户插话（无 continuation marker）不计轮。"""
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        mgr = GoalManager(SID)
        st = mgr.state
        st.turns_used = 2
        save_goal(SID, st)
        driver._apply_action_sync(SID, "pause", {"reason": "user"})

        driver._after_turn_sync(SID, "顺便帮我看看天气", "今天晴")

        assert GoalManager(SID).state.turns_used == 2

    def test_budget_resume_resets_with_monotonic_round(self, driver, reports):
        """预算耗尽的暂停 resume 时必须重置预算（否则立刻再触发预算暂停），
        但 App 侧轮次要靠 sidecar 偏移保持单调 —— 第 5 轮耗尽后继续应显示第 6 轮。"""
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver, max_rounds=5)
        mgr = GoalManager(SID)
        st = mgr.state
        st.turns_used = 5
        st.status = "paused"
        st.paused_reason = "turn budget exhausted (5/5)"
        save_goal(SID, st)

        proj = driver._apply_action_sync(SID, "resume", {})

        assert GoalManager(SID).state.turns_used == 0, "预算暂停的 resume 必须重置预算"
        assert driver._load_sidecar(SID)["rounds_offset"] == 5
        assert proj["round"] == 6, "轮次显示必须单调：5 轮偏移 + 新预算第 1 轮"


class TestAfterTurn:
    def test_continue_verdict_reports_continuation(self, driver, reports):
        _create(driver)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "还有散落文件", False, None)):
            driver._after_turn_sync(SID, "整理下载目录", "本轮移动了 3 个文件")
        assert len(reports) == 1
        r = reports[0]
        assert r["proj"]["state"] == "running"
        assert r["proj"]["round"] == 2  # turns_used=1 + 下一轮
        assert r["continuation"] and "[Continuing toward" in r["continuation"]
        assert r["session_id"] == SID

    def test_done_verdict_reports_done_and_drops_index(self, driver, reports):
        _create(driver)
        with patch("hermes_cli.goals.judge_goal", return_value=("done", "全部归位", False, None)):
            driver._after_turn_sync(SID, "user msg", "最终产出")
        assert reports[-1]["proj"]["state"] == "done"
        assert reports[-1]["continuation"] is None
        assert SID not in driver._index()

    def test_budget_exhausted_reports_paused(self, driver, reports):
        _create(driver, max_rounds=1)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "还没完", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出")
        assert reports[-1]["proj"]["state"] == "paused"
        assert reports[-1]["continuation"] is None

    def test_no_goal_is_noop(self, driver, reports):
        driver._after_turn_sync("zettlab:u1:agent-a:no-goal", "hi", "hello")
        assert reports == []

    def test_continuation_marker_marks_not_user_initiated(self, driver, reports):
        _create(driver)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "again", False, None)) as jg:
            driver._after_turn_sync(SID, "[Continuing toward your standing goal]\nGoal: x", "产出")
        assert jg.called
        assert len(reports) == 1

    def test_compaction_rotation_migrates_before_evaluate(self, driver, reports):
        from hermes_cli.goals import migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "go on", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出", effective_session_id=new_sid)
        assert len(reports) == 1
        # 驱动内部按迁移后的 sid 继续循环（fixture 捕获的是 report 入参）……
        assert reports[0]["session_id"] == new_sid
        assert driver._load_sidecar(new_sid)["goal_id"] == "g_test"
        # ……而真实 report() 会经 _wire_session_id 换回 app 级稳定 id 上报。
        assert driver._wire_session_id(new_sid) == SID


def _wait_until(cond, timeout=3.0):
    import time as _t

    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if cond():
            return True
        _t.sleep(0.02)
    return cond()


class TestInterruptAndInteractions:
    def test_user_interrupt_pauses_goal(self, driver, reports):
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver.on_user_interrupt(SID)
        # pause 在 context 保持的后台线程里落库（不阻塞 interrupt HTTP 响应）。
        assert _wait_until(lambda: GoalManager(SID).state.status == "paused")
        assert _wait_until(lambda: bool(reports) and reports[-1]["proj"]["state"] == "paused")
        # 暂停后 post-turn hook 不再续轮（evaluate 返回 inactive）。
        n = len(reports)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "x", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出")
        assert len(reports) == n, "paused goal 不得自动爬起续轮"

    def test_cancel_mark_beats_racing_evaluate(self, driver, reports):
        """停止落在 judge 评估期间：evaluate 自己的 save 会把 active 写回去，
        cancel 标记是最后的裁决 —— 绝不能上报 continuation。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        # 模拟"评估还没开始/正在进行时用户按了停止"：只打标记，不跑后台 pause
        # 线程（绕过 _spawn 的时序不确定性，聚焦标记裁决本身）。
        driver._mark_user_cancel(SID)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "还没完", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出")
        assert GoalManager(SID).state.status == "paused", "cancel 标记必须压过 continue 判定"
        assert reports[-1]["proj"]["state"] == "paused"
        assert all(r["continuation"] is None for r in reports), "不得下发续轮 continuation"

    def test_interrupt_without_goal_is_noop(self, driver, reports):
        driver.on_user_interrupt("zettlab:u1:agent-a:no-goal")
        import time as _t

        _t.sleep(0.1)  # 后台 pause 线程跑完（no-goal 路径无写入无上报）
        assert reports == []

    def test_interaction_pending_projects_waiting(self, driver, reports):
        _create(driver)
        driver.on_interaction_pending(SID)
        assert reports[-1]["proj"]["state"] == "waiting"
        driver.on_interaction_resolved(SID)
        assert reports[-1]["proj"]["state"] == "running"


class TestCancelMarkMigration:
    def test_migrate_sidecar_moves_cancel_mark(self, driver):
        _create(driver)
        driver._mark_user_cancel(SID)
        new_sid = SID + "--c2"
        driver._migrate_sidecar(SID, new_sid)
        assert driver._consume_user_cancel(new_sid), "cancel 标记必须随轮转迁移"
        assert not driver._consume_user_cancel(SID), "旧 id 的标记应已被搬走"

    def test_stop_after_compaction_rotation_still_pauses(self, driver, reports):
        """压缩轮转后用户按停止：标记落在旧 App sid（local-server 只认它），
        post-turn hook 迁移 sidecar 时必须连带迁移标记，否则 continue 判定
        会压过用户的停止（codex P1）。"""
        from hermes_cli.goals import GoalManager, migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        # 停止落在旧 id：goal 行已迁走，_pause_after_interrupt(old) 扑空，
        # 只剩这个标记承载用户意图。
        driver._mark_user_cancel(SID)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "还没完", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出", effective_session_id=new_sid)
        assert GoalManager(new_sid).state.status == "paused"
        assert all(r["continuation"] is None for r in reports), "停止之后不得下发续轮"


class TestInteractionPersistence:
    def test_pending_flag_persists_and_reconcile_parks(self, driver, reports):
        """approval 等待中 gateway 挂掉：重启 reconcile 不得自驱续轮（HR#3），
        必须 park 成 paused 留给用户显式恢复。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver.on_interaction_pending(SID)
        assert driver._interaction_flag_set(SID), "等待态必须落盘才能撑过重启"
        reports.clear()

        driver._reconcile_sync()
        assert GoalManager(SID).state.status == "paused"
        assert len(reports) == 1
        assert reports[0]["proj"]["state"] == "paused"
        assert reports[0]["continuation"] is None, "reconcile 不得替用户跳过确认"
        assert not driver._interaction_flag_set(SID)

    def test_flag_cleared_after_clean_turn(self, driver, reports):
        """approval 在轮内被处理（同意/拒绝/超时）后轮次正常结束：flag 必须
        随之清掉，循环照常推进。"""
        _create(driver)
        driver.on_interaction_pending(SID)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "继续", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出")
        assert not driver._interaction_flag_set(SID)
        assert reports[-1]["proj"]["state"] == "running"
        assert reports[-1]["continuation"], "已处理完确认的轮次照常续轮"

    def test_auto_resume_with_pending_interaction_parks(self, driver, reports):
        """LS 的 error 重踢（resume）抢在 reconcile 之前落到 active + flag 的
        goal 上：park 而不是续轮。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver.on_interaction_pending(SID)
        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["state"] == "paused"
        assert GoalManager(SID).state.status == "paused"
        assert not driver._interaction_flag_set(SID)

    def test_user_resume_clears_stale_flag(self, driver, reports):
        """等待确认期间用户按停止（paused + flag 残留）后手动继续：正常恢复，
        flag 视为放弃那次确认被清掉。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver.on_interaction_pending(SID)
        GoalManager(SID).pause("user stopped the running turn")
        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["state"] in ("running", "waiting")
        assert GoalManager(SID).state.status == "active"
        assert not driver._interaction_flag_set(SID)


class TestTerminalResumeRejected:
    def test_resume_on_cleared_goal_stays_cleared(self, driver, reports):
        """上游 resume() 不校验状态会把 cleared 复活成 active（codex P1）——
        LS 的重试/过期 resume 不得让已终结的 goal 重新自驱。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver._apply_action_sync(SID, "clear", {})
        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["state"] == "cleared"
        st = GoalManager(SID).state
        assert st is None or st.status == "cleared"
        assert all(r.get("continuation") is None for r in reports), "终态 resume 不得下发续轮"

    def test_resume_on_done_goal_stays_done(self, driver, reports):
        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).mark_done("done")
        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["state"] == "done"
        assert GoalManager(SID).state.status == "done"


class TestStaleCancelMark:
    def test_create_discards_stale_cancel_mark(self, driver, reports):
        """无 goal 会话上按过的停止（mark TTL 180s）不得误暂停之后新建的
        goal（codex P1）：create 时丢弃残留标记。"""
        from hermes_cli.goals import GoalManager

        driver._mark_user_cancel(SID)
        _create(driver)
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "继续", False, None)):
            driver._after_turn_sync(SID, "user msg", "产出")
        assert GoalManager(SID).state.status == "active", "旧 stop 不得暂停新 goal"
        assert reports[-1]["continuation"], "第一轮照常续轮"

    def test_resume_discards_stale_cancel_mark(self, driver, reports):
        """stop 中断的轮次不跑 post-turn hook，mark 不被消费；用户显式恢复
        即宣告该 stop 作废 —— 否则恢复后第一轮结束又被旧标记暂停回去。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver._mark_user_cancel(SID)
        GoalManager(SID).pause("user stopped the running turn")
        driver._apply_action_sync(SID, "resume", {})
        assert GoalManager(SID).state.status == "active"
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "继续", False, None)):
            driver._after_turn_sync(SID, "[Continuing toward your standing goal]\nGoal: x", "产出")
        assert GoalManager(SID).state.status == "active", "作废的 stop 不得再次暂停"


class TestCompactionTimeMigration:
    def test_note_compaction_rotation_migrates_index_immediately(self, driver):
        """压缩当下即时迁移（codex P1）：goal 行已被 migrate_goal_to_session
        迁走（旧行 cleared），若 index 还留在旧 sid，压缩后 turn 结束前挂掉
        的 gateway 重启 reconcile 只会删旧索引，新 sid 的 active goal 永失。"""
        from hermes_cli.goals import GoalManager, migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        driver.note_compaction_rotation(SID, new_sid)

        assert new_sid in driver._index()
        assert SID not in driver._index()
        assert driver._load_sidecar(new_sid)["goal_id"] == "g_test"
        # 模拟压缩后立即崩溃重启：reconcile 必须能沿新 index 找到 active goal。
        assert GoalManager(new_sid).state.status == "active"

    def test_double_migration_is_idempotent_and_preserves_new_side(self, driver):
        """即时迁移后 post-turn hook 还会再调一次 _migrate_sidecar：旧行已
        tombstone，重复迁移不得用旧快照回盖新 sid 上的后续写入。"""
        _create(driver)
        new_sid = SID + "--c2"
        driver.note_compaction_rotation(SID, new_sid)
        # 新 sid 上发生了后续写入（如 rounds_offset）。
        side = driver._load_sidecar(new_sid)
        side["rounds_offset"] = 5
        driver._save_sidecar(new_sid, side)

        driver._migrate_sidecar(SID, new_sid)  # post-turn 兜底重复迁移
        assert driver._load_sidecar(new_sid)["rounds_offset"] == 5
        assert driver._load_sidecar(new_sid)["goal_id"] == "g_test"


class TestRotationAwareInteractions:
    def test_interaction_pending_follows_rotation(self, driver, reports):
        """approval 回调捕获的是 _create_agent 时的旧 sid；本轮已压缩轮转时
        等待标记必须写到新 sid 的 sidecar（codex P1）——写旧 sid 的话
        reconcile 只扫新 index，重启后看不到等待态照样自驱。"""
        import types

        from hermes_cli.goals import migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        driver.note_compaction_rotation(SID, new_sid)
        # 活跃 agent 仍按请求时的旧 sid 注册，但 agent.session_id 已轮转。
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=new_sid)]

        driver.on_interaction_pending(SID)
        assert driver._interaction_flag_set(new_sid), "等待标记必须落在轮转后的 sidecar"
        assert not driver._interaction_flag_set(SID)
        assert reports[-1]["proj"]["state"] == "waiting"

        driver.on_interaction_resolved(SID)
        assert not driver._interaction_flag_set(new_sid), "resolved 也要按轮转后的 sid 清"

    def test_reconcile_skips_rotated_active_turn(self, driver, reports):
        """压缩即时迁移把 index 提前切到新 sid，而在途 turn 仍按旧 App sid
        注册（codex P1）：reconcile 只查新 sid 会误判空闲、并发重踢。"""
        import types

        from hermes_cli.goals import migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        driver.note_compaction_rotation(SID, new_sid)
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=new_sid)]

        driver._reconcile_sync()
        assert all(r.get("continuation") is None for r in reports), "在途轮（旧 sid 注册）不得被并发重踢"


class TestBarrierEdgeCases:
    def test_barrier_already_cleared_kicks_immediately(self, driver, reports, monkeypatch):
        """WAIT verdict 设完 barrier 后被等的 pid 秒退：排定时器时 is_waiting()
        已清掉 barrier —— 不能静默 return 卡死，必须走 wakeup 同款续跑
        （codex P1）。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        mgr = GoalManager(SID)
        st = mgr.state
        st.turns_used = 1
        # 用已过期的时间 barrier 模拟「设置后立刻满足」：is_waiting() 对过期
        # deadline 返回 False 并清 barrier。
        st.waiting_until = time.time() - 1.0
        st.waiting_reason = "waiting for build"
        from hermes_cli.goals import save_goal

        save_goal(SID, st)

        driver._schedule_barrier_wakeup(SID)
        assert reports, "barrier 已满足必须立即下发续轮"
        assert reports[-1]["proj"]["state"] == "running"
        assert reports[-1]["continuation"] and "[Continuing toward" in reports[-1]["continuation"]

    def test_barrier_kick_yields_to_active_turn(self, driver, reports):
        """barrier 清除时用户 turn 在途：不得并发自驱（codex P1），让位给
        该 turn 的 post-turn 评估。"""
        import types

        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        st = GoalManager(SID).state
        st.turns_used = 1
        st.waiting_until = time.time() - 1.0
        save_goal(SID, st)
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=SID)]

        driver._schedule_barrier_wakeup(SID)
        assert all(r.get("continuation") is None for r in reports), "在途 turn 存在时不得下发续轮"


class TestResumeIdempotency:
    def test_resume_retry_with_active_turn_skips_continuation(self, driver, reports):
        """resume 重试撞上已在途的续轮（上一条 resume 已起轮）：不得再发
        continuation 并发起第二轮（codex P1）。"""
        import types

        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).pause("user paused")
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=SID)]

        proj = driver._apply_action_sync(SID, "resume", {})
        assert proj["state"] in ("running", "waiting")
        assert all(r.get("continuation") is None for r in reports), "在途 turn 存在时 resume 不下发续轮"

    def test_resume_on_idle_session_still_kicks(self, driver, reports):
        """会话空闲时的 resume（LS error 重踢的正常场景）照常下发续轮。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).pause("user paused")
        driver._apply_action_sync(SID, "resume", {})
        assert reports and reports[-1]["continuation"], "空闲会话的 resume 必须重踢续轮"


class TestScopedKeys:
    def test_cancel_marks_isolated_per_profile_home(self, driver, monkeypatch):
        """mux 下 driver 是单例、profile 只切 runtime scope：同名 session_id
        的 cancel mark / barrier timer 不得跨 profile 互踩（codex P1）。"""
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        driver._mark_user_cancel(SID)
        token = set_hermes_home_override("/tmp/other-profile-home")
        try:
            assert not driver._consume_user_cancel(SID), "别的 profile 不得消费本 profile 的 mark"
        finally:
            reset_hermes_home_override(token)
        assert driver._consume_user_cancel(SID), "回到原 scope 后 mark 仍在"


class TestTerminalPauseRejected:
    def test_pause_on_done_goal_stays_done(self, driver, reports):
        """与 resume 对称（codex P1）：过期 pause 不得把终态行改成 paused，
        否则后续合法 resume 会复活已终结的 goal。"""
        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).mark_done("done")
        proj = driver._apply_action_sync(SID, "pause", {"reason": "stale"})
        assert proj["state"] == "done"
        assert GoalManager(SID).state.status == "done"

    def test_pause_on_cleared_goal_stays_cleared(self, driver, reports):
        from hermes_cli.goals import GoalManager

        _create(driver)
        driver._apply_action_sync(SID, "clear", {})
        proj = driver._apply_action_sync(SID, "pause", {"reason": "stale"})
        assert proj["state"] == "cleared"
        st = GoalManager(SID).state
        assert st is None or st.status == "cleared"


class TestBarrierTimerLifecycle:
    def _arm_timer(self, driver):
        from hermes_cli.goals import GoalManager, save_goal

        st = GoalManager(SID).state
        st.waiting_until = time.time() + 3600.0
        save_goal(SID, st)
        driver._schedule_barrier_wakeup(SID)
        assert driver._barrier_timers, "前置：timer 已排上"

    def test_create_cancels_stale_barrier_timer(self, driver, reports):
        """旧 goal 在 WAIT 时被 create 覆盖：残留 timer 触发会加载新 goal
        （无 barrier）并下发 continuation，与新 goal 并发起轮（codex P1）。"""
        _create(driver)
        self._arm_timer(driver)
        _create(driver, goal_id="g_new", text="新目标 直到完成")
        assert not driver._barrier_timers, "create 覆盖必须取消旧 barrier timer"

    def test_disconnect_cancels_all_timers(self, driver, reports):
        """adapter teardown：daemon Timer 不在 _background_tasks，不清会在
        断开后触发 wakeup 与新 adapter 并发自驱（codex P1）。"""
        _create(driver)
        self._arm_timer(driver)
        driver.cancel_all_barrier_timers()
        assert not driver._barrier_timers


class TestProjectionReadOnly:
    def test_projection_does_not_clear_satisfied_barrier(self, driver, reports):
        """GET/status 的 projection 无锁运行：is_waiting() 的 lazy auto-clear
        会把旧快照写回 DB、与并发 pause/clear 竞态（codex P1）——投影必须
        只读。barrier 已满足时投影显示 running，但 barrier 字段留给持锁的
        推进路径去清。"""
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        st = GoalManager(SID).state
        st.waiting_until = time.time() - 1.0  # 已满足的时间 barrier
        save_goal(SID, st)

        proj = driver.projection(SID)
        assert proj["state"] == "running", "满足的 barrier 不再算 waiting"
        st2 = GoalManager(SID).state
        assert st2.waiting_until, "projection 不得顺手清 barrier（写副作用）"

    def test_projection_shows_waiting_while_barrier_holds(self, driver, reports):
        from hermes_cli.goals import GoalManager, save_goal

        _create(driver)
        st = GoalManager(SID).state
        st.waiting_until = time.time() + 3600.0
        st.waiting_reason = "waiting for build"
        save_goal(SID, st)

        proj = driver.projection(SID)
        assert proj["state"] == "waiting"
        assert proj["summary"] == "waiting for build"


class TestMigrationCrashRecovery:
    def test_reconcile_follows_migration_pointer(self, driver, reports):
        """迁移半途崩溃（pointer 已写、new 未入 index）：reconcile 必须沿
        migrated_to 恢复新 sid 的 index/sidecar（app_session_id 跟过去，
        上报不漂移）而不是只删旧索引（codex P1）。"""
        from hermes_cli.goals import GoalManager, migrate_goal_to_session

        _create(driver)
        new_sid = SID + "--c2"
        migrate_goal_to_session(SID, new_sid, reason="compression")
        # 模拟崩溃点：只完成了「old sidecar 打 forward pointer」这一步。
        side = driver._load_sidecar(SID)
        driver._save_sidecar(SID, {**side, "migrated_to": new_sid})
        assert SID in driver._index() and new_sid not in driver._index()

        driver._reconcile_sync()

        assert new_sid in driver._index(), "沿指针恢复新 sid 索引"
        assert SID not in driver._index(), "旧索引照常摘除"
        restored = driver._load_sidecar(new_sid)
        assert restored.get("app_session_id") == SID, "app_session_id 必须恢复，上报不得漂移"
        assert GoalManager(new_sid).state.status == "active"
        # 恢复后的 goal 被照常重踢（active 无在途 turn）。
        assert reports and reports[-1]["continuation"]
        assert reports[-1]["session_id"] == new_sid


class TestActiveTurnProfileIsolation:
    def test_other_profile_turn_does_not_block_goal(self, driver, reports):
        """mux 下 active 表是 adapter 级裸 sid：另一 profile 的同名会话在跑
        不得让本 profile 的 goal 误判「已在途」而饿死续轮（codex P1）。"""
        import types

        _create(driver)
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=SID)]
        driver.adapter._active_session_homes = {SID: "/some/other/profile-home"}
        assert not driver._session_turn_active(SID), "别的 profile 的 turn 不算本 goal 在途"

        driver.adapter._active_session_homes = {SID: str(driver_home_of(driver))}
        assert driver._session_turn_active(SID), "本 profile 的 turn 照常算在途"

    def test_unstamped_registration_stays_permissive(self, driver):
        """未打 home 戳的注册（legacy/单 profile）保持宽松命中，行为不变。"""
        import types

        _create(driver)
        driver.adapter._active_session_agents[SID] = [types.SimpleNamespace(session_id=SID)]
        assert driver._session_turn_active(SID)


def driver_home_of(driver):
    from hermes_constants import get_hermes_home

    return get_hermes_home()


class TestJudgeBackgroundProcesses:
    def test_evaluate_passes_background_snapshot(self, driver, reports):
        """judge 的 WAIT 判定依赖后台进程快照（CI/build/watch）——必须像其它
        宿主驱动一样透传 gather_background_processes()（codex P1）。"""
        procs = [{"pid": 4242, "command": "npm run build", "running": True}]
        _create(driver)
        with patch("hermes_cli.goals.gather_background_processes", return_value=procs), \
             patch("hermes_cli.goals.judge_goal", return_value=("continue", "build 还在跑", False, None)) as jg:
            driver._after_turn_sync(SID, "user msg", "产出")
        assert jg.called
        assert jg.call_args.kwargs.get("background_processes") == procs


class TestMultiplexReconcile:
    def test_non_mux_runs_single_pass(self, driver, monkeypatch):
        calls = []
        monkeypatch.setattr(driver, "_reconcile_sync", lambda: calls.append("default"))
        monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
        driver._reconcile_all_scopes()
        assert calls == ["default"]

    def test_mux_reconciles_each_profile_home_once(self, driver, monkeypatch, tmp_path):
        """multiplex 下每个 profile home 的 goal index 各自独立，且 report 依赖
        profile scope 才能解析 ZET_GOAL_ADVANCE_URL —— reconcile 必须逐 profile
        进 scope 跑（codex P1）；default/main 同 home 去重。"""
        import sys
        import types
        from contextlib import contextmanager
        from pathlib import Path

        home_a = tmp_path / "prof-a"
        home_b = tmp_path / "prof-b"
        home_a.mkdir()
        home_b.mkdir()

        entered = []

        @contextmanager
        def fake_scope(home):
            entered.append(str(Path(home)))
            yield

        fake_run = types.ModuleType("gateway.run")
        fake_run._profile_runtime_scope = fake_scope
        monkeypatch.setitem(sys.modules, "gateway.run", fake_run)
        monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
        monkeypatch.setattr(
            driver.adapter,
            "_multiplex_profile_homes",
            lambda: {"default": home_a, "main": home_a, "beta": home_b},
            raising=False,
        )
        passes = []
        monkeypatch.setattr(driver, "_reconcile_sync", lambda: passes.append(entered[-1]))

        driver._reconcile_all_scopes()
        assert sorted(passes) == sorted([str(home_a), str(home_b)]), "同 home 去重、每 profile 各跑一遍"


class TestReconcile:
    def test_reconcile_rekicks_active_goal(self, driver, reports):
        _create(driver)
        driver._reconcile_sync()
        assert len(reports) == 1
        assert reports[0]["proj"]["state"] == "running"
        assert reports[0]["continuation"] and "[Continuing toward" in reports[0]["continuation"]

    def test_reconcile_skips_session_with_active_turn(self, driver, reports):
        _create(driver)
        driver.adapter._active_session_agents[SID] = [object()]
        driver._reconcile_sync()
        assert reports == [], "已有活跃 turn 的会话不得重复 kick"

    def test_reconcile_prunes_done_goal(self, driver, reports):
        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).mark_done("done")
        driver._reconcile_sync()
        assert SID not in driver._index()
        assert reports == []

    def test_reconcile_reports_paused_without_kick(self, driver, reports):
        from hermes_cli.goals import GoalManager

        _create(driver)
        GoalManager(SID).pause("user paused")
        driver._reconcile_sync()
        assert len(reports) == 1
        assert reports[0]["proj"]["state"] == "paused"
        assert reports[0]["continuation"] is None
