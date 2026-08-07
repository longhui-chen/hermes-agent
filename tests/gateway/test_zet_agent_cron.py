from unittest.mock import MagicMock
import inspect
from contextlib import contextmanager

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.zet_agent import ZetAgentAdapter


@pytest.mark.asyncio
async def test_send_requires_explicit_cron_delivery_metadata():
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

    normal = await adapter.send("session-1", "hello")
    assert isinstance(normal, SendResult)
    assert normal.success is False
    assert "no proactive send path" in normal.error

    cron = await adapter.send(
        "session-1",
        "hello",
        metadata={"zet_agent_cron_delivery": True},
    )
    assert cron.success is True
    assert cron.message_id == "zet_agent:session-1"


def test_install_normalizes_legacy_zettlab_origin():
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zet_agent_cron

    zet_agent_cron.install()
    signature = inspect.signature(scheduler.mark_job_run)
    assert "scheduled_at" in signature.parameters
    assert "output_filename" in signature.parameters
    run_signature = inspect.signature(scheduler.run_job)
    assert "defer_agent_teardown" in run_signature.parameters

    job = {
        "origin": {
            "platform": "zettlab",
            "chat_id": "session-123",
            "chat_name": "App Chat",
        }
    }
    result = scheduler._resolve_origin(job)
    assert result == {
        "platform": "zet_agent",
        "chat_id": "session-123",
        "chat_name": "App Chat",
    }
    assert job["origin"]["platform"] == "zettlab"


def test_install_does_not_relabel_generic_api_server_context():
    """Zet's cron extension must not monkey-patch the parent API adapter.

    Zet overrides _create_agent, so the old parent patch never protected Zet
    and instead rewrote ordinary API-server turns as delivering zet_agent
    sessions.
    """
    from gateway.platforms.api_server import APIServerAdapter
    import gateway.platforms.zet_agent_cron as zet_agent_cron

    before = APIServerAdapter._create_agent
    zet_agent_cron.install()
    assert APIServerAdapter._create_agent is before
    assert not getattr(
        APIServerAdapter._create_agent,
        zet_agent_cron._PATCH_SENTINEL,
        False,
    )


def test_install_bypasses_scheduler_delivery_for_zet_agent(monkeypatch):
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zet_agent_cron

    zet_agent_cron.install()

    adapter = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    job = {
        "id": "job-1",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": "session-123"},
    }

    result = scheduler._deliver_result(
        job,
        "Cron output",
        adapters={},
        loop=loop,
    )

    assert result is None
    adapter.send.assert_not_called()


def test_ordinary_once_cron_keeps_completed_row_and_visible_cron_summary(
    tmp_path, monkeypatch,
):
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    import hermes_state
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    zc.install()

    session_id = "zettlab:userA:main:ordinary-cron"
    job_id = "ordinary-once-visible"
    job = {
        "id": job_id,
        "name": "普通一次性任务",
        "prompt": "生成普通报告",
        "schedule": {"kind": "once", "run_at": "2026-07-22T01:00:00Z"},
        "schedule_display": "2026-07-22 01:00",
        "repeat": {"times": 1, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": session_id},
        "timezone": "UTC",
    }
    cron_jobs.save_jobs([job])
    db = SessionDB()
    try:
        db.create_session(session_id, source="zet_agent", user_id="userA")
    finally:
        db.close()

    zc._LATEST_OUTPUT[job_id] = (
        "# Cron Job: 普通一次性任务\n\n## Response\n\n普通任务已完成，报告共 3 项。\n"
    )
    scheduler.mark_job_run(
        job_id,
        True,
        scheduled_at="2026-07-22T01:00:00Z",
        output_filename="run.md",
    )

    stored = cron_jobs.get_job_raw(job_id)
    assert stored["enabled"] is False
    assert stored["state"] == "completed"
    assert stored["repeat"]["completed"] == 1
    db = SessionDB()
    try:
        messages = db.get_messages(session_id)
    finally:
        db.close()
    assert len(messages) == 1
    assert messages[0]["llm_visible"] == 1
    assert "```cron-summary" in messages[0]["content"]
    assert "普通任务已完成，报告共 3 项。" in messages[0]["content"]


def test_detect_fake_success(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    def set_body(body):
        zc._LATEST_OUTPUT["job-x"] = f"## Response\n\n{body}"

    # 意图句 + 0 工具 → 降级（复现 ZET-1048 假成功）
    monkeypatch.setattr(zc, "_count_tool_activity", lambda _jid: 0)
    set_body("I'll fetch the weather data for both cities now.")
    assert zc._detect_fake_success("job-x") is not None
    # 中文意图句带开场白 + 0 工具 → 降级
    set_body("好的，我现在就去帮你查询天气。")
    assert zc._detect_fake_success("job-x") is not None
    # 真实数据（非意图句）→ 不降级
    set_body("Iceland: 5°C cloudy. Norway: 3°C rain.")
    assert zc._detect_fake_success("job-x") is None
    # "let me know" 在句中（非开头意图）→ 不降级
    set_body("Your weekly report is ready, let me know if you want edits.")
    assert zc._detect_fake_success("job-x") is None
    # 意图句但工具跑过了 → 不降级
    monkeypatch.setattr(zc, "_count_tool_activity", lambda _jid: 2)
    set_body("I'll fetch the weather data for both cities now.")
    assert zc._detect_fake_success("job-x") is None
    # ZET-1782：工具跑过但最终正文是残缺 code/markdown 片段，也不能当成功推送。
    set_body("`.app")
    assert zc._detect_fake_success("job-x") is not None
    # 同类：只吐出 helper script 文件名，不是用户可读结果。
    set_body("sync_v2.py")
    assert zc._detect_fake_success("job-x") is not None
    # 正常短文本不误伤。
    set_body("已同步完成。")
    assert zc._detect_fake_success("job-x") is None
    # session 读不到（None）→ fail-open，不降级
    monkeypatch.setattr(zc, "_count_tool_activity", lambda _jid: None)
    set_body("I'll fetch the weather data for both cities now.")
    assert zc._detect_fake_success("job-x") is None

    zc._LATEST_OUTPUT.pop("job-x", None)


def test_notify_chat_append_includes_cron_kind(monkeypatch):
    import json as _json
    import urllib.request

    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:0/append")
    monkeypatch.setenv("ZET_AGENT_ID", "agent-1")

    captured = {}

    class _Resp:
        status = 200

        def read(self, *_a):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    zc._try_notify_chat_append("sess-1", 7, "body text")

    payload = _json.loads(captured["data"].decode("utf-8"))
    assert payload["kind"] == "cron_summary"
    assert payload["session_id"] == "sess-1"
    assert payload["msg_id"] == 7


def test_append_delivery_error_to_output(tmp_path):
    import re

    import gateway.platforms.zet_agent_cron as zc

    run_file = tmp_path / "2026-05-28_15-05-04.md"
    run_file.write_text(
        "# Cron Job: 日报\n\n**Mode:** no_agent (script)\n\n---\n\n"
        "日报已生成：今日新增 3 条热点。\n",
        encoding="utf-8",
    )
    zc._LATEST_OUTPUT_PATH["job-d"] = run_file

    # 执行成功但投递失败 → 追加 "## Delivery Error" 段
    zc._append_delivery_error_to_output(
        "job-d", "platform 'telegram' not configured/enabled"
    )
    content = run_file.read_text(encoding="utf-8")
    assert "## Delivery Error" in content
    assert "platform 'telegram' not configured/enabled" in content
    # 不能引入会被 App parser 误判成"执行失败"的标记
    assert "(FAILED)" not in content
    assert re.search(r"^##\s*Error\b", content, re.M) is None

    # 空原因 / 未知 job → no-op，不抛、不改文件
    before = run_file.read_text(encoding="utf-8")
    zc._append_delivery_error_to_output("job-d", "")
    zc._append_delivery_error_to_output("unknown-job", "boom")
    assert run_file.read_text(encoding="utf-8") == before

    zc._LATEST_OUTPUT_PATH.pop("job-d", None)


def test_append_run_error_to_output(tmp_path):
    import re

    import gateway.platforms.zet_agent_cron as zc

    # scheduler 已把 .md 当成功文档写盘（## Response + 开场白，无 (FAILED)/## Error）
    run_file = tmp_path / "2026-05-28_15-10-04.md"
    run_file.write_text(
        "# Cron Job: 天气推送\n\n**Run Time:** 2026-05-28 15:10:04\n\n## Response\n\n"
        "I'll fetch the weather data for both cities now.\n",
        encoding="utf-8",
    )
    zc._LATEST_OUTPUT_PATH["job-r"] = run_file

    # 假成功降级 → 补 "## Error"，让 parseCronRunStatus 的 ^## Error 分支命中 = failed
    zc._append_run_error_to_output("job-r", "agent executed no tools and produced no result")
    content = run_file.read_text(encoding="utf-8")
    assert re.search(r"^##\s*Error\b", content, re.M) is not None
    assert "agent executed no tools and produced no result" in content
    # 不动标题，不引入 (FAILED) 双重标记
    assert "(FAILED)" not in content

    # 空原因 / 未知 job → no-op
    before = run_file.read_text(encoding="utf-8")
    zc._append_run_error_to_output("job-r", "")
    zc._append_run_error_to_output("unknown-job", "boom")
    assert run_file.read_text(encoding="utf-8") == before

    zc._LATEST_OUTPUT_PATH.pop("job-r", None)


def test_handoff_session_when_origin_deleted(tmp_path, monkeypatch):
    """deliver=origin 但源对话已删 → 新建承接会话、打 origin_recreated、回写 job.origin。"""
    import hermes_state
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    from hermes_state import SessionDB

    AID = "agent-x"
    SID = f"zettlab:userA:{AID}:orig001"
    JID = "jobH"
    job = {
        "id": JID, "name": "喝水提醒", "prompt": "提醒喝水", "skills": [], "skill": None,
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "每天09:00"},
        "schedule_display": "每天09:00", "repeat": {"times": None, "completed": 0},
        "enabled": True, "state": "scheduled", "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": SID, "chat_name": "喝水对话"},
        "timezone": "UTC", "last_status": None, "last_error": None, "last_delivery_error": None,
    }
    cron_jobs.save_jobs([job])

    db = SessionDB()
    db.create_session(SID, source="chat", user_id="userA")
    db.close()
    SessionDB().delete_session(SID)  # 用户删除源对话
    assert SessionDB().get_session(SID) is None

    zc._LATEST_OUTPUT[JID] = "# Cron Job: 喝水提醒\n\n## Response\n\n该喝水啦！💧\n"
    try:
        ret = zc._try_persist_to_session(JID, True, None, None, job)
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    # 不再是投递失败
    assert ret is None
    # job.origin 被重指到一个新的同 user/agent 会话
    new_id = cron_jobs.get_job(JID)["origin"]["chat_id"]
    assert new_id != SID
    assert new_id.startswith(f"zettlab:userA:{AID}:")
    # 新会话真建出来了，老会话没被复活
    assert SessionDB().get_session(new_id) is not None
    assert SessionDB().get_session(SID) is None
    # 承接消息带 origin_recreated 标记（App 暂不渲染，仅数据标记）+ cron 正文
    msgs = SessionDB().get_messages(new_id)
    blob = " ".join((m.get("content") or "") for m in msgs if isinstance(m.get("content"), str))
    assert '"origin_recreated": true' in blob
    assert "该喝水啦" in blob


def test_persist_heals_profile_scoped_internal_origin_key(tmp_path, monkeypatch):
    """Internal multiplex queue keys must never become persisted chat ids."""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)

    session_id = "zettlab:userA:main:origin001"
    internal_key = f"{tmp_path}|{session_id}"
    job_id = "job-scoped-origin"
    job = {
        "id": job_id,
        "name": "喝水提醒",
        "prompt": "提醒喝水",
        "skills": [],
        "skill": None,
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": internal_key},
        "timezone": "UTC",
    }
    cron_jobs.save_jobs([job])
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="zet_agent", user_id="userA")
    db.close()

    zc._LATEST_OUTPUT[job_id] = (
        "# Cron Job: 喝水提醒\n\n## Response\n\n该喝水了，记得补充水分。\n"
    )
    try:
        assert zc._try_persist_to_session(job_id, True, None, None, job) is None
    finally:
        zc._LATEST_OUTPUT.pop(job_id, None)

    stored = cron_jobs.get_job(job_id)
    assert stored["origin"]["chat_id"] == session_id
    db = SessionDB(db_path=tmp_path / "state.db")
    messages = db.get_messages(session_id)
    db.close()
    assert any("该喝水了" in str(message.get("content")) for message in messages)


def test_scoped_origin_normalization_rejects_foreign_profile(tmp_path, monkeypatch):
    """Only this profile's internal key may be reduced to a public chat id."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_id = "zettlab:userA:main:origin001"

    assert zc._normalize_zet_agent_chat_id(f"{tmp_path}|{session_id}") == session_id
    foreign = f"{tmp_path.parent / 'foreign'}|{session_id}"
    assert zc._normalize_zet_agent_chat_id(foreign) == foreign
    malformed = f"{tmp_path}|not-a-public-session"
    assert zc._normalize_zet_agent_chat_id(malformed) == malformed


def test_calendar_reminder_session_is_created_without_handoff(tmp_path, monkeypatch):
    """calendar-reminders 是 APP/local-server 合成会话，不存在时应原地创建。"""
    import hermes_state
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    from hermes_state import SessionDB

    SID = "zettlab:userA:main:calendar-reminders"
    JID = "calJob"
    job = {
        "id": JID,
        "name": "喝水提醒",
        "source": "calendar",
        "content": "喝水提醒",
        "prompt": "喝水提醒",
        "skills": [],
        "skill": None,
        "schedule": {"kind": "once", "run_at": "2026-06-29T07:45:00Z"},
        "repeat": {"times": 1, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": SID, "chat_name": "Calendar Reminders"},
        "timezone": "UTC",
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
    }
    cron_jobs.save_jobs([job])
    assert SessionDB().get_session(SID) is None

    zc._LATEST_OUTPUT[JID] = "# Cron Job: 喝水提醒\n\n## Response\n\n喝水提醒\n"
    try:
        ret = zc._try_persist_to_session(JID, True, None, None, job)
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    assert ret is None
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID
    assert SessionDB().get_session(SID) is not None
    msgs = SessionDB().get_messages(SID)
    blob = " ".join((m.get("content") or "") for m in msgs if isinstance(m.get("content"), str))
    assert '"origin_recreated": true' not in blob
    assert '"source": "calendar"' in blob
    assert "喝水提醒" in blob


def test_persist_targets_call_time_profile_home_not_frozen_default(tmp_path, monkeypatch):
    """Multiplex e2e (real mark_job_run): the cron summary must land in the firing
    profile's state.db (what the App reads /history from), not the top-level DB
    that DEFAULT_DB_PATH is frozen to at import time — the misroute that left the
    per-profile DB with 0 cron summaries."""
    import sqlite3
    import hermes_state
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    top = tmp_path / "top"                       # gateway HERMES_HOME at import
    prof = tmp_path / "profiles" / "eae0707d"    # firing profile home (call-time)
    top.mkdir(parents=True)
    prof.mkdir(parents=True)

    # DEFAULT_DB_PATH frozen to the top-level home, but the live home
    # (get_hermes_home, under the profile cron scope) is the per-profile home.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", top / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(prof))
    monkeypatch.setattr(cron_jobs, "CRON_DIR", prof / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", prof / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", prof / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    zc.install()

    AID = "eae0707d"
    SID = f"zettlab:userA:{AID}:orig001"
    JID = "jobMux"
    job = {
        "id": JID, "name": "论文推荐", "prompt": "推荐论文", "skills": [], "skill": None,
        "schedule": {"kind": "cron", "expr": "30 21 * * *", "display": "每天21:30"},
        "schedule_display": "每天21:30", "repeat": {"times": None, "completed": 1},
        "enabled": True, "state": "scheduled", "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": SID},
        "timezone": "Asia/Shanghai", "last_status": None, "last_error": None,
        "last_delivery_error": None,
    }
    cron_jobs.save_jobs([job])

    # The App-visible origin conversation lives in the PER-PROFILE db.
    prof_db = SessionDB(db_path=prof / "state.db")
    prof_db.create_session(SID, source="zet_agent", user_id="userA")
    prof_db.close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n今日推荐：论文\n"
    try:
        scheduler.mark_job_run(JID, True, scheduled_at="2026-07-30T13:30:00Z", output_filename="run.md")
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    def _cron_summary_count(db_file):
        if not db_file.exists():
            return 0
        c = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            return c.execute(
                "select count(*) from messages where session_id=? and content like '%cron-summary%'",
                (SID,),
            ).fetchone()[0]
        finally:
            c.close()

    # Summary in the per-profile db (App reads it), NOT the frozen top-level db.
    assert _cron_summary_count(prof / "state.db") == 1
    assert _cron_summary_count(top / "state.db") == 0


def _cron_summary_rows(db_file, session_id):
    """Count cron-summary messages for one session in one state.db (read-only)."""
    import sqlite3

    if not db_file.exists():
        return 0
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        return conn.execute(
            "select count(*) from messages where session_id=? "
            "and content like '%cron-summary%'",
            (session_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def _mux_cron_job(job_id, session_id, name="论文推荐"):
    return {
        "id": job_id, "name": name, "prompt": "推荐论文", "skills": [], "skill": None,
        "schedule": {"kind": "cron", "expr": "30 21 * * *", "display": "每天21:30"},
        "schedule_display": "每天21:30", "repeat": {"times": None, "completed": 1},
        "enabled": True, "state": "scheduled", "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": session_id},
        "timezone": "Asia/Shanghai", "last_status": None, "last_error": None,
        "last_delivery_error": None,
    }


def _mux_env(monkeypatch, hermes_home, jobs_home, default_db):
    """Point the process at ``hermes_home`` with DEFAULT_DB_PATH frozen elsewhere."""
    import cron.jobs as cron_jobs
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", default_db)
    monkeypatch.setattr(cron_jobs, "CRON_DIR", jobs_home / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", jobs_home / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", jobs_home / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZET_AGENT_ID", raising=False)


def test_persist_lands_in_profile_db_when_home_override_unbound(tmp_path, monkeypatch):
    """S0 止血：cron 线程里 profile override 没绑上（get_hermes_home() = 根 home）时，
    摘要仍必须落 profiles/<agentID>/state.db —— 这正是线上 61% 记录跳不过去的成因。"""
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    zc.install()

    SID = f"zettlab:userA:{AID}:orig001"
    JID = "jobUnbound"
    cron_jobs.save_jobs([_mux_cron_job(JID, SID)])

    prof_db = SessionDB(db_path=prof / "state.db")
    prof_db.create_session(SID, source="zet_agent", user_id="userA")
    prof_db.close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n今日推荐：论文\n"
    try:
        scheduler.mark_job_run(JID, True, scheduled_at="2026-07-30T13:30:00Z",
                               output_filename="run.md")
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    assert _cron_summary_rows(prof / "state.db", SID) == 1
    assert _cron_summary_rows(root / "state.db", SID) == 0
    # 源会话没被当成「已删除」→ 不许派生承接会话
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID


def test_persist_keeps_legacy_root_store_session_in_place(tmp_path, monkeypatch):
    """存量 root-only 会话（板子上 101 个）不能被当成已删除：继续写根库、
    不派生承接会话。否则每次 cron 都给用户新建一个空对话。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "abf18a05"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:legacy01"
    JID = "jobLegacy"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])

    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session(SID, source="zet_agent", user_id="userA")
    root_db.close()
    # profile 库存在但没有这个会话
    other = SessionDB(db_path=prof / "state.db")
    other.create_session(f"zettlab:userA:{AID}:other01", source="zet_agent", user_id="userA")
    other.close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n旧会话续写\n"
    try:
        assert zc._try_persist_to_session(JID, True, None, None, job) is None
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    assert _cron_summary_rows(root / "state.db", SID) == 1
    assert _cron_summary_rows(prof / "state.db", SID) == 0
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID


def test_persist_prefers_profile_db_for_split_session(tmp_path, monkeypatch):
    """劈裂会话（两库都有，板子上 2 个）：新写入只进 profile 库，根库不再增长——
    S2a 小手术的前置条件就是这个「不再边合边裂」。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:9sKO_OC-HQSg"
    JID = "jobSplit"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])

    for db_file in (root / "state.db", prof / "state.db"):
        db = SessionDB(db_path=db_file)
        db.create_session(SID, source="zet_agent", user_id="userA")
        db.close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n劈裂会话\n"
    try:
        assert zc._try_persist_to_session(JID, True, None, None, job) is None
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    assert _cron_summary_rows(prof / "state.db", SID) == 1
    assert _cron_summary_rows(root / "state.db", SID) == 0


def test_persist_creates_missing_session_in_profile_db(tmp_path, monkeypatch):
    """会话两库都没有（calendar-reminders 合成会话）→ 建在 profile 库，不建在根库。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "main"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:calendar-reminders"
    JID = "jobCal"
    job = _mux_cron_job(JID, SID, name="喝水提醒")
    job["source"] = "calendar"
    cron_jobs.save_jobs([job])

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n该喝水啦\n"
    try:
        assert zc._try_persist_to_session(JID, True, None, None, job) is None
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    prof_db = SessionDB(db_path=prof / "state.db", read_only=True)
    try:
        assert prof_db.get_session(SID) is not None
    finally:
        prof_db.close()
    assert _cron_summary_rows(prof / "state.db", SID) == 1
    assert not (root / "state.db").exists() or _cron_summary_rows(root / "state.db", SID) == 0


def test_persist_follows_session_agent_id_not_process_home(tmp_path, monkeypatch):
    """multiplex 下 override 绑到了「别的 profile」时，摘要仍按会话自带的 agentID
    落库 —— 绝不允许跟着错的 home 跨 profile 写。

    job 必须来自它自己 profile 的 cron store：那是执行身份的服务端来源。
    job 住在 B 的 store 却把 origin 指向 A 是攻击形状，由
    test_cross_agent_origin_refused_by_job_store_profile_when_env_unbound 钉住拒绝。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    OTHER = "d490707e"
    prof = root / "profiles" / AID
    other_prof = root / "profiles" / OTHER
    prof.mkdir(parents=True)
    other_prof.mkdir(parents=True)
    # 进程 home 绑在 OTHER 的 profile 上（错的那个）；job 仍在自己 profile 的 store 里
    _mux_env(monkeypatch, other_prof, prof, root / "state.db")

    SID = f"zettlab:userA:{AID}:orig001"
    JID = "jobCross"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])

    prof_db = SessionDB(db_path=prof / "state.db")
    prof_db.create_session(SID, source="zet_agent", user_id="userA")
    prof_db.close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n跨 profile 保护\n"
    try:
        assert zc._try_persist_to_session(JID, True, None, None, job) is None
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    assert _cron_summary_rows(prof / "state.db", SID) == 1
    assert _cron_summary_rows(other_prof / "state.db", SID) == 0
    assert _cron_summary_rows(root / "state.db", SID) == 0


def test_profile_db_resolution_rejects_traversal_agent_id(tmp_path, monkeypatch):
    """origin.chat_id 是外部输入：agentID 段不是纯 profile 名就不许拼进路径。"""
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    (root / "profiles").mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    for bad in ("../../etc", "..", ".", "", "a/b", "x\x00y", "eae0707d\n"):
        assert zc._profile_state_db_candidates(bad) == [], bad
    assert zc._profile_state_db_candidates("eae0707d") == [
        (root / "profiles" / "eae0707d" / "state.db").resolve()
    ]
    # 整条解析也退回当前 home，不落到 profiles 之外
    assert zc._resolve_persist_db_path("zettlab:userA:../../etc:s1") == (root / "state.db", None)


def test_cron_session_readers_stay_on_default_db_path(tmp_path, monkeypatch):
    """fake-success / silent / 附件三条读路径必须跟 cron/scheduler.py 写 cron session
    的那个 bare SessionDB() 同库（DEFAULT_DB_PATH）；profile 化只这一侧会读空。"""
    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    prof = root / "profiles" / "eae0707d"
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, prof, prof, root / "state.db")

    db = zc._cron_session_db()
    try:
        assert db.db_path == hermes_state.DEFAULT_DB_PATH
    finally:
        db.close()


def test_persist_failure_surfaces_delivery_error_never_silent(tmp_path, monkeypatch):
    """Never-silent: if persisting the summary raises, the failure is folded into
    the job's last_delivery_error (App card shows it) — not swallowed."""
    import hermes_state
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    zc.install()

    SID = "zettlab:userA:main:origFail"
    JID = "jobFail"
    job = {
        "id": JID, "name": "会失败的任务", "prompt": "生成报告",
        "schedule": {"kind": "cron", "expr": "0 9 * * *", "display": "每天09:00"},
        "schedule_display": "每天09:00", "repeat": {"times": None, "completed": 0},
        "enabled": True, "state": "scheduled", "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": SID}, "timezone": "UTC",
    }
    cron_jobs.save_jobs([job])
    db = SessionDB()
    try:
        db.create_session(SID, source="zet_agent", user_id="userA")
    finally:
        db.close()

    # Persisting the summary blows up.
    def _boom(*a, **k):
        raise RuntimeError("db write blew up")
    monkeypatch.setattr(hermes_state.SessionDB, "append_message", _boom)

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n内容\n"
    try:
        scheduler.mark_job_run(JID, True, scheduled_at="2026-07-30T01:00:00Z", output_filename="run.md")
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    stored = cron_jobs.get_job(JID)
    assert stored is not None
    assert stored.get("last_delivery_error")   # failure surfaced, not silently dropped


def test_silent_run_skips_session_persist(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    job = {
        "id": "job-s",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": "sess-s"},
    }

    # 静默运行绝不能落卡 → SessionDB 一旦被构造就说明漏了守卫
    def _boom(*_a, **_k):
        raise AssertionError("SessionDB constructed for a silent run")

    monkeypatch.setattr("hermes_state.SessionDB", _boom)

    # no_agent 脚本无输出（窗口外提醒）→ 静默，跳过落卡
    zc._LATEST_OUTPUT["job-s"] = (
        "# Cron Job: 起身提醒\n\n**Job ID:** job-s\n"
        "**Mode:** no_agent (script)\n**Status:** silent (empty output)\n"
    )
    assert zc._is_silent_run("job-s") is True
    assert zc._try_persist_to_session("job-s", True, None, None, job) is None

    # wakeAgent=false 门控同样是静默
    zc._LATEST_OUTPUT["job-s"] = (
        "# Cron Job: 看门狗\n\n**Mode:** no_agent (script)\n"
        "**Status:** silent (wakeAgent=false)\n"
    )
    assert zc._is_silent_run("job-s") is True

    # agent 回复 [SILENT] → 静默，跳过落卡
    zc._LATEST_OUTPUT["job-s"] = "# Cron Job: x\n\n## Response\n\n[SILENT]\n"
    assert zc._is_silent_run("job-s") is True
    assert zc._try_persist_to_session("job-s", True, None, None, job) is None

    # 真实产出 → 非静默（正常落卡路径不受影响）
    zc._LATEST_OUTPUT["job-s"] = "# Cron Job: x\n\n## Response\n\n日报已生成。\n"
    assert zc._is_silent_run("job-s") is False

    # 行中提到静默协议、但末尾给出真实结果，必须投递。旧 substring
    # 判断会把这种实际模型输出误吞，造成日历已完成而会话无卡片。
    zc._LATEST_OUTPUT["job-s"] = (
        "# Cron Job: x\n\n## Response\n\n"
        "I considered [SILENT], but this run has a result.\n\n"
        "E2E unified reminder delivery executed\n"
    )
    assert zc._is_silent_run("job-s") is False

    # Status-like text inside the untrusted response is not producer metadata.
    zc._LATEST_OUTPUT["job-s"] = (
        "# Cron Job: x\n\n## Response\n\n"
        "**Status:** silent\n\n12 个付款任务执行失败\n"
    )
    assert zc._is_silent_run("job-s") is False

    # 无缓存输出 → 非静默（fail-open，不误吞真运行）
    assert zc._is_silent_run("never-seen-job") is False

    zc._LATEST_OUTPUT.pop("job-s", None)


def test_cron_summary_carries_next_run_at_and_timezone(tmp_path, monkeypatch):
    """App 据 next_run_at 把循环任务展示时间本地化、据 timezone 区分字面可信与否；
    两者都从 job 透传进 cron-summary 元数据，无值时省略。"""
    import json as _json

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    def _meta(content: str) -> dict:
        fence = content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0]
        return _json.loads(fence)

    job = {
        "id": "job-tz",
        "name": "喝水提醒",
        "schedule": {"kind": "cron", "expr": "30 2 * * *", "display": "30 2 * * *"},
        "next_run_at": "2026-05-29T02:30:00+00:00",
        "timezone": "Asia/Shanghai",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": "c1"},
    }
    meta = _meta(zc._build_typed_message_content(job, "job-tz", True, None, None))
    assert meta["next_run_at"] == "2026-05-29T02:30:00+00:00"
    assert meta["timezone"] == "Asia/Shanghai"

    # 旧 job：无 timezone / next_run_at → 字段省略，App 走原字面回退
    legacy = {
        "id": "job-legacy",
        "name": "每日新闻",
        "schedule": {"kind": "cron", "expr": "0 12 * * *", "display": "0 12 * * *"},
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": "c1"},
    }
    meta2 = _meta(zc._build_typed_message_content(legacy, "job-legacy", True, None, None))
    assert "next_run_at" not in meta2
    assert "timezone" not in meta2


def test_collect_produced_files_walks_current_output_run_and_filters_helpers(tmp_path, monkeypatch):
    """ZET-1793: cron attachments come from the current output run, not every
    tool-call path or every historical file in a reused output bucket."""
    import json as _json
    import os
    import time

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    from hermes_state import SessionDB

    agent_id = "d87dcb35-cf11-44d3-8ca3-8d9b01cfb090"
    bucket = tmp_path / "volume1" / "subvol" / "agents" / "data" / agent_id / "output" / "BMlfeWtIkAHx"
    bucket.mkdir(parents=True)
    current_report = bucket / "reddit_report_2026-06-28.html"
    current_report.write_text("<html>today</html>", encoding="utf-8")
    old_report = bucket / "reddit_report_2026-06-27.html"
    old_report.write_text("<html>old</html>", encoding="utf-8")
    tmp_script = tmp_path / "tmp" / "gen_report.py"
    tmp_script.parent.mkdir()
    tmp_script.write_text("print('helper')", encoding="utf-8")

    now = time.time()
    os.utime(current_report, (now, now))
    os.utime(tmp_script, (now, now))
    os.utime(old_report, (now - 86_400, now - 86_400))

    sid = "cron_jobR_20260628_143034"
    db = SessionDB()
    db.create_session(sid, source="cron", user_id="userA")
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 10, sid))
    db.append_message(
        sid,
        role="assistant",
        content="I will write a helper script.",
        tool_calls=[
            {
                "name": "write_file",
                "arguments": _json.dumps({"path": str(tmp_script), "content": "print('helper')"}),
            }
        ],
    )
    db.append_message(
        sid,
        role="assistant",
        content="Running report generator.",
        tool_calls=[
            {
                "name": "terminal",
                "arguments": _json.dumps({"command": f"python3 {tmp_script}"}),
            }
        ],
    )
    db.append_message(
        sid,
        role="tool",
        tool_name="terminal",
        content=f"REPORT: {current_report}\nSIZE: {current_report.stat().st_size}",
    )
    db.close()

    job = {"origin": {"platform": "zet_agent", "chat_id": f"zettlab:userA:{agent_id}:chat1"}}
    attachments = zc._collect_produced_files("jobR", job)

    assert [a["name"] for a in attachments] == ["reddit_report_2026-06-28.html"]
    assert attachments[0]["path"] == str(current_report.resolve())
    assert attachments[0]["mime"] == "text/html"


def test_collect_produced_files_filters_sync_helper_script(tmp_path, monkeypatch):
    """ZET-1782: sync helper scripts are not user-facing cron attachments."""
    import os
    import time

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    from hermes_state import SessionDB

    agent_id = "4c452598-e213-4e86-bc17-412d99bc8ab9"
    bucket = tmp_path / "volume1" / "subvol" / "agents" / "data" / agent_id / "output" / "fdea6ea26aa1"
    bucket.mkdir(parents=True)
    helper = bucket / "sync_v2.py"
    helper.write_text("print('sync helper')", encoding="utf-8")

    now = time.time()
    os.utime(helper, (now, now))

    sid = "cron_jobSync_20260627_140004"
    db = SessionDB()
    db.create_session(sid, source="cron", user_id="userA")
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 5, sid))
    db.append_message(
        sid,
        role="tool",
        tool_name="terminal",
        content=f"checked helper: {helper}",
    )
    db.close()

    job = {"origin": {"platform": "zet_agent", "chat_id": f"zettlab:userA:{agent_id}:chat1"}}
    assert zc._collect_produced_files("jobSync", job) == []


def test_collect_produced_files_allows_external_chat_deliverables(tmp_path, monkeypatch):
    """A report created outside the output bucket is still a user deliverable
    when the current cron run mentions it; temp/helper scripts remain hidden."""
    import json as _json
    import os
    import time

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(zc, "_CRON_ATTACHMENT_TEMP_DIRS", frozenset({"tmp", "var/tmp"}))

    from hermes_state import SessionDB

    root_dir = tmp_path / "root"
    root_dir.mkdir()
    report = root_dir / "nas-youtube-report-2026-06-29.html"
    report.write_text("<html>youtube report</html>", encoding="utf-8")
    tmp_script = tmp_path / "tmp" / "gen_report.py"
    tmp_script.parent.mkdir()
    tmp_script.write_text("open('/root/nas-youtube-report.html', 'w')", encoding="utf-8")

    now = time.time()
    os.utime(report, (now, now))
    os.utime(tmp_script, (now, now))

    sid = "cron_jobY_20260629_143034"
    db = SessionDB()
    db.create_session(sid, source="cron", user_id="userA")
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 5, sid))
    db.append_message(
        sid,
        role="assistant",
        content="Generating the YouTube report.",
        tool_calls=[
            {
                "name": "write_file",
                "arguments": _json.dumps({"path": str(tmp_script), "content": "helper"}),
            }
        ],
    )
    db.append_message(
        sid,
        role="tool",
        tool_name="terminal",
        content=f"wrote report: {report}\nhelper: {tmp_script}",
    )
    db.close()

    attachments = zc._collect_produced_files("jobY", {"origin": {"platform": "zet_agent"}})

    assert [a["name"] for a in attachments] == ["nas-youtube-report-2026-06-29.html"]
    assert attachments[0]["path"] == str(report.resolve())


def test_collect_produced_files_rejects_agent_output_without_agent_scope(tmp_path, monkeypatch):
    """Agent output paths must not fall through to the external-file fallback."""
    import os
    import time

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    from hermes_state import SessionDB

    other_agent = "other-agent"
    bucket = tmp_path / "volume1" / "subvol" / "agents" / "data" / other_agent / "output" / "session-a"
    bucket.mkdir(parents=True)
    report = bucket / "report.html"
    report.write_text("<html>foreign</html>", encoding="utf-8")

    now = time.time()
    os.utime(report, (now, now))

    sid = "cron_jobNoScope_20260629_143034"
    db = SessionDB()
    db.create_session(sid, source="cron", user_id="userA")
    with db._lock:
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (now - 5, sid))
    db.append_message(
        sid,
        role="tool",
        tool_name="terminal",
        content=f"report: {report}",
    )
    db.close()

    assert zc._collect_produced_files("jobNoScope", {"origin": {"platform": "zet_agent"}}) == []


def test_cron_summary_carries_calendar_metadata(tmp_path, monkeypatch):
    import json as _json

    import hermes_state
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    content = zc._build_typed_message_content(
        {
            "id": "cal-job",
            "name": "项目评审",
            "source": "calendar",
            "calendar_source_type": "device_calendar",
            "calendar_source_instance_id": "device:u1:ios:install-1",
            "calendar_source_platform": "ios",
            "calendar_provider": "device_calendar",
            "calendar_connection_id": "install-1",
            "calendar_id": "local-cal",
            "calendar_event_id": "series-1_20260626T070000Z",
            "calendar_series_id": "series-1",
            "calendar_original_start": "2026-06-26T07:00:00Z",
            "content": "项目评审",
            "schedule": {"kind": "once", "run_at": "2026-06-26T07:00:00Z"},
            "deliver": "origin",
            "origin": {"platform": "zet_agent", "chat_id": "zettlab:u1:main:calendar-reminders"},
        },
        "cal-job",
        True,
        None,
        None,
    )
    fence = content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0]
    meta = _json.loads(fence)
    assert meta["source"] == "calendar"
    assert meta["calendar_source_type"] == "device_calendar"
    assert meta["calendar_source_instance_id"] == "device:u1:ios:install-1"
    assert meta["calendar_source_platform"] == "ios"
    assert meta["calendar_provider"] == "device_calendar"
    assert meta["calendar_id"] == "local-cal"
    assert meta["calendar_event_id"] == "series-1_20260626T070000Z"
    assert meta["calendar_original_start"] == "2026-06-26T07:00:00Z"
    assert meta["content"] == "项目评审"
    assert content.endswith("\n项目评审")


# ── ZET-1565: friendly failure messaging + run-level auto-retry ──────

_RAW_502 = (
    "RuntimeError: HTTP 502: Error code: 502 - {'error': 'Post "
    "\"https://us-iam-gw.zettlab.com/ai/v1/chat/completions\": context "
    "deadline exceeded (Client.Timeout exceeded while awaiting headers)'}"
)


def test_is_retryable_error_classification():
    import gateway.platforms.zet_agent_cron as zc

    assert zc._is_retryable_error(_RAW_502) is True
    assert zc._is_retryable_error("HTTP 503 Service Unavailable") is True
    assert zc._is_retryable_error("connection reset by peer") is True
    # real agent/logic errors are NOT retryable
    assert zc._is_retryable_error("ValueError: bad config") is False
    assert zc._is_retryable_error(None) is False
    assert zc._is_retryable_error("") is False


def test_friendly_failure_hides_raw_detail():
    import gateway.platforms.zet_agent_cron as zc

    msg = zc._friendly_failure("群聊日报整点更新", _RAW_502)
    assert "群聊日报整点更新" in msg
    # never leak stack / internal gateway URL / 502
    assert "RuntimeError" not in msg
    assert "us-iam-gw" not in msg
    assert "502" not in msg
    # non-retryable still gets a friendly line, no raw text
    other = zc._friendly_failure("X", "ValueError: bad config")
    assert "ValueError" not in other and "X" in other


def test_build_typed_message_content_failure_is_friendly():
    import gateway.platforms.zet_agent_cron as zc

    job = {"name": "群聊日报整点更新", "schedule": {"display": "每小时"}}
    # On failure, _LATEST_OUTPUT holds the raw FAILED doc (prompt + error).
    zc._LATEST_OUTPUT["job-f"] = (
        "# Cron Job: 群聊日报整点更新 (FAILED)\n\n## Prompt\n\n"
        "用 lark-cli 读取 /volume1/subvol/agents/data/abc/output ...\n\n"
        f"## Error\n\n```\n{_RAW_502}\n```\n"
    )
    try:
        content = zc._build_typed_message_content(job, "job-f", False, _RAW_502, None)
    finally:
        zc._LATEST_OUTPUT.pop("job-f", None)

    # card metadata still marks it failed
    assert '"last_run_result": "failed"' in content
    # body must NOT leak the raw doc
    assert "## Prompt" not in content
    assert "RuntimeError" not in content
    assert "us-iam-gw" not in content
    assert "/volume1/subvol" not in content
    # body carries the friendly line (default-locale reason, no raw error)
    assert "群聊日报整点更新" in content
    assert "AI 服务暂时繁忙" in content


def test_malformed_cron_success_becomes_empty_response_card(monkeypatch):
    import json as _json

    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_count_tool_activity", lambda _jid: 3)
    zc._LATEST_OUTPUT["job-bad"] = "# Cron Job: x\n\n## Response\n\n`.app\n"
    try:
        reason = zc._detect_fake_success("job-bad")
        assert reason is not None
        content = zc._build_typed_message_content(
            {
                "id": "job-bad",
                "name": "飞书审批费用同步",
                "schedule": {"display": "Daily at 22:00"},
            },
            "job-bad",
            False,
            reason,
            None,
        )
    finally:
        zc._LATEST_OUTPUT.pop("job-bad", None)

    meta = _json.loads(content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0])
    assert meta["failure"] == {"code": "empty_response", "retryable": False}
    assert "本次未产出有效结果" in content
    assert "`.app" not in content


def test_run_job_with_retry_retries_clean_transient(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)  # clean

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        # succeed on the 2nd attempt
        if calls["n"] >= 2:
            return (True, "ok", "done", None)
        return (False, "out", "", _RAW_502)

    ok, _out, final, err = zc._run_job_with_retry(orig, {"id": "j1"})
    assert ok is True and final == "done" and err is None
    assert calls["n"] == 2  # retried once


def test_installed_retry_wrapper_releases_only_failed_attempt_agents(monkeypatch):
    from cron import scheduler

    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_MAX_RUN_RETRIES", 2)
    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)

    calls = {"n": 0}
    released = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        calls["n"] += 1
        defer_agent_teardown.append(f"agent-{calls['n']}")
        if calls["n"] == 1:
            return (False, "out", "", _RAW_502)
        return (True, "ok", "done", None)

    # install() uses mark_job_run's sentinel as its global idempotency guard.
    # Replace it together with run_job so this test installs a fresh wrapper
    # around the controlled fake without disturbing the already-installed
    # process-global wrappers after monkeypatch teardown.
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "save_job_output", lambda _job_id, output: output)
    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(
        scheduler,
        "_teardown_cron_agent",
        lambda agent, job_id: released.append((agent, job_id)),
    )
    zc.install()

    deferred = []
    result = scheduler.run_job(
        {"id": "resource-job"},
        defer_agent_teardown=deferred,
    )

    assert result == (True, "ok", "done", None)
    assert calls["n"] == 2
    assert released == [("agent-1", "resource-job")]
    assert deferred == ["agent-2"]


def test_run_job_with_retry_caps_attempts(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j2"})
    assert ok is False
    assert calls["n"] == 1 + zc._MAX_RUN_RETRIES  # initial + capped retries


def test_run_job_with_retry_skips_when_tools_ran(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 3)  # work happened
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    zc._run_job_with_retry(orig, {"id": "j3"})
    assert calls["n"] == 1  # no retry — partially-executed run must not repeat


def test_retry_metadata_explains_partial_tool_activity_skip(monkeypatch):
    import json as _json

    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 4)

    def orig(job):
        return (False, "out", "", _RAW_502)

    zc._run_job_with_retry(orig, {"id": "paper-job"})
    content = zc._build_typed_message_content(
        {"id": "paper-job", "name": "每日论文推荐", "schedule": {"display": "每天10:00"}},
        "paper-job",
        False,
        _RAW_502,
        None,
    )
    meta = _json.loads(content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0])

    assert meta["failure"]["code"] == "upstream_unavailable"
    assert meta["failure"]["retryable"] is True
    assert meta["failure"]["retry"] == {
        "attempts": 0,
        "max_attempts": zc._MAX_RUN_RETRIES,
        "skipped_reason": "tool_activity",
        "tool_activity": 4,
    }
    assert "已执行部分步骤" in content
    assert "为避免重复操作未自动重试" in content
    assert "RuntimeError" not in content and "us-iam-gw" not in content


def test_retry_metadata_records_clean_retry_exhaustion(monkeypatch):
    import json as _json

    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    zc._run_job_with_retry(orig, {"id": "clean-job"})
    content = zc._build_typed_message_content(
        {"id": "clean-job", "name": "早报", "schedule": {"display": "每天"}},
        "clean-job",
        False,
        _RAW_502,
        None,
    )
    meta = _json.loads(content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0])

    assert calls["n"] == 1 + zc._MAX_RUN_RETRIES
    assert meta["failure"]["retry"]["attempts"] == zc._MAX_RUN_RETRIES
    assert meta["failure"]["retry"]["skipped_reason"] == "retry_exhausted"
    assert f"已自动重试 {zc._MAX_RUN_RETRIES} 次仍失败" in content


def test_run_job_with_retry_no_retry_on_non_transient(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", "ValueError: bad config")

    zc._run_job_with_retry(orig, {"id": "j4"})
    assert calls["n"] == 1  # non-retryable error → no retry


def test_run_job_with_retry_no_retry_on_success(monkeypatch):
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (True, "out", "done", None)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j5"})
    assert ok is True and calls["n"] == 1


# ── ZET-1565: retry guard measures the CURRENT attempt's session, not the
# most recent one — exercises the real _attempt_tool_activity snapshot-diff
# (only the SessionDB seams are stubbed, not the decision logic). ──────


def test_retry_fires_despite_prior_run_tool_activity(monkeypatch):
    """Regression: an early transient failure that creates no new session must
    still retry, even when a PRIOR run's session has tool activity. The naive
    "latest session by started_at" signal misread the prior run as work and
    suppressed the retry."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    # DB already holds a prior successful run with 3 tool events.
    store = {"cron_jobX_prior": 3}
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set(store))
    monkeypatch.setattr(zc, "_count_session_tool_activity", lambda sid: store.get(sid))

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        # current attempt 502s before creating any session; succeeds on retry
        if calls["n"] >= 2:
            return (True, "ok", "done", None)
        return (False, "out", "", _RAW_502)

    ok, _o, final, err = zc._run_job_with_retry(orig, {"id": "jobX"})
    assert ok is True and final == "done" and err is None
    assert calls["n"] == 2  # retried — prior run's tool activity correctly ignored


def test_retry_fires_when_attempt_session_has_no_tools(monkeypatch):
    """A 502 *after* the session is created but before any tool ran (zero tool
    activity in this attempt's own session) is still a clean retry."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    store = {}
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set(store))
    monkeypatch.setattr(zc, "_count_session_tool_activity", lambda sid: store.get(sid))

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        if calls["n"] == 1:
            store["cron_j_run1"] = 0  # session created, zero tools
            return (False, "out", "", _RAW_502)
        return (True, "ok", "done", None)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is True and calls["n"] == 2


def test_retry_isolates_per_attempt_session(monkeypatch):
    """The guard measures only the CURRENT attempt: a retry whose own attempt
    runs tools then fails transiently must not retry again — even though the
    earlier attempt ran clean."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    store = {"cron_j_prior": 5}  # prior run, irrelevant to current attempts
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set(store))
    monkeypatch.setattr(zc, "_count_session_tool_activity", lambda sid: store.get(sid))

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        if calls["n"] == 1:
            return (False, "out", "", _RAW_502)  # clean early 502, no session
        if calls["n"] == 2:
            store["cron_j_run2"] = 2  # this attempt ran 2 tools…
            return (False, "out", "", _RAW_502)  # …then failed transiently
        return (True, "ok", "done", None)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is False
    assert calls["n"] == 2  # attempt1 retried; attempt2 ran tools → stop


def test_retry_skipped_when_session_set_unreadable(monkeypatch):
    """If the cron-session set can't be resolved, the attempt's activity is
    unknown → fail-safe to no retry (never repeat possible side effects)."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)  # no real backoff wait
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: None)  # unreadable
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is False and calls["n"] == 1


def test_retry_aborts_on_shutdown(monkeypatch):
    """A shutdown signalled during the backoff must abort the retry so the cron
    pool's atexit drain (wait=True) is never stalled by a sleeping worker."""
    import threading

    import gateway.platforms.zet_agent_cron as zc

    ev = threading.Event()
    ev.set()  # shutdown already in progress
    monkeypatch.setattr(zc, "_shutdown", ev)
    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 30)  # would block if not interruptible
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)  # clean → would retry

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is False and calls["n"] == 1  # shutdown during backoff → no retry


def test_no_agent_script_job_never_retries(monkeypatch):
    """no_agent script jobs create no session, so tool-activity can't measure
    their side effects. They must never retry — even when the script's own
    stdout/stderr (returned as result[3]) looks transient (e.g. "timed out")."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    # If the guard were reached it would read 0 (no session) → "safe to retry".
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        # script failed; result[3] is the script output, which reads transient
        return (False, "doc", "alert", "health probe: connection refused, timed out")

    ok, *_ = zc._run_job_with_retry(orig, {"id": "watchdog", "no_agent": True})
    assert ok is False and calls["n"] == 1  # script executed exactly once


def test_agent_job_with_script_never_retries(monkeypatch):
    """P1 regression: an ordinary (agent) job can also carry job["script"], which
    cron/scheduler.py runs BEFORE the LLM call. That script's side effects (file
    writes / webhooks) aren't Hermes tool activity, so the guard would read 0 and
    wrongly retry — re-executing the script. Any job with a script runs once."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    # If the guard were reached it would read 0 → "safe to retry".
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)  # transient LLM 502 after script ran

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j", "script": "collect.py"})
    assert ok is False and calls["n"] == 1  # script side effects never repeated


def test_retry_disabled_skips_baseline_query_and_never_retries(monkeypatch):
    """ZET_CRON_RETRY_MAX=0 disables retries: the job runs once and the pre-loop
    baseline session-id query is skipped entirely (no per-tick DB round-trip)."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_MAX_RUN_RETRIES", 0)
    baseline_queries = {"n": 0}
    monkeypatch.setattr(
        zc, "_list_cron_session_ids",
        lambda _jid: (baseline_queries.__setitem__("n", baseline_queries["n"] + 1) or set()),
    )

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", _RAW_502)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is False and calls["n"] == 1  # ran once, no retry
    assert baseline_queries["n"] == 0  # disabled → never paid the baseline DB query


def test_retry_counts_tools_in_reused_session_id(monkeypatch):
    """P2 regression: with BACKOFF_S=0 a retry can reuse the same second-precision
    cron session id (INSERT OR IGNORE keeps the existing row). The guard must
    count the per-session tool-count DELTA, not just brand-new ids — otherwise the
    second attempt's tool calls land in the existing id, the new-id set diff reads
    empty, and a third attempt repeats side effects."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    store = {}
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set(store))
    monkeypatch.setattr(zc, "_count_session_tool_activity", lambda sid: store.get(sid))

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        if calls["n"] == 1:
            store["cron_j_T"] = 0  # session created this attempt, zero tools
            return (False, "out", "", _RAW_502)
        if calls["n"] == 2:
            store["cron_j_T"] = 3  # SAME id reused; this attempt ran 3 tools
            return (False, "out", "", _RAW_502)
        return (True, "ok", "done", None)

    ok, *_ = zc._run_job_with_retry(orig, {"id": "j"})
    assert ok is False
    assert calls["n"] == 2  # attempt2's tools seen via delta on reused id → stop


def test_retry_tolerates_malformed_result_tuple(monkeypatch):
    """issue_1 regression: if upstream run_job ever returns a tuple shorter than
    4 elements, the retry guard must treat it as non-retryable rather than crash
    the scheduler thread with an IndexError on result[3]."""
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())

    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "short tuple")  # malformed: only 2 elements

    result = zc._run_job_with_retry(orig, {"id": "j"})
    assert result == (False, "short tuple")  # returned as-is, no crash
    assert calls["n"] == 1  # malformed → not retried


def test_redact_channel_failure_multiline_name_no_leak():
    """ZET-1565 review (KILLY000): job names are stored verbatim, so an interior
    newline defeated the `.*?` regex and leaked the raw error to channel bridges.
    Exact-prefix reconstruction redacts a multi-line-named failure instead."""
    import gateway.platforms.zet_agent_cron as zc

    job = {"id": "j1", "name": "multi\nline"}
    raw = f"⚠️ Cron job 'multi\nline' failed:\n{_RAW_502}"
    out = zc._redact_channel_failure(job, raw)
    # the raw stack / internal URL / 502 must be gone
    assert "RuntimeError" not in out and "us-iam-gw" not in out and "502" not in out
    assert "执行失败" in out  # friendly line rendered


def test_redact_channel_failure_single_line_and_passthrough():
    import gateway.platforms.zet_agent_cron as zc

    job = {"id": "j2", "name": "daily"}
    out = zc._redact_channel_failure(job, f"⚠️ Cron job 'daily' failed:\n{_RAW_502}")
    assert "502" not in out and "daily" in out
    # name-absent falls back to the job id in the upstream template
    job_noname = {"id": "abc123"}
    out2 = zc._redact_channel_failure(
        job_noname, f"⚠️ Cron job 'abc123' failed:\n{_RAW_502}"
    )
    assert "502" not in out2
    # non-failure content (and falsy) passes through untouched
    assert zc._redact_channel_failure(job, "hello world") == "hello world"
    assert zc._redact_channel_failure(job, "") == ""


def test_failure_template_self_check_matches_live_upstream():
    """The marker must be present in the live upstream scheduler source — the
    friendly-ize path silently leaks raw errors once this drifts."""
    import inspect

    import cron.scheduler as sched
    import gateway.platforms.zet_agent_cron as zc

    assert zc._warn_if_failure_template_drifted(inspect.getsource(sched)) is True


def test_failure_template_self_check_warns_on_drift(monkeypatch):
    """When upstream drops/changes the template, the check returns False and
    emits a visible warning instead of degrading silently."""
    import gateway.platforms.zet_agent_cron as zc

    warnings = []
    monkeypatch.setattr(zc, "_dbg", lambda m: warnings.append(m))

    assert zc._warn_if_failure_template_drifted("def _process_job(): pass") is False
    assert any("template changed" in w for w in warnings)


def test_failure_template_self_check_warns_on_partial_drift(monkeypatch):
    """Prefix intact but the "failed:" wording changed → _redact_channel_failure
    would stop matching; the check must still warn (it verifies BOTH fragments,
    not just the prefix), else raw errors leak silently."""
    import gateway.platforms.zet_agent_cron as zc

    warnings = []
    monkeypatch.setattr(zc, "_dbg", lambda m: warnings.append(m))

    # Keeps "⚠️ Cron job '" but drops "' failed:" — a prefix-only check missed this.
    partial = "msg = f\"⚠️ Cron job '{name}' bombed:\\n{error}\""
    assert zc._warn_if_failure_template_drifted(partial) is False
    assert any("template changed" in w for w in warnings)


def test_retry_budget_is_env_configurable(monkeypatch):
    """ZET_CRON_RETRY_MAX=0 disables retries; values are clamped to a sane range."""
    import gateway.platforms.zet_agent_cron as zc

    assert zc._env_int("ZET_X_UNSET", 2, lo=0, hi=10) == 2  # missing → default
    monkeypatch.setenv("ZET_X_BAD", "not-an-int")
    assert zc._env_int("ZET_X_BAD", 2, lo=0, hi=10) == 2  # garbage → default
    monkeypatch.setenv("ZET_X_HI", "999")
    assert zc._env_int("ZET_X_HI", 2, lo=0, hi=10) == 10  # clamped to hi
    monkeypatch.setenv("ZET_X_ZERO", "0")
    assert zc._env_int("ZET_X_ZERO", 2, lo=0, hi=10) == 0  # explicit disable


def test_classify_failure_codes():
    import gateway.platforms.zet_agent_cron as zc

    assert zc._classify_failure(_RAW_502) == ("upstream_unavailable", True)
    assert zc._classify_failure("HTTP 503") == ("upstream_unavailable", True)
    assert zc._classify_failure(
        "httpx.ConnectError: [Errno -3] Temporary failure in name resolution"
    ) == ("network_unavailable", True)
    assert zc._classify_failure(
        "socket.gaierror: [Errno -2] Name or service not known"
    ) == ("network_unavailable", True)
    assert zc._classify_failure("transport connection error") == (
        "network_unavailable",
        True,
    )
    assert zc._classify_failure(
        "read tcp 10.0.0.1:443: connection reset by peer"
    ) == ("network_unavailable", True)
    assert zc._classify_failure("EOF") == ("network_unavailable", True)
    assert zc._classify_failure("response ended with EOF") == (
        "network_unavailable",
        True,
    )
    assert zc._classify_failure("unexpected EOF while parsing JSON") == (
        "agent_error",
        False,
    )
    assert zc._classify_failure("file stream ended with EOF") == (
        "agent_error",
        False,
    )
    assert zc._classify_failure("context deadline exceeded") == ("timeout", True)
    assert zc._classify_failure("HTTP 429: rate limit exceeded") == ("rate_limited", True)
    assert zc._classify_failure("rate_limit exceeded") == ("rate_limited", True)
    assert zc._classify_failure(
        "HTTP 402: insufficient_credits"
    ) == ("insufficient_credits", False)
    assert zc._classify_failure("credit balance exhausted") == (
        "insufficient_credits",
        False,
    )
    assert zc._classify_failure("failed to fetch credit balance") == (
        "agent_error",
        False,
    )
    # Real upstream #8585 sentinel — note it embeds "timeout"; empty-result must
    # win over the retryable-keyword match, else it's mislabeled retryable.
    assert zc._classify_failure(
        "Agent completed but produced empty response "
        "(model error, timeout, or misconfiguration)"
    ) == ("empty_response", False)
    assert zc._classify_failure("Agent produced no usable output") == (
        "empty_response",
        False,
    )
    assert zc._classify_failure("No response generated") == ("empty_response", False)
    assert zc._classify_failure("ValueError: bad config") == ("agent_error", False)
    assert zc._classify_failure("request quota exceeded") == ("agent_error", False)
    assert zc._classify_failure("model not configured") == ("agent_error", False)
    assert zc._classify_failure(None) == ("unknown", False)
    assert zc._classify_failure("") == ("unknown", False)


def test_dns_failure_retries_and_carries_specific_metadata(monkeypatch):
    import json as _json
    import gateway.platforms.zet_agent_cron as zc

    monkeypatch.setattr(zc, "_RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(zc, "_list_cron_session_ids", lambda _jid: set())
    monkeypatch.setattr(zc, "_attempt_tool_activity", lambda _jid, _before: 0)
    dns_error = "httpx.ConnectError: [Errno -3] Temporary failure in name resolution"
    calls = {"n": 0}

    def orig(job):
        calls["n"] += 1
        return (False, "out", "", dns_error)

    zc._run_job_with_retry(orig, {"id": "dns-job"})
    content = zc._build_typed_message_content(
        {"id": "dns-job", "name": "网络验收", "schedule": {"display": "每天"}},
        "dns-job",
        False,
        dns_error,
        None,
    )
    meta = _json.loads(content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0])

    assert calls["n"] == 1 + zc._MAX_RUN_RETRIES
    assert meta["failure"]["code"] == "network_unavailable"
    assert meta["failure"]["retryable"] is True
    assert meta["failure"]["retry"]["skipped_reason"] == "retry_exhausted"
    assert "暂时无法连接服务" in content
    assert "Temporary failure in name resolution" not in content


def test_failure_metadata_carries_structured_code():
    import json as _json
    import gateway.platforms.zet_agent_cron as zc

    job = {"name": "群聊日报整点更新", "schedule": {"display": "每小时"}}
    content = zc._build_typed_message_content(job, "job-fc", False, _RAW_502, None)
    fence = content.split("```cron-summary\n", 1)[1].split("\n```", 1)[0]
    meta = _json.loads(fence)
    assert meta["last_run_result"] == "failed"
    assert meta["failure"] == {"code": "upstream_unavailable", "retryable": True}

    # success path carries no failure field
    ok = zc._build_typed_message_content(job, "job-ok", True, None, None)
    ok_fence = ok.split("```cron-summary\n", 1)[1].split("\n```", 1)[0]
    assert "failure" not in _json.loads(ok_fence)


class _FakeCronAgent:
    def __init__(self, turn_id, interrupted):
        self._current_turn_id = turn_id
        self._interrupt_requested = interrupted


def _install_wrapper_with_agent(monkeypatch, agent, finishes):
    """Install a fresh run_job wrapper around a fake that defers *agent*."""
    from cron import scheduler

    import gateway.platforms.zet_agent_cron as zc
    import tools.zettlab_snapshot_guard as guard

    monkeypatch.setattr(zc, "_MAX_RUN_RETRIES", 0)
    monkeypatch.setattr(
        guard, "finish_turn",
        lambda state, **kw: finishes.append((state, kw.get("turn_id", ""))),
    )

    def fake_run_job(job, *, defer_agent_teardown=None):
        defer_agent_teardown.append(agent)
        return (False, "out", "", "TimeoutError: Cron job 'j' idle for 600s")

    monkeypatch.setattr(scheduler, "mark_job_run", lambda *a, **kw: None)
    monkeypatch.setattr(scheduler, "save_job_output", lambda _jid, output: output)
    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *a, **kw: None)
    zc.install()
    return scheduler


def test_interrupted_cron_run_leaves_snapshot_pin_to_ttl(monkeypatch):
    """inactivity timeout 路径：scheduler 先 interrupt 再 shutdown(wait=False)
    返回失败，工具线程未必已退出——wrapper 不收 pin，留给 LS 侧 PinTTL +
    reconcile 自愈（Codex review P1）。"""
    finishes = []
    scheduler = _install_wrapper_with_agent(
        monkeypatch, _FakeCronAgent("turn_cron_1", interrupted=True), finishes)

    deferred = []
    scheduler.run_job({"id": "timeout-job"}, defer_agent_teardown=deferred)

    assert finishes == [], "被 interrupt 的轮不能提前 finish 解 pin"


def test_normal_failed_cron_run_still_finishes_turn(monkeypatch):
    """普通失败（没有 interrupt）照常收 pin，上报 failed 终态。"""
    finishes = []
    scheduler = _install_wrapper_with_agent(
        monkeypatch, _FakeCronAgent("turn_cron_2", interrupted=False), finishes)

    deferred = []
    scheduler.run_job({"id": "failed-job"}, defer_agent_teardown=deferred)

    assert finishes == [("failed", "turn_cron_2")]


def test_persist_finds_root_store_when_override_points_at_profile_home(tmp_path, monkeypatch):
    """board229 实测缺陷：cron 线程的 home override 指向 <root>/profiles/<agent> 时，
    候选集里没有 root 自己的 state.db —— root-only 存量会话被判成已删，每次 run 派生 handoff。

    原有用例的 override 是根 home，探不到这条路径。"""
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    # 关键差异：home override = profile home，不是根 home
    _mux_env(monkeypatch, prof, root, root / "state.db")
    zc.install()

    SID = f"zettlab:userA:{AID}:rootonly1"
    JID = "jobRootOnly"
    cron_jobs.save_jobs([_mux_cron_job(JID, SID)])

    # 会话只存在于 root 库；profile 库是空的（存量漂移的真实形态）
    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session(SID, source="zet_agent", user_id="userA")
    root_db.close()
    SessionDB(db_path=prof / "state.db").close()

    assert zc._resolve_persist_db_path(SID) == (root / "state.db", None)

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n验证成功\n"
    try:
        scheduler.mark_job_run(JID, True, scheduled_at="2026-08-05T10:00:00Z",
                               output_filename="run.md")
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)

    # 落在原会话（root 库），且没有派生新会话
    assert _cron_summary_rows(root / "state.db", SID) == 1
    import sqlite3
    con = sqlite3.connect(root / "state.db")
    try:
        extra = [r[0] for r in con.execute(
            "SELECT id FROM sessions WHERE id LIKE ? AND id != ?", (f"zettlab:%{AID}:%", SID))]
    finally:
        con.close()
    assert extra == [], f"派生了 handoff 会话: {extra}"


@contextmanager
def _context_home_override(home):
    """cron 线程的真实形态：context 级 override，进程 HERMES_HOME 留在别处。"""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def test_persist_finds_root_store_with_context_level_profile_override(tmp_path, monkeypatch):
    """生产形态：context override = profile home，进程 HERMES_HOME = root。
    monkeypatch.setenv 是进程级，get_hermes_home() 与 get_process_hermes_home()
    会永远相等，那半段候选逻辑一行都覆盖不到 —— 必须走 set_hermes_home_override。"""
    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_constants import get_hermes_home, get_process_hermes_home
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    zc.install()

    SID = f"zettlab:userA:{AID}:rootonly2"
    JID = "jobCtxRootOnly"
    cron_jobs.save_jobs([_mux_cron_job(JID, SID)])

    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session(SID, source="zet_agent", user_id="userA")
    root_db.close()
    SessionDB(db_path=prof / "state.db").close()

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\ncontext override\n"
    with _context_home_override(prof):
        assert get_hermes_home() != get_process_hermes_home(), "两个 home 必须真的分叉"
        assert zc._resolve_persist_db_path(SID) == (root / "state.db", None)
        try:
            scheduler.mark_job_run(JID, True, scheduled_at="2026-08-05T10:00:00Z",
                                   output_filename="run.md")
        finally:
            zc._LATEST_OUTPUT.pop(JID, None)

    assert _cron_summary_rows(root / "state.db", SID) == 1
    assert _cron_summary_rows(prof / "state.db", SID) == 0
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID


def test_new_session_never_lands_in_legacy_nested_profiles_dir(tmp_path, monkeypatch):
    """遗留的 <root>/profiles/<A>/profiles/<A>/ 目录不许成为候选：它排在第一位，
    全新会话（handoff / calendar-reminders）会被建进一个 local-server 永远读不到的库。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "main"
    prof = root / "profiles" / AID
    nested = prof / "profiles" / AID
    nested.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:calendar-reminders"
    JID = "jobNested"
    job = _mux_cron_job(JID, SID, name="喝水提醒")
    cron_jobs.save_jobs([job])

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n该喝水啦\n"
    with _context_home_override(prof):
        assert nested / "state.db" not in zc._profile_state_db_candidates(AID)
        try:
            assert zc._try_persist_to_session(JID, True, None, None, job) is None
        finally:
            zc._LATEST_OUTPUT.pop(JID, None)

    assert not (nested / "state.db").exists(), "新会话落进了嵌套假库"
    assert _cron_summary_rows(prof / "state.db", SID) == 1
    prof_db = SessionDB(db_path=prof / "state.db", read_only=True)
    try:
        assert prof_db.get_session(SID) is not None
    finally:
        prof_db.close()


def test_unreadable_owning_store_fails_closed_and_keeps_origin(tmp_path, monkeypatch):
    """探测失败 != 会话已删。owning store 读不出来时必须 fail-closed：
    落 last_delivery_error，绝不派生 handoff 会话去永久改写 job.origin.chat_id。"""
    import os

    import cron.jobs as cron_jobs
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    if os.geteuid() == 0:
        pytest.skip("root bypasses file mode bits, cannot simulate an unreadable store")

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    zc.install()

    SID = f"zettlab:userA:{AID}:rootonly3"
    JID = "jobUnreadable"
    cron_jobs.save_jobs([_mux_cron_job(JID, SID)])

    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session(SID, source="zet_agent", user_id="userA")
    root_db.close()

    for suffix in ("", "-wal", "-shm"):
        sidecar = root / f"state.db{suffix}"
        if sidecar.exists():
            os.chmod(sidecar, 0o000)

    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n读不出来\n"
    try:
        with _context_home_override(prof):
            _, unresolved = zc._resolve_persist_db_path(SID)
            assert unresolved, "探测失败必须报未决，不能静默当成会话不存在"
            scheduler.mark_job_run(JID, True, scheduled_at="2026-08-05T11:00:00Z",
                                   output_filename="run.md")
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)
        for suffix in ("", "-wal", "-shm"):
            sidecar = root / f"state.db{suffix}"
            if sidecar.exists():
                os.chmod(sidecar, 0o600)

    stored = cron_jobs.get_job(JID)
    assert stored["origin"]["chat_id"] == SID, "fail-open 了：origin 被改写成 handoff 会话"
    assert stored.get("last_delivery_error"), "失败没有落到 last_delivery_error"
    assert _cron_summary_rows(prof / "state.db", SID) == 0


def _fake_probe(monkeypatch, zc, verdicts):
    """按 resolve 后的路径钉死 _db_has_session 三态返回。"""
    resolved = {str(p.resolve()): v for p, v in verdicts.items()}

    def probe(db_path, session_id):
        return resolved.get(str(db_path.resolve()), False)

    monkeypatch.setattr(zc, "_db_has_session", probe)


def test_undecided_profile_probe_blocks_root_fallback(tmp_path, monkeypatch):
    """profile 候选探测未决（None）时，root 命中也必须返回未决——
    否则 profile 其实持有该会话时结果被写进 root，split session 扩大。"""
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:undecided1"
    _fake_probe(monkeypatch, zc, {prof / "state.db": None, root / "state.db": True})

    db_path, unresolved = zc._resolve_persist_db_path(SID)
    assert unresolved, "profile 候选未决时不许降级返回 root"
    assert str((prof / "state.db").resolve()) in unresolved


def test_excluded_profile_falls_back_to_owning_root(tmp_path, monkeypatch):
    """profile 候选明确不拥有（False）+ root 拥有 → 正常降级到 root。"""
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:undecided2"
    _fake_probe(monkeypatch, zc, {prof / "state.db": False, root / "state.db": True})

    assert zc._resolve_persist_db_path(SID) == (root / "state.db", None)


def test_all_candidates_excluded_creates_new_in_profile(tmp_path, monkeypatch):
    """全部候选明确不拥有 → 走 new-in-profile 分支。"""
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:undecided3"
    _fake_probe(monkeypatch, zc, {prof / "state.db": False, root / "state.db": False})

    assert zc._resolve_persist_db_path(SID) == ((prof / "state.db").resolve(), None)


def test_split_session_logs_warning_instead_of_silently_preferring_profile(
    tmp_path, monkeypatch, caplog
):
    """§6.3：两库同时命中时静默取 profile 会掩盖劈裂，必须告警 + 打点。"""
    import logging

    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")

    SID = f"zettlab:userA:{AID}:split01"
    for db_file in (root / "state.db", prof / "state.db"):
        db = SessionDB(db_path=db_file)
        db.create_session(SID, source="zet_agent", user_id="userA")
        db.close()

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.zet_agent_cron"):
        db_path, unresolved = zc._resolve_persist_db_path(SID)

    assert (db_path, unresolved) == ((prof / "state.db").resolve(), None)
    assert any("split session detected" in r.getMessage() for r in caplog.records)


def test_root_candidates_cover_both_context_and_process_homes(tmp_path, monkeypatch):
    """候选集必须同时含 context override 和进程 HERMES_HOME 两条 root。
    所有 _mux_env 用例都用进程级 setenv，两者永远相等 → get_process_hermes_home()
    那半段一行没被覆盖。"""
    import gateway.platforms.zet_agent_cron as zc

    root_a = tmp_path / "rootA"
    root_b = tmp_path / "rootB"
    AID = "eae0707d"
    (root_b / "profiles" / AID).mkdir(parents=True)
    _mux_env(monkeypatch, root_a, root_a, root_a / "state.db")

    with _context_home_override(root_b / "profiles" / AID):
        roots = zc._root_state_db_candidates()
        profiles = zc._profile_state_db_candidates(AID)

    assert root_b / "state.db" in roots, "context override 的 root 丢了"
    assert root_a / "state.db" in roots, "进程 HERMES_HOME 的 root 丢了"
    assert (root_b / "profiles" / AID / "state.db").resolve() in profiles
    assert (root_a / "profiles" / AID / "state.db").resolve() in profiles


def test_profile_candidates_are_symlink_resolved_not_reopened_by_name(tmp_path, monkeypatch):
    """校验用的是 db_path.resolve()，返回的就必须是同一个 resolved 路径 —— 返回未解析的
    名字等于把校验和使用分成两次 lookup（TOCTOU），中间那层 symlink 可以被换掉。"""
    import gateway.platforms.zet_agent_cron as zc

    AID = "eae0707d"
    real_profiles = tmp_path / "real_profiles"
    (real_profiles / AID).mkdir(parents=True)
    home = tmp_path / "linked_home"
    home.mkdir()
    (home / "profiles").symlink_to(real_profiles)
    _mux_env(monkeypatch, home, home, home / "state.db")

    assert zc._profile_state_db_candidates(AID) == [real_profiles / AID / "state.db"]


def test_cross_agent_origin_refused_for_profile_store(tmp_path, monkeypatch):
    """job.origin 是调用方可控输入：嵌的 agentID ≠ 当前执行 agent 时必须 fail-closed，
    绝不返回对方 profile 的库路径，也不往里写。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    EXEC = "agent0a"
    VICTIM = "agent0b"
    victim_prof = root / "profiles" / VICTIM
    victim_prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    monkeypatch.setenv("ZET_AGENT_ID", EXEC)

    SID = f"zettlab:userA:{VICTIM}:orig001"
    victim_db = SessionDB(db_path=victim_prof / "state.db")
    victim_db.create_session(SID, source="zet_agent", user_id="userA")
    victim_db.close()

    db_path, unresolved = zc._resolve_persist_db_path(SID)
    assert unresolved, "跨 agent origin 必须报未决，不能解析成功"
    assert VICTIM in unresolved and EXEC in unresolved
    assert db_path != (victim_prof / "state.db").resolve()

    JID = "jobCrossTenantProf"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])
    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n越权写入\n"
    try:
        err = zc._try_persist_to_session(JID, True, None, None, job)
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)
    assert err, "跨 agent persist 必须以投递失败上抛，不许静默丢弃"
    assert _cron_summary_rows(victim_prof / "state.db", SID) == 0
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID


def test_cross_agent_origin_refused_for_root_store(tmp_path, monkeypatch):
    """root 库是所有 agent 共享的：跨 agent origin 命中 root-only 会话时同样
    fail-closed，不许降级写 root。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    EXEC = "agent0a"
    VICTIM = "agent0b"
    (root / "profiles").mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    monkeypatch.setenv("ZET_AGENT_ID", EXEC)

    SID = f"zettlab:userA:{VICTIM}:legacy01"
    root_db = SessionDB(db_path=root / "state.db")
    root_db.create_session(SID, source="zet_agent", user_id="userA")
    root_db.close()

    _, unresolved = zc._resolve_persist_db_path(SID)
    assert unresolved, "root 命中也不能绕过跨 agent 校验"

    JID = "jobCrossTenantRoot"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])
    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n越权写 root\n"
    try:
        err = zc._try_persist_to_session(JID, True, None, None, job)
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)
    assert err
    assert _cron_summary_rows(root / "state.db", SID) == 0


def test_matching_origin_agent_resolves_normally(tmp_path, monkeypatch):
    """origin 的 agentID == 当前执行 agent（App/Web 的合法流量形状）→ 解析不变。"""
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    monkeypatch.setenv("ZET_AGENT_ID", AID)

    SID = f"zettlab:userA:{AID}:orig001"
    db = SessionDB(db_path=prof / "state.db")
    db.create_session(SID, source="zet_agent", user_id="userA")
    db.close()

    assert zc._resolve_persist_db_path(SID) == ((prof / "state.db").resolve(), None)


def test_missing_exec_agent_identity_keeps_session_routing(tmp_path, monkeypatch):
    """拿不到执行 agent 身份（ZET_AGENT_ID 空，profile override 未绑）时不做校验：
    这正是 61% 记录依赖的降级恢复路径，fail-closed 会把合法流量一起挡死。"""
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")  # 已 delenv ZET_AGENT_ID

    SID = f"zettlab:userA:{AID}:orig001"
    db = SessionDB(db_path=prof / "state.db")
    db.create_session(SID, source="zet_agent", user_id="userA")
    db.close()

    assert zc._resolve_persist_db_path(SID) == ((prof / "state.db").resolve(), None)


def test_cross_agent_origin_refused_by_job_store_profile_when_env_unbound(tmp_path, monkeypatch):
    """核心用例：ZET_AGENT_ID 拿不到（scope 未绑，读的是根 .env）时，执行身份必须
    从「job 所属 profile 的 cron store 路径」盖章 —— 该 store 是服务端事实，调用方
    改不了。跨 agent origin 仍要拦住：不落对方 profile 库，也不落 root 共享库。"""
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    EXEC = "agent0a"
    VICTIM = "agent0b"
    exec_prof = root / "profiles" / EXEC
    victim_prof = root / "profiles" / VICTIM
    exec_prof.mkdir(parents=True)
    victim_prof.mkdir(parents=True)
    # 进程 home 退化到根 home，且 ZET_AGENT_ID 未设 —— 上一轮 fail-open 的触发态；
    # 但 job 是从 EXEC 的 profile cron store 读出来的（jobs_home=exec_prof）。
    _mux_env(monkeypatch, root, exec_prof, root / "state.db")

    SID = f"zettlab:userA:{VICTIM}:orig001"
    victim_db = SessionDB(db_path=victim_prof / "state.db")
    victim_db.create_session(SID, source="zet_agent", user_id="userA")
    victim_db.close()

    assert zc._job_store_agent_id() == EXEC

    db_path, unresolved = zc._resolve_persist_db_path(SID)
    assert unresolved, "scope 未绑时也必须按 job store profile 拦住跨 agent origin"
    assert VICTIM in unresolved and EXEC in unresolved
    assert db_path != (victim_prof / "state.db").resolve()

    JID = "jobStoreCrossTenant"
    job = _mux_cron_job(JID, SID)
    cron_jobs.save_jobs([job])
    zc._LATEST_OUTPUT[JID] = "# Cron\n\n## Response\n\n越权写入\n"
    try:
        err = zc._try_persist_to_session(JID, True, None, None, job)
    finally:
        zc._LATEST_OUTPUT.pop(JID, None)
    assert err, "跨 agent persist 必须以投递失败上抛，不许静默落库"
    assert _cron_summary_rows(victim_prof / "state.db", SID) == 0
    assert _cron_summary_rows(root / "state.db", SID) == 0
    assert cron_jobs.get_job(JID)["origin"]["chat_id"] == SID


def test_job_store_profile_matching_origin_resolves_normally(tmp_path, monkeypatch):
    """合法流量：origin 的 agentID == job 所属 profile（App 提交的形状），即使
    ZET_AGENT_ID 为空也照常解析到该 profile 的库。"""
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, prof, root / "state.db")  # 已 delenv ZET_AGENT_ID

    SID = f"zettlab:userA:{AID}:orig001"
    db = SessionDB(db_path=prof / "state.db")
    db.create_session(SID, source="zet_agent", user_id="userA")
    db.close()

    assert zc._job_store_agent_id() == AID
    assert zc._resolve_persist_db_path(SID) == ((prof / "state.db").resolve(), None)


def test_job_store_profile_wins_over_stale_env_identity(tmp_path, monkeypatch):
    """store 身份和 .env 身份冲突时 store 赢：根 .env 里残留的 legacy ZET_AGENT_ID
    不能把该 profile 自己的合法 job 挡死。"""
    import gateway.platforms.zet_agent_cron as zc
    from hermes_state import SessionDB

    root = tmp_path / "hermes_home"
    AID = "eae0707d"
    prof = root / "profiles" / AID
    prof.mkdir(parents=True)
    _mux_env(monkeypatch, root, prof, root / "state.db")
    monkeypatch.setenv("ZET_AGENT_ID", "legacy-main")

    SID = f"zettlab:userA:{AID}:orig001"
    db = SessionDB(db_path=prof / "state.db")
    db.create_session(SID, source="zet_agent", user_id="userA")
    db.close()

    assert zc._resolve_persist_db_path(SID) == ((prof / "state.db").resolve(), None)


def test_job_store_agent_id_only_matches_profile_store_shape(tmp_path, monkeypatch):
    """root/legacy store（无 profiles/<X>/ 段）不产生身份；此时回退 .env 身份。"""
    import gateway.platforms.zet_agent_cron as zc

    root = tmp_path / "hermes_home"
    (root / "profiles").mkdir(parents=True)
    _mux_env(monkeypatch, root, root, root / "state.db")
    assert zc._job_store_agent_id() == ""
