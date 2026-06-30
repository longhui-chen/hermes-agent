from unittest.mock import MagicMock

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


def test_calendar_reminder_session_is_created_without_handoff(tmp_path, monkeypatch):
    """calendar-reminders 是 APP/local-server 合成会话，不存在时应原地创建。"""
    import hermes_state
    import cron.jobs as cron_jobs
    import gateway.platforms.zet_agent_cron as zc

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
            "calendar_provider": "device",
            "calendar_connection_id": "dev",
            "calendar_id": "local-cal",
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
    assert meta["calendar_provider"] == "device"
    assert meta["calendar_id"] == "local-cal"
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
    # Real upstream #8585 sentinel — note it embeds "timeout"; empty-result must
    # win over the retryable-keyword match, else it's mislabeled retryable.
    assert zc._classify_failure(
        "Agent completed but produced empty response "
        "(model error, timeout, or misconfiguration)"
    ) == ("empty_response", False)
    assert zc._classify_failure("ValueError: bad config") == ("agent_error", False)
    assert zc._classify_failure(None) == ("unknown", False)
    assert zc._classify_failure("") == ("unknown", False)


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
