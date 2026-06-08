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
