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
