"""Flow tests: the protection guard on the real tool-dispatch path.

These go through ``model_tools.handle_function_call`` — the single choke point
every file mutation passes (direct tool calls, the execute_code sandbox RPC and
the MCP bridge all funnel into it) — so the wiring itself is under test, not
just the guard in isolation.
"""

from __future__ import annotations

import io
import json
import types
import urllib.error

import pytest

import model_tools
from tools import zettlab_snapshot_guard as guard


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _Recorder:
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(
            {
                "url": req.full_url,
                "body": json.loads(req.data.decode("utf-8")) if req.data else None,
            }
        )
        reply = self._replies.pop(0) if self._replies else {"ready": True, "operations": []}
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(json.dumps({"code": 200, "data": reply}).encode("utf-8"))


@pytest.fixture(autouse=True)
def _device_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", "http://127.0.0.1:19090/api/v1/internal/chat/append")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "tok123")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    # HOME 隔离到 tmp：开发机真实 HOME 里的 .bashrc 会触发 rc 遮蔽检测。
    monkeypatch.setenv("HOME", str(tmp_path))
    guard.reset_for_test()
    yield
    guard.reset_for_test()


def _install(monkeypatch, *replies):
    rec = _Recorder(replies)
    # 实现走禁用重定向的 _OPENER.open（不是裸 urlopen），mock 也挂在这一层。
    monkeypatch.setattr(guard, "_OPENER", types.SimpleNamespace(open=rec))
    return rec


def test_dispatch_writes_through_when_snapshot_unavailable_flow(monkeypatch, tmp_path):
    """本特性完全不可用时，行为与 2026-07-29 引入它之前完全一致（PRD 附录 B #18，
    验收 §19 #1b）。

    这是 #18 的核心验收项：local-server 挂了 → 拿不到恢复点 → **写入照常发生**。
    原用例断言的正好相反（「文件不该被动」），那是降级前的口径。
    """
    _install(monkeypatch, urllib.error.URLError("local-server down"))
    target = tmp_path / "预算.xlsx"
    target.write_text("original", encoding="utf-8")

    result = model_tools.handle_function_call(
        "write_file",
        {"path": str(target), "content": "overwritten"},
        task_id="t",
        turn_id="turn_1",
    )

    assert "error" not in json.loads(result), "入口挂了不该把用户的活干不下去"
    assert target.read_text(encoding="utf-8") == "overwritten", (
        "#18：拿不到恢复点也要照常写入——本特性只增加恢复点，不增加限制"
    )


def test_dispatch_allows_write_on_unprotected_folder_flow(monkeypatch, tmp_path):
    """Folders that cannot be snapshotted do not stop the agent — they are
    recorded as unprotected and the write goes through (PRD 附录 B #16)."""
    rec = _install(
        monkeypatch,
        {
            "ready": True,
            "operations": [
                {
                    "operationId": "aop_1",
                    "state": "unprotected",
                    "unprotected": True,
                    "unprotectedReason": "target_protection_disabled",
                }
            ],
        },
    )
    target = tmp_path / "a.txt"
    target.write_text("original", encoding="utf-8")

    blocked = guard.maybe_require_snapshot(
        "write_file", {"path": str(target)}, turn_id="turn_1"
    )

    assert blocked is None
    assert len(rec.requests) == 1, "仍然经过统一入口，只是没有恢复点"


def test_dispatch_is_unaffected_off_device_flow(monkeypatch, tmp_path):
    """Without local-server callbacks (CLI/dev), dispatch must not be blocked."""
    monkeypatch.delenv("ZET_CHAT_APPEND_URL", raising=False)
    monkeypatch.delenv("ZETTLAB_AGENT_SHARE_ACTION_URL", raising=False)
    rec = _install(monkeypatch)

    blocked = guard.maybe_require_snapshot(
        "write_file", {"path": str(tmp_path / "a.txt")}, turn_id="turn_1"
    )
    assert blocked is None
    assert rec.requests == []


def test_one_turn_takes_one_snapshot_then_reports_terminal_state_flow(monkeypatch, tmp_path):
    """A whole turn: lazily ensure once per folder, reuse, then release the pin."""
    rec = _install(monkeypatch)
    existing_a = tmp_path / "a.txt"
    existing_b = tmp_path / "b.txt"
    existing_a.write_text("x")
    existing_b.write_text("y")

    # Two edits to pre-existing files in the same turn.
    assert guard.maybe_require_snapshot("write_file", {"path": str(existing_a)}, turn_id="turn_1") is None
    assert guard.maybe_require_snapshot("write_file", {"path": str(existing_b)}, turn_id="turn_1") is None

    # A file the agent creates this turn, then keeps iterating on: no protection
    # needed, its original state is "absent".
    generated = tmp_path / "report.md"
    assert guard.maybe_require_snapshot("write_file", {"path": str(generated)}, turn_id="turn_1") is None
    generated.write_text("draft")
    assert guard.maybe_require_snapshot("write_file", {"path": str(generated)}, turn_id="turn_1") is None

    ensure_calls = [r for r in rec.requests if r["url"].endswith("/ensure")]
    assert len(ensure_calls) == 3, "server-side idempotency dedupes; the client only skips self-created files"

    guard.finish_turn("completed", turn_id="turn_1")
    finish_calls = [r for r in rec.requests if r["url"].endswith("/finish")]
    assert len(finish_calls) == 1
    assert finish_calls[0]["body"]["turnId"] == "turn_1"
    assert finish_calls[0]["body"]["state"] == "completed"


def test_failed_turn_reports_failed_state_flow(monkeypatch, tmp_path):
    rec = _install(monkeypatch)
    target = tmp_path / "a.txt"
    target.write_text("x")

    guard.maybe_require_snapshot("write_file", {"path": str(target)}, turn_id="turn_1")
    guard.finish_turn("failed", turn_id="turn_1", error_code="tool_error", error_stage="mutate")

    finish = [r for r in rec.requests if r["url"].endswith("/finish")][0]
    assert finish["body"]["state"] == "failed"
    assert finish["body"]["errorStage"] == "mutate"


def test_turn_without_protection_reports_nothing_flow(monkeypatch, tmp_path):
    """Read-only turns must not cost a single request."""
    rec = _install(monkeypatch)

    guard.maybe_require_snapshot("read_file", {"path": str(tmp_path / "a.txt")}, turn_id="turn_1")
    guard.maybe_require_snapshot("terminal", {"command": "ls -la"}, turn_id="turn_1")
    guard.finish_turn("completed")

    assert rec.requests == []


def test_snapshot_gate_sees_final_middleware_rewritten_args_flow(monkeypatch, tmp_path):
    """执行 middleware 能在 next_call(next_payload) 里改写工具参数：guard 挪进
    _dispatch 后必须对**最终**参数 ensure，否则真实写入目标没有恢复点
    （Codex review P1）。"""
    from hermes_cli import middleware as mw

    decoy = tmp_path / "decoy.txt"
    real = tmp_path / "real.txt"
    decoy.write_text("d", encoding="utf-8")
    real.write_text("r", encoding="utf-8")

    def rewrite(*, args, next_call, **_ctx):
        changed = dict(args)
        changed["path"] = str(real)
        return next_call(changed)

    monkeypatch.setattr(
        mw,
        "_get_middleware_callbacks",
        lambda kind: [rewrite] if kind == mw.TOOL_EXECUTION_MIDDLEWARE else [],
    )
    rec = _install(monkeypatch, {"ready": False, "operations": []})

    result = model_tools.handle_function_call(
        "write_file",
        {"path": str(decoy), "content": "overwritten"},
        task_id="t",
        turn_id="turn_1",
    )
    # #18：ready=false 不再阻断，所以这条用例的判据从「被挡住了」换成
    # 「ensure 请求里带的是改写**之后**的路径」——那才是它真正要守的东西。
    assert "error" not in json.loads(result)
    assert rec.requests[0]["body"]["paths"] == [str(real)], "ensure 必须看到改写后的最终路径"
    assert real.read_text(encoding="utf-8") == "overwritten", "写入落在改写后的目标上"
    assert decoy.read_text(encoding="utf-8") == "d", "诱饵路径不该被动"
